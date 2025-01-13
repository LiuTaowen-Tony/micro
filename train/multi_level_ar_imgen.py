import math
import torch.utils.data.dataset
import torchvision
import dataclasses
import multiprocessing
import os
import torch
from tqdm import tqdm
import json

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from ml_utils.args import DataClassArgumentParser
from ml_utils.data import load_jsonl
from ml_utils.misc import load_with_config, save_with_config
from ml_utils.dist import all_gather

from model.ar_imgen_transformer import TransformerArImGenConfig, TransformerArImGen
from .base_algorithm import BaseAlgorithm


@dataclasses.dataclass()
class TrainerArgs:
    # val_check_interval: int = 1.0
    gradient_clip_val: float = 1
    max_epochs: int = 20
    # max_steps: int = -1
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 1
    precision: str = "bf16-mixed"


@dataclasses.dataclass()
class LevelImGenTainerArgs:
    total_batch_size: int = 64 * 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    warmup_steps: float = 0.1
    batch_size: int = -1
    min_learning_rate_ratio: float = 0.1
    project_name: str = "arimgen-cifar10-level0"
    conditional: bool = True
    dataset_name: str = "cifar10"
    dataset_cache_dir: str = "data/vqvae_latent_codes"

class LevelImGenTrainer(BaseAlgorithm):

    def __init__(
        self,
        model: TransformerArImGen,
        train_args: LevelImGenTainerArgs,
        wandb: WandbLogger,
    ):
        super().__init__()
        self.train_args = train_args
        self.wandb = wandb
        self.model = model
        self.save_hyperparameters({
            "train_args": dataclasses.asdict(train_args),
            "model_config": dataclasses.asdict(model.config),
        })

    def configure_optimizers(self):
        return self.get_standard_optmizer(
            self.model,
            learning_rate=self.train_args.learning_rate,
            weight_decay=self.train_args.weight_decay,
            warmup_steps=self.train_args.warmup_steps,
            max_steps=self.trainer.estimated_stepping_batches,
            min_learning_rate_ratio=self.train_args.min_learning_rate_ratio,
        )

    def prepare_data(self):
        dataset_path = os.path.join(
            self.train_args.dataset_cache_dir, self.train_args.dataset_name
        )
        if os.path.exists(dataset_path):
            return
        transforms = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transforms = torchvision.transforms.Compose(transforms)
        if self.train_args.dataset_name == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(
                root="data", train=True, download=True, transform=transforms
            )
            val_dataset = torchvision.datasets.CIFAR10(
                root="data", train=False, download=True, transform=transforms
            )
            # concat train and val
            dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        elif self.train_args.dataset_name == "ffhq128":
            transforms = [
                torchvision.transforms.Resize(256),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
            transforms = torchvision.transforms.Compose(transforms)
            dataset = torchvision.datasets.ImageFolder(
                "data/ffhq128", transform=transforms
            )
        idx = 0

        def preprocess(images, labels):
            nonlocal idx
            # level+code_idss
            # level list -> [bsz, n_tokens, tokens_dim]
            images = images.to("cuda")
            model = self.model.to("cuda")
            level_code_idss = model.get_level_code_ids(images)
            items = []
            for label in labels:
                items.append({"idx": idx, "label": label.item()})
                idx += 1
            for i, level_code_ids in enumerate(level_code_idss[::-1]):
                level_i = "level_" + str(i)
                for j, code in enumerate(level_code_ids):
                    code: torch.Tensor
                    shape = list(code.size())
                    items[j][level_i] = code.flatten().cpu().numpy().tolist()
                    items[j][level_i + "_shape"] = shape
            return items

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_args.total_batch_size,
            num_workers=multiprocessing.cpu_count(),
            shuffle=False,
        )
        os.makedirs(dataset_path, exist_ok=True)
        for image, label in tqdm(dataloader, total=len(dataloader)):
            with open(os.path.join(dataset_path, "data.jsonl"), "a+") as f:
                for item in preprocess(image, label):
                    f.write(json.dumps(item) + "\n")

    def setup(self, stage):
        if self.train_args.dataset_name == "cifar10":
            raw_data = load_jsonl("data/vqvae_latent_codes/cifar10/data.jsonl")
            train_idxs, val_idxs = torch.arange(50000), torch.arange(50000, 60000)
        elif self.train_args.dataset_name == "ffhq128":
            raw_data = load_jsonl("data/vqvae_latent_codes/ffhq128/data.jsonl")
            train_idxs, val_idxs = torch.arange(60000), torch.arange(60000, 70000)

        class LevelImGenDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]
        dataset = LevelImGenDataset(raw_data)
        self.train_dataset = torch.utils.data.Subset(dataset, train_idxs)
        self.val_dataset = torch.utils.data.Subset(dataset, val_idxs)

    def build_level_code_ids(self, batch):
        nlevels = self.model.config.num_levels
        level_code_ids = []
        shapes = []
        for i in range(nlevels):
            level_i = "level_" + str(i)
            tsr = torch.stack(batch[level_i], 1)
            w, h = batch[level_i + "_shape"][0][0], batch[level_i + "_shape"][1][0]
            shapes.append((w, h))
            level_code_ids.append(tsr)
        return level_code_ids, shapes, batch["label"]

    def forward(
        self,
        level_code_ids: torch.Tensor,
        shapes: list[tuple],
        class_label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.train_forward(level_code_ids, shapes, class_label)

    def log_reconstruct(self, level_code_ids, shapes, name):
        # for i in range(len(level_code_ids)):
        #     level_code_ids[i] = all_gather(level_code_ids[i])
        image = self.model.decode_codes(level_code_ids, shapes)
        nrow_y = math.isqrt(image.size(0))
        grid_y = torchvision.utils.make_grid(image, nrow=nrow_y, normalize=True)
        self.wandb.log_image(name, [grid_y])

    def training_step(self, batch, batch_idx):
        level_code_ids, shapes, label = self.build_level_code_ids(batch)
        loss, logits, predicted_code_ids = self.forward(level_code_ids, shapes, label)
        level_code_ids[self.model.config.level] = predicted_code_ids
        if batch_idx == 0:
            self.log_reconstruct(level_code_ids, shapes, "train_output")
        self.log_dict(
            {
                "train_loss": loss,
                "logits_norm": logits.norm(),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        level_code_ids, shapes, label = self.build_level_code_ids(batch)
        loss, logits, predicted_code_ids = self.forward(level_code_ids, shapes, label)
        level_code_ids[self.model.config.level] = predicted_code_ids
        print(predicted_code_ids.shape)

        if batch_idx == 0:
            self.log_reconstruct(level_code_ids, shapes, "val_output")
        self.log_dict({"val_loss": loss}, prog_bar=True, sync_dist=True)
        return loss


def parse_args() -> tuple[LevelImGenTainerArgs, TrainerArgs, TransformerArImGenConfig]:
    train_args, trainer_args, model_config = DataClassArgumentParser(
        (LevelImGenTainerArgs, TrainerArgs, TransformerArImGenConfig)
    ).parse_args_into_dataclasses()
    train_args.batch_size = (
        train_args.total_batch_size
        // trainer_args.accumulate_grad_batches
        // torch.cuda.device_count()
    )
    return train_args, trainer_args, model_config


if __name__ == "__main__":
    train_args, trainer_args, model_config = parse_args()
    model = model_config.build_model()
    logger = WandbLogger(project=train_args.project_name)
    algorithm = LevelImGenTrainer(model, train_args, logger)
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=logger,
        **trainer_args.__dict__,
    )
    trainer.fit(algorithm)
    save_with_config(
        model,
        f"trained_models/level_ar_imgen/t_ar_imgen/{train_args.dataset_name}/0",
    )
