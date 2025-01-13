import math
import torchvision
import dataclasses
import multiprocessing
import os
import torch
import pytorch_lightning as pl

import torchvision
from ml_utils.args import DataClassArgumentParser
from ml_utils.data import load_jsonl
from ml_utils.misc import load_with_config, save_with_config 
from ml_utils.dist import all_gather
from model.encoder_decoder_transformer import TransformerConfig, Transformer
from model.multi_level_vqvae import LevelVQVAE, LevelVQVAEConfig
from .base_algorithm import BaseAlgorithm
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm
import json

@dataclasses.dataclass()
class TrainerArgs:
    # val_check_interval: int = 1.0
    gradient_clip_val: float = 1
    max_epochs: int = 50
    # max_steps: int = -1
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 1
    precision: str = "bf16-mixed"


@dataclasses.dataclass()
class LevelImGenTainerArgs:
    total_batch_size: int = 24 * 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    warmup_steps: float = 300
    batch_size: int = -1
    min_learning_rate_ratio: float = 0.1
    level: int = 0
    project_name: str = "arimgen-cifar10-level0"
    conditional: bool = True
    dataset_name: str = "cifar10"
    dataset_cache_dir: str = "data/vqvae_latent_codes"
    vqvae_path: str = "/home/tl2020/micro/trained_models/vqvae_cifar10_earnest-energy-4"

class LevelImGenTrainer(BaseAlgorithm):

    def __init__(
        self,
        vqvae: LevelVQVAE,
        model: Transformer,
        train_args: LevelImGenTainerArgs,
        wandb: WandbLogger,
    ):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.vqvae = vqvae
        self.train_args = train_args
        self.wandb = wandb
        self.model = model
        self.cond_map = torch.nn.Linear(self.vqvae.config.embed_dim, self.model.config.d_model)
        self.target_map = torch.nn.Linear(self.vqvae.config.embed_dim, self.model.config.d_model)
        self.trainable = torch.nn.Sequential(self.cond_map, self.target_map, self.model)
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        return self.get_standard_optmizer(self.trainable,
                                          learning_rate=self.train_args.learning_rate,
                                          weight_decay=self.train_args.weight_decay,
                                          warmup_steps=self.train_args.warmup_steps,
                                          max_steps=self.trainer.estimated_stepping_batches,
                                          min_learning_rate_ratio=self.train_args.min_learning_rate_ratio)

    def prepare_data(self):
        dataset_path = os.path.join(self.train_args.dataset_cache_dir, self.train_args.dataset_name)
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
            self.vqvae = self.vqvae.to("cuda")
            _, _, _, _, level_code_idss = self.vqvae(images)
            items = []
            for label in labels:
                items.append({"idx": idx, "label": label.item()})
                idx += 1
            for i, level_code_ids in enumerate(level_code_idss[::-1]):
                level_i = "level_" + str(i)
                for j, code in enumerate(level_code_ids):
                    code : torch.Tensor
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
        for (image, label) in tqdm(dataloader, total=len(dataloader)):
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

        level = self.train_args.level
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
        nlevels = self.vqvae.n_levels
        level_code_ids = []
        shapes = []
        for i in range(nlevels):
            level_i = "level_" + str(i)
            tsr = torch.stack(batch[level_i], 1)
            w, h = batch[level_i + "_shape"][0][0], batch[level_i + "_shape"][1][0]
            shapes.append((w, h))
            level_code_ids.append(tsr)
        return level_code_ids, shapes, batch["label"]

    def forward(self, level_code_ids, shapes, class_label):
        level = self.train_args.level
        bsz, seq_len = level_code_ids[level].size()
        # if train top level
        if level == self.vqvae.n_levels - 1:
            embedded_condition = torch.nn.functional.one_hot(class_label, num_classes=self.model.config.d_model)
        else:
            cond_seq_len = level_code_ids[level + 1].size(1)
            embedded_condition = self.vqvae.embed_code_id(level_code_ids[level + 1], level + 1, shapes[level + 1]).reshape(bsz, cond_seq_len, -1)

        label_ids = torch.ones((bsz, seq_len + 1), dtype=torch.long, device=level_code_ids[level].device)
        label_ids[:, :-1] = level_code_ids[level]
        embedded_target = self.vqvae.embed_code_id(level_code_ids[level], level, shapes[level]).reshape(bsz, seq_len, -1)

        # pad seq_len + 1 to the left
        embedded_condition = self.cond_map(embedded_condition)
        embedded_target = self.target_map(embedded_target)

        transformed_features = self.model.feature_transform(src=embedded_condition, tgt=embedded_target,)
        logits = self.model.fc_out(transformed_features)
        loss = self.loss(logits.view(-1, logits.size(-1)), label_ids.view(-1))
        code_ids = torch.argmax(logits, dim=-1)
        return loss, logits, code_ids

    def log_reconstruct(self, level_code_ids, shapes, name):
        for i in range(len(level_code_ids)):
            w, h = shapes[i]
            level_code_ids[i] = all_gather(level_code_ids[i].view(-1, w, h))
        image = self.vqvae.decode_codes(*level_code_ids)
        nrow_y = math.isqrt(image.size(0))
        grid_y = torchvision.utils.make_grid(image, nrow=nrow_y, normalize=True)
        self.wandb.log_image(name, [grid_y])

    def training_step(self, batch, batch_idx):
        level_code_ids, shapes, label = self.build_level_code_ids(batch)
        loss, logits, predicted_code_ids = self.forward(level_code_ids, shapes, label)
        level_code_ids[self.train_args.level] = predicted_code_ids
        if batch_idx == 0:
            self.log_reconstruct(level_code_ids, shapes, "train_output")
        self.log_dict({
            "train_loss": loss,
            "logits_norm": logits.norm(),
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"]
        }, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        level_code_ids, shapes, label = self.build_level_code_ids(batch)
        loss, logits, predicted_code_ids = self.forward(level_code_ids, shapes, label)
        level_code_ids[self.train_args.level] = predicted_code_ids
        if batch_idx == 0:
            self.log_reconstruct(level_code_ids, shapes, "val_output")
        self.log_dict({
            "val_loss": loss
        }, prog_bar=True, sync_dist=True)
        return loss

def parse_args() -> tuple[LevelImGenTainerArgs, TrainerArgs, TransformerConfig]:
    train_args, trainer_args, model_config = DataClassArgumentParser(
        (LevelImGenTainerArgs, TrainerArgs, TransformerConfig)
    ).parse_args_into_dataclasses()
    train_args.batch_size = (
        train_args.total_batch_size
        // trainer_args.accumulate_grad_batches
        // torch.cuda.device_count()
    )
    return  train_args, trainer_args, model_config


if __name__ == "__main__":
    train_args, trainer_args, model_config = parse_args()
    vqvae = load_with_config(LevelVQVAEConfig, train_args.vqvae_path)
    model = model_config.build_model()
    logger = WandbLogger(project=train_args.project_name)
    logger.log_hyperparams(train_args.__dict__ | trainer_args.__dict__ | model_config.__dict__)
    algorithm = LevelImGenTrainer(vqvae, model, train_args, logger)
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=logger,
        **trainer_args.__dict__
    )
    trainer.fit(algorithm)
    save_with_config(model, f"trained_models/level_ar_imgen-{train_args.level}-{train_args.dataset_name}")

