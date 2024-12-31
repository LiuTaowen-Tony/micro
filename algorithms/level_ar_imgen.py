import dataclasses
import multiprocessing
import os
import torch
import pytorch_lightning as pl

import torchvision
from ml_utils.args import DataClassArgumentParser
from ml_utils.data import load_jsonl
from ml_utils.misc import load_with_config
from model.level_vqvae import LevelVQVAE, LevelVQVAEConfig
from model.transformer import TransformerDecoder, TransformerDecoderConfig
from .base_algorithm import BaseAlgorithm
from tqdm import tqdm
import json

@dataclasses.dataclass()
class TrainerArgs:
    # val_check_interval: int = 1.0
    gradient_clip_val: float = 1
    max_epochs: int = 30
    # max_steps: int = -1
    accumulate_grad_batches: int = 1
    precision: str = "bf16-mixed"

    
@dataclasses.dataclass()
class LevelImGenTainerArgs:
    total_batch_size: int = 128
    learning_rate: float = 1e-4
    # weight_decay: float = 0.0
    # warmup_steps: int = 1200
    beta: float = 0.25
    batch_size: int = -1
    # min_learning_rate_ratio: float = 0.5
    level: int = 0
    project_name: str = "vqvae-ffhq128"
    dataset_name: str = "ffhq128"
    dataset_cache_dir: str = "data/vqvae_latent_codes"
    vqvae_path: str = "trained_models/vqvae_ffhq128_young-lion-46"

class LevelImGenTrainer(BaseAlgorithm):
    def __init__(self, 
                 vqvae: LevelVQVAE, 
                 model: TransformerDecoder, 
                 train_args: LevelImGenTainerArgs, 
                 wandb):
        super().__init__(model, train_args, wandb)
        self.loss = torch.nn.CrossEntropyLoss()
        self.vqvae = vqvae
        self.train_args = train_args

    def forward(self, x):
        return self.model(x)

    def _loss(self, img, condition):
        return self.model(img, condition)

    @torch.no_grad()
    def _encode(self, x):
        """Encodes input data into latent codes for all levels."""
        _, _, _, _, codes = self.vqvae(x)
        return codes

    def _calculate_loss(self, x, condition):
        """Compute loss and accuracy."""
        x = x.to(self.device)
        condition = [c.to(self.device) for c in condition]
        self._dequantize_condition(condition)

        y, _ = self.prior(x, cs=condition)
        loss = F.cross_entropy(y, x, reduction='none').mean()
        accuracy = (torch.argmax(y, dim=1) == x).float().mean()
        return loss, accuracy

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
            for i, level_code_ids in enumerate(level_code_idss):
                level_i = "level_" + str(i)
                for j, code in enumerate(level_code_ids):
                    items[j][level_i] = code.flatten().cpu().numpy().tolist()
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

    @torch.no_grad()
    def _dequantize_condition(self, condition):
        """Dequantize higher-level latent codes for conditioning."""
        for i, c in enumerate(condition):
            condition[i] = self.vqvae.codebooks[self.level + i + 1].embed_code(c).permute(0, 3, 1, 2)

    def setup(self, stage):
        if self.train_args.dataset_name == "cifar10":
            raw_data = load_jsonl("data/vqvae_latent_codes/cifar10/data.jsonl")
            train_idxs, val_idxs = torch.arange(50000), torch.arange(50000, 60000)
        elif self.train_args.dataset_name == "ffhq128":
            raw_data = load_jsonl("data/vqvae_latent_codes/ffhq128/data.jsonl")
            train_idxs, val_idxs = torch.arange(60000), torch.arange(60000, 70000)

        level = self.train_args.level
        vqvae = self.vqvae
        class LevelImGenDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                item = self.data[idx]
                gt = item["level_" + str(level)]
                if level == 0:
                    condition = item["label"]

                else:
                    condition = item["level_" + str(level - 1)]
                    condition = vqvae.codebooks[level - 1].embed_code(torch.tensor(condition))
                gt = vqvae.codebooks[level].embed_code(torch.tensor(gt))

                return {
                    "input_ids": torch.tensor(gt, dtype=torch.long),
                    "labels": torch.tensor(condition, dtype=torch.long),
                }
        dataset = LevelImGenDataset(raw_data)
        self.train_dataset = torch.utils.data.Subset(dataset, train_idxs)
        self.val_dataset = torch.utils.data.Subset(dataset, val_idxs)

    def training_step(self, batch, batch_idx):
        input_ids, labels, attention_mask = batch
        logits = self.model(tokens=input_ids)
        loss = self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log_dict({
            "train_loss": loss,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"]
        })
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels, attention_mask = batch
        logits = self.model(tokens=input_ids)
        loss = self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log_dict({
            "val_loss": loss
        })
        return loss

def parse_args() -> tuple[LevelImGenTainerArgs, TrainerArgs]:
    train_args, trainer_args = DataClassArgumentParser(
        (LevelImGenTainerArgs, TrainerArgs)
    ).parse_args_into_dataclasses()
    train_args.batch_size = (
        train_args.total_batch_size
        // trainer_args.accumulate_grad_batches
        // torch.cuda.device_count()
    )
    return  train_args, trainer_args



if __name__ == "__main__":
    train_args, trainer_args = parse_args()
    vqvae = load_with_config(LevelVQVAEConfig, train_args.vqvae_path)
    model = None

    algorithm = LevelImGenTrainer(vqvae, model, train_args, None)
    algorithm.prepare_data()
    
