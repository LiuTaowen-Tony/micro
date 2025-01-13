import math
import os
import dataclasses
from typing import Tuple
import torch
import torchvision
from torchvision.utils import save_image
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import ml_utils.dist
from ml_utils.args import DataClassArgumentParser
from ml_utils.misc import save_with_config

from model.multi_level_vqvae import MultiLevelVQVAE, get_model_by_taskname
from train.base_algorithm import BaseAlgorithm


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("medium")


@dataclasses.dataclass()
class TrainerArgs:
    # val_check_interval: int = 1.0
    gradient_clip_val: float = 1
    max_epochs: int = 30
    # max_steps: int = -1
    accumulate_grad_batches: int = 1
    precision: str = "bf16-mixed"


@dataclasses.dataclass()
class VQVAETrainerArgs:
    total_batch_size: int = 128
    learning_rate: float = 1e-4
    # weight_decay: float = 0.0
    # warmup_steps: int = 1200
    beta: float = 0.25
    batch_size: int = -1
    # min_learning_rate_ratio: float = 0.5
    project_name: str = "vqvae-ffhq128"
    dataset_name: str = "ffhq128"


class VQVAETrainer(BaseAlgorithm):

    def __init__(
        self, model: MultiLevelVQVAE, train_args: VQVAETrainerArgs, wandb: WandbLogger
    ) -> None:
        super().__init__()
        self.model = model
        self.train_args = train_args
        self.wandb = wandb

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, _ = batch
        y, d, _, _, _ = self.model(img)
        diff = y.sub(img)
        r_loss = diff.pow(2).mean() 
        l_loss = sum(d)        
        loss = r_loss + self.train_args.beta*l_loss
        self.log_dict({"train_loss": loss, "r_loss": r_loss, "l_loss": l_loss}, prog_bar=True)
        if batch_idx == 0:
            nrow_y = math.isqrt(y.size(0))
            grid_y = torchvision.utils.make_grid(y, nrow=nrow_y, normalize=True)
            self.wandb.log_image("train_output", [wandb.Image(grid_y)])
        return loss

    def validation_step(self, batch, batch_idx):
        img, _ = batch
        y, d, _, _, _ = self.model(img)
        diff = y.sub(img)
        r_loss = diff.pow(2).mean() 
        l_loss = sum(d)
        loss = r_loss + self.train_args.beta*l_loss
        self.log_dict({"val_loss": loss, "val_r_loss": r_loss, "val_l_loss": l_loss}, sync_dist=True, prog_bar=True)
        # logimage  to wandb
        if batch_idx == 0:
            y = ml_utils.dist.all_gather_concat_pl(self, y)
            nrow_y = math.isqrt(y.size(0))
            grid_y = torchvision.utils.make_grid(y, nrow=nrow_y, normalize=True)
            self.wandb.log_image("val_output", [wandb.Image(grid_y)])
            os.makedirs("images", exist_ok=True)
            save_image(y, f"images/recon-{str(self.trainer.current_epoch).zfill(4)}.{'png'}", nrow=nrow_y, normalize=True, value_range=(-1,1))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_args.learning_rate,
        )

    def setup(self, stage):
        transforms = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transforms = torchvision.transforms.Compose(transforms)
        if self.train_args.dataset_name == "cifar10":
            self.train_dataset = torchvision.datasets.CIFAR10(
                root="data", train=True, download=True, transform=transforms
            )
            self.val_dataset = torchvision.datasets.CIFAR10(
                root="data", train=False, download=True, transform=transforms
            )
        elif self.train_args.dataset_name == "imagenet":
            self.train_dataset = torchvision.datasets.ImageNet(
                root="data", split="train", download=True, transform=transforms
            )
            self.val_dataset = torchvision.datasets.ImageNet(
                root="data", split="val", download=True, transform=transforms
            )
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
            train_idx, test_idx = torch.arange(0, 60000), torch.arange(60000, len(dataset))
            self.train_dataset = torch.utils.data.Subset(dataset, train_idx)
            self.val_dataset = torch.utils.data.Subset(dataset, test_idx)
        elif self.train_args.dataset_name == "ffhq1024":
            self.train_dataset = torchvision.datasets.ImageFolder(
                "data/ffhq1024", transform=transforms
            )
            self.val_dataset = torchvision.datasets.ImageFolder(
                "data/ffhq1024", transform=transforms
            )
        else:
            raise ValueError(f"Unknown dataset: {self.train_args.dataset_name}")

def parse_args() -> Tuple[TrainerArgs, VQVAETrainerArgs]:
    trainer_args, train_args = DataClassArgumentParser(
        (TrainerArgs, VQVAETrainerArgs)).parse_args_into_dataclasses()
    train_args: VQVAETrainerArgs
    train_args.batch_size = (
        train_args.total_batch_size //
        torch.cuda.device_count() //
        trainer_args.accumulate_grad_batches
    )
    return trainer_args, train_args

def main():
    trainer_args, train_args = parse_args()
    wandb_logger = WandbLogger(project=train_args.project_name)

    trainer = pl.Trainer(
        logger=wandb_logger,
        **trainer_args.__dict__
    )

    model = get_model_by_taskname(train_args.dataset_name)
    # cfg = HPS_VQVAE[train_args.dataset_name]
    # model = VQVAE(
    #     in_channels=cfg.in_channels,
    #     hidden_channels=cfg.hidden_channels,
    #     embed_dim=cfg.embed_dim,
    #     nb_entries=cfg.nb_entries,
    #     nb_levels=cfg.nb_levels,
    #     scaling_rates=cfg.scaling_rates,
    # )
    # model_config = model.config
    # print(cfg)
    wrapper = VQVAETrainer(model, train_args, wandb_logger)
    trainer.fit(wrapper)
    save_with_config(model, f"trained_models/vqvae_{train_args.dataset_name}_{wandb_logger.experiment.name}")

if __name__ == "__main__":
    main()
