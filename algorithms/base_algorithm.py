import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from ml_utils.optim import LinearWarmupCosineAnnealingLR, escape_non_decay
from lightning.pytorch.utilities import grad_norm

class BaseAlgorithm(pl.LightningModule):
    def __init__(self, model, train_args, wandb: WandbLogger):
        super().__init__()
        self.model = model
        self.train_args = train_args
        self.wandb = wandb

    def configure_optimizers(self):
        return escape_non_decay(
            self.model,
            lambda x : torch.optim.AdamW(x, 
                self.train_args.learning_rate,
                fused=False),
            lambda x : LinearWarmupCosineAnnealingLR(
                x,
                warmup_steps=self.train_args.warmup_steps,
                max_steps=self.trainer.estimated_stepping_batches,
                eta_min=self.train_args.learning_rate * self.train_args.min_learning_rate_ratio),
            self.train_args.weight_decay
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_args.batch_size,
            num_workers=9,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.train_args.batch_size,
            num_workers=9,
        )

    def on_before_optimizer_step(self, optimizer):
        self.log("lr", optimizer.param_groups[0]["lr"])
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)