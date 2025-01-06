import pytorch_lightning as pl
import torch

from ml_utils.optim import LinearWarmupCosineAnnealingLR, get_decay_non_decay
from ml_utils.dist import rank0_print
from lightning.pytorch.utilities import grad_norm

class BaseAlgorithm(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.train_args = None
        self.wandb = None

    def get_standard_optmizer(self, 
                              model,
                              learning_rate: float = None,
                              weight_decay: float = None,
                              warmup_steps: int = None,
                              max_steps: int = None,
                              min_learning_rate_ratio: float = None):
        if learning_rate is None:
            if hasattr(self.train_args, "learning_rate"):
                learning_rate = self.train_args.learning_rate
            else:
                raise ValueError("learning_rate must be provided")
        if weight_decay is None:
            weight_decay = 0.0
            if hasattr(self.train_args, "weight_decay"):
                weight_decay = self.train_args.weight_decay
        if warmup_steps is None:
            warmup_steps = 0
            if hasattr(self.train_args, "warmup_steps"):
                warmup_steps = self.train_args.warmup_steps
        if min_learning_rate_ratio is None:
            min_learning_rate_ratio = 0.0
            if hasattr(self.train_args, "min_learning_rate_ratio"):
                min_learning_rate_ratio = self.train_args.min_learning_rate_ratio
        if max_steps is None:
            max_steps = -1
            if hasattr(self.train_args, "max_steps"):
                max_steps = self.train_args.max_steps

        if max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches

        decay_params, nodecay_params = get_decay_non_decay(model)
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            fused=False,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            eta_min=min_learning_rate_ratio,
            eta_min_type="relative",
        )
        
        result =  {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
        rank0_print(result)
        return result


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_args.batch_size,
            num_workers=9,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.train_args.batch_size,
            num_workers=9,
            collate_fn=self.collate_fn
        )

    def on_before_optimizer_step(self, optimizer):
        result = {}
        for i, param_group in enumerate(optimizer.param_groups):
            result[f"lr_{i}"] = param_group["lr"]
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            norms = grad_norm(self, norm_type=2)
            result.update(norms)
        self.log_dict(result)