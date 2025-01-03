from typing import Tuple
import random
import pytorch_lightning as pl
import dataclasses
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from algorithms.base_algorithm import BaseAlgorithm
from ml_utils.args import DataClassArgumentParser
from sampler.lm_sampler import LMSampler
from pytorch_lightning.loggers import WandbLogger
import torch
from model import transformer
import pretrain_sft_lm_data
import ml_utils
import os
from lightning.pytorch.utilities import grad_norm
import lm_tokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("medium")


@dataclasses.dataclass()
class TrainArgs:
    total_batch_size: int = 384
    accumulate_grad_batches: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 2000
    seed: int = 1
    batch_size: int = -1
    max_steps: int = 60000
    val_check_interval: int = 3000
    min_learning_rate_ratio: float = 0.1
    gradient_clip_val: float = 1
    ckpt_path: str = ""
    max_epochs: int = 1
    data_module_type: str = "fillseq"
    dataset_path: str = "cerebras/SlimPajama-627B"
    model_path: str = ""
    output_path: str = "micro_model.pth"
    project_name: str = "micro-training"


# Argument parser
def parse_args() -> Tuple[TrainArgs, transformer.TransformerDecoderConfig]:
    train_args, model_config = DataClassArgumentParser(
        (TrainArgs, transformer.TransformerDecoderConfig)
    ).parse_args_into_dataclasses()
    train_args: TrainArgs
    train_args.batch_size = (
        train_args.total_batch_size
        // train_args.accumulate_grad_batches
        // torch.cuda.device_count()
    )
    return train_args, model_config


random.seed(1)
torch.manual_seed(1)


class LMDecoderAlgorithm(BaseAlgorithm):

    def __init__(
        self,
        model: transformer.DecoderOnlyTransformer,
        tokenizer: PreTrainedTokenizerFast,
        train_args: TrainArgs,
    ):
        super(LMDecoderAlgorithm, self).__init__()
        self.model = model
        self.args = train_args
        self.tokenizer = tokenizer
        self.loss = torch.nn.CrossEntropyLoss()
        self.check_first_batch = False
        self.prev_loss = float("inf")

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]  # batch x seq_len
        logits = self.model(tokens=input_ids)
        loss = self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        if loss.item() > self.prev_loss * 2:
            print(f"outlier loss: {loss.item()}")
        else:
            self.prev_loss = loss.item()
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]  # batch x seq_len
        logits = self.model(tokens=input_ids)
        loss = self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        if batch_idx == 0:
            test_prompt = "My name is David, and I am"
            chat_warpper = LMSampler(
                self.model, self.tokenizer, torch.bfloat16, self.device
            )
            print(chat_warpper.show_output_probs(test_prompt))
            generated_text = chat_warpper.generate(test_prompt)
            print(generated_text)
        return loss

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.model, norm_type=2)
    #     self.log_dict(norms)

    # def configure_optimizers(self):
    #     param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    #     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    #     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    #     optim_groups = [
    #         {"params": decay_params, "weight_decay": self.args.weight_decay},
    #         {"params": nodecay_params, "weight_decay": 0.0},
    #     ]
    #     num_decay_params = sum(p.numel() for p in decay_params)
    #     num_nodecay_params = sum(p.numel() for p in nodecay_params)
    #     if self.local_rank == 0:
    #         print(
    #             f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    #         )
    #         print(
    #             f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    #         )
    #     optimizer = torch.optim.AdamW(
    #         optim_groups,
    #         lr=self.args.learning_rate,
    #         betas=(self.args.beta1, self.args.beta2),
    #         fused=False,
    #     )
    #     max_steps = self.args.max_steps
    #     if self.args.max_steps == -1:
    #         max_steps = (
    #             self.trainer.max_epochs * self.trainer.estimated_stepping_batches
    #         )
    #     print(f"max_steps: {max_steps}")
    #     scheduler = ml_utils.optim.LinearWarmupCosineAnnealingLR(
    #         optimizer,
    #         warmup_steps=self.args.warmup_steps,
    #         max_steps=max_steps,
    #         eta_min=self.args.learning_rate * self.args.min_learning_rate_ratio,
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",
    #         },
    #     }


# Main training function
def main():
    train_args, model_config = parse_args()
    # Load tokenizer
    wandb_logger = WandbLogger(project=train_args.project_name)
    wandb_logger.log_hyperparams(train_args.__dict__)

    trainer = pl.Trainer(
        max_epochs=train_args.max_epochs,
        accumulate_grad_batches=train_args.accumulate_grad_batches,
        precision="bf16-mixed",
        logger=wandb_logger,
        max_steps=train_args.max_steps,
        val_check_interval=train_args.val_check_interval,
        log_every_n_steps=1,
        gradient_clip_val=train_args.gradient_clip_val,
    )
    # tokenizer = PreTrainedTokenizerFast.from_pretrained("")
    tokenizer = lm_tokenizer.load_tokenizer()
    # model = transformer.get_model_from_config(model_config)
    model = model_config.build_model()
    if train_args.model_path != "":
        model.load_state_dict(torch.load(train_args.model_path))

    if train_args.data_module_type == "fillseq":
        data_module_type = pretrain_sft_lm_data.FillSeqDataModule
    elif train_args.data_module_type == "sft":
        data_module_type = pretrain_sft_lm_data.SFTDataModule

    data_module = data_module_type(
        tokenizer,
        max_seq_len=model_config.max_seq_len,
        batch_size=train_args.batch_size,
        path=train_args.dataset_path,
    )
    model_wrapper = LMDecoderAlgorithm(model, tokenizer, train_args)
    trainer.fit(
        model_wrapper,
        datamodule=data_module,
        ckpt_path=train_args.ckpt_path if train_args.ckpt_path else None,
    )
    torch.save(model.state_dict(), train_args.output_path)


if __name__ == "__main__":
    main()
