from typing import Tuple
import random
import pytorch_lightning as pl
import dataclasses
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from train.base_algorithm import BaseAlgorithm
from ml_utils.args import DataClassArgumentParser
from inference.lm_sampler import LMSampler
from pytorch_lightning.loggers import WandbLogger
import torch
from model import transformer
import pretrain_sft_lm_data
import os
from lightning.pytorch.utilities import grad_norm


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("medium")


@dataclasses.dataclass()
class TrainerArgs:
    val_check_interval: int = 1000
    gradient_clip_val: float = 1
    max_epochs: int = 1
    max_steps: int = 60000
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 1
    precision: str = "bf16-mixed"


@dataclasses.dataclass()
class TrainArgs:
    total_batch_size: int = 384
    accumulate_grad_batches: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 1e-1
    warmup_steps: int = 2000
    seed: int = 1
    batch_size: int = -1
    min_learning_rate_ratio: float = 0.1
    ckpt_path: str = ""
    data_module_type: str = "fillseq"
    dataset_path: str = "cerebras/SlimPajama-627B"
    model_path: str = ""
    output_path: str = "micro_model.pth"
    project_name: str = "micro-training"




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

# Argument parser
def parse_args() -> Tuple[TrainArgs, TrainerArgs, transformer.DecoderOnlyTransformerConfig]:
    train_args, trainer_args, model_config = DataClassArgumentParser(
        (TrainArgs, transformer.DecoderOnlyTransformerConfig)
    ).parse_args_into_dataclasses()
    train_args.batch_size = (
        train_args.total_batch_size
        // train_args.accumulate_grad_batches
        // torch.cuda.device_count()
    )
    return train_args, trainer_args, model_config


# Main training function
def main():
    train_args, trainer_args, model_config = parse_args()
    # Load tokenizer
    wandb_logger = WandbLogger(project=train_args.project_name)
    trainer = pl.Trainer(
        logger=wandb_logger,
        **dataclasses.asdict(trainer_args),
    )
    model = model_config.build_model()
    tokenizer = model.tokenizer
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
