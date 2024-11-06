import random
import pytorch_lightning as pl
import dataclasses
from transformers import PreTrainedTokenizerFast, HfArgumentParser, AutoTokenizer
from chat_wrapper import ChatWrapper
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset
import torch
import micro_model
from torch.utils.data import DataLoader, Dataset
from functools import partial
from token_data import load_tokenizer, data_collator
import ml_utils
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclasses.dataclass()
class Args:
    total_batch_size: int = 2048
    accumulate_grad_batches: int = 40
    learning_rate: float = 7e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 1000
    seed: int = 42
    batch_size: int = -1
    max_steps: int = 10000
    val_check_interval: int = 3000
    ckpt_path: str = ""


# Argument parser
def parse_args() -> Args:
    args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
    args: Args
    args.batch_size = (
        args.total_batch_size
        // args.accumulate_grad_batches
        // torch.cuda.device_count()
    )
    return args


random.seed(42)
torch.manual_seed(42)


class MicroTraining(pl.LightningModule):
    def __init__(
        self,
        model: micro_model.TransformerDecoder,
        tokenizer: PreTrainedTokenizerFast,
        args: Args,
    ):
        super(MicroTraining, self).__init__()
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        self.loss = torch.nn.CrossEntropyLoss()
        self.check_first_batch = False

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]  # batch x seq_len
        logits = self.model(tokens=input_ids)
        loss = self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]  # batch x seq_len
        logits = self.model(tokens=input_ids)
        loss = self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("token_count", batch["token_count"], sync_dist=True)
        if batch_idx == 0:
            test_prompt = "My name is David, and I am"
            chat_warpper = ChatWrapper(
                self.model, self.tokenizer, torch.bfloat16, self.device
            )
            print(chat_warpper.show_output_probs(test_prompt))
            generated_text = chat_warpper.generate(test_prompt)
            print(generated_text)
        return loss

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters()
                      if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.local_rank == 0:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2),
            fused=True,
        )
        scheduler = ml_utils.optim.LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=self.args.warmup_steps,
            max_steps=self.trainer.max_steps,
            eta_min=0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

# Main training function
def main():
    args = parse_args()
    # Load tokenizer
    wandb_logger = WandbLogger(project="micro-training")
    wandb_logger.log_hyperparams(args.__dict__)

    trainer = pl.Trainer(
        max_epochs=1,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision="bf16-mixed",
        logger=wandb_logger,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
    )
    # tokenizer = PreTrainedTokenizerFast.from_pretrained("")
    tokenizer = load_tokenizer()
    model = micro_model.get_model()
    model_wrapper = MicroTraining(model, tokenizer, args)
    train_dataset = load_dataset(
        path="cerebras/SlimPajama-627B",
        split="train",
        trust_remote_code=True,
        streaming=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=partial(data_collator, tokenizer, model.max_seq_len),
    )
    val_dataset = load_dataset(
        path="cerebras/SlimPajama-627B",
        split="validation",
        trust_remote_code=True,
        streaming=True,
    )
    subset = ml_utils.data.IterableSubset(val_dataset, 1000)
    val_dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=partial(data_collator, tokenizer, model.max_seq_len),
    )

    trainer.fit(
        model_wrapper,
        train_dataloader,
        val_dataloader,
        ckpt_path=args.ckpt_path if args.ckpt_path else None,
    )
    torch.save(model.state_dict(), "micro_model.pth")


if __name__ == "__main__":
    main()
