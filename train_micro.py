import math
import random
import pytorch_lightning as pl
import dataclasses
from transformers import ( PreTrainedTokenizerFast, HfArgumentParser)
from chat_wrapper import ChatWrapper
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset
import torch
import micro_model
from torch.utils.data import DataLoader, Dataset
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from functools import partial
from token_data import load_tokenizer, data_collator
from pytorch_lightning.callbacks import ModelCheckpoint


checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="global_step",
    every_n_train_steps=5000,
    dirpath="checkpoints",
    filename="micro-0.2b-{global_step}",
)

@dataclasses.dataclass()
class Args:
    total_batch_size: int = 2048
    accumulate_grad_batches: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 1000
    seed: int = 42
    size: float = 1.0
    batch_size: int = -1
    max_steps: int = 15000
    

# Argument parser
def parse_args() -> Args:
    args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
    args: Args
    args.batch_size = args.total_batch_size // args.accumulate_grad_batches // torch.cuda.device_count()
    return args

random.seed(42)
torch.manual_seed(42)

from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, eta_min=0, last_step=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        self.last_step = 0
        super().__init__(optimizer, last_step)

    def get_lr(self):
        if self.last_step < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_step / float(self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.last_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay for base_lr in self.base_lrs]

    def step(self, step=None):
        """Update the learning rate per step (batch) instead of per epoch."""
        if step is not None:
            self.last_step = step
        else:
            self.last_step += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class MicroTraining(pl.LightningModule):
    def __init__(self, model: micro_model.TransformerDecoder, tokenizer: PreTrainedTokenizerFast, args: Args):
        super(MicroTraining, self).__init__()
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        self.loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100, label_smoothing=0.1)
        self.check_first_batch = False

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"] # batch x seq_len
        logits = self.model(tokens=input_ids)
        loss = self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.sum() / batch["token_count"]
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        if batch_idx % 1000 == 0:
            print(labels.shape)
            test_prompt = "My name is David, and I am"
            chat_warpper = ChatWrapper(self.model, self.tokenizer, torch.bfloat16, self.device)
            print(chat_warpper.show_output_probs(test_prompt))
            generated_text = chat_warpper.generate(test_prompt)
            print(generated_text)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.args.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.local_rank == 0:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optimizer = torch.optim.AdamW(optim_groups, lr=self.args.learning_rate, betas=(self.args.beta1, self.args.beta2), fused=True)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps=self.args.warmup_steps, max_steps=self.trainer.max_steps, eta_min=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    

def make_log(log):
    print(log)

# Main training function
def main():
    args = parse_args()
    # Load tokenizer
    tokenizer = load_tokenizer()
    dataset = load_dataset(path="cerebras/SlimPajama-627B", split='train', trust_remote_code=True, streaming=True)

    model = micro_model.get_model()
    model_wrapper = MicroTraining(model, tokenizer, args)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, collate_fn=partial(data_collator, tokenizer, model.max_seq_len), )

    wandb_logger = WandbLogger(project='micro-training')
    wandb_logger.log_hyperparams(args.__dict__)

    trainer = pl.Trainer(
        devices=4,
        max_epochs=1,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision="bf16",
        logger=wandb_logger,
        max_steps=args.max_steps,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model_wrapper, train_dataloader)
    trainer.save_checkpoint("gpt2_micro_hf.ckpt")

if __name__ == "__main__":
    main()

