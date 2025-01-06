from dataclasses import dataclass
import dataclasses
import os
import datasets
import torch
from torch import nn
import torchvision
from ml_utils.args import DataClassArgumentParser
from ml_utils.misc import save_with_config, to_model_device
from ml_utils.optim import LinearWarmupCosineAnnealingLR
from model.clip import Clip, ClipInitConfig
from train.base_algorithm import BaseAlgorithm
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import grad_norm
from inference.clip import ClipInference

from model.transformer import DecoderOnlyTransformer

torch.set_float32_matmul_precision('medium')

@dataclass()
class TrainerArgs:
    val_check_interval: int = 0.5
    gradient_clip_val: float = 1
    max_epochs: int = 8
    # max_steps: int = -1
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 1
    precision: str = "bf16-mixed"


@dataclass
class ClipTrainArgs:
    batch_size: int = -1
    total_batch_size: int = 32
    img_encoder_lr: float = 1e-4
    text_encoder_lr: float = 1e-5
    img_projection_head_lr: float = 1e-3
    text_projection_head_lr: float = 1e-3
    text_model_trainable_layers: int = 5
    weight_decay: float = 0.0
    warmup_steps: int = 0.1
    min_learning_rate_ratio: float = 0.1
    temperature: float = 0.1
    max_text_len: int = 32
    temperature: float = 0.1
    project_name: str = "clip"


class ClipAlgorithm(BaseAlgorithm):

    def __init__(
        self,
        clip_config: ClipInitConfig,
        train_args: ClipTrainArgs,
        wandb: WandbLogger,
    ):
        super().__init__()
        self.init_config = clip_config
        self.clip_model = clip_config.build_model()
        self.train_args = train_args
        self.tokenizer = self.clip_model.text_encoder.tokenizer
        self.wandb = wandb
        self.freeze_text_encoder()
        self.save_hyperparameters(
            dataclasses.asdict(train_args) | dataclasses.asdict(self.clip_model.config)
            | dataclasses.asdict(self.init_config)
        )

    def forward(self, batch):
        text_features, img_features = self.clip_model(
            batch["input_ids"], batch["image"], batch["attention_mask"]
        )
        # text_features = all_gather_concat_pl(self, text_features)
        # img_features = all_gather_concat_pl(self, img_features)
        # text_features = all_gather_concat(text_features, self.local_rank, torch.cuda.device_count())
        # img_features = all_gather_concat(img_features, self.local_rank, torch.cuda.device_count())
        # img_features = all_gather(img_features)
        # text_features = all_gather(text_features)
        # print(img_features.shape, text_features.shape)

        logits_per_text = (text_features @ img_features.t()) / self.train_args.temperature
        logits_per_img = logits_per_text.t()

        # target = F.softmax(
        #     (images_sim + texts_sim) / 2 * self.train_args.temperature, dim=-1
        # )
        # # why build target like this?
        # # there are some internal similarity between different images and texts
        # # for example: two images of dogs should be similar, they can appear in the same
        # # batch

        # text_loss = self.cross_entropy(logits, target)
        # image_loss = self.cross_entropy(logits.t(), target.t())
        # loss = (text_loss + image_loss) / 2  # shape (batch_size,)
        labels = to_model_device(torch.arange(logits_per_text.shape[0]), self)
        loss1 = nn.functional.cross_entropy(logits_per_text, labels)
        loss2 = nn.functional.cross_entropy(logits_per_img, labels)
        return loss1.mean() + loss2.mean() / 2

    
    def on_before_optimizer_step(self, optimizer):
        result = {}
        for i, param_group in enumerate(optimizer.param_groups):
            result[f"lr_{i}"] = param_group["lr"]
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            text_norms = grad_norm(self.clip_model.text_encoder, norm_type=2)
            img_norms = grad_norm(self.clip_model.image_encoder, norm_type=2)
            img_head_norms = grad_norm(self.clip_model.img_projection_head, norm_type=2)
            text_head_norms = grad_norm(self.clip_model.text_projection_head, norm_type=2)
            result["text_norms"] = text_norms["grad_2.0_norm_total"]
            result["img_norms"] = img_norms["grad_2.0_norm_total"]
            result["img_head_norms"] = img_head_norms["grad_2.0_norm_total"]
            result["text_head_norms"] = text_head_norms["grad_2.0_norm_total"]
        self.log_dict(result)

    def freeze_text_encoder(self):
        if isinstance(self.clip_model.text_encoder, DecoderOnlyTransformer):
            self.clip_model.text_encoder.tok_embeddings.requires_grad_(False)
            for i in range(
                len(self.clip_model.text_encoder.layers)
                - self.train_args.text_model_trainable_layers
            ):
                self.clip_model.text_encoder.layers[i].requires_grad_(False)

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train_loss", loss, prog_bar=True)
        if batch_idx == 0 and self.local_rank == 0 and self.global_rank == 0:
            inference = ClipInference(self.clip_model, self.tokenizer)
            img = inference.show_confusion_matrix(
                batch["image"], batch["caption"]
            )
            self.wandb.log_image("confusion_matrix_train", [img])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        if batch_idx == 0 and self.local_rank == 0 and self.global_rank == 0:
            inference = ClipInference(self.clip_model, self.tokenizer)
            img = inference.show_confusion_matrix(
                batch["image"], batch["caption"]
            )
            self.wandb.log_image("confusion_matrix_val", [img])
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            [
                {
                    "params": self.clip_model.image_encoder.parameters(),
                    "lr": self.train_args.img_encoder_lr,
                },
                {
                    "params": self.clip_model.text_encoder.parameters(),
                    "lr": self.train_args.text_encoder_lr,
                },
                {
                    "params": self.clip_model.img_projection_head.parameters(),
                    "lr": self.train_args.img_projection_head_lr,
                },
                {
                    "params": self.clip_model.text_projection_head.parameters(),
                    "lr": self.train_args.text_projection_head_lr,
                },
            ],
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_steps=self.train_args.warmup_steps,
            max_steps=self.trainer.estimated_stepping_batches,
            eta_min=self.train_args.min_learning_rate_ratio,
            eta_min_type="relative",
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def setup(self, stage):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.clip_model = self.init_config.build_model()

        def batch_transform(batch):
            batch["image"] = [transforms(x) for x in batch["image"]]
            batch["caption"] = [x[0] for x in batch["caption"]]
            return batch

        def collator(batch: list):
            images = torch.stack([x["image"] for x in batch])
            texts = self.tokenizer(
                [x["caption"] for x in batch],
                padding=True,
                truncation=True,
                max_length=self.train_args.max_text_len,
            )
            return {
                "image": images,
                "caption": [x["caption"] for x in batch],
                "input_ids": torch.tensor(texts["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(texts["attention_mask"], dtype=torch.long),
            }

        dataset = (
            datasets.load_dataset("nlphuji/flickr30k", split="test")
            .select_columns(["image", "caption"])
            .with_transform(batch_transform)
            .train_test_split(test_size=0.1)
        )

        self.collate_fn = collator
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

    def prepare_data(self):
        datasets.load_dataset("nlphuji/flickr30k")


def parse_args() -> tuple[ClipInitConfig, TrainerArgs, ClipTrainArgs]:
    model_config, trainer_args, train_args = DataClassArgumentParser(
        (ClipInitConfig, TrainerArgs, ClipTrainArgs)
    ).parse_args_into_dataclasses()
    print(torch.cuda.device_count())
    train_args.batch_size = (
        train_args.total_batch_size
        // trainer_args.accumulate_grad_batches
        // torch.cuda.device_count()
    )
    return train_args, trainer_args, model_config


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_args, trainer_args, model_config = parse_args()
    logger = WandbLogger(project=train_args.project_name)
    algorithm = ClipAlgorithm(model_config, train_args, logger)
    trainer = pl.Trainer(
        **trainer_args.__dict__,
        logger=logger,
    )
    trainer.fit(algorithm)
    save_with_config(algorithm.clip_model, "trained_models/clip-micro-lm-mobilenet-v4-flickr30k")
