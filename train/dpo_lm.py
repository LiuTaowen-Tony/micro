from copy import deepcopy
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from dataclasses import dataclass
import sampler.lm_sampler as lm_sampler
import sampler.lm_tokenizer as lm_tokenizer
from model import micro_lm_model
from ml_utils.args import DataClassArgumentParser
import ml_utils.data
from sequence_modelling import pad_to_length, boolean_triangular_mask
from ml_utils.misc import disable_dropout
import ml_utils.optim
from transformers import PreTrainedTokenizerFast
from pytorch_lightning.loggers import WandbLogger
from typing import Dict, Union, List, Tuple
import dpo_lm_data
import torch.nn.functional as F


@dataclass
class DPOConfig:
    model_path: str
    datasets_names: List[str]

    lr: float = 5e-7
    total_batch_size: int = 32
    accumulate_grad_batches: int = 4
    batch_size: int = -1
    val_check_interval: int = 200
    gradient_clip_val: float = 1
    sample_during_eval: bool = True
    max_epochs: int = 1
    weight_decay: float = 0.0
    warmup_steps: int = 150
    project_name: str = "dpo-training"
    output_path: str = "dpo_model"
    max_seq_len: int = 512
    max_prompt_len: int = 256


@dataclass
class LossConfig:
    loss_name: str = "dpo"
    loss_beta: float = 0.1
    reference_free: bool = False
    label_smoothing: float = 0.0


def parse_args() -> Tuple[DPOConfig, LossConfig]:
    dpo_config, loss_config = DataClassArgumentParser(
        (DPOConfig, LossConfig)
    ).parse_args_into_dataclasses()
    dpo_config: DPOConfig
    dpo_config.batch_size = (
        dpo_config.total_batch_size
        // dpo_config.accumulate_grad_batches
        // torch.cuda.device_count()
    )
    return dpo_config, loss_config


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}


def _get_batch_logps(
    logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    ipo: bool = False,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (
            logits - 1 / (2 * beta)
        ) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = (
        beta * (policy_rejected_logps - reference_rejected_logps).detach()
    )

    return losses, chosen_rewards, rejected_rewards


def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]]
) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(
        batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
    )
    concatenated_batch = {}
    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(
                batch[k], max_length, pad_value=pad_value
            )
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            )
    return concatenated_batch


class DPOModule(pl.LightningModule):
    def __init__(
        self,
        policy: nn.Module,
        reference: nn.Module,
        tokenizer: PreTrainedTokenizerFast,
        config: DPOConfig,
        loss_config: LossConfig,
        wandb_logger: WandbLogger,
        datasets_names: List[str],
    ):
        super().__init__()
        self.policy = policy
        self.reference = reference
        self.tokenizer = tokenizer
        self.loss_config = loss_config
        self.config = config
        self.example_counter = 0
        self.batch_counter = 0
        self.wandb_logger = wandb_logger
        self.datasets_names = datasets_names
        self.train_dataset = None
        self.val_dataset = None

    def forward(self, x):
        return self.policy(x)

    def concatenated_forward(
        self,
        model: micro_lm_model.LMDecoder,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        mask = boolean_triangular_mask(
            concatenated_batch["concatenated_attention_mask"] != 0
        )
        all_logits = model.forward(
            concatenated_batch["concatenated_input_ids"],
            mask=mask,
        ).to(torch.float32)
        all_logps = _get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :]
        return chosen_logps, rejected_logps

    def get_batch_metrics(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        loss_config: LossConfig,
        train=True,
    ):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = "train" if train else "eval"

        if loss_config.loss_name in {"dpo", "ipo"}:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(
                self.policy, batch
            )
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = (
                    self.concatenated_forward(self.reference, batch)
                )

            if loss_config.loss_name == "dpo":
                loss_kwargs = {
                    "beta": loss_config.loss_beta,
                    "reference_free": loss_config.reference_free,
                    "label_smoothing": loss_config.label_smoothing,
                    "ipo": False,
                }
            elif loss_config.loss_name == "ipo":
                loss_kwargs = {"beta": loss_config.loss_beta, "ipo": True}
            else:
                raise ValueError(f"unknown loss {loss_config.loss_name}")

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                **loss_kwargs,
            )

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            # chosen_rewards = all_gather_concat_pl(self,chosen_rewards)
            # rejected_rewards = all_gather_concat_pl(self,rejected_rewards)
            # reward_accuracies = all_gather_concat_pl(self,reward_accuracies)

            metrics[f"rewards_{train_test}/chosen"] = (
                chosen_rewards.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/rejected"] = (
                rejected_rewards.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/accuracies"] = (
                reward_accuracies.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/margins"] = (
                (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            )

            # policy_rejected_logps = all_gather_concat_pl(self,policy_rejected_logps.detach())
            metrics[f"logps_{train_test}/rejected"] = (
                policy_rejected_logps.cpu().numpy().tolist()
            )

        elif loss_config.loss_name == "sft":
            policy_chosen_logits = self.policy(
                batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]
            ).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(
                policy_chosen_logits, batch["chosen_labels"], average_log_prob=False
            )

            losses = -policy_chosen_logps

        # policy_chosen_logps = all_gather_concat_pl(self,policy_chosen_logps.detach())
        metrics[f"logps_{train_test}/chosen"] = (
            policy_chosen_logps.cpu().numpy().tolist()
        )

        # all_devices_losses = all_gather_concat_pl(self,losses.detach())
        all_devices_losses = losses
        metrics[f"loss/{train_test}"] = all_devices_losses.cpu().detach().numpy().tolist()

        return losses.mean(), metrics

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[List[str], List[str]]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        policy_chat = lm_sampler.LMSampler(
            self.policy, self.tokenizer, torch.bfloat16, self.device
        )
        reference_chat = lm_sampler.LMSampler(
            self.reference, self.tokenizer, torch.bfloat16, self.device
        )
        # b, s = batch["chosen_input_ids"].shape
        # generation_max_length = (
        #     self.config.max_seq_len - batch["chosen_input_ids"].shape[1]
        # )
        # print(batch)
        policy_output = policy_chat.generate_ids(
            batch["prompt_input_ids"],
            config=lm_sampler.GenerationConfig(
                generation_max_length=256, temperature=0.9
            ),
        )
        policy_output = pad_to_length(
            policy_output, self.config.max_seq_len, self.tokenizer.pad_token_id
        )
        # print(policy_output)
        # policy_output = all_gather_concat_pl(self,policy_output)
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True
        )
        # print(policy_output_decoded)

        if self.loss_config.loss_name in {"dpo", "ipo"}:
            reference_output = reference_chat.generate_ids(
                batch["prompt_input_ids"],
                config=lm_sampler.GenerationConfig(
                    generation_max_length=256, temperature=0.9
                ),
            )
            reference_output = pad_to_length(
                reference_output, self.config.max_seq_len, self.tokenizer.pad_token_id
            )
            # reference_output = all_gather_concat_pl(self,reference_output)
            reference_output_decoded = self.tokenizer.batch_decode(
                reference_output, skip_special_tokens=True
            )
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.get_batch_metrics(batch, self.loss_config, train=False)
        result = {
            "loss": loss,
            **metrics,
        }
        if self.batch_idx % 10 == 0:
            policy_samples, reference_samples = self.get_batch_samples(batch)
            if self.local_rank == 0:
                prompts = self.tokenizer.batch_decode(batch["prompt_input_ids"], skip_special_tokens=True)
                min_len = min(len(prompts), len(policy_samples), len(reference_samples))
                df = pd.DataFrame(
                    {
                        "prompt": prompts[:min_len],
                        "policy_samples": policy_samples[:min_len],
                        "reference_samples": reference_samples[:min_len],
                    }
                )
                self.wandb_logger.log_table("samples", dataframe=df)
        return result

    def training_step(self, batch, batch_idx):
        loss, metrics = self.get_batch_metrics(batch, self.loss_config, train=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.local_rank == 0:
            print(metrics)
        metrics_mean = {k: sum(v) / len(v) for k, v in metrics.items()}
        self.log_dict(metrics_mean, on_step=True, rank_zero_only=True)
        return loss

    def setup(self, stage: str = "fit"):
        split = "test"
        if stage == "fit":
            split = "train"

        flat_data = []
        for name in self.datasets_names:
            truncation_mode = "keep_end"
            if name == "hh":
                truncation_mode = "keep_start"
            data = dpo_lm_data.get_dataset(name, split)
            for prompt, responses in data.items():
                flat_data.append(
                    (
                        prompt,
                        responses["responses"],
                        responses["pairs"],
                        responses["sft_target"],
                        truncation_mode,
                    )
                )
        if self.train_dataset is None:
            self.train_dataset = dpo_lm_data.DPODataset(
                flat_data, self.tokenizer, self.config.max_seq_len, self.config.max_prompt_len
            )
        if self.val_dataset is None:
            self.val_dataset = dpo_lm_data.DPODataset(
                flat_data, self.tokenizer, self.config.max_seq_len, self.config.max_prompt_len
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            collate_fn=dpo_lm_data.get_collate_fn(self.tokenizer),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            collate_fn=dpo_lm_data.get_collate_fn(self.tokenizer),
        )

    def configure_optimizers(self):
        adam_factory = lambda params: torch.optim.Adam(params, lr=self.config.lr)
        scheduler_factory = (
            lambda optimizer: ml_utils.optim.LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.trainer.estimated_stepping_batches,
            )
        )
        return ml_utils.optim.escape_non_decay(
            self.policy, adam_factory, scheduler_factory, self.config.weight_decay
        )


def main():
    dpo_config, loss_config = parse_args()

    # Load the policy and reference models
    wandb_logger = WandbLogger(project=dpo_config.project_name)
    wandb_logger.log_hyperparams(dpo_config.__dict__)

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        max_epochs=dpo_config.max_epochs,
        gradient_clip_val=dpo_config.gradient_clip_val,
        accumulate_grad_batches=dpo_config.accumulate_grad_batches,
        logger=wandb_logger,
        precision="bf16-mixed",
        log_every_n_steps=1,
        val_check_interval=dpo_config.val_check_interval,
    )

    tokenizer = lm_tokenizer.load_tokenizer()
    policy = micro_lm_model.load_model_from_path(dpo_config.model_path)
    reference = deepcopy(policy)
    disable_dropout(policy)

    # Initialize the DPOModule
    model_wrapper = DPOModule(
        policy, reference, tokenizer, dpo_config, loss_config, wandb_logger, dpo_config.datasets_names
    )

    # Train the model
    trainer.fit(model_wrapper)
    micro_lm_model.save_model(policy, dpo_config.output_path)


if __name__ == "__main__":
    main()
