import torch
import dataclasses
from model import multi_level_vqvae
from model import encoder_decoder_transformer
from typing import Optional
import math


@dataclasses.dataclass
class TransformerArImGenConfig:
    tokenizer_config: multi_level_vqvae.TrainedMLVQVAEConfig = (
        multi_level_vqvae.TrainedMLVQVAEConfig()
    )
    transformer_config: encoder_decoder_transformer.TransformerConfig = (
        encoder_decoder_transformer.TransformerConfig()
    )
    level: int = 0
    num_levels: int = 2

    def build_model(self) -> "TransformerArImGen":
        vqvae = self.tokenizer_config.build_model()
        assert self.num_levels == vqvae.config.num_levels

        num_entries = vqvae.config.num_entries
        d_model = self.transformer_config.d_model
        embed_dim = vqvae.config.embed_dim
        src_vocab_size = self.transformer_config.src_vocab_size
        tgt_vocab_size = self.transformer_config.tgt_vocab_size

        assert src_vocab_size == num_entries
        assert tgt_vocab_size == num_entries

        transformer = self.transformer_config.build_model()

        if d_model == embed_dim:
            cond_proj = torch.nn.Identity()
            tgt_proj = torch.nn.Identity()
        else:
            cond_proj = torch.nn.Linear(vqvae.config.embed_dim, d_model)
            tgt_proj = torch.nn.Linear(vqvae.config.embed_dim, d_model)
        return TransformerArImGen(self, transformer, vqvae, cond_proj, tgt_proj)


class TransformerArImGen(torch.nn.Module):
    def __init__(
        self,
        config: TransformerArImGenConfig,
        transformer: encoder_decoder_transformer.Transformer,
        vqvae: multi_level_vqvae.MultiLevelVQVAE,
        cond_proj: torch.nn.Module,
        tgt_proj: torch.nn.Module,
    ):
        super().__init__()
        self.config = config
        self.vqvae = vqvae
        self.transformer = transformer
        self.cond_proj = cond_proj
        self.tgt_proj = tgt_proj
        self.loss = torch.nn.CrossEntropyLoss()
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def to_bhw(
        self, x: torch.LongTensor, shape: Optional[tuple[int, int]] = None
    ) -> torch.LongTensor:
        assert (
            x.dim() == 2 and x.dtype == torch.long
        ), "Input tensor must be 2D and long"
        if shape is not None:
            return x.view(-1, *shape)  # bsz, w, h
        bsz, n_tokens = x.shape
        l = math.sqrt(n_tokens)
        return x.view(bsz, l, l)

    def embed(
        self, code_ids: torch.LongTensor, shape: tuple[int, int], level
    ) -> torch.FloatTensor:
        bsz, seq_len = code_ids.shape
        code_ids = self.to_bhw(code_ids, shape)
        embed = self.vqvae.code_ids_to_embeddings(code_ids, level)
        embed = embed.permute(0, 2, 3, 1).reshape(
            bsz, seq_len, self.vqvae.config.embed_dim
        )
        return embed

    def class_embed(self, class_labels):
        bsz = class_labels.shape[0]
        return torch.nn.functional.one_hot(
            class_labels, self.vqvae.config.num_entries
        ).view(bsz, 1, -1)

    def forward(self, embed_cond, embed_tgt):
        embed_cond = self.cond_proj(embed_cond)
        embed_tgt = self.tgt_proj(embed_tgt)

        features = self.transformer.feature_transform(embed_cond, tgt=embed_tgt)
        logits = self.transformer.fc_out(features)
        return logits

    def train_forward(
        self,
        level_code_ids: torch.Tensor,
        shapes: list[tuple],
        class_label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l = self.config.level
        target = level_code_ids[l]
        bsz, sl = target.shape

        # train target is [0, x1, x2, x3, x4, x5]
        # train label is  [x1, x2, x3, x4, x5, -100]
        dims = (bsz, sl + 1)
        label_ids = torch.ones(dims, dtype=torch.long, device=target.device) * -100
        label_ids[:, :-1] = target

        dims = (bsz, sl + 1, self.vqvae.config.embed_dim)
        train_inputs = torch.zeros(dims, device=target.device)
        train_inputs[:, 1:] = self.embed(target, shapes[l], l)

        # if train top level
        if l == self.vqvae.config.num_levels - 1:
            cond_embed = self.class_embed(class_label)
        else:
            cond_embed = self.embed(level_code_ids[l + 1], shapes[l + 1], l + 1)

        logits = self.forward(cond_embed, train_inputs) 
        loss = self.loss(logits.view(-1, logits.size(-1)), label_ids.view(-1))
        code_ids = torch.argmax(logits, dim=-1)
        return loss, logits[:, :-1], code_ids[:, :-1]

    def decode_codes(self, level_code_ids, shapes):
        for i in range(len(level_code_ids)):
            level_code_ids[i] = self.to_bhw(level_code_ids[i], shapes[i])
        return self.vqvae.decode_codes(level_code_ids)

    def get_level_code_ids(self, images):
        _, _, _, _, level_code_ids = self.vqvae(images)
        return level_code_ids
