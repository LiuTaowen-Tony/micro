import torch
import torch.nn as nn
import math
from .transformer import MLP as FeedForward
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    src_vocab_size: int = 512
    tgt_vocab_size: int = 512
    d_model: int = 256
    n_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    max_seq_len: int = 4096

    def build_model(self):
        return Transformer(self)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, is_causal: bool, mask=None, kv_cache=None):
        batch_size = q.size(0)
        
        # they need to be both None or not None
        assert (mask is not None) == (kv_cache is not None)
        # if both are None, is_causal -> Decoder, otherwise -> Encoder

        # if is_causal is True, mask should be None
        if is_causal:
            assert mask is not None

        # Linear projections
        q = (
            self.q_linear(q)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(k)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(v)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Key-value caching for generation
        if kv_cache is not None:
            if "k" in kv_cache and "v" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"] = k
            kv_cache["v"] = v

        # Scaled dot-product attention
        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.1,
            is_causal=is_causal,
            scale=self.scale,
        )

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_k)
        )
        return self.out(output), kv_cache


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(
            d_model,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        xnorm = self.norm1(x)
        attn_output, _ = self.attn(xnorm, xnorm, xnorm, is_causal=False)
        h = x + attn_output
        ff_output = self.ff(self.norm2(h))
        return h + ff_output


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None, kv_cache=None):
        xnorm = self.norm1(x)

        # they need to be both None or not None
        assert (tgt_mask is not None) == (kv_cache is not None)

        is_causal = tgt_mask is not None
        self_attn_output, kv_cache = self.self_attn(
            xnorm, xnorm, xnorm, is_causal=is_causal, mask=tgt_mask, kv_cache=kv_cache
        )
        h = x + self_attn_output
        if enc_output is not None:
            cross_attn_output, _ = self.cross_attn(
                self.norm2(h), enc_output, enc_output, is_causal=False
            )
            h = h + cross_attn_output
        ff_output = self.ff(self.norm3(h))
        return h + ff_output, kv_cache


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.encoder_embedding = nn.Embedding(config.src_vocab_size, config.d_model)
        self.decoder_embedding = nn.Embedding(config.tgt_vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_len)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    config.d_model,
                    config.n_heads,
                )
                for _ in range(config.num_encoder_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(config.d_model, config.n_heads)
                for _ in range(config.num_decoder_layers)
            ]
        )

        self.fc_out = nn.Linear(config.d_model, config.tgt_vocab_size)


    def make_tgt_mask(self, tgt):
        seq_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
        return tgt_mask.unsqueeze(0).unsqueeze(1)

    def encode(self, src):
        src = self.encoder_embedding(src) * math.sqrt(src.size(-1))
        src = self.pos_encoder(src)
        for layer in self.encoder_layers:
            src = layer(src)
        return src

    def decode(
        self, tgt, memory,  tgt_mask=None, kv_cache=None
    ):
        tgt = self.decoder_embedding(tgt) * math.sqrt(tgt.size(-1))
        tgt = self.pos_encoder(tgt)

        for layer in self.decoder_layers:
            tgt, kv_cache = layer(tgt, memory, tgt_mask, kv_cache)
        return tgt, kv_cache

    def forward(self, src, tgt):
        memory = None
        if src is not None:
            memory = self.encode(src)
        output, _ = self.decode(tgt, memory)
        return self.fc_out(output)

    # def generate(self, src, max_len, start_symbol):
    #     src_mask = self.make_src_mask(src)
    #     memory = self.encode(src, src_mask)

    #     ys = torch.ones((src.size(0), 1), device=src.device).fill_(start_symbol).long()
    #     kv_cache = [{} for _ in range(len(self.decoder_layers))]

    #     for _ in range(max_len - 1):
    #         tgt_mask = self.make_tgt_mask(ys)
    #         out, kv_cache = self.decode(ys, memory, src_mask, tgt_mask, kv_cache)
    #         prob = self.fc_out(out[:, -1])
    #         _, next_word = torch.max(prob, dim=1)
    #         ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
    #         if torch.all(next_word == 0):
    #             break

    #     return ys
