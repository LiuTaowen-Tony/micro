import copy
import torch
import torch.nn as nn
import dataclasses
from torch import FloatTensor, LongTensor, Tensor, BoolTensor
from typing import Optional, Union
import __future__


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = int(hidden_size * 2.68)

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        gate_proj = nn.functional.silu(self.gate_proj(x))
        up_proj = self.up_proj(x)
        down_proj = self.down_proj(gate_proj * up_proj)
        return down_proj


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        output_proj: nn.Linear,
        pos_embeddings: nn.Module = None,
        max_seq_len: int = 4096,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({embed_dim}) must be between 0.0 and 1.0")

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Set layers
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.pos_embeddings = pos_embeddings

    def forward(
        self,
        x: torch.FloatTensor,
        *,
        mask: BoolTensor = None,
        input_pos: torch.LongTensor = None,
        kv_cache: "KVCache",
    ) -> torch.FloatTensor:
        """
            mask (Optional[Tensor]): Optional boolean tensor which contains the attention mask
                with shape [batch_size x seq_length x seq_length]. This is applied after
                the query-key multiplication and before the softmax. A value of True in row i
                and column j means token i attends to token j. A value of False means token i
                does not attend to token j. If no mask is specified, a causal mask
                is used by default. Default is None.
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - n_kv: num kv heads
            - d: embed dim
            - h_d: head dim

        TODO:
            - Return the attention weights
            - Make application of positional embeddings optional
        """
        # Get shape
        bsz, seq_len, _ = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        # q has shape [b, s, num_heads * head_dim]
        # k has shape [b, s, num_kv_heads * head_dim]
        # v has shape [b, s, num_kv_heads * head_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads

        # q: [b, s, n_kv, q_per_kv, h_d]
        # k: [b, s, n_kv, 1, h_d]
        # v: [b, s, n_kv, 1, h_d]
        q = q.view(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
        k = k.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)
        v = v.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)

        # if needed, expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        if self.num_heads != self.num_kv_heads:
            k = k.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
            v = v.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)

        # llama2 applies the RoPE embeddings on tensors with shape
        # [b, s, n_h, h_d]
        # Reshape the tensors before we apply RoPE
        q = q.reshape(bsz, seq_len, -1, self.head_dim)
        k = k.reshape(bsz, seq_len, -1, self.head_dim)
        v = v.reshape(bsz, seq_len, -1, self.head_dim)

        # Apply positional embeddings
        q = self.pos_embeddings(q, input_pos=input_pos)
        k = self.pos_embeddings(k, input_pos=input_pos)

        # [b, n_h, s, h_d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Update key-value cache
        if kv_cache is not None:
            k, v = kv_cache.update(input_pos, k, v)

        # shape: [b, 1, s, s]
        if mask is not None:
            if mask.dim() != 3:
                raise ValueError(
                    f"Mask should have shape [b x s x s], found {mask.shape}"
                )
            mask = mask[:, None, :, :]

        # Flash attention from https://pytorch.org/blog/accelerating-large-language-models/
        output = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout,
            is_causal=kv_cache is None and mask is None,
        )

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.output_proj(output)


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: Tensor, *, input_pos: Optional[Tensor] = None) -> Tensor:
        """
        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class KVCache(nn.Module):
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache",
            torch.zeros(cache_shape, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(cache_shape, dtype=dtype, device=device),
            persistent=False,
        )
        self.batch_size = batch_size

    def reset(self) -> None:
        """Reset the cache to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()

    def update(
        self, input_pos: Tensor, k_val: Tensor, v_val: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            input_pos (Tensor): Current position tensor with shape [S]
            k_val (Tensor): Current key tensor with shape [B, H, S, D]
            v_val (Tensor): Current value tensor with shape [B, H, S, D]
        Returns:
            Tuple[Tensor, Tensor]: Updated KV cache with key first
        """
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        attn: SelfAttention,
        mlp: nn.Module,
        sa_norm: nn.Module,
        mlp_norm: nn.Module,
    ) -> None:
        super().__init__()
        self.sa_norm = sa_norm
        self.attn = attn
        self.mlp_norm = mlp_norm
        self.mlp = mlp

    def forward(
        self,
        x: Tensor,
        *,
        mask: Optional[BoolTensor] = None,
        input_pos: Optional[LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        # [b, s, d]
        # Norm applied before self-attention
        attn_out = self.attn(
            self.sa_norm(x), mask=mask, input_pos=input_pos, kv_cache=kv_cache
        )

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = attn_out + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + mlp_out
        return out


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


@dataclasses.dataclass
class TransformerDecoderConfig:
    num_heads: int = 16
    head_dim: int = 64
    num_kv_heads: int = 8
    max_seq_len: int = 1024
    attn_dropout: float = 0.1
    vocab_size: int = 32000
    num_layers: int = 10
    attn_bias: bool = False
    ffn_bias: bool = True
    pos_emb_base: int = 10000
    pos_emb_max_seq_len: int = 4096

    def build_model(self) -> "TransformerDecoder":
        num_heads = self.num_heads
        head_dim = self.head_dim
        num_kv_heads = self.num_kv_heads
        max_seq_len = self.max_seq_len
        attn_dropout = self.attn_dropout
        vocab_size = self.vocab_size
        num_layers = self.num_layers

        hidden_size = num_heads * head_dim
        attn = SelfAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(hidden_size, num_heads * head_dim, bias=self.attn_bias),
            k_proj=nn.Linear(hidden_size, num_kv_heads * head_dim, bias=self.attn_bias),
            v_proj=nn.Linear(hidden_size, num_kv_heads * head_dim, bias=self.attn_bias),
            output_proj=nn.Linear(hidden_size, hidden_size, bias=self.attn_bias),
            pos_embeddings=RotaryPositionalEmbeddings(
                head_dim, self.pos_emb_max_seq_len, self.pos_emb_base
            ),
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        layer = TransformerDecoderLayer(
            attn=attn,
            mlp=MLP(hidden_size),
            sa_norm=nn.RMSNorm(hidden_size),
            mlp_norm=nn.RMSNorm(hidden_size),
        )
        decoder = TransformerDecoder(
            tok_embeddings=torch.nn.Embedding(vocab_size, hidden_size),
            layer=layer,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            norm=nn.RMSNorm(hidden_size),
            config=self,
            lm_head=nn.Linear(hidden_size, vocab_size),
        )
        return decoder


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        tok_embeddings: nn.Embedding,
        layer: TransformerDecoderLayer,
        num_layers: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        lm_head: nn.Module,
        config: TransformerDecoderConfig,
    ) -> None:
        super().__init__()

        self.tok_embeddings = tok_embeddings
        self.layers = _get_clones(layer, num_layers)
        self.norm = norm
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        causal_mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool,))
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.lm_head = lm_head
        self.config = config

    def get_kv_caches(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> tuple[BoolTensor, KVCache]:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
        """
        kv_caches = []
        for layer in self.layers:
            kv_caches.append(
                KVCache(
                    batch_size=batch_size,
                    max_seq_len=self.max_seq_len,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    dtype=dtype,
                    device=device,
                )
            )
        return kv_caches

    def forward(
        self,
        tokens: Union[LongTensor, FloatTensor],
        *,
        mask: Optional[BoolTensor] = None,
        input_pos: Optional[LongTensor] = None,
        kv_caches: Optional[list[KVCache]] = None,
    ) -> Tensor:
        """
        Args:
            tokens (LongTensor): Input tensor with shape [b x s] where b is the batch size
            or tokens (FloatTensor): Input tensor with shape [b x s x d] where d is the embedding dimension

            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.


        Note:
            In training mode, provide tokens, optionally mask.

            In inference mode, provide tokens, input_pos and kv_caches. kv_caches will be initialised when
            prefilling. Then increment input_pos and kv_caches in each decoding step.  At the very first step of inference, when the model is provided with a prompt,
            ``input_pos`` would contain the positions of all of the tokens in the prompt
            (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
            KV values for each position.

        Returns:
            Tensor: output tensor with shape [b x s x v]
        """
        # input tensor of shape [b, s]
        # bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        h = tokens
        if tokens.dtype == torch.long:
            h = self.tok_embeddings(tokens)

        if kv_caches is not None:
            if input_pos is None:
                raise ValueError(
                    "Caches are setup, but the position of input token is missing"
                )
            if mask is not None:
                raise ValueError(
                    "Mask is automatically set. Do not supply it again. Cannot use a non-causal mask for inference"
                )
            # shape: [1, input_pos_len, m_s]
            # in most cases input_pos_len should be 1
            mask = self.causal_mask[None, input_pos]

        for layer, kv_cache in zip(self.layers, kv_caches):
            # shape: [b, s, d]
            h = layer(h, mask=mask, input_pos=input_pos, kv_cache=kv_cache)

        # shape: [b, s, d]
        h = self.norm(h)

        output = self.lm_head(h)
        return output
