from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2
from typing import Optional, Tuple

from ml_utils.misc import load_with_config


@dataclass
class MultiLevelVQVAEConfig:
    in_channels: int
    hidden_channels: int
    res_channels: int
    num_res_layers: int
    num_levels: int
    embed_dim: int
    num_entries: int
    scaling_rates: list[int]

    def build_model(self) -> "MultiLevelVQVAE":
        return MultiLevelVQVAE(config=self, **self.__dict__)


@dataclass
class TrainedMLVQVAEConfig:
    path: str = "trained_models/multi_level_vqvae/cifar10_earnest-energy-4"

    def build_model(self) -> "MultiLevelVQVAE":
        return load_with_config(MultiLevelVQVAEConfig, self.path)

class ReZero(torch.nn.Module):
    def __init__(self, in_channels: int, res_channels: int):
        super(ReZero, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x


class ResidualStack(torch.nn.Module):
    def __init__(self, in_channels: int, res_channels: int, nb_layers: int):
        super().__init__()
        self.stack = nn.Sequential(
            *[ReZero(in_channels, res_channels) for _ in range(nb_layers)]
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)


class Encoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        res_channels: int,
        nb_res_layers: int,
        downscale_factor: int,
    ):
        super().__init__()
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(c_channel, n_channel, 4, stride=2, padding=1),
                    nn.BatchNorm2d(n_channel),
                    nn.ReLU(inplace=True),
                )
            )
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


class Decoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        res_channels: int,
        nb_res_layers: int,
        upscale_factor: int,
    ):
        super().__init__()
        assert log2(upscale_factor) % 1 == 0, "Downscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        layers = [nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(ResidualStack(hidden_channels, res_channels, nb_res_layers))
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                    nn.BatchNorm2d(n_channel),
                    nn.ReLU(inplace=True),
                )
            )
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        # layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


"""
    Almost directly taken from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    No reason to reinvent this rather complex mechanism.

    Essentially handles the "discrete" part of the network, and training through EMA rather than 
    third term in loss function.
"""


class CodeLayer(torch.nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, nb_entries: int):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, embed_dim, 1)

        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5

        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer(
            "cluster_size", torch.zeros(nb_entries, dtype=torch.float32)
        )
        self.register_buffer("embed_avg", embed.clone())

    def vector_quantize(self, x: torch.FloatTensor) -> torch.LongTensor:
        x = self.conv_in(x.float()).permute(0, 2, 3, 1)
        flatten = x.reshape(-1, self.dim)
        dist: torch.Tensor = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        return embed_ind.view(*x.shape[:-1])

    def vector_quantize_dist(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        x = self.conv_in(x.float()).permute(0, 2, 3, 1)
        flatten = x.reshape(-1, self.dim)
        dist: torch.Tensor = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        return dist, embed_ind.view(*x.shape[:-1])

    def forward(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:
        x = self.conv_in(x.float()).permute(0, 2, 3, 1)
        flatten = x.reshape(-1, self.dim)
        dist: torch.Tensor = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # TODO: Replace this? Or can we simply comment out?
            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    def embed_code(self, embed_id: torch.LongTensor) -> torch.FloatTensor:
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class Upscaler(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        scaling_rates: list[int],
    ):
        super().__init__()

        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(log2(sr))
            layers = []
            for _ in range(upscale_steps):
                layers.append(
                    nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
                )
                layers.append(nn.BatchNorm2d(embed_dim))
                layers.append(nn.ReLU(inplace=True))
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor, stage: int) -> torch.FloatTensor:
        return self.stages[stage](x)


"""
    Main VQ-VAE-2 Module, capable of support arbitrary number of levels
    TODO: A lot of this class could do with a refactor. It works, but at what cost?
    TODO: Add disrete code decoding function
"""


class MultiLevelVQVAE(torch.nn.Module):

    def __init__(
        self,
        config: MultiLevelVQVAEConfig,
        in_channels: int = 3,
        hidden_channels: int = 128,
        res_channels: int = 32,
        num_res_layers: int = 2,
        num_levels: int = 3,
        embed_dim: int = 64,
        num_entries: int = 512,
        scaling_rates: list[int] = [8, 4, 2],
    ):
        super().__init__()
        self.config = config
        self.num_levels = num_levels
        assert (
            len(scaling_rates) == num_levels
        ), "Number of scaling rates not equal to number of levels!"

        self.encoders = nn.ModuleList(
            [
                Encoder(
                    in_channels,
                    hidden_channels,
                    res_channels,
                    num_res_layers,
                    scaling_rates[0],
                )
            ]
        )
        for i, sr in enumerate(scaling_rates[1:]):
            self.encoders.append(
                Encoder(
                    hidden_channels, hidden_channels, res_channels, num_res_layers, sr
                )
            )

        self.codebooks = nn.ModuleList()
        for i in range(num_levels - 1):
            self.codebooks.append(
                CodeLayer(hidden_channels + embed_dim, embed_dim, num_entries)
            )
        self.codebooks.append(CodeLayer(hidden_channels, embed_dim, num_entries))

        self.decoders = nn.ModuleList(
            [
                Decoder(
                    embed_dim * num_levels,
                    hidden_channels,
                    in_channels,
                    res_channels,
                    num_res_layers,
                    scaling_rates[0],
                )
            ]
        )
        for i, sr in enumerate(scaling_rates[1:]):
            self.decoders.append(
                Decoder(
                    embed_dim * (num_levels - 1 - i),
                    hidden_channels,
                    embed_dim,
                    res_channels,
                    num_res_layers,
                    sr,
                )
            )

        self.upscalers = nn.ModuleList()
        for i in range(num_levels - 1):
            self.upscalers.append(
                Upscaler(embed_dim, scaling_rates[1 : len(scaling_rates) - i][::-1])
            )




    def embeddings_to_code_ids(
        self, x: torch.FloatTensor, level: int
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Args:
            x (torch.FloatTensor): Input tensor, shape [B, C, H, W]
            level (int): Level of the codebook

        Returns:
            tuple[torch.FloatTensor, torch.LongTensor]: Quantized tensor and code ids
        """
        if level >= self.num_levels:
            raise ValueError("Level out of range")
        quantized_code, diff, code_ids = self.codebooks[level](x)
        return quantized_code, code_ids

    def code_ids_to_embeddings(
        self, x: torch.FloatTensor, level: int
    ) -> torch.FloatTensor:
        """
        Args:
            x (torch.FloatTensor): Input tensor, shape [B, C, H, W]
            level (int): Level of the codebook

        Returns:
            torch.FloatTensor: Quantized tensor
        """
        if level >= self.num_levels:
            raise ValueError("Level out of range")
        return self.codebooks[level].embed_code(x).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.FloatTensor):
        encoder_outputs = []
        code_outputs = []
        decoder_outputs = []
        upscale_counts = []
        id_outputs = []
        diffs = []
        # encoder outputs : [xxxx, xx, x]

        for enc in self.encoders:
            if len(encoder_outputs):
                encoder_outputs.append(enc(encoder_outputs[-1]))
            else:
                encoder_outputs.append(enc(x))

        for l in range(self.num_levels - 1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]

            if len(decoder_outputs):  # if we have previous levels to condition on
                code_q, code_d, emb_id = codebook(
                    torch.cat([encoder_outputs[l], decoder_outputs[-1]], axis=1)
                )
            else:
                code_q, code_d, emb_id = codebook(encoder_outputs[l])
            diffs.append(code_d)
            id_outputs.append(emb_id)

            code_outputs = [
                self.upscalers[i](c, upscale_counts[i])
                for i, c in enumerate(code_outputs)
            ]
            upscale_counts = [u + 1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1], diffs, encoder_outputs, decoder_outputs, id_outputs

    def decode_codes(
        self, cs: list[torch.LongTensor]
    ) -> torch.FloatTensor:
        """
        Args:
            cs (list[torch.LongTensor]): List of code tensors

        Returns:
            torch.FloatTensor: Decoded image

        Note:
            code should be in ascending order of levels
            [level0, level1, level2, ...]
            level0 has the largest resolution
        """
        decoder_outputs = []
        code_outputs = []
        upscale_counts = []

        for l in range(self.num_levels - 1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]

            code_q = codebook.embed_code(cs[l]).permute(0, 3, 1, 2)
            code_outputs = [
                self.upscalers[i](c, upscale_counts[i])
                for i, c in enumerate(code_outputs)
            ]
            upscale_counts = [u + 1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1]


_ffhq1024 = {
    "display_name": "FFHQ1024",
    "image_shape": (3, 1024, 1024),
    "in_channels": 3,
    "hidden_channels": 128,
    "res_channels": 32,
    "nb_res_layers": 2,
    "embed_dim": 64,
    "nb_entries": 512,
    "nb_levels": 3,
    "scaling_rates": [8, 2, 2],
    "learning_rate": 1e-4,
    "beta": 0.25,
    "batch_size": 8,
    "mini_batch_size": 8,
    "max_epochs": 100,
}

_ffhq1024_large = {
    "display_name": "FFHQ1024 VQ-VAE++",
    "image_shape": (3, 1024, 1024),
    "in_channels": 3,
    "hidden_channels": 128,
    "res_channels": 32,
    "nb_res_layers": 2,
    "embed_dim": 64,
    "nb_entries": 512,
    "nb_levels": 5,
    "scaling_rates": [4, 2, 2, 2, 2],
    "learning_rate": 1e-4,
    "beta": 0.25,
    "batch_size": 16,
    "mini_batch_size": 8,
    "max_epochs": 100,
}

_ffhq256 = {
    "display_name": "FFHQ256",
    "image_shape": (3, 256, 256),
    "in_channels": 3,
    "hidden_channels": 128,
    "res_channels": 64,
    "nb_res_layers": 2,
    "embed_dim": 64,
    "nb_entries": 512,
    "nb_levels": 2,
    "scaling_rates": [4, 2],
    "learning_rate": 1e-4,
    "beta": 0.25,
    "batch_size": 128,
    "mini_batch_size": 128,
    "max_epochs": 100,
}
_ffhq128 = {
    "display_name": "FFHQ128",
    "image_shape": (3, 128, 128),
    "in_channels": 3,
    "hidden_channels": 128,
    "res_channels": 32,
    "nb_res_layers": 2,
    "embed_dim": 256,
    "nb_entries": 512,
    "nb_levels": 2,
    "scaling_rates": [8, 4],
    "learning_rate": 1e-4,
    "beta": 0.25,
    "batch_size": 128,
    "mini_batch_size": 128,
    "max_epochs": 100,
}

_cifar10 = {
    "display_name": "CIFAR10",
    "image_shape": (3, 32, 32),
    "in_channels": 3,
    "hidden_channels": 128,
    "res_channels": 32,
    "nb_res_layers": 2,
    "embed_dim": 256,
    "nb_entries": 512,
    "nb_levels": 2,
    "scaling_rates": [2, 4],
    "learning_rate": 1e-3,
    "beta": 0.25,
    "batch_size": 128,
    "mini_batch_size": 128,
    "max_epochs": 100,
}

_mnist = {
    "display_name": "MNIST",
    "image_shape": (1, 28, 28),
    "in_channels": 1,
    "hidden_channels": 128,
    "res_channels": 32,
    "nb_res_layers": 2,
    "embed_dim": 32,
    "nb_entries": 128,
    "nb_levels": 2,
    "scaling_rates": [2, 2],
    "learning_rate": 1e-4,
    "beta": 0.25,
    "batch_size": 32,
    "mini_batch_size": 32,
    "max_epochs": 100,
}

_kmnist = {
    "display_name": "Kuzushiji-MNIST",
    "image_shape": (1, 28, 28),
    "in_channels": 1,
    "hidden_channels": 128,
    "res_channels": 32,
    "nb_res_layers": 2,
    "embed_dim": 32,
    "nb_entries": 128,
    "nb_levels": 2,
    "scaling_rates": [2, 2],
    "learning_rate": 1e-4,
    "beta": 0.25,
    "batch_size": 32,
    "mini_batch_size": 32,
    "max_epochs": 100,
}


def get_config_by_taskname(taskname: str) -> MultiLevelVQVAEConfig:
    HPS_VQVAE = {
        "ffhq1024": _ffhq1024,
        "ffhq1024-large": _ffhq1024_large,
        "ffhq256": _ffhq256,
        "ffhq128": _ffhq128,
        "cifar10": _cifar10,
        "mnist": _mnist,
        "kmnist": _kmnist,
    }
    return MultiLevelVQVAEConfig(
        in_channels=HPS_VQVAE[taskname]["in_channels"],
        hidden_channels=HPS_VQVAE[taskname]["hidden_channels"],
        res_channels=HPS_VQVAE[taskname]["res_channels"],
        num_res_layers=HPS_VQVAE[taskname]["nb_res_layers"],
        num_levels=HPS_VQVAE[taskname]["nb_levels"],
        embed_dim=HPS_VQVAE[taskname]["embed_dim"],
        num_entries=HPS_VQVAE[taskname]["nb_entries"],
        scaling_rates=HPS_VQVAE[taskname]["scaling_rates"],
    )


def get_model_by_taskname(taskname: str) -> MultiLevelVQVAE:
    config = get_config_by_taskname(taskname)
    return MultiLevelVQVAE(config, **config.__dict__)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nb_levels = 10
    net = MultiLevelVQVAE(num_levels=nb_levels, scaling_rates=[2] * nb_levels).to(
        device
    )

    x = torch.randn(1, 3, 1024, 1024).to(device)
    _, diffs, enc_out, dec_out = net(x)
    print("\n".join(str(y.shape) for y in enc_out))
    print()
    print("\n".join(str(y.shape) for y in dec_out))
    print()
    print("\n".join(str(y) for y in diffs))
