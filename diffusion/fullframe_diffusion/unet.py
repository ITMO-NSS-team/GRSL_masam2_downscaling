from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .conditioning import resize_condition_pyramid
from .config import NetworkConfig


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    maximum_period: int = 10_000,
) -> torch.Tensor:
    half = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(maximum_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        time = sinusoidal_timestep_embedding(timesteps, self.embedding_dim)
        return self.time_mlp(time)


class FiLMResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_channels: int,
        embedding_dim: int,
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.condition_projection = nn.Conv2d(condition_channels, out_channels * 2, 1)
        self.embedding_projection = nn.Linear(embedding_dim, out_channels * 2)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.output_projection = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        embedding: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(inputs)
        hidden = self.norm1(hidden)

        if condition.shape[-2:] != hidden.shape[-2:]:
            condition = F.interpolate(
                condition,
                size=hidden.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        film = self.condition_projection(condition)
        film = film + self.embedding_projection(embedding).unsqueeze(-1).unsqueeze(-1)
        scale, shift = film.chunk(2, dim=1)
        hidden = hidden * (1.0 + scale) + shift

        hidden = self.output_projection(
            self.dropout(F.silu(self.norm2(F.silu(hidden))))
        )
        return hidden + self.skip(inputs)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self,
        inputs: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        inputs = F.interpolate(inputs, size=target_size, mode="nearest")
        return self.conv(inputs)


class PooledSelfAttention(nn.Module):
    """Bottleneck attention with bounded token count for 16 GB GPUs."""

    def __init__(self, channels: int, heads: int, max_tokens: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("Attention channels must be divisible by attention heads")
        self.max_tokens = max_tokens
        self.norm = nn.GroupNorm(1, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        if height * width > self.max_tokens:
            scale = math.sqrt(self.max_tokens / (height * width))
            pooled_size = (
                max(1, int(height * scale)),
                max(1, int(width * scale)),
            )
            pooled = F.adaptive_avg_pool2d(inputs, pooled_size)
        else:
            pooled = inputs

        normalized = self.norm(pooled)
        tokens = normalized.flatten(2).transpose(1, 2)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        attended = attended.transpose(1, 2).reshape(
            batch, channels, pooled.shape[-2], pooled.shape[-1]
        )
        if attended.shape[-2:] != (height, width):
            attended = F.interpolate(
                attended,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        return inputs + self.output(attended)


class ConditionalDiffusionUNet(nn.Module):
    """RoughNet-style FiLM U-Net operating on an invertible full-frame latent."""

    def __init__(
        self,
        latent_channels: int,
        spatial_condition_channels: int,
        condition_channels: tuple[int, ...],
        config: NetworkConfig,
    ) -> None:
        super().__init__()
        level_channels = tuple(
            config.base_channels * multiplier
            for multiplier in config.channel_multipliers
        )
        if len(level_channels) != len(condition_channels):
            raise ValueError("U-Net and condition encoder level counts must match")

        self.config = config
        self.level_channels = level_channels
        self.embedding = TimeEmbedding(config.time_embedding_dim)
        self.input_projection = nn.Conv2d(
            latent_channels + spatial_condition_channels,
            level_channels[0],
            3,
            padding=1,
        )

        encoder_stages = []
        downsamples = []
        current_channels = level_channels[0]
        for level, (out_channels, cond_channels) in enumerate(
            zip(level_channels, condition_channels)
        ):
            blocks = []
            for block_index in range(config.blocks_per_level):
                blocks.append(
                    FiLMResidualBlock(
                        in_channels=(
                            current_channels if block_index == 0 else out_channels
                        ),
                        out_channels=out_channels,
                        condition_channels=cond_channels,
                        embedding_dim=config.time_embedding_dim,
                        groups=config.group_norm_groups,
                        dropout=config.dropout,
                    )
                )
            encoder_stages.append(nn.ModuleList(blocks))
            current_channels = out_channels
            if level < len(level_channels) - 1:
                downsamples.append(Downsample(current_channels))

        self.encoder_stages = nn.ModuleList(encoder_stages)
        self.downsamples = nn.ModuleList(downsamples)
        bottleneck_channels = level_channels[-1]
        bottleneck_condition = condition_channels[-1]
        self.middle1 = FiLMResidualBlock(
            bottleneck_channels,
            bottleneck_channels,
            bottleneck_condition,
            config.time_embedding_dim,
            config.group_norm_groups,
            config.dropout,
        )
        self.middle_attention = (
            PooledSelfAttention(
                bottleneck_channels,
                config.attention_heads,
                config.max_attention_tokens,
            )
            if config.use_pooled_attention
            else nn.Identity()
        )
        self.middle2 = FiLMResidualBlock(
            bottleneck_channels,
            bottleneck_channels,
            bottleneck_condition,
            config.time_embedding_dim,
            config.group_norm_groups,
            config.dropout,
        )

        decoder_stages = []
        upsamples = []
        current_channels = bottleneck_channels
        for reverse_index in range(len(level_channels) - 2, -1, -1):
            out_channels = level_channels[reverse_index]
            cond_channels = condition_channels[reverse_index]
            upsamples.append(Upsample(current_channels))
            blocks = [
                FiLMResidualBlock(
                    current_channels + out_channels,
                    out_channels,
                    cond_channels,
                    config.time_embedding_dim,
                    config.group_norm_groups,
                    config.dropout,
                )
            ]
            for _ in range(config.blocks_per_level - 1):
                blocks.append(
                    FiLMResidualBlock(
                        out_channels,
                        out_channels,
                        cond_channels,
                        config.time_embedding_dim,
                        config.group_norm_groups,
                        config.dropout,
                    )
                )
            decoder_stages.append(nn.ModuleList(blocks))
            current_channels = out_channels

        self.upsamples = nn.ModuleList(upsamples)
        self.decoder_stages = nn.ModuleList(decoder_stages)
        self.output_norm = nn.GroupNorm(config.group_norm_groups, level_channels[0])
        self.output_projection = nn.Conv2d(
            level_channels[0],
            latent_channels,
            3,
            padding=1,
        )

    def _run_block(
        self,
        block: FiLMResidualBlock,
        hidden: torch.Tensor,
        embedding: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training:
            return checkpoint(
                block,
                hidden,
                embedding,
                condition,
                use_reentrant=False,
            )
        return block(hidden, embedding, condition)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        spatial_condition: torch.Tensor,
        timesteps: torch.Tensor,
        source_condition_features: list[torch.Tensor],
    ) -> torch.Tensor:
        if spatial_condition.shape[0] != noisy_latent.shape[0]:
            raise ValueError("Spatial condition and noisy latent batch sizes differ")
        if spatial_condition.shape[-2:] != noisy_latent.shape[-2:]:
            raise ValueError("Spatial condition and noisy latent resolutions differ")
        embedding = self.embedding(timesteps)
        hidden = self.input_projection(
            torch.cat((noisy_latent, spatial_condition), dim=1)
        )

        target_sizes = []
        probe_size = hidden.shape[-2:]
        for _ in self.level_channels:
            target_sizes.append(tuple(probe_size))
            probe_size = tuple((size + 1) // 2 for size in probe_size)
        conditions = resize_condition_pyramid(
            source_condition_features,
            target_sizes,
        )

        skips = []
        for level, blocks in enumerate(self.encoder_stages):
            for block in blocks:
                hidden = self._run_block(block, hidden, embedding, conditions[level])
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)

        hidden = self._run_block(
            self.middle1, hidden, embedding, conditions[-1]
        )
        hidden = self.middle_attention(hidden)
        hidden = self._run_block(
            self.middle2, hidden, embedding, conditions[-1]
        )

        for decoder_index, (upsample, blocks) in enumerate(
            zip(self.upsamples, self.decoder_stages)
        ):
            skip_level = len(skips) - decoder_index - 2
            skip = skips[skip_level]
            hidden = upsample(hidden, tuple(skip.shape[-2:]))
            hidden = torch.cat((hidden, skip), dim=1)
            for block in blocks:
                hidden = self._run_block(
                    block, hidden, embedding, conditions[skip_level]
                )

        return self.output_projection(F.silu(self.output_norm(hidden)))
