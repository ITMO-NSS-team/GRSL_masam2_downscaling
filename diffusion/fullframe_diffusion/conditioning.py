from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConditionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class OSISAFConditionEncoder(nn.Module):
    """Encode native-grid OSISAF into a source-grid feature pyramid."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        levels: int = 4,
    ) -> None:
        super().__init__()
        if levels <= 0:
            raise ValueError("levels must be positive")

        channels = [base_channels * (2**level) for level in range(levels)]
        blocks = []
        current_channels = in_channels
        for level, out_channels in enumerate(channels):
            blocks.append(
                ConditionBlock(
                    current_channels,
                    out_channels,
                    downsample=level > 0,
                )
            )
            current_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.output_channels = tuple(channels)

    def forward(self, osisaf: torch.Tensor) -> list[torch.Tensor]:
        if osisaf.ndim != 4:
            raise ValueError("OSISAF must have shape [B, C, H, W]")
        features = []
        hidden = osisaf
        for block in self.blocks:
            hidden = block(hidden)
            features.append(hidden)
        return features


def resize_condition_pyramid(
    source_features: list[torch.Tensor],
    target_sizes: list[tuple[int, int]],
) -> list[torch.Tensor]:
    """Resize each native-grid condition level to its target U-Net resolution."""
    if len(source_features) != len(target_sizes):
        raise ValueError("Condition feature and target-size counts differ")

    resized = []
    for feature, target_size in zip(source_features, target_sizes):
        if feature.shape[-2:] == target_size:
            resized.append(feature)
        else:
            resized.append(
                F.interpolate(
                feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            )
    return resized
