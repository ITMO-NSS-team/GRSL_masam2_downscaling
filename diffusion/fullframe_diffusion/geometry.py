from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_to_shape(
    tensor: torch.Tensor,
    padded_shape: tuple[int, int],
    value: float = 0.0,
) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    padded_height, padded_width = padded_shape
    if height > padded_height or width > padded_width:
        raise ValueError("Tensor is larger than requested padded shape")
    padding = (0, padded_width - width, 0, padded_height - height)
    return F.pad(tensor, padding, mode="constant", value=value)


def crop_to_shape(tensor: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return tensor[..., : shape[0], : shape[1]]
