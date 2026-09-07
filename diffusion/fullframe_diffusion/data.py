from __future__ import annotations

import numpy as np
import torch


def validate_fullframe_batch(
    osisaf: torch.Tensor,
    masam2: torch.Tensor | None,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> None:
    if osisaf.ndim != 4 or osisaf.shape[1] != 1:
        raise ValueError("OSISAF batch must have shape [B, 1, H, W]")
    if tuple(osisaf.shape[-2:]) != source_shape:
        raise ValueError(
            f"OSISAF shape {tuple(osisaf.shape[-2:])} does not match {source_shape}"
        )
    if not torch.isfinite(osisaf).all():
        raise ValueError("OSISAF batch contains NaN or infinite values")
    if osisaf.min() < 0.0 or osisaf.max() > 1.0:
        raise ValueError("OSISAF batch must be normalized to [0, 1]")

    if masam2 is None:
        return
    if masam2.ndim != 4 or masam2.shape[1] != 1:
        raise ValueError("MASAM2 batch must have shape [B, 1, H, W]")
    if tuple(masam2.shape[-2:]) != target_shape:
        raise ValueError(
            f"MASAM2 shape {tuple(masam2.shape[-2:])} does not match {target_shape}"
        )
    if not torch.isfinite(masam2).all():
        raise ValueError("MASAM2 batch contains NaN or infinite values")
    if masam2.min() < 0.0 or masam2.max() > 1.0:
        raise ValueError("MASAM2 batch must lie within [0, 1]")
