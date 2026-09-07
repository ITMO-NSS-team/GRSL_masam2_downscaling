from __future__ import annotations

from pathlib import Path

import torch

from .config import FullFrameDiffusionConfig
from .model import FullFrameConditionalDiffusion


def resolve_device(device_name: str = "auto") -> torch.device:
    """Resolve an inference device and fail clearly when CUDA is unavailable."""
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_name)


def create_model(
    config: FullFrameDiffusionConfig,
    device: torch.device,
) -> FullFrameConditionalDiffusion:
    return FullFrameConditionalDiffusion(config).to(device)


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[FullFrameConditionalDiffusion, dict]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "config" not in checkpoint:
        raise KeyError("Checkpoint does not contain a full-frame diffusion config")
    prediction_type = checkpoint["config"].get("diffusion", {}).get(
        "prediction_type"
    )
    if prediction_type != "v":
        raise ValueError(
            "This source version supports experiment-4 v-prediction checkpoints only. "
            "Use the previous code archive for experiments 1-3."
        )
    config = FullFrameDiffusionConfig.from_dict(checkpoint["config"])
    model = create_model(config, device)
    state = checkpoint["model_state_dict"]
    if use_ema and "ema_model_state_dict" in checkpoint:
        state = checkpoint["ema_model_state_dict"]
    elif use_ema and "ema_state_dict" in checkpoint:
        state = dict(state)
        state.update(checkpoint["ema_state_dict"]["shadow"])
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def create_initial_noise(
    model: FullFrameConditionalDiffusion,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Create reproducible initial diffusion noise for a model batch."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        (
            batch_size,
            model.geometry.latent_channels,
            *model.geometry.latent_shape,
        ),
        generator=generator,
        device=device,
    )
