from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from .config import FullFrameDiffusionConfig
from .model import FullFrameConditionalDiffusion


GIB = 1024**3
MEMORY_LIMIT_GIB = 15.5


def reset_peak(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device: torch.device) -> tuple[float, float]:
    torch.cuda.synchronize(device)
    return (
        torch.cuda.max_memory_allocated(device) / GIB,
        torch.cuda.max_memory_reserved(device) / GIB,
    )


def probe(config: FullFrameDiffusionConfig, sampling_steps: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Memory probe requires a CUDA GPU")
    device = torch.device("cuda")
    model = FullFrameConditionalDiffusion(config).to(device).train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    amp_enabled = config.training.use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    osisaf = torch.rand((1, 1, *config.geometry.source_shape), device=device)
    masam2 = torch.rand((1, 1, *config.geometry.target_shape), device=device)

    reset_peak(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        "cuda",
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        loss = model.training_loss(osisaf, masam2)["loss"]
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    optimizer.step()
    scaler.update()
    train_allocated, train_reserved = peak_memory(device)

    optimizer.zero_grad(set_to_none=True)
    del loss
    model.eval()
    initial_noise = torch.randn(
        (1, config.geometry.latent_channels, *config.geometry.latent_shape),
        device=device,
    )
    reset_peak(device)
    with torch.no_grad(), torch.amp.autocast(
        "cuda",
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        prediction = model.sample(
            osisaf,
            sampling_steps=sampling_steps,
            initial_noise=initial_noise,
        )
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Memory-probe sampling produced NaN or Inf")
    sample_allocated, sample_reserved = peak_memory(device)

    maximum = max(train_allocated, sample_allocated)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Parameters: {parameter_count:,}")
    print(
        f"Train peak: allocated={train_allocated:.2f} GiB, "
        f"reserved={train_reserved:.2f} GiB"
    )
    print(
        f"DDIM-{sampling_steps} peak: allocated={sample_allocated:.2f} GiB, "
        f"reserved={sample_reserved:.2f} GiB"
    )
    if maximum > MEMORY_LIMIT_GIB:
        raise RuntimeError(
            f"Peak allocated memory {maximum:.2f} GiB exceeds "
            f"the {MEMORY_LIMIT_GIB:.1f} GiB experiment limit"
        )
    print(f"Result: fits the {MEMORY_LIMIT_GIB:.1f} GiB experiment limit")


def main() -> None:
    defaults = FullFrameDiffusionConfig()
    parser = argparse.ArgumentParser(
        description="Measure experiment-4 training and DDIM sampling memory."
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=defaults.network.base_channels,
    )
    parser.add_argument(
        "--shuffle-factor",
        type=int,
        choices=(1, 2, 4),
        default=defaults.geometry.shuffle_factor,
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=defaults.diffusion.sampling_steps,
    )
    parser.add_argument("--disable-attention", action="store_true")
    args = parser.parse_args()

    config = defaults
    config.geometry = replace(config.geometry, shuffle_factor=args.shuffle_factor)
    config.network = replace(
        config.network,
        base_channels=args.base_channels,
        use_pooled_attention=not args.disable_attention,
    )
    config.diffusion = replace(config.diffusion, sampling_steps=args.sampling_steps)
    config.validate()
    probe(config, sampling_steps=args.sampling_steps)


if __name__ == "__main__":
    main()
