from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    cumulative = torch.cos(
        ((x / timesteps) + offset) / (1.0 + offset) * math.pi * 0.5
    ).pow(2)
    cumulative = cumulative / cumulative[0]
    betas = 1.0 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-8, 0.999).float()


def extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    shape: torch.Size,
) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Cosine DDPM training process and deterministic/stochastic DDIM sampling."""

    def __init__(self, timesteps: int, cosine_offset: float = 0.008) -> None:
        super().__init__()
        betas = cosine_beta_schedule(timesteps, cosine_offset)
        alphas = 1.0 - betas
        alpha_cumulative = torch.cumprod(alphas, dim=0)

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumulative", alpha_cumulative)
        self.register_buffer("sqrt_alphas_cumulative", alpha_cumulative.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumulative",
            (1.0 - alpha_cumulative).sqrt(),
        )

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumulative, timesteps, clean.shape) * clean
            + extract(self.sqrt_one_minus_alphas_cumulative, timesteps, clean.shape)
            * noise
        )

    def velocity_target(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Return the stable v-parameterization target."""
        return (
            extract(self.sqrt_alphas_cumulative, timesteps, clean.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumulative, timesteps, clean.shape)
            * clean
        )

    def predict_clean_from_velocity(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_velocity: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = extract(self.sqrt_alphas_cumulative, timesteps, noisy.shape)
        sqrt_one_minus_alpha = extract(
            self.sqrt_one_minus_alphas_cumulative, timesteps, noisy.shape
        )
        return sqrt_alpha * noisy - sqrt_one_minus_alpha * predicted_velocity

    def predict_noise_from_velocity(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_velocity: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = extract(self.sqrt_alphas_cumulative, timesteps, noisy.shape)
        sqrt_one_minus_alpha = extract(
            self.sqrt_one_minus_alphas_cumulative, timesteps, noisy.shape
        )
        return sqrt_one_minus_alpha * noisy + sqrt_alpha * predicted_velocity

    @torch.no_grad()
    def ddim_sample(
        self,
        denoise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: tuple[int, ...],
        device: torch.device,
        sampling_steps: int,
        eta: float = 0.0,
        initial_noise: torch.Tensor | None = None,
        clip_clean: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        if not 1 <= sampling_steps <= self.timesteps:
            raise ValueError("sampling_steps is outside the diffusion schedule")

        sample = (
            initial_noise
            if initial_noise is not None
            else torch.randn(shape, device=device)
        )
        if tuple(sample.shape) != shape:
            raise ValueError(
                "initial_noise shape does not match requested sample shape"
            )

        sequence = torch.linspace(
            self.timesteps - 1,
            0,
            sampling_steps,
            device=device,
        ).round().long()
        sequence = torch.unique_consecutive(sequence)

        for index, timestep_value in enumerate(sequence):
            timesteps = torch.full(
                (shape[0],),
                int(timestep_value.item()),
                device=device,
                dtype=torch.long,
            )
            predicted_velocity = denoise(sample, timesteps)
            clean = self.predict_clean_from_velocity(
                sample, timesteps, predicted_velocity
            )
            predicted_noise = self.predict_noise_from_velocity(
                sample, timesteps, predicted_velocity
            )
            alpha = self.alphas_cumulative[timestep_value]
            if clip_clean is not None:
                clean = clean.clamp(*clip_clean)

            if index == len(sequence) - 1:
                sample = clean
                continue

            next_timestep = sequence[index + 1]
            alpha_next = self.alphas_cumulative[next_timestep]
            if clip_clean is not None:
                predicted_noise = (
                    sample - alpha.sqrt() * clean
                ) / (1.0 - alpha).sqrt().clamp_min(1e-12)
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_next) / (1.0 - alpha))
                * (1.0 - alpha / alpha_next)
            ).clamp_min(0.0)
            direction = torch.sqrt((1.0 - alpha_next - sigma**2).clamp_min(0.0))
            random_noise = torch.randn_like(sample) if eta > 0.0 else 0.0
            sample = (
                alpha_next.sqrt() * clean
                + direction * predicted_noise
                + sigma * random_noise
            )

        return sample
