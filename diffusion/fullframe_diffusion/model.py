from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import OSISAFConditionEncoder
from .config import FullFrameDiffusionConfig
from .data import validate_fullframe_batch
from .diffusion import GaussianDiffusion
from .geometry import crop_to_shape, pad_to_shape
from .unet import ConditionalDiffusionUNet


class FullFrameConditionalDiffusion(nn.Module):
    """SR3-style strongly conditioned direct diffusion of full MASAM2 fields."""

    def __init__(self, config: FullFrameDiffusionConfig) -> None:
        super().__init__()
        config.validate()
        geometry = config.geometry

        self.config = config
        levels = len(config.network.channel_multipliers)
        self.condition_encoder = OSISAFConditionEncoder(
            in_channels=1,
            base_channels=config.network.condition_base_channels,
            levels=levels,
        )
        self.denoiser = ConditionalDiffusionUNet(
            latent_channels=geometry.latent_channels,
            spatial_condition_channels=geometry.latent_channels,
            condition_channels=self.condition_encoder.output_channels,
            config=config.network,
        )
        self.diffusion = GaussianDiffusion(
            timesteps=config.diffusion.training_timesteps,
            cosine_offset=config.diffusion.cosine_offset,
        )

    @property
    def geometry(self):
        return self.config.geometry

    def full_to_latent(
        self,
        full_resolution: torch.Tensor,
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        padded = pad_to_shape(
            full_resolution,
            self.geometry.padded_shape,
            value=pad_value,
        )
        return F.pixel_unshuffle(padded, self.geometry.shuffle_factor)

    def latent_to_full(self, latent: torch.Tensor) -> torch.Tensor:
        padded = F.pixel_shuffle(latent, self.geometry.shuffle_factor)
        return crop_to_shape(padded, self.geometry.target_shape)

    def target_to_latent(self, masam2: torch.Tensor) -> torch.Tensor:
        normalized = masam2.mul(2.0).sub(1.0)
        return self.full_to_latent(normalized, pad_value=-1.0)

    def osisaf_to_spatial_condition(self, osisaf: torch.Tensor) -> torch.Tensor:
        """Build an SR3-style target-grid condition for input concatenation."""
        normalized = osisaf.mul(2.0).sub(1.0)
        target_grid = F.interpolate(
            normalized,
            size=self.geometry.target_shape,
            mode="bilinear",
            align_corners=False,
        )
        return self.full_to_latent(target_grid, pad_value=-1.0)

    def latent_to_target(self, latent: torch.Tensor) -> torch.Tensor:
        normalized = self.latent_to_full(latent).clamp(-1.0, 1.0)
        return normalized.add(1.0).mul(0.5).clamp(0.0, 1.0)

    def valid_latent_mask(self, reference: torch.Tensor) -> torch.Tensor:
        valid_full = torch.ones_like(reference)
        return self.full_to_latent(valid_full, pad_value=0.0)

    def training_loss(
        self,
        osisaf: torch.Tensor,
        masam2: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        validate_fullframe_batch(
            osisaf,
            masam2,
            self.geometry.source_shape,
            self.geometry.target_shape,
        )
        batch_size = osisaf.shape[0]
        clean_latent = self.target_to_latent(masam2)

        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.diffusion.timesteps,
                (batch_size,),
                device=osisaf.device,
            )
        if noise is None:
            noise = torch.randn_like(clean_latent)
        noisy_latent = self.diffusion.q_sample(clean_latent, timesteps, noise)

        condition_features = self.condition_encoder(osisaf)
        spatial_condition = self.osisaf_to_spatial_condition(osisaf)
        predicted_velocity = self.denoiser(
            noisy_latent,
            spatial_condition,
            timesteps,
            condition_features,
        )
        target_velocity = self.diffusion.velocity_target(
            clean_latent, timesteps, noise
        )

        marginal = (
            (masam2 > self.config.diffusion.marginal_ice_low)
            & (masam2 < self.config.diffusion.marginal_ice_high)
        ).float()
        full_weight = 1.0 + self.config.diffusion.marginal_ice_weight * marginal
        latent_weight = self.full_to_latent(full_weight, pad_value=0.0)
        weight_denominator = latent_weight.sum().clamp_min(1.0)
        velocity_loss = (
            ((predicted_velocity - target_velocity).square() * latent_weight).sum()
            / weight_denominator
        )

        predicted_clean = self.diffusion.predict_clean_from_velocity(
            noisy_latent,
            timesteps,
            predicted_velocity,
        )
        valid_latent = self.valid_latent_mask(masam2)
        valid_denominator = valid_latent.sum().clamp_min(1.0)
        x0_loss = (
            ((predicted_clean - clean_latent).abs() * valid_latent).sum()
            / valid_denominator
        )
        total = velocity_loss + self.config.diffusion.x0_loss_weight * x0_loss
        return {
            "loss": total,
            "velocity_loss": velocity_loss.detach(),
            "x0_loss": x0_loss.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        osisaf: torch.Tensor,
        sampling_steps: int | None = None,
        eta: float | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_fullframe_batch(
            osisaf,
            None,
            self.geometry.source_shape,
            self.geometry.target_shape,
        )
        condition_features = self.condition_encoder(osisaf)
        spatial_condition = self.osisaf_to_spatial_condition(osisaf)
        shape = (
            osisaf.shape[0],
            self.geometry.latent_channels,
            *self.geometry.latent_shape,
        )

        def denoise(noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            return self.denoiser(
                noisy,
                spatial_condition,
                timesteps,
                condition_features,
            )

        target_latent = self.diffusion.ddim_sample(
            denoise=denoise,
            shape=shape,
            device=osisaf.device,
            sampling_steps=sampling_steps or self.config.diffusion.sampling_steps,
            eta=self.config.diffusion.ddim_eta if eta is None else eta,
            initial_noise=initial_noise,
            clip_clean=(-1.0, 1.0),
        )
        return self.latent_to_target(target_latent)

    def forward(
        self,
        osisaf: torch.Tensor,
        sampling_steps: int | None = None,
    ) -> torch.Tensor:
        return self.sample(osisaf, sampling_steps=sampling_steps)
