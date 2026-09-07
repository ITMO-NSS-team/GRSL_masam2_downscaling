from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

# The upstream package took this from the sibling repo's config module. Fall
# back to a path inside this repository so the package works standalone.
try:
    from config import FULLFRAME_DIFFUSION_OUTPUT_DIR
except ImportError:
    from pathlib import Path as _Path
    FULLFRAME_DIFFUSION_OUTPUT_DIR = (
        _Path(__file__).resolve().parents[2] / "outputs" / "fullframe_diffusion"
    )


@dataclass
class GeometryConfig:
    source_shape: tuple[int, int] = (432, 432)
    target_shape: tuple[int, int] = (2100, 2550)
    padded_shape: tuple[int, int] = (2112, 2560)
    shuffle_factor: int = 4

    def validate(self) -> None:
        if len(self.source_shape) != 2 or len(self.target_shape) != 2:
            raise ValueError("source_shape and target_shape must be two-dimensional")
        if any(size <= 0 for size in (*self.source_shape, *self.target_shape)):
            raise ValueError("Grid dimensions must be positive")
        if any(
            padded < target
            for padded, target in zip(self.padded_shape, self.target_shape)
        ):
            raise ValueError("padded_shape must not be smaller than target_shape")
        if self.shuffle_factor not in (1, 2, 4):
            raise ValueError("shuffle_factor must be one of 1, 2, or 4")
        if any(size % self.shuffle_factor for size in self.padded_shape):
            raise ValueError("padded_shape must be divisible by shuffle_factor")

    @property
    def latent_shape(self) -> tuple[int, int]:
        return tuple(size // self.shuffle_factor for size in self.padded_shape)

    @property
    def latent_channels(self) -> int:
        return self.shuffle_factor**2


@dataclass
class NetworkConfig:
    condition_base_channels: int = 16
    base_channels: int = 40
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    blocks_per_level: int = 2
    time_embedding_dim: int = 256
    dropout: float = 0.0
    group_norm_groups: int = 8
    use_pooled_attention: bool = True
    attention_heads: int = 4
    max_attention_tokens: int = 1024
    gradient_checkpointing: bool = True

    def validate(self) -> None:
        if self.base_channels <= 0 or self.condition_base_channels <= 0:
            raise ValueError("Channel counts must be positive")
        if not self.channel_multipliers:
            raise ValueError("At least one channel multiplier is required")
        if self.blocks_per_level <= 0:
            raise ValueError("blocks_per_level must be positive")
        channels = [self.base_channels * value for value in self.channel_multipliers]
        if any(channel % self.group_norm_groups for channel in channels):
            raise ValueError(
                "Every U-Net channel count must be divisible by GroupNorm groups"
            )


@dataclass
class DiffusionConfig:
    training_timesteps: int = 1000
    sampling_steps: int = 30
    cosine_offset: float = 0.008
    ddim_eta: float = 0.0
    prediction_type: str = "v"
    x0_loss_weight: float = 0.5
    marginal_ice_weight: float = 2.0
    marginal_ice_low: float = 0.05
    marginal_ice_high: float = 0.95

    def validate(self) -> None:
        if self.training_timesteps <= 1:
            raise ValueError("training_timesteps must be greater than one")
        if not 1 <= self.sampling_steps <= self.training_timesteps:
            raise ValueError("sampling_steps must lie within the training schedule")
        if self.prediction_type != "v":
            raise ValueError("Experiment 4 supports only v-prediction")
        if self.x0_loss_weight < 0 or self.marginal_ice_weight < 0:
            raise ValueError("Loss weights must be non-negative")


@dataclass
class TrainingConfig:
    train_date_range: tuple[str, str] = ("20120101", "20211231")
    val_date_range: tuple[str, str] = ("20220101", "20231231")
    test_date_range: tuple[str, str] = ("20240101", "20260418")
    batch_size: int = 1
    gradient_accumulation: int = 8
    epochs: int = 80
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    num_workers: int = 2
    validation_batches: int = 8
    sample_every_epochs: int = 5
    checkpoint_every_epochs: int = 1
    checkpoint_every_steps: int = 100
    milestone_epochs: tuple[int, ...] = (20, 50, 80)
    early_stopping_patience: int = 10
    use_amp: bool = True
    with_missed_dates: bool = True
    seed: int = 42
    output_dir: str = str(FULLFRAME_DIFFUSION_OUTPUT_DIR)

    def validate(self) -> None:
        if self.batch_size != 1:
            raise ValueError(
                "The initial full-frame implementation requires batch_size=1"
            )
        if (
            self.gradient_accumulation <= 0
            or self.epochs <= 0
            or self.checkpoint_every_steps <= 0
            or self.early_stopping_patience <= 0
        ):
            raise ValueError("Training counts must be positive")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie between zero and one")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if any(epoch <= 0 or epoch > self.epochs for epoch in self.milestone_epochs):
            raise ValueError("milestone_epochs must lie within the training run")


@dataclass
class FullFrameDiffusionConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.geometry.validate()
        self.network.validate()
        self.diffusion.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FullFrameDiffusionConfig":
        geometry_values = dict(values.get("geometry", {}))
        network_values = dict(values.get("network", {}))
        diffusion_values = dict(values.get("diffusion", {}))
        training_values = dict(values.get("training", {}))

        for key in ("source_shape", "target_shape", "padded_shape"):
            if key in geometry_values:
                geometry_values[key] = tuple(geometry_values[key])
        if "channel_multipliers" in network_values:
            network_values["channel_multipliers"] = tuple(
                network_values["channel_multipliers"]
            )
        for key in ("train_date_range", "val_date_range", "test_date_range"):
            if key in training_values:
                training_values[key] = tuple(training_values[key])
        if "milestone_epochs" in training_values:
            training_values["milestone_epochs"] = tuple(
                training_values["milestone_epochs"]
            )

        config = cls(
            geometry=GeometryConfig(**geometry_values),
            network=NetworkConfig(**network_values),
            diffusion=DiffusionConfig(**diffusion_values),
            training=TrainingConfig(**training_values),
        )
        config.validate()
        return config
