"""Full-frame conditional diffusion for OSISAF-to-MASAM2 downscaling."""

from .config import FullFrameDiffusionConfig
from .model import FullFrameConditionalDiffusion

__all__ = ["FullFrameDiffusionConfig", "FullFrameConditionalDiffusion"]
