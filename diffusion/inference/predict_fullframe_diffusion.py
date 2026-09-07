"""Run experiment-4 diffusion inference for one OSISAF date."""

import argparse
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import (
    FULLFRAME_DIFFUSION_BEST_MODEL_PATH,
    FULLFRAME_DIFFUSION_PREDICTIONS_DIR,
    OSISAF_DIR,
)
from fullframe_diffusion.runtime import (
    create_initial_noise,
    load_checkpoint_model,
    resolve_device,
)


def validate_date(value: str) -> str:
    """Validate a date in YYYYMMDD format."""
    if len(value) != 8 or not value.isdigit():
        raise argparse.ArgumentTypeError("Date must use YYYYMMDD format")
    return value


def extract_date(filename: str, date_pattern: str = r"(\d{8})") -> str | None:
    """Extract a YYYYMMDD date from a filename."""
    match = re.search(date_pattern, filename)
    return match.group(1) if match else None


def find_osisaf_path(osisaf_dir: str | Path, date: str) -> Path:
    """Find the OSISAF NumPy matrix for a date."""
    matches = sorted(
        path
        for path in Path(osisaf_dir).rglob("*.npy")
        if extract_date(path.name) == date
    )
    if not matches:
        raise FileNotFoundError(f"No OSISAF .npy file found for date {date}")
    return matches[0]


def load_osisaf_tensor(osisaf_path: str | Path) -> torch.Tensor:
    """Load an OSISAF percentage matrix as a normalized model input."""
    osisaf_path = Path(osisaf_path)
    values = np.load(osisaf_path)
    if values.ndim != 2:
        raise ValueError(
            f"OSISAF matrix must be two-dimensional: {osisaf_path}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"OSISAF matrix contains NaN or Inf: {osisaf_path}")

    values = np.clip(values, 0, 100).astype(np.float32) / 100.0
    return torch.from_numpy(values).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def predict_fullframe_diffusion(
    date: str,
    checkpoint_path: str | Path = FULLFRAME_DIFFUSION_BEST_MODEL_PATH,
    output_path: str | Path | None = None,
    osisaf_dir: str | Path = OSISAF_DIR,
    sampling_steps: int | None = None,
    seed: int = 42,
    use_ema: bool = True,
    device_name: str = "auto",
) -> Path:
    """Generate and save one full-resolution MASAM2 prediction."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(
        output_path or FULLFRAME_DIFFUSION_PREDICTIONS_DIR / f"{date}.npy"
    )
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint_model(
        checkpoint_path,
        device,
        use_ema=use_ema,
    )

    osisaf_path = find_osisaf_path(osisaf_dir, date)
    osisaf = load_osisaf_tensor(osisaf_path).to(device)
    initial_noise = create_initial_noise(model, 1, device, seed)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast_context:
        prediction = model.sample(
            osisaf,
            sampling_steps=sampling_steps,
            initial_noise=initial_noise,
        )

    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Prediction contains NaN or infinite values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_array = prediction[0, 0].float().cpu().numpy().astype(np.float32)
    np.save(output_path, prediction_array)

    checkpoint_epoch = checkpoint.get("epoch")
    epoch = int(checkpoint_epoch) + 1 if checkpoint_epoch is not None else None
    print(f"Device: {device}")
    print(f"Weights: {'EMA' if use_ema else 'raw'}")
    print(f"Checkpoint epoch: {epoch}")
    print(f"OSISAF: {osisaf_path}")
    print(f"Prediction: {output_path}")
    return output_path


def main() -> None:
    """Parse CLI arguments and run one-date diffusion inference."""
    parser = argparse.ArgumentParser(
        description="Predict a full MASAM2 field with experiment-4 diffusion."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=FULLFRAME_DIFFUSION_BEST_MODEL_PATH,
    )
    parser.add_argument("--date", type=validate_date, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--osisaf-dir", type=Path, default=OSISAF_DIR)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    args = parser.parse_args()

    predict_fullframe_diffusion(
        checkpoint_path=args.checkpoint,
        date=args.date,
        output_path=args.output,
        osisaf_dir=args.osisaf_dir,
        sampling_steps=args.sampling_steps,
        seed=args.seed,
        use_ema=not args.raw_weights,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
