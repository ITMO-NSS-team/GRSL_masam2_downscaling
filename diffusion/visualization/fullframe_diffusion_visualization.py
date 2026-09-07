"""Visualize a saved full-frame diffusion prediction against MASAM2."""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

matplotlib.use("Agg")

from config import (
    FULLFRAME_DIFFUSION_PREDICTIONS_DIR,
    MASAM2_DIR,
    OSISAF_DIR,
)
from dataset import IceConcentrationDataset
from inference.predict_fullframe_diffusion import extract_date, validate_date
from visualization.visualization import visualize_results


def find_sample_by_date(
    dataset: IceConcentrationDataset,
    date: str,
) -> dict:
    """Return the paired OSISAF/MASAM2 sample for one date."""
    for index, (osisaf_path, _) in enumerate(dataset.pairs):
        if extract_date(Path(osisaf_path).name) == date:
            return dataset[index]
    raise ValueError(f"No OSISAF/MASAM2 pair found for date {date}")


def visualize_fullframe_diffusion(
    date: str,
    prediction_path: str | Path | None = None,
    output_path: str | Path | None = None,
    osisaf_dir: str | Path = OSISAF_DIR,
    masam2_dir: str | Path = MASAM2_DIR,
) -> Path:
    """Save OSISAF, prediction, MASAM2, and absolute-error panels."""
    prediction_path = Path(
        prediction_path
        or FULLFRAME_DIFFUSION_PREDICTIONS_DIR / f"{date}.npy"
    )
    output_path = Path(
        output_path
        or FULLFRAME_DIFFUSION_PREDICTIONS_DIR / f"{date}.png"
    )
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    prediction_array = np.load(prediction_path)
    if prediction_array.ndim != 2:
        raise ValueError("Diffusion prediction must be a two-dimensional array")
    if not np.isfinite(prediction_array).all():
        raise ValueError("Diffusion prediction contains NaN or infinite values")
    prediction = torch.from_numpy(
        prediction_array.astype(np.float32, copy=False)
    ).unsqueeze(0).unsqueeze(0)

    dataset = IceConcentrationDataset(
        osisaf_dir=osisaf_dir,
        masam2_dir=masam2_dir,
    )
    sample = find_sample_by_date(dataset, date)
    osisaf = sample["lr"].unsqueeze(0)
    target = sample["hr"].unsqueeze(0)
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError(
            f"Prediction shape {tuple(prediction.shape)} does not match "
            f"MASAM2 shape {tuple(target.shape)}"
        )

    visualize_results(
        lr_images=osisaf,
        hr_images=target,
        pred_images=prediction,
        epoch="inference",
        model_name="Full-frame diffusion",
        save_path=output_path,
        dates=[date],
    )

    mae = torch.mean(torch.abs(prediction - target)).item()
    print(f"Prediction: {prediction_path}")
    print(f"MASAM2: {sample['hr_path']}")
    print(f"MAE vs MASAM2: {mae:.6f}")
    print(f"Saved visualization: {output_path}")
    return output_path


def main() -> None:
    """Parse CLI arguments and visualize one saved prediction."""
    parser = argparse.ArgumentParser(
        description="Visualize a diffusion prediction against MASAM2."
    )
    parser.add_argument("--date", type=validate_date, required=True)
    parser.add_argument("--prediction", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--osisaf-dir", type=Path, default=OSISAF_DIR)
    parser.add_argument("--masam2-dir", type=Path, default=MASAM2_DIR)
    args = parser.parse_args()

    visualize_fullframe_diffusion(
        date=args.date,
        prediction_path=args.prediction,
        output_path=args.output,
        osisaf_dir=args.osisaf_dir,
        masam2_dir=args.masam2_dir,
    )


if __name__ == "__main__":
    main()
