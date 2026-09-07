"""Evaluate experiment-4 diffusion predictions on a dataset split."""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import (
    FULLFRAME_DIFFUSION_BEST_MODEL_PATH,
    MASAM2_DIR,
    OSISAF_DIR,
    OUTPUT_METRICS_DIR,
    SEA_ICE_THRESHOLD,
    TEST_DATE_RANGE,
    TRAIN_DATE_RANGE,
    VAL_DATE_RANGE,
)
from dataset import create_dataloaders
from fullframe_diffusion.runtime import (
    create_initial_noise,
    load_checkpoint_model,
    resolve_device,
)


def balanced_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Calculate balanced ice/no-ice classification accuracy."""
    predicted_ice = prediction >= threshold
    target_ice = target >= threshold

    true_positive = (predicted_ice & target_ice).sum().float()
    true_negative = (~predicted_ice & ~target_ice).sum().float()
    false_positive = (predicted_ice & ~target_ice).sum().float()
    false_negative = (~predicted_ice & target_ice).sum().float()

    recalls = []
    if true_positive + false_negative > 0:
        recalls.append(true_positive / (true_positive + false_negative))
    if true_negative + false_positive > 0:
        recalls.append(true_negative / (true_negative + false_positive))
    if not recalls:
        return prediction.new_tensor(float("nan"))
    return torch.stack(recalls).mean()


def field_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """Calculate whole-grid metrics using the U-Net evaluation convention."""
    try:
        from torchmetrics.functional.image import (
            structural_similarity_index_measure,
        )
    except ImportError:
        from torchmetrics.functional import structural_similarity_index_measure

    error = prediction - target
    mean_squared_error = error.square().mean()
    predicted_ice = prediction >= threshold
    target_ice = target >= threshold
    ssim = structural_similarity_index_measure(
        prediction[None, None],
        target[None, None],
        data_range=1.0,
    )

    return {
        "bacc": balanced_accuracy(prediction, target, threshold).item(),
        "iiee_1e5": ((predicted_ice != target_ice).sum() / 1e5).item(),
        "mae": error.abs().mean().item(),
        "rmse": mean_squared_error.sqrt().item(),
        "psnr": (
            -10.0 * torch.log10(mean_squared_error.clamp_min(1e-12))
        ).item(),
        "ssim": ssim.item(),
    }


@torch.no_grad()
def evaluate_dataloader(
    model,
    dataloader,
    device: torch.device,
    threshold: float,
    max_samples: int | None,
    sampling_steps: int | None,
    seed: int,
) -> dict[str, Any]:
    """Generate predictions and aggregate metrics for one dataloader."""
    rows = []
    model.eval()
    progress = tqdm(dataloader, desc="Full-frame diffusion evaluation")

    for batch in progress:
        if max_samples is not None and len(rows) >= max_samples:
            break

        osisaf = batch["lr"].to(device)
        target = batch["hr"].to(device)
        date_seed = seed + int(str(batch["date"][0]))
        initial_noise = create_initial_noise(
            model,
            osisaf.shape[0],
            device,
            date_seed,
        )
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
            finite_fraction = torch.isfinite(prediction).float().mean().item()
            raise FloatingPointError(
                "Prediction contains NaN or Inf for "
                f"{batch['date'][0]}; finite fraction={finite_fraction:.6f}"
            )

        for item_index, date in enumerate(batch["date"]):
            if max_samples is not None and len(rows) >= max_samples:
                break
            metrics = field_metrics(
                prediction[item_index, 0].float(),
                target[item_index, 0],
                threshold,
            )
            metrics["date"] = str(date)
            rows.append(metrics)

        progress.set_postfix(
            mae=f"{np.nanmean([row['mae'] for row in rows]):.4f}",
            ssim=f"{np.nanmean([row['ssim'] for row in rows]):.4f}",
        )

    if not rows:
        raise ValueError("No samples were evaluated")

    metric_names = [name for name in rows[0] if name != "date"]
    summary = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summary[name] = {
            "mean": float(np.nanmean(values)),
            "std": (
                float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            ),
        }
    return {"summary": summary, "samples": rows}


def evaluate_fullframe_diffusion(
    split: str,
    checkpoint_path: str | Path = FULLFRAME_DIFFUSION_BEST_MODEL_PATH,
    output_path: str | Path | None = None,
    osisaf_dir: str | Path = OSISAF_DIR,
    masam2_dir: str | Path = MASAM2_DIR,
    missed_dates_path: str | Path | None = None,
    threshold: float = SEA_ICE_THRESHOLD,
    max_samples: int | None = None,
    sampling_steps: int | None = None,
    seed: int = 42,
    num_workers: int = 2,
    use_ema: bool = True,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Load a checkpoint, evaluate one split, and save detailed metrics."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(
        output_path
        or OUTPUT_METRICS_DIR / f"fullframe_diffusion_{split}_metrics.json"
    )
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint_model(
        checkpoint_path,
        device,
        use_ema=use_ema,
    )
    config = model.config
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        osisaf_dir=osisaf_dir,
        masam2_dir=masam2_dir,
        train_date_range=config.training.train_date_range or TRAIN_DATE_RANGE,
        val_date_range=config.training.val_date_range or VAL_DATE_RANGE,
        test_date_range=config.training.test_date_range or TEST_DATE_RANGE,
        batch_size=1,
        num_workers=num_workers,
        with_missed=(
            config.training.with_missed_dates
            if missed_dates_path is None
            else False
        ),
        missed_dates_path=missed_dates_path,
    )
    dataloader = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }[split]
    results = evaluate_dataloader(
        model,
        dataloader,
        device,
        threshold=threshold,
        max_samples=max_samples,
        sampling_steps=sampling_steps,
        seed=seed,
    )

    checkpoint_epoch = checkpoint.get("epoch")
    results["checkpoint_epoch_index"] = checkpoint_epoch
    results["checkpoint_epoch"] = (
        int(checkpoint_epoch) + 1 if checkpoint_epoch is not None else None
    )
    results["split"] = split
    results["weights"] = "ema" if use_ema else "raw"
    results["device"] = str(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    print(f"Saved metrics: {output_path}")
    return results


def main() -> None:
    """Parse CLI arguments and evaluate a diffusion checkpoint."""
    parser = argparse.ArgumentParser(
        description="Evaluate experiment-4 full-frame diffusion."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=FULLFRAME_DIFFUSION_BEST_MODEL_PATH,
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--osisaf-dir", type=Path, default=OSISAF_DIR)
    parser.add_argument("--masam2-dir", type=Path, default=MASAM2_DIR)
    parser.add_argument("--missed-dates", type=Path, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=SEA_ICE_THRESHOLD,
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    args = parser.parse_args()

    evaluate_fullframe_diffusion(
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_path=args.output,
        osisaf_dir=args.osisaf_dir,
        masam2_dir=args.masam2_dir,
        missed_dates_path=args.missed_dates,
        threshold=args.threshold,
        max_samples=args.max_samples,
        sampling_steps=args.sampling_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        use_ema=not args.raw_weights,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
