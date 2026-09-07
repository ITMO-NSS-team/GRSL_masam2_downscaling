"""Test-set metrics for the trained U-Nets, land-masked.

  joint     -> the in-network reprojection of native OSISAF, no learning
  enhance  -> the CDO-reprojected OSISAF, no learning
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (
    CACHE, MODELS_DIR, TASKS, U8Pairs, build_splits, index_dir,
)

from paths import LANDMASK


def ocean_mask():
    raw = np.load(LANDMASK)
    land = np.flip(np.rot90(raw, k=1), axis=1) == 120
    print(f"Land fraction: {land.mean():.3f}")
    return torch.from_numpy(np.ascontiguousarray(~land))


def score_masked(pred, target, mask_flat, mask2d, ssim_metric, threshold=0.15):
    """Metrics over the pixels selected by mask_flat.

    mask_flat=None scores the whole field, including land. SSIM is windowed, so
    it cannot be restricted to an irregular region; when a mask is given, land
    is zeroed in both prediction and target instead, which is what the
    "land-zeroed" name refers to.
    """
    pf, tf = pred.reshape(-1), target.reshape(-1)
    if mask_flat is not None:
        pf, tf = pf[mask_flat], tf[mask_flat]
    pb, tb = pf >= threshold, tf >= threshold
    tp = (pb & tb).sum().float()
    tn = (~pb & ~tb).sum().float()
    fp = (pb & ~tb).sum().float()
    fn = (~pb & tb).sum().float()
    parts = []
    if tp + fn > 0:
        parts.append(tp / (tp + fn))
    if tn + fp > 0:
        parts.append(tn / (tn + fp))
    bacc = (sum(parts) / len(parts)).item() if parts else float("nan")
    mse = torch.mean((pf - tf) ** 2).item()
    if mask2d is None:
        pz, tz = pred.unsqueeze(0), target.unsqueeze(0)
        ssim_key = "SSIM"
    else:
        pz = torch.where(mask2d, pred, torch.zeros_like(pred)).unsqueeze(0)
        tz = torch.where(mask2d, target, torch.zeros_like(target)).unsqueeze(0)
        ssim_key = "SSIM"
    return {
        "BACC": bacc,
        "IIEE": (pb != tb).sum().item() / 1e5,
        "MAE": torch.abs(pf - tf).mean().item(),
        "PSNR": 10 * np.log10(1.0 / mse) if mse > 0 else float("inf"),
        ssim_key: ssim_metric(pz, tz).item(),
    }


def score_both(pred, target, ocean_flat, ocean2d, ssim_metric, threshold=0.15):
    """Both views in one pass: the whole field, and ocean pixels only."""
    return {
        "full": score_masked(pred, target, None, None, ssim_metric, threshold),
        "ocean": score_masked(pred, target, ocean_flat, ocean2d, ssim_metric, threshold),
    }


def score(pred, target, ocean_flat, ocean2d, ssim_metric, threshold=0.15):
    """Backwards-compatible alias for the ocean-only view."""
    return score_masked(pred, target, ocean_flat, ocean2d, ssim_metric, threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--baseline", action="store_true", help="score the input, not a model")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1,
                    help="must match the stride the model was trained with")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.55,
                    help="lower it (e.g. 0.25) to run alongside a training job")
    args = ap.parse_args()

    dev = "cuda"
    try:                                       
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass
    if args.gpu_mem_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)

    src_dir, model_cls, kwargs, _ = TASKS[args.task]
    src_index, tgt_index = index_dir(src_dir), index_dir(CACHE / "masam2")
    splits = dict(zip(("train", "val", "test"), build_splits(src_index, tgt_index)))
    days = splits[args.split][:: args.stride]
    print(f"task={args.task} split={args.split}  {len(days)} days")

    loader = DataLoader(U8Pairs(days, src_index, tgt_index), batch_size=1,
                        num_workers=args.workers, pin_memory=True)

    model = None
    if not args.baseline:
        weights = args.weights or MODELS_DIR / f"unet_{args.task}_best.pth"
        st = torch.load(weights, map_location=dev)
        model = model_cls(1, 1, **kwargs).to(dev)
        model.load_state_dict(st["model_state_dict"])
        model.eval()
        print(f"weights {weights}  (epoch {st['epoch']+1}, val {st['val_loss']:.6f})")
    else:
        native_input = Path(src_dir).name == "native"
        if native_input:
            from reprojection import OsisafToMasam2
            model = OsisafToMasam2().to(dev)
            print("scoring the reprojected INPUT (no learning)")
        else:
            print("scoring the INPUT as-is (no learning)")

    om = ocean_mask().to(dev)
    of = om.reshape(-1)
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)

    acc = {}
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc=f"{args.task}/{args.split}"):
            x, y = x.to(dev), y.to(dev)
            pred = x if model is None else model(x)
            pred = pred.float().clamp(0.0, 1.0)
            # keep the channel dim: torchmetrics SSIM needs BxCxHxW, and the
            # unsqueeze(0) inside score() only adds the batch axis
            for p, t in zip(pred, y):
                for view, vals in score_both(p, t, of, om, ssim).items():
                    for k, v in vals.items():
                        acc.setdefault(view, {}).setdefault(k, []).append(v)

    out = {view: {k: {"mean": float(np.nanmean(v)),
                      "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                      "median": float(np.nanmedian(v))} for k, v in vals.items()}
           for view, vals in acc.items()}
    for view in ("full", "ocean"):
        print(f"\n=== {args.task} / {args.split}"
              f"{' / BASELINE' if args.baseline else ''} ({len(days)} days, "
              f"{'whole field' if view == 'full' else 'ocean only'}) ===")
        for k, v in out[view].items():
            print(f"  {k:18s} {v['mean']:10.5f} +- {v['std']:8.5f}   median {v['median']:10.5f}")

    name = f"metrics_{args.task}_{args.split}{'_baseline' if args.baseline else ''}.json"
    path = MODELS_DIR / name
    path.write_text(json.dumps({"task": args.task, "split": args.split,
                                "baseline": args.baseline, "stride": args.stride,
                                "n_days": len(days), "metrics": out}, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
