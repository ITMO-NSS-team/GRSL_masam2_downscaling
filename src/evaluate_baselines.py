"""Score the non-learned baselines with the SAME protocol as the models.


Six baselines:

  bilinear / bicubic / nearest        native OSISAF resized to 2100x2550 with
                                      no map projection at all - pixel count
                                      only, geometry wrong
  bilinear_scrip / bicubic_scrip /    the same field remapped by CDO
  nearest_scrip                       (remapbil / remapbic / remapnn), i.e.
                                      geometrically correct
  manual_hybrid                       CDO-remapped OSISAF constrained by the
                                      MASIE ice extent - not a model, a
                                      hand-built product kept for reference

"""
import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths
from train import CACHE, MODELS_DIR, build_splits, index_dir
from evaluate import ocean_mask, score_both

OUT = (2100, 2550)
SCRIP_DIRS = {
    "nearest_scrip": paths.OSISAF_REPROJ,
    "bilinear_scrip": paths.OSISAF_REPROJ_BILINEAR,
    "bicubic_scrip": paths.OSISAF_REPROJ_BICUBIC,
    "manual_hybrid": paths.MANUAL_HYBRID,
}
RESIZE_MODES = ("bilinear", "bicubic", "nearest")


def index_plain(directory):
    out = {}
    for p in Path(directory).glob("*.npy"):
        m = re.search(r"(\d{8})", p.name)
        if m:
            out[m.group(1)] = str(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.55)
    args = ap.parse_args()

    dev = "cuda"
    if args.gpu_mem_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass

    native = index_dir(CACHE / "native")
    masam2 = index_dir(CACHE / "masam2")
    days = dict(zip(("train", "val", "test"),
                    build_splits(native, masam2)))[args.split][:: args.stride]
    print(f"{args.split}: {len(days)} days (stride {args.stride})")

    om = ocean_mask().to(dev)
    of = om.reshape(-1)
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)

    scrip = {k: index_plain(v) for k, v in SCRIP_DIRS.items()}
    results = {}

    def finish(name, acc):
        results[name] = {
            view: {k: {"mean": float(np.nanmean(v)),
                       "std": float(np.nanstd(v, ddof=1)),
                       "median": float(np.nanmedian(v))} for k, v in vals.items()}
            for view, vals in acc.items()
        }
        for view in ("full", "ocean"):
            m = results[name][view]
            print(f"  {name:16s} [{view:5s}] BACC {m['BACC']['mean']:.3f}+-{m['BACC']['std']:.3f}   "
                  f"IIEE {m['IIEE']['mean']:.3f}+-{m['IIEE']['std']:.3f}   "
                  f"MAE {m['MAE']['mean']:.4f}+-{m['MAE']['std']:.4f}   "
                  f"PSNR {m['PSNR']['mean']:.2f}   SSIM {m['SSIM']['mean']:.3f}")

    # --- plain resizes of the native grid, no reprojection ---
    for mode in RESIZE_MODES:
        acc = {}
        for day in tqdm(days, desc=mode, leave=False):
            x = torch.from_numpy(
                np.load(native[day]).astype(np.float32) / 100.0)[None, None].to(dev)
            y = torch.from_numpy(
                np.load(masam2[day]).astype(np.float32) / 100.0)[None].to(dev)
            kw = {} if mode == "nearest" else {"align_corners": False}
            pred = F.interpolate(x, size=OUT, mode=mode, **kw)[0].clamp(0.0, 1.0)
            for view, vals in score_both(pred, y, of, om, ssim).items():
                for k, v in vals.items():
                    acc.setdefault(view, {}).setdefault(k, []).append(v)
        finish(mode, acc)

    # --- CDO/SCRIP remappings ---
    for name, idx in scrip.items():
        avail = [d for d in days if d in idx]
        if not avail:
            print(f"  {name}: no files, skipped")
            continue
        acc = {}
        for day in tqdm(avail, desc=name, leave=False):
            pred = torch.from_numpy(
                np.load(idx[day]).astype(np.float32))[None].to(dev).clamp(0.0, 1.0)
            y = torch.from_numpy(
                np.load(masam2[day]).astype(np.float32) / 100.0)[None].to(dev)
            for view, vals in score_both(pred, y, of, om, ssim).items():
                for k, v in vals.items():
                    acc.setdefault(view, {}).setdefault(k, []).append(v)
        finish(f"{name} ({len(avail)}d)" if len(avail) != len(days) else name, acc)

    path = MODELS_DIR / f"metrics_baselines_{args.split}.json"
    path.write_text(json.dumps({"split": args.split, "stride": args.stride,
                                "n_days": len(days), "metrics": results}, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
