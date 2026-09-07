"""Score the published Attention U-Net and Light U-Net with the current protocol.

Both take native OSISAF (432x432) and emit the MASAM2 grid directly.

"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "published_models"))

from train import CACHE, MODELS_DIR, build_splits, index_dir
from evaluate import ocean_mask, score_both

PUBLISHED = {
    "attention_unet": ("attention_unet", MODELS_DIR / "attention_unet_final.pth"),
    "light_unet": ("unet_light", MODELS_DIR / "unet_light_final.pth"),
}


def load(module_name, weights, device):
    import importlib
    module = importlib.import_module(module_name)
    model = module.UNet(in_channels=1, out_channels=1).to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.30)
    args = ap.parse_args()

    device = "cuda"
    if args.gpu_mem_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass

    native, masam2 = index_dir(CACHE / "native"), index_dir(CACHE / "masam2")
    days = dict(zip(("train", "val", "test"),
                    build_splits(native, masam2)))[args.split][:: args.stride]
    om = ocean_mask().to(device)
    of = om.reshape(-1)
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    results = {}
    for name, (module_name, weights) in PUBLISHED.items():
        if not weights.exists():
            print(f"{name}: {weights} missing, skipped")
            continue
        model = load(module_name, weights, device)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"\n{name}: {n_par/1e6:.2f}M parameters, {len(days)} {args.split} days")
        acc = {}
        with torch.no_grad():
            for day in tqdm(days, desc=name, leave=False):
                x = torch.from_numpy(
                    np.load(native[day]).astype(np.float32) / 100.0)[None, None].to(device)
                y = torch.from_numpy(
                    np.load(masam2[day]).astype(np.float32) / 100.0)[None].to(device)
                pred = model(x)[0].float().clamp(0.0, 1.0)
                for view, vals in score_both(pred, y, of, om, ssim).items():
                    for k, v in vals.items():
                        acc.setdefault(view, {}).setdefault(k, []).append(v)
        results[name] = {
            view: {k: {"mean": float(np.nanmean(v)),
                       "std": float(np.nanstd(v, ddof=1)),
                       "median": float(np.nanmedian(v))} for k, v in vals.items()}
            for view, vals in acc.items()
        }
        for view in ("full", "ocean"):
            m = results[name][view]
            print(f"  [{view:5s}] BACC {m['BACC']['mean']:.4f}  IIEE {m['IIEE']['mean']:.4f}  "
                  f"MAE {m['MAE']['mean']:.5f}  PSNR {m['PSNR']['mean']:.2f}  "
                  f"SSIM {m['SSIM']['mean']:.4f}")
        del model
        torch.cuda.empty_cache()

    path = MODELS_DIR / f"metrics_published_{args.split}.json"
    path.write_text(json.dumps({"split": args.split, "stride": args.stride,
                                "n_days": len(days), "metrics": results}, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
