"""Rebuild the MASIE-on-target-grid and Manual Hybrid fields.

    MASIE   sea_ice_extent[0][1907:4457, 2000:4100]  ->  ==3 is ice
            zero over land, fliplr, rot90            ->  (2100, 2550)

    hybrid  = OSISAF reprojected by CDO
              zeroed wherever MASIE says no ice
              raised to 0.15 wherever MASIE says ice but OSISAF is below it

"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import netCDF4 as nc
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import paths
from train import CACHE, build_splits, index_dir

MASIE_NC_DIR = paths.MASIE_NC
OSISAF_REPROJ_DIR = paths.OSISAF_REPROJ
LANDMASK = paths.LANDMASK
OUT_MASIE = paths.MASIE_REPROJ
OUT_HYBRID = paths.MANUAL_HYBRID

CROP = (1907, 1907 + 2550, 2000, 2000 + 2100)
ICE_CODE = 3            # MASIE sea_ice_extent code for ice
LAND_CODE = 120         # land in the MASAM2 mask, before reorientation
HYBRID_FLOOR = 0.15     # concentration floor where MASIE says ice


def land_mask_target():
    """Land mask already in the (2100, 2550) target orientation"""
    return np.flip(np.rot90(np.load(LANDMASK), k=1), axis=1) == LAND_CODE


def masie_on_target_grid(date, land_target):
    path = MASIE_NC_DIR / f"{date}.nc"
    if not path.exists():
        return None
    with nc.Dataset(path) as d:
        raw = np.asarray(d.variables["sea_ice_extent"][0])
    y0, y1, x0, x1 = CROP
    ice = (raw[y0:y1, x0:x1] == ICE_CODE).astype(np.float32)
    ice = np.rot90(np.fliplr(ice))          # -> (2100, 2550)
    ice[land_target] = 0.0
    return np.ascontiguousarray(ice)


def manual_hybrid(osisaf_reproj, masie):
    hybrid = osisaf_reproj.copy()
    hybrid[masie != 1] = 0.0
    hybrid[(hybrid < HYBRID_FLOOR) & (masie == 1)] = HYBRID_FLOOR
    return hybrid


def build(days):
    OUT_MASIE.mkdir(parents=True, exist_ok=True)
    OUT_HYBRID.mkdir(parents=True, exist_ok=True)
    land = land_mask_target()
    made, skipped = 0, 0
    for day in tqdm(days, desc="manual hybrid"):
        hyb_path = OUT_HYBRID / f"{day}.npy"
        if hyb_path.exists():
            made += 1
            continue
        masie = masie_on_target_grid(day, land)
        osi_path = OSISAF_REPROJ_DIR / f"{day}.npy"
        if masie is None or not osi_path.exists():
            skipped += 1
            continue
        np.save(OUT_MASIE / f"{day}.npy", masie.astype(np.float32))
        np.save(hyb_path, manual_hybrid(np.load(osi_path).astype(np.float32), masie))
        made += 1
    print(f"built {made}, skipped {skipped} (missing MASIE or OSISAF)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "test"), default=None)
    args = ap.parse_args()
    if args.verify:
        verify()
    if args.split:
        native, masam2 = index_dir(CACHE / "native"), index_dir(CACHE / "masam2")
        days = dict(zip(("train", "val", "test"), build_splits(native, masam2)))[args.split]
        build(days)
