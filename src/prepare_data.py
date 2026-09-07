"""One-off: repack the training arrays as small LOSSLESS integer types.

Each source sits on its own quantisation grid, verified per file:

  native OSISAF  integers 0..100          -> uint8  x100     0.19 MB (was 1.5)
  MASAM2         32 distinct values, all
                 multiples of 0.01        -> uint8  x100     5.36 MB (was 42.8)
  SCRIP reproj   grid of 1e-4             -> uint16 x10000  10.71 MB (was 21.4)

uint8 is NOT lossless for the reprojected field (max error 0.005, checked), so
that one gets uint16; the round trip there is exact to 6e-8.

Re-running skips files that already exist, so it is safe to interrupt.
"""
import os
import re
import sys
import glob
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CACHE, MASAM2, OSISAF_NATIVE, OSISAF_REPROJ

# name: (source dir, cache dir, divisor to reach [0,1], integer scale, dtype)
SOURCES = {
    "native": (OSISAF_NATIVE, CACHE / "native", 100.0,   100, np.uint8),
    "reproj": (OSISAF_REPROJ, CACHE / "reproj",   1.0, 10000, np.uint16),
    "masam2": (MASAM2,        CACHE / "masam2",   1.0,   100, np.uint8),
}

# training code recovers [0,1] from the dtype alone
SCALE_FOR_DTYPE = {np.dtype(np.uint8): 100.0, np.dtype(np.uint16): 10000.0}


def convert(name, verify_every=200):
    src_dir, dst_dir, div, scale, dtype = SOURCES[name]
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "**", "*.npy"), recursive=True))
    print(f"\n[{name}] {len(files)} files -> {dst_dir}")
    t0 = time.time()
    n_done = n_skip = 0
    worst = 0.0
    for i, path in enumerate(files):
        m = re.search(r"(\d{8})", os.path.basename(path))
        if not m:
            continue
        out = os.path.join(dst_dir, f"{m.group(1)}.npy")
        if os.path.exists(out):
            n_skip += 1
            continue
        a = np.load(path)
        if a.ndim != 2:
            print(f"  skip {os.path.basename(path)}: shape {a.shape}")
            continue
        unit = np.clip(a.astype(np.float64) / div, 0.0, 1.0)
        q = np.rint(unit * scale).astype(dtype)
        if i % verify_every == 0:                      # spot-check losslessness
            worst = max(worst, float(np.abs(q.astype(np.float64) / scale - unit).max()))
        np.save(out, q)
        n_done += 1
        if n_done % 500 == 0:
            el = time.time() - t0
            print(f"  {n_done}/{len(files)}  {el:.0f}s  ({n_done/el:.1f} files/s)")
    print(f"[{name}] wrote {n_done}, skipped {n_skip} existing, {time.time()-t0:.0f}s")
    print(f"[{name}] worst spot-checked round-trip error: {worst:.10f}")
    # A genuine quantisation failure shows up as a sizeable fraction of one
    # step (1/scale). Anything near float32 epsilon is just rounding noise
    # from the float64 -> float32 source values, not a lossy cache.
    if worst > 0.01 / scale:
        print(f"[{name}] WARNING: {np.dtype(dtype).name} x{scale} is NOT lossless here "
              f"(worst {worst:.2e} vs step {1.0/scale:.1e}) - pick a finer scale or dtype")


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SOURCES)
    for name in wanted:
        if name not in SOURCES:
            raise SystemExit(f"unknown source {name!r}; choose from {list(SOURCES)}")
        convert(name)
