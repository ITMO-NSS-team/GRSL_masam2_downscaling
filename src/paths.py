"""Where the data lives.

Every path can be overridden with an environment variable, so nothing in this
repository is tied to one machine:

    ICE_DATA_ROOT     parent of everything below           (default D:/ML4RS_data)
    ICE_OSISAF_NATIVE native OSISAF .npy, 432x432, 0..100 int
    ICE_OSISAF_REPROJ OSISAF remapped onto the MASAM2 grid by `cdo remapnn`
    ICE_MASAM2        MASAM2 .npy, 2100x2550, 0..1 float
    ICE_MASIE_NC      raw MASIE netCDF, 6144x6144 sea_ice_extent
    ICE_CACHE         integer cache written by prepare_data.py
    ICE_LANDMASK      MASAM2 land mask .npy (land == 120 before reorientation)

Linux/macOS:  export ICE_DATA_ROOT=/data/ice
Windows:      set ICE_DATA_ROOT=E:\\ice
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("ICE_DATA_ROOT", "D:/ML4RS_data"))


def _p(var, default):
    return Path(os.environ.get(var, default))


# --- inputs -----------------------------------------------------------------
OSISAF_NATIVE = _p("ICE_OSISAF_NATIVE", ROOT / "OSISAF_native_matrices_aiice")
MASAM2 = _p("ICE_MASAM2", ROOT / "MASAM2_matrices")
LANDMASK = _p("ICE_LANDMASK", ROOT / "masam2landmask.npy")
MASIE_NC = _p("ICE_MASIE_NC", ROOT / "MASIE")

# --- CDO/SCRIP remappings, only needed for the corresponding table columns ---
OSISAF_REPROJ = _p("ICE_OSISAF_REPROJ", ROOT / "OSISAF_reproj_matrices")
OSISAF_REPROJ_BILINEAR = _p("ICE_OSISAF_REPROJ_BILINEAR", ROOT / "OSISAF_reproj_matrices_bilinear")
OSISAF_REPROJ_BICUBIC = _p("ICE_OSISAF_REPROJ_BICUBIC", ROOT / "OSISAF_reproj_matrices_bicubic")

# --- derived, written by build_manual_hybrid.py -----------------------------
MASIE_REPROJ = _p("ICE_MASIE_REPROJ", ROOT / "MASIE_reproj_matrices")
MANUAL_HYBRID = _p("ICE_MANUAL_HYBRID", ROOT / "ManualHybrid_reproj_matrices")

# --- written by prepare_data.py ---------------------------------------------
CACHE = _p("ICE_CACHE", ROOT / "cache_u8")

# dates excluded because the MASAM2 side is missing or broken
MISSED_DATES_FILE = _p(
    "ICE_MISSED_DATES", Path(__file__).resolve().parents[1] / "data" / "masam2_missed.txt"
)

TRAIN_RANGE = ("20120701", "20201231")
VAL_RANGE = ("20210101", "20221231")
TEST_RANGE = ("20230101", "20260607")


def describe():
    rows = [
        ("OSISAF native", OSISAF_NATIVE), ("MASAM2", MASAM2),
        ("land mask", LANDMASK), ("MASIE netCDF", MASIE_NC),
        ("OSISAF reproj (nn)", OSISAF_REPROJ),
        ("OSISAF reproj (bil)", OSISAF_REPROJ_BILINEAR),
        ("OSISAF reproj (bic)", OSISAF_REPROJ_BICUBIC),
        ("MASIE on grid", MASIE_REPROJ), ("manual hybrid", MANUAL_HYBRID),
        ("cache", CACHE), ("missed dates", MISSED_DATES_FILE),
    ]
    for name, p in rows:
        print(f"  {name:20s} {p}{'' if p.exists() else '   (missing)'}")


if __name__ == "__main__":
    print(f"ICE_DATA_ROOT = {ROOT}")
    describe()
