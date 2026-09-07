"""Analytic OSISAF -> MASAM2 reprojection as a network layer.

Replaces the offline `cdo remapnn` preprocessing step. The reprojection is a
known, fixed geometric map, so there is nothing to learn: the sampling grid is
computed once from the two grid definitions and applied with F.grid_sample.

Verified against the existing `cdo remapnn` output over 18 days spanning
2023-01..2024-11:

    MAE vs cdo remapnn, whole field : 0.00260  (max 0.00457)
    MAE vs cdo remapnn, OCEAN ONLY  : 0.00122  (max 0.00173)

No stored grid: it is regenerated in __init__ from the ten constants below and
registered with persistent=False, so checkpoints stay the size of the weights.
Regeneration was verified equal to the grid derived from target_grid_cropped.nc
to 1.3e-5 source pixels.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- source: OSISAF Lambert Azimuthal Equal Area (from the netCDF CRS attrs) ---
SRC_PROJ = "+proj=laea +lon_0=0 +datum=WGS84 +ellps=WGS84 +lat_0=90.0"
SRC_XC0_KM, SRC_YC0_KM, SRC_STEP_KM, SRC_N = -5387.5, 5387.5, 25.0, 432

# --- target: MASIE 4 km polar stereographic, cropped to the MASAM2 window ---
# (matches target_grid_cropped.nc to ~2e-6 m; see prepare_masie_grid.py)
DST_PROJ = "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-80 +datum=WGS84 +ellps=WGS84"
DST_X0, DST_Y0, DST_STEP = -4286000.0, -4658000.0, 4000.0
DST_SHAPE = (2550, 2100)          # (y, x) of the target grid file
OUT_SHAPE = (2100, 2550)          # orientation used by MASAM2 .npy in this repo


def build_resample_grid(dtype=torch.float32):
    """Sampling grid for F.grid_sample(align_corners=True), shape [1, 2550, 2100, 2].

    Requires pyproj at construction time only.
    """
    from pyproj import CRS, Transformer

    h, w = DST_SHAPE
    xs = DST_X0 + DST_STEP * np.arange(w, dtype=np.float64)
    ys = DST_Y0 + DST_STEP * np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)

    to_lonlat = Transformer.from_crs(CRS.from_proj4(DST_PROJ), CRS.from_epsg(4326), always_xy=True)
    lon, lat = to_lonlat.transform(xx, yy)

    to_laea = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(SRC_PROJ), always_xy=True)
    x_m, y_m = to_laea.transform(lon, lat)

    # fractional index into the source array (yc descends)
    fi = (x_m / 1000.0 - SRC_XC0_KM) / SRC_STEP_KM
    fj = (SRC_YC0_KM - y_m / 1000.0) / SRC_STEP_KM

    gx = 2.0 * fi / (SRC_N - 1) - 1.0
    gy = 2.0 * fj / (SRC_N - 1) - 1.0
    grid = np.stack([gx, gy], axis=-1)[None]           # 1, 2550, 2100, 2
    return torch.from_numpy(grid).to(dtype)


class OsisafToMasam2(nn.Module):
    """Resample a native OSISAF field (B, C, 432, 432) onto the MASAM2 grid,
    returning (B, C, 2100, 2550).

    The transpose at the end matches the orientation of the MASAM2 .npy arrays
    used throughout this repo (established by comparing against the CDO output).
    """

    def __init__(self, mode="nearest", padding_mode="border", grid=None):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        if grid is None:
            grid = build_resample_grid()
        # persistent=False: regenerated on construction, kept out of state_dict
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, x):
        g = self.grid.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1, -1)
        out = F.grid_sample(
            x, g, mode=self.mode, padding_mode=self.padding_mode, align_corners=True
        )
        return out.transpose(-1, -2).contiguous()      # (B, C, 2550, 2100) -> (B, C, 2100, 2550)
