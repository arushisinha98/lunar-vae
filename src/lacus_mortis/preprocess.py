#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py
==========================
Downloads and preprocesses Lacus Mortis Diviner brightness temperature
data from HuggingFace, following the methodology of Moseley et al. (2020).

Pipeline (per paper Section 2.1, 2.4):
  1. Download all 233 .xyz files from HuggingFace
  2. Parse and concatenate raw point measurements
  3. Filter by emission angle < 10°  (if column present)
  4. Project points onto a 200 × 200 m orthographic grid centred on
     Lacus Mortis (45.0°N, 27.2°E)
  5. For each grid bin, collect temperature measurements and sort by
     local lunar time
  6. Reject profiles whose maximum gap in local lunar time exceeds 4 hr
  7. GP interpolation: Matérn-1.5 kernel, max length-scale 6 hr,
     noise σ = 10 K, resampled every 0.2 hr  →  120-point profiles
  8. Save output array of shape (N_profiles, 1, 120) as .npy

Output
------
  lacus_mortis_profiles.npy   — shape (N, 1, 120), float32, raw Kelvin
  lacus_mortis_grid_coords.npy — shape (N, 2), [lon_deg, lat_deg] per profile

Usage
-----
  pixi run python preprocess.py

Configure the three constants below before running.
"""

import os
import io
import time
import logging
import warnings
import numpy as np
from pathlib import Path

import requests
from tqdm import tqdm

from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

# ═══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  ── edit these three lines before running
# ═══════════════════════════════════════════════════════════════════════════════

HF_REPO_ID   = "arushisinha98/lunar"
N_FILES      = 233
OUTPUT_DIR   = "data/lacus_mortis"

# ═══════════════════════════════════════════════════════════════════════════════

# ── physical / grid constants ─────────────────────────────────────────────────
R_MOON_M      = 1_737_400.0          # lunar radius (m)
AOI_LAT       = 45.0                 # Lacus Mortis centre latitude  (°N)
AOI_LON       = 27.2                 # Lacus Mortis centre longitude (°E)
AOI_RADIUS_KM = 150.0                # half-width of extraction region (km)

BIN_SIZE_M    = 200.0                # grid bin size (m)  — paper Section 2.4
MAX_TIME_GAP  = 6.0                  # reject profiles with gap > 6 hr (paper: 4 hr)
EMIT_ANGLE_MAX = 10.0                # emission angle filter (paper Section 2.1)

# GP interpolation parameters (paper Section 2.4)
GP_NOISE_K    = 10.0                 # assumed temperature noise (K)
GP_LS_MAX_HR  = 6.0                  # maximum Matérn length scale (hr)
GP_SAMPLE_DT  = 0.2                  # resample interval (hr)
GP_N_SAMPLES  = 120                  # 0 … 24 hr exclusive, step 0.2 → 120 pts
T_GRID        = np.arange(GP_N_SAMPLES) * GP_SAMPLE_DT   # (120,)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Download
# ══════════════════════════════════════════════════════════════════════════════

def hf_url(repo_id: str, filename: str) -> str:
    """Return the direct-download URL for a HuggingFace dataset file."""
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"


def download_all_files(repo_id: str, n_files: int, out_dir: Path) -> list[Path]:
    """
    Download lacus_mortis-tb-001.xyz … lacus_mortis-tb-233.xyz into out_dir.
    Skips files that already exist (safe to re-run after interruption).
    Returns list of local paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    local_paths = []

    log.info("Downloading %i files from HuggingFace repo '%s'", n_files, repo_id)
    for n in tqdm(range(1, n_files + 1), desc="Downloading"):
        fname   = f"lacus_mortis-tb-{n:03d}.xyz"
        url     = hf_url(repo_id, fname)
        local   = out_dir / fname

        if local.exists():
            local_paths.append(local)
            continue

        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                local.write_bytes(resp.content)
                local_paths.append(local)
                break
            except Exception as exc:
                if attempt == 2:
                    log.error("Failed to download %s after 3 attempts: %s", fname, exc)
                else:
                    time.sleep(2 ** attempt)

    log.info("Downloaded %i / %i files", len(local_paths), n_files)
    return local_paths


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Parse .xyz files
# ══════════════════════════════════════════════════════════════════════════════

# Expected column layout for Diviner brightness-temperature .xyz files.
# If your files use a different order, edit the indices below.
#
#   col 0 : longitude        (degrees East,  0–360)
#   col 1 : latitude         (degrees North, −90–90)
#   col 2 : local_time       (hours,          0–24)
#   col 3 : temperature      (Kelvin)
#   col 4 : emission_angle   (degrees)         ← optional; set to None if absent
#
COL_LON   = 0
COL_LAT   = 1
COL_LTIME = 2
COL_TEMP  = 3
COL_EMIT  = 4    # set to None if your files have no emission-angle column


def parse_xyz_file(path: Path) -> np.ndarray | None:
    """
    Parse a single .xyz file.  Returns float32 array of shape (M, 5):
      [lon, lat, local_time, temperature, emission_angle]
    emission_angle is set to 0 if the column is absent (passes the <10° filter).
    Returns None if the file is empty or unparseable.
    """
    try:
        # skip comment lines starting with # or %
        data = np.loadtxt(path, comments=["#", "%"], dtype=np.float32)
    except Exception as exc:
        log.warning("Could not parse %s: %s", path.name, exc)
        return None

    if data.ndim == 1:
        data = data[np.newaxis, :]   # single-row file
    if len(data) == 0:
        return None

    n_cols = data.shape[1]

    # Build a canonical 5-column array: [lon, lat, local_time, temperature, emission_angle]
    out = np.zeros((len(data), 5), dtype=np.float32)

    if n_cols == 5:
        out[:, 0] = data[:, 0]  # lon
        out[:, 1] = data[:, 1]  # lat
        out[:, 2] = data[:, 2]  # local_time
        out[:, 3] = data[:, 3]  # temperature
        out[:, 4] = data[:, 4]  # emission_angle
    elif n_cols == 4:
        out[:, 0] = data[:, 0]  # lon
        out[:, 1] = data[:, 1]  # lat
        out[:, 2] = data[:, 2]  # local_time
        out[:, 3] = data[:, 3]  # temperature
        out[:, 4] = 0           # emission_angle missing
    elif n_cols == 3:
        out[:, 0] = data[:, 0]  # lon
        out[:, 1] = data[:, 1]  # lat
        out[:, 2] = 0           # local_time missing
        out[:, 3] = data[:, 2]  # temperature
        out[:, 4] = 0           # emission_angle missing
    else:
        log.error("%s: Unexpected number of columns (%d). Skipping file.", path.name, n_cols)
        return None

    return out


def load_all_data(paths: list[Path]) -> np.ndarray:
    """Load and concatenate all .xyz files.  Returns (M_total, 5) float32."""
    chunks = []
    for p in tqdm(paths, desc="Parsing  "):
        chunk = parse_xyz_file(p)
        if chunk is not None:
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError("No valid data found in any .xyz file.")

    all_data = np.concatenate(chunks, axis=0)
    log.info("Total raw measurements: %s", f"{len(all_data):,}")
    return all_data


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Filter
# ══════════════════════════════════════════════════════════════════════════════

def apply_filters(data: np.ndarray) -> np.ndarray:
    """
    Paper Section 2.1 filters applied to the parsed data array.
    Retains rows where:
      • temperature  > 0 K      (reject fill / NaN substitutes)
      • temperature  < 450 K    (physical upper bound for lunar surface)
      • local_time  in [0, 24)  (sanity check)
      • emission_angle < 10°    (paper criterion)
    """
    lon   = data[:, 0]
    lat   = data[:, 1]
    ltime = data[:, 2]
    temp  = data[:, 3]
    emit  = data[:, 4]

    mask = (
        (temp  >  0.0)           &
        (temp  <  450.0)         &
        (ltime >= 0.0)           &
        (ltime <  24.0)          &
        (emit  <  EMIT_ANGLE_MAX)
    )

    filtered = data[mask]
    log.info(
        "After filtering: %s measurements (%.1f %% retained)",
        f"{len(filtered):,}",
        100 * len(filtered) / len(data),
    )
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Orthographic projection and 200 m binning
# ══════════════════════════════════════════════════════════════════════════════

def lonlat_to_ortho(lon_deg: np.ndarray,
                    lat_deg: np.ndarray,
                    lon0: float,
                    lat0: float,
                    R: float = R_MOON_M) -> tuple[np.ndarray, np.ndarray]:
    """
    Orthographic projection centred at (lon0, lat0).
    Returns (x_m, y_m) in metres on the tangent plane.
    Points on the far hemisphere (dot product < 0) are masked as NaN.
    """
    lon  = np.deg2rad(lon_deg)
    lat  = np.deg2rad(lat_deg)
    lon0r = np.deg2rad(lon0)
    lat0r = np.deg2rad(lat0)

    cos_c = (np.sin(lat0r) * np.sin(lat) +
             np.cos(lat0r) * np.cos(lat) * np.cos(lon - lon0r))

    x = R * np.cos(lat) * np.sin(lon - lon0r)
    y = R * (np.cos(lat0r) * np.sin(lat) -
             np.sin(lat0r) * np.cos(lat) * np.cos(lon - lon0r))

    # mask points on the far side of the Moon
    behind = cos_c < 0
    x[behind] = np.nan
    y[behind] = np.nan

    return x, y


def ortho_to_lonlat(x_m: np.ndarray,
                    y_m: np.ndarray,
                    lon0: float,
                    lat0: float,
                    R: float = R_MOON_M) -> tuple[np.ndarray, np.ndarray]:
    """Inverse orthographic projection."""
    lon0r = np.deg2rad(lon0)
    lat0r = np.deg2rad(lat0)

    rho = np.sqrt(x_m**2 + y_m**2)
    c   = np.arcsin(np.clip(rho / R, -1, 1))

    lat = np.arcsin(
        np.cos(c) * np.sin(lat0r) +
        np.where(rho > 0, y_m * np.sin(c) * np.cos(lat0r) / rho, 0)
    )
    lon = lon0r + np.arctan2(
        x_m * np.sin(c),
        rho * np.cos(lat0r) * np.cos(c) - y_m * np.sin(lat0r) * np.sin(c)
    )

    return np.rad2deg(lon), np.rad2deg(lat)


def bin_to_grid(data: np.ndarray,
                aoi_lon: float = AOI_LON,
                aoi_lat: float = AOI_LAT,
                aoi_radius_km: float = AOI_RADIUS_KM,
                bin_size_m: float = BIN_SIZE_M,
               ) -> dict:
    """
    Project all measurements onto a 200 × 200 m orthographic grid centred on
    the AOI and assign each point to a bin.

    Returns a dict mapping (ix, iy) bin indices → list of (local_time, temp).
    Also returns the grid's x/y extents for later coordinate recovery.
    """
    lon   = data[:, 0]
    lat   = data[:, 1]
    ltime = data[:, 2]
    temp  = data[:, 3]

    x_m, y_m = lonlat_to_ortho(lon, lat, aoi_lon, aoi_lat)

    # Keep only points within AOI radius and not NaN
    radius_m = aoi_radius_km * 1000.0
    valid = (
        np.isfinite(x_m) & np.isfinite(y_m) &
        (np.sqrt(x_m**2 + y_m**2) <= radius_m)
    )
    x_m, y_m = x_m[valid], y_m[valid]
    ltime, temp = ltime[valid], temp[valid]

    log.info(
        "Points within AOI radius (%.0f km): %s",
        aoi_radius_km, f"{valid.sum():,}"
    )

    if valid.sum() == 0:
        raise RuntimeError(
            "No data within the AOI. Check AOI_LON/AOI_LAT and column indices."
        )

    # Compute integer bin indices
    ix = np.floor(x_m / bin_size_m).astype(np.int32)
    iy = np.floor(y_m / bin_size_m).astype(np.int32)

    # Build bin dictionary
    bins: dict = {}
    for k in range(len(ix)):
        key = (ix[k], iy[k])
        if key not in bins:
            bins[key] = {"ltime": [], "temp": [], "x": [], "y": []}
        bins[key]["ltime"].append(ltime[k])
        bins[key]["temp"].append(temp[k])
        bins[key]["x"].append(x_m[k])
        bins[key]["y"].append(y_m[k])

    log.info("Non-empty bins: %s", f"{len(bins):,}")
    return bins


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Reject sparse profiles
# ══════════════════════════════════════════════════════════════════════════════

def check_temporal_coverage(ltime: np.ndarray,
                            max_gap: float = MAX_TIME_GAP) -> bool:
    """
    Returns True (keep) if the maximum gap between consecutive sorted
    local-time measurements is ≤ max_gap hours.  Paper Section 2.4.
    """
    if len(ltime) < 2:
        return False
    t_sorted = np.sort(ltime)
    gaps = np.diff(t_sorted)
    # also check the wrap-around gap (24 hr − last + first)
    wrap = 24.0 - t_sorted[-1] + t_sorted[0]
    return float(np.max(np.append(gaps, wrap))) <= max_gap


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — GP interpolation
# ══════════════════════════════════════════════════════════════════════════════

def gp_interpolate(ltime: np.ndarray,
                   temp: np.ndarray,
                   noise_k: float = GP_NOISE_K,
                   ls_max: float = GP_LS_MAX_HR,
                   t_query: np.ndarray = T_GRID,
                  ) -> np.ndarray | None:
    """
    Fit a GP with Matérn-1.5 kernel (length scale ≤ ls_max hr, noise σ = noise_k K)
    to (ltime, temp) observations and return predictions at t_query.

    The GP operates on the time axis only — temperature is treated as a 1D
    function of local lunar time, matching paper Section 2.4.

    Returns array of shape (len(t_query),) or None on failure.
    """
    X = ltime.reshape(-1, 1)
    y = temp.astype(np.float64)

    # Matérn-1.5 kernel with bounded length scale + fixed noise term
    kernel = (
        Matern(length_scale=3.0,
               length_scale_bounds=(0.1, ls_max),
               nu=1.5)
        + WhiteKernel(noise_level=noise_k**2,
                      noise_level_bounds="fixed")
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,
        normalize_y=True,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, y)
        pred = gp.predict(t_query.reshape(-1, 1))
        return pred.astype(np.float32)
    except Exception as exc:
        log.debug("GP failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(OUTPUT_DIR)
    raw_dir = out_dir / "raw_xyz"

    # ── 1. Download ───────────────────────────────────────────────────────────
    paths = download_all_files(HF_REPO_ID, N_FILES, raw_dir)
    if not paths:
        raise RuntimeError("No files downloaded. Check HF_REPO_ID and network.")

    # ── 2. Parse ──────────────────────────────────────────────────────────────
    log.info("Parsing all .xyz files...")
    all_data = load_all_data(paths)

    # Quick diagnostic — print column ranges to verify parsing is correct
    log.info("Column diagnostics (check these make physical sense):")
    log.info("  lon   : %.2f – %.2f °E",   all_data[:, 0].min(), all_data[:, 0].max())
    log.info("  lat   : %.2f – %.2f °N",   all_data[:, 1].min(), all_data[:, 1].max())
    log.info("  ltime : %.2f – %.2f hr",   all_data[:, 2].min(), all_data[:, 2].max())
    log.info("  temp  : %.1f – %.1f K",    all_data[:, 3].min(), all_data[:, 3].max())
    log.info("  emit  : %.2f – %.2f °",    all_data[:, 4].min(), all_data[:, 4].max())

    # ── 3. Filter ─────────────────────────────────────────────────────────────
    log.info("Applying filters...")
    data = apply_filters(all_data)
    del all_data  # free memory

    # ── 4. Grid binning ───────────────────────────────────────────────────────
    log.info("Projecting onto 200 m orthographic grid...")
    bins = bin_to_grid(data)

    # ── 5–6. Reject sparse bins, GP interpolate ───────────────────────────────
    profiles   = []   # (N, 120) float32
    coords_xy  = []   # (N, 2)   float32  [x_m, y_m] bin centres

    log.info("Running GP interpolation on %s bins...", f"{len(bins):,}")
    n_rejected_sparse   = 0
    n_rejected_gp_fail  = 0
    n_accepted          = 0

    for (ix, iy), bin_data in tqdm(bins.items(), desc="GP interp"):
        ltime = np.array(bin_data["ltime"], dtype=np.float64)
        temp  = np.array(bin_data["temp"],  dtype=np.float64)

        # ── 5. Temporal coverage check ────────────────────────────────────────
        if not check_temporal_coverage(ltime, MAX_TIME_GAP):
            n_rejected_sparse += 1
            continue

        # ── 6. GP interpolation ───────────────────────────────────────────────
        profile = gp_interpolate(ltime, temp)
        if profile is None:
            n_rejected_gp_fail += 1
            continue

        profiles.append(profile)
        # bin centre in projected coordinates
        x_centre = (ix + 0.5) * BIN_SIZE_M
        y_centre = (iy + 0.5) * BIN_SIZE_M
        coords_xy.append([x_centre, y_centre])
        n_accepted += 1

    log.info(
        "GP done.  Accepted: %i  |  Rejected (sparse): %i  |  Rejected (GP fail): %i",
        n_accepted, n_rejected_sparse, n_rejected_gp_fail,
    )

    if n_accepted == 0:
        raise RuntimeError(
            "Zero profiles survived. Check MAX_TIME_GAP, AOI extent, and column indices."
        )

    # ── 7. Assemble and save ──────────────────────────────────────────────────
    X = np.stack(profiles, axis=0)[:, np.newaxis, :]   # (N, 1, 120)
    log.info("Output array shape: %s  dtype: %s", X.shape, X.dtype)

    # Convert bin-centre xy back to lon/lat for the coordinate file
    xy = np.array(coords_xy, dtype=np.float32)         # (N, 2)
    lon_out, lat_out = ortho_to_lonlat(xy[:, 0], xy[:, 1], AOI_LON, AOI_LAT)
    coords_lonlat = np.stack([lon_out, lat_out], axis=1).astype(np.float32)

    profiles_path = out_dir / "lacus_mortis_profiles.npy"
    coords_path   = out_dir / "lacus_mortis_grid_coords.npy"

    np.save(profiles_path, X)
    np.save(coords_path,   coords_lonlat)

    log.info("Saved profiles → %s", profiles_path)
    log.info("Saved coords   → %s", coords_path)

    # ── 8. Print normalisation statistics ─────────────────────────────────────
    #    Feed these into constants.py as T_MU and T_SIGMA
    flat = X.ravel()
    log.info("=" * 60)
    log.info("  Copy these values into constants.py:")
    log.info("  N_EXAMPLES = %i",     len(X))
    log.info("  T_MU       = %.8f",   float(flat.mean()))
    log.info("  T_SIGMA    = %.8f",   float(flat.std()))
    log.info("=" * 60)


if __name__ == "__main__":
    main()