#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py
==========================
Downloads and preprocesses Lacus Mortis Diviner brightness temperature
data from HuggingFace, following the methodology of Moseley et al. (2020).

Pipeline (per paper Section 2.1, 2.4):
  1. Download all 240 .xyz files from HuggingFace (NOTE: some files are missing)
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
import sys
import time
import logging
import warnings
import argparse
import pickle
import numpy as np
from pathlib import Path
import math

import requests
from tqdm import tqdm

from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from concurrent.futures import ProcessPoolExecutor, as_completed

# ═══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  ── edit these three lines before running
# ═══════════════════════════════════════════════════════════════════════════════

HF_REPO_ID   = "arushisinha98/lunar"
N_FILES      = 240
OUTPUT_DIR   = "data/lacus_mortis"
NCPU         = 8

# ═══════════════════════════════════════════════════════════════════════════════

# ── physical / grid constants ─────────────────────────────────────────────────
R_MOON_M      = 1_737_400.0          # lunar radius (m)
AOI_LAT       = 45.0                 # Lacus Mortis centre latitude  (°N)
AOI_LON       = 27.2                 # Lacus Mortis centre longitude (°E)
AOI_RADIUS_KM = None                 # No AOI radius constraint (include all points)

BIN_SIZE_M    = 200.0                # grid bin size (m)  — paper Section 2.4
MAX_TIME_GAP  = 6                    # (paper uses 4 hr)
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


def download_chunk(args):
    repo_id, out_dir_str, start, end = args
    out_dir = Path(out_dir_str)
    chunk_paths = []
    for n in range(start, end + 1):
        fname = f"lacus_mortis-tb-{n:03d}.xyz"
        url = hf_url(repo_id, fname)
        local = out_dir / fname
        if local.exists():
            chunk_paths.append(local)
            continue
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                local.write_bytes(resp.content)
                chunk_paths.append(local)
                break
            except Exception as exc:
                if attempt == 2:
                    log.error("Failed to download %s after 3 attempts: %s", fname, exc)
                else:
                    time.sleep(2 ** attempt)
    return chunk_paths

def download_all_files(repo_id: str, n_files: int, out_dir: Path) -> list[Path]:
    """
    Download lacus_mortis-tb-001.xyz … lacus_mortis-tb-240.xyz into out_dir in parallel using NCPU.
    Each CPU is responsible for a chunk of files. Skips files that already exist.
    Returns list of local paths.
    """
    import math
    import threading
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out_dir.mkdir(parents=True, exist_ok=True)
    local_paths = []
    local_paths_lock = threading.Lock()

    # Divide file indices among NCPU
    indices = list(range(1, n_files + 1))
    chunk_size = math.ceil(len(indices) / NCPU)
    chunks = [indices[i:i+chunk_size] for i in range(0, len(indices), chunk_size)]

    log.info(f"Downloading {n_files} files from HuggingFace repo '{repo_id}' using {NCPU} CPUs")
    results = []
    with ProcessPoolExecutor(max_workers=NCPU) as executor:
        futures = [executor.submit(download_chunk, (repo_id, str(out_dir), chunk[0], chunk[-1])) for chunk in chunks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading (parallel)"):
            chunk_paths = fut.result()
            with local_paths_lock:
                local_paths.extend(chunk_paths)

    log.info("Downloaded %i / %i files", len(local_paths), n_files)
    return local_paths


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Parse .xyz files
# ══════════════════════════════════════════════════════════════════════════════

"""
Expected column layout for your .xyz files:
    col 0 : longitude        (degrees East,  0–360)
    col 1 : latitude         (degrees North, −90–90)
    col 2 : temperature      (Kelvin)
No local_time or emission_angle columns are present.
"""
COL_LON   = 0
COL_LAT   = 1
COL_TEMP  = 2

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

    # Expecting data shape (N, 3): each row is [lon, lat, temp]
    if data.ndim != 2 or data.shape[1] != 3:
        log.error("%s: Expected shape (N, 3), got %s. Skipping file.", path.name, data.shape)
        return None

    n_points = data.shape[0]
    # Build a canonical 5-column array: [lon, lat, local_time, temperature, emission_angle]
    out = np.zeros((n_points, 5), dtype=np.float32)
    out[:, 0] = data[:, 0]  # lon (X)
    out[:, 1] = data[:, 1]  # lat (Y)
    out[:, 2] = np.nan      # local_time missing
    out[:, 3] = data[:, 2]  # temperature (Z)
    out[:, 4] = np.nan      # emission_angle missing
    return out
    

def parse_and_annotate_chunk(args):
    chunk_paths, time_pattern_str = args
    import re
    time_pattern = re.compile(time_pattern_str)
    chunk_results = []
    for path in chunk_paths:
        chunk = parse_xyz_file(path)
        if chunk is not None:
            m = time_pattern.search(str(path))
            if m:
                time_idx = int(m.group(1))
                local_time = (time_idx - 1) * 0.1
            else:
                log.warning(f"Could not extract time from filename: {path}, using 0.0.")
                local_time = 0.0
            chunk[:, 2] = local_time
            chunk_results.append(chunk)
    return chunk_results

def load_all_data(paths: list[Path]) -> np.ndarray:
    """Load and concatenate all .xyz files in parallel using NCPU. Returns (M_total, 5) float32."""
    import math
    from concurrent.futures import ProcessPoolExecutor, as_completed
    time_pattern_str = r"lacus_mortis-tb-(\d{3})\.xyz"

    # Divide paths among NCPU
    chunk_size = math.ceil(len(paths) / NCPU)
    chunks = [paths[i:i+chunk_size] for i in range(0, len(paths), chunk_size)]

    all_chunks = []
    with ProcessPoolExecutor(max_workers=NCPU) as executor:
        futures = [executor.submit(parse_and_annotate_chunk, (chunk, time_pattern_str)) for chunk in chunks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Parsing (parallel)"):
            chunk_result = fut.result()
            all_chunks.extend(chunk_result)

    if not all_chunks:
        raise RuntimeError("No valid data found in any .xyz file.")

    all_data = np.concatenate(all_chunks, axis=0)
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
    """
    lon   = data[:, 0]
    lat   = data[:, 1]
    temp  = data[:, 3]

    n_total = len(data)
    mask_temp_valid = (~np.isnan(temp))
    mask_temp_gt0 = temp > 0.0
    mask_temp_lt450 = temp < 450.0

    n_valid = mask_temp_valid.sum()
    n_gt0 = (mask_temp_valid & mask_temp_gt0).sum()
    n_lt450 = (mask_temp_valid & mask_temp_lt450).sum()
    n_both = (mask_temp_valid & mask_temp_gt0 & mask_temp_lt450).sum()

    log.info("Filter: valid temps (not NaN): %d / %d (%.1f%%)", n_valid, n_total, 100 * n_valid / n_total)
    log.info("Filter: temp > 0: %d / %d (%.1f%%)", n_gt0, n_total, 100 * n_gt0 / n_total)
    log.info("Filter: temp < 450: %d / %d (%.1f%%)", n_lt450, n_total, 100 * n_lt450 / n_total)
    log.info("Filter: temp > 0 & < 450: %d / %d (%.1f%%)", n_both, n_total, 100 * n_both / n_total)

    mask = mask_temp_valid & mask_temp_gt0 & mask_temp_lt450
    filtered = data[mask]
    log.info(
        "After filtering: %s measurements (%.1f %% retained)",
        f"{len(filtered):,}",
        100 * len(filtered) / n_total,
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

    Returns a dict mapping (ix, iy) bin indices → {"ltime": array, "temp": array}.
    """
    lon   = data[:, 0]
    lat   = data[:, 1]
    ltime = data[:, 2]
    temp  = data[:, 3]

    x_m, y_m = lonlat_to_ortho(lon, lat, aoi_lon, aoi_lat)

    # Remove NaN projections (far-side points)
    valid_proj = ~np.isnan(x_m)
    n_removed = int((~valid_proj).sum())
    if n_removed > 0:
        log.info("Removed %d far-side points with NaN projection", n_removed)
    x_m   = x_m[valid_proj]
    y_m   = y_m[valid_proj]
    ltime = ltime[valid_proj]
    temp  = temp[valid_proj]

    # Compute integer bin indices
    ix = np.floor(x_m / bin_size_m).astype(np.int32)
    iy = np.floor(y_m / bin_size_m).astype(np.int32)

    # Vectorized grouping via composite bin ID (avoids slow per-row Python loop)
    log.info("Grouping %s points into spatial bins...", f"{len(ix):,}")
    ix_off = (ix - ix.min()).astype(np.int64)
    iy_off = (iy - iy.min()).astype(np.int64)
    ny = int(iy_off.max()) + 1
    bin_id = ix_off * ny + iy_off

    order = np.argsort(bin_id, kind='mergesort')
    sorted_bin_id = bin_id[order]
    unique_ids, counts = np.unique(sorted_bin_id, return_counts=True)
    offsets = np.zeros(len(unique_ids) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    bins: dict = {}
    for i in range(len(unique_ids)):
        s, e = int(offsets[i]), int(offsets[i + 1])
        idx = order[s:e]
        key = (int(ix[idx[0]]), int(iy[idx[0]]))
        bins[key] = {
            "ltime": ltime[idx].copy(),
            "temp":  temp[idx].copy(),
        }

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

def process_bin_chunk(chunk):
    chunk_profiles = []
    chunk_coords_xy = []
    n_rejected_sparse = 0
    n_rejected_gp_fail = 0
    n_accepted = 0
    for (ix, iy), bin_data in chunk:
        ltime = np.asarray(bin_data["ltime"], dtype=np.float64)
        temp  = np.asarray(bin_data["temp"],  dtype=np.float64)
        # Remove entries where either ltime or temp is NaN
        valid = ~np.isnan(temp) & ~np.isnan(ltime)
        temp  = temp[valid]
        ltime = ltime[valid]
        if len(temp) < 2:
            n_rejected_sparse += 1
            continue
        # Check temporal coverage
        if not check_temporal_coverage(ltime):
            n_rejected_sparse += 1
            continue
        # GP interpolate
        profile = gp_interpolate(ltime, temp)
        if profile is None or profile.shape != (GP_N_SAMPLES,):
            n_rejected_gp_fail += 1
            continue
        chunk_profiles.append(profile)
        x_centre = (ix + 0.5) * BIN_SIZE_M
        y_centre = (iy + 0.5) * BIN_SIZE_M
        chunk_coords_xy.append([x_centre, y_centre])
        n_accepted += 1
    return chunk_profiles, chunk_coords_xy, n_accepted, n_rejected_sparse, n_rejected_gp_fail


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
        normalize_y=False,
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
#  STAGED PIPELINE  —  prepare → gp (sharded) → merge
# ══════════════════════════════════════════════════════════════════════════════

def stage_prepare():
    """Stage 1: Download, parse, filter, bin → save bins.pkl to disk."""
    out_dir = Path(OUTPUT_DIR)
    raw_dir = out_dir / "raw_xyz"

    # ── 1. Download ───────────────────────────────────────────────────────────
    paths = download_all_files(HF_REPO_ID, N_FILES, raw_dir)
    if not paths:
        raise RuntimeError("No files downloaded. Check HF_REPO_ID and network.")

    # ── 2. Parse ──────────────────────────────────────────────────────────────
    log.info("Parsing all .xyz files...")
    all_data = load_all_data(paths)

    log.info("Column diagnostics (check these make physical sense):")
    log.info("  lon   : %.2f – %.2f °E",   all_data[:, 0].min(), all_data[:, 0].max())
    log.info("  lat   : %.2f – %.2f °N",   all_data[:, 1].min(), all_data[:, 1].max())
    log.info("  temp  : %.1f – %.1f K",    np.nanmin(all_data[:, 3]), np.nanmax(all_data[:, 3]))
    log.info("  time  : %.1f – %.1f hr",   np.nanmin(all_data[:, 2]), np.nanmax(all_data[:, 2]))

    # ── 3. Filter ─────────────────────────────────────────────────────────────
    log.info("Applying filters...")
    data = apply_filters(all_data)
    del all_data

    # ── 4. Grid binning ───────────────────────────────────────────────────────
    log.info("Projecting onto 200 m orthographic grid...")
    bins = bin_to_grid(data)
    del data

    # ── 5. Bin-level diagnostics ──────────────────────────────────────────────
    sizes = np.array([len(b["temp"]) for b in bins.values()])
    log.info("Bin sizes: min=%d  median=%d  mean=%.1f  max=%d",
             sizes.min(), int(np.median(sizes)), sizes.mean(), sizes.max())

    # Quick temporal-coverage check on a sample
    n_pass = sum(1 for b in bins.values()
                 if check_temporal_coverage(
                     np.asarray(b["ltime"], dtype=np.float64)))
    log.info("Bins passing temporal-coverage check: %d / %d (%.1f%%)",
             n_pass, len(bins), 100 * n_pass / max(len(bins), 1))

    # ── 6. Save bins ──────────────────────────────────────────────────────────
    bins_path = out_dir / "bins.pkl"
    log.info("Saving %d bins to %s ...", len(bins), bins_path)
    with open(bins_path, "wb") as f:
        pickle.dump(bins, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Prepare stage complete.  bins.pkl size: %.1f MB",
             bins_path.stat().st_size / 1e6)


def stage_gp(shard: int, n_shards: int, ncpu: int):
    """Stage 2: Load bins, GP-interpolate one shard, save shard results."""
    out_dir = Path(OUTPUT_DIR)
    bins_path = out_dir / "bins.pkl"

    log.info("Loading bins from %s ...", bins_path)
    with open(bins_path, "rb") as f:
        bins = pickle.load(f)

    # Sort keys for reproducible sharding
    bin_items = sorted(bins.items())
    del bins
    total = len(bin_items)
    shard_size = math.ceil(total / n_shards)
    start = shard * shard_size
    end = min(start + shard_size, total)
    my_bins = bin_items[start:end]
    del bin_items

    log.info("Shard %d/%d: bins %d–%d of %d (%d bins, %d CPUs)",
             shard, n_shards, start, end - 1, total, len(my_bins), ncpu)

    # Split into many small chunks for finer progress updates
    n_chunks = min(ncpu * 20, max(len(my_bins), 1))
    chunk_size = max(1, math.ceil(len(my_bins) / n_chunks))
    chunks = [my_bins[i:i+chunk_size] for i in range(0, len(my_bins), chunk_size)]

    profiles = []
    coords_xy = []
    total_accepted = 0
    total_rejected_sparse = 0
    total_rejected_gp_fail = 0

    log.info("Dispatching %d chunks to %d workers ...", len(chunks), ncpu)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=ncpu) as executor:
        futures = [executor.submit(process_bin_chunk, chunk) for chunk in chunks]
        for i, fut in enumerate(as_completed(futures)):
            chunk_profiles, chunk_coords_xy, n_acc, n_rej_sparse, n_rej_gp = fut.result()
            profiles.extend(chunk_profiles)
            coords_xy.extend(chunk_coords_xy)
            total_accepted += n_acc
            total_rejected_sparse += n_rej_sparse
            total_rejected_gp_fail += n_rej_gp
            # Log every 10 chunks or on the last chunk
            if (i + 1) % 10 == 0 or (i + 1) == len(futures):
                elapsed = time.time() - t0
                bins_done = (i + 1) * chunk_size
                rate = bins_done / max(elapsed, 1)
                log.info("  [shard %d] %d/%d chunks | accepted=%d  sparse=%d  gp_fail=%d  "
                         "(%.0f bins/s, %.1f min elapsed)",
                         shard, i + 1, len(futures),
                         total_accepted, total_rejected_sparse,
                         total_rejected_gp_fail, rate, elapsed / 60)

    elapsed_total = time.time() - t0
    log.info("Shard %d complete in %.1f min.  Accepted: %d | Sparse: %d | GP fail: %d",
             shard, elapsed_total / 60,
             total_accepted, total_rejected_sparse, total_rejected_gp_fail)

    # Save shard results
    shard_path = out_dir / f"shard_{shard:03d}.npz"
    if profiles:
        X_shard = np.stack(profiles, axis=0)       # (n_accepted, 120)
        xy_shard = np.array(coords_xy, dtype=np.float32)  # (n_accepted, 2)
        np.savez(shard_path, profiles=X_shard, coords_xy=xy_shard)
    else:
        np.savez(shard_path,
                 profiles=np.empty((0, GP_N_SAMPLES), dtype=np.float32),
                 coords_xy=np.empty((0, 2), dtype=np.float32))
    log.info("Saved shard → %s", shard_path)


def stage_merge(n_shards: int):
    """Stage 3: Merge all shard .npz files into final profiles + coords."""
    out_dir = Path(OUTPUT_DIR)

    all_profiles = []
    all_coords = []
    total = 0
    for s in range(n_shards):
        shard_path = out_dir / f"shard_{s:03d}.npz"
        if not shard_path.exists():
            log.warning("Missing shard file: %s", shard_path)
            continue
        d = np.load(shard_path)
        n = d["profiles"].shape[0]
        if n > 0:
            all_profiles.append(d["profiles"])
            all_coords.append(d["coords_xy"])
        total += n
        log.info("Shard %d: %d profiles  (running total: %d)", s, n, total)

    if not all_profiles:
        raise RuntimeError("No profiles found in any shard.")

    profiles_flat = np.concatenate(all_profiles, axis=0)   # (N, 120)
    X = profiles_flat[:, np.newaxis, :]                    # (N, 1, 120)
    xy = np.concatenate(all_coords, axis=0)                # (N, 2)

    log.info("Total profiles: %d | Output shape: %s  dtype: %s", len(X), X.shape, X.dtype)

    # Convert bin-centre xy back to lon/lat
    lon_out, lat_out = ortho_to_lonlat(xy[:, 0], xy[:, 1], AOI_LON, AOI_LAT)
    coords_lonlat = np.stack([lon_out, lat_out], axis=1).astype(np.float32)

    profiles_path = out_dir / "lacus_mortis_profiles.npy"
    coords_path   = out_dir / "lacus_mortis_grid_coords.npy"
    np.save(profiles_path, X)
    np.save(coords_path,   coords_lonlat)

    log.info("Saved profiles → %s", profiles_path)
    log.info("Saved coords   → %s", coords_path)

    flat = X.ravel()
    log.info("=" * 60)
    log.info("  Copy these values into constants.py:")
    log.info("  N_EXAMPLES = %i",   len(X))
    log.info("  T_MU       = %.8f", float(flat.mean()))
    log.info("  T_SIGMA    = %.8f", float(flat.std()))
    log.info("=" * 60)


def main(ncpu: int = NCPU):
    """Run the full pipeline on a single node (prepare → gp → merge)."""
    stage_prepare()
    stage_gp(shard=0, n_shards=1, ncpu=ncpu)
    stage_merge(n_shards=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lacus Mortis preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single node (all stages in one go):
  python preprocess.py all --ncpu 8

  # Multi-node via PBS (run each stage separately):
  python preprocess.py prepare
  python preprocess.py gp --shard 0 --n-shards 16 --ncpu 8
  python preprocess.py merge --n-shards 16
""",
    )
    sub = parser.add_subparsers(dest="stage")

    p_prep = sub.add_parser("prepare", help="Download, parse, filter, bin → save bins.pkl")

    p_gp = sub.add_parser("gp", help="GP interpolation on one shard")
    p_gp.add_argument("--shard", type=int, required=True, help="Shard index (0-based)")
    p_gp.add_argument("--n-shards", type=int, required=True, help="Total number of shards")
    p_gp.add_argument("--ncpu", type=int, default=NCPU, help="CPUs per node")

    p_merge = sub.add_parser("merge", help="Merge shard results into final output")
    p_merge.add_argument("--n-shards", type=int, required=True, help="Total number of shards")

    p_all = sub.add_parser("all", help="Run entire pipeline on one node")
    p_all.add_argument("--ncpu", type=int, default=NCPU, help="CPUs to use")

    args = parser.parse_args()

    if args.stage == "prepare":
        stage_prepare()
    elif args.stage == "gp":
        stage_gp(args.shard, args.n_shards, args.ncpu)
    elif args.stage == "merge":
        stage_merge(args.n_shards)
    elif args.stage == "all":
        main(ncpu=args.ncpu)
    else:
        # No subcommand: run everything
        main()