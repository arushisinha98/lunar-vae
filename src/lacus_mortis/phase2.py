#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase2.py — Rille-Proximity Hypothesis Test
============================================
Test whether the VAE's latent 3 (thermal inertia proxy) detects subsurface
structural heterogeneity along the Rimae Bürg rille system in Lacus Mortis.

Hypotheses
----------
  H₀: Latent 3 shows no statistically significant difference between
      profiles within 2 km of the rille and latitude-matched control
      points in the surrounding mare fill (≥5 km from rille).

  H₁: Rille-proximal profiles show systematically elevated latent 3
      values, consistent with reduced regolith thickness or exposed
      higher-conductivity subsurface material.

Method
------
  1. Load latent maps from Phase 1
  2. Compute great-circle distance from each profile to the Rimae Bürg
     rille centerline (approximate polyline)
  3. Select two populations:
       • Rille-proximal : distance ≤ 2 km
       • Control         : distance ≥ 5 km, matched latitude ± 0.5°
  4. Compare latent 3 distributions via:
       • Two-sample Kolmogorov–Smirnov test
       • Mann–Whitney U test
  5. Apply physical transform:  Î = exp(0.93 · z₃ / 2)
  6. Generate spatial residual maps

Inputs
------
  results/lacus_mortis/phase1/latent_means.npy
  results/lacus_mortis/phase1/latent_logvars.npy
  data/lacus_mortis/lacus_mortis_grid_coords.npy
  data/lacus_mortis/lacus_mortis_profiles.npy

Outputs (saved to results/lacus_mortis/phase2/)
-------
  rille_proximity_map.png
  latent3_comparison.png
  thermal_inertia_comparison.png
  latent3_vs_distance.png
  summary.txt

Usage
-----
  pixi run python src/lacus_mortis/phase2.py

Note
----
  Rille waypoints are approximate and should be refined using
  high-resolution LROC imagery.  Slope matching is omitted because
  slope data is not available in the preprocessed output; this is
  noted as a limitation.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ═════════════════════════════════════════════════════════════════════════════
#  Paths
# ═════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR   = PROJECT_ROOT / "results" / "lacus_mortis" / "phase1"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "lacus_mortis" / "phase2"
COORDS_PATH  = PROJECT_ROOT / "data" / "lacus_mortis" / "lacus_mortis_grid_coords.npy"
PROFILES_PATH = PROJECT_ROOT / "data" / "lacus_mortis" / "lacus_mortis_profiles.npy"

# ═════════════════════════════════════════════════════════════════════════════
#  Rille geometry
# ═════════════════════════════════════════════════════════════════════════════

# Approximate centerline waypoints for the main segment of Rimae Bürg
# (lon °E, lat °N).  Digitised from published maps; the rille trends
# roughly NE → SW across the Lacus Mortis mare fill.
#
# IMPORTANT: refine these with LROC NAC imagery for publication-quality work.
RILLE_WAYPOINTS = np.array([
    [28.50, 46.10],
    [28.30, 45.85],
    [28.05, 45.60],
    [27.80, 45.40],
    [27.55, 45.20],
    [27.35, 45.00],
    [27.15, 44.80],
    [26.95, 44.60],
    [26.80, 44.40],
    [26.70, 44.25],
])

# Selection thresholds
RILLE_PROX_KM   = 2.0    # ≤ 2 km from rille centerline
CONTROL_MIN_KM  = 5.0    # ≥ 5 km from rille
LAT_MATCH_DEG   = 0.5    # control latitude band: rille lat ± 0.5°

R_MOON_KM       = 1737.4


# ═════════════════════════════════════════════════════════════════════════════
#  Distance computation
# ═════════════════════════════════════════════════════════════════════════════

def lonlat_to_ortho_km(lon, lat, lon0=27.2, lat0=45.0):
    """
    Orthographic projection centred on Lacus Mortis → (x_km, y_km).
    Accurate to < 0.1% within ~200 km of the centre.
    """
    lon_r, lat_r = np.deg2rad(lon), np.deg2rad(lat)
    lon0_r, lat0_r = np.deg2rad(lon0), np.deg2rad(lat0)

    x = R_MOON_KM * np.cos(lat_r) * np.sin(lon_r - lon0_r)
    y = R_MOON_KM * (
        np.cos(lat0_r) * np.sin(lat_r)
        - np.sin(lat0_r) * np.cos(lat_r) * np.cos(lon_r - lon0_r)
    )
    return x, y


def min_distance_to_polyline_km(lon, lat, waypoints):
    """
    Minimum distance (km) from each profile location to any segment of
    the rille polyline, computed in the orthographic tangent plane.

    Parameters
    ----------
    lon, lat : (N,) arrays in degrees
    waypoints : (M, 2) array of [lon, lat] in degrees

    Returns
    -------
    dist_km : (N,) minimum distance in km
    """
    x_p, y_p = lonlat_to_ortho_km(lon, lat)
    x_w, y_w = lonlat_to_ortho_km(waypoints[:, 0], waypoints[:, 1])

    min_dist2 = np.full(len(x_p), np.inf)

    for j in range(len(x_w) - 1):
        ax, ay = x_w[j], y_w[j]
        bx, by = x_w[j + 1], y_w[j + 1]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy

        if seg_len2 < 1e-12:
            d2 = (x_p - ax) ** 2 + (y_p - ay) ** 2
        else:
            # Parameter t of closest point on segment, clamped to [0, 1]
            t = np.clip(((x_p - ax) * dx + (y_p - ay) * dy) / seg_len2, 0, 1)
            px = ax + t * dx
            py = ay + t * dy
            d2 = (x_p - px) ** 2 + (y_p - py) ** 2

        min_dist2 = np.minimum(min_dist2, d2)

    return np.sqrt(min_dist2)


# ═════════════════════════════════════════════════════════════════════════════
#  Population selection
# ═════════════════════════════════════════════════════════════════════════════

def select_populations(coords, dist_km):
    """
    Select rille-proximal and latitude-matched control populations.

    Returns
    -------
    rille_mask   : (N,) bool — True for rille-proximal profiles
    control_mask : (N,) bool — True for control profiles
    """
    lat = coords[:, 1]

    rille_mask = dist_km <= RILLE_PROX_KM

    # Latitude band spanned by rille-proximal profiles
    if rille_mask.sum() == 0:
        print("  WARNING: No profiles found within {:.1f} km of the rille.".format(
            RILLE_PROX_KM))
        return rille_mask, np.zeros(len(coords), dtype=bool)

    lat_rille_min = lat[rille_mask].min() - LAT_MATCH_DEG
    lat_rille_max = lat[rille_mask].max() + LAT_MATCH_DEG

    control_mask = (
        (dist_km >= CONTROL_MIN_KM)
        & (lat >= lat_rille_min)
        & (lat <= lat_rille_max)
    )

    return rille_mask, control_mask


# ═════════════════════════════════════════════════════════════════════════════
#  Statistical tests
# ═════════════════════════════════════════════════════════════════════════════

def run_tests(z3_rille, z3_control):
    """Two-sample KS test and Mann–Whitney U test on latent 3 values."""
    ks_stat, ks_p = stats.ks_2samp(z3_rille, z3_control)
    mw_stat, mw_p = stats.mannwhitneyu(z3_rille, z3_control, alternative="greater")
    return {
        "ks_stat": ks_stat, "ks_p": ks_p,
        "mw_stat": mw_stat, "mw_p": mw_p,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═════════════════════════════════════════════════════════════════════════════

def plot_rille_proximity_map(coords, dist_km, rille_mask, control_mask, out_dir):
    """Map showing rille, proximal, and control populations."""
    lon, lat = coords[:, 0], coords[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: distance to rille
    vmax = np.percentile(dist_km, 95)
    sc = axes[0].scatter(lon, lat, c=dist_km, s=0.02, cmap="viridis_r",
                         vmin=0, vmax=vmax, rasterized=True, alpha=0.7)
    plt.colorbar(sc, ax=axes[0], label="Distance to rille (km)")
    # Overlay rille centerline
    axes[0].plot(RILLE_WAYPOINTS[:, 0], RILLE_WAYPOINTS[:, 1],
                 "r-", lw=2, label="Rille centerline")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].set_title("Distance to Rimae Bürg")
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    axes[0].set_aspect("equal")

    # Right: population classification
    bg = axes[1].scatter(lon, lat, c="lightgray", s=0.01, rasterized=True, alpha=0.3)
    if control_mask.any():
        axes[1].scatter(lon[control_mask], lat[control_mask],
                        c="steelblue", s=0.02, label="Control (≥5 km)", alpha=0.5, rasterized=True)
    if rille_mask.any():
        axes[1].scatter(lon[rille_mask], lat[rille_mask],
                        c="red", s=0.05, label="Rille-proximal (≤2 km)", alpha=0.7, rasterized=True)
    axes[1].plot(RILLE_WAYPOINTS[:, 0], RILLE_WAYPOINTS[:, 1],
                 "k-", lw=2, label="Rille centerline")
    axes[1].legend(fontsize=9, markerscale=10, loc="upper right")
    axes[1].set_title("Population Classification")
    axes[1].set_xlabel("Longitude (°E)")
    axes[1].set_ylabel("Latitude (°N)")
    axes[1].set_aspect("equal")

    fig.suptitle("Rimae Bürg Rille — Proximity Classification", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "rille_proximity_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved rille_proximity_map.png")


def plot_latent3_comparison(z3_rille, z3_control, test_results, out_dir):
    """Histograms and box plots of latent 3 for both populations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograms
    bins = np.linspace(
        min(z3_rille.min(), z3_control.min()),
        max(z3_rille.max(), z3_control.max()),
        60,
    )
    axes[0].hist(z3_control, bins=bins, density=True, alpha=0.6,
                 color="steelblue", label=f"Control (N={len(z3_control)})")
    axes[0].hist(z3_rille, bins=bins, density=True, alpha=0.6,
                 color="red", label=f"Rille-proximal (N={len(z3_rille)})")
    axes[0].axvline(z3_control.mean(), color="steelblue", ls="--", lw=1.5)
    axes[0].axvline(z3_rille.mean(), color="red", ls="--", lw=1.5)
    axes[0].set_xlabel(r"Latent $z_3$ (thermal inertia proxy)")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=9)
    axes[0].set_title(
        f"KS p = {test_results['ks_p']:.2e}  |  "
        f"MW p = {test_results['mw_p']:.2e}"
    )

    # Box plot
    bp = axes[1].boxplot(
        [z3_control, z3_rille],
        tick_labels=["Control", "Rille-proximal"],
        widths=0.5,
        showfliers=False,
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor("red")
    bp["boxes"][1].set_alpha(0.5)
    axes[1].set_ylabel(r"Latent $z_3$")
    axes[1].set_title("Distribution Comparison")

    fig.suptitle("Latent 3 — Rille vs Control", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "latent3_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved latent3_comparison.png")


def plot_thermal_inertia_comparison(z3_rille, z3_control, out_dir):
    """Compare the physical thermal-inertia transform Î = exp(0.93·z₃/2)."""
    ti_rille   = np.exp(0.93 * z3_rille / 2.0)
    ti_control = np.exp(0.93 * z3_control / 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(
        min(ti_rille.min(), ti_control.min()),
        max(ti_rille.max(), ti_control.max()),
        60,
    )
    axes[0].hist(ti_control, bins=bins, density=True, alpha=0.6,
                 color="steelblue", label=f"Control (N={len(ti_control)})")
    axes[0].hist(ti_rille, bins=bins, density=True, alpha=0.6,
                 color="red", label=f"Rille-proximal (N={len(ti_rille)})")
    axes[0].set_xlabel(r"$\hat{I} = \exp(0.93\,z_3\,/\,2)$  (thermal inertia units)")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=9)
    axes[0].set_title("Thermal Inertia Proxy Distribution")

    # Effect size: difference of means relative to pooled std
    pooled_std = np.sqrt(
        ((len(ti_rille) - 1) * ti_rille.std() ** 2
         + (len(ti_control) - 1) * ti_control.std() ** 2)
        / (len(ti_rille) + len(ti_control) - 2)
    )
    cohens_d = (ti_rille.mean() - ti_control.mean()) / pooled_std if pooled_std > 0 else 0.0

    bp = axes[1].boxplot(
        [ti_control, ti_rille],
        tick_labels=["Control", "Rille-proximal"],
        widths=0.5,
        showfliers=False,
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor("red")
    bp["boxes"][1].set_alpha(0.5)
    axes[1].set_ylabel(r"$\hat{I}$")
    axes[1].set_title(f"Cohen's d = {cohens_d:.3f}")

    fig.suptitle("Thermal Inertia Proxy — Rille vs Control", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "thermal_inertia_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved thermal_inertia_comparison.png")

    return cohens_d


def plot_latent3_vs_distance(mus, dist_km, out_dir):
    """Scatter plot of latent 3 vs distance to rille (binned)."""
    z3 = mus[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw scatter
    mask = dist_km < 30  # focus on profiles near the rille
    axes[0].scatter(dist_km[mask], z3[mask], s=0.02, alpha=0.2,
                    color="gray", rasterized=True)
    axes[0].set_xlabel("Distance to rille (km)")
    axes[0].set_ylabel(r"Latent $z_3$")
    axes[0].set_title("Raw scatter (dist < 30 km)")
    axes[0].axvline(RILLE_PROX_KM, color="red", ls="--", alpha=0.7, label="2 km threshold")
    axes[0].axvline(CONTROL_MIN_KM, color="steelblue", ls="--", alpha=0.7, label="5 km threshold")
    axes[0].legend(fontsize=8)

    # Right: binned means with error bars
    bin_edges = np.arange(0, 31, 1.0)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = np.full(len(bin_centres), np.nan)
    bin_stds  = np.full(len(bin_centres), np.nan)
    bin_ns    = np.zeros(len(bin_centres), dtype=int)

    for k in range(len(bin_centres)):
        in_bin = (dist_km >= bin_edges[k]) & (dist_km < bin_edges[k + 1])
        if in_bin.sum() >= 10:
            bin_means[k] = z3[in_bin].mean()
            bin_stds[k]  = z3[in_bin].std() / np.sqrt(in_bin.sum())  # SEM
            bin_ns[k]    = in_bin.sum()

    valid = ~np.isnan(bin_means)
    axes[1].errorbar(bin_centres[valid], bin_means[valid], yerr=bin_stds[valid],
                     fmt="o-", color="black", markersize=4, capsize=3)
    axes[1].axvline(RILLE_PROX_KM, color="red", ls="--", alpha=0.7)
    axes[1].axvline(CONTROL_MIN_KM, color="steelblue", ls="--", alpha=0.7)
    axes[1].set_xlabel("Distance to rille (km)")
    axes[1].set_ylabel(r"Mean latent $z_3$ ± SEM")
    axes[1].set_title("Binned mean (1 km bins)")

    fig.suptitle("Latent 3 vs Distance to Rimae Bürg", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "latent3_vs_distance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved latent3_vs_distance.png")


def plot_spatial_residual(mus, coords, dist_km, rille_mask, control_mask, out_dir):
    """
    Map latent 3 residual after removing the latitude trend,
    to highlight rille-correlated anomalies.
    """
    lon, lat = coords[:, 0], coords[:, 1]
    z3 = mus[:, 3]

    # Fit a latitude-only trend on the control population
    if control_mask.sum() < 10:
        print("  Skipping spatial residual (insufficient control points).")
        return

    from numpy.polynomial.polynomial import polyfit, polyval
    coeffs = polyfit(lat[control_mask], z3[control_mask], deg=2)
    z3_trend = polyval(lat, coeffs)
    residual = z3 - z3_trend

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: latent 3 residual map
    vmin, vmax = np.percentile(residual, [2, 98])
    sc = axes[0].scatter(lon, lat, c=residual, s=0.02, cmap="RdBu_r",
                         vmin=vmin, vmax=vmax, rasterized=True, alpha=0.7)
    plt.colorbar(sc, ax=axes[0], label=r"$z_3$ residual (lat trend removed)")
    axes[0].plot(RILLE_WAYPOINTS[:, 0], RILLE_WAYPOINTS[:, 1],
                 "k-", lw=2, label="Rille")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].set_title("Latent 3 — Latitude-Detrended Residual")
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    axes[0].set_aspect("equal")

    # Right: residual vs distance (binned)
    bin_edges = np.arange(0, 31, 1.0)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = np.full(len(bin_centres), np.nan)
    bin_sems  = np.full(len(bin_centres), np.nan)

    for k in range(len(bin_centres)):
        in_bin = (dist_km >= bin_edges[k]) & (dist_km < bin_edges[k + 1])
        if in_bin.sum() >= 10:
            bin_means[k] = residual[in_bin].mean()
            bin_sems[k]  = residual[in_bin].std() / np.sqrt(in_bin.sum())

    valid = ~np.isnan(bin_means)
    axes[1].errorbar(bin_centres[valid], bin_means[valid], yerr=bin_sems[valid],
                     fmt="o-", color="black", markersize=4, capsize=3)
    axes[1].axhline(0, color="gray", ls=":", alpha=0.5)
    axes[1].axvline(RILLE_PROX_KM, color="red", ls="--", alpha=0.7, label="2 km")
    axes[1].axvline(CONTROL_MIN_KM, color="steelblue", ls="--", alpha=0.7, label="5 km")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_xlabel("Distance to rille (km)")
    axes[1].set_ylabel(r"Mean $z_3$ residual ± SEM")
    axes[1].set_title("Detrended Residual vs Distance")

    fig.suptitle("Spatial Residual Analysis — Latent 3", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "spatial_residual.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved spatial_residual.png")


# ═════════════════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════════════════

def write_summary(z3_rille, z3_control, test_results, cohens_d,
                  n_total, n_rille, n_control, dist_km, out_dir):
    """Write and print a numerical summary."""
    ti_rille   = np.exp(0.93 * z3_rille / 2.0)
    ti_control = np.exp(0.93 * z3_control / 2.0)

    lines = []
    lines.append("Phase 2: Rille-Proximity Hypothesis Test")
    lines.append("=" * 50)
    lines.append("")
    lines.append("POPULATIONS:")
    lines.append(f"  Total profiles         : {n_total}")
    lines.append(f"  Rille-proximal (≤{RILLE_PROX_KM} km): {n_rille}")
    lines.append(f"  Control (≥{CONTROL_MIN_KM} km, lat-matched): {n_control}")
    lines.append(f"  Distance range         : [{dist_km.min():.2f}, {dist_km.max():.2f}] km")
    lines.append("")
    lines.append("LATENT 3 (z₃):")
    lines.append(f"  Rille-proximal : mean={z3_rille.mean():.4f} ± {z3_rille.std():.4f}")
    lines.append(f"  Control        : mean={z3_control.mean():.4f} ± {z3_control.std():.4f}")
    lines.append(f"  Difference     : {z3_rille.mean() - z3_control.mean():.4f}")
    lines.append("")
    lines.append("THERMAL INERTIA PROXY (Î = exp(0.93·z₃/2)):")
    lines.append(f"  Rille-proximal : mean={ti_rille.mean():.4f} ± {ti_rille.std():.4f}")
    lines.append(f"  Control        : mean={ti_control.mean():.4f} ± {ti_control.std():.4f}")
    lines.append(f"  Difference     : {ti_rille.mean() - ti_control.mean():.4f}")
    lines.append(f"  Cohen's d      : {cohens_d:.4f}")
    lines.append("")
    lines.append("STATISTICAL TESTS:")
    lines.append(f"  KS test   : D = {test_results['ks_stat']:.4f},  "
                 f"p = {test_results['ks_p']:.2e}")
    lines.append(f"  MW U test : U = {test_results['mw_stat']:.0f},  "
                 f"p = {test_results['mw_p']:.2e}  (one-sided: rille > control)")
    lines.append("")

    if test_results["ks_p"] < 0.05 and test_results["mw_p"] < 0.05:
        lines.append("CONCLUSION: Reject H₀ at α = 0.05.")
        if z3_rille.mean() > z3_control.mean():
            lines.append("  Rille-proximal profiles show elevated latent 3,")
            lines.append("  consistent with H₁ (higher thermal inertia near rille).")
        else:
            lines.append("  However, the direction is opposite to H₁ prediction —")
            lines.append("  rille profiles have LOWER latent 3 than controls.")
    else:
        lines.append("CONCLUSION: Cannot reject H₀ at α = 0.05.")
        lines.append("  No statistically significant difference detected.")

    lines.append("")
    lines.append("LIMITATIONS:")
    lines.append("  - Rille waypoints are approximate (refine with LROC NAC)")
    lines.append("  - No slope matching (DEM data not available)")
    lines.append("  - Preprocessed profiles have low T_MU (~53 K) due to")
    lines.append("    GP kernel misconfiguration; latent values may not")
    lines.append("    reflect the full diurnal thermal cycle.")

    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text)
    print(text)


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Phase 1 results ──────────────────────────────────────────────
    print("Loading Phase 1 results...")
    mus     = np.load(PHASE1_DIR / "latent_means.npy")     # (N, 4)
    logvars = np.load(PHASE1_DIR / "latent_logvars.npy")   # (N, 4)
    coords  = np.load(COORDS_PATH).astype(np.float32)      # (N, 2)
    print(f"  Loaded {len(mus)} latent vectors + coordinates")

    # ── Compute distances ─────────────────────────────────────────────────
    print("Computing distances to Rimae Bürg rille...")
    dist_km = min_distance_to_polyline_km(
        coords[:, 0], coords[:, 1], RILLE_WAYPOINTS
    )
    print(f"  Distance range: [{dist_km.min():.2f}, {dist_km.max():.2f}] km")
    print(f"  Profiles within {RILLE_PROX_KM} km: {(dist_km <= RILLE_PROX_KM).sum()}")
    print(f"  Profiles beyond {CONTROL_MIN_KM} km: {(dist_km >= CONTROL_MIN_KM).sum()}")

    # ── Select populations ────────────────────────────────────────────────
    print("\nSelecting populations...")
    rille_mask, control_mask = select_populations(coords, dist_km)
    n_rille  = rille_mask.sum()
    n_control = control_mask.sum()
    print(f"  Rille-proximal : {n_rille}")
    print(f"  Control (lat-matched): {n_control}")

    if n_rille < 10 or n_control < 10:
        print("\n  ERROR: Insufficient profiles in one or both populations.")
        print("  Check rille waypoints and selection thresholds.")
        return

    z3_rille   = mus[rille_mask, 3]
    z3_control = mus[control_mask, 3]

    # ── Statistical tests ─────────────────────────────────────────────────
    print("\nRunning statistical tests...")
    test_results = run_tests(z3_rille, z3_control)
    print(f"  KS test : D = {test_results['ks_stat']:.4f}, "
          f"p = {test_results['ks_p']:.2e}")
    print(f"  MW U    : U = {test_results['mw_stat']:.0f}, "
          f"p = {test_results['mw_p']:.2e}")

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_rille_proximity_map(coords, dist_km, rille_mask, control_mask, OUTPUT_DIR)
    plot_latent3_comparison(z3_rille, z3_control, test_results, OUTPUT_DIR)
    cohens_d = plot_thermal_inertia_comparison(z3_rille, z3_control, OUTPUT_DIR)
    plot_latent3_vs_distance(mus, dist_km, OUTPUT_DIR)
    plot_spatial_residual(mus, coords, dist_km, rille_mask, control_mask, OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n")
    write_summary(z3_rille, z3_control, test_results, cohens_d,
                  len(mus), n_rille, n_control, dist_km, OUTPUT_DIR)

    # ── Save distance array for downstream use ────────────────────────────
    np.save(OUTPUT_DIR / "rille_distance_km.npy", dist_km)
    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
