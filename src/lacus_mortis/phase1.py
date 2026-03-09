#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase1.py — Zero-Shot Generalization Test
==========================================
Run the pre-trained VAE (trained on 48 AOIs from Moseley et al. 2020)
directly on Lacus Mortis profiles *without retraining* to test whether the
learned latent space generalizes to an unseen region.

Inputs
------
  data/lacus_mortis/lacus_mortis_profiles.npy      — (N, 1, 120) float32 Kelvin
  data/lacus_mortis/lacus_mortis_grid_coords.npy   — (N, 2) float32 [lon°E, lat°N]
  results/models/vae_0.20_4_32_f_l2_fin/model_final.torch

Outputs  (saved to results/lacus_mortis/phase1/)
-------
  latent_means.npy            — (N, 4)  latent posterior means
  latent_logvars.npy          — (N, 4)  latent posterior log-variances
  reconstruction_l1.npy       — (N,)    per-profile L1 loss (normalised)
  reconstruction_l2.npy       — (N,)    per-profile L2 loss (normalised)
  latent_maps.png             — 2×2 spatial maps of each latent dimension
  latent3_thermal_inertia.png — detailed latent-3 / thermal-inertia proxy map
  reconstruction_loss_map.png — spatial reconstruction quality
  sample_profiles.png         — example profiles with VAE reconstructions
  latent_distributions.png    — histograms per latent dimension
  posterior_uncertainty.png    — encoder confidence map
  summary.txt                 — numerical summary

Usage
-----
  pixi run python src/lacus_mortis/phase1.py

DATA-QUALITY NOTE
-----------------
The preprocessed Lacus Mortis profiles have T_MU ≈ 53 K, far below the
original training data T_MU ≈ 192 K.  Root cause: the GP kernel in
preprocess.py uses  Matern(amplitude=1) + WhiteKernel(noise=100)  without
an explicit ConstantKernel for the signal amplitude, giving an SNR of
~0.01.  Combined with normalize_y=False, the GP predictions collapse
toward the zero-mean prior instead of tracking the observed 90–350 K
diurnal cycle.

Recommended fixes to preprocess.py (see also suggestions below):
  1. Add ConstantKernel:  ConstantKernel(1e4) * Matern(...) + WhiteKernel(...)
  2. Set normalize_y=True so the GP prior mean equals the data mean.
  3. Optionally wrap observations periodically (replicate ±24 hr) so
     the GP sees the wraparound and interpolates smoothly at t=0/24.

Despite this, Phase 1 uses the ORIGINAL training normalisation (T_MU=192.39,
T_SIGMA=99.14) because the model was trained on those statistics.  This is
the only correct approach for a zero-shot test; using the Lacus Mortis
statistics would distort the input distribution the model expects.
"""

import sys
import os

# Add parent directory (src/) to the Python path so we can import
# constants, models, losses, etc.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from constants import Constants
from models import VAE

# ═════════════════════════════════════════════════════════════════════════════
#  Paths
# ═════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROFILES_PATH = PROJECT_ROOT / "data" / "lacus_mortis" / "lacus_mortis_profiles.npy"
COORDS_PATH   = PROJECT_ROOT / "data" / "lacus_mortis" / "lacus_mortis_grid_coords.npy"
MODEL_PATH    = PROJECT_ROOT / "results" / "models" / "vae_0.20_4_32_f_l2_fin" / "model_final.torch"
OUTPUT_DIR    = PROJECT_ROOT / "results" / "lacus_mortis" / "phase1"

# Time grid for profile x-axis (0 to 23.8 hr in 0.2 hr steps, 120 points)
T_GRID = np.arange(120) * 0.2


# ═════════════════════════════════════════════════════════════════════════════
#  Core functions
# ═════════════════════════════════════════════════════════════════════════════

def load_model():
    """Load the pre-trained VAE with the original training constants."""
    c = Constants()
    model = VAE(c)

    # model_final.torch was saved with torch.save(model.state_dict(), ...)
    # while numbered checkpoints use {'model_state_dict': ...}.
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # model_final.torch is a raw state_dict
        model.load_state_dict(checkpoint)

    model.eval()
    return model, c


def load_data():
    """Load preprocessed Lacus Mortis profiles and grid coordinates."""
    profiles = np.load(PROFILES_PATH).astype(np.float32)   # (N, 1, 120)
    coords   = np.load(COORDS_PATH).astype(np.float32)     # (N, 2) [lon, lat]
    return profiles, coords


def apply_periodic_padding(profiles):
    """
    Same cyclic padding as TtDataset.__init__:
      prepend last time-step, append first → (N, 1, 120) → (N, 1, 122).
    This lets the convolutional encoder treat t=0 and t=23.8 hr as
    neighbours on a periodic domain.
    """
    return np.concatenate(
        [profiles[:, :, -1:], profiles, profiles[:, :, :1]], axis=2
    ).astype(np.float32)


def normalize(X_padded, c):
    """Normalize using the ORIGINAL training statistics."""
    return ((X_padded - c.T_MU) / c.T_SIGMA).astype(np.float32)


def encode_all(model, X_norm, batch_size=2048):
    """
    Encode all profiles.  Uses mu directly (no reparameterisation noise)
    for deterministic, reproducible latent maps.

    Returns
    -------
    mus      : (N, N_LATENT) latent posterior means
    logvars  : (N, N_LATENT) latent posterior log-variances
    recons   : (N, 1, 122)  reconstructed profiles (normalised, padded)
    """
    all_mu, all_logvar, all_recon = [], [], []

    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):
            batch = torch.from_numpy(X_norm[i : i + batch_size])
            mu, logvar = model.encode(batch)
            # Decode deterministically from the posterior mean
            recon = model.decode(mu)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            all_recon.append(recon.cpu().numpy())

    mus     = np.concatenate(all_mu, axis=0).squeeze(-1)       # (N, 4)
    logvars = np.concatenate(all_logvar, axis=0).squeeze(-1)   # (N, 4)
    recons  = np.concatenate(all_recon, axis=0)                # (N, 1, 122)
    return mus, logvars, recons


def reconstruction_metrics(X_norm, recons):
    """
    Per-profile L1 and L2 on the core 120 points (strip periodic padding).
    Both X_norm and recons are (N, 1, 122) in normalised units.
    """
    x = X_norm[:, :, 1:-1]     # (N, 1, 120)
    r = recons[:, :, 1:-1]
    l1 = np.mean(np.abs(x - r), axis=(1, 2))
    l2 = np.mean((x - r) ** 2, axis=(1, 2))
    return l1, l2


# ═════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

LATENT_LABELS = [
    "Latent 0 (solar onset / slope aspect)",
    "Latent 1 (effective albedo / peak T)",
    "Latent 2 (cumulative illumination)",
    "Latent 3 (thermal inertia / conductivity)",
]


def _scatter_map(ax, lon, lat, values, cmap, label, vmin=None, vmax=None):
    """Reusable scatter-plot on a lon/lat axis."""
    if vmin is None:
        vmin, vmax = np.percentile(values, [2, 98])
    sc = ax.scatter(lon, lat, c=values, s=0.02, cmap=cmap,
                    vmin=vmin, vmax=vmax, rasterized=True, alpha=0.7)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(label, fontsize=9)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal")


def plot_latent_maps(mus, coords, out_dir):
    """2×2 panel of all four latent dimensions."""
    lon, lat = coords[:, 0], coords[:, 1]
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    for i, ax in enumerate(axes.flat):
        _scatter_map(ax, lon, lat, mus[:, i], "viridis", LATENT_LABELS[i])
        ax.set_title(LATENT_LABELS[i], fontsize=11)

    fig.suptitle("Lacus Mortis — VAE Latent Maps (Zero-Shot)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "latent_maps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved latent_maps.png")


def plot_latent3_detailed(mus, coords, out_dir):
    """Detailed side-by-side: raw latent 3 and thermal-inertia transform."""
    lon, lat = coords[:, 0], coords[:, 1]
    z3 = mus[:, 3]
    thermal_inertia = np.exp(0.93 * z3 / 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    _scatter_map(axes[0], lon, lat, z3, "magma", r"$z_3$")
    axes[0].set_title("Latent 3 — Raw VAE Output", fontsize=12)

    _scatter_map(axes[1], lon, lat, thermal_inertia, "inferno",
                 r"$\hat{I} = \exp(0.93\,z_3\,/\,2)$")
    axes[1].set_title("Thermal Inertia Proxy", fontsize=12)

    fig.suptitle("Lacus Mortis — Latent 3 / Thermal Inertia", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "latent3_thermal_inertia.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved latent3_thermal_inertia.png")


def plot_reconstruction_loss_map(l1, l2, coords, out_dir):
    """Spatial maps of per-profile reconstruction loss."""
    lon, lat = coords[:, 0], coords[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, metric, name in [(axes[0], l1, "L1"), (axes[1], l2, "L2")]:
        _scatter_map(ax, lon, lat, metric, "hot_r", f"Mean {name} (normalised)")
        ax.set_title(f"Reconstruction {name} Loss", fontsize=12)

    fig.suptitle("Lacus Mortis — Reconstruction Quality (Zero-Shot)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "reconstruction_loss_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved reconstruction_loss_map.png")


def plot_sample_profiles(profiles, recons_K, mus, coords, out_dir, n_samples=8):
    """Show example profiles (Kelvin) with their VAE reconstructions."""
    np.random.seed(42)
    n = len(profiles)
    indices = np.random.choice(n, size=min(n_samples, n), replace=False)
    indices = indices[np.argsort(profiles[indices, 0, :].mean(axis=1))]

    n_cols = 4
    n_rows = int(np.ceil(len(indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows), squeeze=False)

    for idx_plot, idx_data in enumerate(indices):
        row, col = divmod(idx_plot, n_cols)
        ax = axes[row, col]

        orig  = profiles[idx_data, 0, :]       # (120,) Kelvin
        recon = recons_K[idx_data, 0, :]        # (120,) Kelvin

        ax.plot(T_GRID, orig,  "b-",  lw=1.0, label="Original", alpha=0.8)
        ax.plot(T_GRID, recon, "r--", lw=1.0, label="Reconstruction", alpha=0.8)

        lon_i, lat_i = coords[idx_data]
        z = mus[idx_data]
        ax.set_title(
            f"({lon_i:.2f}°E, {lat_i:.2f}°N)\n"
            f"z=[{z[0]:.1f}, {z[1]:.1f}, {z[2]:.1f}, {z[3]:.1f}]",
            fontsize=8,
        )
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 6))
        if idx_plot == 0:
            ax.legend(fontsize=7)
        if col == 0:
            ax.set_ylabel("Temperature (K)")
        if row == n_rows - 1:
            ax.set_xlabel("Local time (hr)")

    for idx_plot in range(len(indices), n_rows * n_cols):
        row, col = divmod(idx_plot, n_cols)
        axes[row, col].axis("off")

    fig.suptitle("Sample Profiles: Original vs VAE Reconstruction (Kelvin)",
                 fontsize=13, y=1.0)
    plt.tight_layout()
    fig.savefig(out_dir / "sample_profiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sample_profiles.png")


def plot_latent_distributions(mus, out_dir):
    """Histogram of each latent dimension."""
    n_latent = mus.shape[1]
    fig, axes = plt.subplots(1, n_latent, figsize=(4 * n_latent, 4))

    short_labels = [r"$z_0$ (slope)", r"$z_1$ (albedo)",
                    r"$z_2$ (illumination)", r"$z_3$ (thermal inertia)"]

    for i, ax in enumerate(axes):
        ax.hist(mus[:, i], bins=100, density=True, alpha=0.7, color="steelblue")
        mu_i = mus[:, i].mean()
        ax.axvline(mu_i, color="red", ls="--",
                   label=f"μ = {mu_i:.2f}")
        ax.set_xlabel(short_labels[i])
        ax.set_ylabel("Density" if i == 0 else "")
        ax.legend(fontsize=8)

    fig.suptitle("Latent Distributions — Lacus Mortis (Zero-Shot)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "latent_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved latent_distributions.png")


def plot_posterior_uncertainty(logvars, coords, out_dir):
    """Map of total encoder posterior uncertainty (root-sum-square of σ)."""
    lon, lat = coords[:, 0], coords[:, 1]
    stds = np.exp(0.5 * logvars)                          # (N, 4)
    total_std = np.sqrt(np.sum(stds ** 2, axis=1))        # (N,)

    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_map(ax, lon, lat, total_std, "plasma",
                 r"$\sqrt{\sum_i \sigma_i^2}$")
    ax.set_title("Encoder Posterior Uncertainty", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "posterior_uncertainty.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved posterior_uncertainty.png")


# ═════════════════════════════════════════════════════════════════════════════
#  Reference-data comparison (framework)
# ═════════════════════════════════════════════════════════════════════════════

def compare_with_reference(mus, coords, out_dir):
    """
    Compare latent 3 against external reference datasets if available.

    Expected files (save as .npy arrays matched to the same grid, or as
    (M, 3) arrays of [lon, lat, value] that will be nearest-neighbour
    interpolated onto the profile grid):

      data/lacus_mortis/h_parameter.npy       — Hayne et al. 2017
      data/lacus_mortis/rock_abundance.npy    — Bandfield et al. 2011

    If the files don't exist, this function prints instructions and returns.
    """
    from scipy.spatial import cKDTree

    z3 = mus[:, 3]
    lon, lat = coords[:, 0], coords[:, 1]

    ref_datasets = {
        "H-parameter (Hayne et al. 2017)": (
            PROJECT_ROOT / "data" / "lacus_mortis" / "h_parameter.npy",
            "H-parameter",
        ),
        "Rock abundance (Bandfield et al. 2011)": (
            PROJECT_ROOT / "data" / "lacus_mortis" / "rock_abundance.npy",
            "Rock abundance",
        ),
    }

    for label, (ref_path, short) in ref_datasets.items():
        if not ref_path.exists():
            print(f"  ⚠  {label} not found at {ref_path}")
            print(f"      → Download from PDS and save as (M, 3) array [lon, lat, value]")
            continue

        ref = np.load(ref_path).astype(np.float64)  # (M, 3): lon, lat, value
        if ref.ndim != 2 or ref.shape[1] != 3:
            print(f"  ⚠  {label}: expected shape (M, 3), got {ref.shape}. Skipping.")
            continue

        # Nearest-neighbour interpolation onto profile grid
        tree = cKDTree(ref[:, :2])
        _, idx = tree.query(np.column_stack([lon, lat]))
        ref_values = ref[idx, 2]

        # Pearson correlation
        mask = np.isfinite(ref_values) & np.isfinite(z3)
        r, p = stats.pearsonr(z3[mask], ref_values[mask])
        print(f"  {label}:")
        print(f"    Pearson r = {r:.4f},  p = {p:.2e}  (N = {mask.sum()})")

        # Scatter plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(z3[mask], ref_values[mask], s=0.05, alpha=0.3, rasterized=True)
        axes[0].set_xlabel(r"Latent $z_3$")
        axes[0].set_ylabel(short)
        axes[0].set_title(f"Pearson r = {r:.4f} (p = {p:.2e})")

        # Spatial residual: ref_value - linear_prediction(z3)
        slope, intercept = np.polyfit(z3[mask], ref_values[mask], 1)
        residual = np.full(len(z3), np.nan)
        residual[mask] = ref_values[mask] - (slope * z3[mask] + intercept)
        _scatter_map(axes[1], lon[mask], lat[mask], residual[mask],
                     "RdBu_r", f"Residual ({short})")
        axes[1].set_title("Spatial Residual (observed − predicted)")

        fig.suptitle(f"Latent 3 vs {label}", fontsize=13, y=1.01)
        plt.tight_layout()
        fname = f"comparison_{short.lower().replace(' ', '_')}.png"
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {fname}")


# ═════════════════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════════════════

def write_summary(profiles, mus, logvars, l1, l2, c, out_dir):
    """Write and print a numerical summary of the analysis."""
    lines = []
    lines.append("Phase 1: Zero-Shot Generalization — Lacus Mortis")
    lines.append("=" * 55)
    lines.append("")
    lines.append("DATA QUALITY:")
    lines.append(f"  Lacus Mortis  T_MU    = {profiles.mean():.2f} K")
    lines.append(f"  Lacus Mortis  T_SIGMA = {profiles.std():.2f} K")
    lines.append(f"  Orig training T_MU    = {c.T_MU:.2f} K")
    lines.append(f"  Orig training T_SIGMA = {c.T_SIGMA:.2f} K")
    lines.append(f"  Normalised LM range   = [{(profiles.min()-c.T_MU)/c.T_SIGMA:.2f}, "
                 f"{(profiles.max()-c.T_MU)/c.T_SIGMA:.2f}]")
    lines.append("")
    lines.append("INPUT:")
    lines.append(f"  N profiles   = {len(profiles)}")
    lines.append(f"  Profile shape = {profiles.shape}")
    lines.append(f"  T range       = [{profiles.min():.1f}, {profiles.max():.1f}] K")
    lines.append(f"  Model         = {MODEL_PATH.name}")
    lines.append(f"  Normalization = original training "
                 f"(T_MU={c.T_MU:.2f}, T_SIGMA={c.T_SIGMA:.2f})")
    lines.append("")
    lines.append("LATENT SPACE:")
    for i in range(mus.shape[1]):
        lines.append(
            f"  z{i}: mean={mus[:,i].mean():.4f}  "
            f"std={mus[:,i].std():.4f}  "
            f"range=[{mus[:,i].min():.4f}, {mus[:,i].max():.4f}]"
        )
    lines.append("")
    lines.append("RECONSTRUCTION QUALITY (normalised scale):")
    lines.append(f"  L1: mean={l1.mean():.6f}  std={l1.std():.6f}  "
                 f"median={np.median(l1):.6f}")
    lines.append(f"  L2: mean={l2.mean():.6f}  std={l2.std():.6f}  "
                 f"median={np.median(l2):.6f}")
    lines.append("")
    lines.append("POSTERIOR UNCERTAINTY:")
    stds = np.exp(0.5 * logvars)
    for i in range(logvars.shape[1]):
        lines.append(f"  σ(z{i}): mean={stds[:,i].mean():.4f}  "
                     f"std={stds[:,i].std():.4f}")
    lines.append("")
    lines.append(f"LATENT CORRELATIONS (Pearson r, N={len(mus)}):")
    for i in range(mus.shape[1]):
        for j in range(i + 1, mus.shape[1]):
            r, p = stats.pearsonr(mus[:, i], mus[:, j])
            lines.append(f"  z{i} vs z{j}: r={r:.4f}, p={p:.2e}")

    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text)
    print(text)


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading model...")
    model, c = load_model()

    print("Loading Lacus Mortis data...")
    profiles, coords = load_data()
    print(f"  Profiles : {profiles.shape}  "
          f"[{profiles.min():.1f}, {profiles.max():.1f}] K  "
          f"mean={profiles.mean():.1f} K")
    print(f"  Coords   : {coords.shape}  "
          f"lon=[{coords[:,0].min():.2f}, {coords[:,0].max():.2f}]  "
          f"lat=[{coords[:,1].min():.2f}, {coords[:,1].max():.2f}]")

    print(f"\n  ⚠  Lacus Mortis T_MU = {profiles.mean():.2f} K  "
          f"(training T_MU = {c.T_MU:.2f} K)")
    print(f"  → Using ORIGINAL normalisation for valid zero-shot test.\n")

    # ── Prepare input ─────────────────────────────────────────────────────
    print("Applying periodic padding + normalisation...")
    X_padded = apply_periodic_padding(profiles)          # (N, 1, 122) Kelvin
    X_norm   = normalize(X_padded, c)                    # (N, 1, 122) normalised

    # ── Encode ────────────────────────────────────────────────────────────
    print("Running VAE encoder (deterministic)...")
    mus, logvars, recons = encode_all(model, X_norm)
    print(f"  Latent means shape : {mus.shape}")
    print(f"  Reconstruction shape: {recons.shape}")

    # ── Metrics ───────────────────────────────────────────────────────────
    l1, l2 = reconstruction_metrics(X_norm, recons)
    print(f"  L1 loss: {l1.mean():.6f} ± {l1.std():.6f}")
    print(f"  L2 loss: {l2.mean():.6f} ± {l2.std():.6f}")

    # Denormalise reconstructions for per-profile Kelvin plots
    recons_K = (recons * c.T_SIGMA + c.T_MU)            # (N, 1, 122) Kelvin
    recons_K_core = recons_K[:, :, 1:-1]                 # (N, 1, 120) Kelvin

    # ── Save arrays ───────────────────────────────────────────────────────
    print("\nSaving numerical outputs...")
    np.save(OUTPUT_DIR / "latent_means.npy",       mus)
    np.save(OUTPUT_DIR / "latent_logvars.npy",     logvars)
    np.save(OUTPUT_DIR / "reconstruction_l1.npy",  l1)
    np.save(OUTPUT_DIR / "reconstruction_l2.npy",  l2)

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_latent_maps(mus, coords, OUTPUT_DIR)
    plot_latent3_detailed(mus, coords, OUTPUT_DIR)
    plot_reconstruction_loss_map(l1, l2, coords, OUTPUT_DIR)
    plot_sample_profiles(profiles, recons_K_core, mus, coords, OUTPUT_DIR)
    plot_latent_distributions(mus, OUTPUT_DIR)
    plot_posterior_uncertainty(logvars, coords, OUTPUT_DIR)

    # ── Reference comparison ──────────────────────────────────────────────
    print("\nReference-data comparison:")
    compare_with_reference(mus, coords, OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n")
    write_summary(profiles, mus, logvars, l1, l2, c, OUTPUT_DIR)

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
