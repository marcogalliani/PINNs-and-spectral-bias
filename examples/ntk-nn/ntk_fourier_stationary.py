"""Fourier-feature NTK is stationary and its kernel is a Fourier expansion.

Companion to `examples/gp-equivalence/nn_vs_krr.py` (MLP learning a sinusoidal
signal).  Here we look only at the *kernel* of the Fourier-feature network and
establish two facts about its infinite-width NTK Theta(a, b):

  (1) STATIONARITY.  With the random Fourier feature map
          gamma(t) = [ sin(2 pi f_j t), cos(2 pi f_j t) ]_j          (no linear t)
      the first-layer covariance is
          Sigma(a, b) = sigma_w^2 <gamma(a), gamma(b)> + sigma_b^2
                      = sigma_w^2 sum_j cos(2 pi f_j (a - b)) + sigma_b^2,
      a function of tau = a - b ONLY.  The diagonal Sigma(a, a) is then constant
      in a, so the whole NTK recursion stays a function of tau and
          Theta(a, b) = k(a - b)        (shift-invariant / Toeplitz).
      Numerically: the sampled Gram is Toeplitz, and rows re-centred at tau = 0
      collapse onto a single profile k(tau).

  (2) FOURIER-BASIS EXPANSION.  The stationary profile k(tau) is built from the
      cosines cos(2 pi f_j tau): its power spectrum is supported on the feature
      frequencies {|f_j|}.  For a single layer this is *exact* (k is literally
      sum_j cos(2 pi f_j tau)); deeper tanh layers keep stationarity but the
      nonlinearity mixes the base tones into harmonics / sum-frequencies.

The appended linear feature `t` used in `infinite_width_ntk_fourier_fn`
(the `a*b` term) is what breaks exact stationarity -- shown in the comparison
panels as a non-Toeplitz Gram and a growing diagonal spread.
"""
from config import ROOT_DIR

import math
import numpy as np
import torch

from spectral_analysis import (
    tanh_expectation_scalar,
    eval_kernel_matrix,
    compute_fft,
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Fourier-feature infinite-width NTK as a differentiable scalar kernel.
#
# Mirrors `infinite_width_ntk_fourier_fn` in spectral_analysis.py, but exposes a
# switch for the appended linear feature `t`.  With `include_linear=False` the
# feature map is the pure random-Fourier map and the kernel is exactly
# stationary; `include_linear=True` reproduces the kernel used in nn_vs_krr.
# ---------------------------------------------------------------------------
def fourier_ntk_fn(depth, ff_freqs, *, include_linear=False,
                   exp_scalar_fn=tanh_expectation_scalar,
                   sigma_w2=0.5, sigma_b2=0.05):
    freqs = torch.as_tensor(np.asarray(ff_freqs, dtype=float))

    def sigma1(a, b):
        f = freqs.to(a.dtype)
        # <gamma(a), gamma(b)> = sum_j cos(2 pi f_j (a - b))  [+ a b if linear]
        feat = torch.cos(2 * math.pi * f * (a - b)).sum()
        if include_linear:
            feat = feat + a * b
        return sigma_w2 * feat + sigma_b2

    def kernel(a, b):
        saa, sbb, sab = sigma1(a, a), sigma1(b, b), sigma1(a, b)
        tab = sab
        for _ in range(depth - 1):
            Ess, Edd = exp_scalar_fn(saa, sbb, sab)
            Eaa, _   = exp_scalar_fn(saa, saa, saa)
            Ebb, _   = exp_scalar_fn(sbb, sbb, sbb)
            sab = sigma_w2 * Ess + sigma_b2
            saa = sigma_w2 * Eaa + sigma_b2
            sbb = sigma_w2 * Ebb + sigma_b2
            tab = sab + tab * (sigma_w2 * Edd)
        return tab

    return kernel


def toeplitz_diag_spread(K):
    """For each offset m, std and |mean| of the m-th diagonal of K.

    A stationary (shift-invariant) kernel is Toeplitz: K[i, j] depends only on
    i - j, so every diagonal is constant and `std` is ~0 (machine precision).
    """
    N = K.shape[0]
    offsets = np.arange(N)
    std = np.empty(N)
    mean = np.empty(N)
    for m in offsets:
        d = np.diagonal(K, offset=m)
        std[m] = d.std()
        mean[m] = d.mean()
    return offsets, mean, std


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    FIGURES_DIR = ROOT_DIR / "figures/ntk"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---PARAMETERS--- (same signal/grid setup as nn_vs_krr.py) ---
    T_0, T_F = 0.0, 2.0
    N_GRID = 500
    # float64 grid: the high feature frequencies amplify float32 quantization of
    # t[i]-t[j], which would otherwise blur the Toeplitz structure (~1e-3).
    t_grid = np.linspace(T_0, T_F, N_GRID, dtype=np.float64)
    dt = (T_F - T_0) / (N_GRID - 1)
    sample_rate = 1.0 / dt

    DEPTH = 3

    # Random Fourier-feature frequencies (as in nn_vs_krr.py)
    FF_SIGMA = 10.0
    FF_N_FREQS = 16
    rng = np.random.default_rng(0)
    FOURIER_FREQS = rng.normal(loc=0.0, scale=FF_SIGMA, size=(FF_N_FREQS,))
    feat_freqs = np.sort(np.abs(FOURIER_FREQS))

    print(f"Fourier feature frequencies |f_j| (Hz):")
    print("  " + ", ".join(f"{f:.2f}" for f in feat_freqs))

    # --- Kernels: pure RFF (stationary) vs with appended linear feature ---
    print("\nBuilding infinite-width NTK Gram matrices on the grid ...")
    K_pure = eval_kernel_matrix(
        fourier_ntk_fn(DEPTH, FOURIER_FREQS, include_linear=False), t_grid)
    K_lin = eval_kernel_matrix(
        fourier_ntk_fn(DEPTH, FOURIER_FREQS, include_linear=True), t_grid)
    # Single layer: NTK == first-layer covariance == EXACT cosine expansion
    K_pure_d1 = eval_kernel_matrix(
        fourier_ntk_fn(1, FOURIER_FREQS, include_linear=False), t_grid)

    # --- (1) Stationarity: Toeplitz check ---
    off, mean_p, std_p = toeplitz_diag_spread(K_pure)
    _,   mean_l, std_l = toeplitz_diag_spread(K_lin)

    # relative spread over diagonals with a meaningful mean
    rel_p = np.max(std_p[np.abs(mean_p) > 1e-9] /
                   np.abs(mean_p[np.abs(mean_p) > 1e-9]))
    rel_l = np.max(std_l[np.abs(mean_l) > 1e-9] /
                   np.abs(mean_l[np.abs(mean_l) > 1e-9]))
    print("\nStationarity (max relative spread of a Gram diagonal; 0 == Toeplitz):")
    print(f"  pure Fourier features   : {rel_p:.2e}   -> stationary")
    print(f"  with linear feature `t` : {rel_l:.2e}   -> NOT stationary")

    # --- (2) Stationary profile k(tau) and its spectrum ---
    # A row of a Toeplitz kernel anchored at the centre IS the profile k(tau).
    mid = N_GRID // 2
    tau = (np.arange(N_GRID) - mid) * dt
    k_profile = K_pure[mid]               # depth-DEPTH profile
    k_profile_d1 = K_pure_d1[mid]         # single-layer profile (exact cosines)

    # Power spectrum of the (real, even) profile -> peaks at the feature freqs.
    f_spec, spec_d1 = compute_fft(k_profile_d1, sample_rate)
    _,      spec_dn = compute_fft(k_profile, sample_rate)

    # =====================  PLOTS  =====================
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) Gram heatmap -- pure RFF: banded / Toeplitz
    im0 = axes[0, 0].imshow(K_pure, origin="lower",
                            extent=[T_0, T_F, T_0, T_F], aspect="auto",
                            cmap="viridis")
    axes[0, 0].set(title=f"Pure Fourier-feature NTK  (depth {DEPTH})\n"
                         "Gram $\\Theta(t,t')$ is Toeplitz (stationary)",
                   xlabel="$t'$", ylabel="$t$")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # (0,1) Gram heatmap -- with linear feature: not Toeplitz
    im1 = axes[0, 1].imshow(K_lin, origin="lower",
                            extent=[T_0, T_F, T_0, T_F], aspect="auto",
                            cmap="viridis")
    axes[0, 1].set(title="With appended linear feature $t$\n"
                         "$a\\,b$ term breaks shift-invariance",
                   xlabel="$t'$", ylabel="$t$")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # (0,2) per-diagonal spread: 0 <=> Toeplitz <=> stationary
    axes[0, 2].semilogy(off * dt, std_p + 1e-18, color="C0",
                        label=f"pure RFF (max rel {rel_p:.0e})")
    axes[0, 2].semilogy(off * dt, std_l + 1e-18, color="C3",
                        label=f"with linear $t$ (max rel {rel_l:.0e})")
    axes[0, 2].set(title="Spread within each Gram diagonal\n"
                         "(std over a constant-$\\tau$ band)",
                   xlabel=r"offset $\tau$", ylabel="std along diagonal")
    axes[0, 2].legend(fontsize=8)

    # (1,0) shift-invariance: rows re-centred at tau=0 collapse onto k(tau)
    anchors = [80, 180, 250, 320, 420]
    for i in anchors:
        taus = (np.arange(N_GRID) - i) * dt
        axes[1, 0].plot(taus, K_pure[i], lw=1.2, alpha=0.8,
                        label=f"$t={t_grid[i]:.2f}$")
    axes[1, 0].set(title="Rows $\\Theta(t_i,\\cdot)$ re-centred at $\\tau=0$\n"
                         "collapse onto a single profile $k(\\tau)$",
                   xlabel=r"$\tau = t' - t_i$", ylabel=r"$\Theta$")
    axes[1, 0].set_xlim(-0.6, 0.6)
    axes[1, 0].legend(fontsize=7, ncol=2)

    # (1,1) the stationary profile k(tau)
    axes[1, 1].plot(tau, k_profile, color="C0", lw=1.5,
                    label=f"depth {DEPTH} (tanh)")
    axes[1, 1].plot(tau, k_profile_d1, color="C1", lw=1.2, alpha=0.8,
                    label="depth 1 (exact $\\sum_j\\cos 2\\pi f_j\\tau$)")
    axes[1, 1].set(title="Stationary kernel profile $k(\\tau)$",
                   xlabel=r"$\tau$", ylabel=r"$k(\tau)$")
    axes[1, 1].set_xlim(-0.6, 0.6)
    axes[1, 1].legend(fontsize=8)

    # (1,2) power spectrum of k(tau): support on the feature frequencies
    f_max = feat_freqs.max() * 1.3
    axes[1, 2].plot(f_spec, spec_d1 / spec_d1.max(), color="C1", lw=1.3,
                    label="depth 1 (exact)")
    axes[1, 2].plot(f_spec, spec_dn / spec_dn.max(), color="C0", lw=1.3,
                    alpha=0.8, label=f"depth {DEPTH} (+ harmonics)")
    for f in feat_freqs:
        axes[1, 2].axvline(f, color="k", ls=":", lw=0.8, alpha=0.5)
    axes[1, 2].axvline(feat_freqs[0], color="k", ls=":", lw=0.8, alpha=0.5,
                       label="feature freqs $|f_j|$")
    axes[1, 2].set(title="Spectrum of $k(\\tau)$: peaks at the feature freqs",
                   xlabel="frequency (Hz)", ylabel="normalised power")
    axes[1, 2].set_xlim(0, f_max)
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("Fourier-feature NTK: stationary kernel built from the "
                 "feature frequencies", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGURES_DIR / "fourier_ntk_stationarity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure -> {out}")


if __name__ == "__main__":
    main()
