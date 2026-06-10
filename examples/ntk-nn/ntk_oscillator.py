# Neural Tangent Kernel Visualisation for a SYSTEM of linear ODEs.
#
# Test system: the harmonic oscillator  x'' = -w0^2 x,  written first-order as
#   z = [x, v],   z' = A z,   A = [[0, 1], [-w0^2, 0]]   (eigenvalues +- i w0).
#
# This exercises the multi-output (block) NTK path: for a d-variable system the
# empirical NTK is a (d, d) block per time pair, assembled into an (N*d, N*d)
# Gram.  The PINN residual operator is matrix-valued, L = (d/dt) I - A, and the
# eigenmodes are vector-valued: each carries a temporal frequency AND a
# direction (polarisation) in state space (cf. mode_dominant_freq/mode_decompose).
from config import ROOT_DIR

import numpy as np
import torch

from PINNs import PINN, PINNConfig
from SIRENs import SIREN, SIRENConfig
from spectral_analysis import (
    empirical_ntk_block_fn,
    eval_kernel_matrix,
    condition_kernel_block_autodiff,
    mode_dominant_freq,
    mode_decompose,
)
from numerical_solvers import rk4_solve

import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # ---SETUP---
    torch.manual_seed(42)
    np.random.seed(42)

    FIGURES_DIR = ROOT_DIR / "figures/ntk_oscillator"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---SYSTEM: harmonic oscillator z' = A z---
    F0 = 2.0                       # natural frequency [Hz]
    W0 = 2 * np.pi * F0
    A = np.array([[0.0, W0], [-W0, 0.0]])     # z = [x, v]
    D = 2                          # number of state variables

    # Time horizon / grid (the kernel grid -- kept modest: the Gram is N*D square)
    T_0, T_F = 0.0, 2.0
    N_GRID = 120
    t_grid = np.linspace(T_0, T_F, N_GRID, dtype=np.float64)
    sample_rate = N_GRID / (T_F - T_0)

    # Reference solution (finer grid, just for the plot)
    t_plot = np.linspace(T_0, T_F, 400)
    z_ref = rk4_solve(lambda t, z: A @ np.asarray(z), [1.0, 0.0], t_plot)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
    axL.plot(t_plot, z_ref[:, 0], label="x(t)")
    axL.plot(t_plot, z_ref[:, 1] / W0, label="v(t) / w0", alpha=0.8)
    axL.set_xlabel("t"); axL.set_title(f"Oscillator solution (f0 = {F0:.1f} Hz)")
    axL.legend(fontsize=9)
    axR.plot(z_ref[:, 0], z_ref[:, 1])
    axR.set_xlabel("x"); axR.set_ylabel("v"); axR.set_title("Phase portrait")
    axR.set_aspect("auto")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ref_sol.png", dpi=150, bbox_inches="tight")

    # ---NN ARCHITECTURES (n_vars = D)---
    WIDTH, DEPTH = 24, 3
    SIREN_OMEGA = 20
    FF_SIGMA, FF_N_FREQS = 10.0, 16
    FOURIER_FREQS = np.random.default_rng(0).normal(0.0, FF_SIGMA, size=FF_N_FREQS)

    MODELS = {
        "MLP (Tanh)":  PINN(PINNConfig(n_vars=D, width=WIDTH, depth=DEPTH)),
        "Fourier MLP": PINN(PINNConfig(n_vars=D, width=WIDTH, depth=DEPTH,
                                       fourier_freqs=FOURIER_FREQS)),
        "SIREN":       SIREN(SIRENConfig(n_vars=D, width=WIDTH, depth=DEPTH,
                                         omega_0=SIREN_OMEGA)),
    }

    # ---BLOCK NTK: base + PINN-conditioned on L = d/dt I - A---
    print("Computing block NTK matrices at initialisation...")
    results = {}
    for name, model in MODELS.items():
        print(f"  {name}...", end=" ", flush=True)
        kfn = empirical_ntk_block_fn(model)              # k(a,b) -> (D, D) block

        K_base = eval_kernel_matrix(kfn, t_grid)         # (N*D, N*D) physics-blind
        # PINN NTK of the *coupled* system: L mixes the state components.
        K_pinn = condition_kernel_block_autodiff(kfn, t_grid, A=A)

        vals, vecs = np.linalg.eigh(K_pinn)
        vals, vecs = vals[::-1].copy(), vecs[:, ::-1].copy()
        eff_rank = int((vals.sum() ** 2) / (vals ** 2).sum())
        results[name] = dict(base=K_base, pinn=K_pinn, vals=vals, vecs=vecs)
        print(f"Gram {K_pinn.shape} | lam_max={vals[0]:.2e} | eff_rank~{eff_rank}")

    ARCH_NAMES = list(MODELS.keys())
    PALETTE = dict(zip(ARCH_NAMES, sns.color_palette("tab10", len(ARCH_NAMES))))

    # ---HEATMAPS of the conditioned (N*D, N*D) block Gram---
    # The 2x2 macro-block structure [[xx, xv], [vx, vv]] is the output coupling.
    fig, axes = plt.subplots(1, len(ARCH_NAMES), figsize=(4.6 * len(ARCH_NAMES), 4.2))
    fig.suptitle("PINN block-NTK Gram (normalised)  --  oscillator system", fontsize=13)
    for ax, name in zip(axes, ARCH_NAMES):
        K = results[name]["pinn"]
        K = K.reshape(N_GRID, D, N_GRID, D).transpose(1, 0, 3, 2).reshape(D * N_GRID, D * N_GRID)
        im = ax.imshow(K / np.abs(K).max(), cmap="viridis", aspect='auto', vmin=-1, vmax=1)
        # delineate the D x D macro-blocks (each N x N)
        for k in range(1, D):
            ax.axhline(k * N_GRID - 0.5, color="k", lw=0.6)
            ax.axvline(k * N_GRID - 0.5, color="k", lw=0.6)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "block_ntk_heatmaps.png", dpi=150, bbox_inches="tight")

    # ---SPECTRUM vs dominant frequency of the (vector-valued) eigenmode---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in ARCH_NAMES:
        vals, vecs = results[name]["vals"], results[name]["vecs"]
        pos = vals > 0                                   # drop numerical-noise tail
        freqs = np.array([mode_dominant_freq(vecs[:, k], sample_rate, n_vars=D)
                          for k in np.where(pos)[0]])
        order = np.argsort(freqs)
        ax.plot(freqs[order], vals[pos][order], label=name,
                color=PALETTE[name], alpha=0.8)
    ax.axvline(F0, color="black", ls="--", lw=1.0, label=f"system f0 = {F0:.1f} Hz")
    ax.set_yscale("log")
    ax.set_xlim(0, sample_rate / 2)
    ax.set_xlabel("Dominant frequency of eigenmode [Hz]  (Option A: summed power)")
    ax.set_ylabel("Eigenvalue (log)")
    ax.set_title("PINN block-NTK spectrum  (small lambda <-> slow convergence)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "block_ntk_spectrum.png", dpi=150, bbox_inches="tight")

    # ---POLARISATION: top modes carry a direction in state space---
    # mode_decompose splits each eigenmode into (frequency, direction, separability).
    N_TOP = 6
    fig, axes = plt.subplots(1, len(ARCH_NAMES), figsize=(4.6 * len(ARCH_NAMES), 4.4),
                             subplot_kw=dict(aspect="equal"))
    fig.suptitle("Top eigenmode polarisations in state space (x, v)", fontsize=13)
    print("\nTop-mode (frequency, polarisation, separability):")
    for ax, name in zip(axes, ARCH_NAMES):
        vecs, vals = results[name]["vecs"], results[name]["vals"]
        print(f"  {name}:")
        for k in range(N_TOP):
            f, w, sep = mode_decompose(vecs[:, k], sample_rate, n_vars=D)
            ax.annotate("", xy=(w[0], w[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=plt.cm.viridis(k / N_TOP)))
            ax.text(w[0] * 1.08, w[1] * 1.08, f"{f:.1f}Hz", fontsize=7)
            print(f"    mode {k}: f={f:5.2f} Hz  dir=({w[0]:+.2f},{w[1]:+.2f})  sep={sep:.3f}")
        lim = 1.2
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
        ax.set_title(name, fontsize=10); ax.set_xlabel("x dir"); ax.set_ylabel("v dir")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "block_ntk_polarisation.png", dpi=150, bbox_inches="tight")

    print(f"\nFigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
