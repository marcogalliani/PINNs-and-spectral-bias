# Neural Tangent Kernel Visualisation
from config import ROOT_DIR

import torch
import numpy as np

from PINNs import PINN, PINNConfig, TrainingConfig
from SIRENs import SIREN, SIRENConfig
from spectral_analysis import compute_fft
from spectral_analysis import infinite_width_ntk_fn, empirical_ntk_fn
from spectral_analysis import eval_kernel_matrix, condition_kernel_autodiff
from numerical_solvers import rk4_solve    

import matplotlib.pyplot as plt
import seaborn as sns

def corr_from_kernel(K):
    d = np.sqrt(np.diag(K).clip(0)); d = np.where(d == 0, 1.0, d) 
    return K / np.outer(d, d)
    
def sinusoidal_signal(t, amps, freqs, phases, sin_fn = np.sin):
    return sum(A * sin_fn(2 * np.pi * f * t + phi)
                for A,f,phi in zip(amps, freqs, phases))

def dominant_freq(signal, sample_rate):
    """Dominant Fourier frequency of a signal."""
    frqs, spec = compute_fft(signal, sample_rate)
    return frqs[1:][np.argmax(spec[1:])]

def main():
    # ---SETUP---
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print(f"Device: {DEVICE}")
    
    FIGURES_DIR = ROOT_DIR / "figures/ntk"
            
    # ---PARAMETERS---
    # Forcing function: \sum_i(A_i sin(2\pi t f_i + phi_i))
    FREQS = [1.0, 2.0, 5.0, 10.0, 20.0]
    AMPS = [1.0, 1.0, 1.0, 1.0, 1.0]
    PHASES = [0, 0, 0, 0, 0]
    
    # Time horizon
    T_0 = 0.0; T_F = 2.0
    N_GRID = 500
    t_grid = np.linspace(T_0, T_F, N_GRID, dtype=np.float32)
    sample_rate = N_GRID / (T_F - T_0)
    
    # ODE: first-order, we just need the rhs to define it
    def ode_rhs(t, y, sin_fn=np.sin):
        return sinusoidal_signal(t, AMPS, FREQS, PHASES, sin_fn)

    y_rk4 = rk4_solve(ode_rhs, [0.0], t_grid, args=(np.sin,))
    plt.plot(t_grid, y_rk4)
    plt.savefig(FIGURES_DIR / "ref_sol.png")
    
    # ---NN ARCHITECTURES---
    WIDTH, DEPTH = 32, 3
    SIREN_OMEGA = 20
    FF_SIGMA = 10.0
    FF_N_FREQS = 16
    rng = np.random.default_rng()
    FOURIER_FREQS = rng.normal(loc=0.0, scale=FF_SIGMA, size=(FF_N_FREQS,))
    
    mlp     = PINN(PINNConfig(n_vars=1, width=WIDTH, depth=DEPTH)).to(DEVICE)
    fourier = PINN(PINNConfig(n_vars=1, width=WIDTH, depth=DEPTH, fourier_freqs=FOURIER_FREQS)).to(DEVICE)
    siren   = SIREN(SIRENConfig(n_vars=1, width=WIDTH, depth=DEPTH, omega_0=SIREN_OMEGA)).to(DEVICE)

    MODELS = {
        "MLP (Tanh)":  mlp,
        "Fourier MLP": fourier,
        "SIREN":       siren,
    }

    print(f"{'Architecture':<20} {'Params':>10}")
    print("-" * 32)
    for name, m in MODELS.items():
        P = sum(p.numel() for p in m.parameters())
        print(f"{name:<20} {P:>10,}")
        
    # ---EMPIRICAL NTK---
    print("Computing NTK matrices at initialisation...")
    
    ntk_matrices    = {}
    ntk_eigenvalues = {}
    ntk_eigenvectors = {}

    for name, model in MODELS.items():
        print(f"  {name}...", end=" ", flush=True)
        # empirical NTK as a differentiable kernel function k(t, t') -- same
        # interface as the infinite-width path (autodiff base + conditioning)
        kfn = empirical_ntk_fn(model, dtype=torch.float32)
        K = eval_kernel_matrix(kfn, t_grid, dtype=torch.float32)
        vals, vecs = np.linalg.eigh(K)
        ntk_matrices[name]     = K
        ntk_eigenvalues[name]  = vals[::-1].copy()
        ntk_eigenvectors[name] = vecs[:, ::-1].copy()
        eff_rank = int((vals.sum() ** 2) / (vals ** 2).sum())   # participation ratio
        print(f"λ_max={vals[-1]:.2e}  λ_min={vals[vals > 0].min():.2e}  eff_rank≈{eff_rank}")

        print(f"Condition on linear op (autodiff)...", end=" ", flush=True)
        K = condition_kernel_autodiff(kfn, t_grid, dtype=torch.float32)
        vals, vecs = np.linalg.eigh(K)
        ntk_matrices[name + " PINN"]     = K
        ntk_eigenvalues[name + " PINN"]  = vals[::-1].copy()
        ntk_eigenvectors[name + " PINN"] = vecs[:, ::-1].copy()
        eff_rank = int((vals.sum() ** 2) / (vals ** 2).sum())   # participation ratio
        print(f"λ_max={vals[-1]:.2e}  λ_min={vals[vals > 0].min():.2e}  eff_rank≈{eff_rank}")
        
    ARCH_NAMES = list(ntk_matrices.keys())
    PALETTE    = dict(zip(ARCH_NAMES, sns.color_palette("tab10", len(ARCH_NAMES))))
    
    # Heatmap visual (normalized)
    N = len(ARCH_NAMES)
    fig, axes = plt.subplots(1, N, figsize=(4.5 * N, 4.2))
    fig.suptitle("NTK Heatmaps (normalised)", fontsize=13)

    for ax, name in zip(axes, ARCH_NAMES):
        K     = ntk_matrices[name]
        K_vis = K / K.max()
        im = ax.imshow(
            K_vis, cmap="viridis", aspect="auto",
            extent=[t_grid[0], t_grid[-1], t_grid[-1], t_grid[0]],
            vmin=-1, vmax=1,
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("t")
        ax.set_ylabel("t'")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ntk_heatmaps.png", dpi=150, bbox_inches="tight")    

    # Kernel functions
    T_STAR = 1.0
    center_idx   = np.argmin(np.abs(t_grid - T_STAR))
    t_star_actual = float(t_grid[center_idx])

    fig, ax = plt.subplots(figsize=(10, 4))

    for name in ARCH_NAMES:
        corr_K     = corr_from_kernel(ntk_matrices[name])
        slice_ = corr_K[:, center_idx] 
        ax.plot(t_grid, slice_, label=name, color=PALETTE[name], linewidth=1.4)

    ax.axvline(t_star_actual, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$k(t,\, t^\star)\;/\; k(t^\star,\, t^\star)$")
    ax.set_title(f"Kernel slice at $t^\\star = {t_star_actual:.2f}$", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ntk_kernel_slice.png", dpi=150, bbox_inches="tight")
    
    # Kernel spectrum
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("NTK Eigenvalue Spectra", fontsize=13)

    idx = np.arange(1, len(t_grid) + 1)

    ax = axes[0]
    for name in ARCH_NAMES:
        vals = ntk_eigenvalues[name]
        ax.semilogy(idx, vals, label=name, color=PALETTE[name], linewidth=1.4)
    ax.set_xlabel("Eigenvalue index (descending)")
    ax.set_ylabel("Eigenvalue  (log scale)")
    ax.set_title("Spectrum")
    ax.legend(fontsize=9)

    ax = axes[1]
    for name in ARCH_NAMES:
        vals = ntk_eigenvalues[name]
        cum  = np.cumsum(vals) / vals.sum()
        ax.plot(idx, cum, label=name, color=PALETTE[name], linewidth=1.4)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.9, label="95 % trace")
    ax.set_xlabel("Eigenvalue index (descending)")
    ax.set_ylabel("Cumulative fraction of trace")
    ax.set_title("Spectral energy distribution  (effective rank)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ntk_spectra.png", dpi=150, bbox_inches="tight")
    
    # ---INFINITE-WIDTH KERNEL COMPUTATION---
    print("Computing infinite-width NTK ...", flush=True)
        
    ntk_inf_matrices    = {}
    ntk_inf_eigenvalues = {}
    ntk_inf_eigenvectors = {}
    
    
    INF_KIND = {"Fourier MLP": "fourier", "SIREN": "siren"}
    for name, model in MODELS.items():
        print(f"  {name}...", end=" ", flush=True)
        kind = INF_KIND.get(name, "mlp")
        kfn = infinite_width_ntk_fn(DEPTH, kind=kind,
                                    ff_freqs=FOURIER_FREQS if kind == "fourier" else None)
        K_inf_raw = eval_kernel_matrix(kfn, t_grid)
        # rescale so λ_max matches the empirical MLP (NTK scale is parametrisation-dependent)
        lam_inf_max = np.linalg.eigvalsh(K_inf_raw)[-1]
        scale = ntk_eigenvalues["MLP (Tanh)"][0] / lam_inf_max
        K_inf = K_inf_raw * scale
        vals, vecs = np.linalg.eigh(K_inf)
        # store
        ntk_inf_matrices[name] = K_inf
        ntk_inf_eigenvalues[name] = vals[::-1].copy()
        ntk_inf_eigenvectors[name] = vecs[:, ::-1].copy()

        eff_rank = int((vals.sum() ** 2) / (vals ** 2).sum())   # participation ratio
        print(f"λ_max={vals[-1]:.2e}  λ_min={vals[vals > 0].min():.2e}  eff_rank≈{eff_rank}")

        print(f"Condition on linear op (autodiff)...", end=" ", flush=True)
        # L_t L_{t'} applied exactly by autodiff -> PSD, no finite-difference noise tail
        K_inf = condition_kernel_autodiff(kfn, t_grid) * scale
        vals, vecs = np.linalg.eigh(K_inf)
        ntk_inf_matrices[name + " PINN"]     = K_inf
        ntk_inf_eigenvalues[name + " PINN"]  = vals[::-1].copy()
        ntk_inf_eigenvectors[name + " PINN"] = vecs[:, ::-1].copy()
        eff_rank = int((vals.sum() ** 2) / (vals ** 2).sum())   # participation ratio
        print(f"λ_max={vals[-1]:.2e}  λ_min={vals[vals > 0].min():.2e}  eff_rank≈{eff_rank}")
        
    # Heatmap visual (normalized)
    N = len(ARCH_NAMES)
    fig, axes = plt.subplots(1, N, figsize=(4.5 * N, 4.2))
    fig.suptitle("NTK Heatmaps (normalised)", fontsize=13)

    for ax, name in zip(axes, ARCH_NAMES):
        K     = ntk_inf_matrices[name]
        K_vis = K / K.max()
        im = ax.imshow(
            K_vis, cmap="viridis", aspect="auto",
            extent=[t_grid[0], t_grid[-1], t_grid[-1], t_grid[0]],
            vmin=-1, vmax=1,
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("t")
        ax.set_ylabel("t'")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "inf_ntk_heatmaps.png", dpi=150, bbox_inches="tight")    

    # Kernel functions
    T_STAR = 1.0
    center_idx   = np.argmin(np.abs(t_grid - T_STAR))
    t_star_actual = float(t_grid[center_idx])

    fig, ax = plt.subplots(figsize=(10, 4))

    for name in ARCH_NAMES:
        corr_K     = corr_from_kernel(ntk_inf_matrices[name])
        slice_ = corr_K[:, center_idx] 
        ax.plot(t_grid, slice_, label=name, color=PALETTE[name], linewidth=1.4)

    ax.axvline(t_star_actual, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$k(t,\, t^\star)\;/\; k(t^\star,\, t^\star)$")
    ax.set_title(f"Kernel slice at $t^\\star = {t_star_actual:.2f}$", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "inf_ntk_kernel_slice.png", dpi=150, bbox_inches="tight")
    
    # Kernel spectrum: eigenvalue vs dominant frequency of the eigenvector
    fig, ax = plt.subplots(figsize=(10,4))
    fig.suptitle("NTK Eigenvalue Spectra", fontsize=13)

    idx = np.arange(1, len(t_grid) + 1)

    for name in ARCH_NAMES:
        vals  = ntk_inf_eigenvalues[name]
        freqs = np.array(
            [dominant_freq(ntk_inf_eigenvectors[name][:, k],
                           sample_rate) for k in range(ntk_inf_eigenvectors[name].shape[1])])
        pos   = vals > 0   # drop the numerical noise tail (non-PSD float artifacts)
        vals, freqs = vals[pos], freqs[pos]
        sort_indices = np.argsort(freqs)
        ax.plot(freqs[sort_indices], vals[sort_indices], label=name, color=PALETTE[name], alpha=0.7)
    for f in FREQS:
        ax.axvline(f, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlim(0, max(FREQS) * 1.5)
    ax.set_xlabel("Dominant frequency of eigenvector [Hz]")
    ax.set_ylabel("Eigenvalue  (log scale)")
    ax.set_title("Spectrum  (small λ ↔ slow convergence)")
    ax.legend(fontsize=9)


    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "inf_ntk_spectra.png", dpi=150, bbox_inches="tight")
    
    
if __name__ == "__main__":
    main()    