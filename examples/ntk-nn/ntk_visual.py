# Neural Tangent Kernel Visualisation
from config import ROOT_DIR

import torch
import numpy as np

from PINNs import PINN, PINNConfig, TrainingConfig
from SIRENs import SIREN, SIRENConfig
from spectral_analysis import compute_ntk, condition_ntk_on_operator
from numerical_solvers import rk4_solve    

import matplotlib.pyplot as plt
import seaborn as sns

def corr_from_kernel(K):
    d = np.sqrt(np.diag(K).clip(0)); d = np.where(d == 0, 1.0, d) 
    return K / np.outer(d, d)
    
def sinusoidal_signal(t, amps, freqs, phases, sin_fn = np.sin):
    return sum(A * sin_fn(2 * np.pi * f * t + phi)
                for A,f,phi in zip(amps, freqs, phases))

# Utilities to compute infinite-width kernels
# Expectations, computed using Gaussian-Hermite quadrature
# E[f(u)f(v)] where (u,v) ~ N(0,Cov)
#
# Reparametrisation trick: (u,v) are reparametrised in terms of 
# independent variables (x,y)
# u = sigma_u * x
# v = sigma_v (rho * x + sqrt(1 - rho^2) * y)

# tanh activation function
def tanh_expectations(Cov, n_gh_pts = 20):
    """
    E[tanh(u)tanh(v)] and 
    E[tanh'(u)tanh'(v)] 
    for (u,v)~N(0,Λ), elementwise over a grid.
    
    Parameters:
    -----------
    - Cov: np.ndarray
        covariance matrix of preactivations, preactivations are assumed
        centred in 0
    """
    var = np.diag(Cov)
    # convert to correlation matrix
    sd = np.sqrt(np.clip(var, 1e-30, None))
    rho = np.clip(Cov / np.outer(sd, sd), -1.0, 1.0)
    # pearson coefficients
    comp = np.sqrt(np.clip(1.0 - rho ** 2, 0.0, None))
    Ess = np.zeros_like(Cov) # init E[tanh(u)tanh(v)]
    Edd = np.zeros_like(Cov) # init E[tanh'(u)tanh'(v)]
    # Gaussian-Hermite quadrature: points and weights
    ghx, ghw = np.polynomial.hermite.hermgauss(n_gh_pts)
    for p in range(n_gh_pts):
        u = sd[:, None] * ghx[p]; 
        tu = np.tanh(u); du = 1.0 - tu ** 2 # activ. and derivative
        for q in range(n_gh_pts):
            v = sd[None, :] * (rho * ghx[p] + comp * ghx[q]); 
            w = ghw[p] * ghw[q]
            Ess += w * tu * np.tanh(v)
            Edd += w * du * (1.0 - np.tanh(v) ** 2)
    return (Ess + Ess.T) / 2.0, (Edd + Edd.T) / 2.0

# sin activation function (SIREN)
# here the analytical solution is used (no GH quadrature needed)
def sin_expectations(Cov):
    """
    Expectations:
    E[sin u sin v]=e^{-(a+b)/2}sinh(c) and
    E[sin'u sin'v]=E[cos u cos v]=e^{-(a+b)/2}cosh(c)
    
    where a = Var(u), b = Var(v), c = Cov(u,v)
    
    Parameters:
    -----------
    - Cov: np.ndarray
        covariance matrix of preactivations, preactivations are assumed
        centred in 0
    """
    var = np.diag(Cov)
    
    a = var[:, None]; b = var[None, :]; c = Cov
    half = (a + b) / 2.0
    Ess = 0.5 * (np.exp(c - half) - np.exp(-c - half))
    Ecc = 0.5 * (np.exp(c - half) + np.exp(-c - half))
    return Ess, Ecc

# TODO: add variant for SIREN, it is an intrisecally different architecture
def infinite_width_ntk(t_grid, depth, expectation_fn, sigma_w2=1.0, sigma_b2=0.2):
    """Deterministic NTK of an infinitely wide MLP
    
    Parameters:
    ----------
    - t: 
        time grid
    - depth: float
        depth of the MLP
    - expectation_fn: 
        function returning E[f(u)f(v)] and E[f'(u)f'(v)] for a specific
        activation function f
    - sigma_w2
    - sigma_b2
    """
    X = np.asarray(t_grid, dtype=float).reshape(-1, 1)
    Sigma = sigma_w2 * (X @ X.T) + sigma_b2          # Σ^(1)
    Theta = Sigma.copy()                              # Θ^(1)
    for _ in range(depth - 1):
        Ess, Edd = expectation_fn(Sigma)
        Sigma = sigma_w2 * Ess + sigma_b2            # Σ^(l+1)
        Theta = Sigma + Theta * (sigma_w2 * Edd)     # Θ^(l+1)
    return Theta


def infinite_width_ntk_fourier(t_grid, depth, expectation_fn, ff_freqs, sigma_w2=0.5, sigma_b2=0.05):
    """Deterministic NTK of an infinitely wide MLP
    
    Parameters:
    ----------
    - t: 
        time grid
    - depth: float
        depth of the MLP
    - expectation_fn: 
        function returning E[f(u)f(v)] and E[f'(u)f'(v)] for a specific
        activation function f
    - ff_freqs
    - sigma_w2
    - sigma_b2
    """
    tt = np.asarray(t_grid, dtype=float)
    ff_embeds = [np.sin(2*np.pi*f*tt) for f in ff_freqs] + [np.cos(2*np.pi*f*tt) for f in ff_freqs] + [tt]
    Gram = np.stack(ff_embeds, 1)
    # Covariance of the mapped features
    Sigma = sigma_w2 * (Gram @ Gram.T) + sigma_b2          # Σ^(1)
    # Same as standard infinite width kernel
    Theta = Sigma.copy()                              # Θ^(1)
    for _ in range(depth - 1):
        Ess, Edd = expectation_fn(Sigma)
        Sigma = sigma_w2 * Ess + sigma_b2            # Σ^(l+1)
        Theta = Sigma + Theta * (sigma_w2 * Edd)     # Θ^(l+1)
    return Theta

def infinite_width_ntk_siren(t_grid, depth, expectation_fn, sigma_w2=600.0, sigma_b2=300.0, gw_hidden = 2.0):
    """Deterministic NTK of an infinitely wide MLP
    
    Parameters:
    ----------
    - t: 
        time grid
    - depth: float
        depth of the MLP
    - expectation_fn: 
        function returning E[f(u)f(v)] and E[f'(u)f'(v)] for a specific
        activation function f
    - sigma_w2
    - sigma_b2
    - gw_hidden
    """
    X = np.asarray(t_grid, dtype=float).reshape(-1, 1)
    Sigma = sigma_w2 * (X @ X.T) + sigma_b2 
    Theta = Sigma.copy()                              # Θ^(1)
    for _ in range(depth - 2): #last layer is linear
        Ess, Edd = expectation_fn(Sigma)
        Sigma = gw_hidden * Ess
        Theta = Sigma + Theta * (gw_hidden * Edd)     # Θ^(l+1)
    Ess, _ = expectation_fn(Sigma)
    return Ess + Theta

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
    
    # ODE: first-order, we just need the rhs to define it
    def ode_rhs(t, y, sin_fn=np.sin):
        return sinusoidal_signal(t, AMPS, FREQS, PHASES, sin_fn)

    # torch-valued rhs for the autodiff PINN NTK (PINN.ntk / SIREN.ntk)
    def ode_rhs_torch(t, z):
        return sinusoidal_signal(t, AMPS, FREQS, PHASES, torch.sin)

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
    
    def ode_diff_op(f,t_grid):
        h = float(t_grid[1] - t_grid[0]) # assuming uniform grid
        return np.gradient(f, h) 

    for name, model in MODELS.items():
        print(f"  {name}...", end=" ", flush=True)
        K = compute_ntk(model, t_grid, device=DEVICE)
        vals, vecs = np.linalg.eigh(K)
        ntk_matrices[name]     = K
        ntk_eigenvalues[name]  = vals[::-1].copy()
        ntk_eigenvectors[name] = vecs[:, ::-1].copy()
        eff_rank = int((vals.sum() ** 2) / (vals ** 2).sum())   # participation ratio
        print(f"λ_max={vals[-1]:.2e}  λ_min={vals[vals > 0].min():.2e}  eff_rank≈{eff_rank}")
        
        print(f"Condition on linear op...", end=" ", flush=True)
        K = condition_ntk_on_operator(K, t_grid, ode_diff_op)
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
    
    
    for name, model in MODELS.items():
        print(f"  {name}...", end=" ", flush=True)
        if name == "Fourier MLP":
            K_inf_raw = infinite_width_ntk_fourier(t_grid, DEPTH, tanh_expectations, FOURIER_FREQS)
        elif name == "SIREN":
            K_inf_raw = infinite_width_ntk_siren(t_grid, DEPTH, sin_expectations)
        else:
            K_inf_raw = infinite_width_ntk(t_grid, DEPTH, tanh_expectations)
        # rescale so λ_max matches the empirical MLP (NTK scale is parametrisation-dependent)
        lam_inf_max = np.linalg.eigvalsh(K_inf_raw)[-1]
        K_inf = K_inf_raw * (ntk_eigenvalues["MLP (Tanh)"][0] / lam_inf_max)
        vals, vecs = np.linalg.eigh(K_inf)
        # store
        ntk_inf_matrices[name] = K_inf
        ntk_inf_eigenvalues[name] = vals[::-1].copy()
        ntk_inf_eigenvectors[name] = vecs[:, ::-1].copy()
        
        eff_rank = int((vals.sum() ** 2) / (vals ** 2).sum())   # participation ratio
        print(f"λ_max={vals[-1]:.2e}  λ_min={vals[vals > 0].min():.2e}  eff_rank≈{eff_rank}")
        
        print(f"Condition on linear op...", end=" ", flush=True)
        K_inf = condition_ntk_on_operator(K_inf, t_grid, ode_diff_op)
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
    
    # Kernel spectrum
    # TODO: modify to plot eigenvalues vs frequency
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("NTK Eigenvalue Spectra", fontsize=13)

    idx = np.arange(1, len(t_grid) + 1)

    ax = axes[0]
    for name in ARCH_NAMES:
        vals = ntk_inf_eigenvalues[name]
        ax.semilogy(idx, vals, label=name, color=PALETTE[name], linewidth=1.4)
    ax.set_xlabel("Eigenvalue index (descending)")
    ax.set_ylabel("Eigenvalue  (log scale)")
    ax.set_title("Spectrum")
    ax.legend(fontsize=9)

    ax = axes[1]
    for name in ARCH_NAMES:
        vals = ntk_inf_eigenvalues[name]
        cum  = np.cumsum(vals) / vals.sum()
        ax.plot(idx, cum, label=name, color=PALETTE[name], linewidth=1.4)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.9, label="95 % trace")
    ax.set_xlabel("Eigenvalue index (descending)")
    ax.set_ylabel("Cumulative fraction of trace")
    ax.set_title("Spectral energy distribution  (effective rank)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "inf_ntk_spectra.png", dpi=150, bbox_inches="tight")
    
    
if __name__ == "__main__":
    main()    