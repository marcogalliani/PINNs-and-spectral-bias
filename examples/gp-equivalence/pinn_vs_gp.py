"""A script to compare PINN and GPs
"""
from config import ROOT_DIR

import torch
import numpy as np

from PINNs import PINN, PINNConfig, TrainingConfig
from SIRENs import SIREN, SIRENConfig
from spectral_analysis import compute_ntk, condition_ntk_on_operator
from spectral_analysis import tanh_expectations, sin_expectations
from spectral_analysis import infinite_width_ntk, infinite_width_ntk_fourier, infinite_width_ntk_siren
from numerical_solvers import rk4_solve    

import matplotlib.pyplot as plt
import seaborn as sns

def sinusoidal_signal(t, amps, freqs, phases, sin_fn = np.sin):
    return sum(A * sin_fn(2 * np.pi * f * t + phi)
                for A,f,phi in zip(amps, freqs, phases))

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
    FIGURES_DIR = ROOT_DIR / "figures/pinn-gp-equivalence"
    
    #---PARAMETERS---
    # Forcing function: \sum_i(A_i sin(2\pi t f_i + phi_i))
    FREQS = [1.0, 2.0, 5.0, 10.0, 20.0]
    AMPS = [1.0, 1.0, 1.0, 1.0, 1.0]
    PHASES = [0, 0, 0, 0, 0]
    
    # Time horizon
    T_0 = 0.0; T_F = 2.0
    N_GRID = 500
    Y0 = 0.0
    t_grid = np.linspace(T_0, T_F, N_GRID, dtype=np.float32)
    
    # ODE: first-order, we just need the rhs to define it
    def ode_rhs(t, y, sin_fn=np.sin):
        return sinusoidal_signal(t, AMPS, FREQS, PHASES, sin_fn)
    
    y_rk4 = rk4_solve(ode_rhs, [0.0], t_grid, args=(np.sin,))
    plt.plot(t_grid, y_rk4)
    plt.savefig(FIGURES_DIR / "ref_sol.png")
    
    def ode_diff_op(f, t_grid):
        h = float(t_grid[1] - t_grid[0]) # assuming uniform grid
        return np.gradient(f, h) 
    
    #---NN ARCHITECTURES---
    DEPTH = 3 
    
    # ---INFINITE WIDTH KERNEL---
    Theta = infinite_width_ntk_siren(t_grid, DEPTH, sin_expectations)

    # Observations: residual = 0 at N_COL interior collocation points + IC at t = 0
    N_COL  = N_GRID - 2
    ix_f   = np.arange(1, N_GRID - 1)      # interior collocation indices (1 .. N_GRID-2)
    ic_idx = 0

    Gram = condition_ntk_on_operator(
        Theta, t_grid, ode_diff_op,
        ic_idx = ic_idx,
        residual_idx = ix_f,
        )

    # Right-hand side: residual L f = g = F(t) at the collocation points, then IC f(0) = y0
    targets = np.concatenate([sinusoidal_signal(t_grid[ix_f], AMPS, FREQS, PHASES), [Y0]])

    # cross-covariance of every eval point with the observations
    def apply_L(M, axis):
        return np.apply_along_axis(lambda f: ode_diff_op(f, t_grid), axis, M)

    # L acts on the observation argument (axis 1) for the residual block; IC is physics-blind
    Kvec = np.concatenate([apply_L(Theta, axis=1)[:, ix_f], Theta[:, ic_idx][:, None]], axis=1)
    sol = np.linalg.solve(Gram, targets)
    f_gp = Kvec @ sol                                           # posterior mean
    post_cov = Theta - Kvec @ np.linalg.solve(Gram, Kvec.T)
    f_gp_std = np.sqrt(np.clip(np.diag(post_cov), 0.0, None))

    print(f"GP mean vs true ODE solution: RMSE = {np.sqrt(np.mean((f_gp - y_rk4) ** 2)):.2e}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_grid, f_gp, color="crimson", lw=2.0,
            label=r"GP mean ($\Theta_\infty$ kernel regression)")
    ax.plot(t_grid, y_rk4, "k--", lw=1.5, label="true solution")

    ax.fill_between(t_grid, f_gp - 2 * f_gp_std, f_gp + 2 * f_gp_std,
                    color="crimson", alpha=0.2, label=r"GP $\pm 2\sigma$")
    ax.scatter(t_grid[ix_f], np.zeros(N_COL), marker="|", s=180, color="steelblue",
            label="collocation (residual = 0)", zorder=5)
    ax.scatter([0], [Y0], color="green", zorder=6, label="IC")
    ax.set(xlabel="t", ylabel="z(t)",
        title=r"Infinite-width PINN $=$ GP with kernel $\Theta_\infty$")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pinn_gp_predictor.png", dpi=150, bbox_inches="tight")
    
    # TODO: introduce comparison with trained PINN fit
    
if __name__ == "__main__":
    main()