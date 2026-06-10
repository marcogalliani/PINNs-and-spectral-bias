"""A script to compare PINNs and GPs (kernel-ridge / GP equivalence).

The infinite-width PINN predictor equals a GP whose kernel is the residual NTK
Θ_RR = L_t L_{t'} Θ_∞.  This script builds that kernel with the *autodiff*
infinite-width tools (`infinite_width_ntk_mlp_fn` + `condition_kernel_autodiff`):
the differential operator L is applied exactly by automatic differentiation,
avoiding the 1/h² finite-difference noise of the matrix-based
`condition_ntk_on_operator`.
"""
from config import ROOT_DIR

import torch
import numpy as np
from torch.func import grad, vmap

from PINNs import PINN, PINNConfig, TrainingConfig
from spectral_analysis import (
    infinite_width_ntk_mlp_fn,
    tanh_expectation_scalar,
    eval_kernel_matrix,
    condition_kernel_autodiff,
)
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
    FREQS = [1.0, 1.0, 1.0, 1.0, 1.0]
    AMPS = [1.0, 1.0, 1.0, 1.0, 1.0]
    PHASES = [0, 0, 0, 0, 0]

    # Time horizon
    T_0 = 0.0; T_F = 2.0
    N_GRID = 50
    Y0 = 0.0
    t_grid = np.linspace(T_0, T_F, N_GRID, dtype=np.float32)

    # ODE: first-order, we just need the rhs to define it
    def ode_rhs(t, y, sin_fn=np.sin):
        return sinusoidal_signal(t, AMPS, FREQS, PHASES, sin_fn)

    y_rk4 = rk4_solve(ode_rhs, [0.0], t_grid, args=(np.sin,))
    plt.plot(t_grid, y_rk4)
    plt.savefig(FIGURES_DIR / "ref_sol.png")

    #---NN ARCHITECTURES---
    DEPTH = 3

    # ---INFINITE-WIDTH KERNEL (autodiff)---
    # Differentiable scalar infinite-width NTK k(t, t') -> Θ_∞(t, t') of a tanh MLP.
    kernel_fn = infinite_width_ntk_mlp_fn(DEPTH, exp_scalar_fn=tanh_expectation_scalar)

    # Residual operator of the first-order ODE z' = F(t):  L = d/dt.
    # operator(g, s) applies L to a scalar function g at the point s (exact autodiff).
    def operator(g, s):
        return grad(g)(s)

    # One-sided operator: L on the *first* kernel argument, sampled on the grid,
    #   K_Lt[i, j] = L_t Θ_∞(t, t')|_{t=t_i, t'=t_j}.
    # (L on the second argument is its transpose, by symmetry of the kernel.)
    t_torch = torch.as_tensor(t_grid, dtype=torch.float64)
    def L_left(a, b):
        return operator(lambda x: kernel_fn(x, b), a)
    K_Lt = vmap(vmap(L_left, in_dims=(None, 0)), in_dims=(0, None))(t_torch, t_torch)
    K_Lt = K_Lt.detach().cpu().numpy()

    # Physics-blind prior Θ_∞ and residual–residual block K_RR = L_t L_{t'} Θ_∞.
    Theta = eval_kernel_matrix(kernel_fn, t_grid)
    K_RR  = condition_kernel_autodiff(kernel_fn, t_grid, operator)

    # Observations: residual = 0 at N_COL interior collocation points + IC at t = 0
    N_COL  = N_GRID - 2
    ix_f   = np.arange(1, N_GRID - 1)      # interior collocation indices (1 .. N_GRID-2)
    ic_idx = 0

    # Gram matrix of the PINN <-> GP equivalence:
    #   [[ K_RR        ,  L_t Θ(t, t_ic) ],
    #    [  (·)ᵀ        ,   Θ(t_ic, t_ic) ]]
    A = K_RR[np.ix_(ix_f, ix_f)]                     # residual–residual
    B = K_Lt[ix_f, ic_idx][:, None]                  # residual–IC (operator on residual side)
    C = np.array([[Theta[ic_idx, ic_idx]]])          # IC–IC (physics-blind)
    Gram = np.block([[A, B], [B.T, C]])
    Gram += 1e-8 * np.mean(np.diag(Gram)) * np.eye(N_COL + 1)  

    # Right-hand side: residual L f = g = F(t) at the collocation points, then IC f(0) = y0
    targets = np.concatenate([sinusoidal_signal(t_grid[ix_f], AMPS, FREQS, PHASES), [Y0]])

    # Cross-covariance of every eval point with the observations:
    #   residual side uses L on the observation argument (K_Lt.T), IC is physics-blind.
    Kvec = np.concatenate([K_Lt.T[:, ix_f], Theta[:, ic_idx][:, None]], axis=1)
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
