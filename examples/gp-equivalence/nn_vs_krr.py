"""A script to compare MLP and KRR
"""
from config import ROOT_DIR

import copy
import torch
import numpy as np

from PINNs import PINN, PINNConfig, TrainingConfig
from SIRENs import SIREN, SIRENConfig
from spectral_analysis import eval_kernel_matrix
from spectral_analysis import infinite_width_ntk_mlp_fn, infinite_width_ntk_siren_fn, infinite_width_ntk_fourier_fn
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
    # Target signal: \sum_i(A_i sin(2\pi t f_i + phi_i))
    FREQS = [1.0, 2.0, 5.0, 10.0, 20.0]
    AMPS = [1.0, 1.0, 1.0, 1.0, 1.0]
    PHASES = [0, 0, 0, 0, 0]

    # Time horizon
    T_0 = 0.0; T_F = 2.0
    N_GRID = 500
    t_grid = np.linspace(T_0, T_F, N_GRID, dtype=np.float32)

    # Function to fit (plain regression, no ODE)
    y_true = sinusoidal_signal(t_grid, AMPS, FREQS, PHASES).astype(np.float32)

    # Training observations: a subsample of the grid
    N_TRAIN = 100
    train_idx = np.unique(np.linspace(0, N_GRID - 1, N_TRAIN).astype(int))
    t_train = t_grid[train_idx]
    y_train = y_true[train_idx]

    #---NN ARCHITECTURE---
    DEPTH = 3
    WIDTH = 1024

    # ---KERNEL RIDGE REGRESSION---
    
    # INFINITE-WIDTH MLP KERNEL
    # Theoretical counterpart of the (wide) MLP trained to convergence with MSE:
    #     f(t*) = K(t*, X) (K(X, X) + lambda I)^{-1} y
    # with K = Theta_inf the infinite-width NTK of a depth-DEPTH tanh MLP.
    mlp_inf_ker_fn = infinite_width_ntk_mlp_fn(DEPTH)
    Theta = eval_kernel_matrix(mlp_inf_ker_fn, t_grid)
    K_tt = Theta[np.ix_(train_idx, train_idx)]                  # train-train
    K_st = Theta[:, train_idx]                                  # grid-train
    ridge = 1e-3 * np.mean(np.diag(K_tt))                       # numerical jitter
    coeffs = np.linalg.solve(K_tt + ridge * np.eye(len(train_idx)), y_train)
    f_krr_mlp = K_st @ coeffs                                       # KRR predictor on grid

    print(f"KRR vs true signal: RMSE = {np.sqrt(np.mean((f_krr_mlp - y_true) ** 2)):.2e}")
    
    # SIREN
    siren_inf_ker_fn = infinite_width_ntk_siren_fn(DEPTH)
    Theta = eval_kernel_matrix(siren_inf_ker_fn, t_grid)
    K_tt = Theta[np.ix_(train_idx, train_idx)]                  # train-train
    K_st = Theta[:, train_idx]                                  # grid-train
    ridge = 1e-6 * np.mean(np.diag(K_tt))                       # numerical jitter
    coeffs = np.linalg.solve(K_tt + ridge * np.eye(len(train_idx)), y_train)
    f_krr_siren = K_st @ coeffs                                       # KRR predictor on grid

    print(f"KRR vs true signal: RMSE = {np.sqrt(np.mean((f_krr_siren - y_true) ** 2)):.2e}")
    
    # FOURIER FEATURES
    FF_SIGMA = 10.0
    FF_N_FREQS = 16
    rng = np.random.default_rng()
    FOURIER_FREQS = rng.normal(loc=0.0, scale=FF_SIGMA, size=(FF_N_FREQS,))
    
    fourier_inf_ker_fn = infinite_width_ntk_fourier_fn(DEPTH,ff_freqs = FOURIER_FREQS)
    Theta = eval_kernel_matrix(fourier_inf_ker_fn, t_grid)
    K_tt = Theta[np.ix_(train_idx, train_idx)]                  # train-train
    K_st = Theta[:, train_idx]                                  # grid-train
    ridge = 1e-6 * np.mean(np.diag(K_tt))                       # numerical jitter
    coeffs = np.linalg.solve(K_tt + ridge * np.eye(len(train_idx)), y_train)
    f_krr_fourier = K_st @ coeffs                                       # KRR predictor on grid

    print(f"KRR vs true signal: RMSE = {np.sqrt(np.mean((f_krr_fourier - y_true) ** 2)):.2e}")

    # ---LAZY-REGIME ("NTK") TRAINING---
    # Chizat-Bach lazy training: scale the output by a large factor ALPHA and
    # center it at initialization,
    #     f(t) = ALPHA * ( g_theta(t) - g_theta0(t) ),
    # so the net starts at the zero function and the parameters need only move
    # O(1/ALPHA) to fit the data.  The first-order (tangent) expansion then stays
    # valid throughout training, i.e. training reduces to kernel regression with
    # the *fixed* empirical NTK at initialization -- no feature learning.
    # We use plain gradient descent (NOT Adam: its per-coordinate gradient
    # normalisation would undo the ALPHA scaling) with the learning rate
    # compensated by 1/ALPHA^2 so the prediction dynamics stay O(1).
    ALPHA = 100.0
    
    # MLP
    mlp = PINN(PINNConfig(n_vars=1, width=WIDTH, depth=DEPTH)).to(DEVICE)
    mlp0 = copy.deepcopy(mlp).eval()                  # frozen init g_theta0
    for p in mlp0.parameters():
        p.requires_grad_(False)

    def lazy_forward(t):
        return ALPHA * (mlp(t) - mlp0(t))

    t_tr = torch.tensor(t_train, device=DEVICE).view(-1, 1)
    y_tr = torch.tensor(y_train, device=DEVICE).view(-1, 1)

    BASE_LR = 1e-2
    optimizer = torch.optim.SGD(mlp.parameters(), lr=BASE_LR / ALPHA ** 2)
    N_ITER = 20_000
    mlp.train()
    for it in range(N_ITER + 1):
        optimizer.zero_grad()
        loss = ((lazy_forward(t_tr) - y_tr) ** 2).mean()
        loss.backward()
        optimizer.step()
        if it % (N_ITER // 5) == 0:
            # parameter drift from init -- stays small in the lazy regime
            drift = max((p - p0).abs().max().item()
                        for p, p0 in zip(mlp.parameters(), mlp0.parameters()))
            print(f"  iter {it:6d} | mse {loss.item():.3e} | max|theta-theta0| {drift:.2e}")

    mlp.eval()
    with torch.no_grad():
        f_mlp = lazy_forward(torch.tensor(t_grid, device=DEVICE).view(-1, 1)).cpu().numpy().ravel()

    print(f"Lazy MLP vs true signal: RMSE = {np.sqrt(np.mean((f_mlp - y_true) ** 2)):.2e}")
    
    # SIREN 
    SIREN_OMEGA = 8.25 #8.25
    siren = SIREN(SIRENConfig(n_vars=1, width=WIDTH, depth=DEPTH, omega_0=SIREN_OMEGA)).to(DEVICE)
    siren0 = copy.deepcopy(siren).eval()                  # frozen init g_theta0
    for p in siren0.parameters():
        p.requires_grad_(False)

    def lazy_forward(t):
        return ALPHA * (siren(t) - siren0(t))

    t_tr = torch.tensor(t_train, device=DEVICE).view(-1, 1)
    y_tr = torch.tensor(y_train, device=DEVICE).view(-1, 1)

    BASE_LR = 1e-2
    optimizer = torch.optim.SGD(siren.parameters(), lr=BASE_LR / ALPHA ** 2)
    N_ITER = 20_000
    siren.train()
    for it in range(N_ITER + 1):
        optimizer.zero_grad()
        loss = ((lazy_forward(t_tr) - y_tr) ** 2).mean()
        loss.backward()
        optimizer.step()
        if it % (N_ITER // 5) == 0:
            # parameter drift from init -- stays small in the lazy regime
            drift = max((p - p0).abs().max().item()
                        for p, p0 in zip(siren.parameters(), siren0.parameters()))
            print(f"  iter {it:6d} | mse {loss.item():.3e} | max|theta-theta0| {drift:.2e}")

    siren.eval()
    with torch.no_grad():
        f_siren = lazy_forward(torch.tensor(t_grid, device=DEVICE).view(-1, 1)).cpu().numpy().ravel()

    print(f"Lazy SIREN vs true signal: RMSE = {np.sqrt(np.mean((f_siren - y_true) ** 2)):.2e}")
    
    # FOURIER FEATURES
    ff_mlp = PINN(PINNConfig(n_vars=1, width=WIDTH, depth=DEPTH, fourier_freqs=FOURIER_FREQS)).to(DEVICE)
    ff_mlp0 = copy.deepcopy(ff_mlp).eval()                  # frozen init g_theta0
    for p in ff_mlp0.parameters():
        p.requires_grad_(False)

    def lazy_forward(t):
        return ALPHA * (ff_mlp(t) - ff_mlp0(t))

    t_tr = torch.tensor(t_train, device=DEVICE).view(-1, 1)
    y_tr = torch.tensor(y_train, device=DEVICE).view(-1, 1)

    BASE_LR = 1e-2
    optimizer = torch.optim.SGD(ff_mlp.parameters(), lr=BASE_LR / ALPHA ** 2)
    N_ITER = 20_000
    ff_mlp.train()
    for it in range(N_ITER + 1):
        optimizer.zero_grad()
        loss = ((lazy_forward(t_tr) - y_tr) ** 2).mean()
        loss.backward()
        optimizer.step()
        if it % (N_ITER // 5) == 0:
            # parameter drift from init -- stays small in the lazy regime
            drift = max((p - p0).abs().max().item()
                        for p, p0 in zip(ff_mlp.parameters(), ff_mlp0.parameters()))
            print(f"  iter {it:6d} | mse {loss.item():.3e} | max|theta-theta0| {drift:.2e}")

    ff_mlp.eval()
    with torch.no_grad():
        f_fourier = lazy_forward(torch.tensor(t_grid, device=DEVICE).view(-1, 1)).cpu().numpy().ravel()

    print(f"Lazy FOURIER vs true signal: RMSE = {np.sqrt(np.mean((f_fourier - y_true) ** 2)):.2e}")

    # ---PLOTS---
    
    # Simple MLP
    fig, ax = plt.subplots(figsize=(10, 4))
    # KRR regression
    ax.plot(t_grid, f_krr_mlp, color="red", lw=2.0, alpha = 0.7,
            label=r"KRR MLP($\Theta_\infty$, infinite width)")
    # NN training
    ax.plot(t_grid, f_mlp, color="blue", lw=2.0, alpha = 0.7,
            label=f"lazy MLP (width {WIDTH}, $\\alpha$={ALPHA:g})")
    # Signal and data
    ax.plot(t_grid, y_true, "k--", lw=1.5, label="true signal")
    ax.scatter(t_train, y_train, color="green", s=20, zorder=5,
               label="training points")
    # Settings
    ax.set(xlabel="t", ylabel="y(t)",
           title=r"Lazy-regime training vs infinite-width NTK kernel ridge regression")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mlp_vs_krr_predictor.png", dpi=150, bbox_inches="tight")
    
    # SIRENs
    fig, ax = plt.subplots(figsize=(10, 4))
    # KRR regression
    ax.plot(t_grid, f_krr_siren, color="red", lw=2.0, alpha = 0.7,
            label=r"KRR SIREN($\Theta_\infty$, infinite width)")
    # NN training
    ax.plot(t_grid, f_siren, color="blue", lw=2.0, alpha = 0.7,
            label=f"lazy SIREN (width {WIDTH}, $\\alpha$={ALPHA:g})")
    # True signal and data
    ax.plot(t_grid, y_true, "k--", lw=1.5, label="true signal")
    ax.scatter(t_train, y_train, color="green", s=20, zorder=5,
               label="training points")
    # Settings
    ax.set(xlabel="t", ylabel="y(t)",
           title=r"Lazy-regime training vs infinite-width NTK kernel ridge regression")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "siren_vs_krr_predictor.png", dpi=150, bbox_inches="tight")
    
    # Fourier features
    fig, ax = plt.subplots(figsize=(10, 4))
    # KRR regression
    ax.plot(t_grid, f_krr_fourier, color="red", lw=2.0, alpha = 0.7,
            label=r"KRR FOURIER($\Theta_\infty$, infinite width)")
    # NN training
    ax.plot(t_grid, f_fourier, color="blue", lw=2.0, alpha = 0.7,
            label=f"lazy FOURIER (width {WIDTH}, $\\alpha$={ALPHA:g})")
    # True signal and data
    ax.plot(t_grid, y_true, "k--", lw=1.5, label="true signal")
    ax.scatter(t_train, y_train, color="green", s=20, zorder=5,
               label="training points")
    # Settings
    ax.set(xlabel="t", ylabel="y(t)",
           title=r"Lazy-regime training vs infinite-width NTK kernel ridge regression")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fourier_vs_krr_predictor.png", dpi=150, bbox_inches="tight")
    

if __name__ == "__main__":
    main()