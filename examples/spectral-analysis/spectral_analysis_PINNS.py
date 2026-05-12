"""
Spectral bias of PINNs vs Picard iteration on a forced linear ODE.

Expected contrast:

  * PINN — gradient descent on a tanh MLP exhibits spectral bias: low
    frequencies of the solution land first, high frequencies arrive much later (or never within the training budget).

  * Picard — z^{k+1}(t) = z0 + ∫₀ᵗ f(s, z^k(s)) ds.  Starting from the
    constant z0, the first iterate already integrates the full forcing once, so high-frequency content appears immediately and successive iterates refine the *amplitude* of every frequency in lockstep.

ODE choice — first-order linear, multi-frequency forcing:
  z' = -α z + Σ_k A_k sin(2π f_k t + φ_k)

The Forcing is chosen so the steady-state amplitude |H(ω)| = 1/√(α²+ω²) gives *comparable* response at the two frequencies — otherwise one component dominates the FFT and the comparison loses its point.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import Namespace
from scipy.integrate import solve_ivp

from src.PINNs import (PINN, train_pinn, plot_spectral_dynamics, plot_loss_curves,
                       compute_pinn_residuals, plot_residual_dynamics)
from src.numerical_solvers import picard_solve
from src.spectral_analysis import compute_fft, plot_ntk_analysis

sns.set_theme(style="whitegrid")
torch.manual_seed(42)
np.random.seed(42)


# ODE
class ForcedFirstOrderODE:
    """
    z' = -α z + Σ_k A_k sin(2π f_k t + φ_k)

    Provides two RHS callables with identical math:
      rhs_np    -- (t: float, z: ndarray(1,)) -> list[float]            (scipy/Picard)
      rhs_torch -- (t: Tensor(N,1), z: Tensor(N,1)) -> Tensor(N,1)      (PINN)
    """

    def __init__(self, alpha, freqs, amps, phases):
        self.alpha = alpha
        self.freqs = list(freqs)
        self.amps = list(amps)
        self.phases = list(phases)

    def _forcing(self, t, sin_fn):
        return sum(A * sin_fn(2 * np.pi * f * t + ph)
                   for f, A, ph in zip(self.freqs, self.amps, self.phases))

    def rhs_np(self, t, z):
        return [-self.alpha * z[0] + self._forcing(t, np.sin)]

    def rhs_torch(self, t, z):
        F = self._forcing(t, torch.sin)
        return -self.alpha * z + F


# Plot utils
def plot_solution_comparison(t, solutions, labels, colors, opt, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Final solutions  -  forcing frequencies: {opt.freqs} Hz",
                 fontsize=12)
    sample_rate = opt.N_POINTS / opt.T

    for sol, lab, col in zip(solutions, labels, colors):
        axes[0].plot(t, sol[:, 0], label=lab, color=col, alpha=0.85, linewidth=1.4)
    axes[0].set_xlabel("Time [s]"); axes[0].set_ylabel("z(t)")
    axes[0].set_title("Solution"); axes[0].legend(fontsize=8)

    for sol, lab, col in zip(solutions, labels, colors):
        frqs, spec = compute_fft(sol[:, 0], sample_rate)
        axes[1].plot(frqs, spec, label=lab, color=col, alpha=0.85)
    for f in opt.freqs:
        axes[1].axvline(f, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    axes[1].set_xlim([0, max(opt.freqs) * 1.5])
    axes[1].set_xlabel("Frequency [Hz]"); axes[1].set_ylabel("|FFT|")
    axes[1].set_title("Spectrum"); axes[1].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def main():
    opt = Namespace()

    # ODE
    opt.alpha = 1.0
    opt.freqs = [1.0, 3.0, 5.0, 10.0]
    opt.amps = [1.0, 1.0, 1.0, 1.0]            
    # equal forcing energy; |H(ω)| then sets the spectrum
    opt.phases = [0.0, 0.0, 0.0, 0.0]
    opt.y0 = [0.0]

    # Time domain
    opt.T = 2.0
    opt.N_POINTS = 200               # >> 2·f_max·T = 12

    # PINN — small enough for fast iteration, large enough to converge
    opt.WIDTH = 512
    opt.DEPTH = 4
    opt.NUM_ITER = 5_000
    opt.REC_FRQ = 100
    opt.LR = 1e-3
    opt.N_COLLOC = 512
    # unused under hard_ic
    opt.IC_WEIGHT = 100.0            

    # Picard
    opt.PICARD_ITERS = 20

    if torch.cuda.is_available():
        opt.DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        opt.DEVICE = torch.device("mps")
    else:
        opt.DEVICE = torch.device("cpu")
    print(f"Device: {opt.DEVICE}")

    ode = ForcedFirstOrderODE(opt.alpha, opt.freqs, opt.amps, opt.phases)
    t_eval = np.linspace(0, opt.T, opt.N_POINTS)
    sample_rate = opt.N_POINTS / opt.T

    # Reference
    print("Reference (RK45)...")
    sol_ref = solve_ivp(ode.rhs_np, [0, opt.T], opt.y0, t_eval=t_eval,
                        method="RK45", rtol=1e-9, atol=1e-11)
    y_ref = sol_ref.y.T

    # Picard iterates
    print(f"Picard ({opt.PICARD_ITERS} iterations)...")
    picard_iterates = picard_solve(ode.rhs_np, opt.y0, t_eval, opt.PICARD_ITERS)
    y_picard = picard_iterates[-1]

    # PINN
    print(f"PINN ({opt.NUM_ITER} iters, width={opt.WIDTH}, depth={opt.DEPTH})...")
    model = PINN(n_vars=1, width=opt.WIDTH, depth=opt.DEPTH).to(opt.DEVICE)
    frames = train_pinn(
        model,
        ode.rhs_torch,
        t_span=(0.0, opt.T),
        y0=opt.y0,
        n_iter=opt.NUM_ITER,
        n_colloc=opt.N_COLLOC,
        lr=opt.LR,
        ic_weight=opt.IC_WEIGHT,
        hard_ic=True,
        rec_frq=opt.REC_FRQ,
        t_eval_np=t_eval,
        save_snapshots=True,
        verbose=True,
    )
    # hard_ic reparametrisation z(t) = z0 + (t - t0) NN(t) must be applied at
    # eval time too — model(t) alone returns the raw NN output.
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32, device=opt.DEVICE).view(-1, 1)
        y0_t = torch.tensor([opt.y0], dtype=torch.float32, device=opt.DEVICE)
        y_pinn = (y0_t + t_tensor * model(t_tensor)).cpu().numpy()

    # Plots
    print("Plotting...")
    plot_solution_comparison(
        t_eval,
        [y_ref, y_picard, y_pinn],
        ["Reference (RK45)", f"Picard (k={opt.PICARD_ITERS})", "PINN"],
        ["black", "steelblue", "darkorange"],
        opt,
        save_path="solution_comparison.png",
    )

    plot_spectral_dynamics(
        [f.prediction for f in frames],
        [f.iter_num for f in frames],
        opt.freqs,
        sample_rate,
        reference=y_ref,
        title="PINN — Spectral Dynamics During Training",
        iter_label="Training Iteration",
        save_path="pinn_spectral_dynamics.png",
    )

    plot_spectral_dynamics(
        picard_iterates,
        list(range(len(picard_iterates))),
        opt.freqs,
        sample_rate,
        reference=y_ref,
        title="Picard — Spectral Dynamics Across Iterations",
        iter_label="Picard Iteration",
        save_path="picard_spectral_dynamics.png",
    )

    plot_loss_curves(frames, save_path="pinn_loss.png")

    print("-> Residual dynamics...")
    residuals = compute_pinn_residuals(
        frames, model, ode.rhs_torch, t_eval, opt.y0,
        hard_ic=True, device=opt.DEVICE,
    )
    plot_residual_dynamics(
        residuals,
        [f.iter_num for f in frames],
        t_eval,
        opt.freqs,
        sample_rate,
        title="PINN — ODE Residual u(t;θ) Dynamics",
        save_path="pinn_residual_dynamics.png",
    )

    print("->NTK analysis")
    snapshots = [(f.iter_num, f.model_state) for f in frames]
    plot_ntk_analysis(
        model,
        t_eval,
        sample_rate,
        freqs=opt.freqs,
        snapshots=snapshots,
        ntk_subsample=5,
        device=opt.DEVICE,
        transform=lambda t, z: t * z,       # hard_ic: actual trajectory = t · NN(t)
        title="NTK Analysis — PINN",
        save_path="ntk_analysis.png",
    )

    # Summary
    print("\n=== Spectral amplitude at forcing frequencies ===")
    print(f"{'Method':<25}", end="")
    for f in opt.freqs:
        print(f"  {f} Hz", end="")
    print()
    for sol, lab in [(y_ref, "Reference (RK45)"),
                     (y_picard, f"Picard (k={opt.PICARD_ITERS})"),
                     (y_pinn, "PINN")]:
        frqs_s, spec_s = compute_fft(sol[:, 0], sample_rate)
        print(f"{lab:<25}", end="")
        for f in opt.freqs:
            print(f"  {spec_s[np.argmin(np.abs(frqs_s - f))]:.4f}", end="")
        print()


if __name__ == "__main__":
    main()