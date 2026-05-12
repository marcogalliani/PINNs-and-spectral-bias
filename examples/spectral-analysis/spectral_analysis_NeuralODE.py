"""
Spectral bias of a Universal Differential Equation (NeuralODE).

Setup
-----
True ODE  : z' = -alpha * z + sum_k A_k sin(2 pi f_k t)
Known part: z' = -alpha * z                  (decay only)
Unknown   : sum_k A_k sin(2 pi f_k t)        (multi-frequency forcing)

The NeuralODE parametrises
    dz/dt = -alpha * z + NN(t, z; theta)
and integrates forward with fixed-step RK4.  Training minimises the MSE
between the integrated trajectory and sparse observations of the true solution.

Research question
-----------------
Does the correction NN(t, z(t); theta) — which must converge to the true
multi-frequency forcing — exhibit spectral bias?  I.e. do low-frequency
components of the forcing appear in the correction before high-frequency ones?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import seaborn as sns
from argparse import Namespace
from scipy.integrate import solve_ivp

from src.NeuralODE import (
    NeuralODEFunc, train_ude, compute_ude_correction,
    plot_correction_dynamics,
)
from src.spectral_analysis import compute_fft
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# ODE
# ---------------------------------------------------------------------------

class ForcedFirstOrderODE:
    def __init__(self, alpha, freqs, amps, phases):
        self.alpha  = alpha
        self.freqs  = list(freqs)
        self.amps   = list(amps)
        self.phases = list(phases)

    def _forcing(self, t, sin_fn):
        return sum(
            A * sin_fn(2 * np.pi * f * t + ph)
            for f, A, ph in zip(self.freqs, self.amps, self.phases)
        )

    def rhs_np(self, t, z):
        return [-self.alpha * z[0] + self._forcing(t, np.sin)]

    def known_rhs_torch(self, t, z):
        return -self.alpha * z


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    opt = Namespace()

    # ODE
    opt.alpha  = 1.0
    opt.freqs  = [1.0, 3.0, 5.0, 10.0]
    opt.amps   = [1.0, 1.0, 1.0, 1.0]
    opt.phases = [0.0, 0.0, 0.0, 0.0]
    opt.y0     = [0.0]

    # Time grid — kept short to limit the number of RK4 steps per iteration
    opt.T        = 2.0
    opt.N_POINTS = 200          # integration + eval grid
    opt.OBS_STEP = 4            # observe every 4th point → 25 observations

    # Training — backprop runs only through N_obs steps (N_POINTS // OBS_STEP = 25)
    opt.NUM_ITER = 10_000
    opt.REC_FRQ  = 200
    opt.LR       = 1e-3
    opt.LR_DECAY = 0.9998

    # NeuralODE architecture
    opt.NeuralODE_WIDTH = 256
    opt.NeuralODE_DEPTH = 3

    if torch.cuda.is_available():
        opt.DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        opt.DEVICE = torch.device("mps")
    else:
        opt.DEVICE = torch.device("cpu")
    print(f"Device: {opt.DEVICE}")

    t_eval      = np.linspace(0, opt.T, opt.N_POINTS)
    sample_rate = opt.N_POINTS / opt.T

    ode = ForcedFirstOrderODE(opt.alpha, opt.freqs, opt.amps, opt.phases)

    u_true = np.array([
        sum(A * np.sin(2 * np.pi * f * t + ph)
            for f, A, ph in zip(opt.freqs, opt.amps, opt.phases))
        for t in t_eval
    ])

    print("Reference (RK45)...")
    sol  = solve_ivp(ode.rhs_np, [0, opt.T], opt.y0, t_eval=t_eval,
                     method="RK45", rtol=1e-9, atol=1e-11)
    y_ref = sol.y.T   # (N, 1)

    obs_idx = np.arange(0, opt.N_POINTS, opt.OBS_STEP)
    t_obs   = t_eval[obs_idx]
    y_obs   = y_ref[obs_idx]

    # ---------------------------------------------------------------------------
    # NeuralODE
    # ---------------------------------------------------------------------------
    print(f"\nNeuralODE ({opt.NUM_ITER} iters, width={opt.NeuralODE_WIDTH}, depth={opt.NeuralODE_DEPTH})...")
    ude_func = NeuralODEFunc(
        n_vars=1, width=opt.NeuralODE_WIDTH, depth=opt.NeuralODE_DEPTH,
        known_rhs=ode.known_rhs_torch, zero_init=True,
    ).to(opt.DEVICE)

    ude_frames = train_ude(
        ude_func,
        t_span=(0.0, opt.T), y0=opt.y0,
        t_obs_np=t_obs, y_obs_np=y_obs,
        n_iter=opt.NUM_ITER, lr=opt.LR, lr_decay=opt.LR_DECAY,
        rec_frq=opt.REC_FRQ, t_eval_np=t_eval,
        save_snapshots=True, verbose=True,
    )

    y_ude = ude_frames[-1].prediction

    print("  Computing NeuralODE corrections...")
    ude_corrections = compute_ude_correction(
        ude_frames, ude_func, t_eval, opt.y0, device=opt.DEVICE,
    )

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    print("\nPlotting...")

    # Solution + spectrum
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"NeuralODE solution  —  forcing frequencies: {opt.freqs} Hz", fontsize=12)
    axes[0].plot(t_eval, y_ref[:, 0], "k-",         linewidth=1.4, label="Reference (RK45)")
    axes[0].plot(t_eval, y_ude[:, 0], color="steelblue", linewidth=1.2, alpha=0.9, label="NeuralODE")
    axes[0].scatter(t_obs, y_obs[:, 0], s=8, c="gray", alpha=0.6, zorder=3, label="Observations")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("z(t)"); axes[0].legend(fontsize=8)
    for sol, lab, col in [(y_ref, "Reference", "black"), (y_ude, "NeuralODE", "steelblue")]:
        frqs, spec = compute_fft(sol[:, 0], sample_rate)
        axes[1].plot(frqs, spec, label=lab, color=col, alpha=0.85)
    for f in opt.freqs:
        axes[1].axvline(f, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    axes[1].set_xlim([0, max(opt.freqs) * 1.5])
    axes[1].set_xlabel("Frequency [Hz]"); axes[1].set_ylabel("|FFT|"); axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("ude_solution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    plot_correction_dynamics(
        ude_corrections,
        [f.iter_num for f in ude_frames],
        t_eval, opt.freqs, sample_rate,
        u_true=u_true,
        title="NeuralODE — Correction NN(t, z(t); θ) Spectral Dynamics",
        save_path="ude_correction_dynamics.png",
    )

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n=== Spectral amplitude of NN correction at forcing frequencies ===")
    frqs_ref, _ = compute_fft(u_true, sample_rate)
    header = f"{'Signal':<28}" + "".join(f"  {f} Hz" for f in opt.freqs)
    print(header)
    for label, sig in [
        ("True forcing", u_true),
        ("NeuralODE correction (final)",
         next(c for c in reversed(ude_corrections) if c is not None)[:, 0]),
    ]:
        _, spec = compute_fft(sig, sample_rate)
        row = f"{label:<28}" + "".join(
            f"  {spec[np.argmin(np.abs(frqs_ref - f))]:.4f}" for f in opt.freqs
        )
        print(row)

    mse = np.mean((y_ude - y_ref) ** 2)
    print(f"\nNeuralODE trajectory MSE vs reference: {mse:.3e}")


if __name__ == "__main__":
    main()