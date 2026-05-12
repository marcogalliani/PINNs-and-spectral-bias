"""
Universal Differential Equation (NeuralODE) / NeuralODE implementation.

A NeuralODE extends a known (possibly incomplete) mechanistic model with a neural
network correction term:

    dz/dt = f_known(t, z) + NN(t, z; θ)

Setting f_known=None gives a pure NeuralODE; setting f_known to the full true
RHS forces NN → 0 at convergence.

Training integrates the augmented ODE forward with a differentiable fixed-step
RK4 solver (discretize-then-optimize) and minimises a trajectory / data loss.
Backpropagation flows through all RK4 stages automatically — no adjoint needed
at the scales used here.

Contrast with PINNs:
  PINN : z(t) parametrised directly as NN(t;θ); ODE enforced via residual loss.
  NeuralODE  : dz/dt = g(t,z;θ) parametrised; z(t) obtained by integrating g.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from src.PINNs import FourierEmbedding, TrainingFrame
from src.spectral_analysis import compute_fft


# ---------------------------------------------------------------------------
# Differentiable integrators
# ---------------------------------------------------------------------------

def odeint_euler(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    t_eval: torch.Tensor,
) -> torch.Tensor:
    """
    Explicit Euler integration.  1 NN call per step — fast for training.

    func: callable(t, z) -> dz/dt, t: 0-d tensor, z: (n_vars,).
    Returns (N, n_vars).
    """
    traj = [z0]
    for i in range(len(t_eval) - 1):
        dt = t_eval[i + 1] - t_eval[i]
        traj.append(traj[-1] + dt * func(t_eval[i], traj[-1]))
    return torch.stack(traj, dim=0)


def _rk4_step(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    z: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    half = 0.5 * dt
    k1 = func(t,        z)
    k2 = func(t + half, z + half * k1)
    k3 = func(t + half, z + half * k2)
    k4 = func(t + dt,   z + dt   * k3)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def odeint_rk4(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    t_eval: torch.Tensor,
) -> torch.Tensor:
    """
    Fixed-step RK4 integration.  4 NN calls per step — accurate for eval.

    func: callable(t, z) -> dz/dt, t: 0-d tensor, z: (n_vars,).
    Returns (N, n_vars).
    """
    traj = [z0]
    for i in range(len(t_eval) - 1):
        dt = t_eval[i + 1] - t_eval[i]
        traj.append(_rk4_step(func, t_eval[i], traj[-1], dt))
    return torch.stack(traj, dim=0)


# ---------------------------------------------------------------------------
# NeuralODE model
# ---------------------------------------------------------------------------

class NeuralODEFunc(nn.Module):
    """
    RHS of the augmented ODE:  dz/dt = f_known(t, z) + NN(t, z; θ).

    The forward pass uses a scalar-time / unbatched interface compatible with
    the RK4 integrator above.  A separate correction_batch method accepts the
    (N, 1) / (N, n_vars) batched convention used elsewhere in the project.

    Parameters
    ----------
    n_vars : int
    width, depth : int
        MLP hidden-layer width and total depth (including input and output).
    activation : nn.Module class
    fourier_freqs : sequence of float or None
        If provided, embed t with sinusoidal features at these frequencies
        before the MLP.  Helps overcome spectral bias for oscillatory dynamics.
    known_rhs : callable(t, z) -> (N, n_vars) or None
        Mechanistic component.  Uses the batched interface
        (t: Tensor(N,1), z: Tensor(N,n_vars)) as in PINNs.py.
        None → pure NeuralODE.
    zero_init : bool
        Initialise the output layer to zero so training starts from f_known.
        This is a strong inductive bias: the network only needs to learn the
        discrepancy, not the full dynamics.
    """

    def __init__(
        self,
        n_vars: int,
        width: int = 64,
        depth: int = 3,
        activation: type[nn.Module] = nn.Tanh,
        fourier_freqs: Optional[Sequence[float]] = None,
        known_rhs: Optional[Callable] = None,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.n_vars    = n_vars
        self.known_rhs = known_rhs

        if fourier_freqs is not None:
            self.embedding: Optional[FourierEmbedding] = FourierEmbedding(fourier_freqs)
            t_dim = self.embedding.out_dim
        else:
            self.embedding = None
            t_dim = 1

        in_dim = t_dim + n_vars
        layers: list[nn.Module] = [nn.Linear(in_dim, width), activation()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), activation()]
        out_layer = nn.Linear(width, n_vars)
        if zero_init:
            nn.init.zeros_(out_layer.weight)
            nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)
        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Internal helper: works for any batch size N ≥ 1
    # ------------------------------------------------------------------

    def _correction(self, t_in: torch.Tensor, z_in: torch.Tensor) -> torch.Tensor:
        """t_in: (N, 1), z_in: (N, n_vars) → (N, n_vars)"""
        t_feat = self.embedding(t_in) if self.embedding is not None else t_in
        return self.net(torch.cat([t_feat, z_in], dim=1))

    # ------------------------------------------------------------------
    # Scalar interface for the RK4 integrator
    # ------------------------------------------------------------------

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """t: 0-d tensor, z: (n_vars,) → (n_vars,)"""
        t1 = t.view(1, 1)
        z1 = z.unsqueeze(0)                           # (1, n_vars)
        correction = self._correction(t1, z1).squeeze(0)

        if self.known_rhs is not None:
            known = self.known_rhs(t1, z1).squeeze(0)
            return known + correction
        return correction

    # ------------------------------------------------------------------
    # Batched correction only (for spectral analysis post-processing)
    # ------------------------------------------------------------------

    def correction_batch(
        self, t_batch: torch.Tensor, z_batch: torch.Tensor
    ) -> torch.Tensor:
        """t_batch: (N, 1), z_batch: (N, n_vars) → (N, n_vars)"""
        return self._correction(t_batch, z_batch)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ude(
    func: NeuralODEFunc,
    t_span: tuple[float, float],
    y0: Sequence[float],
    t_obs_np: np.ndarray,
    y_obs_np: np.ndarray,
    *,
    n_iter: int = 20_000,
    lr: float = 1e-3,
    lr_decay: float = 0.9995,
    rec_frq: int = 100,
    t_eval_np: Optional[np.ndarray] = None,
    n_eval: int = 200,
    save_snapshots: bool = False,
    verbose: bool = True,
) -> list[TrainingFrame]:
    """
    Train the NeuralODE on trajectory observations.

    Loss = MSE between the integrated trajectory at observation times and y_obs.
    The initial condition is enforced by construction (z(t0) = y0 always).

    Backpropagation runs only on the coarse training grid (t_obs_np), whose
    depth equals the number of observations.  The dense eval grid (t_eval_np)
    is used only under no_grad when recording frames, keeping per-iteration
    cost independent of the plotting resolution.

    Parameters
    ----------
    func : NeuralODEFunc  (already moved to the desired device)
    t_span : (t0, T)
    y0 : initial condition (length n_vars)
    t_obs_np : (N_obs,) observation times — also used as the training grid
    y_obs_np : (N_obs, n_vars) observed state values
    t_eval_np : dense evaluation grid for recording; defaults to n_eval
                uniformly-spaced points over t_span
    n_eval : grid size if t_eval_np is None
    """
    device = next(func.parameters()).device
    t0, T  = t_span

    if t_eval_np is None:
        t_eval_np = np.linspace(t0, T, n_eval)

    # Training grid = observation times (coarse, drives backprop depth)
    t_train_t = torch.tensor(t_obs_np, dtype=torch.float32, device=device)
    y_obs_t   = torch.tensor(y_obs_np, dtype=torch.float32, device=device)
    z0_t      = torch.tensor(list(y0), dtype=torch.float32, device=device)

    # Dense eval grid (no_grad only)
    t_eval_t  = torch.tensor(t_eval_np, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(func.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    frames: list[TrainingFrame] = []
    func.train()

    for it in range(n_iter + 1):
        optimizer.zero_grad()

        # Euler: 1 NN call per step, 4x fewer than RK4 → fast backprop graph
        traj_train = odeint_euler(func, z0_t, t_train_t)   # (N_obs, n_vars)
        loss = ((traj_train - y_obs_t) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if it % rec_frq == 0:
            func.eval()
            with torch.no_grad():
                pred_np = odeint_rk4(func, z0_t, t_eval_t).cpu().numpy()
            func.train()

            snapshot = None
            if save_snapshots:
                snapshot = {k: v.detach().cpu().clone()
                            for k, v in func.state_dict().items()}

            frames.append(
                TrainingFrame(it, pred_np, loss.item(), loss.item(), 0.0, snapshot)
            )

            if verbose and n_iter > 0 and it % max(1, n_iter // 5) == 0:
                print(f"  iter {it:6d} | loss {loss.item():.3e}")

    func.eval()
    return frames


# ---------------------------------------------------------------------------
# Post-processing: NN correction spectrum across snapshots
# ---------------------------------------------------------------------------

def compute_ude_correction(
    frames: list[TrainingFrame],
    func: NeuralODEFunc,
    t_eval_np: np.ndarray,
    y0: Sequence[float],
    device=None,
) -> list[Optional[np.ndarray]]:
    """
    For each snapshot in frames, integrate the trajectory and return the
    NN correction  NN(t, z(t); θ).

    This is the NeuralODE analogue of compute_pinn_residuals: both represent
    the signal the network must express to bridge the mechanistic model
    and the true dynamics.

    Returns
    -------
    List of (N, n_vars) arrays (or None for frames without a saved state).
    """
    if device is None:
        device = next(func.parameters()).device

    saved_state = {k: v.clone() for k, v in func.state_dict().items()}

    t_eval_t = torch.tensor(t_eval_np, dtype=torch.float32, device=device)
    z0_t     = torch.tensor(list(y0),  dtype=torch.float32, device=device)
    t_batch  = t_eval_t.view(-1, 1)   # (N, 1)

    corrections = []
    for frame in frames:
        if frame.model_state is None:
            corrections.append(None)
            continue

        func.load_state_dict(frame.model_state)
        func.eval()

        with torch.no_grad():
            traj = odeint_rk4(func, z0_t, t_eval_t)          # (N, n_vars)
            corr = func.correction_batch(t_batch, traj)       # (N, n_vars)

        corrections.append(corr.cpu().numpy())

    func.load_state_dict(saved_state)
    func.eval()
    return corrections


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_correction_dynamics(
    corrections: list[np.ndarray],
    iter_nums,
    t_eval: np.ndarray,
    freqs,
    sample_rate: float,
    u_true: Optional[np.ndarray] = None,
    *,
    title: str = "NeuralODE — NN Correction NN(t, z(t); θ) Dynamics",
    save_path=None,
):
    """
    Three-panel figure mirroring plot_residual_dynamics from PINNs.py.

    The NN correction NN(t, z(t);θ) plays the same role as the PINN residual
    u(t;θ) = z'(t;θ) − F_known(t,z): both converge to the true forcing as
    training proceeds.  Spectral bias predicts low frequencies converge first.

    Panel (left)   — |FFT(correction)| at each forcing frequency vs iteration.
    Panel (centre) — Full correction spectrum at the final iterate.
    Panel (right)  — Time-domain correction at three training snapshots
                     (+ true forcing if u_true is provided).
    """
    valid = [(it, c) for it, c in zip(iter_nums, corrections) if c is not None]
    if not valid:
        return
    iters, corr_list = zip(*valid)
    iters   = np.asarray(iters)
    palette = sns.color_palette("husl", len(freqs))

    amp_curves = {f: [] for f in freqs}
    for c in corr_list:
        frqs_k, spec_k = compute_fft(c[:, 0], sample_rate)
        for f in freqs:
            amp_curves[f].append(float(spec_k[np.argmin(np.abs(frqs_k - f))]))

    frqs_final, spec_final = compute_fft(corr_list[-1][:, 0], sample_rate)
    snap_idx = [0, len(corr_list) // 2, len(corr_list) - 1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(title, fontsize=12)

    ax = axes[0]
    for f, col in zip(freqs, palette):
        ax.plot(iters, amp_curves[f], label=f"{f} Hz", color=col, linewidth=1.2)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("|FFT(correction)| at forcing frequency")
    ax.set_yscale("log")
    ax.set_title("Spectral dynamics of correction\n(spectral bias → high freq decays last)")
    ax.legend(fontsize=8)

    ax = axes[1]
    mask = frqs_final <= max(freqs) * 1.5
    ax.plot(frqs_final[mask], spec_final[mask], color="steelblue", linewidth=1.2,
            label="NeuralODE correction (final)")
    if u_true is not None:
        frqs_true, spec_true = compute_fft(u_true, sample_rate)
        ax.plot(frqs_true[mask], spec_true[mask], color="firebrick", linewidth=1.2,
                alpha=0.7, linestyle="--", label="True forcing")
    for f, col in zip(freqs, palette):
        ax.axvline(f, color=col, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|FFT(correction)|")
    ax.set_title("Final correction spectrum")
    ax.legend(fontsize=8)

    ax = axes[2]
    snap_palette = sns.color_palette("Blues_d", len(snap_idx))
    for idx, col in zip(snap_idx, snap_palette):
        ax.plot(t_eval, corr_list[idx][:, 0], color=col, alpha=0.85,
                linewidth=1.1, label=f"iter {iters[idx]}")
    if u_true is not None:
        ax.plot(t_eval, u_true, color="firebrick", linewidth=1.2,
                alpha=0.7, label="True forcing")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("t")
    ax.set_ylabel("NN correction")
    ax.set_title("Correction waveform (first / mid / last)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ude_vs_pinn_spectral_dynamics(
    ude_corrections: list[np.ndarray],
    pinn_residuals: list[np.ndarray],
    ude_iters,
    pinn_iters,
    freqs,
    sample_rate: float,
    *,
    title: str = "NeuralODE correction vs PINN residual — Spectral Dynamics",
    save_path=None,
):
    """
    Side-by-side comparison of how quickly NeuralODE and PINN learn each forcing
    frequency component.

    For each method the plot shows  |FFT(signal)| / |FFT(signal)|_{final}
    so that both axes are on a normalised scale despite possibly different
    absolute amplitudes.
    """
    def _amp_curves(signals, iters_arr):
        curves = {f: [] for f in freqs}
        for sig in signals:
            if sig is None:
                for f in freqs:
                    curves[f].append(np.nan)
                continue
            frqs_k, spec_k = compute_fft(sig[:, 0], sample_rate)
            for f in freqs:
                curves[f].append(float(spec_k[np.argmin(np.abs(frqs_k - f))]))
        return curves

    ude_iters  = np.asarray(ude_iters)
    pinn_iters = np.asarray(pinn_iters)
    ude_curves  = _amp_curves(ude_corrections, ude_iters)
    pinn_curves = _amp_curves(pinn_residuals,  pinn_iters)

    palette = sns.color_palette("husl", len(freqs))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle(title, fontsize=12)

    for ax, curves, iters, method in [
        (axes[0], ude_curves,  ude_iters,  "NeuralODE correction"),
        (axes[1], pinn_curves, pinn_iters, "PINN residual"),
    ]:
        for f, col in zip(freqs, palette):
            vals = np.array(curves[f], dtype=float)
            final = vals[np.isfinite(vals)][-1] if np.any(np.isfinite(vals)) else 1.0
            ax.plot(iters, vals / (final + 1e-30), label=f"{f} Hz", color=col,
                    linewidth=1.2)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Normalised |FFT| at forcing freq")
        ax.set_title(f"{method}\n(normalised to final amplitude)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
