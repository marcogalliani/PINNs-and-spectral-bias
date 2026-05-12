"""
General-purpose PINN for first-order ODE systems  z' = f(t, z).

The architecture is an MLP t -> z(t).  The physics loss enforces the ODE
residual via automatic differentiation through the network w.r.t. time.
The caller supplies the RHS as a callable

    rhs_torch(t: Tensor, z: Tensor) -> Tensor

where both t and z are (N, *) tensors on the same device as the model.
This keeps the module ODE-agnostic: any autonomous or non-autonomous system
can be plugged in without touching the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from src.spectral_analysis import compute_fft

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrainingFrame:
    iter_num: int
    prediction: Optional[np.ndarray]   # (N_eval, n_vars) or None
    loss: float
    phys_loss: float
    ic_loss: float
    model_state: Optional[dict] = field(default=None, repr=False)
    # CPU-side copy of state_dict; populated only when save_snapshots=True

class FourierEmbedding(nn.Module):
    """
    Map  t (N, 1)  ->  [sin(2π f₁ t), cos(2π f₁ t), ..., sin(2π fₙ t), cos(2π fₙ t), t]

    Providing the network with sinusoidal basis functions at the frequencies
    present in the solution directly overcomes spectral bias: the first linear
    layer can trivially combine them without having to grow the needed
    oscillations from a flat initialisation.

    Parameters
    ----------
    frequencies : sequence of float
        The frequencies (Hz) to embed.  Typically the forcing frequencies of
        the ODE or a set of Nyquist-spaced frequencies up to the expected
        bandwidth.
    """

    def __init__(self, frequencies: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.tensor(frequencies, dtype=torch.float32))

    @property
    def out_dim(self) -> int:
        return 2 * len(self.freqs) + 1

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = 2 * np.pi * self.freqs * t          # (N, n_freqs)
        return torch.cat([torch.sin(args), torch.cos(args), t], dim=1)


class PINN(nn.Module):
    """
    MLP  t (scalar) -> z(t) (n_vars-dimensional state vector).

    Parameters
    ----------
    n_vars : int
        Dimensionality of the ODE state vector.
    width : int
        Hidden layer width.
    depth : int
        Total number of layers (including input and output).
    activation : nn.Module class
        Pointwise activation.  Tanh is the default because it is smooth and
        its derivatives do not vanish on a dense set, which matters for
        autograd-based residuals.
    fourier_freqs : sequence of float or None
        If provided, prepend a FourierEmbedding at the known frequencies
        before the MLP.  This is the recommended option whenever the ODE
        solution is known to be oscillatory at specific frequencies.
    """

    def __init__(
        self,
        n_vars: int,
        width: int = 128,
        depth: int = 5,
        activation: type[nn.Module] = nn.Tanh,
        fourier_freqs: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if fourier_freqs is not None:
            self.embedding: Optional[FourierEmbedding] = FourierEmbedding(fourier_freqs)
            in_dim = self.embedding.out_dim
        else:
            self.embedding = None
            in_dim = 1

        layers: list[nn.Module] = [nn.Linear(in_dim, width), activation()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), activation()]
        layers.append(nn.Linear(width, n_vars))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.embedding(t) if self.embedding is not None else t
        return self.net(x)


def pinn_loss(
    model: PINN,
    t_col: torch.Tensor,
    rhs_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t0: torch.Tensor,
    z0: torch.Tensor,
    ic_weight: float,
    hard_ic: bool = False,
) -> tuple[torch.Tensor, float, float]:
    """
    Compute total PINN loss = ODE residual  +  ic_weight * IC loss.

    Parameters
    ----------
    hard_ic : bool
        If True, enforce the initial condition exactly via the reparametrisation
            z(t) = z0 + (t - t0) * NN(t)
    """
    t0_val = t0[0, 0]
    t_col = t_col.detach().requires_grad_(True)

    if hard_ic:
        # z(t) = z0 + (t - t0) * NN(t)  --  exactly z(t0) = z0
        z = z0 + (t_col - t0_val) * model(t_col)
        ic_val = 0.0
    else:
        z = model(t_col)

    n_vars = z.shape[1]

    # dz/dt via one autograd pass per output dimension
    dz_dt = torch.cat(
        [torch.autograd.grad(z[:, i].sum(), t_col, create_graph=True)[0]
         for i in range(n_vars)],
        dim=1,
    )                                     # (N, n_vars)

    residual = dz_dt - rhs_torch(t_col, z)
    phys = (residual ** 2).mean()

    if hard_ic:
        total = phys
    else:
        ic_loss = ((model(t0) - z0) ** 2).mean()
        ic_val = ic_loss.item()
        total = phys + ic_weight * ic_loss

    return total, phys.item(), ic_val


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_pinn(
    model: PINN,
    rhs_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t_span: tuple[float, float],
    y0: Sequence[float],
    *,
    n_iter: int = 20_000,
    n_colloc: int = 300,
    lr: float = 5e-4,
    ic_weight: float = 100.0,
    hard_ic: bool = False,
    rec_frq: int = 200,
    lr_decay: float = 0.9995,
    t_eval_np: Optional[np.ndarray] = None,
    save_snapshots: bool = False,
    verbose: bool = True,
) -> list[TrainingFrame]:

    device = next(model.parameters()).device
    t0_val, T = t_span

    y0_tensor = torch.tensor([y0], dtype=torch.float32, device=device)
    t0_tensor = torch.tensor([[t0_val]], dtype=torch.float32, device=device)

    t_eval: Optional[torch.Tensor] = None
    if t_eval_np is not None:
        t_eval = torch.tensor(t_eval_np, dtype=torch.float32, device=device).view(-1, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    frames: list[TrainingFrame] = []
    model.train()

    for it in range(n_iter + 1):
        optimizer.zero_grad()

        t_col = torch.rand(n_colloc, 1, device=device) * (T - t0_val) + t0_val
        total, phys, ic = pinn_loss(
            model, t_col, rhs_torch, t0_tensor, y0_tensor, ic_weight, hard_ic
        )
        total.backward()
        optimizer.step()
        scheduler.step()

        if it % rec_frq == 0:
            model.eval()
            pred: Optional[np.ndarray] = None
            if t_eval is not None:
                with torch.no_grad():
                    raw = model(t_eval)
                    if hard_ic:
                        raw = y0_tensor + (t_eval - t0_tensor[0, 0]) * raw
                    pred = raw.cpu().numpy()
            model.train()

            snapshot = None
            if save_snapshots:
                snapshot = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            frames.append(TrainingFrame(it, pred, total.item(), phys, ic, snapshot))

            if verbose and n_iter > 0 and it % max(1, n_iter // 5) == 0:
                print(
                    f"  iter {it:6d} | total {total.item():.3e}"
                    f" | phys {phys:.3e} | ic {ic:.3e}"
                )

    model.eval()
    return frames


def plot_spectral_dynamics(
    predictions,
    iter_nums,
    freqs,
    sample_rate,
    *,
    reference=None,
    title="Spectral Dynamics",
    iter_label="Iteration",
    save_path=None,
):
    """
    Track the amplitude at each forcing frequency across iterations.

    For each iterate in `predictions`, extracts the FFT amplitude at every
    frequency in `freqs` and plots two panels:
      - left  : amplitude at each forcing frequency over iterations
                (dashed horizontal lines = reference amplitudes when provided)
      - right : |reference_amplitude - learned_amplitude| over iterations
                (only shown when `reference` is provided)

    Parameters
    ----------
    predictions : sequence of ndarray (N_eval, n_vars)
    iter_nums   : sequence of int, same length as predictions
    freqs       : sequence of float — the frequencies to track [Hz]
    sample_rate : float — samples per second of the evaluation grid
    reference   : ndarray (N_eval, n_vars) or None — ground-truth trajectory
                  used to compute target amplitudes and amplitude errors
    """
    iter_nums = np.asarray(iter_nums)
    palette = sns.color_palette("husl", len(freqs))

    amp_curves = {f: [] for f in freqs}
    for pred in predictions:
        frqs_fft, spec = compute_fft(pred[:, 0], sample_rate)
        for f in freqs:
            amp_curves[f].append(float(spec[np.argmin(np.abs(frqs_fft - f))]))

    ref_amps = {}
    if reference is not None:
        frqs_ref, spec_ref = compute_fft(reference[:, 0], sample_rate)
        for f in freqs:
            ref_amps[f] = float(spec_ref[np.argmin(np.abs(frqs_ref - f))])

    n_panels = 2 if reference is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12)

    ax = axes[0]
    for f, col in zip(freqs, palette):
        ax.plot(iter_nums, amp_curves[f], label=f"{f} Hz", color=col)
    if ref_amps:
        for f, col in zip(freqs, palette):
            ax.axhline(ref_amps[f], color=col, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(iter_label)
    ax.set_ylabel("|FFT| at forcing frequency")
    ax.set_title("Amplitude at forcing frequencies\n(dashed = reference)")
    ax.legend(fontsize=8)

    if reference is not None:
        ax = axes[1]
        for f, col in zip(freqs, palette):
            errs = [abs(a - ref_amps[f]) for a in amp_curves[f]]
            ax.plot(iter_nums, errs, label=f"{f} Hz", color=col)
        ax.set_xlabel(iter_label)
        ax.set_ylabel("|ref − learned| amplitude")
        ax.set_title("Amplitude error at forcing frequencies")
        ax.set_yscale("log")
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_pinn_residuals(
    frames: list[TrainingFrame],
    model: "PINN",
    rhs_torch: Callable,
    t_eval: np.ndarray,
    y0: Sequence[float],
    *,
    hard_ic: bool = True,
    device=None,
) -> list[np.ndarray]:
    """
    Compute u(t;θ) = ẑ'(t;θ) − F(t, ẑ(t;θ)) for every snapshot in frames.

    Returns a list of (N, n_vars) arrays, one per frame that has a saved
    model_state.  Frames without a snapshot contribute None.
    """
    if device is None:
        device = next(model.parameters()).device

    saved_state = {k: v.clone() for k, v in model.state_dict().items()}

    t_t  = torch.tensor(t_eval, dtype=torch.float32, device=device).view(-1, 1)
    y0_t = torch.tensor([y0],   dtype=torch.float32, device=device)

    residuals: list[np.ndarray] = []
    for frame in frames:
        if frame.model_state is None:
            residuals.append(None)
            continue

        model.load_state_dict(frame.model_state)
        model.eval()

        t_in = t_t.clone().detach().requires_grad_(True)
        if hard_ic:
            z = y0_t + t_in * model(t_in)          # t0 = 0 assumed
        else:
            z = model(t_in)

        n_vars = z.shape[1]
        dz_dt = torch.cat([
            torch.autograd.grad(
                z[:, i].sum(), t_in,
                create_graph=False,
                retain_graph=(i < n_vars - 1),
            )[0]
            for i in range(n_vars)
        ], dim=1)

        with torch.no_grad():
            F_val = rhs_torch(t_in.detach(), z.detach())

        residuals.append((dz_dt.detach() - F_val).cpu().numpy())

    model.load_state_dict(saved_state)
    model.eval()
    return residuals


def plot_residual_dynamics(
    residuals: list[np.ndarray],
    iter_nums,
    t_eval: np.ndarray,
    freqs,
    sample_rate: float,
    *,
    title: str = "PINN — ODE Residual u(t;θ) Dynamics",
    save_path=None,
):
    """
    Three-panel visualisation of the ODE residual u(t;θ) = ẑ' − F across
    training, mirroring the spectral-dynamics plots of the R data-smoothing
    script.

    Panel (left) — Amplitude of |FFT(u)| at each forcing frequency vs
        training iteration.  Spectral bias appears as high frequencies
        decaying more slowly than low ones.

    Panel (centre) — Full spectrum |FFT(u)| at the final iterate, with
        forcing frequencies marked.  Shows which residual components remain.

    Panel (right) — Time-domain u(t;θ) at three snapshots (first, middle,
        last) to show how the residual waveform collapses during training.
    """
    valid = [(it, r) for it, r in zip(iter_nums, residuals) if r is not None]
    if not valid:
        return
    iters, res_list = zip(*valid)
    iters    = np.asarray(iters)
    palette  = sns.color_palette("husl", len(freqs))

    # Spectral amplitude at forcing freqs across iterations
    amp_curves = {f: [] for f in freqs}
    for r in res_list:
        frqs_k, spec_k = compute_fft(r[:, 0], sample_rate)
        for f in freqs:
            amp_curves[f].append(float(spec_k[np.argmin(np.abs(frqs_k - f))]))

    # Final spectrum
    frqs_final, spec_final = compute_fft(res_list[-1][:, 0], sample_rate)

    # Snapshot indices for time-domain panel
    snap_idx = [0, len(res_list) // 2, len(res_list) - 1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(title, fontsize=12)

    # Left — spectral dynamics
    ax = axes[0]
    for f, col in zip(freqs, palette):
        ax.plot(iters, amp_curves[f], label=f"{f} Hz", color=col, linewidth=1.2)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("|FFT(u)| at forcing frequency")
    ax.set_yscale("log")
    ax.set_title("Spectral dynamics of residual\n(spectral bias → high freq decays last)")
    ax.legend(fontsize=8)

    # Centre — final residual spectrum
    ax = axes[1]
    mask = frqs_final <= max(freqs) * 1.5
    ax.plot(frqs_final[mask], spec_final[mask], color="steelblue", linewidth=1.2)
    for f, col in zip(freqs, palette):
        ax.axvline(f, color=col, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|FFT(u)|")
    ax.set_title("Final residual spectrum")

    # Right — time-domain snapshots
    ax = axes[2]
    snap_palette = sns.color_palette("Blues_d", len(snap_idx))
    for idx, col in zip(snap_idx, snap_palette):
        it  = iters[idx]
        r   = res_list[idx]
        ax.plot(t_eval, r[:, 0], color=col, alpha=0.85, linewidth=1.1,
                label=f"iter {it}")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("t")
    ax.set_ylabel("u(t;θ)")
    ax.set_title("Residual waveform (first / mid / last)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(frames, save_path=None):
    iter_nums = [f.iter_num for f in frames]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iter_nums, [f.loss for f in frames], label="Total loss")
    ax.plot(iter_nums, [f.phys_loss for f in frames], label="Physics residual")
    if any(f.ic_loss > 0 for f in frames):
        ax.plot(iter_nums, [f.ic_loss for f in frames], label="IC loss")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("PINN Training Loss")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
