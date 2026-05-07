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

from dataclasses import dataclass
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
                    pred = model(t_eval).cpu().numpy()
            model.train()

            frames.append(TrainingFrame(it, pred, total.item(), phys, ic))

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
    max_freq_show=None,
    title="Spectral Dynamics",
    iter_label="Iteration",
    save_path=None,
):
    """
    Plot the spectrum of the first state variable across iterations.

    Works for any iterative solver that exposes a sequence of trajectories on
    the same time grid: PINN training snapshots, Picard iterates, etc.

    Parameters
    ----------
    predictions : sequence of ndarray (N_eval, n_vars)
    iter_nums : sequence of int, same length as predictions
    freqs : sequence of float — forcing frequencies, marked on the heatmaps and
        used as the per-frequency learning curves on the right panel.
    sample_rate : float — samples per second of the evaluation grid.
    """
    if max_freq_show is None:
        max_freq_show = max(freqs) * 1.8

    dynamics = []
    for pred in predictions:
        frqs, spec = compute_fft(pred[:, 0], sample_rate)
        mask = frqs <= max_freq_show
        dynamics.append(spec[mask])
    dynamics = np.array(dynamics)
    frqs_trimmed = frqs[mask]
    iter_nums = np.asarray(iter_nums)

    col_max = dynamics.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1.0
    norm_dynamics = dynamics / col_max

    cmap = sns.cubehelix_palette(8, start=0.5, rot=-0.75, reverse=True, as_cmap=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(title, fontsize=13)

    n_frames, n_freqs = dynamics.shape
    x_tick_pos = np.linspace(0, n_freqs - 1, min(10, n_freqs), dtype=int)

    for ax, data, sub_title, cbar_label in [
        (axes[0], dynamics,      "Spectrum (raw amplitude)",                                "|FFT|"),
        (axes[1], norm_dynamics, "Spectrum (column-normalised)\nRed: forcing frequencies", "normalised amplitude"),
    ]:
        im = ax.imshow(data, aspect="auto", origin="upper", cmap=cmap,
                       extent=[0, n_freqs, iter_nums[-1], iter_nums[0]])
        fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

        ax.set_xticks(x_tick_pos + 0.5)
        ax.set_xticklabels([f"{frqs_trimmed[i]:.1f}" for i in x_tick_pos], fontsize=8)
        ax.set_yticks(np.linspace(iter_nums[0], iter_nums[-1], min(5, n_frames)))
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(iter_label)
        ax.set_title(sub_title)
        for f in freqs:
            x_idx = np.argmin(np.abs(frqs_trimmed - f))
            ax.axvline(x_idx, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

    palette = sns.color_palette("husl", len(freqs))
    for f, col in zip(freqs, palette):
        idx = np.argmin(np.abs(frqs_trimmed - f))
        axes[2].plot(iter_nums, dynamics[:, idx], label=f"{f} Hz", color=col)
    axes[2].set_xlabel(iter_label)
    axes[2].set_ylabel("|FFT| at forcing frequency")
    axes[2].set_title("Per-frequency Learning Curves")
    axes[2].legend(fontsize=8)

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
