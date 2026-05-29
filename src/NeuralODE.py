"""
NeuralODE: dz/dt = f(t, z) + NN(t, z; θ)
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.PINNs import FourierEmbedding, TrainingFrame


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
# NeuralODE / UDE model
# ---------------------------------------------------------------------------

class NeuralODEFunc(nn.Module):
    """
    Parametrised right-hand side for a NeuralODE or UDE.

    known_rhs=None  →  pure NeuralODE:  dz/dt = NN(t, z; θ)
    known_rhs=f     →  UDE:             dz/dt = f(t, z) + NN(t, z; θ)

    Parameters
    ----------
    n_vars : int
        State dimension.
    width, depth : int
        MLP hidden-layer width and total depth (including input/output layers).
    activation : nn.Module subclass
        Pointwise activation applied after every hidden layer.
    fourier_freqs : sequence of float or None
        If provided, embed t with sinusoidal features at these frequencies
        before the MLP.  Helps overcome spectral bias for oscillatory dynamics.
    known_rhs : callable(t, z) -> Tensor(N, n_vars) or None
        Mechanistic component f(t, z).  Must follow the batched convention
        (t: Tensor(N,1), z: Tensor(N,n_vars)), consistent with PINNs.py.
        None → pure NeuralODE (no physics prior).
    zero_init : bool
        Initialise the output layer weights/bias to zero.  When known_rhs is
        provided this means training starts from the mechanistic solution and
        the network only needs to learn the residual discrepancy.
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
    # Scalar interface for the RK4 / Euler integrators
    # ------------------------------------------------------------------

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        t: 0-d tensor, z: (n_vars,) → (n_vars,)

        NeuralODE (known_rhs=None): returns NN(t, z; θ)
        UDE       (known_rhs=f)   : returns f(t, z) + NN(t, z; θ)
        """
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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
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
        Train on trajectory observations.

        Loss = MSE between the integrated trajectory at observation times and
        y_obs.  The initial condition is enforced by construction (z(t0) = y0).

        Backpropagation runs only on the coarse training grid (t_obs_np), whose
        depth equals the number of observations.  The dense eval grid
        (t_eval_np) is used only under no_grad when recording frames, keeping
        per-iteration cost independent of the plotting resolution.

        Training uses Euler integration (1 NN call / step, shallow graph).
        Evaluation snapshots use RK4 (4 calls / step, higher accuracy).

        Parameters
        ----------
        t_span : (t0, T)
        y0 : initial condition, length n_vars
        t_obs_np : (N_obs,) observation times — also used as the training grid
        y_obs_np : (N_obs, n_vars) observed state values
        t_eval_np : dense evaluation grid for snapshots; defaults to `n_eval`
                    uniformly-spaced points over t_span
        n_eval : grid size when t_eval_np is None
        save_snapshots : if True, store model state_dict in each TrainingFrame
                         (enables post-hoc spectral analysis via
                         compute_ude_correction)
        verbose : print loss every n_iter/5 iterations

        Returns
        -------
        list[TrainingFrame]  — one frame per rec_frq iterations
        """
        device = next(self.parameters()).device
        t0, T  = t_span

        if t_eval_np is None:
            t_eval_np = np.linspace(t0, T, n_eval)

        # Training grid = observation times (coarse, drives backprop depth)
        t_train_t = torch.tensor(t_obs_np, dtype=torch.float32, device=device)
        y_obs_t   = torch.tensor(y_obs_np, dtype=torch.float32, device=device)
        z0_t      = torch.tensor(list(y0), dtype=torch.float32, device=device)

        # Dense eval grid (no_grad only)
        t_eval_t  = torch.tensor(t_eval_np, dtype=torch.float32, device=device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

        frames: list[TrainingFrame] = []
        self.train()  # nn.Module.train() — sets training mode

        for it in range(n_iter + 1):
            optimizer.zero_grad()

            # Euler: 1 NN call per step, 4x fewer than RK4 → fast backprop graph
            traj_train = odeint_euler(self, z0_t, t_train_t)   # (N_obs, n_vars)
            loss = ((traj_train - y_obs_t) ** 2).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if it % rec_frq == 0:
                self.eval()  # nn.Module.eval() — sets eval mode
                with torch.no_grad():
                    pred_np = odeint_rk4(self, z0_t, t_eval_t).cpu().numpy()
                self.train()

                snapshot = None
                if save_snapshots:
                    snapshot = {k: v.detach().cpu().clone()
                                for k, v in self.state_dict().items()}

                frames.append(
                    TrainingFrame(it, pred_np, loss.item(), loss.item(), 0.0, snapshot)
                )

                if verbose and n_iter > 0 and it % max(1, n_iter // 5) == 0:
                    print(f"  iter {it:6d} | loss {loss.item():.3e}")

        self.eval()
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
