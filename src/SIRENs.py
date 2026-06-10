"""
SIREN (SINusoidal REpresentation Network) for first-order ODE systems  z' = f(t, z).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from PINNs import TrainingConfig, TrainingFrame


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SIRENConfig:
    n_vars: int
    width: int = 128
    depth: int = 4
    omega_0: float = 30.0   # frequency scaling for all hidden layers


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class SirenLayer(nn.Linear):
    """Single SIREN hidden layer: sin(ω₀ · Wx + b) with Sitzmann et al. init."""

    def __init__(self, in_f: int, out_f: int, omega_0: float = 30.0, is_first: bool = False):
        super().__init__(in_f, out_f)
        with torch.no_grad():
            if is_first:
                self.weight.uniform_(-1.0 / in_f, 1.0 / in_f)
            else:
                self.weight.uniform_(
                    -np.sqrt(6.0 / in_f) / omega_0,
                     np.sqrt(6.0 / in_f) / omega_0,
                )
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * super().forward(x))


class SIREN(nn.Module):
    """
    SINusoidal REpresentation Network  t (scalar) -> z(t) (n_vars-dimensional).

    Architecture: SirenLayer stack with a final linear head (no sine on output).
    """

    def __init__(self, config: SIRENConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = [
            SirenLayer(1, config.width, omega_0=config.omega_0, is_first=True)
        ]
        for _ in range(config.depth - 2):
            layers.append(SirenLayer(config.width, config.width, omega_0=config.omega_0))
        layers.append(nn.Linear(config.width, config.n_vars))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        t_col: torch.Tensor,
        rhs_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        t0: torch.Tensor,
        z0: torch.Tensor,
        ic_weight: float,
        hard_ic: bool = False,
    ) -> tuple[torch.Tensor, float, float]:
        """Compute total loss = ODE residual + ic_weight * IC loss."""
        t0_val = t0[0, 0]
        t_col = t_col.detach().requires_grad_(True)

        if hard_ic:
            z = z0 + (t_col - t0_val) * self(t_col)
            ic_val = 0.0
        else:
            z = self(t_col)

        n_vars = z.shape[1]
        dz_dt = torch.cat(
            [torch.autograd.grad(z[:, i].sum(), t_col, create_graph=True)[0]
             for i in range(n_vars)],
            dim=1,
        )

        residual = dz_dt - rhs_torch(t_col, z)
        phys = (residual ** 2).mean()

        if hard_ic:
            total = phys
        else:
            ic_loss = ((self(t0) - z0) ** 2).mean()
            ic_val = ic_loss.item()
            total = phys + ic_weight * ic_loss

        return total, phys.item(), ic_val

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _ntk_mean_diag(
        self,
        t_col: torch.Tensor,
        rhs_torch: Callable,
        t0: torch.Tensor,
        z0: torch.Tensor,
        hard_ic: bool,
        n_samples: int,
    ) -> tuple[float, float]:
        """
        Estimate mean(diag(K_ff)) and mean(diag(K_uu)) without forming the full NTK.

            mean(diag(K_ff)) = (1/N_f) Σ_i  ||∂R(t_i; θ)/∂θ||²
            mean(diag(K_uu)) = (1/N_u) Σ_j  ||∂z(t_j; θ)/∂θ||²

        Returns (mean_diag_ff, mean_diag_uu); mean_diag_uu = 1 when hard_ic=True.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        device = t_col.device
        t0_val = t0[0, 0]

        idx = torch.randperm(t_col.shape[0], device=device)[:n_samples]
        t_sub = t_col[idx].detach()

        trace_ff = 0.0
        for i in range(len(t_sub)):
            t_i = t_sub[i:i+1].requires_grad_(True)
            if hard_ic:
                z_i = z0 + (t_i - t0_val) * self(t_i)
            else:
                z_i = self(t_i)
            n_vars = z_i.shape[1]
            dz_dt = torch.cat(
                [torch.autograd.grad(z_i[:, k].sum(), t_i, create_graph=True)[0]
                 for k in range(n_vars)],
                dim=1,
            )
            R_i = dz_dt - rhs_torch(t_i, z_i)
            grads = torch.autograd.grad(R_i.sum(), params, allow_unused=True)
            trace_ff += sum(
                g.detach().pow(2).sum().item() if g is not None else 0.0
                for g in grads
            )
        mean_diag_ff = trace_ff / len(t_sub)

        if hard_ic:
            return mean_diag_ff, 1.0

        trace_uu = 0.0
        N_u = t0.shape[0]
        for i in range(N_u):
            t_i = t0[i:i+1].detach()
            z_i = self(t_i)
            grads = torch.autograd.grad(z_i.sum(), params, allow_unused=True)
            trace_uu += sum(
                g.detach().pow(2).sum().item() if g is not None else 0.0
                for g in grads
            )
        mean_diag_uu = trace_uu / N_u

        return mean_diag_ff, mean_diag_uu

    def fit(
        self,
        rhs_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        t_span: tuple[float, float],
        y0: Sequence[float],
        *,
        tr_cfg: TrainingConfig,
        t_eval_np: Optional[np.ndarray] = None,
    ) -> list[TrainingFrame]:
        """
        Train the SIREN.

        When tr_cfg.adaptive_weights=True the IC penalty weight is updated every
        ntk_update_freq iterations using the ratio of NTK trace means:
            w_new = mean(diag(K_ff)) / mean(diag(K_uu))
        An EMA with factor ntk_ema smooths successive estimates.
        """
        device = next(self.parameters()).device
        t0_val, T = t_span

        y0_tensor = torch.tensor([y0], dtype=torch.float32, device=device)
        t0_tensor = torch.tensor([[t0_val]], dtype=torch.float32, device=device)

        t_eval: Optional[torch.Tensor] = None
        if t_eval_np is not None:
            t_eval = torch.tensor(t_eval_np, dtype=torch.float32, device=device).view(-1, 1)

        optimizer = torch.optim.Adam(self.parameters(), lr=tr_cfg.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=tr_cfg.lr_decay)

        ic_weight = tr_cfg.ic_weight
        frames: list[TrainingFrame] = []
        self.train()

        for it in range(tr_cfg.n_iter + 1):
            optimizer.zero_grad()

            t_col = torch.rand(tr_cfg.n_colloc, 1, device=device) * (T - t0_val) + t0_val
            total, phys, ic = self.loss(
                t_col, rhs_torch, t0_tensor, y0_tensor, ic_weight, tr_cfg.hard_ic
            )
            total.backward()
            optimizer.step()
            scheduler.step()

            if (
                tr_cfg.adaptive_weights
                and not tr_cfg.hard_ic
                and it > 0
                and it % tr_cfg.ntk_update_freq == 0
            ):
                self.eval()
                mean_ff, mean_uu = self._ntk_mean_diag(
                    t_col, rhs_torch, t0_tensor, y0_tensor,
                    tr_cfg.hard_ic, tr_cfg.ntk_n_samples,
                )
                self.train()
                if mean_uu > 1e-30:
                    w_new = mean_ff / mean_uu
                    ic_weight = tr_cfg.ntk_ema * ic_weight + (1 - tr_cfg.ntk_ema) * w_new
                    if tr_cfg.verbose:
                        print(f"  [NTK] iter {it:6d} | ic_weight {ic_weight:.3e}"
                              f"  (raw={w_new:.3e})")

            if it % tr_cfg.rec_frq == 0:
                self.eval()
                pred: Optional[np.ndarray] = None
                if t_eval is not None:
                    with torch.no_grad():
                        raw = self(t_eval)
                        if tr_cfg.hard_ic:
                            raw = y0_tensor + (t_eval - t0_tensor[0, 0]) * raw
                        pred = raw.cpu().numpy()
                self.train()

                snapshot = None
                if tr_cfg.save_snapshots:
                    snapshot = {k: v.detach().cpu().clone()
                                for k, v in self.state_dict().items()}
                frames.append(TrainingFrame(
                    iter_num=it,
                    prediction=pred,
                    loss=total.item(),
                    phys_loss=phys,
                    ic_loss=ic,
                    model_state=snapshot,
                    ic_weight=ic_weight,
                ))

                if tr_cfg.verbose and tr_cfg.n_iter > 0 and it % max(1, tr_cfg.n_iter // 5) == 0:
                    print(
                        f"  iter {it:6d} | total {total.item():.3e}"
                        f" | phys {phys:.3e} | ic {ic:.3e}"
                        f" | w_ic {ic_weight:.3e}"
                    )

        self.eval()
        return frames

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def compute_residuals(
        self,
        frames: list[TrainingFrame],
        rhs_torch: Callable,
        t_eval: np.ndarray,
        y0: Sequence[float],
        *,
        hard_ic: bool = True,
        device=None,
    ) -> list[Optional[np.ndarray]]:
        """
        Compute u(t;θ) = ẑ'(t;θ) − F(t, ẑ(t;θ)) for every snapshot in frames.

        Returns a list of (N, n_vars) arrays; frames without a saved snapshot
        contribute None.
        """
        if device is None:
            device = next(self.parameters()).device

        saved_state = {k: v.clone() for k, v in self.state_dict().items()}

        t_t  = torch.tensor(t_eval, dtype=torch.float32, device=device).view(-1, 1)
        y0_t = torch.tensor([y0],   dtype=torch.float32, device=device)

        residuals: list[Optional[np.ndarray]] = []
        for frame in frames:
            if frame.model_state is None:
                residuals.append(None)
                continue

            self.load_state_dict(frame.model_state)
            self.eval()

            t_in = t_t.clone().detach().requires_grad_(True)
            if hard_ic:
                z = y0_t + t_in * self(t_in)
            else:
                z = self(t_in)

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

        self.load_state_dict(saved_state)
        self.eval()
        return residuals
