"""
Misspecified-PINN spectral analysis
====================================
Python equivalent of spectral_analysis_misspec.R.

Setup (mirrors the R script exactly)
-------------------------------------
- True ODE  : z' = -alpha * z + sum_k A_k sin(2 pi f_k t)   [with forcing]
- Base model: z' = -alpha * z                                  [no forcing]
- The PINN is trained with
    physics loss : || z'(t;θ) + alpha * z(t;θ) ||²   (misspecified base ODE)
    data loss    : || z(t_obs;θ) - y_obs ||²          (observations of true traj)
- The ODE residual  u(t;θ) = z'(t;θ) + alpha * z(t;θ)
  bridges the model-data gap and should converge to the true forcing
  u_true(t) = sum_k A_k sin(2 pi f_k t).

Question
--------
Does u(t;θ) recover the true forcing frequency-by-frequency, with low
frequencies emerging first?  (Spectral bias predicts: yes.)
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

from src.PINNs import PINN, TrainingFrame, compute_pinn_residuals
from src.spectral_analysis import compute_fft

sns.set_theme(style="whitegrid")
torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_misspec_pinn(
    model,
    alpha,
    t_span,
    y0,
    t_obs_t,
    y_obs_t,
    *,
    n_iter=5_000,
    n_colloc=512,
    lr=1e-3,
    data_weight=50.0,
    lr_decay=0.9995,
    rec_frq=100,
    t_eval_t=None,
    device=None,
    verbose=True,
):
    """
    Train a PINN on the misspecified base ODE with additional data loss.

    Physics loss  : mean || z'(t_i;θ) + alpha * z(t_i;θ) ||²
    Data loss     : mean || z(t_obs_j;θ) - y_obs_j ||²
    Total loss    : physics + data_weight * data

    Hard-IC reparametrisation z(t;θ) = y0 + t * NN(t) (t0 = 0 assumed).
    """
    if device is None:
        device = next(model.parameters()).device

    t0_val, T = t_span
    y0_t = torch.tensor([y0], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    frames: list[TrainingFrame] = []
    model.train()

    for it in range(n_iter + 1):
        optimizer.zero_grad()

        # --- physics loss (misspecified base ODE) ---
        t_col = (
            torch.rand(n_colloc, 1, device=device) * (T - t0_val) + t0_val
        ).requires_grad_(True)
        z_col = y0_t + t_col * model(t_col)           # hard IC
        dz_dt = torch.autograd.grad(z_col.sum(), t_col, create_graph=True)[0]
        phys_loss = ((dz_dt - (-alpha * z_col)) ** 2).mean()

        # --- data loss ---
        z_obs = y0_t + t_obs_t * model(t_obs_t)
        data_loss = ((z_obs - y_obs_t) ** 2).mean()

        total = phys_loss + data_weight * data_loss
        total.backward()
        optimizer.step()
        scheduler.step()

        if it % rec_frq == 0:
            model.eval()
            pred = None
            if t_eval_t is not None:
                with torch.no_grad():
                    pred = (y0_t + t_eval_t * model(t_eval_t)).cpu().numpy()
            snapshot = {k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()}
            frames.append(
                TrainingFrame(it, pred, total.item(),
                              phys_loss.item(), data_loss.item(), snapshot)
            )
            if verbose and it % max(1, n_iter // 5) == 0:
                print(
                    f"  iter {it:6d} | total {total.item():.3e}"
                    f" | phys {phys_loss.item():.3e}"
                    f" | data {data_loss.item():.3e}"
                )
            model.train()

    model.eval()
    return frames


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_misspec_analysis(
    residuals,
    iter_nums,
    t_eval,
    freqs,
    amps,
    sample_rate,
    u_true,
    y_ref,
    predictions,
    t_obs,
    y_obs,
    *,
    title="Misspecified PINN — Residual Spectral Dynamics",
    save_path=None,
):
    """
    Four-panel figure mirroring the R script output.

    Panel (0,0) — State trajectory: true vs PINN fit + observation points.
    Panel (0,1) — Normalised spectral dynamics: |FFT(u)| / |u_true| at each
        forcing frequency vs training iteration.  The dashed line at 1
        marks full recovery of the forcing component.
    Panel (1,0) — Final forcing spectrum: estimated u(t;θ) vs true forcing,
        both truncated to 1.5 × max forcing frequency.
    Panel (1,1) — Time-domain forcing: u(t;θ) at three training snapshots
        (first / mid / last) overlaid on the true forcing.
    """
    valid = [(it, r) for it, r in zip(iter_nums, residuals) if r is not None]
    iters, res_list = zip(*valid)
    iters   = np.asarray(iters)
    palette = sns.color_palette("husl", len(freqs))

    # --- true forcing amplitude at each target frequency (one-sided spectrum) ---
    frqs_true, spec_true = compute_fft(u_true, sample_rate)
    true_amp = np.array([
        2.0 * float(spec_true[np.argmin(np.abs(frqs_true - f))])
        for f in freqs
    ])

    # --- normalised spectral dynamics ---
    norm_curves = {f: [] for f in freqs}
    for r in res_list:
        frqs_k, spec_k = compute_fft(r[:, 0], sample_rate)
        for f, a_true in zip(freqs, true_amp):
            amp = 2.0 * float(spec_k[np.argmin(np.abs(frqs_k - f))])
            norm_curves[f].append(amp / a_true if a_true > 1e-12 else 0.0)

    # --- final residual spectrum ---
    frqs_fin, spec_fin = compute_fft(res_list[-1][:, 0], sample_rate)
    mask = frqs_fin <= max(freqs) * 1.5

    # --- snapshot indices for time-domain panel ---
    snap_idx    = [0, len(res_list) // 2, len(res_list) - 1]
    snap_colors = sns.color_palette("Blues_d", len(snap_idx))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=13)

    # (0,0) — state trajectory
    ax = axes[0, 0]
    ax.plot(t_eval, y_ref[:, 0], "k-", linewidth=1.4, label="True y(t)")
    ax.plot(t_eval, predictions[-1][:, 0], color="steelblue",
            linewidth=1.2, alpha=0.9, label="PINN fit (final)")
    ax.scatter(t_obs, y_obs[:, 0], s=8, c="gray", alpha=0.6, zorder=3,
               label="Observations")
    ax.set_xlabel("t")
    ax.set_ylabel("z(t)")
    ax.set_title("State trajectory")
    ax.legend(fontsize=8)

    # (0,1) — normalised spectral dynamics
    ax = axes[0, 1]
    for f, col in zip(freqs, palette):
        ax.plot(iters, norm_curves[f], color=col, linewidth=1.2, label=f"{f} Hz")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("|FFT(u)| / |u_true|  at forcing freq")
    ax.set_title("Spectral dynamics of residual u(t;θ)\n"
                 "(dashed = true amplitude fully recovered)")
    ax.legend(fontsize=8)

    # (1,0) — final spectrum
    ax = axes[1, 0]
    ax.plot(frqs_true[mask], spec_true[mask], color="firebrick",
            linewidth=1.2, alpha=0.8, label="True forcing u_true(t)")
    ax.plot(frqs_fin[mask],  spec_fin[mask],  color="steelblue",
            linewidth=1.2, alpha=0.9, label="Estimated u(t;θ) — final")
    for f, col in zip(freqs, palette):
        ax.axvline(f, color=col, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|FFT|")
    ax.set_title("Final forcing spectrum: estimated vs true")
    ax.legend(fontsize=8)

    # (1,1) — time-domain forcing
    ax = axes[1, 1]
    for idx, col in zip(snap_idx, snap_colors):
        ax.plot(t_eval, res_list[idx][:, 0], color=col, alpha=0.85,
                linewidth=1.0, label=f"iter {iters[idx]}")
    ax.plot(t_eval, u_true, color="firebrick", linewidth=1.2,
            alpha=0.7, label="True u(t)")
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.set_title("Forcing — time domain\n(estimated at 3 snapshots vs true)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

    # Time grid (mirrors R: 200 pts over [0, 2])
    opt.T        = 2.0
    opt.N_POINTS = 200
    opt.OBS_STEP = 4          # observe every 4th point (50 obs) — same as R

    # PINN
    opt.WIDTH       = 128
    opt.DEPTH       = 4
    opt.NUM_ITER    = 20_000
    opt.REC_FRQ     = 50
    opt.LR          = 1e-3
    opt.N_COLLOC    = 512
    opt.DATA_WEIGHT = 1e6    # weight of observation loss vs physics loss

    if torch.cuda.is_available():
        opt.DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        opt.DEVICE = torch.device("mps")
    else:
        opt.DEVICE = torch.device("cpu")
    print(f"Device: {opt.DEVICE}")

    t_eval      = np.linspace(0, opt.T, opt.N_POINTS)
    sample_rate = opt.N_POINTS / opt.T

    # --- true forcing (the signal u_true that the misspecified model is missing) ---
    u_true = np.array([
        sum(A * np.sin(2 * np.pi * f * t + ph)
            for f, A, ph in zip(opt.freqs, opt.amps, opt.phases))
        for t in t_eval
    ])

    # --- reference trajectory (true ODE, RK45) ---
    def rhs_true(t, z):
        forcing = sum(A * np.sin(2 * np.pi * f * t + ph)
                      for f, A, ph in zip(opt.freqs, opt.amps, opt.phases))
        return [-opt.alpha * z[0] + forcing]

    print("Reference (RK45)...")
    sol = solve_ivp(rhs_true, [0, opt.T], opt.y0, t_eval=t_eval,
                    method="RK45", rtol=1e-9, atol=1e-11)
    y_ref = sol.y.T                               # (N, 1)

    # --- observation grid (every OBS_STEP-th point) ---
    obs_idx = np.arange(0, opt.N_POINTS, opt.OBS_STEP)
    t_obs   = t_eval[obs_idx]
    y_obs   = y_ref[obs_idx]                      # (N_obs, 1)

    t_obs_t = torch.tensor(t_obs, dtype=torch.float32, device=opt.DEVICE).view(-1, 1)
    y_obs_t = torch.tensor(y_obs, dtype=torch.float32, device=opt.DEVICE)
    t_eval_t = torch.tensor(t_eval, dtype=torch.float32, device=opt.DEVICE).view(-1, 1)

    # --- train misspecified PINN ---
    print(f"Training misspecified PINN ({opt.NUM_ITER} iters)...")
    model  = PINN(n_vars=1, width=opt.WIDTH, depth=opt.DEPTH).to(opt.DEVICE)
    frames = train_misspec_pinn(
        model, opt.alpha,
        t_span=(0.0, opt.T),
        y0=opt.y0,
        t_obs_t=t_obs_t,
        y_obs_t=y_obs_t,
        n_iter=opt.NUM_ITER,
        n_colloc=opt.N_COLLOC,
        lr=opt.LR,
        data_weight=opt.DATA_WEIGHT,
        rec_frq=opt.REC_FRQ,
        t_eval_t=t_eval_t,
        device=opt.DEVICE,
        verbose=True,
    )

    # --- compute residual u(t;θ) = z'(t;θ) + alpha * z(t;θ) from snapshots ---
    print("Computing residuals from snapshots...")
    rhs_misspec_torch = lambda t, z: -opt.alpha * z   # base ODE (no forcing)
    residuals = compute_pinn_residuals(
        frames, model, rhs_misspec_torch, t_eval, opt.y0,
        hard_ic=True, device=opt.DEVICE,
    )

    # --- plots ---
    print("Plotting...")
    plot_misspec_analysis(
        residuals,
        [f.iter_num for f in frames],
        t_eval,
        opt.freqs,
        opt.amps,
        sample_rate,
        u_true,
        y_ref,
        [f.prediction for f in frames],
        t_obs,
        y_obs,
        title="Misspecified PINN — Residual Spectral Dynamics",
        save_path="misspec_pinn_analysis.png",
    )

    # --- summary table ---
    print("\n=== Normalised amplitude at forcing frequencies (final iterate) ===")
    frqs_true, spec_true = compute_fft(u_true, sample_rate)
    frqs_fin,  spec_fin  = compute_fft(
        next(r for r in reversed(residuals) if r is not None)[:, 0], sample_rate
    )
    print(f"{'Freq (Hz)':<12}", end="")
    for f in opt.freqs:
        print(f"  {f} Hz", end="")
    print()
    for label, spec in [("u_true", spec_true), ("u_hat (final)", spec_fin)]:
        print(f"{label:<12}", end="")
        for f in opt.freqs:
            print(f"  {spec[np.argmin(np.abs(frqs_true - f))]:.4f}", end="")
        print()


if __name__ == "__main__":
    main()
