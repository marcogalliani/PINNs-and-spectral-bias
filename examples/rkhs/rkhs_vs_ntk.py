# How close is a PINN's kernel to the reproducing kernel of the PDE operator?
#
# Problem: the 1-D Helmholtz boundary-value problem on [0, L]
#   u''(x) + k^2 u(x) = f(x),    u(0) = u(L) = 0   (homogeneous Dirichlet).
#
# We compare two kernels, both built around the linear operator  L = D^2 + k^2:
#
#   * K_RKHS  -- the *reproducing kernel* of L with the boundary conditions, i.e.
#     the kernel whose RKHS norm is the physics penalty  J(u) = \int |L u|^2.
#     Built from the Dirichlet Green's function:  K_RKHS = (L^2)^{-1} = \int G G.
#   * K_PINN  -- the *PINN kernel* of each architecture: its NTK CONDITIONED on
#     the linear operator,  K_RR(x,y) = L_x L_y Theta(x,y)  (the residual NTK that
#     governs the physics-loss training dynamics).  Shown for a tanh MLP and a
#     SIREN, at finite (empirical) and infinite width.
#
# Enforcing the boundaries (every kernel respects u(0)=u(L)=0).
#   - K_RKHS uses the *two-sided* Dirichlet Green's function of L,
#       G(x,s) = sin(k x_<) sin(k (L - x_>)) / (k sin kL),   x_< = min, x_> = max,
#     so  K_RKHS(0,.) = K_RKHS(L,.) = 0.  (Contrast the oscillator IVP, whose
#     causal Green's function vanished only at the initial time.)
#   - K_PINN uses the hard-constraint ansatz  u_hat(x) = D(x) NN(x),  D(x)=x(L-x),
#     which vanishes at both ends.  D is theta-independent, so the base NTK is
#     D(x) D(y) Theta(x,y);  L = D^2 + k^2 is then applied to each argument
#     exactly by autodiff (condition_kernel_autodiff), giving the residual kernel.
#
# What the comparison reveals.  K_RKHS = (L^2)^{-1} is RESONANT: its eigenvalues
# ~ 1/(k^2 - (n pi/L)^2)^2 peak at the Dirichlet mode n pi/L ~ k, so it concentrates
# on functions oscillating AT the operator frequency k.  The PINN residual kernel
# K_RR = L_x L_y Theta carries the opposite tilt -- the operator multiplier
# (k^2 - (2 pi f)^2)^2 *suppresses* the resonance at k and *amplifies* high
# frequencies -- so where the conditioned kernel concentrates is set by the
# architecture's own spectrum: a tanh MLP stays stuck at the lowest mode (spectral
# bias), a SIREN pushes to high frequency.  Off-the-shelf, neither reproduces the
# operator's resonant kernel; the architecture must be frequency-matched to k.
from config import ROOT_DIR

import copy

import numpy as np
import torch
from torch.func import grad

from PINNs import PINN, PINNConfig
from SIRENs import SIREN, SIRENConfig
from spectral_analysis import (
    empirical_ntk_fn,
    infinite_width_ntk_mlp_fn,
    infinite_width_ntk_siren_fn,
    tanh_expectation_scalar,
    sin_expectation_scalar,
    condition_kernel_autodiff,
    mode_dominant_freq,
)

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def corr(K):
    """Correlation form  K[i,j] / sqrt(K[i,i] K[j,j])  in [-1, 1] (0-diag safe)."""
    d = np.sqrt(np.clip(np.diag(K), 0.0, None))
    d = np.where(d <= 0.0, 1.0, d)
    return K / np.outer(d, d)


def eff_rank(K):
    """Participation ratio (sum w)^2 / sum w^2 of the correlation-form spectrum."""
    w = np.linalg.eigvalsh(corr(K))
    w = w[w > 1e-10]
    return float(w.sum() ** 2 / (w ** 2).sum())


def spectrum_freqs(K, sample_rate):
    """(eigenvalues desc, dominant freq of each eigenmode) for positive eigenvalues."""
    vals, vecs = np.linalg.eigh(K)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    keep = vals > vals.max() * 1e-6
    freqs = np.array([mode_dominant_freq(vecs[:, j], sample_rate, n_vars=1)
                      for j in np.where(keep)[0]])
    return vals[keep], freqs


def dirichlet_green(x_grid, k, L):
    """Dirichlet Green's function matrix G(x_i, x_j) of  L = D^2 + k^2  on [0, L]."""
    x = np.asarray(x_grid, float)
    lo, hi = np.minimum(x[:, None], x[None, :]), np.maximum(x[:, None], x[None, :])
    return np.sin(k * lo) * np.sin(k * (L - hi)) / (k * np.sin(k * L))


def bc_wrap(kernel_fn, L):
    """Lift a base kernel k(a,b) to the boundary-constrained NTK  D(a) D(b) k(a,b),
    D(x) = x (L - x).  D is theta-independent, so this is exactly the empirical NTK
    of the hard-constrained network  u_hat = D * NN  -- and applies to inf-width too.
    """
    def kernel(a, b):
        return a * (L - a) * b * (L - b) * kernel_fn(a, b)
    return kernel


def u_hat(model, x, L):
    """Hard-constraint ansatz  u(x) = x (L - x) NN(x)  (u(0) = u(L) = 0)."""
    return x * (L - x) * model(x)


def train_siren_pinn(model, k2, f_fn, L, *, n_iter=5000, n_colloc=128, lr=1e-3, seed=0):
    """Train the hard-BC SIREN as a PINN: minimise the Helmholtz residual
    ||u'' + k^2 u - f||^2 over random collocation points (boundaries are exact)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_iter):
        opt.zero_grad()
        xc = (torch.rand(n_colloc, 1) * L).requires_grad_(True)
        u = u_hat(model, xc, L)
        u_x = torch.autograd.grad(u.sum(), xc, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), xc, create_graph=True)[0]
        loss = ((u_xx + k2 * u - f_fn(xc)) ** 2).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
def main():
    # ---SETUP---
    torch.manual_seed(42)
    np.random.seed(42)
    sns.set_theme(style="whitegrid")

    FIGURES_DIR = ROOT_DIR / "figures/rkhs_vs_ntk"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---PROBLEM: Helmholtz BVP  u'' + k^2 u = f,  u(0)=u(L)=0---
    L_DOM = 1.0
    K_WAVE = 4.5 * np.pi                     # non-resonant (sin kL != 0)
    F0 = K_WAVE / (2 * np.pi)                # operator spatial frequency [cyc/unit]
    K2 = K_WAVE ** 2

    N_GRID = 120
    x_grid = np.linspace(0.0, L_DOM, N_GRID, dtype=np.float64)
    dx = x_grid[1] - x_grid[0]
    sample_rate = N_GRID / L_DOM             # for eigenmode frequency analysis

    # Reference BVP solution (u'' + k^2 u = 0, u(0)=0, u(L)=1): u = sin(kx)/sin(kL)
    x_plot = np.linspace(0.0, L_DOM, 400)
    u_ref = np.sin(K_WAVE * x_plot) / np.sin(K_WAVE * L_DOM)

    # ---RESIDUAL OPERATOR  L = D^2 + k^2  (applied exactly by autodiff)---
    def operator(g, s):
        return grad(grad(g))(s) + K2 * g(s)

    # ---THE KERNELS (all enforce the Dirichlet boundaries)---
    print(f"Building kernels on a {N_GRID}-point grid ...")

    # (1) operator reproducing kernel with Dirichlet BCs:  K_RKHS = \int G G ds
    G = dirichlet_green(x_grid, K_WAVE, L_DOM)
    K_rkhs = (G @ G.T) * dx                  # PSD, resonant at k, vanishes at 0, L

    # (2..) PINN kernel = boundary-constrained NTK CONDITIONED on L, per architecture
    WIDTH, DEPTH = 24, 3
    SIREN_OMEGA = 7.0
    mlp   = PINN(PINNConfig(n_vars=1, width=WIDTH, depth=DEPTH))
    siren = SIREN(SIRENConfig(n_vars=1, width=WIDTH, depth=DEPTH, omega_0=SIREN_OMEGA))

    BASES = {
        r"$K_{\mathrm{tanh}}^{\mathrm{emp}}$":  empirical_ntk_fn(mlp),
        r"$K_{\mathrm{tanh}}^{\infty}$":        infinite_width_ntk_mlp_fn(
            DEPTH, exp_scalar_fn=tanh_expectation_scalar),
        r"$K_{\mathrm{SIREN}}^{\mathrm{emp}}$": empirical_ntk_fn(siren),
        r"$K_{\mathrm{SIREN}}^{\infty}$":       infinite_width_ntk_siren_fn(
            DEPTH, exp_scalar_fn=sin_expectation_scalar),
    }

    KERNELS = {r"$K_{\mathrm{RKHS}}$": K_rkhs}
    for name, base in BASES.items():
        print(f"  conditioning {name} on L = D^2 + k^2 ...", flush=True)
        KERNELS[name] = condition_kernel_autodiff(bc_wrap(base, L_DOM), x_grid, operator)

    STYLE = {                                # (color, linestyle, lw)
        r"$K_{\mathrm{RKHS}}$":                 ("black",      "-",  2.2),
        r"$K_{\mathrm{tanh}}^{\mathrm{emp}}$":  ("darkorange", "-",  1.5),
        r"$K_{\mathrm{tanh}}^{\infty}$":        ("darkorange", "--", 1.5),
        r"$K_{\mathrm{SIREN}}^{\mathrm{emp}}$": ("crimson",    "-",  1.5),
        r"$K_{\mathrm{SIREN}}^{\infty}$":       ("crimson",    "--", 1.5),
    }

    # ---QUANTITATIVE SUMMARY: where does each kernel concentrate in frequency?---
    # top-mode freq:  frequency of the leading eigenmode  (RKHS resonates at ~ k)
    # centroid:       energy-weighted mean eigenmode frequency
    print(f"\nOperator spatial frequency  k/2pi = {F0:.2f} cycles/unit")
    print(f"{'kernel':24s} {'eff_rank':>9s} {'top-mode f':>11s} {'centroid f':>11s}")
    for name, K in KERNELS.items():
        vals, freqs = spectrum_freqs(K, sample_rate)
        top_f = freqs[np.argmax(vals)]
        centroid = float((vals * freqs).sum() / vals.sum())
        print(f"{name:24s} {eff_rank(K):9.2f} {top_f:11.2f} {centroid:11.2f}")

    # ---HEATMAPS: correlation form---
    fig, axes = plt.subplots(1, len(KERNELS), figsize=(3.6 * len(KERNELS), 4.0))
    fig.suptitle("Helmholtz: operator reproducing kernel vs PINN kernel "
                 "(NTK conditioned on L)  --  correlation form", fontsize=13)
    for ax, (name, K) in zip(axes, KERNELS.items()):
        im = ax.imshow(corr(K), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto",
                       extent=[0, L_DOM, L_DOM, 0])
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("x'")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kernel_heatmaps.png", dpi=150, bbox_inches="tight")

    # ---SLICES at x* = L/2 (correlation-normalised, in [-1, 1])---
    star = np.argmin(np.abs(x_grid - L_DOM / 2))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.suptitle(r"Kernel slices  $\tilde K(x, x^\star)$  at $x^\star = L/2$", fontsize=12)
    for name, K in KERNELS.items():
        color, ls, lw = STYLE[name]
        ax.plot(x_grid, corr(K)[:, star], label=name, color=color, ls=ls, lw=lw, alpha=0.9)
    ax.axvline(x_grid[star], color="gray", ls=":", lw=1.0)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set(xlabel="x", ylim=(-1.05, 1.05),
           title="K_RKHS resonates at the operator frequency k")
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kernel_slices.png", dpi=150, bbox_inches="tight")

    # ---SPECTRUM vs eigenmode frequency + reference solution---
    # K_RKHS peaks at k (resonance); the conditioned NTKs concentrate where the
    # architecture has power -- tanh low (spectral bias), SIREN high.
    fig, (ax_sol, ax_spec) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_sol.plot(x_plot, u_ref, color="steelblue")
    ax_sol.set(xlabel="x", ylabel="u(x)",
               title=f"Helmholtz BVP solution  (k/2pi = {F0:.2f} cyc/unit)")

    for name, K in KERNELS.items():
        color, ls, lw = STYLE[name]
        vals, freqs = spectrum_freqs(K, sample_rate)
        order = np.argsort(freqs)
        ax_spec.semilogy(freqs[order], vals[order] / vals.max(), label=name,
                         color=color, ls=ls, lw=lw, alpha=0.85)
    ax_spec.axvline(F0, color="gray", ls="--", lw=1.0, label=f"k/2pi = {F0:.2f}")
    ax_spec.set(xlim=(0, sample_rate / 2),
                xlabel="Dominant frequency of eigenmode [cycles/unit]",
                ylabel="Eigenvalue / max (log)", title="Spectrum vs eigenmode frequency")
    ax_spec.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "spectra.png", dpi=150, bbox_inches="tight")

    # ---TRAIN THE SIREN PINN (feature-learning regime), then inspect its NTK-----
    # The width-24 SIREN is narrow enough to leave the lazy/kernel regime: training
    # moves the parameters enough to *deform* the NTK.  Manufactured solution
    # u*(x) = sin(4 pi x) (Dirichlet, near the resonance n pi ~ k), with forcing
    # f = u*'' + k^2 u* = (k^2 - (4 pi)^2) sin(4 pi x).  Question: does feature
    # learning warp the empirical PINN kernel toward the resonant operator kernel?
    m_star = 4
    def f_fn(x):
        return (K2 - (m_star * np.pi) ** 2) * torch.sin(m_star * np.pi * x)
    u_star = np.sin(m_star * np.pi * x_grid)

    print("\nTraining SIREN PINN (hard-BC ansatz, feature-learning regime) ...")
    siren_trained = copy.deepcopy(siren)         # identical init to K_SIREN^emp
    train_siren_pinn(siren_trained, K2, f_fn, L_DOM)
    with torch.no_grad():
        u_fit = u_hat(siren_trained,
                      torch.as_tensor(x_grid, dtype=torch.float32).view(-1, 1),
                      L_DOM).numpy().ravel()
    print(f"  solution fit RMSE = {np.sqrt(np.mean((u_fit - u_star) ** 2)):.2e}")

    print("  empirical NTK of the trained PINN (conditioned on L) ...", flush=True)
    K_init = KERNELS[r"$K_{\mathrm{SIREN}}^{\mathrm{emp}}$"]   # at initialisation
    K_tr = condition_kernel_autodiff(bc_wrap(empirical_ntk_fn(siren_trained), L_DOM),
                                     x_grid, operator)
    drift = (np.linalg.norm(corr(K_tr) - corr(K_init))
             / np.linalg.norm(corr(K_init)))
    v0, f0s = spectrum_freqs(K_init, sample_rate)
    v1, f1s = spectrum_freqs(K_tr, sample_rate)
    c0, c1 = (v0 * f0s).sum() / v0.sum(), (v1 * f1s).sum() / v1.sum()
    print(f"  NTK drift  ||corr_trained - corr_init|| / ||corr_init|| = {drift:.3f}"
          f"   (large => feature learning, NTK deformed)")
    print(f"  centroid f:  init {c0:.2f}  ->  trained {c1:.2f}   (k/2pi = {F0:.2f})")

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    fig.suptitle(rf"Feature-learning SIREN PINN ($\omega_0$={SIREN_OMEGA:g}, width={WIDTH}): "
                 rf"NTK deforms under training (drift={drift:.2f})", fontsize=12)
    axes[0].plot(x_grid, u_star, "k--", lw=1.5, label=r"$u^\star=\sin(4\pi x)$")
    axes[0].plot(x_grid, u_fit, color="crimson", lw=1.6, label="trained PINN")
    axes[0].set(xlabel="x", ylabel="u(x)", title="solution fit")
    axes[0].legend(fontsize=8)
    for ax, K, ttl in [(axes[1], K_init, "init"), (axes[2], K_tr, "after training")]:
        im = ax.imshow(corr(K), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto",
                       extent=[0, L_DOM, L_DOM, 0])
        ax.set_title(rf"$\tilde K_{{\mathrm{{SIREN}}}}$ {ttl}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("x'")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for (v, f), c, lab in [((v0, f0s), "lightcoral", "init"),
                           ((v1, f1s), "crimson", "trained")]:
        o = np.argsort(f)
        axes[3].semilogy(f[o], v[o] / v.max(), color=c, lw=1.5, label=f"SIREN {lab}")
    vr, fr = spectrum_freqs(K_rkhs, sample_rate); o = np.argsort(fr)
    axes[3].semilogy(fr[o], vr[o] / vr.max(), color="black", lw=2.0, label=r"$K_{\mathrm{RKHS}}$")
    axes[3].axvline(F0, color="gray", ls="--", lw=1.0)
    axes[3].set(xlim=(0, sample_rate / 2), ylim=(1e-4, 2),
                xlabel="eigenmode frequency [cycles/unit]", ylabel="eigenvalue / max (log)",
                title="spectrum: init vs trained vs RKHS")
    axes[3].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "siren_trained_ntk.png", dpi=150, bbox_inches="tight")

    print(f"\nFigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
