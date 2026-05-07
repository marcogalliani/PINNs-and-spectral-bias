import numpy as np


def compute_fft(signal, sample_rate):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal)) * 2.0 / n
    return freqs, spectrum


# ---------------------------------------------------------------------------
# Neural Tangent Kernel
# ---------------------------------------------------------------------------

def compute_ntk(model, t_eval, *, device=None, transform=None):
    """
    Empirical NTK matrix at the given time points.

    For a model f(t; θ) the NTK is defined as

        K[i, j] = ∑_k  (∂f/∂θ_k)(t_i) · (∂f/∂θ_k)(t_j)

    where the sum runs over all trainable parameters and the dot product is
    over output dimensions (trace of the block NTK for multi-output models).

    Under NTK theory, gradient-descent (with step size η) decays the
    projection of the error onto the k-th eigenvector of K at rate
    exp(-η λ_k t).  Spectral bias follows directly: small eigenvalues ↔
    slow convergence ↔ high-frequency eigenvectors.

    Parameters
    ----------
    model : nn.Module  (call model.eval() before passing)
    t_eval : array-like (N,) — time points
    device : torch.device or None
    transform : callable (t_tensor, model_output) -> Tensor, optional
        Apply before differentiating, e.g. for hard_ic:
            transform = lambda t, z: t * z
        The transform must be differentiable w.r.t. model_output.

    Returns
    -------
    K : ndarray (N, N), symmetric positive semi-definite
    """
    import torch

    t_eval = np.asarray(t_eval, dtype=np.float32)
    N = len(t_eval)

    if device is None:
        device = next(model.parameters()).device

    t_tensor = torch.tensor(t_eval, device=device).view(-1, 1)

    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    J = np.zeros((N, P), dtype=np.float32)

    model.eval()
    for i in range(N):
        t_i = t_tensor[i : i + 1]                    # (1, 1)
        model.zero_grad()
        out = model(t_i)                              # (1, n_vars)
        if transform is not None:
            out = transform(t_i.detach(), out)
        out.sum().backward()                          # sum over output dims
        J[i] = np.concatenate(
            [p.grad.detach().cpu().numpy().ravel() if p.grad is not None
             else np.zeros(p.numel())
             for p in params]
        )

    model.zero_grad()
    return J @ J.T                                    # (N, N)


def ntk_spectrum(model, t_eval, *, device=None, transform=None):
    """
    Eigendecomposition of the NTK matrix.

    Returns
    -------
    eigenvalues  : ndarray (N,), descending order
    eigenvectors : ndarray (N, N), columns are eigenvectors
    """
    K = compute_ntk(model, t_eval, device=device, transform=transform)
    vals, vecs = np.linalg.eigh(K)          # eigh returns ascending order
    return vals[::-1].copy(), vecs[:, ::-1].copy()


def plot_ntk_analysis(
    model,
    t_eval,
    sample_rate,
    freqs=(),
    *,
    snapshots=None,
    ntk_subsample=5,
    ntk_t_eval=None,
    device=None,
    transform=None,
    title="NTK Analysis",
    save_path=None,
):
    """
    Four-panel NTK analysis figure.

    Panel (0,0) — NTK eigenvalue spectrum at start vs end of training.
        Requires `snapshots`; if not provided, shows the current model only.

    Panel (0,1) — Relative parameter change and relative kernel change
        over training iterations.  Uses a subsampled subset of snapshots
        (every `ntk_subsample`-th) to keep computation tractable.
        Requires `snapshots`.

    Panel (1,0) — Dominant frequency of each NTK eigenvector vs its
        eigenvalue (scatter, log-scale y-axis).  The spectral-bias
        signature is a monotone decreasing trend: high frequency ↔
        small eigenvalue ↔ slow convergence.

    Panel (1,1) — FFT of the eigenvectors most associated with each
        target forcing frequency, with the corresponding eigenvalue in
        the legend.  Shows which convergence speed is attached to each
        ODE frequency component.

    Parameters
    ----------
    model : nn.Module  (the final trained model)
    t_eval : array-like (N,)
    sample_rate : float
    freqs : sequence of float — target forcing frequencies [Hz]
    snapshots : sequence of (iter_num, state_dict) or None
        Intermediate model states saved during training (e.g. from
        TrainingFrame.model_state).  Required for panels (0,0)/(0,1).
    ntk_subsample : int
        Use only every nth snapshot when computing NTK evolution.
    transform : callable (t_tensor, model_output) -> Tensor or None
        Applied before differentiating; use for hard-IC wrappers.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    t_eval = np.asarray(t_eval, dtype=np.float32)
    N = len(t_eval)

    if device is None:
        device = next(model.parameters()).device

    # ntk_t_eval: coarser grid used for all NTK computations.  Reduces cost
    # proportionally to len(ntk_t_eval) when the model is wide.  The
    # eigenvector FFTs and dominant-frequency scatter use this grid.
    t_ntk = np.asarray(ntk_t_eval, dtype=np.float32) \
            if ntk_t_eval is not None else t_eval
    N_ntk = len(t_ntk)
    sr_ntk = sample_rate * N_ntk / N   # scale sample_rate to ntk grid

    # --- Final NTK (always computed) ---
    eigenvalues_final, eigenvectors_final = ntk_spectrum(
        model, t_ntk, device=device, transform=transform
    )

    # --- Dominant frequency per final eigenvector ---
    dom_freqs = np.zeros(N_ntk)
    for k in range(N_ntk):
        frqs_k, spec_k = compute_fft(eigenvectors_final[:, k], sr_ntk)
        dom_freqs[k] = frqs_k[1 + np.argmax(spec_k[1:])] if len(frqs_k) > 1 \
                       else frqs_k[np.argmax(spec_k)]

    # --- Snapshot-based quantities ---
    eigenvalues_init = None
    evo_iters, evo_param, evo_kernel = [], [], []

    if snapshots is not None:
        saved_state = {k: v.clone() for k, v in model.state_dict().items()}

        sub = snapshots[::ntk_subsample]
        if snapshots[-1] not in sub:
            sub = list(sub) + [snapshots[-1]]

        # Initial params and kernel (first snapshot)
        _, state0 = snapshots[0]
        model.load_state_dict(state0)
        theta_0 = np.concatenate([v.cpu().numpy().ravel() for v in state0.values()])
        K_0 = compute_ntk(model, t_ntk, device=device, transform=transform)
        eigenvalues_init, _ = ntk_spectrum(model, t_ntk, device=device,
                                           transform=transform)

        for it, state in sub:
            model.load_state_dict(state)
            theta_t = np.concatenate([v.cpu().numpy().ravel() for v in state.values()])
            K_t = compute_ntk(model, t_ntk, device=device, transform=transform)

            norm0_theta = np.linalg.norm(theta_0)
            norm0_K = np.linalg.norm(K_0, "fro")
            evo_iters.append(it)
            evo_param.append(np.linalg.norm(theta_t - theta_0) / (norm0_theta + 1e-30))
            evo_kernel.append(np.linalg.norm(K_t - K_0, "fro") / (norm0_K + 1e-30))

        model.load_state_dict(saved_state)     # restore final model

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=13)
    palette = sns.color_palette("husl", max(len(freqs), 1))

    # (0,0) — Eigenvalue spectrum: initial vs final
    ax = axes[0, 0]
    idx = np.arange(1, N_ntk + 1)
    ax.semilogy(idx, eigenvalues_final, "o-", markersize=3, linewidth=1,
                label="final" if eigenvalues_init is not None else None)
    if eigenvalues_init is not None:
        ax.semilogy(idx, eigenvalues_init, "s--", markersize=3, linewidth=1,
                    color="gray", label="initial")
        ax.legend(fontsize=8)
    ax.set_xlabel("Eigenvalue index (descending)")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("NTK Eigenvalue Spectrum — initial vs final")

    # (0,1) — Relative parameter and kernel change
    ax = axes[0, 1]
    if evo_iters:
        ax.plot(evo_iters, evo_param,  "o-", markersize=4, label="||theta_t - theta_0|| / ||theta_0||")
        ax.plot(evo_iters, evo_kernel, "s--", markersize=4, label="||K_t - K_0||_F / ||K_0||_F")
        ax.set_yscale("log")
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Relative change")
        ax.set_title("Parameter and kernel stability")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Pass snapshots= to enable",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("Parameter and kernel stability")

    # (1,0) — Dominant frequency vs eigenvalue scatter
    ax = axes[1, 0]
    valid = eigenvalues_final > eigenvalues_final.max() * 1e-10
    sc = ax.scatter(dom_freqs[valid], eigenvalues_final[valid], s=8, alpha=0.6,
                    c=np.log10(eigenvalues_final[valid] + 1e-30), cmap="viridis")
    fig.colorbar(sc, ax=ax, label="log10(eigenvalue)", fraction=0.046, pad=0.04)
    for f in freqs:
        ax.axvline(f, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Dominant frequency of eigenvector [Hz]")
    ax.set_ylabel("Eigenvalue")
    ax.set_yscale("log")
    ax.set_title("Eigenvalue vs dominant frequency\n(small λ → slow convergence)")

    # (1,1) — FFT of eigenvectors closest to each target frequency
    ax = axes[1, 1]
    used = set()
    for f, col in zip(freqs, palette):
        best_k, best_dist = None, float("inf")
        for k in range(N):
            if k in used:
                continue
            d = abs(dom_freqs[k] - f)
            if d < best_dist:
                best_dist, best_k = d, k
        if best_k is None:
            continue
        used.add(best_k)
        frqs_k, spec_k = compute_fft(eigenvectors_final[:, best_k], sample_rate)
        ax.plot(frqs_k, spec_k, color=col, alpha=0.85,
                label=f"{f} Hz  λ={eigenvalues_final[best_k]:.2e}")
        ax.axvline(f, color=col, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|FFT| of eigenvector")
    ax.set_title("Eigenvectors at target frequencies\n(with associated eigenvalue)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)