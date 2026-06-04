import numpy as np
import torch

def compute_fft(signal, sample_rate):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal)) * 2.0 / n
    return freqs, spectrum


# ---------------------------------------------------------------------------
# Neural Tangent Kernel (generic / physics-blind)
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
             for p in params] #p.grad computes df/dp for single output models
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
