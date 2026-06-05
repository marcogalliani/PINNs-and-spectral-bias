import numpy as np
import torch

# Compute Fourier Transform of the signal
def compute_fft(signal, sample_rate):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal)) * 2.0 / n
    return freqs, spectrum

# Physics blind NTK
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

# Conditioning a physics-blind NTK on a linear operator
def condition_ntk_on_operator(
    K,
    t,
    operator,
    *,
    ic_idx=None,
    residual_idx=None,
    jitter=0.0,
    project_psd=False,
):
    """
    Condition a physics-blind NTK on a linear differential operator.

    Given a physics-blind (function-space) kernel  K(t, t') — e.g. the empirical
    base NTK from `compute_ntk`, or an infinite-width Θ_∞ — and a linear operator
    L, this forms the residual / PINN kernel by applying L to *both* kernel
    arguments:

        K_RR(t, t') = L_t L_{t'} K(t, t').

    Numerical note
    --------------
    For a finite-difference operator L the resulting matrix equals  G K Gᵀ  with G
    the difference stencil (‖G‖ ~ 1/h).  Although G K Gᵀ is PSD for an *exactly*
    PSD K, the operator amplifies any non-PSD float noise in K by ~1/h²: an
    empirical NTK from `compute_ntk` (a float32 Gram, min-eig ~ −1e-2) develops
    large *spurious negative* tail eigenvalues here.  Consequences:
      * top spectrum, heatmaps and correlation slices stay accurate;
      * the small-eigenvalue tail / effective-rank tail is unreliable.
    Prefer the autodiff PINN NTK (`PINN.ntk` / `SIREN.ntk`, which differentiate the
    factored Jacobian and stay exactly PSD) for *empirical* kernels; reserve this
    routine for smooth analytic / infinite-width kernels.  `project_psd=True`
    symmetrises and clips negative eigenvalues to 0 as a damage-limiting measure
    (it zeroes the corrupted tail rather than recovering it).

    When `ic_idx` is given, the initial-condition functional z(t_{ic}) is appended
    so the result is the full Gram matrix of the PINN <-> GP equivalence,

        Gram = [[ K_RR ,  K_R·IC ],
                [ K_IC·R, K_IC·IC ]],

    with K_R·IC = L_t K(t, t_{ic}) (operator on the residual side only) and
    K_IC·IC = K(t_{ic}, t_{ic}) (physics-blind). 

    Parameters
    ----------
    K : ndarray (N, N)
        Physics-blind kernel sampled on the grid `t`, symmetric PSD.
    t : array-like (N,)
        Uniform grid the kernel is sampled on (needed by the finite-difference
        operator).
    operator : callable (f, t) -> ndarray
        The linear operator L acting on a *function* f sampled on `t`: it maps a
        1-D array f = [f(t_0), ..., f(t_{N-1})] to L f sampled on the same grid.
        `condition_ntk_on_operator` applies it to each kernel argument internally.
        Example (first-order residual operator L = ∂_t + α of z' = -α z + F):
            def operator(f, t):
                h = float(t[1] - t[0])  # uniform grid
                return np.gradient(f, h) + alpha * f
    ic_idx : int or sequence of int, optional
        Grid index (or indices) where the initial condition z(t_{ic}) is imposed.
        If None, only the residual block K_RR is returned.
    residual_idx : sequence of int, optional
        Grid indices used as residual collocation points.  Defaults to every grid
        point (the full N points).
    jitter : float, optional
        Relative diagonal jitter added for numerical stability:
        jitter * mean(diag(Gram)) * I.  Default 0.0 (no jitter).
    project_psd : bool, optional
        If True, symmetrise the result and clip negative eigenvalues to 0 before
        any jitter is added (see Numerical note).  Default False; the output is
        always symmetrised regardless.

    Returns
    -------
    Gram : ndarray
        (N_res, N_res) residual kernel K_RR if `ic_idx` is None, otherwise the
        (N_res + N_ic, N_res + N_ic) Gram matrix described above.
    """
    K = np.asarray(K, dtype=float)
    t = np.asarray(t, dtype=float)

    # Apply the function-space operator L to one kernel argument: treat every
    # slice along `axis` as a function f(t) and map it to L f(t).
    def apply_L(M, axis):
        return np.apply_along_axis(lambda f: operator(f, t), axis, M)

    # Symmetrise (always) and optionally clip the spectrum to PSD.
    def finalize(M):
        M = (M + M.T) / 2.0
        if project_psd:
            w, V = np.linalg.eigh(M)
            M = (V * np.clip(w, 0.0, None)) @ V.T
        return M

    # Residual–residual block:  K_RR = L_t L_{t'} K
    K_RL = apply_L(K, axis=1)             # L on the second argument
    K_RR = apply_L(K_RL, axis=0)          # then L on the first argument

    if residual_idx is None:
        residual_idx = np.arange(K.shape[0])
    residual_idx = np.asarray(residual_idx, dtype=int)

    A = K_RR[np.ix_(residual_idx, residual_idx)]

    # If no IC is given, return the NTK
    if ic_idx is None:
        return finalize(A)
    # Otherwise, GRAM matrix
    ic_idx = np.atleast_1d(np.asarray(ic_idx, dtype=int))

    # Residual–IC cross block:  L_t K(t, t_ic)  (operator on the residual side only)
    K_R_left = apply_L(K, axis=0)
    B = K_R_left[np.ix_(residual_idx, ic_idx)]        # (N_res, N_ic)

    # IC–IC block:  physics-blind K(t_ic, t_ic)
    C = K[np.ix_(ic_idx, ic_idx)]                      # (N_ic, N_ic)

    Gram = finalize(np.block([[A, B], [B.T, C]]))

    if jitter:
        Gram = Gram + jitter * np.mean(np.diag(Gram)) * np.eye(Gram.shape[0])

    return Gram

