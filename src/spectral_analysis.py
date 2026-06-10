import numpy as np
import torch

# Compute Fourier Transform of the signal
def compute_fft(signal, sample_rate):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal)) * 2.0 / n
    return freqs, spectrum


# Spectral analysis of (possibly vector-valued) NTK eigenmodes
#
# For a scalar ODE an NTK eigenvector v in R^N is a time signal v(t): FFT it,
# read off the peak frequency.  For a system z(t) in R^d the NTK is (N*d, N*d),
# so an eigenvector lives in R^{N*d}; reshaped to (N, d) it is a *vector-valued*
# time signal phi(t) in R^d -- a mode carrying both a temporal profile and a
# direction in output (state) space.  Both helpers below use the flat-index
# convention of `eval_kernel_matrix` / `condition_kernel_block_autodiff`: the
# (time i, component o) entry sits at index i*d + o, i.e. `mode.reshape(N, d)`.

def mode_dominant_freq(mode, sample_rate, n_vars=1):
    """Dominant temporal frequency of an NTK eigenmode.

    Collapses the output axis by summing the per-component power spectra
    (Parseval-correct: it preserves the mode's total energy) and returns the
    peak frequency,

        P(f) = sum_o |FFT(phi_o)|^2 ,   f* = argmax_{f>0} P(f).

    Reduces to the ordinary dominant Fourier frequency when `n_vars == 1`.
    """
    M = np.asarray(mode).reshape(-1, n_vars)              # (N, d)
    N = M.shape[0]
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)
    power = (np.abs(np.fft.rfft(M, axis=0)) ** 2).sum(axis=1)   # (n_freqs,)
    return freqs[1:][np.argmax(power[1:])]                # skip DC


def mode_decompose(mode, sample_rate, n_vars=1):
    """Split an NTK eigenmode into a frequency and an output direction.

    SVD of the reshaped mode  M = sum_i s_i u_i(t) w_i^T  isolates the dominant
    *separable* component: u_1 is a scalar time profile (its peak Fourier
    frequency is returned) and w_1 is the unit direction in state space the mode
    excites.  `separability = s_1^2 / sum_i s_i^2` in (0, 1] reports how rank-1
    the mode is -- ~1 means a single (frequency, direction) describes it, which
    holds exactly when `n_vars == 1` and at initialisation (NTK = K_scalar ⊗ I).

    Returns
    -------
    freq : float            dominant frequency of the leading time profile u_1
    direction : ndarray (d,)  unit vector w_1 in output/state space
    separability : float    s_1^2 / sum_i s_i^2
    """
    M = np.asarray(mode).reshape(-1, n_vars)              # (N, d)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    freqs = np.fft.rfftfreq(M.shape[0], d=1.0 / sample_rate)
    power = np.abs(np.fft.rfft(U[:, 0])) ** 2
    freq = freqs[1:][np.argmax(power[1:])]
    separability = float(S[0] ** 2 / (S ** 2).sum()) if S.size else 0.0
    return freq, Vt[0], separability


# Physics blind NTK
def compute_ntk(model, t_eval, *, device=None, transform=None):
    """
    Empirical NTK matrix at the given time points.

    For a model f(t; θ) the NTK is defined as

        K[i, j] = ∑_k  (∂f/∂θ_k)(t_i) · (∂f/∂θ_k)(t_j)

    where the sum runs over all trainable parameters and the dot product is
    over output dimensions (trace of the block NTK for multi-output models).
    
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

# Infinite-width NTK
# Utilities to compute infinite-width kernels
# Expectations, computed using Gaussian-Hermite quadrature
# E[f(u)f(v)] where (u,v) ~ N(0,Cov)
#
# Reparametrisation trick: (u,v) are reparametrised in terms of 
# independent variables (x,y)
# u = sigma_u * x
# v = sigma_v (rho * x + sqrt(1 - rho^2) * y)

# tanh activation function
def tanh_expectations(Cov, n_gh_pts = 20):
    """
    E[tanh(u)tanh(v)] and 
    E[tanh'(u)tanh'(v)] 
    for (u,v)~N(0,Λ), elementwise over a grid.
    
    Parameters:
    -----------
    - Cov: np.ndarray
        covariance matrix of preactivations, preactivations are assumed
        centred in 0
    """
    var = np.diag(Cov)
    # convert to correlation matrix
    sd = np.sqrt(np.clip(var, 1e-30, None))
    rho = np.clip(Cov / np.outer(sd, sd), -1.0, 1.0)
    # pearson coefficients
    comp = np.sqrt(np.clip(1.0 - rho ** 2, 0.0, None))
    Ess = np.zeros_like(Cov) # init E[tanh(u)tanh(v)]
    Edd = np.zeros_like(Cov) # init E[tanh'(u)tanh'(v)]
    # Gaussian-Hermite quadrature: points and weights
    ghx, ghw = np.polynomial.hermite.hermgauss(n_gh_pts)
    for p in range(n_gh_pts):
        u = sd[:, None] * ghx[p]; 
        tu = np.tanh(u); du = 1.0 - tu ** 2 # activ. and derivative
        for q in range(n_gh_pts):
            v = sd[None, :] * (rho * ghx[p] + comp * ghx[q]); 
            w = ghw[p] * ghw[q]
            Ess += w * tu * np.tanh(v)
            Edd += w * du * (1.0 - np.tanh(v) ** 2)
    return (Ess + Ess.T) / 2.0, (Edd + Edd.T) / 2.0

# sin activation function (SIREN)
# here the analytical solution is used (no GH quadrature needed)
def sin_expectations(Cov):
    """
    Expectations:
    E[sin u sin v]=e^{-(a+b)/2}sinh(c) and
    E[sin'u sin'v]=E[cos u cos v]=e^{-(a+b)/2}cosh(c)
    
    where a = Var(u), b = Var(v), c = Cov(u,v)
    
    Parameters:
    -----------
    - Cov: np.ndarray
        covariance matrix of preactivations, preactivations are assumed
        centred in 0
    """
    var = np.diag(Cov)
    
    a = var[:, None]; b = var[None, :]; c = Cov
    half = (a + b) / 2.0
    Ess = 0.5 * (np.exp(c - half) - np.exp(-c - half))
    Ecc = 0.5 * (np.exp(c - half) + np.exp(-c - half))
    return Ess, Ecc

def infinite_width_ntk(t_grid, depth, expectation_fn, sigma_w2=1.0, sigma_b2=0.2):
    """Deterministic NTK of an infinitely wide MLP
    
    Parameters:
    ----------
    - t: 
        time grid
    - depth: float
        depth of the MLP
    - expectation_fn: 
        function returning E[f(u)f(v)] and E[f'(u)f'(v)] for a specific
        activation function f
    - sigma_w2
    - sigma_b2
    """
    X = np.asarray(t_grid, dtype=float).reshape(-1, 1)
    Sigma = sigma_w2 * (X @ X.T) + sigma_b2          # Σ^(1)
    Theta = Sigma.copy()                              # Θ^(1)
    for _ in range(depth - 1):
        Ess, Edd = expectation_fn(Sigma)
        Sigma = sigma_w2 * Ess + sigma_b2            # Σ^(l+1)
        Theta = Sigma + Theta * (sigma_w2 * Edd)     # Θ^(l+1)
    return Theta


def infinite_width_ntk_fourier(t_grid, depth, expectation_fn, ff_freqs, sigma_w2=0.5, sigma_b2=0.05):
    """Deterministic NTK of an infinitely wide MLP
    
    Parameters:
    ----------
    - t: 
        time grid
    - depth: float
        depth of the MLP
    - expectation_fn: 
        function returning E[f(u)f(v)] and E[f'(u)f'(v)] for a specific
        activation function f
    - ff_freqs
    - sigma_w2
    - sigma_b2
    """
    tt = np.asarray(t_grid, dtype=float)
    ff_embeds = [np.sin(2*np.pi*f*tt) for f in ff_freqs] + [np.cos(2*np.pi*f*tt) for f in ff_freqs] + [tt]
    Gram = np.stack(ff_embeds, 1)
    # Covariance of the mapped features
    Sigma = sigma_w2 * (Gram @ Gram.T) + sigma_b2          # Σ^(1)
    # Same as standard infinite width kernel
    Theta = Sigma.copy()                              # Θ^(1)
    for _ in range(depth - 1):
        Ess, Edd = expectation_fn(Sigma)
        Sigma = sigma_w2 * Ess + sigma_b2            # Σ^(l+1)
        Theta = Sigma + Theta * (sigma_w2 * Edd)     # Θ^(l+1)
    return Theta

def infinite_width_ntk_siren(t_grid, depth, expectation_fn, sigma_w2=600.0, sigma_b2=300.0, gw_hidden = 2.0):
    """Deterministic NTK of an infinitely wide MLP
    
    Parameters:
    ----------
    - t: 
        time grid
    - depth: float
        depth of the MLP
    - expectation_fn: 
        function returning E[f(u)f(v)] and E[f'(u)f'(v)] for a specific
        activation function f
    - sigma_w2
    - sigma_b2
    - gw_hidden
    """
    X = np.asarray(t_grid, dtype=float).reshape(-1, 1)
    Sigma = sigma_w2 * (X @ X.T) + sigma_b2 
    Theta = Sigma.copy()                              # Θ^(1)
    for _ in range(depth - 2): #last layer is linear
        Ess, Edd = expectation_fn(Sigma)
        Sigma = gw_hidden * Ess
        Theta = Sigma + Theta * (gw_hidden * Edd)     # Θ^(l+1)
    Ess, _ = expectation_fn(Sigma)
    return Ess + Theta


# ---------------------------------------------------------------------------
# Autodiff operator conditioning (differentiable kernel functions)
# ---------------------------------------------------------------------------
# The routines above sample a kernel on a grid as a *matrix*; applying a
# differential operator to it then requires finite differences (see the
# numerical note in `condition_ntk_on_operator`, which amplifies non-PSD float
# noise by ~1/h^2).  The tools below instead express the infinite-width kernel
# as a *differentiable function* k(t, t') in torch, so the operator
# L_t L_{t'} k can be applied exactly by autodiff -- no finite-difference noise.
#
# All three infinite-width recursions are pointwise-separable: k(a, b) depends
# on the pair (a, b) only through the running scalars Σ(a,a), Σ(b,b), Σ(a,b)
# and Θ(a,b).  We therefore evaluate a scalar k(a, b), differentiate it w.r.t.
# each argument, and vmap over the grid.

import math
from numpy.polynomial.hermite_e import hermeval
from torch.func import grad, vmap, jacrev, functional_call

# --- Mehler / Hermite-series basis for the tanh expectations ---------------
# A 2-D Gauss-Hermite quadrature of E[φ(u)φ(v)] needs the reparametrisation
# v = σ_v(ρ x + √(1-ρ²) y), whose √(1-ρ²) term has an infinite derivative at
# ρ=1 (the diagonal t=t').  Differentiating it (the autodiff operator) exposes
# that branch point and destroys PSD-ness.  Instead we expand the activation in
# probabilists' Hermite polynomials,  φ(σξ) = Σ_k c_k(σ)/k! · He_k(ξ),  and use
# Mehler's identity  E[He_k(ξ)He_l(η)] = δ_kl k! ρ^k  to get
#     E[φ(u)φ(v)] = Σ_k c_k(σ_u) c_k(σ_v) ρ^k / k!,
# a *polynomial in ρ* — smooth through ρ=1, no √(1-ρ²).

def tanh_expectation_scalar(var_u, var_v, cov_uv, n_gh_nodes = 40, hermite_terms = 12):
    """E[tanh u tanh v] and E[tanh'u tanh'v] for (u,v)~N(0,Cov).

    Pointwise torch analogue of `tanh_expectations`, via the smooth Mehler
    expansion (correctly normalised; the matrix version drops the 1/√π·√2
    Gauss-Hermite factors).  Differentiable through ρ=1.
    """
    
    sdu = torch.sqrt(torch.clamp(var_u, min=1e-30))
    sdv = torch.sqrt(torch.clamp(var_v, min=1e-30))
    
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_gh_nodes)
    # Hermite polynomials expansion
    he_xi = torch.tensor(np.sqrt(2.0) * gh_x, dtype=torch.float64)         # ξ nodes
    he_w = torch.tensor(gh_w / np.sqrt(np.pi), dtype=torch.float64)        # E_{N(0,1)} weights
    he_basis = torch.tensor(                                                # He_k(ξ_p), (K, G)
        np.stack([hermeval(np.sqrt(2.0) * gh_x,
                        np.eye(hermite_terms)[k]) for k in range(hermite_terms)]),
        dtype=torch.float64,
    )
    he_fact = torch.tensor([math.factorial(k) for k in range(hermite_terms)], dtype=torch.float64)

    def hermite_coeffs(sigma, deriv):
        """Hermite coefficients c_k(σ) = E[φ(σξ) He_k(ξ)] for φ=tanh (or tanh')."""
        xi = he_xi.to(sigma.dtype)
        phi = (1.0 - torch.tanh(sigma * xi) ** 2) if deriv else torch.tanh(sigma * xi)
        return (he_w.to(sigma.dtype)[None, :] * he_basis.to(sigma.dtype) * phi[None, :]).sum(dim=1)
    
    rho = cov_uv / (sdu * sdv)
    rho_pow = rho ** torch.arange(hermite_terms, dtype=var_u.dtype)
    fact = he_fact.to(var_u.dtype)
    Ess = (hermite_coeffs(sdu, False) * hermite_coeffs(sdv, False) * rho_pow / fact).sum()
    Edd = (hermite_coeffs(sdu, True) * hermite_coeffs(sdv, True) * rho_pow / fact).sum()
    return Ess, Edd

def sin_expectation_scalar(var_u, var_v, cov_uv):
    """E[sin u sin v] and E[cos u cos v] for (u,v)~N(0,[[a,c],[c,b]])."""

    half = (var_u + var_v) / 2.0
    Ess = 0.5 * (torch.exp(cov_uv - half) - torch.exp(-cov_uv - half))
    Ecc = 0.5 * (torch.exp(cov_uv - half) + torch.exp(-cov_uv - half))
    return Ess, Ecc


def infinite_width_ntk_mlp_fn(depth, *, exp_scalar_fn = tanh_expectation_scalar, sigma_w2=1.0, sigma_b2=0.2, gw_hidden=2.0):
    """Differentiable scalar infinite-width NTK  k(a, b) -> Θ_∞(a, b). Returns a closure suitable for autodiff (e.g. `condition_kernel_autodiff`) and vmap.
    """
    def sigma1(a, b):
        return sigma_w2 * a * b + sigma_b2

    
    def kernel(a, b):
        saa, sbb, sab = sigma1(a, a), sigma1(b, b), sigma1(a, b)
        tab = sab
        for _ in range(depth - 1):
            Ess, Edd = exp_scalar_fn(saa, sbb, sab)
            Eaa, _   = exp_scalar_fn(saa, saa, saa)
            Ebb, _   = exp_scalar_fn(sbb, sbb, sbb)
            sab = sigma_w2 * Ess + sigma_b2
            saa = sigma_w2 * Eaa + sigma_b2
            sbb = sigma_w2 * Ebb + sigma_b2
            tab = sab + tab * (sigma_w2 * Edd)
        return tab

    return kernel

def infinite_width_ntk_siren_fn(depth, *, exp_scalar_fn = sin_expectation_scalar, sigma_w2=600.0, sigma_b2=300.0, gw_hidden=2.0):
    """Differentiable scalar infinite-width NTK  k(a, b) -> Θ_∞(a, b). Returns a closure suitable for autodiff (e.g. `condition_kernel_autodiff`) and vmap.
    """   
    def sigma1(a, b):
        return sigma_w2 * a * b + sigma_b2

    def kernel(a, b):
        saa, sbb, sab = sigma1(a, a), sigma1(b, b), sigma1(a, b)
        tab = sab
        for _ in range(depth - 2):              # last layer is linear
            Ess, Edd = exp_scalar_fn(saa, sbb, sab)
            Eaa, _   = exp_scalar_fn(saa, saa, saa)
            Ebb, _   = exp_scalar_fn(sbb, sbb, sbb)
            sab, saa, sbb = gw_hidden * Ess, gw_hidden * Eaa, gw_hidden * Ebb
            tab = sab + tab * (gw_hidden * Edd)
        Ess, _ = exp_scalar_fn(saa, sbb, sab)
        return Ess + tab
        
    return kernel

def infinite_width_ntk_fourier_fn(depth, *, ff_freqs=None, exp_scalar_fn = tanh_expectation_scalar,
                          sigma_w2=0.5, sigma_b2=0.05, gw_hidden=2.0):
    """Differentiable scalar infinite-width NTK for a fourier architecture k(a, b) -> Θ_∞(a, b). Returns a closure suitable for autodiff (e.g. `condition_kernel_autodiff`) and vmap.
    """
    freqs = torch.as_tensor(np.asarray(ff_freqs, dtype=float))
    
    def sigma1(a, b):
        f = freqs.to(a.dtype)
        return sigma_w2 * (torch.cos(2 * math.pi * f * (a - b)).sum() + a * b) + sigma_b2

    def kernel(a, b):
        saa, sbb, sab = sigma1(a, a), sigma1(b, b), sigma1(a, b)
        tab = sab
        for _ in range(depth - 1):
            Ess, Edd = exp_scalar_fn(saa, sbb, sab)
            Eaa, _   = exp_scalar_fn(saa, saa, saa)
            Ebb, _   = exp_scalar_fn(sbb, sbb, sbb)
            sab = sigma_w2 * Ess + sigma_b2
            saa = sigma_w2 * Eaa + sigma_b2
            sbb = sigma_w2 * Ebb + sigma_b2
            tab = sab + tab * (sigma_w2 * Edd)
        return tab

    return kernel


def eval_kernel_matrix(kernel_fn, t_grid, *, dtype=torch.float64):
    """Sample a kernel on the grid -> numpy Gram matrix.

    Scalar kernel:  k(a, b) -> (N, N).  
    Matrix-valued (block) kernel: k(a, b) -> (d, d)  ->  (N*d, N*d), 
    with flat index (time i, component o) mapped to i*d + o 
    (so an eigenvector reshapes as `v.reshape(N, d)`).
    """
    t = torch.as_tensor(np.asarray(t_grid), dtype=dtype)
    K = vmap(vmap(kernel_fn, in_dims=(None, 0)), in_dims=(0, None))(t, t)
    K = K.detach().cpu().numpy()
    if K.ndim == 2:                                   # scalar kernel -> (N, N)
        return K
    N, _, d, _ = K.shape                              # block kernel (N, N, d, d)
    return K.transpose(0, 2, 1, 3).reshape(N * d, N * d)


def condition_kernel_autodiff(kernel_fn, t_grid, operator, *, dtype=torch.float64):
    """Residual kernel  K_RR = L_t L_{t'} k(t, t')  via autodiff, applies L exactly.

    Parameters
    ----------
    kernel_fn : callable(a, b) -> scalar tensor
        Differentiable (torch) kernel, e.g. from `infinite_width_ntk_fn`.
    operator : callable(g, s) -> scalar, optional
        Applies the linear operator L to a scalar function g at the point s.

    """
    t = torch.as_tensor(np.asarray(t_grid), dtype=dtype)

    def LL(a, b):
        # L on the second argument (as a function of the first), then L on the first
        Lb = lambda x: operator(lambda s: kernel_fn(x, s), b)
        return operator(Lb, a)

    K = vmap(vmap(LL, in_dims=(None, 0)), in_dims=(0, None))(t, t)
    K = (K + K.T) / 2.0
    return K.detach().cpu().numpy()


def empirical_ntk_fn(model, *, transform=None, dtype=torch.float64):
    """Differentiable empirical NTK  k(a, b) = ⟨∂z(a)/∂θ, ∂z(b)/∂θ⟩  of `model`.

    The finite-network counterpart of `infinite_width_ntk_fn`: it returns a
    differentiable scalar kernel k(a, b) (the physics-blind NTK, traced over
    outputs).  

    The model is used only to *build* k. Works for any nn.Module t -> z(t).
    """
    # capture on CPU (the autodiff conditioner builds CPU grids; NTK is small)
    params  = {n: p.detach().cpu().to(dtype) for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n: b.detach().cpu().to(dtype) for n, b in model.named_buffers()}

    def z(p, t):                                   # scalar t -> (n_vars,)
        out = functional_call(model, (p, buffers), (t.reshape(1, 1),))
        if transform is not None:
            out = transform(t.reshape(1, 1), out)
        return out.reshape(-1)

    jac = jacrev(z, argnums=0)                      # ∂z/∂θ as a param-pytree

    def kernel(a, b):
        Ja, Jb = jac(params, a), jac(params, b)
        # sum over output dim and all parameters -> trace NTK
        return sum((Ja[n] * Jb[n]).sum() for n in Ja)

    return kernel


# ---------------------------------------------------------------------------
# Multi-output (system of ODEs) NTK: block kernels  K(a, b)_{o,o'}
# ---------------------------------------------------------------------------
# The scalar kernels above trace over the output dimension, collapsing a system
# z(t) in R^d to a single (N, N) matrix.  For a *coupled* system z' = A z + F
# the residual operator L = (d/dt) I - A mixes the output components, so the
# cross-output blocks K(a,b)_{o,o'} (o != o') must be retained.  The functions
# below keep the full (d, d) block; `eval_kernel_matrix` assembles them into the
# (N*d, N*d) Gram and `condition_kernel_block_autodiff` applies L exactly.

def empirical_ntk_block_fn(model, *, transform=None, dtype=torch.float64):
    """Differentiable empirical NTK block  K(a,b)_{o,o'} = ⟨∂z_o(a)/∂θ, ∂z_{o'}(b)/∂θ⟩.

    Multi-output analogue of `empirical_ntk_fn`: instead of tracing over the
    output dimension it returns the full (d, d) block for a system z(t) in R^d
    (a (1, 1) block when d == 1).  Plugs into `eval_kernel_matrix` and
    `condition_kernel_block_autodiff`.
    """
    params  = {n: p.detach().cpu().to(dtype) for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n: b.detach().cpu().to(dtype) for n, b in model.named_buffers()}

    def z(p, t):                                   # scalar t -> (d,)
        out = functional_call(model, (p, buffers), (t.reshape(1, 1),))
        if transform is not None:
            out = transform(t.reshape(1, 1), out)
        return out.reshape(-1)

    jac = jacrev(z, argnums=0)                      # ∂z/∂θ as a param-pytree

    def kernel(a, b):
        Ja, Jb = jac(params, a), jac(params, b)
        # Ja[n], Jb[n]: (d, *param_shape); contract parameter axes -> (d, d)
        return sum(
            torch.tensordot(Ja[n].reshape(Ja[n].shape[0], -1),
                            Jb[n].reshape(Jb[n].shape[0], -1),
                            dims=([1], [1]))
            for n in Ja
        )

    return kernel


def as_block_kernel(scalar_kernel_fn, n_vars):
    """Lift a scalar kernel k(a, b) to the (d, d) block  k(a, b) * I_d.

    At initialisation the multi-output NTK of an MLP with independent output
    heads is K_scalar(t, t') ⊗ I_d (block-diagonal, no cross-output coupling).
    Wrapping an infinite-width scalar kernel (`infinite_width_ntk_fn`) this way
    lets it reuse `eval_kernel_matrix` / `condition_kernel_block_autodiff`; the
    coupling then enters solely through the residual operator A.
    """
    def kernel(a, b):
        return scalar_kernel_fn(a, b) * torch.eye(n_vars, dtype=a.dtype)
    return kernel


def condition_kernel_block_autodiff(kernel_fn, t_grid, A=None, *, dtype=torch.float64):
    """Residual (PINN) kernel of a first-order ODE *system*  z' = A z + F.

    Multi-output analogue of `condition_kernel_autodiff`.  The residual operator
    is matrix-valued,  L = (d/dt) I_d - A, so for a block base kernel
    M(a, b) = [K(a,b)_{o,o'}] (e.g. from `empirical_ntk_block_fn`, or a scalar
    infinite-width kernel lifted by `as_block_kernel`) the conditioned block is

        K_RR(a, b) = ∂_a∂_b M − (∂_a M) Aᵀ − A (∂_b M) + A M Aᵀ,

    i.e. L applied to each kernel argument (the cross-output blocks of M are
    essential here — tracing them away loses the coupling).  `A` is the (d, d)
    system matrix; ``A=None`` means L = (d/dt) I (decoupled).  Returns the
    symmetrised (N*d, N*d) Gram, flat index (time i, component o) -> i*d + o.
    """
    t = torch.as_tensor(np.asarray(t_grid), dtype=dtype)
    d = kernel_fn(t[0], t[0]).shape[0]
    A_t = (torch.zeros((d, d), dtype=dtype) if A is None
           else torch.as_tensor(np.asarray(A), dtype=dtype))

    def LL(a, b):
        M   = kernel_fn(a, b)                                           # (d, d)
        dMa = jacrev(lambda x: kernel_fn(x, b))(a)                      # ∂_a M
        dMb = jacrev(lambda y: kernel_fn(a, y))(b)                      # ∂_b M
        d2M = jacrev(lambda x: jacrev(lambda y: kernel_fn(x, y))(b))(a) # ∂_a∂_b M
        return d2M - dMa @ A_t.T - A_t @ dMb + A_t @ M @ A_t.T

    K = vmap(vmap(LL, in_dims=(None, 0)), in_dims=(0, None))(t, t)
    K = K.detach().cpu().numpy()                                        # (N, N, d, d)
    N = K.shape[0]
    K = K.transpose(0, 2, 1, 3).reshape(N * d, N * d)
    return (K + K.T) / 2.0