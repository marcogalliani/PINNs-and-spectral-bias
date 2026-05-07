import numpy as np

def euler_solve(rhs_fn, y0, times):
    ns, nv = len(times), len(y0)
    y = np.zeros((ns, nv))
    y[0] = y0
    for i in range(ns - 1):
        dt = times[i + 1] - times[i]
        y[i + 1] = y[i] + dt * np.array(rhs_fn(times[i], y[i]))
    return y


def rk4_solve(rhs_fn, y0, times):
    ns, nv = len(times), len(y0)
    y = np.zeros((ns, nv))
    y[0] = y0
    for i in range(ns - 1):
        dt = times[i + 1] - times[i]
        t = times[i]
        k1 = np.array(rhs_fn(t, y[i]))
        k2 = np.array(rhs_fn(t + dt / 2, y[i] + dt / 2 * k1))
        k3 = np.array(rhs_fn(t + dt / 2, y[i] + dt / 2 * k2))
        k4 = np.array(rhs_fn(t + dt, y[i] + dt * k3))
        y[i + 1] = y[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def picard_solve(rhs_fn, y0, times, n_iter):
    """
    Picard iteration on the integral form of the IVP:

        z^{k+1}(t) = z0 + ∫₀ᵗ f(s, z^k(s)) ds

    Iterates refine the *whole trajectory* at once, which makes them directly
    comparable to PINN training snapshots: each iterate is a function of t and
    its FFT can be tracked across iterations. For Lipschitz f on a finite
    interval the iteration converges to the unique solution.

    Parameters
    ----------
    rhs_fn : (t: float, z: ndarray) -> sequence[float]
        Pointwise RHS — same signature as for the other solvers in this module.
    y0 : sequence of float
    times : ndarray of shape (N,)
    n_iter : int
        Number of Picard iterations to perform.

    Returns
    -------
    iterates : list of ndarray, length n_iter + 1
        iterates[0] is the constant trajectory z(t) ≡ z0; iterates[k] is the
        k-th Picard refinement evaluated at `times`.
    """
    times = np.asarray(times)
    y0 = np.asarray(y0, dtype=float)
    n_steps, n_vars = len(times), len(y0)

    z = np.tile(y0, (n_steps, 1))
    iterates = [z.copy()]

    dt = np.diff(times)[:, None]
    for _ in range(n_iter):
        rhs_vals = np.array([rhs_fn(t, z[i]) for i, t in enumerate(times)])
        increments = 0.5 * (rhs_vals[:-1] + rhs_vals[1:]) * dt
        integral = np.concatenate(
            [np.zeros((1, n_vars)), np.cumsum(increments, axis=0)], axis=0
        )
        z = y0 + integral
        iterates.append(z.copy())

    return iterates