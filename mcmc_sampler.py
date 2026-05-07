"""
mcmc_sampler.py
===============
Two fully self-contained MCMC samplers drawing from the vine copula density
conditioned on the market-index crisis event: u_market ≤ alpha.

Because the crisis boundary is the flat hyperplane u[market_idx] = alpha,
both samplers become significantly cleaner than the general case:

  HMC  : When a leapfrog step would push u[market_idx] above alpha, the
          momentum in the market dimension is simply negated (exact specular
          reflection off a flat boundary).  All other dimensions move freely
          subject to the Metropolis criterion.

  Gibbs: When updating dimension j ≠ market_idx, the market coordinate does
          not change, so the crisis constraint is trivially satisfied and the
          draw is always accepted.  When updating the market dimension itself,
          the grid is restricted to (0, alpha) so every draw stays in crisis.

Public API
----------
  run_sampler(vine_result, crisis_spec, sampler, N_samples, **kwargs)
      → np.ndarray, shape (N_samples, N)
        All rows satisfy u[:, market_idx] ≤ alpha.
"""

from __future__ import annotations

import numpy as np

from vine_copula import VineFitResult
from crisis_event import CrisisSpec, in_crisis_region, distance_to_boundary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vine_log_pdf(vine_result: VineFitResult, u: np.ndarray) -> float:
    """
    Log-density of the fitted vine at a single point.

    Parameters
    ----------
    vine_result : VineFitResult
    u           : np.ndarray, shape (N,)

    Returns
    -------
    float — log-density; -inf outside (0, 1)^N
    """
    N = vine_result.N
    u_c = np.clip(u, 1e-6, 1 - 1e-6)
    try:
        pdf_val = float(vine_result.vine.pdf(u_c.reshape(1, N))[0])
        return np.log(pdf_val) if pdf_val > 0 else -np.inf
    except Exception:
        return -np.inf


def _vine_log_pdf_gradient(vine_result: VineFitResult, u: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Numerical gradient of the vine log-density via central differences.

    Parameters
    ----------
    vine_result : VineFitResult
    u           : np.ndarray, shape (N,)
    eps         : float — finite difference step

    Returns
    -------
    grad : np.ndarray, shape (N,)
    """
    N = vine_result.N
    u = np.asarray(u, dtype=float).ravel()
    grad = np.zeros(N, dtype=float)

    if not np.isfinite(_vine_log_pdf(vine_result, u)):
        return grad

    for j in range(N):
        u_fwd = u.copy()
        u_bwd = u.copy()
        u_fwd[j] = min(u[j] + eps, 1 - 1e-6)
        u_bwd[j] = max(u[j] - eps, 1e-6)
        step = u_fwd[j] - u_bwd[j]
        ld_fwd = _vine_log_pdf(vine_result, u_fwd)
        ld_bwd = _vine_log_pdf(vine_result, u_bwd)
        if np.isfinite(ld_fwd) and np.isfinite(ld_bwd):
            grad[j] = (ld_fwd - ld_bwd) / (step + 1e-14)

    return np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


def _find_crisis_start(crisis_spec: CrisisSpec, rng: np.random.Generator) -> np.ndarray:
    """
    Find a starting point inside the crisis region.

    For the box constraint u[market_idx] ≤ alpha, we simply draw the market
    coordinate uniformly from (0, alpha) and the rest from (0, 1).

    Parameters
    ----------
    crisis_spec : CrisisSpec
    rng         : np.random.Generator

    Returns
    -------
    u0 : np.ndarray, shape (N,)
    """
    N = crisis_spec.N
    u0 = rng.uniform(1e-4, 1 - 1e-4, size=N)
    u0[crisis_spec.market_idx] = rng.uniform(1e-4, crisis_spec.alpha - 1e-6)
    return u0


def _gibbs_conditional_draw(
    vine_result: VineFitResult,
    u_current: np.ndarray,
    j: int,
    rng: np.random.Generator,
    grid_size: int,
    u_lo: float,
    u_hi: float,
) -> float:
    """
    Draw a new value for u_j from the conditional distribution p(u_j | u_{-j})
    evaluated on a 1-D grid over (u_lo, u_hi).

    The grid range allows the market dimension to be restricted to (0, alpha)
    while other dimensions use (0, 1).

    Parameters
    ----------
    vine_result : VineFitResult
    u_current   : np.ndarray, shape (N,)
    j           : int  — dimension to update (0-based)
    rng         : np.random.Generator
    grid_size   : int  — number of grid points
    u_lo        : float — lower bound of grid
    u_hi        : float — upper bound of grid

    Returns
    -------
    u_j_new : float in (u_lo, u_hi)
    """
    grid = np.linspace(u_lo + 1e-6, u_hi - 1e-6, grid_size)
    log_weights = np.empty(grid_size, dtype=float)

    for k, uj_val in enumerate(grid):
        u_probe = u_current.copy()
        u_probe[j] = uj_val
        log_weights[k] = _vine_log_pdf(vine_result, u_probe)

    log_weights -= np.nanmax(log_weights)
    weights = np.exp(np.where(np.isfinite(log_weights), log_weights, -np.inf))
    w_sum = weights.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        return float(u_current[j])

    weights /= w_sum
    k_drawn = rng.choice(grid_size, p=weights)
    return float(grid[k_drawn])


# ---------------------------------------------------------------------------
# HMC sampler
# ---------------------------------------------------------------------------

def _hmc_sample(
    vine_result: VineFitResult,
    crisis_spec: CrisisSpec,
    N_samples: int,
    n_leapfrog: int,
    step_size: float,
    n_warmup: int,
    seed: int,
) -> np.ndarray:
    """
    HMC sampler with exact specular reflection at the flat crisis boundary.

    The crisis boundary u[market_idx] = alpha is a flat hyperplane, so
    reflection reduces to negating the market momentum component whenever a
    leapfrog step would push u[market_idx] above alpha.

    Parameters
    ----------
    vine_result : VineFitResult
    crisis_spec : CrisisSpec
    N_samples   : int  — post-warmup samples
    n_leapfrog  : int  — leapfrog steps per proposal
    step_size   : float — initial step size (adapted during warmup)
    n_warmup    : int  — warmup iterations discarded
    seed        : int

    Returns
    -------
    samples : np.ndarray, shape (N_samples, N)
    """
    rng = np.random.default_rng(seed)
    N = crisis_spec.N
    mkt = crisis_spec.market_idx
    alpha = crisis_spec.alpha

    def potential(u: np.ndarray) -> float:
        ld = _vine_log_pdf(vine_result, u)
        return -ld if np.isfinite(ld) else np.inf

    def grad_potential(u: np.ndarray) -> np.ndarray:
        return -_vine_log_pdf_gradient(vine_result, u)

    current_u = _find_crisis_start(crisis_spec, rng)

    samples: list[np.ndarray] = []
    adapt_step = float(step_size)
    total_iter = n_warmup + N_samples

    for iteration in range(total_iter):
        p = rng.standard_normal(N)

        proposed_u = current_u.copy()
        proposed_p = p.copy()

        proposed_p -= 0.5 * adapt_step * grad_potential(proposed_u)

        for _ in range(n_leapfrog):
            u_next = proposed_u + adapt_step * proposed_p

            # Reflect off the flat boundary u[mkt] = alpha
            if u_next[mkt] > alpha:
                proposed_p[mkt] = -proposed_p[mkt]
                u_next = proposed_u + adapt_step * proposed_p

            # Reflect off the unit cube walls
            for j in range(N):
                if u_next[j] <= 0:
                    proposed_p[j] = abs(proposed_p[j])
                    u_next[j] = max(u_next[j], 1e-6)
                elif u_next[j] >= 1:
                    proposed_p[j] = -abs(proposed_p[j])
                    u_next[j] = min(u_next[j], 1 - 1e-6)

            proposed_u = u_next
            proposed_p -= adapt_step * grad_potential(proposed_u)

        proposed_p -= 0.5 * adapt_step * grad_potential(proposed_u)

        # Safety: enforce crisis constraint after leapfrog
        proposed_u[mkt] = min(proposed_u[mkt], alpha - 1e-8)
        proposed_u = np.clip(proposed_u, 1e-6, 1 - 1e-6)

        H_curr = potential(current_u) + 0.5 * float(np.dot(p, p))
        H_prop = potential(proposed_u) + 0.5 * float(np.dot(proposed_p, proposed_p))

        log_accept = -(H_prop - H_curr)
        accept_prob = min(1.0, np.exp(log_accept)) if np.isfinite(log_accept) else 0.0

        if np.log(rng.uniform()) < log_accept:
            current_u = proposed_u.copy()

        if iteration >= n_warmup:
            samples.append(current_u.copy())

        if iteration < n_warmup:
            adapt_step *= np.exp(0.05 * (accept_prob - 0.65))
            adapt_step = float(np.clip(adapt_step, 1e-4, 0.5))

    samples_arr = np.array(samples[:N_samples])
    if samples_arr.shape[0] < N_samples:
        pad = np.tile(current_u, (N_samples - samples_arr.shape[0], 1))
        samples_arr = np.vstack([samples_arr, pad])

    return samples_arr


# ---------------------------------------------------------------------------
# Gibbs sampler
# ---------------------------------------------------------------------------

def _gibbs_sample(
    vine_result: VineFitResult,
    crisis_spec: CrisisSpec,
    N_samples: int,
    n_warmup: int,
    seed: int,
    grid_size: int,
) -> np.ndarray:
    """
    Gibbs sampler exploiting the box-constraint structure.

    For dimensions j ≠ market_idx: draw from the conditional p(u_j | u_{-j})
    over the full (0, 1) interval — no crisis rejection needed because the
    market coordinate stays fixed.

    For j == market_idx: draw from the conditional restricted to (0, alpha),
    so every draw automatically satisfies the crisis constraint.

    Parameters
    ----------
    vine_result : VineFitResult
    crisis_spec : CrisisSpec
    N_samples   : int
    n_warmup    : int
    seed        : int
    grid_size   : int — grid points for 1-D conditional (default 80)

    Returns
    -------
    samples : np.ndarray, shape (N_samples, N)
    """
    rng = np.random.default_rng(seed)
    N = crisis_spec.N
    mkt = crisis_spec.market_idx
    alpha = crisis_spec.alpha

    current_u = _find_crisis_start(crisis_spec, rng)

    samples: list[np.ndarray] = []
    total_iter = n_warmup + N_samples

    for iteration in range(total_iter):
        for j in range(N):
            # Market dimension: restrict grid to (0, alpha)
            u_lo = 1e-6
            u_hi = alpha - 1e-6 if j == mkt else 1 - 1e-6

            u_j_new = _gibbs_conditional_draw(
                vine_result, current_u, j, rng, grid_size, u_lo, u_hi
            )
            current_u[j] = float(np.clip(u_j_new, u_lo, u_hi))

        if iteration >= n_warmup:
            samples.append(current_u.copy())

    samples_arr = np.array(samples[:N_samples])
    if samples_arr.shape[0] < N_samples:
        pad = np.tile(current_u, (N_samples - samples_arr.shape[0], 1))
        samples_arr = np.vstack([samples_arr, pad])

    return samples_arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sampler(
    vine_result: VineFitResult,
    crisis_spec: CrisisSpec,
    sampler: str = "gibbs",
    N_samples: int = 2000,
    n_warmup: int = 500,
    seed: int = 42,
    # HMC-specific
    n_leapfrog: int = 10,
    step_size: float = 0.05,
    # Gibbs-specific
    grid_size: int = 80,
) -> np.ndarray:
    """
    Draw samples from the joint vine density conditioned on the crisis event.

    The crisis event is u[market_idx] ≤ alpha, so all returned samples have
    their market coordinate below alpha.

    Parameters
    ----------
    vine_result : VineFitResult — fitted vine on (N_sectors + 1) variables
    crisis_spec : CrisisSpec   — built from build_crisis_spec
    sampler     : str          — 'hmc' or 'gibbs' (default 'gibbs')
    N_samples   : int          — post-warmup samples (default 2000)
    n_warmup    : int          — warmup iterations discarded (default 500)
    seed        : int          — random seed (default 42)
    n_leapfrog  : int          — HMC leapfrog steps (default 10)
    step_size   : float        — HMC initial step size (default 0.05)
    grid_size   : int          — Gibbs grid points per dimension (default 80)

    Returns
    -------
    samples : np.ndarray, shape (N_samples, N)
        N = N_sectors + 1.  Column market_idx satisfies ≤ alpha.
    """
    sampler_lower = sampler.strip().lower()
    if sampler_lower not in ("hmc", "gibbs"):
        raise ValueError(f"sampler must be 'hmc' or 'gibbs', got '{sampler}'.")

    if vine_result.N != crisis_spec.N:
        raise ValueError(
            f"vine_result.N={vine_result.N} != crisis_spec.N={crisis_spec.N}."
        )

    if sampler_lower == "hmc":
        return _hmc_sample(
            vine_result=vine_result,
            crisis_spec=crisis_spec,
            N_samples=N_samples,
            n_leapfrog=n_leapfrog,
            step_size=step_size,
            n_warmup=n_warmup,
            seed=seed,
        )
    else:
        return _gibbs_sample(
            vine_result=vine_result,
            crisis_spec=crisis_spec,
            N_samples=N_samples,
            n_warmup=n_warmup,
            seed=seed,
            grid_size=grid_size,
        )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    N_sectors = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sampler_test = sys.argv[2] if len(sys.argv) > 2 else "gibbs"
    N_total = N_sectors + 1
    T_test = 300
    N_samp = 100
    alpha = 0.05
    rng_main = np.random.default_rng(99)
    U_test = rng_main.uniform(0.01, 0.99, size=(T_test, N_total))

    print(f"Testing mcmc_sampler.py: N_sectors={N_sectors}, N_total={N_total}, sampler={sampler_test}")

    from vine_copula import fit_vine
    from crisis_event import build_crisis_spec

    vine_res = fit_vine(U_test)
    spec = build_crisis_spec(U_test, alpha=alpha, market_name="OMXS30")
    print(f"  {vine_res}")
    print(f"  {spec}")

    samples = run_sampler(
        vine_res, spec,
        sampler=sampler_test,
        N_samples=N_samp,
        n_warmup=50,
        seed=7,
        grid_size=40,
    )
    print(f"  Samples shape: {samples.shape}")
    mkt_col = samples[:, spec.market_idx]
    print(f"  Market col max: {mkt_col.max():.4f}  (must be ≤ alpha={alpha})")
    print(f"  All in crisis: {(mkt_col <= alpha).all()}")
