"""
mcmc_sampler.py
===============
Two samplers drawing from the vine copula density conditioned on the
market-index crisis event: u_market ≤ alpha.

  rejection (default, recommended):
      Simulates directly from the fitted vine with pyvinecopulib's fast C++
      backend and discards rows where u[crisis_idx] is outside the crisis
      region.  Exact, no convergence issues, no grid resolution artifacts.
      Expected acceptance rate ≈ alpha, so ~100x oversampling for alpha=0.01.

  mwg (Metropolis-within-Gibbs):
      Cycles through each variable with a scalar Metropolis-Hastings step.
      Crisis variable: uniform proposal over the crisis region (symmetric,
      exact for any alpha).  Sector variables: reflected Gaussian random walk
      with step sizes adapted during warmup to ~44% acceptance.
      Slower than rejection but useful for MCMC convergence diagnostics.

Public API
----------
  run_sampler(vine_result, crisis_spec, sampler, N_samples, **kwargs)
      → np.ndarray, shape (N_samples, N)
        All rows satisfy the crisis constraint.
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
    idx = crisis_spec.crisis_idx
    if crisis_spec.tail == "lower":
        u0[idx] = rng.uniform(1e-4, crisis_spec.alpha - 1e-6)
    else:
        u0[idx] = rng.uniform(crisis_spec.threshold + 1e-6, 1 - 1e-4)
    return u0


def _reflect_into_unit_interval(x: float) -> float:
    """Fold x back into (0, 1) by reflection at 0 and 1."""
    x = abs(x)
    while x > 1.0:
        x = abs(2.0 - x)
    return float(np.clip(x, 1e-6, 1 - 1e-6))


# ---------------------------------------------------------------------------
# Rejection sampler (recommended)
# ---------------------------------------------------------------------------

def _rejection_sample(
    vine_result: VineFitResult,
    crisis_spec: CrisisSpec,
    N_samples: int,
    seed: int,
    max_batches: int = 500,
) -> np.ndarray:
    """
    Sample from the vine conditioned on the crisis region via rejection sampling.

    Uses pyvinecopulib's C++ vine.simulate() for fast batch generation, then
    discards rows outside the crisis region.  Exact and statistically correct;
    no convergence issues, no grid resolution artifacts.

    Expected acceptance rate ≈ alpha (e.g. 5% for alpha=0.05), so each batch
    of size ≈ N_samples/alpha rows is drawn until N_samples accepted.

    Parameters
    ----------
    vine_result : VineFitResult
    crisis_spec : CrisisSpec
    N_samples   : int
    seed        : int
    max_batches : int — safety cap on simulation rounds

    Returns
    -------
    samples : np.ndarray, shape (N_samples, N)
        All rows satisfy the crisis constraint.
    """
    rng = np.random.default_rng(seed)
    idx = crisis_spec.crisis_idx
    tail = crisis_spec.tail
    threshold = crisis_spec.threshold
    alpha = crisis_spec.alpha

    batch_size = max(int(N_samples / alpha * 4), 10_000)

    collected: list[np.ndarray] = []
    total_accepted = 0

    for _ in range(max_batches):
        vine_seed = [int(rng.integers(0, 2**31))]
        u_batch = vine_result.vine.simulate(batch_size, seeds=vine_seed)

        if tail == "lower":
            mask = u_batch[:, idx] <= threshold
        else:
            mask = u_batch[:, idx] >= threshold

        accepted = u_batch[mask]
        if len(accepted) > 0:
            collected.append(accepted)
            total_accepted += len(accepted)

        if total_accepted >= N_samples:
            break

    if total_accepted == 0:
        raise RuntimeError(
            f"Rejection sampling: 0 accepted in {max_batches} batches. "
            f"alpha={alpha}, tail='{tail}', threshold={threshold:.4f}."
        )

    all_samples = np.vstack(collected)

    if total_accepted < N_samples:
        idx_rs = rng.integers(0, total_accepted, size=N_samples)
        return all_samples[idx_rs]

    return all_samples[:N_samples]


# ---------------------------------------------------------------------------
# Metropolis-within-Gibbs sampler
# ---------------------------------------------------------------------------

def _mwg_sample(
    vine_result: VineFitResult,
    crisis_spec: CrisisSpec,
    N_samples: int,
    n_warmup: int,
    seed: int,
    sigma_init: float = 0.15,
) -> np.ndarray:
    """
    Metropolis-within-Gibbs: cycle through each variable with a scalar MH step.

    Crisis variable (u_market):
        Propose uniformly from the crisis region (0, threshold) or (threshold, 1).
        The proposal is symmetric so the MH ratio is just the vine pdf ratio.
        This correctly samples from the vine conditional restricted to the crisis
        region — no grid resolution issues, exact for any alpha.

    Sector variables (j ≠ crisis_idx):
        Propose u_j' = reflect(u_j + N(0, sigma_j^2)) into (0, 1).
        Reflection at 0 and 1 keeps the proposal symmetric so the MH ratio is
        again the pure pdf ratio.  sigma_j is adapted during warmup to keep
        per-variable acceptance rates near 0.44 (optimal for 1-D MH).

    Parameters
    ----------
    vine_result : VineFitResult
    crisis_spec : CrisisSpec
    N_samples   : int  — post-warmup samples to collect
    n_warmup    : int  — adaptation / burn-in iterations (discarded)
    seed        : int
    sigma_init  : float — initial random-walk step size for sector variables

    Returns
    -------
    samples : np.ndarray, shape (N_samples, N)
        All rows satisfy the crisis constraint.
    """
    rng = np.random.default_rng(seed)
    N = crisis_spec.N
    idx = crisis_spec.crisis_idx
    tail = crisis_spec.tail
    threshold = crisis_spec.threshold
    target_rate = 0.44

    current_u = _find_crisis_start(crisis_spec, rng)
    current_ld = _vine_log_pdf(vine_result, current_u)

    # Per-variable step sizes (crisis variable doesn't use sigma)
    sigmas = np.full(N, sigma_init, dtype=float)

    # Running accept counts for adaptation (reset every adapt_interval)
    adapt_interval = 50
    accept_counts = np.zeros(N, dtype=float)

    samples: list[np.ndarray] = []
    total_iter = n_warmup + N_samples

    for iteration in range(total_iter):
        for j in range(N):
            u_prop = current_u.copy()

            if j == idx:
                # Uniform proposal inside the crisis region — symmetric, exact
                if tail == "lower":
                    u_prop[j] = rng.uniform(1e-6, threshold - 1e-6)
                else:
                    u_prop[j] = rng.uniform(threshold + 1e-6, 1 - 1e-6)
            else:
                # Reflected Gaussian random walk — symmetric proposal
                u_prop[j] = _reflect_into_unit_interval(
                    current_u[j] + rng.normal(0.0, sigmas[j])
                )

            prop_ld = _vine_log_pdf(vine_result, u_prop)
            log_a = prop_ld - current_ld

            if np.log(rng.uniform()) < log_a:
                current_u = u_prop
                current_ld = prop_ld
                accept_counts[j] += 1

        # Dual-averaging step-size adaptation during warmup
        if iteration < n_warmup and (iteration + 1) % adapt_interval == 0:
            rates = accept_counts / adapt_interval
            for j in range(N):
                if j != idx:
                    sigmas[j] *= np.exp(rates[j] - target_rate)
                    sigmas[j] = float(np.clip(sigmas[j], 1e-4, 1.0))
            accept_counts[:] = 0.0

        if iteration >= n_warmup:
            samples.append(current_u.copy())

    return np.array(samples[:N_samples])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sampler(
    vine_result: VineFitResult,
    crisis_spec: CrisisSpec,
    sampler: str = "rejection",
    N_samples: int = 2000,
    n_warmup: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Draw samples from the joint vine density conditioned on the crisis event.

    Parameters
    ----------
    vine_result : VineFitResult — fitted vine on (N_sectors + 1) variables
    crisis_spec : CrisisSpec   — built from build_crisis_spec
    sampler     : str          — 'rejection' (default) or 'mwg'
    N_samples   : int          — post-warmup samples (default 2000)
    n_warmup    : int          — MWG warmup iterations (default 500)
    seed        : int          — random seed (default 42)

    Returns
    -------
    samples : np.ndarray, shape (N_samples, N)
        All rows satisfy the crisis constraint on crisis_spec.crisis_idx.
    """
    sampler_lower = sampler.strip().lower()
    if sampler_lower not in ("rejection", "mwg", "gibbs", "hmc"):
        raise ValueError(f"sampler must be 'rejection' or 'mwg', got '{sampler}'.")

    if vine_result.N != crisis_spec.N:
        raise ValueError(
            f"vine_result.N={vine_result.N} != crisis_spec.N={crisis_spec.N}."
        )

    if sampler_lower == "rejection":
        return _rejection_sample(
            vine_result=vine_result,
            crisis_spec=crisis_spec,
            N_samples=N_samples,
            seed=seed,
        )
    else:
        # 'mwg', 'gibbs', 'hmc' all route to MWG (gibbs/hmc kept as aliases)
        return _mwg_sample(
            vine_result=vine_result,
            crisis_spec=crisis_spec,
            N_samples=N_samples,
            n_warmup=n_warmup,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    N_sectors = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sampler_test = sys.argv[2] if len(sys.argv) > 2 else "mwg"
    N_total = N_sectors + 1
    T_test = 300
    N_samp = 200
    alpha = 0.05
    rng_main = np.random.default_rng(99)
    U_test = rng_main.uniform(0.01, 0.99, size=(T_test, N_total))

    print(f"Testing mcmc_sampler.py: N_sectors={N_sectors}, N_total={N_total}, sampler={sampler_test}")

    from vine_copula import fit_vine
    from crisis_event import build_crisis_spec

    vine_res = fit_vine(U_test)
    spec = build_crisis_spec(U_test, alpha=alpha, crisis_name="OMXS30")
    print(f"  {vine_res}")
    print(f"  {spec}")

    samples = run_sampler(
        vine_res, spec,
        sampler=sampler_test,
        N_samples=N_samp,
        n_warmup=100,
        seed=7,
    )
    print(f"  Samples shape: {samples.shape}")
    mkt_col = samples[:, spec.crisis_idx]
    print(f"  Market col max: {mkt_col.max():.4f}  (must be ≤ alpha={alpha})")
    print(f"  All in crisis: {(mkt_col <= alpha).all()}")
    print(f"  Sector col means: {samples[:, :N_sectors].mean(axis=0).round(3)}")
