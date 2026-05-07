"""
crisis_event.py
===============
Define the crisis event as: the market index return falls below its
empirical VaR at level alpha.

Because the market series is transformed to uniform space via the PIT,
the event {R_market ≤ VaR_α} is exactly equivalent to {U_market ≤ α}.
This gives a simple box constraint on the last column of the joint
uniform array, with a flat boundary and a trivial gradient.

The full uniform array has shape (T, N_sectors + 1), where the market
occupies column index `market_idx` (default: last column, N_sectors).

Exposes
-------
build_crisis_spec(U_full, alpha, market_idx)  → CrisisSpec
in_crisis_region(spec, u)                     → bool
distance_to_boundary(spec, u)                 → float  (positive = outside crisis)
boundary_gradient(spec, u)                    → np.ndarray, shape (N,)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

class CrisisSpec:
    """
    Specification of the market-index crisis event.

    Attributes
    ----------
    alpha      : float — VaR probability level (e.g. 0.05)
    N          : int   — total number of variables (sectors + market)
    market_idx : int   — column index of the market in the uniform array
    market_name: str   — name label for the market series
    """

    def __init__(self, alpha: float, N: int, market_idx: int, market_name: str = "market"):
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if not (0 <= market_idx < N):
            raise ValueError(f"market_idx={market_idx} out of range [0, {N}).")
        self.alpha = alpha
        self.N = N
        self.market_idx = market_idx
        self.market_name = market_name

    @property
    def threshold(self) -> float:
        """The crisis threshold in uniform space equals alpha by PIT definition."""
        return self.alpha

    def __repr__(self) -> str:
        return (
            f"CrisisSpec(N={self.N}, alpha={self.alpha:.3f}, "
            f"market='{self.market_name}' [col {self.market_idx}])"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_crisis_spec(
    U_full: np.ndarray,
    alpha: float = 0.05,
    market_idx: int | None = None,
    market_name: str = "market",
) -> CrisisSpec:
    """
    Build a CrisisSpec from the joint uniform array.

    The crisis event is {U_market ≤ α}, which by the PIT is equivalent to
    {R_market ≤ VaR_α(R_market)}.  The threshold in uniform space is exactly
    α — no empirical quantile estimation is needed.

    Parameters
    ----------
    U_full     : np.ndarray, shape (T, N_sectors + 1)
        Joint uniform pseudo-observations; the market column is the last by
        convention unless market_idx is specified.
    alpha      : float, optional — VaR level; default 0.05
    market_idx : int, optional  — column index of the market in U_full;
                                  defaults to the last column (N_sectors)
    market_name: str, optional  — label for the market index; default "market"

    Returns
    -------
    CrisisSpec
    """
    if U_full.ndim != 2:
        raise ValueError(f"U_full must be 2-D, got shape {U_full.shape}.")
    T, N = U_full.shape
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    if market_idx is None:
        market_idx = N - 1   # last column by convention

    empirical_fraction = float((U_full[:, market_idx] <= alpha).mean())

    return CrisisSpec(
        alpha=alpha,
        N=N,
        market_idx=market_idx,
        market_name=market_name,
    )


def in_crisis_region(spec: CrisisSpec, u: np.ndarray) -> bool:
    """
    Check whether a point satisfies the crisis constraint.

    The constraint is u[market_idx] ≤ alpha.

    Parameters
    ----------
    spec : CrisisSpec
    u    : np.ndarray, shape (N,)

    Returns
    -------
    bool — True if the market uniform coordinate is at or below alpha
    """
    u = np.asarray(u, dtype=float).ravel()
    if u.shape[0] != spec.N:
        raise ValueError(f"u has length {u.shape[0]}, expected N={spec.N}.")
    return bool(u[spec.market_idx] <= spec.alpha)


def distance_to_boundary(spec: CrisisSpec, u: np.ndarray) -> float:
    """
    Signed distance from a point to the crisis boundary.

    Positive means *outside* the crisis region (market above VaR threshold).
    Negative means *inside* the crisis region (market below VaR threshold).

    Parameters
    ----------
    spec : CrisisSpec
    u    : np.ndarray, shape (N,)

    Returns
    -------
    dist : float  — u[market_idx] - alpha
    """
    u = np.asarray(u, dtype=float).ravel()
    if u.shape[0] != spec.N:
        raise ValueError(f"u has length {u.shape[0]}, expected N={spec.N}.")
    return float(u[spec.market_idx] - spec.alpha)


def boundary_gradient(spec: CrisisSpec, u: np.ndarray) -> np.ndarray:
    """
    Gradient of the distance-to-boundary function w.r.t. u.

    Because the boundary is the flat hyperplane u[market_idx] = alpha,
    the gradient is simply the unit vector in the market dimension.
    This is used by the HMC sampler to compute the reflection normal.

    Parameters
    ----------
    spec : CrisisSpec
    u    : np.ndarray, shape (N,)  — current point (unused, included for API consistency)

    Returns
    -------
    grad : np.ndarray, shape (N,)  — unit vector with 1 at market_idx, 0 elsewhere
    """
    grad = np.zeros(spec.N, dtype=float)
    grad[spec.market_idx] = 1.0
    return grad


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    N_sectors = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    N_total = N_sectors + 1   # sectors + market
    T_test = 500
    alpha = 0.05
    rng = np.random.default_rng(7)
    U_test = rng.uniform(0.01, 0.99, size=(T_test, N_total))

    print(f"Testing crisis_event.py with T={T_test}, N_sectors={N_sectors}")
    spec = build_crisis_spec(U_test, alpha=alpha, market_name="OMXS30")
    print(f"  {spec}")
    print(f"  threshold (= alpha) = {spec.threshold:.4f}")

    # Crisis point: market coordinate (last) is well below alpha
    u_crisis = np.full(N_total, 0.5)
    u_crisis[spec.market_idx] = 0.01

    # Normal point: market coordinate above alpha
    u_normal = np.full(N_total, 0.5)
    u_normal[spec.market_idx] = 0.80

    print(f"  in_crisis(u_market=0.01): {in_crisis_region(spec, u_crisis)}")
    print(f"  in_crisis(u_market=0.80): {in_crisis_region(spec, u_normal)}")
    print(f"  dist_to_boundary(crisis): {distance_to_boundary(spec, u_crisis):.4f}")
    print(f"  dist_to_boundary(normal): {distance_to_boundary(spec, u_normal):.4f}")
    print(f"  boundary_gradient: {boundary_gradient(spec, u_crisis)}")
    print(f"  Empirical fraction in crisis: {(U_test[:, spec.market_idx] <= alpha).mean():.3f}")
