"""
crisis_event.py
===============
Define the crisis event as a tail constraint on one conditioning variable.

Two supported crisis types:

  'lower' (default) — market crash:
      The market index return falls below its VaR_alpha.
      In uniform space: u[market_idx] ≤ alpha.
      Example: S&P 500 falls below its 5th percentile return.

  'upper' — interest rate spike:
      The rate change exceeds its (1-alpha) quantile (top alpha% of moves).
      In uniform space: u[rate_idx] ≥ 1 - alpha.
      Example: 10Y Treasury yield rises by more than its 95th percentile daily change.

In both cases the boundary is a flat hyperplane in uniform space, giving
a trivial gradient (unit vector) and exact reflection in the MCMC samplers.

Exposes
-------
build_crisis_spec(U_full, alpha, crisis_idx, tail, crisis_name)  → CrisisSpec
in_crisis_region(spec, u)                                         → bool
distance_to_boundary(spec, u)                                     → float
boundary_gradient(spec, u)                                        → np.ndarray
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

class CrisisSpec:
    """
    Specification of the crisis event.

    Attributes
    ----------
    alpha       : float — tail probability level (e.g. 0.05)
    N           : int   — total variables in the vine (sectors + conditioning var)
    crisis_idx  : int   — column index of the conditioning variable in U_full
    tail        : str   — 'lower' (market crash) or 'upper' (rate spike)
    crisis_name : str   — label for the conditioning variable
    threshold   : float — boundary value in uniform space
                          'lower': alpha     (u ≤ alpha = crisis)
                          'upper': 1 - alpha (u ≥ 1-alpha = crisis)
    """

    def __init__(
        self,
        alpha: float,
        N: int,
        crisis_idx: int,
        tail: str = "lower",
        crisis_name: str = "market",
    ):
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if not (0 <= crisis_idx < N):
            raise ValueError(f"crisis_idx={crisis_idx} out of range [0, {N}).")
        if tail not in ("lower", "upper"):
            raise ValueError(f"tail must be 'lower' or 'upper', got '{tail}'.")

        self.alpha = alpha
        self.N = N
        self.crisis_idx = crisis_idx
        self.tail = tail
        self.crisis_name = crisis_name

    @property
    def threshold(self) -> float:
        """Boundary value in uniform space."""
        return self.alpha if self.tail == "lower" else 1.0 - self.alpha

    def __repr__(self) -> str:
        direction = f"≤ {self.threshold:.4f}" if self.tail == "lower" else f"≥ {self.threshold:.4f}"
        return (
            f"CrisisSpec(N={self.N}, alpha={self.alpha:.3f}, tail='{self.tail}', "
            f"'{self.crisis_name}' [col {self.crisis_idx}] {direction})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_crisis_spec(
    U_full: np.ndarray,
    alpha: float = 0.05,
    crisis_idx: int | None = None,
    tail: str = "lower",
    crisis_name: str = "market",
) -> CrisisSpec:
    """
    Build a CrisisSpec from the joint uniform array.

    For tail='lower' (market crash):
        Crisis event = { u[crisis_idx] ≤ alpha }
        Equivalent to R_market ≤ VaR_alpha(R_market) after the PIT.

    For tail='upper' (rate spike):
        Crisis event = { u[crisis_idx] ≥ 1 - alpha }
        Equivalent to Δrate ≥ Q_{1-alpha}(Δrate) after the PIT — i.e.
        the rate change is in the top alpha% of historical moves.

    Parameters
    ----------
    U_full     : np.ndarray, shape (T, N)
        Joint uniform pseudo-observations. The conditioning variable
        occupies column crisis_idx (default: last column).
    alpha      : float — tail probability level; default 0.05
    crisis_idx : int   — column of the conditioning variable; default last column
    tail       : str   — 'lower' or 'upper'; default 'lower'
    crisis_name: str   — label for the conditioning variable; default 'market'

    Returns
    -------
    CrisisSpec
    """
    if U_full.ndim != 2:
        raise ValueError(f"U_full must be 2-D, got shape {U_full.shape}.")
    T, N = U_full.shape
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    if crisis_idx is None:
        crisis_idx = N - 1

    return CrisisSpec(
        alpha=alpha,
        N=N,
        crisis_idx=crisis_idx,
        tail=tail,
        crisis_name=crisis_name,
    )


def in_crisis_region(spec: CrisisSpec, u: np.ndarray) -> bool:
    """
    Check whether a point satisfies the crisis constraint.

    lower tail: u[crisis_idx] ≤ alpha
    upper tail: u[crisis_idx] ≥ 1 - alpha

    Parameters
    ----------
    spec : CrisisSpec
    u    : np.ndarray, shape (N,)

    Returns
    -------
    bool
    """
    u = np.asarray(u, dtype=float).ravel()
    if u.shape[0] != spec.N:
        raise ValueError(f"u has length {u.shape[0]}, expected N={spec.N}.")
    val = u[spec.crisis_idx]
    if spec.tail == "lower":
        return bool(val <= spec.threshold)
    else:
        return bool(val >= spec.threshold)


def distance_to_boundary(spec: CrisisSpec, u: np.ndarray) -> float:
    """
    Signed distance from a point to the crisis boundary.

    Positive = outside the crisis region.
    Negative = inside the crisis region.

    lower tail: dist = u[crisis_idx] - alpha
    upper tail: dist = (1-alpha) - u[crisis_idx]

    Parameters
    ----------
    spec : CrisisSpec
    u    : np.ndarray, shape (N,)

    Returns
    -------
    float
    """
    u = np.asarray(u, dtype=float).ravel()
    if u.shape[0] != spec.N:
        raise ValueError(f"u has length {u.shape[0]}, expected N={spec.N}.")
    val = u[spec.crisis_idx]
    if spec.tail == "lower":
        return float(val - spec.threshold)
    else:
        return float(spec.threshold - val)


def boundary_gradient(spec: CrisisSpec, u: np.ndarray) -> np.ndarray:
    """
    Gradient of the distance-to-boundary function w.r.t. u.

    The boundary is a flat hyperplane so the gradient is a unit vector
    at the crisis_idx dimension:
      lower tail: +1 at crisis_idx (distance increases as u increases)
      upper tail: -1 at crisis_idx (distance increases as u decreases)

    Used by the HMC sampler to compute the reflection normal.

    Parameters
    ----------
    spec : CrisisSpec
    u    : np.ndarray, shape (N,)  — unused, included for API consistency

    Returns
    -------
    grad : np.ndarray, shape (N,)
    """
    grad = np.zeros(spec.N, dtype=float)
    grad[spec.crisis_idx] = 1.0 if spec.tail == "lower" else -1.0
    return grad


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    N_sectors = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    N_total = N_sectors + 1
    T_test = 500
    rng = np.random.default_rng(7)
    U_test = rng.uniform(0.01, 0.99, size=(T_test, N_total))

    for tail, name in [("lower", "^GSPC"), ("upper", "^TNX")]:
        spec = build_crisis_spec(U_test, alpha=0.05, tail=tail, crisis_name=name)
        print(spec)

        u_in  = np.full(N_total, 0.5)
        u_out = np.full(N_total, 0.5)
        u_in[spec.crisis_idx]  = 0.01 if tail == "lower" else 0.99
        u_out[spec.crisis_idx] = 0.80 if tail == "lower" else 0.20

        print(f"  in_crisis(deep in):  {in_crisis_region(spec, u_in)}")
        print(f"  in_crisis(deep out): {in_crisis_region(spec, u_out)}")
        print(f"  dist(deep in):  {distance_to_boundary(spec, u_in):.4f}")
        print(f"  dist(deep out): {distance_to_boundary(spec, u_out):.4f}")
        print(f"  gradient: {boundary_gradient(spec, u_in)}")
        frac = (U_test[:, spec.crisis_idx] <= spec.alpha if tail == 'lower'
                else U_test[:, spec.crisis_idx] >= spec.threshold)
        print(f"  Empirical fraction in crisis: {frac.mean():.3f}  (expect ~0.05)")
        print()
