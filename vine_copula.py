"""
vine_copula.py
==============
Fit an R-vine copula to (T x M) uniform pseudo-observations using
pyvinecopulib with AIC-based bivariate family selection.

M = N_sectors + 1 when a market index is included as the last column.
The vine is agnostic to this — it models the full joint distribution.

Exposes
-------
fit_vine(U)                        → VineFitResult
log_density(vine_result, u)        → float
conditional_cdf(vine_result, u, j) → float
"""

from __future__ import annotations

import numpy as np
import pyvinecopulib as pv


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class VineFitResult:
    """Container for a fitted vine copula."""

    def __init__(self, vine: pv.Vinecop, N: int, family_names: list[str]):
        """
        Parameters
        ----------
        vine         : pv.Vinecop  — fitted vine object
        N            : int         — total number of variables (sectors + market)
        family_names : list[str]   — selected bivariate families, all trees
        """
        self.vine = vine
        self.N = N
        self.family_names = family_names

    def __repr__(self) -> str:
        return f"VineFitResult(N={self.N}, pairs={len(self.family_names)})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_vine(U: np.ndarray) -> VineFitResult:
    """
    Fit an R-vine copula to uniform pseudo-observations.

    The structure is selected via maximum spanning tree on Kendall's τ.
    The bivariate copula family per pair is chosen by AIC from a rich
    parametric set. For N=2 this degenerates to a single bivariate copula
    and is handled transparently by pyvinecopulib.

    Parameters
    ----------
    U : np.ndarray, shape (T, N)
        Uniform pseudo-observations in (0, 1).
        When a market index is included, it occupies column N-1.

    Returns
    -------
    VineFitResult
    """
    if U.ndim != 2:
        raise ValueError(f"U must be 2-D, got shape {U.shape}.")
    T, N = U.shape
    if N < 2:
        raise ValueError(f"Need at least 2 variables, got N={N}.")

    U = np.clip(U, 1e-6, 1 - 1e-6)

    family_set = [
        pv.BicopFamily.gaussian,
        pv.BicopFamily.student,
        pv.BicopFamily.clayton,
        pv.BicopFamily.gumbel,
        pv.BicopFamily.frank,
        pv.BicopFamily.joe,
        pv.BicopFamily.bb1,
        pv.BicopFamily.bb7,
    ]

    controls = pv.FitControlsVinecop(
        family_set=family_set,
        parametric_method="mle",
        selection_criterion="aic",
        num_threads=1,
        trunc_lvl=N - 1,
    )

    vine = pv.Vinecop.from_data(data=U, controls=controls)

    family_names: list[str] = []
    for tree_families in vine.families:
        for fam in tree_families:
            family_names.append(str(fam))

    return VineFitResult(vine=vine, N=N, family_names=family_names)


def log_density(vine_result: VineFitResult, u: np.ndarray) -> float:
    """
    Evaluate the log-density of the fitted vine at a single point.

    Parameters
    ----------
    vine_result : VineFitResult
    u           : np.ndarray, shape (N,)  — point in (0, 1)^N

    Returns
    -------
    ld : float  — log-density; -inf outside (0, 1)^N
    """
    u = np.asarray(u, dtype=float).ravel()
    N = vine_result.N
    if u.shape[0] != N:
        raise ValueError(f"u has length {u.shape[0]}, expected N={N}.")
    if np.any(u <= 0) or np.any(u >= 1):
        return -np.inf

    try:
        pdf_val = float(vine_result.vine.pdf(u.reshape(1, N))[0])
        return np.log(pdf_val) if pdf_val > 0 else -np.inf
    except Exception:
        return -np.inf


def conditional_cdf(vine_result: VineFitResult, u: np.ndarray, j: int) -> float:
    """
    Evaluate the conditional CDF for variable j via the Rosenblatt transform.

    Returns P(U_j ≤ u_j | U_1, ..., U_{j-1}) in the vine's internal ordering,
    used in the Gibbs sampler.

    Parameters
    ----------
    vine_result : VineFitResult
    u           : np.ndarray, shape (N,)
    j           : int  — 0-based variable index

    Returns
    -------
    p : float in (0, 1)
    """
    u = np.asarray(u, dtype=float).ravel()
    N = vine_result.N
    if u.shape[0] != N:
        raise ValueError(f"u has length {u.shape[0]}, expected N={N}.")
    if not (0 <= j < N):
        raise ValueError(f"j={j} out of range [0, {N}).")

    try:
        rosenblatt = vine_result.vine.rosenblatt(np.clip(u, 1e-6, 1 - 1e-6).reshape(1, N))
        p = float(rosenblatt[0, j])
    except Exception:
        p = float(u[j])

    return float(np.clip(p, 1e-8, 1 - 1e-8))


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    N_sectors = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    N_total = N_sectors + 1   # sectors + market
    T_test = 300
    rng = np.random.default_rng(0)
    U_test = rng.uniform(0.01, 0.99, size=(T_test, N_total))

    print(f"Testing vine_copula.py with T={T_test}, N_sectors={N_sectors}, N_total={N_total}")
    result = fit_vine(U_test)
    print(f"  {result}")
    print(f"  Families selected: {result.family_names}")

    u_pt = rng.uniform(0.1, 0.9, size=N_total)
    ld = log_density(result, u_pt)
    print(f"  log_density at random point: {ld:.4f}")

    cdf_j = conditional_cdf(result, u_pt, j=0)
    print(f"  conditional_cdf(j=0): {cdf_j:.4f}")

    # N=2 edge case (1 sector + market)
    print("  Testing N=2 edge case (1 sector + market)...")
    U2 = rng.uniform(0.01, 0.99, size=(200, 2))
    r2 = fit_vine(U2)
    print(f"  N=2 result: {r2}, families={r2.family_names}")
