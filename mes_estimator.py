"""
mes_estimator.py
================
Estimate Marginal Expected Shortfall (MES) for each of N_sectors sector
indices from MCMC samples drawn from the joint vine conditioned on the
crisis event.

MES_j = E[R_j | crisis]

The MCMC samples have shape (S, N_sectors + 1).  The last column is the
conditioning variable (market or rate); it is NOT included in the MES output.
Only the first N_sectors columns are reported.

Back-transformation uses the empirical quantile function of each sector's
historical return series, which bounds MES to the observed data range and
avoids parametric tail extrapolation.

Exposes
-------
estimate_mes(samples_U, returns, market, alpha, batch_size) → pd.DataFrame
    DataFrame with columns [sector, MES, CI_lower, CI_upper], shape (N_sectors, 4)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


# ---------------------------------------------------------------------------
# Back-transformation
# ---------------------------------------------------------------------------

def _back_transform_column(u_col: np.ndarray, returns_series: np.ndarray) -> np.ndarray:
    """
    Transform a column of uniform MCMC samples to the return scale using the
    empirical quantile function of the historical return series.

    This maps each u in [0, 1] to the historical return at that quantile,
    bounding MES by the observed data range and avoiding parametric
    tail extrapolation that produces unrealistic values when u ≈ 0 or 1.

    Parameters
    ----------
    u_col          : np.ndarray, shape (S,)  — uniform MCMC samples
    returns_series : np.ndarray, shape (T,)  — historical return series

    Returns
    -------
    returns_col : np.ndarray, shape (S,)
    """
    u_col = np.clip(u_col, 1e-6, 1 - 1e-6)
    return np.quantile(returns_series, u_col)


# ---------------------------------------------------------------------------
# Batch-means CI
# ---------------------------------------------------------------------------

def _batch_means_ci(samples: np.ndarray, batch_size: int, conf: float = 0.95) -> tuple[float, float]:
    """
    95% CI for E[samples] using batch means (accounts for MCMC serial correlation).

    Parameters
    ----------
    samples    : np.ndarray, shape (S,)
    batch_size : int
    conf       : float

    Returns
    -------
    (ci_lower, ci_upper) : tuple of floats
    """
    S = len(samples)
    n_batches = max(S // batch_size, 2)
    truncated = samples[: n_batches * batch_size]
    batch_means = truncated.reshape(n_batches, batch_size).mean(axis=1)
    grand_mean = float(batch_means.mean())
    se = float(np.sqrt(batch_means.var(ddof=1) / n_batches))

    from scipy.stats import t as _t
    t_crit = float(_t.ppf((1 + conf) / 2, df=max(n_batches - 1, 1)))
    return (grand_mean - t_crit * se, grand_mean + t_crit * se)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_mes(
    samples_U: np.ndarray,
    returns: pd.DataFrame,
    market: pd.Series,
    alpha: float = 0.05,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Estimate Marginal Expected Shortfall for each of N_sectors sector indices.

    MES_j = E[R_j | R_market ≤ VaR_α(R_market)]

    GARCH-skewed-t marginals are re-fitted independently here.

    Parameters
    ----------
    samples_U  : np.ndarray, shape (S, N_sectors + 1)
        MCMC samples in uniform space.  The last column corresponds to the
        market index; the first N_sectors columns correspond to the sectors.
    returns    : pd.DataFrame, shape (T, N_sectors)
        Original daily return series for the N sector indices.
    market     : pd.Series, shape (T,)
        Original daily return series for the market index (e.g., OMXS30).
    alpha      : float — VaR level used to define the crisis event
    batch_size : int   — batch size for batch-means CI (default 100)

    Returns
    -------
    result : pd.DataFrame with columns [sector, MES, CI_lower, CI_upper]
        Shape (N_sectors, 4).  MES values are in the same units as `returns`.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if not isinstance(market, pd.Series):
        raise TypeError("market must be a pandas Series.")
    if samples_U.ndim != 2:
        raise ValueError(f"samples_U must be 2-D, got shape {samples_U.shape}.")

    T, N_sectors = returns.shape
    S, N_total = samples_U.shape

    if N_total != N_sectors + 1:
        raise ValueError(
            f"samples_U has {N_total} columns but expected N_sectors+1={N_sectors+1}. "
            f"The last column must be the market index."
        )

    # Back-transform sector columns (first N_sectors columns of samples_U)
    # using the empirical quantile function of each sector's historical returns.
    rows = []
    for j, col in enumerate(returns.columns):
        series_j = returns[col].dropna().values
        r_j = _back_transform_column(samples_U[:, j], series_j)
        mes = float(np.mean(r_j))

        if S >= 2 * batch_size:
            ci_lo, ci_hi = _batch_means_ci(r_j, batch_size=batch_size)
        else:
            from scipy.stats import t as _t_fb
            se = float(np.std(r_j, ddof=1) / np.sqrt(max(S, 2)))
            t_crit = float(_t_fb.ppf(0.975, df=max(S - 1, 1)))
            ci_lo = mes - t_crit * se
            ci_hi = mes + t_crit * se

        rows.append({"sector": col, "MES": mes, "CI_lower": ci_lo, "CI_upper": ci_hi})

    return pd.DataFrame(rows, columns=["sector", "MES", "CI_lower", "CI_upper"])


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    N_sectors = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    T_test = 500
    S_test = 300
    rng = np.random.default_rng(55)

    fake_returns = pd.DataFrame(
        rng.standard_normal((T_test, N_sectors)) * 1.5,
        columns=[f"Sector_{i}" for i in range(N_sectors)],
    )
    fake_market = pd.Series(rng.standard_normal(T_test) * 1.2, name="OMXS30")

    # Synthetic crisis samples: market col (last) drawn from (0, 0.05)
    fake_U = np.column_stack([
        rng.uniform(0.01, 0.99, size=(S_test, N_sectors)),
        rng.uniform(0.001, 0.05, size=S_test),
    ])

    print(f"Testing mes_estimator.py with T={T_test}, N_sectors={N_sectors}, S={S_test}")
    result = estimate_mes(fake_U, fake_returns, fake_market, alpha=0.05, batch_size=50)
    print(result.to_string(index=False))
