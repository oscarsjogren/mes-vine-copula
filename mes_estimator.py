"""
mes_estimator.py
================
Estimate Marginal Expected Shortfall (MES) for each of N_sectors sector
indices from MCMC samples drawn from the joint vine conditioned on the
market crisis event.

MES_j = E[R_j | R_market ≤ VaR_α(R_market)]

The MCMC samples have shape (S, N_sectors + 1).  The last column is the
market index; it is back-transformed but NOT included in the MES output.
Only the first N_sectors columns are reported.

GARCH(1,1)-skewed-t marginals are re-fitted here independently (all logic
self-contained — no imports from marginals.py).

Exposes
-------
estimate_mes(samples_U, returns, market, alpha, batch_size) → pd.DataFrame
    DataFrame with columns [sector, MES, CI_lower, CI_upper], shape (N_sectors, 4)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t as student_t
from scipy.optimize import brentq, minimize
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Self-contained Hansen skewed-t implementation
# ---------------------------------------------------------------------------

def _skt_constants_local(nu: float, lam: float) -> tuple[float, float, float]:
    """Compute (c, a, b) for Hansen's skewed-t using log-gamma."""
    log_c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    c = np.exp(log_c)
    a = 4.0 * lam * c * ((nu - 2) / (nu - 1))
    b = float(np.sqrt(max(1.0 + 3.0 * lam**2 - a**2, 1e-10)))
    return c, a, b


def _skt_pdf_local(x: np.ndarray, nu: float, lam: float) -> np.ndarray:
    """Hansen skewed-t PDF. Shape (T,) → (T,)."""
    nu = float(max(nu, 2.001))
    lam = float(np.clip(lam, -0.999, 0.999))
    c, a, b = _skt_constants_local(nu, lam)
    pdf = np.empty_like(x, dtype=float)
    left = x < -a / b
    right = ~left
    pdf[left]  = b * c * (1 + (1/(nu-2)) * ((b*x[left]  + a)/(1-lam))**2)**(-(nu+1)/2)
    pdf[right] = b * c * (1 + (1/(nu-2)) * ((b*x[right] + a)/(1+lam))**2)**(-(nu+1)/2)
    return pdf


def _skt_cdf_local(x: np.ndarray, nu: float, lam: float) -> np.ndarray:
    """Hansen skewed-t CDF. Shape (T,) → (T,)."""
    nu = float(max(nu, 2.001))
    lam = float(np.clip(lam, -0.999, 0.999))
    c, a, b = _skt_constants_local(nu, lam)
    cdf = np.empty_like(x, dtype=float)
    left = x < -a / b
    right = ~left
    z_left  = (b * x[left]  + a) / (1 - lam)
    cdf[left] = (1 - lam) * student_t.cdf(z_left, df=nu)
    z_right = (b * x[right] + a) / (1 + lam)
    cdf[right] = (1 - lam) / 2.0 + (1 + lam) * (student_t.cdf(z_right, df=nu) - 0.5)
    return np.clip(cdf, 1e-8, 1 - 1e-8)


def _skt_ppf_local(p: float, nu: float, lam: float) -> float:
    """Quantile of Hansen's skewed-t. Scalar → scalar."""
    p = float(np.clip(p, 1e-8, 1 - 1e-8))
    f = lambda x: float(_skt_cdf_local(np.array([x]), nu, lam)[0]) - p
    try:
        return brentq(f, -200.0, 200.0, xtol=1e-8, maxiter=300)
    except ValueError:
        return float("nan")


def _fit_skt_local(residuals: np.ndarray) -> dict:
    """Fit Hansen skewed-t to residuals via MLE. Returns {'nu': float, 'lam': float}."""
    def neg_ll(theta: np.ndarray) -> float:
        nu, lam = theta
        if nu <= 2.001 or abs(lam) >= 0.999:
            return 1e10
        pdf_vals = _skt_pdf_local(residuals, nu, lam)
        return -float(np.sum(np.log(np.maximum(pdf_vals, 1e-300))))

    best_val, best_x = np.inf, None
    for nu0, lam0 in [(5.0, -0.1), (10.0, 0.0), (20.0, 0.1)]:
        try:
            res = minimize(neg_ll, [nu0, lam0], method="Nelder-Mead",
                           options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000})
            if res.fun < best_val:
                best_val, best_x = res.fun, res.x
        except Exception:
            pass

    if best_x is None:
        return {"nu": 10.0, "lam": 0.0}
    nu, lam = best_x
    return {"nu": float(max(nu, 2.001)), "lam": float(np.clip(lam, -0.999, 0.999))}


def _fit_marginal_local(series: np.ndarray, label: str) -> dict:
    """
    Fit GARCH(1,1)-skewed-t to a single return series.

    Parameters
    ----------
    series : np.ndarray, shape (T,)
    label  : str

    Returns
    -------
    dict with keys: label, sigma (T,), std_residuals (T,), skt_params, scale_factor
    """
    scaled = series * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(scaled, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
        res = am.fit(disp="off", options={"maxiter": 500})
    sigma = res.conditional_volatility
    valid = sigma > 0
    std_resid = np.where(valid, scaled / sigma, 0.0)
    return {
        "label": label,
        "sigma": sigma,
        "std_residuals": std_resid,
        "skt_params": _fit_skt_local(std_resid[valid]),
        "scale_factor": 100.0,
    }


# ---------------------------------------------------------------------------
# Back-transformation
# ---------------------------------------------------------------------------

def _back_transform_column(u_col: np.ndarray, mp: dict) -> np.ndarray:
    """
    Transform a column of uniform MCMC samples to the return scale.

    z = skt_ppf(u; nu, lam)                — standardised residual
    r = z * last_sigma / scale_factor       — approximate return

    Parameters
    ----------
    u_col : np.ndarray, shape (S,)
    mp    : dict from _fit_marginal_local

    Returns
    -------
    returns_col : np.ndarray, shape (S,)
    """
    skt_p = mp["skt_params"]
    nu, lam = skt_p["nu"], skt_p["lam"]
    last_sigma = float(mp["sigma"][-1])
    sf = mp["scale_factor"]
    z = np.array([_skt_ppf_local(float(u), nu, lam) for u in u_col])
    return z * last_sigma / sf


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

    # Re-fit sector marginals independently
    sector_params = [
        _fit_marginal_local(returns[col].dropna().values, col)
        for col in returns.columns
    ]

    # Re-fit market marginal independently
    mkt_label = market.name if market.name is not None else "market"
    market_params = _fit_marginal_local(market.dropna().values, mkt_label)

    # Back-transform sector columns (first N_sectors columns of samples_U)
    rows = []
    for j, col in enumerate(returns.columns):
        r_j = _back_transform_column(samples_U[:, j], sector_params[j])
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
