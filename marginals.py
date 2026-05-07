"""
marginals.py
============
Fit GARCH(1,1) with skewed-t innovations to each of N return series
and to a separate market index series.

Exposes
-------
fit_marginals(returns)                      → list of N dicts  (marginals)
pit_transform(returns, params_list)         → np.ndarray (T, N)  uniform pseudo-obs
fit_market(market)                          → dict               (market marginal)
pit_transform_market(market, params)        → np.ndarray (T,)    uniform pseudo-obs
inverse_cdf(u, params)                      → np.ndarray (S,)    back-transform

All functions infer N from input shape — it is never hardcoded.
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
# Hansen (1994) skewed-t helpers
# ---------------------------------------------------------------------------

def _skt_constants(nu: float, lam: float) -> tuple[float, float, float]:
    """
    Compute the normalising constants (c, a, b) for Hansen's skewed-t.

    Uses log-gamma for numerical stability at large nu.

    Parameters
    ----------
    nu  : float — degrees of freedom > 2
    lam : float — skewness in (-1, 1)

    Returns
    -------
    (c, a, b) : tuple of floats
    """
    log_c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    c = np.exp(log_c)
    a = 4.0 * lam * c * ((nu - 2) / (nu - 1))
    b = float(np.sqrt(max(1.0 + 3.0 * lam**2 - a**2, 1e-10)))
    return c, a, b


def _skt_pdf(x: np.ndarray, nu: float, lam: float) -> np.ndarray:
    """
    Probability density of Hansen's skewed-t.

    Parameters
    ----------
    x   : np.ndarray, shape (T,)
    nu  : float — degrees of freedom > 2
    lam : float — skewness in (-1, 1)

    Returns
    -------
    pdf : np.ndarray, shape (T,)
    """
    nu = float(max(nu, 2.001))
    lam = float(np.clip(lam, -0.999, 0.999))
    c, a, b = _skt_constants(nu, lam)

    pdf = np.empty_like(x, dtype=float)
    left = x < -a / b
    right = ~left
    pdf[left] = (b * c * (1 + (1 / (nu - 2)) * ((b * x[left] + a) / (1 - lam)) ** 2)
                 ** (-(nu + 1) / 2))
    pdf[right] = (b * c * (1 + (1 / (nu - 2)) * ((b * x[right] + a) / (1 + lam)) ** 2)
                  ** (-(nu + 1) / 2))
    return pdf


def _skt_cdf(x: np.ndarray, nu: float, lam: float) -> np.ndarray:
    """
    Cumulative distribution function of Hansen's skewed-t.

    Parameters
    ----------
    x   : np.ndarray, shape (T,)
    nu  : float — degrees of freedom > 2
    lam : float — skewness in (-1, 1)

    Returns
    -------
    cdf : np.ndarray, shape (T,)  — values in (0, 1)
    """
    nu = float(max(nu, 2.001))
    lam = float(np.clip(lam, -0.999, 0.999))
    c, a, b = _skt_constants(nu, lam)

    cdf = np.empty_like(x, dtype=float)
    left = x < -a / b
    right = ~left
    z_left = (b * x[left] + a) / (1 - lam)
    cdf[left] = (1 - lam) * student_t.cdf(z_left, df=nu)
    z_right = (b * x[right] + a) / (1 + lam)
    cdf[right] = (1 - lam) / 2.0 + (1 + lam) * (student_t.cdf(z_right, df=nu) - 0.5)
    return np.clip(cdf, 1e-8, 1 - 1e-8)


def _skt_ppf(p: float, nu: float, lam: float) -> float:
    """
    Quantile (inverse CDF) of Hansen's skewed-t via Brent root-finding.

    Parameters
    ----------
    p   : float — probability in (0, 1)
    nu  : float — degrees of freedom > 2
    lam : float — skewness in (-1, 1)

    Returns
    -------
    q : float
    """
    p = float(np.clip(p, 1e-8, 1 - 1e-8))
    f = lambda x: float(_skt_cdf(np.array([x]), nu, lam)[0]) - p
    try:
        return brentq(f, -200.0, 200.0, xtol=1e-8, maxiter=300)
    except ValueError:
        return float("nan")


def _fit_skt(residuals: np.ndarray) -> dict:
    """
    Fit Hansen's skewed-t to standardised residuals via MLE.

    Parameters
    ----------
    residuals : np.ndarray, shape (T,)

    Returns
    -------
    dict with keys 'nu' (float) and 'lam' (float)
    """
    def neg_ll(theta: np.ndarray) -> float:
        nu, lam = theta
        if nu <= 2.001 or abs(lam) >= 0.999:
            return 1e10
        pdf_vals = _skt_pdf(residuals, nu, lam)
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


def _fit_garch_skt(series: np.ndarray, label: str) -> dict:
    """
    Fit GARCH(1,1) then skewed-t to standardised residuals for a single series.

    Parameters
    ----------
    series : np.ndarray, shape (T,)  — raw return series
    label  : str                     — name label for the output dict

    Returns
    -------
    dict with keys:
        'col'           : str
        'garch_params'  : dict — omega, alpha, beta
        'skt_params'    : dict — nu, lam
        'sigma'         : np.ndarray, shape (T,)
        'std_residuals' : np.ndarray, shape (T,)
        'scale_factor'  : float  (100.0 — series was scaled before GARCH)
    """
    scaled = series * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(scaled, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
        res = am.fit(disp="off", options={"maxiter": 500})

    sigma = res.conditional_volatility
    valid = sigma > 0
    std_resid = np.where(valid, scaled / sigma, 0.0)

    garch_p = {
        "omega": float(res.params.get("omega", res.params.iloc[0])),
        "alpha": float(res.params.get("alpha[1]", res.params.iloc[1])),
        "beta":  float(res.params.get("beta[1]", res.params.iloc[2])),
    }
    skt_p = _fit_skt(std_resid[valid])

    return {
        "col": label,
        "garch_params": garch_p,
        "skt_params": skt_p,
        "sigma": sigma,
        "std_residuals": std_resid,
        "scale_factor": 100.0,
    }


# ---------------------------------------------------------------------------
# Public API — marginals
# ---------------------------------------------------------------------------

def fit_marginals(returns: pd.DataFrame) -> list[dict]:
    """
    Fit GARCH(1,1)-skewed-t to each of the N return series.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Daily return series; column names are series labels.

    Returns
    -------
    params_list : list of N dicts, each containing:
        'col'           : str
        'garch_params'  : dict  — omega, alpha, beta
        'skt_params'    : dict  — nu, lam
        'sigma'         : np.ndarray, shape (T,)
        'std_residuals' : np.ndarray, shape (T,)
        'scale_factor'  : float
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    T, N = returns.shape
    if T < 30:
        raise ValueError(f"Too few observations: got T={T}, need at least 30.")

    return [
        _fit_garch_skt(returns[col].dropna().values, col)
        for col in returns.columns
    ]


def pit_transform(returns: pd.DataFrame, params_list: list[dict] | None = None) -> np.ndarray:
    """
    Apply the probability integral transform to sector returns.

    Parameters
    ----------
    returns     : pd.DataFrame, shape (T, N)
    params_list : list of N dicts from fit_marginals, or None (fits internally)

    Returns
    -------
    U : np.ndarray, shape (T, N)  — uniform pseudo-observations in (0, 1)
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    T, N = returns.shape

    if params_list is None:
        params_list = fit_marginals(returns)
    if len(params_list) != N:
        raise ValueError(f"params_list length {len(params_list)} != N={N}.")

    U = np.empty((T, N), dtype=float)
    for j, p in enumerate(params_list):
        skt_p = p["skt_params"]
        U[:, j] = _skt_cdf(p["std_residuals"], skt_p["nu"], skt_p["lam"])
    return U


# ---------------------------------------------------------------------------
# Public API — market index marginal
# ---------------------------------------------------------------------------

def fit_market(market: pd.Series) -> dict:
    """
    Fit GARCH(1,1)-skewed-t to the market index return series.

    Parameters
    ----------
    market : pd.Series, shape (T,)
        Daily return series for the market index (e.g., OMXS30).

    Returns
    -------
    params : dict with same structure as one element of fit_marginals output.
        'col'           : str  — series name
        'garch_params'  : dict
        'skt_params'    : dict
        'sigma'         : np.ndarray, shape (T,)
        'std_residuals' : np.ndarray, shape (T,)
        'scale_factor'  : float
    """
    if not isinstance(market, pd.Series):
        raise TypeError("market must be a pandas Series.")
    if len(market) < 30:
        raise ValueError(f"Too few observations: got T={len(market)}, need at least 30.")

    label = market.name if market.name is not None else "market"
    return _fit_garch_skt(market.dropna().values, label)


def pit_transform_market(market: pd.Series, params: dict | None = None) -> np.ndarray:
    """
    Apply the probability integral transform to the market index return series.

    After the PIT, the event {u_market ≤ α} is equivalent to
    {R_market ≤ VaR_α(R_market)} — the crisis constraint.

    Parameters
    ----------
    market : pd.Series, shape (T,)
    params : dict from fit_market, or None (fits internally)

    Returns
    -------
    u_market : np.ndarray, shape (T,)  — uniform values in (0, 1)
    """
    if not isinstance(market, pd.Series):
        raise TypeError("market must be a pandas Series.")
    if params is None:
        params = fit_market(market)

    skt_p = params["skt_params"]
    return _skt_cdf(params["std_residuals"], skt_p["nu"], skt_p["lam"])


# ---------------------------------------------------------------------------
# Public API — inverse CDF (back-transformation)
# ---------------------------------------------------------------------------

def inverse_cdf(u: np.ndarray, params: dict) -> np.ndarray:
    """
    Back-transform uniform values to the standardised residual scale.

    Parameters
    ----------
    u      : np.ndarray, shape (S,)  — uniform values in (0, 1)
    params : dict — single element from fit_marginals or fit_market

    Returns
    -------
    z : np.ndarray, shape (S,)
        Standardised residual values.  Multiply by params['sigma'][-1]
        and divide by params['scale_factor'] to get approximate returns.
    """
    u = np.asarray(u, dtype=float)
    skt_p = params["skt_params"]
    z = np.array([_skt_ppf(float(ui), skt_p["nu"], skt_p["lam"]) for ui in u.ravel()])
    return z.reshape(u.shape)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    N_test = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    T_test = 500
    rng = np.random.default_rng(42)

    fake_returns = pd.DataFrame(
        rng.standard_normal((T_test, N_test)) * 1.5,
        columns=[f"Sector_{i}" for i in range(N_test)],
    )
    fake_market = pd.Series(
        rng.standard_normal(T_test) * 1.2, name="OMXS30"
    )

    print(f"Testing marginals.py with T={T_test}, N={N_test}")

    p_list = fit_marginals(fake_returns)
    print(f"  Fitted {len(p_list)} sector marginals.")
    for p in p_list:
        print(f"    {p['col']}: SKT={p['skt_params']}")

    U = pit_transform(fake_returns, p_list)
    print(f"  Sector U shape: {U.shape}, range [{U.min():.4f}, {U.max():.4f}]")

    mkt_params = fit_market(fake_market)
    print(f"  Market marginal: {mkt_params['col']}, SKT={mkt_params['skt_params']}")

    u_mkt = pit_transform_market(fake_market, mkt_params)
    print(f"  Market u shape: {u_mkt.shape}, range [{u_mkt.min():.4f}, {u_mkt.max():.4f}]")
    print(f"  Fraction of market u <= 0.05: {(u_mkt <= 0.05).mean():.3f}  (expect ~0.05)")

    u_sample = rng.uniform(0.01, 0.99, size=5)
    z_back = inverse_cdf(u_sample, p_list[0])
    print(f"  Inverse CDF check: u={u_sample.round(3)}, z={z_back.round(3)}")
