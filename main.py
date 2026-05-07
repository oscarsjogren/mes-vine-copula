"""
main.py
=======
Orchestrate the full MES estimation pipeline using a market index
to define the crisis event.

Data sources
------------
  --real                     Download real sector data via data_loader.py
                             (default: S&P 500 GICS sectors vs ^GSPC)
  --manifest path/to/csv     Use a custom sector manifest CSV
  --market TICKER            Override the market index ticker
  --start / --end            Date range for download (default 2010-01-01 → today)
  --force                    Re-download even if cache exists

  If --real is not passed, synthetic correlated returns are generated.

Pipeline
--------
  1. Load sector returns (T x N) and market index returns (T,).
  2. Fit GARCH(1,1)-skewed-t marginals to each series.
  3. PIT transform → uniform pseudo-observations.
  4. Stack into a (T, N+1) array: [sector cols | market col].
  5. Fit R-vine copula on all N+1 variables.
  6. Build crisis spec: u_market ≤ alpha  ⟺  R_market ≤ VaR_alpha.
  7. Run MCMC (Gibbs or HMC) — samples in (N+1)-dimensional uniform space.
  8. Back-transform sector columns → estimate MES with 95% CI.
  9. Plot bar chart.

Usage
-----
  python3.11 main.py --real                            # real S&P 500 sector data
  python3.11 main.py --real --start 2018-01-01         # custom date range
  python3.11 main.py --real --manifest data/us_sector_index.csv --market ^GSPC
  python3.11 main.py --N 6 --T 800                     # synthetic data
  python3.11 main.py --real --sampler hmc --alpha 0.10
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marginals import fit_marginals, pit_transform, fit_market, pit_transform_market
from vine_copula import fit_vine
from crisis_event import build_crisis_spec
from mcmc_sampler import run_sampler
from mes_estimator import estimate_mes
from data_loader import download_sector_data, DEFAULT_MARKET


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MES estimation via vine copula + MCMC, conditioned on a market index."
    )
    # Real data download
    parser.add_argument("--real", action="store_true",
                        help="Download real sector data via Yahoo Finance.")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Sector manifest CSV (default: data/us_sector_index.csv).")
    parser.add_argument("--market", type=str, default=DEFAULT_MARKET,
                        help=f"Market index ticker (default: {DEFAULT_MARKET}).")
    parser.add_argument("--start", type=str, default="2010-01-01",
                        help="Start date for download (default: 2010-01-01).")
    parser.add_argument("--end", type=str, default=None,
                        help="End date for download (default: today).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cache exists.")
    # Synthetic data fallback
    parser.add_argument("--N", type=int, default=5,
                        help="Sectors for synthetic data (ignored with --real).")
    parser.add_argument("--T", type=int, default=600,
                        help="Time steps for synthetic data.")
    # Pipeline parameters
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="VaR level for crisis event (default 0.05).")
    parser.add_argument("--sampler", type=str, default="gibbs",
                        choices=["gibbs", "hmc"])
    parser.add_argument("--N_samples", type=int, default=2000)
    parser.add_argument("--n_warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="mes_chart.png")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading / generation
# ---------------------------------------------------------------------------

def _load_or_generate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (sector_returns, market_returns).

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    returns : pd.DataFrame, shape (T, N)
    market  : pd.Series,    shape (T,)
    """
    if args.real:
        returns, market = download_sector_data(
            start=args.start,
            end=args.end,
            manifest_path=args.manifest,
            market_ticker=args.market,
            force_download=args.force,
        )
        return returns, market

    # Synthetic: N correlated sector series + 1 market series
    N = args.N
    T = args.T
    market_name = args.market
    rng = np.random.default_rng(args.seed)

    raw = rng.standard_normal((N + 1, N + 1))
    cov = raw @ raw.T / (N + 1) + np.eye(N + 1) * 0.5
    vols = np.append(rng.uniform(0.5, 2.0, size=N), 1.2)
    cov_scaled = np.diag(vols) @ cov @ np.diag(vols)

    data = rng.multivariate_normal(np.zeros(N + 1), cov_scaled, size=T) / 100.0
    data *= (1 + 0.3 * np.abs(rng.standard_normal(T)))[:, None]

    returns = pd.DataFrame(data[:, :N], columns=[f"Sector_{i+1}" for i in range(N)])
    market = pd.Series(data[:, N], name=market_name)

    print(f"Generated synthetic returns: T={T}, N={N} sectors, market='{market.name}'")
    return returns, market


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    """
    Execute the full MES estimation pipeline.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    mes_df : pd.DataFrame, shape (N_sectors, 4)
        Columns: sector, MES, CI_lower, CI_upper
    """
    returns, market = _load_or_generate(args)
    T, N_sectors = returns.shape
    print(f"\n--- Pipeline: T={T}, N_sectors={N_sectors}, market='{market.name}' ---")

    # Step 1: Fit marginals
    print("[1/5] Fitting GARCH(1,1)-skewed-t marginals...")
    sector_params = fit_marginals(returns)
    market_params = fit_market(market)
    U_sectors = pit_transform(returns, sector_params)     # (T, N_sectors)
    u_market  = pit_transform_market(market, market_params)  # (T,)

    # Stack: sectors first, market last
    U_full = np.column_stack([U_sectors, u_market])       # (T, N_sectors + 1)
    print(f"      U_full shape: {U_full.shape}")
    print(f"      Fraction of market u ≤ alpha={args.alpha}: "
          f"{(u_market <= args.alpha).mean():.3f}  (expect ~{args.alpha})")

    # Step 2: Fit R-vine on all N+1 variables
    N_total = N_sectors + 1
    print(f"[2/5] Fitting R-vine copula (N_total={N_total} = {N_sectors} sectors + market)...")
    vine_result = fit_vine(U_full)
    print(f"      {vine_result}")
    print(f"      Families (first 5): {vine_result.family_names[:5]}")

    # Step 3: Build crisis spec
    print(f"[3/5] Building crisis event: u_{market.name} ≤ {args.alpha} ...")
    crisis_spec = build_crisis_spec(
        U_full, alpha=args.alpha,
        market_idx=N_sectors,          # last column
        market_name=market.name,
    )
    print(f"      {crisis_spec}")

    # Step 4: MCMC sampling
    print(f"[4/5] Running {args.sampler.upper()} sampler "
          f"(N_samples={args.N_samples}, warmup={args.n_warmup})...")
    samples_U = run_sampler(
        vine_result=vine_result,
        crisis_spec=crisis_spec,
        sampler=args.sampler,
        N_samples=args.N_samples,
        n_warmup=args.n_warmup,
        seed=args.seed,
    )
    print(f"      Samples shape: {samples_U.shape}")
    mkt_col = samples_U[:, crisis_spec.market_idx]
    print(f"      Market col: max={mkt_col.max():.5f}, all ≤ alpha: {(mkt_col <= args.alpha).all()}")

    # Step 5: Estimate MES
    print("[5/5] Estimating MES with 95% CI (batch means)...")
    mes_df = estimate_mes(
        samples_U=samples_U,
        returns=returns,
        market=market,
        alpha=args.alpha,
        batch_size=max(50, args.N_samples // 20),
    )
    print(f"\nMES Results  (crisis = {market.name} ≤ VaR_{args.alpha:.0%}):")
    print(mes_df.to_string(index=False))
    return mes_df


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def plot_mes(
    mes_df: pd.DataFrame,
    N_sectors: int,
    market_name: str,
    alpha: float,
    sampler: str,
    out_path: str,
) -> None:
    """
    Bar chart of MES estimates with 95% CI error bars.

    Parameters
    ----------
    mes_df      : pd.DataFrame, shape (N_sectors, 4)
    N_sectors   : int
    market_name : str — name of market index used for crisis definition
    alpha       : float
    sampler     : str
    out_path    : str
    """
    sectors  = mes_df["sector"].tolist()
    mes_vals = mes_df["MES"].values
    ci_lo    = mes_df["CI_lower"].values
    ci_hi    = mes_df["CI_upper"].values

    yerr = np.array([np.abs(mes_vals - ci_lo), np.abs(ci_hi - mes_vals)])

    fig, ax = plt.subplots(figsize=(max(6, N_sectors * 1.2), 5))
    colors = plt.cm.tab20(np.linspace(0, 1, N_sectors))

    bars = ax.bar(
        sectors, mes_vals,
        color=colors, edgecolor="black", linewidth=0.7,
        yerr=yerr, capsize=5,
        error_kw={"elinewidth": 1.5, "ecolor": "black"},
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"Marginal Expected Shortfall — {N_sectors} Sectors\n"
        f"Crisis: {market_name} ≤ VaR_{alpha:.0%}   |   Sampler: {sampler.upper()}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel(f"Sector  (N={N_sectors})", fontsize=11)
    ax.set_ylabel(f"MES  |  E[R_sector | R_{market_name} ≤ VaR_{alpha:.0%}]", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for bar, val in zip(bars, mes_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.0001 if val >= 0 else -0.0005),
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nChart saved to: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    mes_df = run_pipeline(args)
    N_sectors = len(mes_df)
    plot_mes(
        mes_df,
        N_sectors=N_sectors,
        market_name=args.market,
        alpha=args.alpha,
        sampler=args.sampler,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
