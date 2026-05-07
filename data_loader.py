"""
data_loader.py
==============
Download sector index price data from Yahoo Finance and prepare return
series for the MES pipeline.

Supports two manifests out of the box:
  - ./data/us_sector_index.csv    — S&P 500 GICS sector indices (default)
  - ./data/nordic_index.csv       — Nasdaq Nordic sector indices
                                    (NOTE: Yahoo Finance only serves today's
                                     quote for those tickers; no history.)

The market index defaults to ^GSPC (S&P 500) for US sectors, or ^OMX
(OMXS30) when using the Nordic manifest.

Exposes
-------
load_manifest(csv_path)
    → pd.DataFrame  — columns [description, ticker]

download_prices(tickers, start, end)
    → pd.DataFrame, shape (T, M)  — adjusted close prices, date-indexed

download_sector_data(start, end, manifest_path, market_ticker, force_download)
    → tuple(pd.DataFrame, pd.Series)
       sector_returns : (T, N)   daily log-returns for N sector indices
       market_returns : (T,)     daily log-returns for the market index

Results are cached in ./data/ as CSVs; set force_download=True to refresh.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MANIFEST = DATA_DIR / "us_sector_index.csv"
DEFAULT_MARKET = "^GSPC"   # S&P 500


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Parse a sector index manifest CSV.

    The CSV must be semicolon-delimited with at minimum two columns:
      Description  — human-readable index name
      Ticker       — Yahoo Finance ticker symbol

    Parameters
    ----------
    csv_path : str or Path, optional
        Defaults to ./data/us_sector_index.csv.

    Returns
    -------
    manifest : pd.DataFrame
        Columns: description (str), ticker (str).  Shape (N, 2).
    """
    if csv_path is None:
        csv_path = DEFAULT_MANIFEST
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=";", usecols=[0, 1], header=0)
    df.columns = ["description", "ticker"]
    df = df.dropna(subset=["description", "ticker"]).reset_index(drop=True)
    df["description"] = df["description"].str.strip()
    df["ticker"] = df["ticker"].str.strip()
    return df


# ---------------------------------------------------------------------------
# Price download
# ---------------------------------------------------------------------------

def download_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of Yahoo Finance tickers.

    Uses Ticker.history() per symbol so each series can have its own
    available date range without a single missing ticker killing the batch.

    Parameters
    ----------
    tickers : list[str]  — Yahoo Finance ticker symbols
    start   : str        — start date 'YYYY-MM-DD' (inclusive)
    end     : str        — end date   'YYYY-MM-DD' (inclusive)

    Returns
    -------
    prices : pd.DataFrame, shape (T, len(tickers))
        Date-indexed adjusted close prices.  Columns are the ticker strings.
        A ticker with no data produces an all-NaN column.
    """
    frames: dict[str, pd.Series] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ticker in tickers:
            hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if len(hist) > 0:
                s = hist["Close"].copy()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                frames[ticker] = s
            else:
                frames[ticker] = pd.Series(dtype=float, name=ticker)

    prices = pd.DataFrame(frames)
    prices.index.name = "date"
    return prices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_sector_data(
    start: str = "2010-01-01",
    end: str | None = None,
    manifest_path: str | Path | None = None,
    market_ticker: str = DEFAULT_MARKET,
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Download sector index and market index price data from Yahoo Finance,
    compute log-returns, align on common trading days, and cache to ./data/.

    Parameters
    ----------
    start          : str   — start date 'YYYY-MM-DD' (default '2010-01-01')
    end            : str   — end date 'YYYY-MM-DD' (default: today)
    manifest_path  : path  — sector manifest CSV (default: us_sector_index.csv)
    market_ticker  : str   — Yahoo Finance market index ticker (default '^GSPC')
    force_download : bool  — ignore cache and re-download (default False)

    Returns
    -------
    sector_returns : pd.DataFrame, shape (T, N)
        Log-returns for N sector indices.  Column names are the descriptions
        from the manifest.  N is determined entirely by the manifest.
    market_returns : pd.Series, shape (T,)
        Log-returns for the market index.  Name is market_ticker.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    manifest = load_manifest(manifest_path)
    N = len(manifest)

    # Derive cache filenames from the manifest stem
    manifest_stem = Path(manifest_path or DEFAULT_MANIFEST).stem
    returns_cache = DATA_DIR / f"{manifest_stem}_returns.csv"
    market_cache  = DATA_DIR / f"{manifest_stem}_market_returns.csv"

    if not force_download and returns_cache.exists() and market_cache.exists():
        print(f"Loading cached returns from ./data/")
        sector_returns = pd.read_csv(returns_cache, index_col=0, parse_dates=True)
        market_returns = pd.read_csv(market_cache, index_col=0, parse_dates=True).iloc[:, 0]
        market_returns.name = market_ticker
        print(f"  Sector: {sector_returns.shape}   Market: {market_returns.shape}")
        print(f"  Date range: {sector_returns.index[0].date()} → {sector_returns.index[-1].date()}")
        return sector_returns, market_returns

    print(f"Manifest: {N} sector indices")
    print(f"Tickers:  {manifest['ticker'].tolist()}")
    print(f"Dates:    {start} → {end}")
    print(f"Market:   {market_ticker}\n")

    # Download sector prices
    print(f"Downloading {N} sector indices...")
    sector_prices = download_prices(manifest["ticker"].tolist(), start=start, end=end)

    # Rename columns: ticker → description
    ticker_to_desc = dict(zip(manifest["ticker"], manifest["description"]))
    sector_prices = sector_prices.rename(columns=ticker_to_desc)

    for col in sector_prices.columns:
        n = int(sector_prices[col].notna().sum())
        status = "OK" if n > 0 else "MISSING"
        print(f"  {status:7s}  {col}  ({n} obs)")

    # Download market index
    print(f"\nDownloading market index {market_ticker}...")
    mkt_prices = download_prices([market_ticker], start=start, end=end)
    mkt_series = mkt_prices[market_ticker]
    n_mkt = int(mkt_series.notna().sum())
    print(f"  Market observations: {n_mkt}")

    if n_mkt == 0:
        raise RuntimeError(f"No data returned for market ticker '{market_ticker}'.")

    # Log-returns
    def _log_ret(df: pd.DataFrame) -> pd.DataFrame:
        return np.log(df / df.shift(1)).dropna(how="all")

    sector_returns = _log_ret(sector_prices)
    market_returns = _log_ret(mkt_prices)[market_ticker]
    market_returns.name = market_ticker

    # Inner-join align: only days where every series has data
    combined = sector_returns.join(market_returns, how="inner").dropna(how="any")
    sector_returns = combined.drop(columns=[market_ticker])
    market_returns = combined[market_ticker]
    market_returns.name = market_ticker

    T, N_final = sector_returns.shape
    print(f"\nAligned: T={T} days, N={N_final} sectors")
    print(f"Date range: {sector_returns.index[0].date()} → {sector_returns.index[-1].date()}")

    # Cache
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sector_returns.to_csv(returns_cache)
    pd.DataFrame({market_ticker: market_returns}).to_csv(market_cache)
    print(f"Cached to ./data/{returns_cache.name} and ./data/{market_cache.name}")

    return sector_returns, market_returns


# ---------------------------------------------------------------------------
# Standalone test / CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download sector index return data.")
    parser.add_argument("--manifest", default=None, help="Path to sector manifest CSV.")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--market", default=DEFAULT_MARKET)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sector_rets, market_rets = download_sector_data(
        start=args.start,
        end=args.end,
        manifest_path=args.manifest,
        market_ticker=args.market,
        force_download=args.force,
    )

    print("\n--- Summary ---")
    print(f"Sector returns shape : {sector_rets.shape}")
    print(f"Market returns shape : {market_rets.shape}")
    print("\nAnnualised volatility (%):")
    vols = sector_rets.std() * np.sqrt(252) * 100
    for col, v in vols.items():
        print(f"  {col:<45s} {v:6.2f}%")
    print(f"\n{market_rets.name} ann. vol: {market_rets.std()*np.sqrt(252)*100:.2f}%")
    print("\nCorrelation with market:")
    corr = sector_rets.corrwith(market_rets).sort_values(ascending=False)
    print(corr.to_string())
