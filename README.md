# MES Estimation via Vine Copulas

Estimates **Marginal Expected Shortfall (MES)** for N sector indices conditioned on a market index (e.g. S&P 500) falling below its Value-at-Risk, following the framework of [Koike & Hofert (2020)](https://doi.org/10.1515/strm-2019-0033) and Guterstam & Trojenborg (2021).

---

## What is MES?

MES measures how much a sector is expected to lose *on days when the overall market is in crisis*:

```
MES_j = E[ R_j | R_market ≤ VaR_α(R_market) ]
```

A sector with a more negative MES contributes more to systemic risk — it tends to fall hardest precisely when the market is already in distress. This makes MES a useful input for capital allocation, risk budgeting, and stress testing.

---

## Method

The estimation follows a five-stage pipeline:

### 1. Marginal models — two options

**Empirical PIT (default):** rank-based probability integral transform,
`u_jt = rank(r_jt) / (T+1)`, strictly in (0,1). Nonparametric, no distributional assumptions, exactly calibrated by construction — the fraction of crisis observations equals α to numerical precision.

**GARCH(1,1) + skewed-t (`--marginals garch`):** fits a GARCH(1,1) to strip time-varying volatility and a Hansen (1994) skewed-t to the standardised residuals by MLE, then applies the parametric CDF as the PIT. More principled but miscalibrated at extreme quantiles (at α=1% the empirical crisis fraction can be as low as 0.6% instead of 1.0%).

### 2. R-vine copula

An **R-vine copula** is fitted to the (T × N+1) matrix of uniform pseudo-observations (N sectors + market). A vine decomposes the joint density into a cascade of bivariate copulas:

```
c(u_1,...,u_{N+1}) = ∏_{trees} ∏_{edges} c_{jk|D}( F(u_j|u_D), F(u_k|u_D) )
```

Each pair gets its own bivariate copula family, selected by AIC from **Gaussian, Student-t, and Clayton**. The vine structure is chosen by maximum spanning tree on Kendall's τ.

Clayton is included because it has lower-tail dependence (λ_l = 2^{-1/θ} > 0), which directly captures crash-clustering — the tendency for sectors and the market to fall together in the lower tail. Families with 90°/270° rotations (Joe, BB1, BB7) are excluded as AIC tends to select their negative-τ variants, which produce counter-intuitive conditional distributions.

For N+1 = 11 variables (10 sectors + market) this gives 55 bivariate copulas across 10 trees.

### 3. Crisis event

The crisis is defined as `u_market ≤ α`, equivalent after the PIT to `R_market ≤ VaR_α`. The boundary is a flat hyperplane in uniform space, making constrained sampling tractable.

A rate-spike crisis (`--crisis rates`) is also supported: `u_yield_change ≥ 1 − α` (upper tail of daily yield changes).

### 4. Sampling

Two samplers draw from the vine density restricted to the crisis region:

**Rejection sampler (default, recommended):** simulate from the fitted vine using pyvinecopulib's C++ backend, keep rows where `u_market ≤ α`. Statistically exact, no convergence issues, no grid artifacts. Expected acceptance rate ≈ α, so ~100x oversampling at α=1%.

**Metropolis-within-Gibbs (`--sampler mwg`):** cycles through each variable with a scalar MH step.
- *Crisis variable*: uniform proposal over (0, α) — symmetric, exact for any α.
- *Sector variables*: reflected Gaussian random walk with per-variable step sizes adapted during warmup to target 44% acceptance (optimal for 1D MH).

MWG is slower due to serial correlation (CI widths 3–5x wider than rejection) but useful for convergence diagnostics and cross-validation.

### 5. MES estimation

MES is estimated as the sample mean of back-transformed sector returns across all samples, with **95% confidence intervals via batch means** to account for any serial correlation.

---

## Project structure

```
.
├── marginals.py        Empirical PIT and GARCH(1,1)-skewed-t marginals
├── vine_copula.py      R-vine copula fit (Gaussian + Student-t + Clayton)
├── crisis_event.py     Crisis event definition (market tail or rate spike)
├── mcmc_sampler.py     Rejection sampler and Metropolis-within-Gibbs
├── mes_estimator.py    MES point estimates with batch-means 95% CI
├── main.py             End-to-end pipeline orchestration and chart output
├── data_loader.py      Yahoo Finance downloader for sector index data
└── data/
    ├── us_sector_index.csv                   S&P 500 GICS sector manifest
    ├── us_sector_index_returns.csv           Cached log-returns (sectors)
    └── us_sector_index_market_returns.csv    Cached log-returns (market)
```

Each module is independently runnable with a synthetic dataset for testing:

```bash
python3.11 marginals.py 5
python3.11 vine_copula.py 5
python3.11 crisis_event.py 5
python3.11 mcmc_sampler.py 5 mwg
python3.11 mes_estimator.py 5
```

---

## Requirements

```
Python 3.11
arch
pyvinecopulib
numpy
scipy
pandas
matplotlib
yfinance
```

---

## Usage

### Real data — S&P 500 sectors vs S&P 500

```bash
# Download data and run full pipeline (uses cached data on subsequent runs)
python3.11 main.py --real

# Custom date range
python3.11 main.py --real --start 2018-01-01 --end 2024-12-31

# Tighter crisis threshold (1% VaR)
python3.11 main.py --real --alpha 0.01 --N_samples 3000

# GARCH marginals instead of empirical PIT
python3.11 main.py --real --marginals garch

# Metropolis-within-Gibbs for MCMC diagnostics
python3.11 main.py --real --sampler mwg --n_warmup 1000

# Rate-spike crisis (yield change upper tail instead of market lower tail)
python3.11 main.py --real --crisis rates --alpha 0.05

# Force re-download
python3.11 main.py --real --force --start 2015-01-01
```

### Synthetic data

```bash
# 6 synthetic sectors, T=800 observations
python3.11 main.py --N 6 --T 800
```

### Custom sector manifest

Create a semicolon-delimited CSV with columns `Description` and `Ticker`:

```
Description;Ticker
S&P 500 Financials;^SP500-40
S&P 500 Health Care;^SP500-35
```

Then run:

```bash
python3.11 main.py --real --manifest data/my_sectors.csv --market ^GSPC
```

---

## Output

The pipeline produces:

1. **Console log** — step-by-step progress including vine structure, sampler diagnostics, and the MES table.
2. **`mes_chart.png`** — bar chart of MES per sector with 95% CI error bars.
3. **Cached CSVs** in `./data/` — return series saved after download to avoid repeated API calls.

Example output (S&P 500 sectors, empirical PIT, α=1%, rejection sampler, 2015–2026):

```
                        sector       MES  CI_lower  CI_upper
S&P 500 Information Technology -0.050379 -0.051307 -0.049451
S&P 500 Consumer Discretionary -0.046188 -0.047029 -0.045347
            S&P 500 Financials -0.045494 -0.046775 -0.044213
           S&P 500 Industrials -0.042922 -0.043558 -0.042286
             S&P 500 Materials -0.039251 -0.040091 -0.038410
S&P 500 Communication Services -0.038587 -0.039268 -0.037907
           S&P 500 Real Estate -0.029652 -0.030610 -0.028694
           S&P 500 Health Care -0.029760 -0.030415 -0.029105
      S&P 500 Consumer Staples -0.021482 -0.022307 -0.020658
             S&P 500 Utilities -0.018878 -0.019875 -0.017880
```

---

## References

- Koike, T. & Hofert, M. (2020). *Markov chain Monte Carlo methods for estimating systemic risk allocations.* Statistics & Risk Modeling.
- Guterstam, A. & Trojenborg, E. (2021). *Systemic Risk Measures: Monte Carlo Estimation Using Vine Copulas.* KTH Royal Institute of Technology.
- Hansen, B.E. (1994). *Autoregressive conditional density estimation.* International Economic Review.
- Aas, K., Czado, C., Frigessi, A. & Bakken, H. (2009). *Pair-copula constructions of multiple dependence.* Insurance: Mathematics and Economics.
- Bollerslev, T. (1986). *Generalized autoregressive conditional heteroskedasticity.* Journal of Econometrics.
