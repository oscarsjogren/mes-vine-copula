# MES Estimation via Vine Copulas and MCMC

Estimates **Marginal Expected Shortfall (MES)** for N sector indices conditioned on a market index (e.g. S&P 500) falling below its Value-at-Risk, following the framework of [Koike & Hofert (2020)](https://doi.org/10.1515/strm-2019-0033).

---

## What is MES?

MES measures how much a sector is expected to lose *on days when the overall market is in crisis*:

```
MES_j = E[ R_j | R_market ≤ VaR_α(R_market) ]
```

A sector with a more negative MES contributes more to systemic risk — it tends to fall hardest precisely when the market is already in distress. This makes MES a useful input for capital allocation, risk budgeting, and stress testing.

---

## Method

The estimation follows a three-stage pipeline:

### 1. Marginal models — GARCH(1,1) + Hansen skewed-t

Each return series (sectors and market) is modelled with a **GARCH(1,1)** to strip out time-varying volatility:

```
r_t = σ_t · ε_t
σ²_t = ω + α · r²_{t-1} + β · σ²_{t-1}
```

The standardised residuals `ε_t = r_t / σ_t` are approximately i.i.d. A **Hansen (1994) skewed-t** distribution is then fitted to them by MLE, capturing fat tails (ν) and return asymmetry (λ).

The full marginal CDF is the composition:

```
F(r_t) = F_skewed-t( r_t / σ_t ;  ν, λ )
```

Applying this CDF to each series produces **uniform pseudo-observations** U ∈ (0,1) via the probability integral transform (PIT). By Sklar's theorem, all remaining dependence structure lives in the joint distribution of these uniforms — which is what the copula models.

### 2. R-vine copula

An **R-vine copula** is fitted to the (T × N+1) matrix of uniform pseudo-observations (N sectors + market). A vine decomposes the joint density into a cascade of bivariate copulas:

```
c(u_1,...,u_{N+1}) = ∏_{trees} ∏_{edges} c_{jk|D}( F(u_j|u_D), F(u_k|u_D) )
```

Each pair gets its own bivariate copula family, selected by AIC from: Gaussian, Student-t, Clayton, Gumbel, Frank, Joe, BB1, BB7 (and their rotations). The vine structure itself is chosen by maximum spanning tree on Kendall's τ, so the strongest pairwise dependencies are captured in Tree 1.

For N+1 = 11 variables (10 sectors + market) this gives 55 bivariate copulas across 10 trees.

### 3. MCMC in the crisis region

The crisis event is defined as:

```
u_market ≤ α
```

which after the PIT is exactly equivalent to `R_market ≤ VaR_α(R_market)`. The boundary is a flat hyperplane in uniform space, making sampling tractable.

Two samplers are implemented:

- **Gibbs**: cycles through each dimension, drawing from the conditional `p(u_j | u_{-j})` evaluated on a fine grid, restricting the market dimension to `(0, α)`.
- **HMC**: leapfrog proposals using the vine log-density gradient, with exact specular reflection off the flat crisis boundary.

MES is estimated as the sample mean of back-transformed sector returns across all MCMC samples, with **95% confidence intervals via batch means** to account for MCMC serial correlation.

---

## Project structure

```
.
├── marginals.py        GARCH(1,1)-skewed-t fit and PIT transform
├── vine_copula.py      R-vine copula fit with AIC family selection
├── crisis_event.py     Crisis event definition and boundary geometry
├── mcmc_sampler.py     Gibbs and HMC samplers in the crisis region
├── mes_estimator.py    MES point estimates with batch-means 95% CI
├── main.py             End-to-end pipeline orchestration and chart output
├── data_loader.py      Yahoo Finance downloader for sector index data
└── data/
    ├── us_sector_index.csv          S&P 500 GICS sector manifest
    ├── nordic_index.csv             Nasdaq Nordic sector manifest
    ├── us_sector_index_returns.csv  Cached log-returns (sectors)
    └── us_sector_index_market_returns.csv  Cached log-returns (market)
```

Each module is independently runnable with a synthetic dataset for testing:

```bash
python3.11 marginals.py 5        # test with N=5 synthetic series
python3.11 vine_copula.py 5      # test vine fit on N+1=6 variables
python3.11 crisis_event.py 5
python3.11 mcmc_sampler.py 5 gibbs
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

# Force re-download
python3.11 main.py --real --force --start 2015-01-01

# Use HMC sampler, 10% VaR level
python3.11 main.py --real --sampler hmc --alpha 0.10

# More MCMC samples for tighter confidence intervals
python3.11 main.py --real --N_samples 5000 --n_warmup 1000
```

### Synthetic data

```bash
# 6 synthetic sectors, T=800 observations
python3.11 main.py --N 6 --T 800
```

### Download data only

```bash
python3.11 data_loader.py --start 2015-01-01
python3.11 data_loader.py --start 2015-01-01 --force   # re-download
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

1. **Console log** — step-by-step progress including vine structure, MCMC diagnostics, and the MES table.
2. **`mes_chart.png`** — bar chart of MES per sector with 95% CI error bars.
3. **Cached CSVs** in `./data/` — return series saved after download to avoid repeated API calls.

Example MES output (S&P 500 sectors, α=5%, 2015–2026):

```
                       sector       MES  CI_lower  CI_upper
          S&P 500 Communication -0.286   -0.301    -0.271
      S&P 500 Info Technology   -0.200   -0.214    -0.186
            S&P 500 Industrials -0.173   -0.185    -0.161
              S&P 500 Utilities -0.159   -0.171    -0.147
             S&P 500 Real Estate -0.132  -0.143    -0.121
```

---

## References

- Koike, T. & Hofert, M. (2020). *Markov chain Monte Carlo methods for estimating systemic risk allocations.* Statistics & Risk Modeling.
- Hansen, B.E. (1994). *Autoregressive conditional density estimation.* International Economic Review.
- Aas, K., Czado, C., Frigessi, A. & Bakken, H. (2009). *Pair-copula constructions of multiple dependence.* Insurance: Mathematics and Economics.
- Bollerslev, T. (1986). *Generalized autoregressive conditional heteroskedasticity.* Journal of Econometrics.
