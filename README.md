# Replication Package: "The Effective Rate" (Dixon 2026)

This package reproduces all empirical results and simulations in the paper.
All estimation scripts run from the pre-built CSV panel files — no internet connection
or raw data downloads are required beyond the single data-build step below.

---

## Requirements

```
Python ≥ 3.9
pandas, numpy, scipy, linearmodels, matplotlib, openpyxl, xlrd
```

Install in one step:

```bash
pip install pandas numpy scipy linearmodels matplotlib openpyxl xlrd
```

---

## Repository Structure

```
.
├── README.md
├── panel_data.csv                 ← 856-obs monthly US panel (1954–2025)
├── cross_country_panel_data.csv   ← 2506-obs cross-country panel (6 countries)
├── effective_rate_model.py        ← Tables 2–5: IS curve LIML estimation
├── nk_simulation.py               ← Tables 6, 8: NK three-equation simulation
├── robustness_analysis.py         ← Tables 7, 10–14 + SCF validation: all robustness
├── cross_country_analysis.py      ← Table 13: α vs rate-gap cross-country regression
└── data_pipeline.py               ← builds panel_data.csv from scratch (see below)
```

---

## Quick Start: Reproducing All Results

Each script is standalone. Run from any directory — paths resolve relative to the
script file, not the working directory.

```bash
# IS curve estimation (Tables 2–5)
python effective_rate_model.py

# NK simulation, stance, Taylor rule (Tables 6, 8, 9)
python nk_simulation.py

# All robustness, split-coefficient, MC resolution, SCF validation (Tables 7, 10–14)
python robustness_analysis.py

# Cross-country α gradient (Table 13)
python cross_country_analysis.py
```

`robustness_analysis.py` accepts `--skip-mc` to skip the Monte Carlo section (~4 min)
if you only need sections 1–5 and 7.

Expected runtimes on a modern laptop:

| Script | Runtime |
|---|---|
| `effective_rate_model.py` | ~2 min (bootstrap 3,000 draws × 12 specs) |
| `nk_simulation.py` | < 30 s |
| `robustness_analysis.py` | ~8 min full; ~4 min with `--skip-mc` |
| `cross_country_analysis.py` | < 5 s |

---

## Panel Data: Column Reference

### `panel_data.csv` (856 obs, 1954-07 to 2025-10, 62 columns)

All columns are `float64` except `date` (string, ISO format `YYYY-MM-DD`).

**Note on runtime-generated columns.** The following columns are *not* stored in
the panel — every analysis script constructs them on load from the columns that are
stored: `x_ip_f1`, `x_ip_l1`, `mps_cum36_l1`, `ffr_cum36_l1`, `fg_cum36_l1`,
`lsap_cum36_l1`. This keeps the stored panel compact and avoids redundancy.

#### Core series

| Column | Description |
|---|---|
| `date` | Month-start date (YYYY-MM-DD) |
| `fed_funds` | Effective federal funds rate (%) |
| `eff_rate` | Effective rate R̄ = (1−ᾱ)·FF + ᾱ·rᴱ (%) |
| `alpha` | Household equity share ᾱ (8Q moving avg of FoF BOGZ1FL153064486Q) |
| `r_equity` | Trailing 7yr real total S&P 500 return, 12m MA (%) |
| `pi_pce` | PCE inflation, MoM annualised (%) |
| `pi_cpce` | Core PCE inflation, MoM annualised (%) |
| `ip` | Industrial Production index |
| `x_ip` | IP output gap, HP-filtered (λ=129,600) |
| `pces` | PCE Services expenditure (billions, chained) |
| `x_pces` | PCE services output gap, HP-filtered |
| `x_combined` | Combined gap: 0.3·x_ip + 0.7·x_pces |

#### Demand controls and activity

| Column | Description |
|---|---|
| `cfnai` | Chicago Fed National Activity Index |
| `candh` | CFNAI Consumption & Housing sub-index |
| `euandh` | CFNAI Employment, Unemployment & Hours sub-index |
| `houst` | Housing starts (thousands, SAAR) |
| `umcsent` | University of Michigan consumer sentiment |
| `nfcicredit` | Chicago Fed NFCI: Credit sub-index |
| `candh_l1` | CANDH lagged one month (primary demand control) |
| `euandh_l1` | EUANDH lagged one month |
| `houst_l1` | HOUST lagged one month |
| `umcsent_l1` | UMCSENT lagged one month |
| `nfcicredit_l1` | NFCICREDIT lagged one month |

#### Monetary policy surprises and instruments

| Column | Description |
|---|---|
| `mps_orth` | Bauer–Swanson orthogonalised MPS (monthly sum of event-level surprises) |
| `ffr` | Swanson FFR factor (reconstructed; monthly sum of event-level values) |
| `fg` | Swanson Forward Guidance factor |
| `lsap` | Swanson LSAP/QE factor |
| `mps_cum12` | 12-month rolling sum of mps_orth |
| `mps_cum24` | 24-month rolling sum of mps_orth |
| `mps_cum36` | 36-month rolling sum of mps_orth (primary instrument) |
| `ffr_cum36` | 36-month rolling sum of FFR factor |
| `fg_cum36` | 36-month rolling sum of FG factor |
| `lsap_cum36` | 36-month rolling sum of LSAP factor |

#### Financial conditions controls

| Column | Description |
|---|---|
| `anfci` | Adjusted National Financial Conditions Index |
| `baa_spread` | Moody's BAA–AAA spread (pp) |
| `term_spread` | 10y–2y Treasury spread (pp) |
| `vix` | VIX (spliced with VXO pre-1990) |

#### Rate changes and derived rate series

| Column | Description |
|---|---|
| `d_ff` | Month-on-month change in fed_funds |
| `d_eff` | Month-on-month change in eff_rate |
| `rate_gap` | R̄ − FF = ᾱ·(rᴱ − FF) |
| `equity_spread` | rᴱ − FF (the equity-risk-free spread before alpha weighting) |
| `r_real_ff` | fed_funds − pi_cpce (ex-post real federal funds rate) |
| `r_real_eff` | eff_rate − pi_cpce (ex-post real effective rate) |
| `stance_ff` | r_real_ff − r* (policy stance relative to r*=0.5%) |
| `stance_eff` | r_real_eff − r* |

#### Services and combined output gap leads/lags

| Column | Description |
|---|---|
| `x_pces_f1` | x_pces shifted forward one month |
| `x_pces_l1` | x_pces shifted back one month |
| `x_combined_f1` | x_combined shifted forward one month |
| `x_combined_l1` | x_combined shifted back one month |

#### Supply shocks

| Column | Description |
|---|---|
| `wti` | WTI crude oil price ($/bbl) |
| `d12_oil` | 12-month log change in WTI (%). Can reach ±135% during extreme events (e.g. April 2020 WTI crash) — correct, not an error |
| `gscpi` | NY Fed Global Supply Chain Pressure Index (1998-present) |

#### Alternative effective rate constructions (robustness, Table 10)

| Column | Description |
|---|---|
| `eff_rate_5yr` | Effective rate with 5yr trailing equity return |
| `eff_rate_10yr` | Effective rate with 10yr trailing equity return |
| `eff_rate_exp3` | Effective rate with exponential-decay return (half-life 3yr) |
| `eff_rate_exp2` | Effective rate with exponential-decay return (half-life 2yr) |
| `x_fwd_spf` | SPF-based forward output gap forecast: one-quarter-ahead real GDP forecast (Philadelphia Fed SPF) converted to implied output gap change; used as $\mathbb{E}_t^F[x_{t+1}]$ in the $\beta_f$ term of the NK simulation |
| `g_spf` | SPF annualised quarterly GDP growth forecast (%) |
| `excess_monthly` | Monthly excess growth: $(g^{SPF}_t - g^{potential})/3$, used to construct `x_fwd_spf` |

#### SCF aggregation validation (Section 2.6, sparse — non-NaN only at survey-year Decembers)

These columns hold Survey of Consumer Finances microdata at the December of each
triennial survey year (1989, 1992, …, 2022). All other months are NaN. The 12
non-NaN rows are used by `robustness_analysis.py` Section 7.

| Column | Description |
|---|---|
| `scf_fof_narrow` | FoF-comparable narrow equity share computed from SCF microdata |
| `scf_mpc_broad` | MPC-weighted broad equity share (includes retirement account equity; baseline MPC schedule from Parker et al. 2013) |
| `scf_mpc_broad_lo` | Lower bound: very steep MPC schedule |
| `scf_mpc_broad_hi` | Upper bound: very flat MPC schedule |
| `scf_ratio` | scf_mpc_broad / scf_fof_narrow. Above 1.0 in every wave since 2001, meaning FoF measure understates the IS-curve-relevant equity share and all paper results are conservative |

---

### `cross_country_panel_data.csv` (2506 obs, 7 country-series)

| Column | Description |
|---|---|
| `date` | Month-start date (YYYY-MM-DD) |
| `country` | Country identifier: US, UK, Japan, Canada, Sweden, Australia_narrow, Australia_broad |
| `alpha` | Household equity share (country-specific source, 8Q MA) |
| `policy_rate` | Central bank policy rate (%) |
| `r_equity` | Trailing 7yr US real equity return (common factor, %). Max ~22% in 1950s US data — correct |
| `eff_rate` | R̄ = (1−α)·policy_rate + α·r_equity |
| `rate_gap` | R̄ − policy_rate = α·(rᴱ − policy_rate) |
| `inflation` | Headline CPI inflation (%) |
| `cpi` | CPI index level |

**Note:** `Australia_broad` covers only 2011–2013 (ABS Table 36 data availability)
and is excluded from the primary cross-country regression. `Australia_narrow` (OECD
Financial Balance Sheets) covers the full sample and is used instead.

---

## Key Parameters (Headline Specification)

These values are hard-coded in the simulation scripts; they come from
the `effective_rate_model.py` headline estimates (Combined instruments, CorePCE,
CANDH_{t-1} demand control):

| Parameter | Value | Source |
|---|---|---|
| β_f (forward IS) | 0.271 | Table 2, Combined / CorePCE / Eff Rate |
| β_b (backward IS) | 0.687 | Table 2, Combined / CorePCE / Eff Rate |
| σ (IS rate elasticity) | 0.108 | Table 2, Combined / CorePCE / Eff Rate |
| κ (Phillips curve slope) | 0.041 | Estimated directly from data (OLS full sample) |
| γ_b (backward PC) | 0.19 | Backward-looking Phillips curve term |
| PC constant | 1.62 | Calibrated to 2% long-run target: 2.0×(1−γ_b) |
| λ_oil | 0.005 | Supply-augmented Phillips curve |
| λ_GSCPI | 0.367 | Supply-augmented Phillips curve |
| r* | 0.5% | Holston–Laubach–Williams (2017) |
| Trailing horizon | 7 years | Malmendier–Nagel (2016) |
| Alpha MA | 8 quarters | Smoothing (FoF quarterly → monthly) |

**Note on estimators.** LIML is used for all point estimates (median-unbiased under
weak instruments). Block bootstrap draws use 2SLS — LIML lacks finite moments under
weak instruments, making its bootstrap distribution heavy-tailed and unreliable.
See paper Section 3.6.

---

## Rebuilding the Panel from Raw Data (`data_pipeline.py`)

> **This step is not required for replication.** `panel_data.csv` is pre-built
> and already included. Run `data_pipeline.py` only if you want to extend the
> sample, change parameters, or verify the construction.

`data_pipeline.py` requires the following raw files in `DATA_DIR`
(default: the current working directory; override by editing `DATA_DIR` at the top):

**FRED CSVs** (download from https://fred.stlouisfed.org):

| Filename | Series | Description |
|---|---|---|
| `FEDFUNDS__1_.csv` | FEDFUNDS | Effective federal funds rate (monthly) |
| `BOGZ1FL153064486Q.csv` | BOGZ1FL153064486Q | Household equity share, FoF (quarterly) |
| `PCEPI.csv` | PCEPI | PCE price index |
| `PCEPILFE__1_.csv` | PCEPILFE | Core PCE price index |
| `INDPRO.csv` | INDPRO | Industrial Production index |
| `CFNAI.csv` | CFNAI | Chicago Fed National Activity Index |
| `CANDH.csv` | CFNAI: Consumption & Housing | |
| `EUANDH.csv` | CFNAI: Employment & Hours | |
| `ANFCI.csv` | ANFCI | Adjusted National Financial Conditions |
| `BAAFFM.csv` | BAAFFM | BAA–AAA corporate spread |
| `T10Y2Y.csv` | T10Y2Y | 10y–2y Treasury spread |
| `VIXCLS__1_.csv` | VIXCLS | VIX |
| `VXOCLS.csv` | VXOCLS | VXO (pre-1990 VIX proxy) |
| `PCES.csv` | PCES | PCE Services |
| `HOUST.csv` | HOUST | Housing starts |
| `UMCSENT.csv` | UMCSENT | Michigan consumer sentiment |
| `NFCICREDIT.csv` | NFCICREDIT | National Financial Conditions: Credit |
| `WTISPLC__1_.csv` | WTISPLC | WTI crude oil spot price |

**External sources:**

| Filename | Source | Description |
|---|---|---|
| `ie_data.xls` | http://www.econ.yale.edu/~shiller/data/ie_data.xls | Shiller S&P 500 data (price, dividend, CPI) |
| `monetary-policy-surprises-data__3_.xlsx` | Bauer & Swanson (2023) | FOMC monetary policy surprises |
| `pre-and-post-ZLB-factors-extended.xlsx` | Swanson (2021) | FFR/FG/LSAP factors for sign normalisation |
| `gscpi_data__1_.xls` | https://www.newyorkfed.org/research/policy/gscpi | NY Fed Global Supply Chain Pressure Index |

Run:
```bash
python data_pipeline.py
```

Output: `panel_data.csv` written to the same directory as the script.

---

## Notes on Swanson Factor Reconstruction

`data_pipeline.py` reconstructs the Swanson (2021) FFR/FG/LSAP factors via
subspace identification on the Bauer–Swanson FOMC futures surprise data.
Correlations with the original published factors over the 1994–2019 overlap
period are ρ(FFR)=0.93, ρ(FG)=0.92, ρ(LSAP)=0.85. Sign normalisations are
corrected against the original when `pre-and-post-ZLB-factors-extended.xlsx` is
present; otherwise heuristic sign rules are applied.

---

## Notes on `Australia_broad`

The `Australia_broad` series in `cross_country_panel_data.csv` covers only
2011–2013 (ABS Table 36 data availability) and is excluded from the primary
cross-country regression. `Australia_narrow` (OECD Financial Balance Sheets,
full sample) is used instead. Both series are retained for transparency.

---

## Citation

If you use this replication package, please cite:

> Dixon, A. (2026). *The Effective Rate: Household Portfolio Returns and
> Monetary Policy Transmission*. Working paper.


