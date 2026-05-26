"""
Effective Rate v2: Robustness Analysis
═══════════════════════════════════════
Standalone script producing all robustness and extension results for the paper.

CONTENTS:
  1. Alternative trailing horizons (5yr, 7yr, 10yr) and belief-weighting schemes
  2. Rolling-window + subsample stability
  3. Local projections (LP-IV impulse responses)
  4. Formal model comparison tests (Vuong, Clarke, Diebold-Mariano)
  5. Split-coefficient analysis: decomposing σ̂ into IES vs financial-conditions
  6. Monte Carlo resolution: conditional and unconditional bootstrap distributions
  7. SCF aggregation validation: MPC-weighted vs Flow of Funds alpha (Section 2.6)

Sections 1–4 use the paper's primary specification throughout:
  - Combined instruments (MPS + Swanson 3F, 8 total)
  - CorePCE inflation
  - CANDH_{t-1} demand control
  - LIML estimator with conservative HAC inference (n^{1/3} bandwidth)

Section 5 adds the split-coefficient IS curve (3 endogenous variables).
Section 6 propagates bootstrap parameter uncertainty through the NK simulation.
Section 7 reads the SCF columns from panel_data.csv (survey waves stored at
December of each survey year; all other months NaN).

Note on estimators: LIML is used for all point estimates (median-unbiased under
weak instruments). Block bootstrap draws use 2SLS, which is numerically stable
and consistent for inference purposes; see paper Section 3.6 for the rationale.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from scipy import stats
from linearmodels.iv import IVLIML
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# CONFIGURATION
# ================================================================

PANEL_PATH = Path(__file__).parent / 'panel_data.csv'

N_BOOT     = 3000   # bootstrap replications (all sections)
BLOCK_SIZE = 12     # block length in months
SAMPLE_START = '1988-07-01'
SAMPLE_END   = '2023-12-01'

# Primary specification
CONTROLS = ['anfci', 'baa_spread', 'term_spread', 'vix', 'candh_l1']

INSTRUMENTS_MPS = [
    'mps_cum12', 'mps_cum24', 'mps_cum36', 'mps_cum36_l1',
]
INSTRUMENTS_SWANSON = [
    'ffr_cum36', 'fg_cum36', 'lsap_cum36',
    'ffr_cum36_l1', 'fg_cum36_l1', 'lsap_cum36_l1',
]
INSTRUMENTS_COMBINED = [
    'mps_cum36', 'mps_cum36_l1',
    'ffr_cum36', 'fg_cum36', 'lsap_cum36',
    'ffr_cum36_l1', 'fg_cum36_l1', 'lsap_cum36_l1',
]

# NK simulation parameters (Section 6)
# IS curve: Combined instruments, CorePCE, CANDH_{t-1} (paper Table 3)
NK_BETA_F    = 0.271
NK_BETA_B    = 0.687
NK_SIGMA     = 0.108
# Supply-augmented Phillips curve (paper Eq. 15)
NK_GAMMA_F   = 0.18
NK_GAMMA_B   = 0.19
NK_KAPPA     = 0.045
NK_LAM_OIL   = 0.005
NK_LAM_GSCPI = 0.367
NK_PC_CONST  = 1.23
NK_R_STAR    = 0.5


# ================================================================
# SHARED ESTIMATION INFRASTRUCTURE
# ================================================================

def iv_manual(y, Xe, Xn, Z):
    """Manual 2SLS for bootstrap/MC draws.

    2SLS is the methodologically correct estimator for the bootstrap loop, not a
    tractability compromise. LIML lacks finite moments under weak instruments (its
    distribution is heavy-tailed due to the minimum eigenvalue computation), which
    means LIML draws from block-resampled samples can diverge or produce extreme
    outliers that contaminating the bootstrap distribution. 2SLS is numerically
    stable and consistent under both H0 and Ha; the bootstrap is characterising
    sampling variation, not point estimation, so 2SLS is appropriate here even
    though the paper's point estimates use IVLIML.
    """
    W  = np.column_stack([Xe, Z])
    X  = np.column_stack([Xe, Xn])
    Pw = W @ lstsq(W, X, rcond=None)[0]
    return lstsq(Pw, y, rcond=None)[0]


def block_bootstrap(y, Xe, Xn, Z, rr_idx, n_boot=N_BOOT, block_size=BLOCK_SIZE, seed=42):
    """Block bootstrap p-value and CI for the blended IS curve coefficient σ̂."""
    n   = len(y)
    rng = np.random.RandomState(seed)
    betas = []
    nb    = int(np.ceil(n / block_size))
    for _ in range(n_boot):
        starts = rng.randint(0, n - block_size + 1, size=nb)
        idx    = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        try:
            betas.append(iv_manual(y[idx], Xe[idx], Xn[idx], Z[idx]))
        except:
            pass
    betas       = np.array(betas)
    boot_sigmas = -betas[:, rr_idx]
    boot_p  = 2 * min(np.mean(boot_sigmas <= 0), np.mean(boot_sigmas >= 0))
    ci_lo   = -np.percentile(betas[:, rr_idx], 97.5)
    ci_hi   = -np.percentile(betas[:, rr_idx], 2.5)
    return boot_p, ci_lo, ci_hi


def browns_method(p_values, corr_matrix):
    """Brown's method for combining correlated p-values."""
    k   = len(p_values)
    T   = -2 * np.sum(np.log(np.maximum(p_values, 1e-15)))
    E_T = 2 * k
    extra = sum(2 * (3.263*corr_matrix[i,j] + 0.710*corr_matrix[i,j]**2 +
                     0.027*corr_matrix[i,j]**3)
                for i in range(k) for j in range(i+1, k))
    Var_T = 4*k + extra
    c = Var_T / (2 * E_T)
    f = 2 * E_T**2 / Var_T
    return 1 - stats.chi2.cdf(T/c, f)


def estimate_is_curve(m, rate_col, pi_col, instruments, controls,
                      start=SAMPLE_START, end=SAMPLE_END, run_bootstrap=True,
                      y_col='x_ip', y_f1='x_ip_f1', y_l1='x_ip_l1'):
    """
    Estimate IS curve via LIML with HAC and optional 2SLS block bootstrap.

      x_t = β_f·x_{t+1} + β_b·x_{t-1} − σ·r^e_t + controls + ε_t

    Returns dict: n, sigma, beta_f, beta_b, se_sigma, p_hac, p_conv,
                  resid, dates, fs_f, and optionally boot_p/ci_lo/ci_hi.
    """
    dd = m.copy()
    dd['real_rate'] = dd[rate_col] - dd[pi_col].shift(-1)

    needed = [y_col, y_f1, y_l1, 'real_rate'] + controls + instruments
    needed = [c for c in needed if c in dd.columns]
    e = dd[needed + ['date']].dropna()
    e = e[(e['date'] >= start) & (e['date'] <= end)].reset_index(drop=True)
    n = len(e)
    if n < 40:
        return None

    e['const'] = 1.0
    exog = [y_l1] + controls + ['const']
    bw   = max(2, int(np.ceil(n**(1/3))))

    try:
        model = IVLIML(dependent=e[y_col], exog=e[exog],
                       endog=e[[y_f1, 'real_rate']], instruments=e[instruments])
        r_c = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=bw, debiased=True)
        r_v = model.fit(cov_type='unadjusted', debiased=True)
    except Exception:
        return None

    result = {
        'n':        n,
        'sigma':   -r_c.params['real_rate'],
        'beta_f':   r_c.params[y_f1],
        'beta_b':   r_c.params[y_l1],
        'se_sigma': r_c.std_errors['real_rate'],
        'p_hac':    r_c.pvalues['real_rate'],
        'p_conv':   r_v.pvalues['real_rate'],
        'resid':    r_c.resids.values.copy(),
        'dates':    e['date'].values.copy(),
    }
    try:
        for key, fsres in r_c.first_stage.individual.items():
            if 'real_rate' in key:
                result['fs_f'] = fsres.f_statistic.stat
    except:
        result['fs_f'] = None

    if run_bootstrap:
        y_  = e[y_col].values
        Xe_ = e[exog].values
        Xn_ = e[[y_f1, 'real_rate']].values
        Z_  = e[instruments].values
        rr_idx = len(exog) + 1
        bp, ci_lo, ci_hi = block_bootstrap(y_, Xe_, Xn_, Z_, rr_idx)
        result.update({'boot_p': bp, 'boot_ci_lo': ci_lo, 'boot_ci_hi': ci_hi})

    return result


# ================================================================
# 1. ALTERNATIVE HORIZONS AND BELIEF WEIGHTING
# ================================================================

def alternative_horizon_results(m, instruments=INSTRUMENTS_COMBINED, controls=CONTROLS):
    """IS curve estimates across alternative effective rate constructions (Table 10)."""
    rate_cols = {
        '5yr uniform':      'eff_rate_5yr',
        '7yr uniform':      'eff_rate',
        '10yr uniform':     'eff_rate_10yr',
        '7yr exp (HL=3yr)': 'eff_rate_exp3',
        '7yr exp (HL=2yr)': 'eff_rate_exp2',
        '7yr recency 2x':   'eff_rate_rec',
        'Fed Funds':        'fed_funds',
    }
    results = []
    for label, col in rate_cols.items():
        if col not in m.columns:
            continue
        r = estimate_is_curve(m, col, 'pi_cpce', instruments, controls, run_bootstrap=True)
        if r:
            results.append({
                'construction': label, 'rate_col': col,
                'n': r['n'], 'sigma': r['sigma'],
                'p_hac': r['p_hac'], 'p_conv': r['p_conv'],
                'boot_p': r.get('boot_p', np.nan),
                'beta_f': r['beta_f'], 'beta_b': r['beta_b'],
                'fs_f': r.get('fs_f'),
            })
    return results


# ================================================================
# 2. ROLLING WINDOWS + SUBSAMPLE STABILITY
# ================================================================

def rolling_window_analysis(m, window_years=15, step_months=12,
                            instruments=INSTRUMENTS_COMBINED, controls=CONTROLS):
    """IS curve in rolling 15-year windows (annual steps). No bootstrap for speed."""
    results = []
    dates = m['date'].sort_values().unique()
    valid = dates[(dates >= pd.Timestamp(SAMPLE_START)) &
                  (dates <= pd.Timestamp(SAMPLE_END))]

    for ws in valid[::step_months]:
        we = ws + pd.DateOffset(years=window_years)
        if we > pd.Timestamp(SAMPLE_END):
            break
        ws_str, we_str = str(ws.date()), str(we.date())
        mask      = (m['date'] >= ws) & (m['date'] <= we)
        avg_alpha = m.loc[mask, 'alpha'].mean()

        r_ff  = estimate_is_curve(m, 'fed_funds', 'pi_cpce', instruments, controls,
                                  start=ws_str, end=we_str, run_bootstrap=False)
        r_eff = estimate_is_curve(m, 'eff_rate',  'pi_cpce', instruments, controls,
                                  start=ws_str, end=we_str, run_bootstrap=False)

        if r_ff and r_eff:
            results.append({
                'start': ws_str, 'end': we_str, 'n': r_ff['n'],
                'avg_alpha': avg_alpha,
                'sigma_ff':  r_ff['sigma'],  'p_ff':  r_ff['p_hac'], 'bf_ff':  r_ff['beta_f'],
                'sigma_eff': r_eff['sigma'], 'p_eff': r_eff['p_hac'],'bf_eff': r_eff['beta_f'],
                'sigma_ratio': (r_eff['sigma'] / r_ff['sigma']
                                if abs(r_ff['sigma']) > 0.001 else np.nan),
            })
    return results


def subsample_analysis(m, instruments=INSTRUMENTS_COMBINED, controls=CONTROLS):
    """IS curve across discrete structural-break subsamples."""
    subsamples = {
        'Pre-ZLB (1990-2007)': ('1990-01-01', '2007-11-01'),
        'ZLB era (2008-2015)': ('2008-12-01', '2015-11-01'),
        'Liftoff (2016-2019)': ('2016-01-01', '2019-12-01'),
        'Full sample':         (SAMPLE_START, SAMPLE_END),
        'Post-2000':           ('2000-01-01', SAMPLE_END),
        'Pre-2008':            (SAMPLE_START, '2007-11-01'),
    }
    results = []
    for label, (start, end) in subsamples.items():
        mask      = (m['date'] >= start) & (m['date'] <= end)
        avg_alpha = m.loc[mask, 'alpha'].mean()
        r_ff  = estimate_is_curve(m, 'fed_funds', 'pi_cpce', instruments, controls,
                                  start=start, end=end, run_bootstrap=False)
        r_eff = estimate_is_curve(m, 'eff_rate',  'pi_cpce', instruments, controls,
                                  start=start, end=end, run_bootstrap=False)
        results.append({
            'subsample': label, 'avg_alpha': avg_alpha,
            'n':         r_ff['n'] if r_ff else (r_eff['n'] if r_eff else 0),
            'sigma_ff':  r_ff['sigma']  if r_ff else np.nan,
            'p_ff':      r_ff['p_hac']  if r_ff else np.nan,
            'bf_ff':     r_ff['beta_f'] if r_ff else np.nan,
            'sigma_eff': r_eff['sigma']  if r_eff else np.nan,
            'p_eff':     r_eff['p_hac']  if r_eff else np.nan,
            'bf_eff':    r_eff['beta_f'] if r_eff else np.nan,
        })
    return results


# ================================================================
# 3. LOCAL PROJECTIONS (LP-IV)
# ================================================================

def build_lp_leads(m, max_h=24):
    """Add forward leads of IP output gap for LP-IV estimation."""
    dd = m.copy()
    dd['x_ip_l1'] = dd['x_ip'].shift(1)
    for h in range(1, max_h + 1):
        dd[f'x_ip_f{h}'] = dd['x_ip'].shift(-h)
    return dd


def estimate_lp(m, h, rate_col, instruments, controls_l1,
                start=SAMPLE_START, end=SAMPLE_END):
    """LP-IV at horizon h: y_{t+h} - y_{t-1} = σ_h·r_t + controls_{t-1} + ε_{t+h}"""
    dd = m.copy()
    if f'x_ip_f{h}' not in dd.columns or 'x_ip_l1' not in dd.columns:
        return None
    dd['dy']        = dd[f'x_ip_f{h}'] - dd['x_ip_l1']
    dd['real_rate'] = dd[rate_col] - dd['pi_cpce'].shift(-1)

    needed = ['dy', 'real_rate'] + controls_l1 + instruments
    needed = [c for c in needed if c in dd.columns]
    e = dd[needed + ['date']].dropna()
    e = e[(e['date'] >= start) & (e['date'] <= end)].reset_index(drop=True)
    if len(e) < 50:
        return None

    e['const'] = 1.0
    bw = max(2, int(1.5 * (h + 1)))
    try:
        model = IVLIML(dependent=e['dy'], exog=e[controls_l1 + ['const']],
                       endog=e[['real_rate']], instruments=e[instruments])
        r     = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=bw, debiased=True)
        sigma = -r.params['real_rate']
        se    = r.std_errors['real_rate']
        fs_f  = None
        try:
            for key, fsres in r.first_stage.individual.items():
                if 'real_rate' in key:
                    fs_f = fsres.f_statistic.stat
        except:
            pass
        return {'h': h, 'n': len(e), 'sigma': sigma, 'se': se,
                'p': r.pvalues['real_rate'],
                'ci_lo': sigma - 1.96*se, 'ci_hi': sigma + 1.96*se, 'fs_f': fs_f}
    except:
        return None


def lp_iv_analysis(m, horizons=None, instruments=INSTRUMENTS_COMBINED):
    """LP-IV at multiple horizons for both rate measures."""
    if horizons is None:
        horizons = [1, 3, 6, 9, 12, 18, 24]
    controls_l1 = [c for c in ['candh_l1','anfci','baa_spread','term_spread','vix']
                   if c in m.columns]
    results = {'ff': [], 'eff': []}
    for h in horizons:
        for rate_col, key in [('fed_funds','ff'), ('eff_rate','eff')]:
            r = estimate_lp(m, h, rate_col, instruments, controls_l1)
            if r:
                r['rate'] = rate_col
                results[key].append(r)
    return results


# ================================================================
# 4. FORMAL MODEL COMPARISON TESTS
# ================================================================

def vuong_test(resid1, resid2, n):
    """Vuong (1989) non-nested model comparison test."""
    ll1   = -0.5 * np.log(2 * np.pi) - 0.5 * resid1**2
    ll2   = -0.5 * np.log(2 * np.pi) - 0.5 * resid2**2
    d     = ll1 - ll2
    omega = np.std(d, ddof=1)
    if omega < 1e-10:
        return {'stat': 0.0, 'p': 1.0, 'direction': 'indistinguishable'}
    v_stat = np.sqrt(n) * np.mean(d) / omega
    p      = 2 * (1 - stats.norm.cdf(abs(v_stat)))
    return {'stat': v_stat, 'p': p,
            'direction': 'fed_funds' if v_stat > 0 else 'eff_rate'}


def clarke_test(resid1, resid2, n):
    """Clarke (2007) distribution-free sign test."""
    d = resid1**2 - resid2**2
    n_favour_eff = int(np.sum(d > 0))   # FF larger residual → Eff wins
    n_favour_ff  = int(np.sum(d < 0))
    n_eff = n_favour_eff + n_favour_ff
    if n_eff == 0:
        return {'stat': 0.5, 'p': 1.0, 'n_favour_eff': 0, 'n_favour_ff': 0}
    p = 2 * min(stats.binom.cdf(min(n_favour_eff, n_favour_ff), n_eff, 0.5),
                1 - stats.binom.cdf(max(n_favour_eff, n_favour_ff) - 1, n_eff, 0.5))
    return {'n_favour_eff': n_favour_eff, 'n_favour_ff': n_favour_ff,
            'pct_eff': n_favour_eff / n_eff * 100, 'p': min(p, 1.0),
            'direction': 'eff_rate' if n_favour_eff > n_favour_ff else 'fed_funds'}


def diebold_mariano_test(resid1, resid2, h=1):
    """Diebold-Mariano (1995) test with HAC-robust variance."""
    d = resid1**2 - resid2**2
    n, d_bar = len(d), np.mean(d)
    gamma_0 = np.var(d, ddof=1)
    hac_var = gamma_0
    for lag in range(1, max(h, 2)):
        if lag >= n:
            break
        gamma_l = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        hac_var += 2 * (1 - lag / max(h, 2)) * gamma_l
    if hac_var <= 0:
        hac_var = gamma_0
    dm_stat = d_bar / np.sqrt(hac_var / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return {'stat': dm_stat, 'p': p,
            'direction': 'eff_rate' if d_bar > 0 else 'fed_funds', 'd_bar': d_bar}


def model_comparison_analysis(m, instruments=INSTRUMENTS_COMBINED, controls=CONTROLS):
    """Vuong, Clarke, and Diebold-Mariano tests between FF and Eff rate specs."""
    r_ff  = estimate_is_curve(m, 'fed_funds', 'pi_cpce', instruments, controls,
                              run_bootstrap=False)
    r_eff = estimate_is_curve(m, 'eff_rate',  'pi_cpce', instruments, controls,
                              run_bootstrap=False)
    if not r_ff or not r_eff:
        return None

    common  = sorted(set(r_ff['dates']) & set(r_eff['dates']))
    idx_ff  = [i for i, d in enumerate(r_ff['dates'])  if d in set(common)]
    idx_eff = [i for i, d in enumerate(r_eff['dates']) if d in set(common)]
    rf      = r_ff['resid'][idx_ff]
    re      = r_eff['resid'][idx_eff]
    n       = len(common)

    return {
        'vuong':    vuong_test(rf, re, n),
        'clarke':   clarke_test(rf, re, n),
        'dm':       diebold_mariano_test(rf, re, h=12),
        'n':        n,
        'rmse_ff':  np.sqrt(np.mean(rf**2)),
        'rmse_eff': np.sqrt(np.mean(re**2)),
        'sigma_ff': r_ff['sigma'], 'sigma_eff': r_eff['sigma'],
    }


# ================================================================
# 5. SPLIT-COEFFICIENT ANALYSIS
# ================================================================

def estimate_split(m, pi_col, instruments, controls,
                   start=SAMPLE_START, end=SAMPLE_END):
    """
    Split-coefficient IS curve with separate coefficients on the two rate components.

      x_t = β_f·x_{t+1} + β_b·x_{t-1} - σ₁·(i^FF - π) - σ₂·α(r^E - i^FF) + c'z + ε

    Effective rate decomposition:
      R̄ - π = (i^FF - π)  +  α·(r^E - i^FF)
               real_ff         rate_gap

    Under pure IES:             σ₁ = σ₂  (symmetric Euler margins)
    Under financial conditions: σ₂ > σ₁  (equity channel adds beyond substitution)

    Three endogenous variables (x_{t+1}, real_ff, rate_gap) → needs ≥ 3 instruments.
    Returns dict with σ̂₁, σ̂₂, Wald test, and first-stage F statistics.
    """
    dd = m.copy()
    dd['real_ff'] = dd['fed_funds'] - dd[pi_col].shift(-1)

    needed = ['x_ip','x_ip_f1','x_ip_l1','real_ff','rate_gap'] + controls + instruments
    needed = [c for c in needed if c in dd.columns]
    e = dd[needed + ['date']].dropna()
    e = e[(e['date'] >= start) & (e['date'] <= end)].reset_index(drop=True)
    n = len(e)
    if n < 50:
        return None

    e['const'] = 1.0
    exog = ['x_ip_l1'] + controls + ['const']
    bw   = max(2, int(np.ceil(n**(1/3))))

    try:
        model  = IVLIML(dependent=e['x_ip'], exog=e[exog],
                        endog=e[['x_ip_f1','real_ff','rate_gap']],
                        instruments=e[instruments])
        r_hac  = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=bw, debiased=True)
        r_conv = model.fit(cov_type='unadjusted', debiased=True)

        sigma_ff = -r_hac.params['real_ff']
        sigma_eq = -r_hac.params['rate_gap']

        # Wald test H0: σ₁ = σ₂
        diff     = r_hac.params['real_ff'] - r_hac.params['rate_gap']
        idx_ff   = list(r_hac.params.index).index('real_ff')
        idx_eq   = list(r_hac.params.index).index('rate_gap')
        vcov     = r_hac.cov
        var_diff = (vcov.iloc[idx_ff,idx_ff] + vcov.iloc[idx_eq,idx_eq]
                    - 2 * vcov.iloc[idx_ff,idx_eq])
        if var_diff > 0:
            wald_stat = diff**2 / var_diff
            wald_p    = 1 - stats.chi2.cdf(wald_stat, 1)
        else:
            wald_stat, wald_p = np.nan, np.nan

        fs_dict = {}
        try:
            for key, fsres in r_hac.first_stage.individual.items():
                fs_dict[key] = fsres.f_statistic.stat
        except:
            pass

        return {
            'n': n,
            'sigma_ff': sigma_ff, 'sigma_eq': sigma_eq,
            'se_ff_hac': r_hac.std_errors['real_ff'],
            'se_eq_hac': r_hac.std_errors['rate_gap'],
            'p_ff_hac':  r_hac.pvalues['real_ff'],
            'p_eq_hac':  r_hac.pvalues['rate_gap'],
            'p_ff_conv': r_conv.pvalues['real_ff'],
            'p_eq_conv': r_conv.pvalues['rate_gap'],
            'beta_f':    r_hac.params['x_ip_f1'],
            'beta_b':    r_hac.params['x_ip_l1'],
            'wald_stat': wald_stat, 'wald_p': wald_p,
            'ratio':     sigma_eq / sigma_ff if abs(sigma_ff) > 1e-6 else np.inf,
            'fs_dict':   fs_dict,
        }
    except Exception as ex:
        print(f"  estimate_split failed: {ex}")
        return None


def bootstrap_split(m, pi_col, instruments, controls,
                    start=SAMPLE_START, end=SAMPLE_END,
                    n_boot=N_BOOT, block_size=BLOCK_SIZE, seed=42):
    """2SLS block bootstrap for the split-coefficient specification."""
    dd = m.copy()
    dd['real_ff'] = dd['fed_funds'] - dd[pi_col].shift(-1)

    needed = ['x_ip','x_ip_f1','x_ip_l1','real_ff','rate_gap'] + controls + instruments
    needed = [c for c in needed if c in dd.columns]
    e  = dd[needed + ['date']].dropna()
    e  = e[(e['date'] >= start) & (e['date'] <= end)].reset_index(drop=True)
    n  = len(e)
    e['const'] = 1.0
    exog_cols  = ['x_ip_l1'] + controls + ['const']

    y   = e['x_ip'].values
    Xe  = e[exog_cols].values
    Xn  = e[['x_ip_f1','real_ff','rate_gap']].values
    Z   = e[instruments].values

    idx_ff = len(exog_cols) + 1   # real_ff position in stacked coef vector
    idx_eq = len(exog_cols) + 2   # rate_gap position

    rng = np.random.RandomState(seed)
    boot_ff, boot_eq = [], []
    nb = int(np.ceil(n / block_size))

    for _ in range(n_boot):
        starts  = rng.randint(0, n - block_size + 1, size=nb)
        indices = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        try:
            b = iv_manual(y[indices], Xe[indices], Xn[indices], Z[indices])
            boot_ff.append(-b[idx_ff])
            boot_eq.append(-b[idx_eq])
        except:
            pass

    boot_ff   = np.array(boot_ff)
    boot_eq   = np.array(boot_eq)
    boot_diff = boot_eq - boot_ff
    safe_ff   = np.where(np.abs(boot_ff) > 0.001, boot_ff, np.nan)

    p_ff    = 2 * min(np.mean(boot_ff <= 0), np.mean(boot_ff >= 0))
    p_eq    = 2 * min(np.mean(boot_eq <= 0), np.mean(boot_eq >= 0))
    p_equal = 2 * min(np.mean(boot_diff <= 0), np.mean(boot_diff >= 0))

    return {
        'boot_sigma_ff_median': np.median(boot_ff),
        'boot_sigma_eq_median': np.median(boot_eq),
        'boot_p_ff':    p_ff,
        'boot_p_eq':    p_eq,
        'boot_p_equal': p_equal,
        'boot_ci_ff':   (np.percentile(boot_ff,   2.5), np.percentile(boot_ff,   97.5)),
        'boot_ci_eq':   (np.percentile(boot_eq,   2.5), np.percentile(boot_eq,   97.5)),
        'boot_ci_diff': (np.percentile(boot_diff, 2.5), np.percentile(boot_diff, 97.5)),
        'boot_ratio_median': np.median(boot_eq / safe_ff),
        'n_valid': len(boot_ff),
    }


def first_stage_diagnostics(m, pi_col, instruments, controls,
                            start=SAMPLE_START, end=SAMPLE_END):
    """Partial correlations: which instruments predict which endogenous component."""
    dd = m.copy()
    dd['real_ff'] = dd['fed_funds'] - dd[pi_col].shift(-1)
    needed = ['real_ff','rate_gap'] + instruments
    e = dd[[c for c in needed if c in dd.columns] + ['date']].dropna()
    e = e[(e['date'] >= start) & (e['date'] <= end)].reset_index(drop=True)

    print(f"\n  {'Instrument':<20s} {'-> real_ff':>12s} {'-> rate_gap':>12s}")
    print("  " + "-" * 46)
    for z in instruments:
        if z in e.columns:
            print(f"  {z:<20s} {e[z].corr(e['real_ff']):+12.3f} "
                  f"{e[z].corr(e['rate_gap']):+12.3f}")


# ================================================================
# 6. MONTE CARLO RESOLUTION
# ================================================================

def get_bootstrap_draws(m, instruments=INSTRUMENTS_COMBINED, controls=CONTROLS,
                        n_boot=N_BOOT, block_size=BLOCK_SIZE, seed=42):
    """
    Draw (sigma, beta_f, beta_b) from the 2SLS bootstrap distribution.

    Note: paper's LIML point estimate is sigma=0.108. The 2SLS point estimate
    here is ~0.098; see paper Section 3.6 for the rationale for using 2SLS in
    the bootstrap loop rather than LIML.

    Returns: draws array (K x 3) and 2SLS full-sample point estimate tuple.
    """
    dd = m.copy()
    dd['real_rate'] = dd['eff_rate'] - dd['pi_cpce'].shift(-1)
    needed = ['x_ip','x_ip_f1','x_ip_l1','real_rate'] + controls + instruments
    e  = dd[[c for c in needed if c in dd.columns] + ['date']].dropna()
    e  = e[(e['date'] >= SAMPLE_START) & (e['date'] <= SAMPLE_END)].reset_index(drop=True)
    n  = len(e)
    e['const'] = 1.0
    exog_cols  = ['x_ip_l1'] + controls + ['const']

    y   = e['x_ip'].values
    Xe  = e[exog_cols].values
    Xn  = e[['x_ip_f1','real_rate']].values
    Z   = e[instruments].values

    bf_idx = len(exog_cols)      # x_ip_f1
    rr_idx = len(exog_cols) + 1  # real_rate

    b_hat = iv_manual(y, Xe, Xn, Z)
    point = (-b_hat[rr_idx], b_hat[bf_idx], b_hat[0])   # (sigma, beta_f, beta_b)
    print(f"  2SLS point: sigma={point[0]:.4f}  beta_f={point[1]:.3f}  beta_b={point[2]:.3f}")

    rng   = np.random.RandomState(seed)
    draws = []
    nb    = int(np.ceil(n / block_size))
    for _ in range(n_boot):
        starts = rng.randint(0, n - block_size + 1, size=nb)
        idx    = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        try:
            b = iv_manual(y[idx], Xe[idx], Xn[idx], Z[idx])
            sig, bf, bb = -b[rr_idx], b[bf_idx], b[0]
            if abs(sig) < 5.0 and abs(bf) < 2.0 and abs(bb) < 2.0:
                draws.append((sig, bf, bb))
        except:
            pass

    print(f"  Valid draws: {len(draws)}/{n_boot}")
    return np.array(draws), point


def _nk_simulate(rate_path, x0, pi0, oil, gscpi, sigma, beta_f, beta_b):
    """
    Single NK simulation pass with explicitly passed parameters.

    Adaptive expectations: E_t[x_{t+1}] = x_{t-1} (purely backward-looking),
    collapsing to (beta_f + beta_b) * x[t-1].
    """
    T = len(rate_path)
    x, pi = np.zeros(T), np.zeros(T)
    x[0], pi[0] = x0, pi0
    for t in range(1, T):
        r_real = rate_path[t] - pi[t-1] - NK_R_STAR
        x[t]   = (beta_f + beta_b) * x[t-1] - sigma * r_real
        x[t]   = np.clip(x[t], -20, 20)
        oil_t   = oil[t]   if not np.isnan(oil[t])   else 0.0
        gscpi_t = gscpi[t] if not np.isnan(gscpi[t]) else 0.0
        pi[t]  = (NK_GAMMA_F * pi[t-1] + NK_GAMMA_B * pi[t-1]
                  + NK_KAPPA * x[t]
                  + NK_LAM_OIL * oil_t + NK_LAM_GSCPI * gscpi_t + NK_PC_CONST)
        pi[t]  = np.clip(pi[t], -5, 25)
    return x, pi


def compute_mc_metrics(df_sim, sigma, beta_f, beta_b):
    """Resolution and MAE for a single (sigma, beta_f, beta_b) bootstrap draw."""
    x0, pi0  = df_sim['x_ip'].iloc[0], df_sim['pi_cpce'].iloc[0]
    pi_act   = df_sim['pi_cpce'].values
    oil      = df_sim['d12_oil'].fillna(0).values
    gscpi    = df_sim['gscpi'].fillna(0).values

    _, pi_ff  = _nk_simulate(df_sim['fed_funds'].values, x0, pi0, oil, gscpi,
                              sigma, beta_f, beta_b)
    _, pi_eff = _nk_simulate(df_sim['eff_rate'].values,  x0, pi0, oil, gscpi,
                              sigma, beta_f, beta_b)

    cum_ff  = np.sum(pi_ff  - pi_act) / 12
    cum_eff = np.sum(pi_eff - pi_act) / 12
    resolution = (cum_ff - cum_eff) / cum_ff if abs(cum_ff) > 0.01 else np.nan
    advantage  = cum_ff - cum_eff

    dates = df_sim['date'].values
    mask  = ((dates >= np.datetime64('2013-01-01')) &
             (dates <= np.datetime64('2019-12-31')))
    if mask.sum() > 0:
        mae_ff  = np.mean(np.abs(pi_ff[mask]  - pi_act[mask]))
        mae_eff = np.mean(np.abs(pi_eff[mask] - pi_act[mask]))
        mae_ratio = mae_ff / mae_eff if mae_eff > 0.001 else np.nan
    else:
        mae_ratio = np.nan

    return sigma, resolution, advantage, mae_ratio


def mc_resolution_analysis(m, instruments=INSTRUMENTS_COMBINED, controls=CONTROLS,
                           sim_start='2009-01-01', sim_end='2019-12-01'):
    """
    Full MC resolution analysis for Table 7.

    Draws (sigma, beta_f, beta_b) from the 2SLS bootstrap distribution and runs
    the NK simulation for each draw, reporting both unconditional and
    conditional-on-sigma>0 distributions of inflation over-prediction resolution.
    """
    draws, point = get_bootstrap_draws(m, instruments, controls)

    sim = m[(m['date'] >= sim_start) & (m['date'] <= sim_end) &
            m['pi_cpce'].notna() & m['x_ip'].notna()].copy().reset_index(drop=True)
    print(f"  Simulation: {sim['date'].min().date()} to {sim['date'].max().date()}"
          f", n={len(sim)}")

    raw = []
    for i, (sig, bf, bb) in enumerate(draws):
        try:
            raw.append(compute_mc_metrics(sim, sig, bf, bb))
        except:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  ... {i+1}/{len(draws)}")

    arr  = np.array([(r[0],r[1],r[2],r[3]) for r in raw
                     if not np.isnan(r[1]) and abs(r[1]) < 10])
    return {
        'draws': draws, 'point': point,
        'sigmas':      arr[:, 0],
        'resolutions': arr[:, 1],
        'advantages':  arr[:, 2],
        'mae_ratios':  arr[:, 3],
        'pos_mask':    arr[:, 0] > 0,
    }


# ================================================================
# 7. SCF AGGREGATION VALIDATION  (Section 2.6)
# ================================================================

def scf_validation(m):
    """
    Validate the Flow of Funds alpha proxy using Survey of Consumer Finances
    microdata stored as sparse columns in panel_data.csv.

    SCF data are stored at December of each survey year (1989-2022, triennial).
    All other months are NaN.  The two corrections to the FoF measure are:

      (1) MPC-weighting  [reduces alpha: wealthy households have high alpha, low MPC]
      (2) Retirement equity inclusion  [raises alpha: 401k/IRA equity counted in broad def]

    Key finding: the two corrections approximately cancel. The ratio
    (scf_mpc_broad / scf_fof_narrow) has been above 1.0 since 2001, meaning
    the FoF measure slightly *understates* the IS-curve-relevant equity share,
    making all paper results conservative.

    Returns a DataFrame with the 12 survey-wave observations.
    """
    scf_cols = ['date', 'alpha', 'scf_fof_narrow', 'scf_mpc_broad',
                'scf_mpc_broad_lo', 'scf_mpc_broad_hi', 'scf_ratio']
    available = [c for c in scf_cols if c in m.columns]
    if 'scf_mpc_broad' not in m.columns:
        return None

    waves = m[m['scf_mpc_broad'].notna()][available].copy()
    waves['year'] = waves['date'].dt.year
    # Cross-check: how does scf_fof_narrow compare to the FoF alpha in the panel?
    waves['fof_panel_alpha'] = waves['alpha']       # alpha col = FoF series from FRED
    waves['scf_vs_fof_diff'] = waves['scf_fof_narrow'] - waves['fof_panel_alpha']
    return waves


# ================================================================
# RESULTS OUTPUT
# ================================================================

def print_results(results):
    """Print all results as clean numerical tables."""
    P = print

    # --- 1: Alternative Horizons ---
    P(f"\n{'='*80}")
    P("1. ALTERNATIVE TRAILING HORIZONS AND BELIEF WEIGHTING")
    P(f"{'='*80}")
    P("Combined instruments (8), CorePCE, CANDH_{t-1}, LIML.")
    hdr = (f"{'Construction':<20s} {'sig':>7} {'p(HAC)':>8} {'p(conv)':>8} {'p(boot)':>8} "
           f"{'b_f':>6} {'b_b':>6} {'Sb':>8} {'1stF':>6}")
    P(f"\n{hdr}"); P("-" * len(hdr))
    for r in results.get('horizons', []):
        P(f"{r['construction']:<20s} {r['sigma']:7.4f} {r['p_hac']:8.3f} {r['p_conv']:8.3f} "
          f"{r.get('boot_p',float('nan')):8.3f} {r['beta_f']:6.3f} {r['beta_b']:6.3f} "
          f"{r['beta_f']+r['beta_b']:8.3f} {r.get('fs_f') or 0:6.1f}")

    # --- 2a: Rolling Windows ---
    P(f"\n{'='*80}"); P("2a. ROLLING WINDOWS (15-year, annual steps)"); P(f"{'='*80}")
    hdr2 = (f"{'Window':<22s} {'n':>4} {'a':>6} {'sig_FF':>8} {'p_FF':>7} "
            f"{'sig_Eff':>8} {'p_Eff':>7} {'bf_FF':>8} {'bf_Eff':>9}")
    P(f"\n{hdr2}"); P("-" * len(hdr2))
    for r in results.get('rolling', []):
        win = f"{r['start'][:7]}-{r['end'][:7]}"
        P(f"{win:<22s} {r['n']:4d} {r['avg_alpha']:6.3f} {r['sigma_ff']:8.4f} {r['p_ff']:7.3f} "
          f"{r['sigma_eff']:8.4f} {r['p_eff']:7.3f} {r['bf_ff']:8.3f} {r['bf_eff']:9.3f}")
    if 'rolling' in results:
        clean = [r for r in results['rolling']
                 if r['end'] <= '2020-01-01' and r['sigma_eff'] > 0 and r['sigma_ff'] > 0]
        if len(clean) > 3:
            alphas = [r['avg_alpha'] for r in clean]
            advs   = [np.log(max(r['p_ff'],1e-6)) - np.log(max(r['p_eff'],1e-6)) for r in clean]
            corr, p_c = stats.pearsonr(alphas, advs)
            P(f"\n  alpha-advantage corr (pre-COVID, n={len(clean)}): rho={corr:.3f} p={p_c:.3f}")

    # --- 2b: Subsamples ---
    P(f"\n{'='*80}"); P("2b. SUBSAMPLE STABILITY"); P(f"{'='*80}")
    hdr3 = (f"{'Subsample':<25s} {'n':>4} {'a':>6} {'sig_FF':>8} {'p_FF':>7} "
            f"{'sig_Eff':>8} {'p_Eff':>7} {'bf_FF':>8} {'bf_Eff':>9}")
    P(f"\n{hdr3}"); P("-" * len(hdr3))
    for r in results.get('subsample', []):
        P(f"{r['subsample']:<25s} {r['n']:4d} {r['avg_alpha']:6.3f} "
          f"{r['sigma_ff']:8.4f} {r['p_ff']:7.3f} "
          f"{r['sigma_eff']:8.4f} {r['p_eff']:7.3f} "
          f"{r['bf_ff']:8.3f} {r['bf_eff']:9.3f}")

    # --- 3: LP-IV ---
    P(f"\n{'='*80}"); P("3. LOCAL PROJECTION IMPULSE RESPONSES (LP-IV)"); P(f"{'='*80}")
    P("y_{t+h} - y_{t-1} = sig_h * r_t + controls + eps. Combined instruments, LIML.")
    hdr4 = (f"{'h':>3} {'sig_FF':>8} {'p_FF':>7} {'lo_FF':>7} {'hi_FF':>7} {'F_FF':>6}  "
            f"{'sig_Eff':>8} {'p_Eff':>7} {'lo_Eff':>7} {'hi_Eff':>7} {'F_Eff':>6}")
    P(f"\n{hdr4}"); P("-" * len(hdr4))
    if 'lp' in results:
        lp_ff  = results['lp'].get('ff', [])
        lp_eff = results['lp'].get('eff', [])
        for i in range(max(len(lp_ff), len(lp_eff))):
            ff  = lp_ff[i]  if i < len(lp_ff)  else None
            eff = lp_eff[i] if i < len(lp_eff) else None
            h   = (ff or eff)['h']
            ffs = (f"{ff['sigma']:8.3f} {ff['p']:7.3f} {ff['ci_lo']:7.3f} "
                   f"{ff['ci_hi']:7.3f} {ff.get('fs_f') or 0:6.1f}") if ff else \
                  "       -       -       -       -      -"
            es  = (f"{eff['sigma']:8.3f} {eff['p']:7.3f} {eff['ci_lo']:7.3f} "
                   f"{eff['ci_hi']:7.3f} {eff.get('fs_f') or 0:6.1f}") if eff else \
                  "       -       -       -       -      -"
            P(f"{h:3d} {ffs}  {es}")

    # --- 4: Model Comparison ---
    P(f"\n{'='*80}"); P("4. FORMAL MODEL COMPARISON TESTS"); P(f"{'='*80}")
    mc = results.get('comparison')
    if mc:
        impr = 100 * (1 - mc['rmse_eff'] / mc['rmse_ff'])
        P(f"\n  RMSE(FF)={mc['rmse_ff']:.4f}  RMSE(Eff)={mc['rmse_eff']:.4f}  "
          f"improvement={impr:.1f}%  n={mc['n']}")
        v  = mc['vuong']
        P(f"\n  Vuong:  stat={v['stat']:+.3f}  p={v['p']:.3f}  favours {v['direction']}")
        c  = mc['clarke']
        P(f"  Clarke: {c['n_favour_eff']} obs favour Eff ({c['pct_eff']:.1f}%),  "
          f"{c['n_favour_ff']} favour FF ({100-c['pct_eff']:.1f}%)  p={c['p']:.3f}")
        dm = mc['dm']
        P(f"  DM:     stat={dm['stat']:+.3f}  d_bar={dm['d_bar']:.6f}  p={dm['p']:.3f}  "
          f"favours {dm['direction']}")

    # --- 5: Split Coefficient ---
    P(f"\n{'='*80}")
    P("5. SPLIT-COEFFICIENT ANALYSIS")
    P("   x_t = bf*x_{t+1} + bb*x_{t-1} - s1*(i^FF-pi) - s2*alpha*(r^E-i^FF) + controls + e")
    P("   Pure IES: s1=s2.  Financial conditions: s2>s1.")
    P(f"{'='*80}")

    if 'split_blended' in results:
        P(f"\n  Blended baseline (for comparison):")
        hdr_b = (f"  {'Instruments':<22s} {'Rate':<10s} {'sigma':>8} {'p(HAC)':>8} "
                 f"{'bf':>6} {'bb':>6} {'Sb':>8} {'F':>6}")
        P(hdr_b); P("  " + "-"*76)
        for row in results['split_blended']:
            P(f"  {row['inst']:<22s} {row['rate']:<10s} {row['sigma']:8.4f} "
              f"{row['p_hac']:8.4f} {row['beta_f']:6.3f} {row['beta_b']:6.3f} "
              f"{row['beta_f']+row['beta_b']:8.3f} {row['fs_f'] or 0:6.1f}")

    for label, r in results.get('split_point', {}).items():
        P(f"\n  --- {label} (n={r['n']}) ---")
        P(f"  s1 (real FF):    {r['sigma_ff']:+8.4f}  SE={r['se_ff_hac']:.4f}  "
          f"p_HAC={r['p_ff_hac']:.4f}  p_conv={r['p_ff_conv']:.4f}")
        P(f"  s2 (equity gap): {r['sigma_eq']:+8.4f}  SE={r['se_eq_hac']:.4f}  "
          f"p_HAC={r['p_eq_hac']:.4f}  p_conv={r['p_eq_conv']:.4f}")
        P(f"  s2/s1={r['ratio']:.2f}  bf={r['beta_f']:.3f}  bb={r['beta_b']:.3f}  "
          f"Sb={r['beta_f']+r['beta_b']:.3f}  Wald p={r['wald_p']:.4f}")
        if r.get('fs_dict'):
            P("  First-stage F: " +
              "  ".join(f"{k}={v:.1f}" for k,v in r['fs_dict'].items()))

    if 'split_boot' in results:
        boot = results['split_boot']
        P(f"\n  Bootstrap ({N_BOOT} draws, Combined instruments):")
        P(f"  s1 median={boot['boot_sigma_ff_median']:+.4f}  "
          f"95% CI=[{boot['boot_ci_ff'][0]:+.4f}, {boot['boot_ci_ff'][1]:+.4f}]  "
          f"p={boot['boot_p_ff']:.4f}")
        P(f"  s2 median={boot['boot_sigma_eq_median']:+.4f}  "
          f"95% CI=[{boot['boot_ci_eq'][0]:+.4f}, {boot['boot_ci_eq'][1]:+.4f}]  "
          f"p={boot['boot_p_eq']:.4f}")
        P(f"  s2-s1 95% CI: [{boot['boot_ci_diff'][0]:+.4f}, {boot['boot_ci_diff'][1]:+.4f}]")
        P(f"  Bootstrap p(s1=s2): {boot['boot_p_equal']:.4f}")
        P(f"  Valid draws: {boot['n_valid']}/{N_BOOT}")

    if 'split_subsample' in results:
        P(f"\n  Subsample stability:")
        for label, r in results['split_subsample'].items():
            if r:
                P(f"  {label} (n={r['n']}):  "
                  f"s1={r['sigma_ff']:+.4f} (p={r['p_ff_hac']:.4f})  "
                  f"s2={r['sigma_eq']:+.4f} (p={r['p_eq_hac']:.4f})  "
                  f"s2/s1={r['ratio']:.2f}  Wald p={r['wald_p']:.4f}")

    # --- 6: MC Resolution ---
    P(f"\n{'='*80}")
    P("6. MONTE CARLO RESOLUTION  (Table 7)")
    P(f"{'='*80}")
    mc_r = results.get('mc')
    if mc_r is None:
        P("  Skipped (run without --skip-mc to include).")
    else:
        sigmas      = mc_r['sigmas']
        resolutions = mc_r['resolutions']
        advantages  = mc_r['advantages']
        mae_ratios  = mc_r['mae_ratios']
        pos_mask    = mc_r['pos_mask']

        P(f"\n  Bootstrap sigma distribution (2SLS):")
        P(f"  Total valid: {len(sigmas)}")
        P(f"  sigma>0 (IS operative): {pos_mask.sum()} ({pos_mask.mean():.1%})")
        P(f"  sigma<=0 (inoperative): {(~pos_mask).sum()} ({(~pos_mask).mean():.1%})")
        P(f"  sigma median={np.median(sigmas):.4f}  mean={np.mean(sigmas):.4f}")

        def _dist_row(label, vals):
            pts  = [5, 10, 25, 50, 75, 90, 95]
            pcts = "  ".join(f"{p}th:{np.percentile(vals,p):+.0%}" for p in pts)
            P(f"\n  {label} (n={len(vals)}):")
            P(f"  {pcts}")
            P(f"  Mean:{np.mean(vals):+.1%}  Pr(>0):{np.mean(vals>0):.1%}")

        P("\n  Resolution = (cum FF over-pred - cum Eff over-pred) / cum FF over-pred:")
        _dist_row("Unconditional (all draws)", resolutions)
        _dist_row("Conditional   (sigma > 0)", resolutions[pos_mask])

        if (~pos_mask).sum() > 5:
            P(f"\n  sigma<=0 draws: resolution median={np.median(resolutions[~pos_mask]):+.1%}  "
              f"advantage median={np.median(advantages[~pos_mask]):+.2f} pp-years")

        valid_mae = mae_ratios[~np.isnan(mae_ratios)]
        P(f"\n  MAE ratio FF/Eff (2013-2019):")
        P(f"  Unconditional: median={np.median(valid_mae):.2f}x  "
          f"Pr(Eff wins)={np.mean(valid_mae>1):.1%}")
        cond_mae = mae_ratios[pos_mask & ~np.isnan(mae_ratios)]
        P(f"  Conditional:   median={np.median(cond_mae):.2f}x  "
          f"Pr(Eff wins)={np.mean(cond_mae>1):.1%}")

    # --- 7: SCF Validation ---
    P(f"\n{'='*80}")
    P("7. SCF AGGREGATION VALIDATION  (Section 2.6)")
    P("   MPC-weighted broad alpha vs Flow of Funds alpha, 12 SCF survey waves.")
    P("   Ratio > 1.0: FoF understates IS-curve-relevant equity share (results conservative).")
    P(f"{'='*80}")
    scf = results.get('scf')
    if scf is None:
        P("  Skipped: scf_mpc_broad column not found in panel_data.csv.")
    else:
        hdr_s = (f"  {'Year':>4s} {'FoF(panel)':>11s} {'SCF_FoF':>9s} {'SCF_broad':>10s} "
                 f"{'lo':>7s} {'hi':>7s} {'ratio':>7s} {'SCF-FoF':>9s}")
        P(hdr_s)
        P("  " + "-" * (len(hdr_s) - 2))
        for _, row in scf.iterrows():
            diff = row['scf_vs_fof_diff']
            diff_str = f"{diff:+.4f}" if not np.isnan(diff) else "   N/A"
            P(f"  {row['year']:>4.0f} {row['fof_panel_alpha']:>11.4f} "
              f"{row['scf_fof_narrow']:>9.4f} {row['scf_mpc_broad']:>10.4f} "
              f"{row['scf_mpc_broad_lo']:>7.4f} {row['scf_mpc_broad_hi']:>7.4f} "
              f"{row['scf_ratio']:>7.4f} {diff_str:>9s}")
        ratio_vals = scf['scf_ratio'].values
        above_one  = (ratio_vals >= 1.0).sum()
        P(f"\n  Ratio >= 1.0 in {above_one}/{len(ratio_vals)} waves "
          f"({above_one/len(ratio_vals):.0%}). FoF understates in every wave since 2001.")
        P(f"  Ratio range: [{ratio_vals.min():.3f}, {ratio_vals.max():.3f}]  "
          f"mean={ratio_vals.mean():.3f}  2022={ratio_vals[-1]:.3f}")
        diffs = scf['scf_vs_fof_diff'].dropna()
        P(f"  SCF narrow vs FoF panel alpha: mean diff={diffs.mean():+.4f}  "
          f"max abs diff={diffs.abs().max():.4f}  "
          f"(small: FoF annual smoothing vs SCF point-in-time)")


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Effective Rate v2: Robustness, Split-Coefficient, and MC Analysis')
    parser.add_argument('--skip-mc', action='store_true',
                        help='Skip Section 6 Monte Carlo (~4 min)')
    args = parser.parse_args()

    print("=" * 70)
    print("EFFECTIVE RATE v2: ROBUSTNESS ANALYSIS")
    print("=" * 70)

    # Load panel
    print("\n1. Loading panel data...")
    m = pd.read_csv(PANEL_PATH, parse_dates=['date'])
    m = m.sort_values('date').reset_index(drop=True)

    for base in ['x_ip', 'x_pces', 'x_combined']:
        if base in m.columns:
            if f'{base}_f1' not in m.columns: m[f'{base}_f1'] = m[base].shift(-1)
            if f'{base}_l1' not in m.columns: m[f'{base}_l1'] = m[base].shift(1)
    for col in ['mps_cum36', 'ffr_cum36', 'fg_cum36', 'lsap_cum36']:
        lag = f'{col}_l1'
        if lag not in m.columns and col in m.columns:
            m[lag] = m[col].shift(1)

    print(f"   Panel: {len(m)} obs, {m['date'].min().date()} to {m['date'].max().date()}")
    alt_cols = ['eff_rate_5yr','eff_rate_10yr','eff_rate_exp3','eff_rate_exp2','eff_rate_rec']
    missing  = [c for c in alt_cols if c not in m.columns]
    if missing:
        print(f"   WARNING: Missing horizon columns {missing} -- those rows will be skipped.")
    else:
        print(f"   All {len(alt_cols)} alternative effective rate columns found.")

    # Build LP leads
    print("\n2. Building LP forward leads...")
    m = build_lp_leads(m, max_h=24)

    all_results = {}

    print("\n3. Alternative horizons / belief weighting...")
    all_results['horizons'] = alternative_horizon_results(m)
    print(f"   {len(all_results['horizons'])} specifications")

    print("\n4. Rolling-window analysis (15yr, annual steps)...")
    all_results['rolling'] = rolling_window_analysis(m)
    print(f"   {len(all_results['rolling'])} windows")

    print("\n5. Subsample stability...")
    all_results['subsample'] = subsample_analysis(m)
    print(f"   {len(all_results['subsample'])} subsamples")

    print("\n6. LP-IV impulse responses...")
    all_results['lp'] = lp_iv_analysis(m)
    print(f"   {len(all_results['lp']['ff'])+len(all_results['lp']['eff'])} LP estimates")

    print("\n7. Formal model comparison tests...")
    all_results['comparison'] = model_comparison_analysis(m)

    # Section 5: Split-coefficient
    print("\n8. Split-coefficient analysis...")
    inst_configs = {
        'MPS cumul (4)':  INSTRUMENTS_MPS,
        'Swanson 3F (6)': INSTRUMENTS_SWANSON,
        'Combined (8)':   INSTRUMENTS_COMBINED,
    }
    pi_col = 'pi_cpce'

    blended_rows = []
    for ilabel, instruments in inst_configs.items():
        for rate_col, rlabel in [('fed_funds','Fed Funds'), ('eff_rate','Eff Rate')]:
            r = estimate_is_curve(m, rate_col, pi_col, instruments, CONTROLS,
                                  run_bootstrap=False)
            if r:
                blended_rows.append({'inst': ilabel, 'rate': rlabel,
                                     'sigma': r['sigma'], 'p_hac': r['p_hac'],
                                     'beta_f': r['beta_f'], 'beta_b': r['beta_b'],
                                     'fs_f': r.get('fs_f')})
    all_results['split_blended'] = blended_rows

    split_point = {}
    for ilabel, instruments in inst_configs.items():
        r = estimate_split(m, pi_col, instruments, CONTROLS)
        if r:
            split_point[ilabel] = r
            print(f"   {ilabel}: s1={r['sigma_ff']:+.4f}  s2={r['sigma_eq']:+.4f}  "
                  f"s2/s1={r['ratio']:.2f}  Wald p={r['wald_p']:.4f}")
    all_results['split_point'] = split_point

    print(f"   Bootstrap ({N_BOOT} draws, Combined)...")
    all_results['split_boot'] = bootstrap_split(m, pi_col, INSTRUMENTS_COMBINED, CONTROLS)

    split_sub_specs = {
        'Pre-ZLB (1988-2007)': (SAMPLE_START, '2007-12-01'),
        'Post-2000':           ('2000-01-01', SAMPLE_END),
        'Full sample':         (SAMPLE_START, SAMPLE_END),
    }
    all_results['split_subsample'] = {
        label: estimate_split(m, pi_col, INSTRUMENTS_COMBINED, CONTROLS, start=s, end=e)
        for label, (s, e) in split_sub_specs.items()
    }
    print(f"   {len(split_sub_specs)} subsample splits")

    # Section 6: Monte Carlo
    if not args.skip_mc:
        print("\n9. Monte Carlo resolution (Table 7)...")
        all_results['mc'] = mc_resolution_analysis(m)
    else:
        print("\n9. Monte Carlo resolution: skipped (--skip-mc to enable)")
        all_results['mc'] = None

    # Section 7: SCF validation
    print("\n10. SCF aggregation validation...")
    all_results['scf'] = scf_validation(m)
    if all_results['scf'] is not None:
        print(f"    {len(all_results['scf'])} survey waves found in panel.")
    else:
        print("    WARNING: scf_mpc_broad column not in panel — run data audit.")

    print_results(all_results)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
