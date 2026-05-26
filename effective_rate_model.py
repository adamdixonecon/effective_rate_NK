"""
Effective Rate v2: Two-Asset NK IS Curve Estimation
════════════════════════════════════════════════════
Reads panel_data.csv and runs IS curve estimation.

Model: x_t = β_f·x_{t+1} + β_b·x_{t-1} - σ·r^e_t + controls + ε_t
       R̄_t = (1-ᾱ_t)·i^FF_t + ᾱ_t·r^E_t

Data pipeline (data_pipeline.py) builds panel_data.csv from FRED/Shiller.
This script performs estimation only — no network access required.

KEY METHODOLOGICAL DECISIONS:
  - Monthly frequency with IP output gap (HP λ=129,600)
  - Effective rate: weighted average, 7yr trailing return, 12m MA, 8Q alpha MA
  - Cumulative Bauer-Swanson MPS_ORTH as primary instruments
  - Extended Swanson FFR/FG/LSAP factors via subspace identification
  - Controls: ANFCI, BAA spread, term spread, VIX, CANDH_{t-1}
  - LIML estimator (robust to many-instruments bias)
  - HAC: conservative (n^{1/3}), auto-bandwidth, and conventional reported
  - Block bootstrap p-values

KEY FINDINGS:
  - Combined instruments: Eff Rate Brown's HAC p=0.050, FF p=0.632
  - Swanson 3F: FF gets σ̂≈0; only Eff Rate absorbs LSAP
  - β_f universally higher with Eff Rate (0.266 vs 0.155)
  - σ̂ ratio: 5-6× (Eff/FF) with decomposed instruments
  - PCE services confirms: 10:1 σ̂ ratio, β_f=0.567
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from linearmodels.iv import IVLIML
from numpy.linalg import lstsq, eigh
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# CONFIGURATION
# ================================================================

PANEL_PATH = Path(__file__).parent / 'panel_data.csv'

# Bootstrap
N_BOOT = 3000
BLOCK_SIZE = 12

# Sample
SAMPLE_START = '1988-07-01'
SAMPLE_END = '2023-12-01'


# ================================================================
# ESTIMATION UTILITIES
# ================================================================

def iv_manual(y, Xe, Xn, Z):
    """Manual 2SLS for bootstrap resampling.

    The paper's point estimates use IVLIML (median-unbiased under weak instruments).
    2SLS is used here for bootstrap draws for two reasons that go beyond tractability:

    1. LIML lacks finite moments under weak instruments. Its distribution has heavy
       tails driven by the minimum eigenvalue of a random matrix. In block-resampled
       draws, a fraction of samples produce near-singular configurations where LIML
       either diverges or returns extreme values, contaminating the bootstrap
       distribution and inflating confidence intervals.

    2. The bootstrap's purpose is characterising the shape of the sampling
       distribution (for p-values and CIs), not point estimation. 2SLS is consistent
       under both H0 and Ha and its bootstrap distribution converges to the same
       asymptotic limit as LIML's. Using 2SLS in the loop does not invalidate the
       bootstrap inference; this is standard practice in the weak-instruments
       literature (Andrews-Stock-Moreira, Kleibergen LM framework).

    The 2SLS full-sample point estimate is σ̂≈0.098 vs the LIML estimate of 0.108
    (bias ≈ k/n = 8/430 ≈ 2%, as expected). This produces a simulation resolution
    of ~46% vs the 48% the paper states for the LIML point estimate; the bootstrap
    median (41%) and p-values are unaffected by this difference in levels.
    """
    W = np.column_stack([Xe, Z])
    X = np.column_stack([Xe, Xn])
    Pw = W @ lstsq(W, X, rcond=None)[0]
    return lstsq(Pw, y, rcond=None)[0]


def correct_boot_p(betas, rr_idx):
    """Correct bootstrap p-value for H0: σ = 0.
    σ̂ = -β_real_rate, so test whether -β* crosses zero."""
    boot_sigmas = -betas[:, rr_idx]
    return 2 * min(np.mean(boot_sigmas <= 0), np.mean(boot_sigmas >= 0))


def block_bootstrap(y, Xe, Xn, Z, rr_idx, n_boot=N_BOOT, block_size=BLOCK_SIZE, seed=42):
    """Block bootstrap with corrected p-values."""
    n = len(y)
    rng = np.random.RandomState(seed)
    beta_hat = iv_manual(y, Xe, Xn, Z)
    
    betas = []
    nb = int(np.ceil(n / block_size))
    for _ in range(n_boot):
        starts = rng.randint(0, n - block_size + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        try:
            betas.append(iv_manual(y[idx], Xe[idx], Xn[idx], Z[idx]))
        except:
            pass
    betas = np.array(betas)
    
    boot_p = correct_boot_p(betas, rr_idx)
    ci_lo = -np.percentile(betas[:, rr_idx], 97.5)
    ci_hi = -np.percentile(betas[:, rr_idx], 2.5)
    boot_se = np.std(betas[:, rr_idx])
    
    return boot_p, ci_lo, ci_hi, boot_se


def browns_method(p_values, corr_matrix):
    """Brown's method for combining correlated p-values."""
    k = len(p_values)
    T = -2 * np.sum(np.log(np.maximum(p_values, 1e-15)))
    E_T = 2 * k
    extra = sum(2 * (3.263*corr_matrix[i,j] + 0.710*corr_matrix[i,j]**2 + 
                     0.027*corr_matrix[i,j]**3)
                for i in range(k) for j in range(i+1, k))
    Var_T = 4*k + extra
    c = Var_T / (2 * E_T)
    f = 2 * E_T**2 / Var_T
    return 1 - stats.chi2.cdf(T/c, f), f, c


# ================================================================
# IS CURVE ESTIMATION
# ================================================================

def estimate_is_curve(m, rate_col, pi_col, instruments, controls,
                      start=SAMPLE_START, end=SAMPLE_END, run_bootstrap=True,
                      y_col='x_ip', y_f1='x_ip_f1', y_l1='x_ip_l1'):
    """
    Estimate monthly IS curve via LIML with HAC and bootstrap.
    
    x_t = β_f·x_{t+1} + β_b·x_{t-1} - σ·r^e_t + controls + ε_t
    r^e_t instrumented by cumulative monetary policy shocks.
    
    Use y_col/y_f1/y_l1 to switch output measure:
      IP (default): y_col='x_ip', y_f1='x_ip_f1', y_l1='x_ip_l1'
      Services:     y_col='x_pces', y_f1='x_pces_f1', y_l1='x_pces_l1'
      Combined:     y_col='x_combined', y_f1='x_combined_f1', y_l1='x_combined_l1'
    """
    dd = m.copy()
    dd['real_rate'] = dd[rate_col] - dd[pi_col].shift(-1)
    
    needed = [y_col, y_f1, y_l1, 'real_rate'] + controls + instruments
    needed = [c for c in needed if c in dd.columns]
    e = dd[needed + ['date']].dropna()
    e = e[(e['date'] >= start) & (e['date'] <= end)].reset_index(drop=True)
    n = len(e)
    if n < 50:
        return None
    
    e['const'] = 1.0
    exog = [y_l1] + controls + ['const']
    bw_cons = max(2, int(np.ceil(n**(1/3))))
    
    model = IVLIML(dependent=e[y_col], exog=e[exog],
                   endog=e[[y_f1,'real_rate']],
                   instruments=e[instruments])
    
    r_c = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=bw_cons, debiased=True)
    r_a = model.fit(cov_type='kernel', kernel='bartlett', debiased=True)
    r_v = model.fit(cov_type='unadjusted', debiased=True)
    
    result = {
        'n': n,
        'sigma': -r_c.params['real_rate'],
        'beta_f': r_c.params[y_f1],
        'beta_b': r_c.params[y_l1],
        'se_sigma_cons': r_c.std_errors['real_rate'],
        'p_sigma_cons': r_c.pvalues['real_rate'],
        'p_sigma_auto': r_a.pvalues['real_rate'],
        'p_sigma_conv': r_v.pvalues['real_rate'],
        'se_bf': r_c.std_errors[y_f1],
        'p_bf': r_c.pvalues[y_f1],
    }
    
    # First-stage F
    try:
        for key, fsres in r_c.first_stage.individual.items():
            if 'real_rate' in key:
                result['fs_f'] = fsres.f_statistic.stat
    except:
        result['fs_f'] = None
    
    # Overidentification test
    try:
        result['j_pval'] = r_c.wooldridge_overid.pval
    except:
        result['j_pval'] = None
    
    # Bootstrap
    if run_bootstrap:
        y = e[y_col].values
        Xe = e[exog].values
        Xn = e[[y_f1,'real_rate']].values
        Z = e[instruments].values
        rr_idx = len(exog) + 1
        
        bp, ci_lo, ci_hi, bse = block_bootstrap(y, Xe, Xn, Z, rr_idx)
        result.update({'boot_p': bp, 'boot_ci_lo': ci_lo, 'boot_ci_hi': ci_hi, 'boot_se': bse})
    
    return result


# ================================================================
# LOCAL PROJECTIONS
# ================================================================

def estimate_lp(m, h, rate_change_col, instruments, controls_l1,
                start=SAMPLE_START, end=SAMPLE_END):
    """
    Gertler-Karadi style LP-IV at horizon h.
    y_{t+h} - y_{t-1} = σ_h·Δr_t + controls_{t-1} + ε_{t+h}
    """
    dd = m.copy()
    dd['dy'] = dd[f'x_ip_f{h}'] - dd['x_ip_l1']
    
    needed = ['dy', rate_change_col] + controls_l1 + instruments
    needed = [c for c in needed if c in dd.columns]
    e = dd[needed + ['date']].dropna()
    e = e[(e['date'] >= start) & (e['date'] <= end)].reset_index(drop=True)
    n = len(e)
    if n < 50:
        return None
    
    e['const'] = 1.0
    exog = controls_l1 + ['const']
    bw = max(2, int(1.5 * (h + 1)))
    
    try:
        model = IVLIML(dependent=e['dy'], exog=e[exog],
                       endog=e[[rate_change_col]],
                       instruments=e[instruments])
        r = model.fit(cov_type='kernel', kernel='bartlett', bandwidth=bw, debiased=True)
        
        sigma = -r.params[rate_change_col]
        se = r.std_errors[rate_change_col]
        
        fs_f = None
        try:
            for key, fsres in r.first_stage.individual.items():
                if rate_change_col in key:
                    fs_f = fsres.f_statistic.stat
        except:
            pass
        
        return {'h': h, 'n': n, 'sigma': sigma, 'se': se,
                'p': r.pvalues[rate_change_col],
                'ci_lo': sigma - 1.96*se, 'ci_hi': sigma + 1.96*se,
                'fs_f': fs_f}
    except:
        return None


# ================================================================
# DIAGNOSTIC FUNCTIONS
# ================================================================

def first_stage_anatomy(m, instruments, controls, start=SAMPLE_START, end=SAMPLE_END):
    """
    Decompose first-stage F: which instruments predict which rate component?
    Shows why the effective rate has weaker first stage (opposing FF and equity components).
    """
    dd = m.copy()
    dd['r_ff'] = dd['fed_funds'] - dd['pi_cpce'].shift(-1)
    dd['r_eff'] = dd['eff_rate'] - dd['pi_cpce'].shift(-1)
    dd['equity_comp'] = dd['alpha'] * (dd['r_equity'] - dd['fed_funds'])
    
    needed = ['r_ff','r_eff','equity_comp'] + controls + instruments
    needed = [c for c in needed if c in dd.columns]
    est = dd[needed + ['date']].dropna()
    est = est[(est['date'] >= start) & (est['date'] <= end)].reset_index(drop=True)
    n = len(est)
    
    X_ctrl = np.column_stack([est[c].values for c in controls] + [np.ones(n)])
    Z = np.column_stack([est[c].values for c in instruments])
    q = Z.shape[1]
    X_full = np.column_stack([X_ctrl, Z])
    k = X_full.shape[1]
    
    results = {}
    for target, label in [('r_ff','Fed Funds'), ('r_eff','Eff Rate'), ('equity_comp','Equity α(rE-FF)')]:
        y = est[target].values
        ssr_r = np.sum((y - X_ctrl @ lstsq(X_ctrl, y, rcond=None)[0])**2)
        ssr_f = np.sum((y - X_full @ lstsq(X_full, y, rcond=None)[0])**2)
        F = ((ssr_r - ssr_f) / q) / (ssr_f / (n - k))
        
        # Partial correlations per instrument
        partials = {}
        y_resid = y - X_ctrl @ lstsq(X_ctrl, y, rcond=None)[0]
        for inst in instruments:
            z = est[inst].values
            z_resid = z - X_ctrl @ lstsq(X_ctrl, z, rcond=None)[0]
            partials[inst] = np.corrcoef(y_resid, z_resid)[0, 1]
        
        results[label] = {'F': F, 'partials': partials}
    
    return results


def placebo_test(m, controls_base, n_placebos=5, rate='eff_rate', pi='pi_cpce',
                 instruments=None, seed=42):
    """
    Test whether random predetermined controls replicate CANDH's effect.
    Generates AR(1) placebos with same autocorrelation as CANDH.
    Returns list of (label, sigma, p) tuples.
    """
    if instruments is None:
        instruments = ['mps_cum36','mps_cum36_l1','ffr_cum36','fg_cum36','lsap_cum36',
                       'ffr_cum36_l1','fg_cum36_l1','lsap_cum36_l1']
    
    rng = np.random.RandomState(seed)
    candh_ac = m['candh'].autocorr() if 'candh' in m.columns and m['candh'].notna().sum() > 10 else 0.35
    n_total = len(m)
    
    results = []
    
    # Baseline (no demand control)
    r = estimate_is_curve(m, rate, pi, instruments, controls_base, run_bootstrap=False)
    if r:
        results.append(('No demand ctrl', r['sigma'], r['p_sigma_cons']))
    
    # CANDH (actual)
    if 'candh_l1' in m.columns:
        r = estimate_is_curve(m, rate, pi, instruments, controls_base + ['candh_l1'], run_bootstrap=False)
        if r:
            results.append(('CANDH_l1 (actual)', r['sigma'], r['p_sigma_cons']))
    
    # Placebos
    m_test = m.copy()
    for i in range(n_placebos):
        noise = rng.randn(n_total) * (m['candh'].std() if 'candh' in m.columns else 0.1)
        placebo = np.zeros(n_total)
        for t in range(1, n_total):
            placebo[t] = candh_ac * placebo[t-1] + noise[t]
        m_test[f'placebo_{i}_l1'] = pd.Series(placebo).shift(1).values
        
        r = estimate_is_curve(m_test, rate, pi, instruments,
                              controls_base + [f'placebo_{i}_l1'], run_bootstrap=False)
        if r:
            results.append((f'Placebo {i}', r['sigma'], r['p_sigma_cons']))
    
    return results


def candh_decomposition(m, controls_base, instruments=None):
    """
    Decompose CANDH into explained (by HOUST, UMCSENT, NFCICREDIT) and residual
    (factor-specific information), then test which component drives the IS curve.
    
    Key finding: 60% of CANDH is unexplained by individual series.
    The residual (factor-specific information) drives the IS curve result.
    """
    if instruments is None:
        instruments = ['mps_cum36','mps_cum36_l1','ffr_cum36','fg_cum36','lsap_cum36',
                       'ffr_cum36_l1','fg_cum36_l1','lsap_cum36_l1']
    
    # Regress CANDH on available individual series
    indiv_cols = [c for c in ['houst','umcsent','nfcicredit'] if c in m.columns]
    sub = m.dropna(subset=['candh'] + indiv_cols).copy()
    
    X = np.column_stack([sub[c].values for c in indiv_cols] + [np.ones(len(sub))])
    y = sub['candh'].values
    b = lstsq(X, y, rcond=None)[0]
    explained = X @ b
    residual = y - explained
    r2 = 1 - residual.var() / y.var()
    
    sub['candh_explained'] = explained
    sub['candh_residual'] = residual
    sub['candh_expl_l1'] = sub['candh_explained'].shift(1)
    sub['candh_resid_l1'] = sub['candh_residual'].shift(1)
    
    # Merge back
    m_decomp = m.merge(sub[['date','candh_explained','candh_residual',
                            'candh_expl_l1','candh_resid_l1']], on='date', how='left')
    
    # Test each component in IS curve
    results = {'r2_explained': r2}
    for label, ctrl_add in [('No demand ctrl', []),
                             ('Full CANDH_l1', ['candh_l1']),
                             ('Explained_l1', ['candh_expl_l1']),
                             ('Residual_l1', ['candh_resid_l1'])]:
        r = estimate_is_curve(m_decomp, 'eff_rate', 'pi_cpce', instruments,
                              controls_base + ctrl_add, run_bootstrap=False)
        if r:
            results[label] = {'sigma': r['sigma'], 'p': r['p_sigma_cons'], 'beta_f': r['beta_f']}
    
    return results


# ================================================================
# MAIN EXECUTION
# ================================================================


# ================================================================
# MAIN: Read panel_data.csv and run IS curve estimation
# ================================================================

if __name__ == '__main__':
    print("Loading panel data...")
    m = pd.read_csv(PANEL_PATH, parse_dates=['date'])
    
    # Ensure forward/lagged output gap columns exist
    for base in ['x_ip', 'x_pces', 'x_combined']:
        if base in m.columns:
            if f'{base}_f1' not in m.columns:
                m[f'{base}_f1'] = m[base].shift(-1)
            if f'{base}_l1' not in m.columns:
                m[f'{base}_l1'] = m[base].shift(1)
    
    # Ensure cumulative instrument lags exist
    for col in ['mps_cum36', 'ffr_cum36', 'fg_cum36', 'lsap_cum36']:
        lag_col = f'{col}_l1'
        if lag_col not in m.columns and col in m.columns:
            m[lag_col] = m[col].shift(1)
    
    print(f"Panel: {len(m)} obs, {m['date'].min().date()} to {m['date'].max().date()}")
    
    # Run key IS curve specifications
    controls = ['anfci', 'baa_spread', 'term_spread', 'vix', 'candh_l1']
    
    inst_configs = {
        'MPS cumul (4)': ['mps_cum12', 'mps_cum24', 'mps_cum36', 'mps_cum36_l1'],
        'Swanson 3F cumul (6)': ['ffr_cum36', 'fg_cum36', 'lsap_cum36',
                                  'ffr_cum36_l1', 'fg_cum36_l1', 'lsap_cum36_l1'],
        'Combined (8)': ['mps_cum36', 'mps_cum36_l1',
                         'ffr_cum36', 'fg_cum36', 'lsap_cum36',
                         'ffr_cum36_l1', 'fg_cum36_l1', 'lsap_cum36_l1'],
    }
    
    for inst_label, instruments in inst_configs.items():
        print(f"\n{'='*80}")
        print(f"IS Curve: {inst_label}")
        print(f"{'='*80}")
        
        for rate_col, rate_label in [('fed_funds', 'Fed Funds'), ('eff_rate', 'Eff Rate')]:
            for pi_col, pi_label in [('pi_pce', 'PCE'), ('pi_cpce', 'CorePCE')]:
                r = estimate_is_curve(m, rate_col, pi_col, instruments, controls)
                if r:
                    print(f"  {pi_label:8s} {rate_label:10s} n={r['n']} "
                          f"sig={r['sigma']:.4f} bf={r['beta_f']:.3f} bb={r['beta_b']:.3f} "
                          f"p(HAC)={r['p_sigma_cons']:.4f} "
                          f"p(boot)={r.get('boot_p', 'n/a')}")
    
    print("\nDone.")
