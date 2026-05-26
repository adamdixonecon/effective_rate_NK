"""
NK Three-Equation Simulation: Missing Inflation Resolution
============================================================
Companion to the Effective Rate v2 model (Dixon 2026).

Simulates the standard 3-equation NK system under two rate measures:
  1. Federal funds rate (standard model)
  2. Effective rate R-bar = (1-alpha)FF + alpha*r^E (two-asset model)

Uses the SUPPLY-AUGMENTED Phillips curve (oil + GSCPI) matching the paper's
Section 5.1 / Equation (14).

USAGE:
  python nk_simulation.py
  python nk_simulation.py --data path/to/panel_data.csv

Requires: panel_data.csv with columns: date, fed_funds, eff_rate, alpha,
  r_equity, pi_cpce, x_ip, d12_oil, gscpi, stance_ff, stance_eff
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# ================================================================
# CONFIGURATION
# ================================================================

# IS curve: Combined instruments, CorePCE, CANDH_{t-1} (paper Eq. 14)
BETA_F = 0.271
BETA_B = 0.687
SIGMA  = 0.108

# Supply-augmented Phillips curve (paper below Eq. 14)
GAMMA_F   = 0.18
GAMMA_B   = 0.19
KAPPA     = 0.045
LAM_OIL   = 0.005
LAM_GSCPI = 0.367
PC_CONST  = 1.23

# Natural rate (Holston-Laubach-Williams)
R_STAR = 0.5

# Sub-periods (fine-grained decomposition)
PERIODS_FINE = {
    'GFC crash 2008-09':      ('2008-01-01', '2009-06-01'),
    'Early ZLB 2009Q3-2011':  ('2009-07-01', '2011-09-01'),
    'Gap opens 2011Q4-2013':  ('2011-10-01', '2013-06-01'),
    'Gap widens 2013Q3-2015': ('2013-07-01', '2015-12-01'),
    'Liftoff 2016-17':        ('2016-01-01', '2017-12-01'),
    'Tightening 2018-19':     ('2018-01-01', '2019-12-01'),
    'COVID 2020-21':          ('2020-01-01', '2021-12-01'),
    'Hiking 2022-23':         ('2022-01-01', '2023-06-01'),
}

PERIODS_COARSE = {
    'Pre-crisis 2005-07': ('2005-01-01', '2007-12-01'),
    'ZLB 2009-15':        ('2009-01-01', '2015-12-01'),
    'Recovery 2016-19':   ('2016-01-01', '2019-12-01'),
    'COVID ZLB 2020-21':  ('2020-03-01', '2021-12-01'),
    'Hiking 2022-23':     ('2022-03-01', '2023-06-01'),
}


# ================================================================
# SIMULATION ENGINE
# ================================================================

def simulate_adaptive(rate_path, x_init, pi_init, oil_path, gscpi_path,
                       params=None):
    """
    Dynamic forward simulation with adaptive (purely backward-looking) expectations.

    Expectation rule: E_{t}[x_{t+1}] = x_{t-1}.
    This collapses the forward and backward terms so the combined persistence
    coefficient is (beta_f + beta_b), making the simulation recursively tractable.

    Uses the supply-augmented Phillips curve:
      pi_t = gamma_f*pi_{t-1} + gamma_b*pi_{t-1} + kappa*x_t
             + lam_oil*oil_t + lam_gscpi*gscpi_t + const

    Parameters
    ----------
    rate_path  : array, nominal rate (FF or effective)
    x_init     : float, initial output gap
    pi_init    : float, initial inflation
    oil_path   : array, 12-month log change in WTI crude
    gscpi_path : array, NY Fed GSCPI (0 where missing)
    params     : dict, override defaults

    Returns
    -------
    x, pi : np.ndarray
    """
    p = dict(beta_f=BETA_F, beta_b=BETA_B, sigma=SIGMA,
             gamma_f=GAMMA_F, gamma_b=GAMMA_B, kappa=KAPPA,
             lam_oil=LAM_OIL, lam_gscpi=LAM_GSCPI,
             pc_const=PC_CONST, r_star=R_STAR)
    if params:
        p.update(params)

    T = len(rate_path)
    x, pi = np.zeros(T), np.zeros(T)
    x[0], pi[0] = x_init, pi_init

    for t in range(1, T):
        # IS curve: adaptive expectations E_{t}[x_{t+1}] = x_{t-1}
        # (purely backward-looking special case; both beta_f and beta_b act on x_{t-1})
        r_real = rate_path[t] - pi[t-1] - p['r_star']
        x[t] = (p['beta_f'] + p['beta_b']) * x[t-1] - p['sigma'] * r_real
        x[t] = np.clip(x[t], -20, 20)

        # Supply-augmented Phillips curve
        oil_t = oil_path[t] if not np.isnan(oil_path[t]) else 0.0
        gscpi_t = gscpi_path[t] if not np.isnan(gscpi_path[t]) else 0.0
        pi[t] = (p['gamma_f'] * pi[t-1] + p['gamma_b'] * pi[t-1]
                 + p['kappa'] * x[t]
                 + p['lam_oil'] * oil_t + p['lam_gscpi'] * gscpi_t
                 + p['pc_const'])
        pi[t] = np.clip(pi[t], -5, 25)

    return x, pi


# ================================================================
# EXERCISE 1: REAL RATE STANCE
# ================================================================

def stance_diagnostic(df, r_star=R_STAR, periods=None):
    if periods is None:
        periods = PERIODS_COARSE
    sub = df[['date','fed_funds','eff_rate','pi_cpce']].dropna().copy()
    sub['r_real_ff']  = sub['fed_funds'] - sub['pi_cpce']
    sub['r_real_eff'] = sub['eff_rate'] - sub['pi_cpce']
    sub['stance_ff']  = sub['r_real_ff'] - r_star
    sub['stance_eff'] = sub['r_real_eff'] - r_star

    print(f"\n{'='*80}")
    print("EXERCISE 1: Real Rate Policy Stance (r_real - r*)")
    print(f"{'='*80}")
    print(f"{'Period':<28s} {'r(FF)':>7s} {'r(Eff)':>8s} {'s(FF)':>7s} {'s(Eff)':>8s} {'Gap':>6s}")
    print("-"*68)
    for label, (s, e) in periods.items():
        m = sub[(sub['date'] >= s) & (sub['date'] <= e)]
        if len(m) == 0: continue
        rff, reff = m['r_real_ff'].mean(), m['r_real_eff'].mean()
        sff, seff = m['stance_ff'].mean(), m['stance_eff'].mean()
        print(f"{label:<28s} {rff:+7.2f} {reff:+7.2f} {sff:+7.2f} {seff:+7.2f} {seff-sff:+5.2f}")
    return sub


# ================================================================
# EXERCISE 2: GAP DECOMPOSITION
# ================================================================

def gap_decomposition(df):
    sub = df[df['r_equity'].notna() & df['alpha'].notna()].copy()
    print(f"\n{'='*80}")
    print("EXERCISE 2: Rate Gap = alpha * (r^E - FF)")
    print(f"{'='*80}")
    print(f"{'Year':>6s} {'alpha':>6s} {'r^E':>7s} {'FF':>6s} {'r^E-FF':>7s} {'gap':>6s} {'Eff':>6s}")
    print("-"*55)
    for yr in range(2007, 2026):
        m = sub[sub['date'].dt.year == yr]
        if len(m) == 0: continue
        a, re, ff = m['alpha'].mean(), m['r_equity'].mean(), m['fed_funds'].mean()
        gap = m['eff_rate'].mean() - ff
        print(f"  {yr} {a:6.3f} {re:7.2f} {ff:6.2f} {re-ff:+7.2f} {gap:+6.2f} {m['eff_rate'].mean():6.2f}")


# ================================================================
# EXERCISE 3: DYNAMIC SIMULATION
# ================================================================

def run_simulation(df, sim_start='2007-01-01', sim_end='2023-06-01',
                   periods=None, print_results=True):
    if periods is None:
        periods = PERIODS_FINE

    sim = df[(df['date'] >= sim_start) & (df['date'] <= sim_end) &
             df['pi_cpce'].notna() & df['x_ip'].notna()].copy().reset_index(drop=True)
    if len(sim) < 12:
        raise ValueError(f"Insufficient data: {len(sim)}")

    x0, pi0 = sim['x_ip'].iloc[0], sim['pi_cpce'].iloc[0]
    pi_act = sim['pi_cpce'].values
    oil = sim['d12_oil'].fillna(0).values
    gscpi = sim['gscpi'].fillna(0).values

    results = {}
    for label, col in [('Fed Funds','fed_funds'), ('Eff Rate','eff_rate')]:
        x_ad, pi_ad = simulate_adaptive(sim[col].values, x0, pi0, oil, gscpi)
        results[label] = dict(x_ad=x_ad, pi_ad=pi_ad)

    cum_ff  = np.cumsum(results['Fed Funds']['pi_ad'] - pi_act) / 12
    cum_eff = np.cumsum(results['Eff Rate']['pi_ad'] - pi_act) / 12

    if print_results:
        print(f"\n{'='*80}")
        print(f"EXERCISE 3: NK Simulation, Supply-Augmented PC ({sim_start[:4]}-{sim_end[:4]}, n={len(sim)})")
        print(f"IS: bf={BETA_F}, bb={BETA_B}, sigma={SIGMA}")
        print(f"PC: gf={GAMMA_F}, gb={GAMMA_B}, kappa={KAPPA}, "
              f"lam_oil={LAM_OIL}, lam_gscpi={LAM_GSCPI}, c={PC_CONST}")
        print(f"r* = {R_STAR}%. Adaptive expectations.")
        print(f"{'='*80}")

        # Phase-level table (matches paper Table 6)
        print(f"\n{'Phase':<28s} {'pi_FF':>7s} {'pi_Eff':>8s} {'pi_act':>7s} "
              f"{'MAE_FF':>7s} {'MAE_Eff':>8s} {'Ratio':>6s}")
        print("-"*75)
        for plabel, (ps, pe) in periods.items():
            mask = (sim['date'] >= ps) & (sim['date'] <= pe)
            if mask.sum() == 0: continue
            idx = mask.values
            pff  = np.mean(results['Fed Funds']['pi_ad'][idx])
            peff = np.mean(results['Eff Rate']['pi_ad'][idx])
            pact = np.mean(pi_act[idx])
            mae_ff  = np.mean(np.abs(results['Fed Funds']['pi_ad'][idx] - pi_act[idx]))
            mae_eff = np.mean(np.abs(results['Eff Rate']['pi_ad'][idx] - pi_act[idx]))
            ratio = mae_ff / mae_eff if mae_eff > 0.001 else float('inf')
            print(f"{plabel:<28s} {pff:7.2f} {peff:7.2f} {pact:7.2f} "
                  f"{mae_ff:7.3f} {mae_eff:8.3f} {ratio:5.2f}x")

        # Year-by-year
        print(f"\n{'Year':>6s} {'pi_FF':>7s} {'piEff':>7s} {'Dpi':>6s} {'piAct':>7s} {'closer':>7s}")
        print("-"*50)
        for yr in range(int(sim_start[:4]), int(sim_end[:4])+1):
            mask = sim['date'].dt.year == yr
            if mask.sum() == 0: continue
            idx = mask.values
            pff  = np.mean(results['Fed Funds']['pi_ad'][idx])
            peff = np.mean(results['Eff Rate']['pi_ad'][idx])
            pact = np.mean(pi_act[idx])
            closer = "Eff" if abs(peff-pact) < abs(pff-pact) else "FF"
            print(f"  {yr} {pff:7.2f} {peff:7.2f} {pff-peff:+5.2f} {pact:7.2f} {closer:>7s}")

        # Cumulative (2009-2019): fresh simulation from 2009-01-01
        # to avoid pre-ZLB initial conditions affecting the cumulative
        sim09 = df[(df['date'] >= '2009-01-01') & (df['date'] <= '2019-12-01') &
                   df['pi_cpce'].notna() & df['x_ip'].notna()].copy().reset_index(drop=True)
        if len(sim09) > 0:
            x0_09, pi0_09 = sim09['x_ip'].iloc[0], sim09['pi_cpce'].iloc[0]
            oil09 = sim09['d12_oil'].fillna(0).values
            gscpi09 = sim09['gscpi'].fillna(0).values
            _, pi_ff_09 = simulate_adaptive(sim09['fed_funds'].values, x0_09, pi0_09, oil09, gscpi09)
            _, pi_eff_09 = simulate_adaptive(sim09['eff_rate'].values, x0_09, pi0_09, oil09, gscpi09)
            pi_act_09 = sim09['pi_cpce'].values
            c_ff = np.sum(pi_ff_09 - pi_act_09) / 12
            c_eff = np.sum(pi_eff_09 - pi_act_09) / 12
            print(f"\nCumulative over-prediction (2009-19, fresh start): "
                  f"FF={c_ff:+.2f}pp, Eff={c_eff:+.2f}pp, Advantage={c_ff-c_eff:+.2f}pp")

    return dict(dates=sim['date'].tolist(), actual_pi=pi_act,
                actual_x=sim['x_ip'].values,
                actual_ff=sim['fed_funds'].values, actual_eff=sim['eff_rate'].values,
                sim=results, cum_ff=cum_ff, cum_eff=cum_eff, df_sim=sim)


# ================================================================
# EXERCISE 4: SENSITIVITY
# ================================================================

def sensitivity_analysis(df, sim_start='2009-01-01', sim_end='2019-12-01'):
    sim = df[(df['date'] >= sim_start) & (df['date'] <= sim_end) &
             df['pi_cpce'].notna() & df['x_ip'].notna()].copy().reset_index(drop=True)
    x0, pi0 = sim['x_ip'].iloc[0], sim['pi_cpce'].iloc[0]
    pi_act = sim['pi_cpce'].values
    oil, gscpi = sim['d12_oil'].fillna(0).values, sim['gscpi'].fillna(0).values

    def sweep(pname, values):
        print(f"\n--- Sensitivity to {pname} ---")
        print(f"{'Value':>7s} {'cum_FF':>8s} {'cum_Eff':>9s} {'Advtg':>8s}")
        print("-"*35)
        for v in values:
            _, pi_ff  = simulate_adaptive(sim['fed_funds'].values, x0, pi0, oil, gscpi, {pname:v})
            _, pi_eff = simulate_adaptive(sim['eff_rate'].values, x0, pi0, oil, gscpi, {pname:v})
            c_ff  = np.sum(pi_ff - pi_act) / 12
            c_eff = np.sum(pi_eff - pi_act) / 12
            print(f"{v:7.3f} {c_ff:+8.2f} {c_eff:+8.2f} {c_ff-c_eff:+8.2f}")

    print(f"\n{'='*80}")
    print("EXERCISE 4: Sensitivity Analysis (Supply-Augmented PC)")
    print(f"{'='*80}")
    sweep('r_star', [0.0, 0.25, 0.5, 1.0, 1.5])
    sweep('kappa',  [0.020, 0.045, 0.080, 0.100])
    sweep('sigma',  [0.050, 0.108, 0.150, 0.200])


# ================================================================
# EXERCISE 5: POST-COVID CONVERGENCE
# ================================================================

def convergence_diagnostic(df):
    print(f"\n{'='*80}")
    print("EXERCISE 5: Post-COVID Convergence Diagnostic")
    print(f"{'='*80}")
    print(f"\n{'Year':>6s} {'alpha':>6s} {'r^E':>7s} {'FF':>6s} {'r^E-FF':>7s} {'gap':>6s} {'Eff':>6s}")
    print("-"*55)
    for yr in range(2019, 2026):
        m = df[(df['date'].dt.year == yr) & df['alpha'].notna() & df['r_equity'].notna()]
        if len(m) == 0: continue
        a, re, ff = m['alpha'].mean(), m['r_equity'].mean(), m['fed_funds'].mean()
        gap = m['eff_rate'].mean() - ff
        print(f"  {yr} {a:6.3f} {re:7.2f} {ff:6.2f} {re-ff:+7.2f} {gap:+6.2f} {m['eff_rate'].mean():6.2f}")

    print(f"\n  Convergence channels:")
    print(f"    1. FF rising:      OPERATING (0.08 -> 5.02)")
    print(f"    2. r^E rotating:   SLOW (7yr trailing window)")
    print(f"    3. alpha:")
    print(f"       - Cyclical Sharpe-ratio component: SHOULD fall, but overwhelmed")
    print(f"       - Low-freq institutional component: RISING (401k, platforms)")
    print(f"       - Net: alpha rose 0.375 -> 0.431 (institutional > Sharpe)")


# ================================================================
# MAIN
# ================================================================

def main(data_path=None):
    if data_path is None:
        for p in [Path(__file__).parent / 'panel_data.csv', Path('panel_data.csv')]:
            if p.exists():
                data_path = p; break
    if data_path is None:
        raise FileNotFoundError("Cannot find panel_data.csv")

    df = pd.read_csv(data_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Data: {len(df)} obs, {df['date'].min().date()} to {df['date'].max().date()}")

    stance_diagnostic(df)
    gap_decomposition(df)
    results = run_simulation(df, '2007-01-01', '2023-06-01')
    sensitivity_analysis(df)
    convergence_diagnostic(df)

    print(f"\n{'='*80}\nALL EXERCISES COMPLETE\n{'='*80}")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()
    main(args.data)
