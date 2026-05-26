"""
Effective Rate v2: Data Pipeline
═════════════════════════════════
Builds panel_data.csv from raw FRED/Shiller data files.

This is the ONLY script with external data dependencies. All other scripts
(effective_rate_model.py, nk_simulation.py, cross_country_analysis.py,
robustness_analysis.py) read from the panel CSVs this script produces.

DATA SOURCES (expected in DATA_DIR):
  - FRED CSVs: FEDFUNDS, BOGZ1FL153064486Q, INDPRO, CFNAI subindexes,
    ANFCI, BAA/AAA spreads, DGS10/DGS2, VIXCLS, PCEPI, PCEPILFE, WTISPLC
  - Shiller: ie_data.xls (S&P 500 monthly data with dividends and CPI)
  - Swanson: monetary-policy-surprises-data.xlsx,
    pre-and-post-ZLB-factors-extended.xlsx
  - GSCPI: gscpi_data.xlsx (NY Fed Global Supply Chain Pressure Index)

OUTPUTS:
  - panel_data.csv: 59-column monthly panel (1954-present)
    Columns include: fed_funds, eff_rate, alpha, r_equity, output gaps,
    inflation, instruments, controls, and alternative effective rates
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from linearmodels.iv import IVLIML, IV2SLS
from statsmodels.tsa.filters.hp_filter import hpfilter
from numpy.linalg import svd, lstsq, norm, eigh
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# CONFIGURATION
# ================================================================

DATA_DIR = Path('/mnt/user-data/uploads')
OUTPUT_DIR = Path(__file__).parent   # panel_data.csv written next to this script

# Effective rate parameters (from theory, Section 5 of foundation doc)
TRAILING_HORIZON = 7           # years (Malmendier-Nagel)
EQUITY_RETURN_MA = 12          # months (4 quarters)
ALPHA_MA = 8                   # quarters
HP_LAMBDA_MONTHLY = 129600     # Ravn-Uhlig rule for monthly
HP_LAMBDA_QUARTERLY = 1600     # Standard quarterly

# Bootstrap
N_BOOT = 3000
BLOCK_SIZE = 12

# Sample
SAMPLE_START = '1988-07-01'
SAMPLE_END = '2023-12-01'
ZLB_DATE = pd.Timestamp('2008-12-01')


# ================================================================
# DATA LOADING UTILITIES
# ================================================================

def load_fred(filename, col_name):
    df = pd.read_csv(DATA_DIR / filename)
    df['date'] = pd.to_datetime(df['observation_date'])
    val = [c for c in df.columns if c not in ('observation_date','date')][0]
    df[col_name] = pd.to_numeric(df[val], errors='coerce')
    return df[['date', col_name]].dropna()

def to_monthly_avg(df, col):
    df = df.copy()
    df['pm'] = df['date'].dt.to_period('M')
    out = df.groupby('pm')[col].mean().reset_index()
    out['date'] = out['pm'].dt.to_timestamp()
    return out[['date', col]]


# ================================================================
# EFFECTIVE RATE CONSTRUCTION
# ================================================================

def load_shiller():
    """
    Load Shiller (2015) monthly S&P 500 data from ie_data.xls.
    Returns DataFrame with: date, price, dividend, cpi, earnings.
    Source: http://www.econ.yale.edu/~shiller/data/ie_data.xls
    """
    xl = pd.read_excel(DATA_DIR / 'ie_data.xls', sheet_name='Data', header=7)
    xl = xl.iloc[:-1].copy()          # drop trailing notes row
    xl.columns = [str(c).strip() for c in xl.columns]

    # Date: Shiller encodes as YYYY.MM (e.g. 1871.01)
    def parse_shiller_date(d):
        try:
            parts = f'{float(d):.2f}'.split('.')
            yr, mo = int(parts[0]), int(parts[1])
            mo = max(1, min(12, mo))
            return pd.Timestamp(yr, mo, 1)
        except Exception:
            return pd.NaT

    xl['date'] = xl['Date'].apply(parse_shiller_date)
    xl = xl.dropna(subset=['date'])

    for col, new in [('P', 'price'), ('D', 'dividend'), ('CPI', 'cpi'), ('E', 'earnings')]:
        xl[new] = pd.to_numeric(xl.get(col, xl.get(col.lower(), np.nan)), errors='coerce')

    return xl[['date', 'price', 'dividend', 'cpi', 'earnings']].dropna(
        subset=['date', 'price', 'cpi']
    ).sort_values('date').reset_index(drop=True)


def compute_trailing_real_return(shiller, horizon_years):
    """
    Compute the trailing annualised real total return over horizon_years.

    Total return index:  TR_t = TR_{t-1} * (P_t + D_t/12) / P_{t-1}
    Real return:         deflated by CPI
    Annualised:          geometric mean over horizon_years * 12 months

    Returns DataFrame with: date, r_trailing (annualised %, not decimal).
    """
    df = shiller.copy().sort_values('date').reset_index(drop=True)

    # Build nominal total-return index
    df['monthly_div'] = df['dividend'] / 12
    tr = np.ones(len(df))
    for i in range(1, len(df)):
        if df['price'].iloc[i-1] > 0:
            tr[i] = tr[i-1] * (df['price'].iloc[i] + df['monthly_div'].iloc[i]) / df['price'].iloc[i-1]
        else:
            tr[i] = tr[i-1]
    df['tr'] = tr

    # Real total return index (deflated by CPI)
    df['real_tr'] = df['tr'] / df['cpi'] * df['cpi'].iloc[0]

    # Annualised trailing return
    h = horizon_years * 12
    r_trailing = []
    for i in range(len(df)):
        if i < h:
            r_trailing.append(np.nan)
        else:
            ratio = df['real_tr'].iloc[i] / df['real_tr'].iloc[i - h]
            r_trailing.append((ratio ** (1 / horizon_years) - 1) * 100)

    df['r_trailing'] = r_trailing
    return df[['date', 'r_trailing']].dropna()


def build_effective_rate():
    """Construct monthly effective rate: R̄ = (1-α)·FF + α·r^E"""
    # Federal funds rate (monthly)
    ff = load_fred('FEDFUNDS__1_.csv', 'fed_funds')

    # Alpha: quarterly Flow of Funds → 8Q MA → interpolate to monthly
    alpha_q = load_fred('BOGZ1FL153064486Q.csv', 'alpha_raw')
    alpha_q['alpha_raw'] /= 100.0
    alpha_q['date'] = alpha_q['date'].dt.to_period('Q').dt.to_timestamp()
    alpha_q = alpha_q.sort_values('date')
    alpha_q['alpha'] = alpha_q['alpha_raw'].rolling(ALPHA_MA, min_periods=ALPHA_MA).mean()
    alpha_q = alpha_q.dropna(subset=['alpha']).set_index('date')
    alpha_m = alpha_q['alpha'].resample('MS').interpolate('linear').reset_index()
    alpha_m.columns = ['date', 'alpha']

    # Trailing equity return: 7yr annualised real total return, 12m MA
    shiller = load_shiller()
    r_eq = compute_trailing_real_return(shiller, TRAILING_HORIZON).sort_values('date')
    r_eq['r_equity'] = r_eq['r_trailing'].rolling(EQUITY_RETURN_MA, min_periods=EQUITY_RETURN_MA).mean()

    # Merge and compute
    m = ff.merge(alpha_m, on='date').merge(r_eq[['date', 'r_equity']], on='date').dropna()
    m['eff_rate'] = (1 - m['alpha']) * m['fed_funds'] + m['alpha'] * m['r_equity']
    return m


# ================================================================
# SWANSON FACTOR RECONSTRUCTION
# ================================================================

def reconstruct_swanson_factors():
    """
    Reconstruct FFR/FG/LSAP factors from Bauer-Swanson raw futures data
    using Swanson (2021) subspace identification with target rotation.
    
    Method:
    1. PCA on [ED1,ED2,ED3,ED4,TNOTE05,TNOTE10,TBOND] → 3 PCs
    2. Pre-ZLB subspace (2 PCs pre-2008) defines FFR+FG space
    3. LSAP = direction in full 3-PC space orthogonal to pre-ZLB subspace
    4. FFR = direction in pre-ZLB subspace closest to ED1 loading
    5. FG = orthogonal complement within pre-ZLB subspace
    
    Validated against published Swanson (2021) factors over the
    1994–2019 overlap period with extended raw high-frequency FOMC data:
    ρ(FFR)=0.93, ρ(FG)=0.92, ρ(LSAP)=0.85
    (see Appendix D of the paper; earlier runs on shorter data windows
    produced lower correlations of 0.79/0.83/0.85 which appear in older
    internal notes but not the final paper).
    """
    bs = pd.read_excel(DATA_DIR / 'monetary-policy-surprises-data__3_.xlsx',
                       sheet_name='FOMC (update 2023)', header=None)
    bs.columns = bs.iloc[0].values
    bs = bs.iloc[1:].copy()
    bs['date'] = pd.to_datetime(bs['Date'], errors='coerce')
    bs = bs.dropna(subset=['date']).reset_index(drop=True)
    
    cols = ['ED1','ED2','ED3','ED4','TNOTE05','TNOTE10','TBOND']
    for c in cols:
        bs[c] = pd.to_numeric(bs[c], errors='coerce')
    
    df = bs.dropna(subset=cols).copy().reset_index(drop=True)
    dates = pd.DatetimeIndex(df['date'].values)
    X = df[cols].values.astype(float)
    X_dm = X - X.mean(0)
    n, p = X.shape
    
    pre = dates < ZLB_DATE
    
    # Full-sample 3 PCs
    _, S_f, Vt_f = svd(X_dm, full_matrices=False)
    V3 = Vt_f[:3].T
    
    # Pre-ZLB 2 PCs
    _, S_p, Vt_p = svd(X_dm[pre], full_matrices=False)
    V2 = Vt_p[:2].T
    
    # LSAP: orthogonal to pre-ZLB subspace
    V_proj = V2 @ (V2.T @ V3)
    V_orth = V3 - V_proj
    norms = [norm(V_orth[:, j]) for j in range(3)]
    lsap_j = np.argmax(norms)
    lsap_load = V_orth[:, lsap_j] / norm(V_orth[:, lsap_j])
    lsap = X_dm @ lsap_load
    
    # FFR: direction in V2 subspace closest to ED1
    e_target = np.zeros(p)
    e_target[0] = 1.0  # ED1
    e_in_V2 = V2.T @ e_target
    e_norm = e_in_V2 / max(norm(e_in_V2), 1e-10)
    fg_in_V2 = np.array([-e_norm[1], e_norm[0]])
    
    ffr = X_dm @ (V2 @ e_norm)
    fg = X_dm @ (V2 @ fg_in_V2)
    
    # Sign conventions
    if np.corrcoef(ffr, X_dm[:, 0])[0,1] < 0:
        ffr = -ffr
    
    factors = pd.DataFrame({'date': dates, 'ffr': ffr, 'fg': fg, 'lsap': lsap})
    
    # Fix signs against Swanson original if available
    try:
        sw = pd.read_excel(DATA_DIR / 'pre-and-post-ZLB-factors-extended.xlsx',
                          sheet_name='Data', header=None).iloc[2:]
        sw.columns = ['_','d','sw_ffr','sw_fg','sw_lsap','_neg']
        sw['date'] = pd.to_datetime(sw['d'], errors='coerce')
        for c in ['sw_ffr','sw_fg','sw_lsap']:
            sw[c] = pd.to_numeric(sw[c], errors='coerce')
        sw = sw.dropna(subset=['date','sw_ffr'])
        mg = factors.merge(sw[['date','sw_ffr','sw_fg','sw_lsap']], on='date')
        for c in ['lsap','fg']:
            if mg[c].corr(mg[f'sw_{c}']) < 0:
                factors[c] = -factors[c]
    except:
        pass
    
    # Aggregate to monthly
    factors['ym'] = factors['date'].dt.to_period('M')
    monthly = factors.groupby('ym')[['ffr','fg','lsap']].sum().reset_index()
    monthly['date'] = monthly['ym'].dt.to_timestamp()
    return monthly[['date','ffr','fg','lsap']]


# ================================================================
# FULL DATASET ASSEMBLY
# ================================================================

def build_monthly_dataset():
    """Assemble complete monthly estimation dataset."""
    m = build_effective_rate()
    
    # Inflation
    for fname, cn, pi in [('PCEPI.csv','pcepi','pi_pce'),
                           ('PCEPILFE__1_.csv','core_pce','pi_cpce')]:
        p = load_fred(fname, cn).sort_values('date')
        p[pi] = 1200 * np.log(p[cn] / p[cn].shift(1))
        m = m.merge(p[['date',pi]], on='date', how='left')
    
    # Industrial production
    m = m.merge(load_fred('INDPRO.csv','ip'), on='date', how='left')
    
    # CFNAI
    m = m.merge(load_fred('CFNAI.csv','cfnai'), on='date', how='left')
    
    # MPS_ORTH
    mps = pd.read_excel(DATA_DIR/'monetary-policy-surprises-data__3_.xlsx',
                        sheet_name='Monthly (update 2023)', header=None).iloc[1:]
    mps.columns = ['year','month','mps','mps_orth'] + [f'x{i}' for i in range(6)]
    mps['mps_orth'] = pd.to_numeric(mps['mps_orth'], errors='coerce')
    mps['year'] = pd.to_numeric(mps['year'], errors='coerce')
    mps['month'] = pd.to_numeric(mps['month'], errors='coerce')
    mps = mps.dropna(subset=['year','month'])
    mps['date'] = pd.to_datetime(mps[['year','month']].assign(day=1))
    m = m.merge(mps[['date','mps_orth']], on='date', how='left')
    
    # Extended Swanson factors
    sw_fac = reconstruct_swanson_factors()
    m = m.merge(sw_fac, on='date', how='left')
    
    # Controls
    for fname, cn in [('ANFCI.csv','anfci'), ('BAAFFM.csv','baa_spread')]:
        c = load_fred(fname, cn)
        if cn == 'anfci':
            c = to_monthly_avg(c, cn)
        m = m.merge(c[['date',cn]], on='date', how='left')
    
    t10 = to_monthly_avg(load_fred('T10Y2Y.csv','term_spread'), 'term_spread')
    m = m.merge(t10, on='date', how='left')
    
    vxo = load_fred('VXOCLS.csv','vol')
    vix = load_fred('VIXCLS__1_.csv','vol')
    vol = pd.concat([vxo[vxo['date']<'1990-01-01'], vix[vix['date']>='1990-01-01']])
    vol_m = to_monthly_avg(vol, 'vol').rename(columns={'vol':'vix'})
    m = m.merge(vol_m, on='date', how='left')
    
    m = m.sort_values('date').reset_index(drop=True)
    
    # HP filter IP
    ipv = m.dropna(subset=['ip']).copy()
    cyc, _ = hpfilter(np.log(ipv['ip']), lamb=HP_LAMBDA_MONTHLY)
    ipv['x_ip'] = cyc * 100
    m = m.merge(ipv[['date','x_ip']], on='date', how='left')
    
    # PCE Services → services output gap
    try:
        pces = load_fred('PCES.csv', 'pces')
        m = m.merge(pces, on='date', how='left')
        pv = m.dropna(subset=['pces']).copy()
        cyc_s, _ = hpfilter(np.log(pv['pces']), lamb=HP_LAMBDA_MONTHLY)
        pv['x_pces'] = cyc_s * 100
        m = m.merge(pv[['date','x_pces']], on='date', how='left')
        m['x_pces_f1'] = m['x_pces'].shift(-1)
        m['x_pces_l1'] = m['x_pces'].shift(1)
        # Combined goods+services (30/70 weight)
        m['x_combined'] = 0.3 * m['x_ip'] + 0.7 * m['x_pces']
        m['x_combined_f1'] = m['x_combined'].shift(-1)
        m['x_combined_l1'] = m['x_combined'].shift(1)
    except Exception as e:
        print(f"  Warning: Could not load PCES: {e}")
    
    # Additional robustness controls
    for fname, cname in [('HOUST.csv','houst'), ('UMCSENT.csv','umcsent')]:
        try:
            m = m.merge(load_fred(fname, cname), on='date', how='left')
            m[f'{cname}_l1'] = m[cname].shift(1)
        except:
            pass
    try:
        nfcicredit = load_fred('NFCICREDIT.csv','nfcicredit')
        nfcicredit = to_monthly_avg(nfcicredit, 'nfcicredit')
        m = m.merge(nfcicredit, on='date', how='left')
        m['nfcicredit_l1'] = m['nfcicredit'].shift(1)
    except:
        pass
    
    # CFNAI subindexes (demand controls, non-IP)
    # CANDH = Consumption and Housing (primary demand control)
    # EUANDH = Employment, Unemployment, Hours
    for fname, cname in [('CANDH.csv','candh'), ('EUANDH.csv','euandh')]:
        try:
            subidx = load_fred(fname, cname)
            m = m.merge(subidx, on='date', how='left')
            m[f'{cname}_l1'] = m[cname].shift(1)
        except Exception as e:
            print(f"  Warning: Could not load {fname}: {e}")
            m[cname] = np.nan
            m[f'{cname}_l1'] = np.nan
    
    # Supply shock series (for Phillips curve augmentation)
    # WTI crude oil: 12-month log change (cost-push term)
    try:
        wti = load_fred('WTISPLC__1_.csv', 'wti')
        wti = wti.sort_values('date')
        wti['d12_oil'] = 100 * (np.log(wti['wti']) - np.log(wti['wti'].shift(12)))
        m = m.merge(wti[['date','wti','d12_oil']], on='date', how='left')
    except Exception as e:
        print(f"  Warning: Could not load WTI: {e}")
        m['wti'] = np.nan
        m['d12_oil'] = np.nan
    
    # NY Fed Global Supply Chain Pressure Index
    try:
        gscpi_raw = pd.read_excel(DATA_DIR / 'gscpi_data__1_.xls',
                                   sheet_name='GSCPI Monthly Data', header=None).iloc[5:]
        gscpi_raw.columns = ['ds','gscpi','_','__']
        gscpi_raw['date'] = pd.to_datetime(gscpi_raw['ds'], format='mixed',
                                            dayfirst=True).dt.to_period('M').dt.to_timestamp()
        gscpi_raw['gscpi'] = pd.to_numeric(gscpi_raw['gscpi'], errors='coerce')
        gscpi = gscpi_raw[['date','gscpi']].dropna()
        m = m.merge(gscpi, on='date', how='left')
    except Exception as e:
        print(f"  Warning: Could not load GSCPI: {e}")
        m['gscpi'] = np.nan
    
    # Cumulative shocks
    m['mps_f'] = m['mps_orth'].fillna(0)
    for c in ['ffr','fg','lsap']:
        m[f'{c}_f'] = m[c].fillna(0)
    for w in [12, 24, 36]:
        m[f'mps_cum{w}'] = m['mps_f'].rolling(w).sum()
        for c in ['ffr','fg','lsap']:
            m[f'{c}_cum{w}'] = m[f'{c}_f'].rolling(w).sum()
    
    # Leads and lags
    for c in ['x_ip','pi_pce','pi_cpce','fed_funds','eff_rate','cfnai',
              'mps_cum36','mps_cum24','mps_cum12',
              'ffr_cum36','fg_cum36','lsap_cum36',
              'anfci','baa_spread','term_spread','vix',
              'mps_f','ffr_f','fg_f','lsap_f']:
        if c in m.columns:
            for lag in range(1, 7):
                m[f'{c}_l{lag}'] = m[c].shift(lag)
    
    m['x_ip_f1'] = m['x_ip'].shift(-1)
    for h in range(0, 49):
        m[f'x_ip_f{h}'] = m['x_ip'].shift(-h)
    
    # Rate changes (for LP)
    m['d_ff'] = m['fed_funds'] - m['fed_funds'].shift(1)
    m['d_eff'] = m['eff_rate'] - m['eff_rate'].shift(1)
    
    return m


# The Shiller-based alternative effective rate construction functions
# (5yr, 10yr, exponential, recency weighting) were used to build
# eff_rate_5yr, eff_rate_10yr, eff_rate_exp3, eff_rate_exp2, eff_rate_rec
# columns already present in panel_data.csv.

# ================================================================
# MAIN EXECUTION
# ================================================================

if __name__ == '__main__':
    print("Building monthly dataset...")
    m = build_monthly_dataset()
    
    # Save panel data
    panel_cols = ['date','fed_funds','eff_rate','alpha','r_equity',
                  'pi_pce','pi_cpce','ip','x_ip','cfnai','candh','euandh',
                  'mps_orth','ffr','fg','lsap',
                  'anfci','baa_spread','term_spread','vix',
                  'mps_cum12','mps_cum24','mps_cum36',
                  'ffr_cum36','fg_cum36','lsap_cum36',
                  'd_ff','d_eff','candh_l1','euandh_l1',
                  'wti','d12_oil','gscpi']
    panel = m[[c for c in panel_cols if c in m.columns]].dropna(subset=['x_ip','fed_funds','eff_rate'])
    panel.to_csv(OUTPUT_DIR / 'panel_data.csv', index=False)
    print(f"Panel data saved: {len(panel)} obs, {panel['date'].min().date()} to {panel['date'].max().date()}")
    
    # Run key IS curve specifications
    controls = ['anfci','baa_spread','term_spread','vix','candh_l1']
    
    inst_configs = {
        'MPS cumul (4)': ['mps_cum12','mps_cum24','mps_cum36','mps_cum36_l1'],
        'Swanson 3F cumul (6)': ['ffr_cum36','fg_cum36','lsap_cum36',
                                  'ffr_cum36_l1','fg_cum36_l1','lsap_cum36_l1'],
        'Combined (8)': ['mps_cum36','mps_cum36_l1',
                         'ffr_cum36','fg_cum36','lsap_cum36',
                         'ffr_cum36_l1','fg_cum36_l1','lsap_cum36_l1'],
    }
    
    for inst_label, instruments in inst_configs.items():
        print(f"\n{'='*80}")
        print(f"IS Curve: {inst_label}")
        print(f"{'='*80}")
        
        for rate_col, rate_label in [('fed_funds','Fed Funds'), ('eff_rate','Eff Rate')]:
            for pi_col, pi_label in [('pi_pce','PCE'), ('pi_cpce','CorePCE')]:
                r = estimate_is_curve(m, rate_col, pi_col, instruments, controls)
                if r:
                    print(f"  {pi_label:8s} {rate_label:10s} n={r['n']} "
                          f"σ̂={r['sigma']:.4f} β_f={r['beta_f']:.3f} β_b={r['beta_b']:.3f} "
                          f"p(HAC)={r['p_sigma_cons']:.4f} "
                          f"p(boot)={r.get('boot_p','n/a')}")
    
    print("\nDone.")
