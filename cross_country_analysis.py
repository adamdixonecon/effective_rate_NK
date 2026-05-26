"""
Cross-Country Effective Rate Analysis
======================================
Constructs R̄ = (1-α)·i^policy + α·r^E for Japan, UK, Australia, Canada,
Sweden, and the US baseline. Tests the prediction that the rate gap
(R̄ − i^policy) scales monotonically with α.

Key finding: R² = 0.976 across 6 countries (2016-2019).
Intercept ≈ 0, confirming R̄ = i^policy when α = 0.

USAGE:
  python cross_country_analysis.py --data-dir /path/to/uploads

Requires: country-specific data files (see DATA SOURCES below)
"""

import pandas as pd
import numpy as np
import csv
import argparse
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# CONFIGURATION
# ================================================================

TRAILING_HORIZON = 7  # years (same as US model)


# ================================================================
# ALPHA CONSTRUCTION
# ================================================================

def load_alpha_japan(data_dir):
    """BoJ Flow of Funds: equity+funds / total financial assets."""
    jp = pd.read_csv(data_dir / 'nme_R031_1554585_20260524004106_02.csv')
    for c in jp.columns[1:]:
        jp[c] = pd.to_numeric(jp[c], errors='coerce')
    jp_d = jp.iloc[1:].copy()
    
    def parse_date(s):
        try:
            parts = str(s).split('/')
            return pd.Timestamp(f'{parts[0]}-{parts[1]}-01')
        except:
            return pd.NaT
    
    jp_d['date'] = jp_d['Series code'].apply(parse_date)
    jp_d['alpha'] = jp_d[jp.columns[2]] / jp_d[jp.columns[-1]]
    return jp_d[['date', 'alpha']].dropna()


def load_alpha_uk(data_dir):
    """ONS Blue Book Table 6.2.11: AF.5 / AF.A."""
    uk = pd.read_excel(data_dir / 'uknationalaccountsthebluebook2025referencetables.xlsx',
                        sheet_name='6.2.11', header=None)
    uk_d = uk.iloc[7:].copy()
    uk_d['year'] = pd.to_numeric(uk_d[0], errors='coerce')
    uk_d['af5'] = pd.to_numeric(uk_d[27], errors='coerce')
    uk_d['afa'] = pd.to_numeric(uk_d[35], errors='coerce')
    uk_d = uk_d.dropna(subset=['year', 'af5', 'afa'])
    uk_d['alpha'] = uk_d['af5'] / uk_d['afa']
    uk_d['date'] = pd.to_datetime(uk_d['year'].astype(int).astype(str) + '-12-31')
    return uk_d[['date', 'alpha']]


def load_alpha_canada(data_dir):
    """Statistics Canada Table 36-10-0580-01: equity+funds / total FA."""
    with open(data_dir / '3610058001-eng.csv', 'r') as f:
        rows = list(csv.reader(f))
    
    dates_raw = [d.strip() for d in rows[11][1:] if d.strip()]
    
    def parse_row(row):
        return [float(v.replace(',', '')) if v.strip() else None for v in row[1:]]
    
    total_fa = parse_row(rows[40])
    equity = parse_row(rows[72])
    
    dates, alphas = [], []
    for i in range(min(len(dates_raw), len(total_fa), len(equity))):
        if total_fa[i] and equity[i] and total_fa[i] > 0:
            parts = dates_raw[i].split()
            q, yr = int(parts[0][1]), int(parts[1])
            dates.append(pd.Timestamp(f'{yr}-{q * 3:02d}-01'))
            alphas.append(equity[i] / total_fa[i])
    
    return pd.DataFrame({'date': dates, 'alpha': alphas})


def load_alpha_oecd(data_dir, country_code):
    """OECD Financial Balance Sheets: AF.5 / F for S14 households."""
    oecd = pd.read_csv(data_dir /
        'OECD_SDD_NAD_DSD_NASEC20_DF_T710R_A__A__AUS_SWE_KOR__S14_S1_____F_F5__XDC______.csv')
    oecd['OBS_VALUE'] = pd.to_numeric(oecd['OBS_VALUE'], errors='coerce')
    hh = oecd[oecd['SECTOR'] == 'S14']
    assets = hh[(hh['REF_AREA'] == country_code) & (hh['ACCOUNTING_ENTRY'] == 'A')]
    
    f5 = assets[assets['INSTR_ASSET'] == 'F5'].set_index('TIME_PERIOD')['OBS_VALUE']
    ft = assets[assets['INSTR_ASSET'] == 'F'].set_index('TIME_PERIOD')['OBS_VALUE']
    alpha = (f5 / ft).dropna()
    
    return pd.DataFrame({
        'date': pd.to_datetime(alpha.index.astype(str) + '-12-31'),
        'alpha': alpha.values
    })


def load_alpha_australia_broad(data_dir):
    """ABS Table 36: (listed + unlisted + 0.6×super) / total FA."""
    aus = pd.read_excel(data_dir / '5232036.xls', sheet_name='Data1')
    
    col_map = {}
    for c in aus.columns:
        cs = str(c).lower()
        if 'total financial assets' in cs and 'outstanding' in cs:
            col_map['total'] = c
        elif 'listed shares' in cs and 'outstanding' in cs:
            col_map['listed'] = c
        elif 'unlisted shares' in cs and 'outstanding' in cs:
            col_map['unlisted'] = c
        elif 'pension funds' in cs and 'outstanding' in cs and 'net equity' in cs:
            col_map['super'] = c
    
    df = pd.DataFrame({'date': pd.to_datetime(aus.iloc[:, 0], errors='coerce')})
    for k, c in col_map.items():
        df[k] = pd.to_numeric(aus[c], errors='coerce')
    df = df.dropna()
    df['alpha'] = (df['listed'] + df['unlisted'] + 0.6 * df['super']) / df['total']
    return df[['date', 'alpha']]


# ================================================================
# POLICY RATE CONSTRUCTION
# ================================================================

def load_rate_fred(data_dir, filename, col_name):
    """Load monthly rate from FRED CSV."""
    df = pd.read_csv(data_dir / filename)
    df['date'] = pd.to_datetime(df['observation_date'])
    df['rate'] = pd.to_numeric(df[col_name], errors='coerce')
    return df[['date', 'rate']].dropna().sort_values('date')


def load_rate_boe(data_dir):
    """BoE Bank Rate: event-based → monthly average."""
    boe = pd.read_csv(data_dir / 'Bank_Rate_history_and_data__Bank_of_England_Database.csv')
    boe['date'] = pd.to_datetime(boe['Date Changed'], format='%d %b %y', dayfirst=True)
    boe['rate'] = pd.to_numeric(boe['Rate'], errors='coerce')
    boe = boe.sort_values('date')
    daily = boe.set_index('date')['rate'].resample('D').ffill()
    monthly = daily.resample('MS').mean().reset_index()
    monthly.columns = ['date', 'rate']
    return monthly


def load_rate_rba(data_dir):
    """RBA cash rate target: daily → monthly."""
    rba = pd.read_excel(data_dir / 'f01d.xlsx', sheet_name='Data', header=None)
    rba_d = rba.iloc[11:].copy()
    rba_d['date'] = pd.to_datetime(rba_d[0], errors='coerce')
    rba_d['rate'] = pd.to_numeric(rba_d[1], errors='coerce')
    rba_d = rba_d.dropna(subset=['date', 'rate']).sort_values('date')
    monthly = rba_d.set_index('date')['rate'].resample('MS').mean().reset_index()
    monthly.columns = ['date', 'rate']
    return monthly


def load_rate_riksbank(data_dir):
    """Riksbank repo rate: event-based → monthly."""
    swe = pd.read_excel(data_dir / 'styrrantan-effektiv.xlsx', header=None)
    swe = swe.iloc[1:].copy()
    swe['date'] = pd.to_datetime(swe[0], errors='coerce')
    swe['rate'] = pd.to_numeric(swe[1], errors='coerce')
    swe = swe.dropna().sort_values('date')
    daily = swe.set_index('date')['rate'].resample('D').ffill()
    monthly = daily.resample('MS').mean().reset_index()
    monthly.columns = ['date', 'rate']
    return monthly


# ================================================================
# EFFECTIVE RATE CONSTRUCTION
# ================================================================

def construct_effective_rate(alpha_df, rate_df, sp_return_df, alpha_ma=8):
    """
    Construct R̄ = (1-α)·i^policy + α·r^E for a given country.
    
    alpha_df: DataFrame with 'date', 'alpha'
    rate_df: DataFrame with 'date', 'rate'
    sp_return_df: DataFrame with 'date', 'r_equity' (trailing 7yr S&P real return)
    """
    # Interpolate alpha to monthly
    a = alpha_df.set_index('date')['alpha'].sort_index()
    a_monthly = a.resample('MS').interpolate('linear')
    a_smooth = a_monthly.rolling(alpha_ma, min_periods=1).mean()
    
    # Merge
    r = rate_df.set_index('date')['rate'].sort_index()
    sp = sp_return_df.set_index('date')['r_equity'].sort_index()
    
    df = pd.DataFrame({'alpha': a_smooth, 'policy_rate': r, 'r_equity': sp}).dropna()
    
    if len(df) < 12:
        return None
    
    df['eff_rate'] = (1 - df['alpha']) * df['policy_rate'] + df['alpha'] * df['r_equity']
    df['rate_gap'] = df['eff_rate'] - df['policy_rate']
    
    return df.reset_index()


# ================================================================
# CROSS-COUNTRY REGRESSION
# ================================================================

def cross_country_regression(results, period_start, period_end):
    """
    Compute average α and rate gap for each country in the given period,
    then regress gap on α.
    """
    points = []
    for country, df in results.items():
        sub = df[(df['date'] >= period_start) & (df['date'] <= period_end)]
        if len(sub) > 6:
            points.append({
                'country': country,
                'alpha': sub['alpha'].mean(),
                'gap': sub['rate_gap'].mean(),
                'ff': sub['policy_rate'].mean(),
                'eff': sub['eff_rate'].mean(),
                'n': len(sub),
            })
    
    if len(points) < 3:
        return None
    
    x = np.array([p['alpha'] for p in points])
    y = np.array([p['gap'] for p in points])
    slope, intercept, r, p, se = stats.linregress(x, y)
    
    return {
        'points': points,
        'slope': slope,
        'intercept': intercept,
        'r2': r ** 2,
        'p': p,
        'se': se,
    }


# ================================================================
# MAIN
# ================================================================

def main(data_dir=None, us_panel_path=None):
    if data_dir is None:
        data_dir = Path('/mnt/user-data/uploads')
    else:
        data_dir = Path(data_dir)
    
    if us_panel_path is None:
        for p in [Path(__file__).parent / 'panel_data.csv',
                  Path('panel_data.csv')]:
            if p.exists():
                us_panel_path = p
                break
    
    # Check for pre-built cross-country panel (avoids need for raw country files)
    cc_panel_path = None
    for p in [Path(__file__).parent / 'cross_country_panel_data.csv',
              Path('cross_country_panel_data.csv')]:
        if p.exists():
            cc_panel_path = p
            break
    
    print("=" * 80)
    print("CROSS-COUNTRY EFFECTIVE RATE ANALYSIS")
    print("=" * 80)
    
    results = {}
    
    if cc_panel_path is not None:
        # --- Fast path: load from pre-built panel ---
        print(f"\nLoading from pre-built panel: {cc_panel_path}")
        cc = pd.read_csv(cc_panel_path, parse_dates=['date'])
        for country in cc['country'].unique():
            cdf = cc[cc['country'] == country].copy()
            results[country] = cdf
            sub = cdf[(cdf['date'] >= '2016-01-01') & (cdf['date'] <= '2019-12-01')]
            a_avg = sub['alpha'].mean() if len(sub) > 0 else float('nan')
            gap   = sub['rate_gap'].mean() if len(sub) > 0 else float('nan')
            a_str   = f'{a_avg:.3f}' if not np.isnan(a_avg) else 'N/A'
            gap_str = f'{gap:+.2f}pp' if not np.isnan(gap) else 'N/A'
            print(f"  {country:20s}: n={len(cdf):>4d}, alpha(16-19)={a_str}, gap(16-19)={gap_str}")
    else:
        # --- Full path: build from raw country data files ---
        print(f"\nNo pre-built panel found. Loading raw country data from {data_dir}...")
        
        # Load US baseline
        us = pd.read_csv(us_panel_path, parse_dates=['date'])
        sp_return = us[['date', 'r_equity']].dropna()
        
        # Load all alpha series
        print("\nLoading alpha series...")
        alpha_series = {
            'Japan': load_alpha_japan(data_dir),
            'UK': load_alpha_uk(data_dir),
            'Canada': load_alpha_canada(data_dir),
            'Australia_narrow': load_alpha_oecd(data_dir, 'AUS'),
            'Australia_broad': load_alpha_australia_broad(data_dir),
            'Sweden': load_alpha_oecd(data_dir, 'SWE'),
        }
        
        for name, df in alpha_series.items():
            print(f"  {name:20s}: {len(df):>4d} obs, a=[{df['alpha'].min():.3f}, {df['alpha'].max():.3f}]")
        
        # Load policy rates
        print("\nLoading policy rates...")
        rate_series = {
            'Japan': load_rate_fred(data_dir, 'IRSTCI01JPM156N__2_.csv', 'IRSTCI01JPM156N'),
            'UK': load_rate_boe(data_dir),
            'Canada': load_rate_fred(data_dir, 'IRSTCI01CAM156N__1_.csv', 'IRSTCI01CAM156N'),
            'Australia': load_rate_rba(data_dir),
            'Sweden': load_rate_riksbank(data_dir),
        }
        
        for name, df in rate_series.items():
            print(f"  {name:20s}: {df['date'].min().date()} to {df['date'].max().date()}")
        
        # Construct effective rates
        print("\nConstructing effective rates...")
        
        # US baseline
        us_eff = us[['date', 'alpha', 'fed_funds', 'r_equity', 'eff_rate']].dropna()
        us_eff['policy_rate'] = us_eff['fed_funds']
        us_eff['rate_gap'] = us_eff['eff_rate'] - us_eff['policy_rate']
        results['US'] = us_eff
        
        # Other countries
        for country in ['Japan', 'UK', 'Canada', 'Australia_narrow', 'Australia_broad', 'Sweden']:
            base = country.replace('_narrow', '').replace('_broad', '')
            rate_key = base if base in rate_series else country
            
            df = construct_effective_rate(
                alpha_series[country],
                rate_series[rate_key],
                sp_return,
            )
            if df is not None:
                results[country] = df
                sub = df[(df['date'] >= '2016-01-01') & (df['date'] <= '2019-12-01')]
                gap = sub['rate_gap'].mean() if len(sub) > 0 else float('nan')
                print(f"  {country:20s}: n={len(df):>4d}, gap(2016-19)={gap:+.2f}pp")
    
    # Cross-country regressions
    for label, s, e in [("2016–2019", "2016-01-01", "2019-12-01"),
                         ("2009–2019", "2009-01-01", "2019-12-01")]:
        # Exclude Australia_broad from primary regression (use narrow for consistency)
        primary = {k: v for k, v in results.items() if k != 'Australia_broad'}
        reg = cross_country_regression(primary, s, e)
        
        if reg:
            print(f"\n{'='*80}")
            print(f"REGRESSION: α vs Rate Gap ({label})")
            print(f"{'='*80}")
            print(f"  gap = {reg['slope']:.2f}·α + {reg['intercept']:.2f}")
            print(f"  R² = {reg['r2']:.3f}, p = {reg['p']:.4f}, n = {len(reg['points'])}")
            print(f"\n  {'Country':<20s} {'α':>8} {'Gap':>8}")
            for p in sorted(reg['points'], key=lambda x: x['alpha']):
                print(f"  {p['country']:<20s} {p['alpha']:8.3f} {p['gap']:+8.2f}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--us-panel', type=str, default=None)
    args = parser.parse_args()
    main(args.data_dir, args.us_panel)
