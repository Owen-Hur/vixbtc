#!/usr/bin/env python3
"""
BTC ETF Flow vs BTC Price Correlation Analysis
================================================
Analyzes the relationship between BTC spot ETF trading activity
and BTC price movements since ETF launch (Jan 11, 2024).
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────
# 1. Download ETF Data via yfinance
# ─────────────────────────────────────────────────────────────
print("=" * 80)
print("STEP 1: Downloading BTC ETF daily data (yfinance)")
print("=" * 80)

import yfinance as yf

ETF_TICKERS = ['IBIT', 'FBTC', 'GBTC', 'ARKB', 'BITB']
ETF_START = '2024-01-11'
ETF_END = '2026-05-28'

etf_data = {}
for ticker in ETF_TICKERS:
    print(f"  Downloading {ticker}...")
    df = yf.download(ticker, start=ETF_START, end=ETF_END, auto_adjust=True, progress=False)
    if len(df) > 0:
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        etf_data[ticker] = df
        print(f"    {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    else:
        print(f"    WARNING: No data for {ticker}")

# Build combined ETF DataFrame
print("\nBuilding combined ETF DataFrame...")

# Dollar volume for each ETF
dv_frames = {}
ret_frames = {}
vol_frames = {}
close_frames = {}

for ticker, df in etf_data.items():
    close_frames[ticker] = df['Close']
    vol_frames[ticker] = df['Volume']
    dv_frames[ticker] = df['Close'] * df['Volume']
    ret_frames[ticker] = df['Close'].pct_change()

etf_dollar_vol = pd.DataFrame(dv_frames)
etf_returns = pd.DataFrame(ret_frames)
etf_volume = pd.DataFrame(vol_frames)
etf_close = pd.DataFrame(close_frames)

# Combined metrics
etf_dollar_vol['Combined'] = etf_dollar_vol.sum(axis=1)
etf_volume['Combined'] = etf_volume.sum(axis=1)

# Save to parquet
save_df = pd.DataFrame({
    'ibit_close': etf_close.get('IBIT'),
    'fbtc_close': etf_close.get('FBTC'),
    'gbtc_close': etf_close.get('GBTC'),
    'arkb_close': etf_close.get('ARKB'),
    'bitb_close': etf_close.get('BITB'),
    'ibit_volume': etf_volume.get('IBIT'),
    'fbtc_volume': etf_volume.get('FBTC'),
    'gbtc_volume': etf_volume.get('GBTC'),
    'arkb_volume': etf_volume.get('ARKB'),
    'bitb_volume': etf_volume.get('BITB'),
    'combined_dollar_vol': etf_dollar_vol['Combined'],
    'combined_volume': etf_volume['Combined'],
})
save_path = Path(__file__).parent.parent / 'data' / 'btc_etf_daily.parquet'
save_df.to_parquet(save_path)
print(f"\nSaved ETF data to {save_path}")
print(f"  Shape: {save_df.shape}, Date range: {save_df.index[0].date()} to {save_df.index[-1].date()}")

# ─────────────────────────────────────────────────────────────
# 2. Load BTC Daily Returns
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 2: Loading BTC daily returns")
print("=" * 80)

from rv_regime.data_loader import load_and_prepare
rv_daily, daily_ret = load_and_prepare()

# Make BTC returns tz-naive for alignment
btc_ret = daily_ret.copy()
if hasattr(btc_ret.index, 'tz') and btc_ret.index.tz is not None:
    btc_ret.index = btc_ret.index.tz_localize(None)
btc_ret.index = pd.to_datetime(btc_ret.index).normalize()
btc_ret = btc_ret[~btc_ret.index.duplicated(keep='last')]

# Also get RV tz-naive
rv = rv_daily.copy()
if hasattr(rv.index, 'tz') and rv.index.tz is not None:
    rv.index = rv.index.tz_localize(None)
rv.index = pd.to_datetime(rv.index).normalize()
rv = rv[~rv.index.duplicated(keep='last')]

# Align to ETF period
etf_start_dt = pd.Timestamp('2024-01-11')
btc_ret_etf = btc_ret[btc_ret.index >= etf_start_dt]
rv_etf = rv[rv.index >= etf_start_dt]

print(f"\nBTC returns in ETF period: {len(btc_ret_etf)} days")
print(f"  {btc_ret_etf.index[0].date()} to {btc_ret_etf.index[-1].date()}")

# ─────────────────────────────────────────────────────────────
# 3. Merge ETF and BTC data
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 3: Merging datasets")
print("=" * 80)

# Normalize ETF index
etf_dollar_vol.index = pd.to_datetime(etf_dollar_vol.index).normalize()
etf_returns.index = pd.to_datetime(etf_returns.index).normalize()
etf_volume.index = pd.to_datetime(etf_volume.index).normalize()

# Create merged DataFrame
merged = pd.DataFrame({
    'btc_ret': btc_ret_etf,
    'combined_dvol': etf_dollar_vol['Combined'],
    'combined_vol': etf_volume['Combined'],
    'ibit_ret': etf_returns.get('IBIT'),
    'gbtc_ret': etf_returns.get('GBTC'),
    'ibit_vol': etf_volume.get('IBIT'),
    'gbtc_vol': etf_volume.get('GBTC'),
    'rv': rv_etf,
}).dropna(subset=['btc_ret', 'combined_dvol'])

# Add lagged BTC returns (next day)
merged['btc_ret_next1d'] = merged['btc_ret'].shift(-1)
merged['btc_ret_next2d'] = merged['btc_ret'].shift(-1).rolling(2).sum()  # cumulative 2d
merged['btc_ret_next3d'] = merged['btc_ret'].shift(-1).rolling(3).sum()
merged['btc_ret_next5d'] = merged['btc_ret'].shift(-1).rolling(5).sum()

# Log dollar volume
merged['log_dvol'] = np.log(merged['combined_dvol'].replace(0, np.nan))

# Volume percentile (expanding window to avoid lookahead)
merged['dvol_pctl'] = merged['combined_dvol'].expanding().rank(pct=True)

# GBTC share of volume
if 'gbtc_vol' in merged.columns and 'ibit_vol' in merged.columns:
    merged['gbtc_ibit_ratio'] = merged['gbtc_vol'] / merged['ibit_vol'].replace(0, np.nan)

print(f"Merged dataset: {len(merged)} trading days")
print(f"  {merged.index[0].date()} to {merged.index[-1].date()}")
print(f"  Columns: {list(merged.columns)}")

# ─────────────────────────────────────────────────────────────
# Helper: correlation with t-test
# ─────────────────────────────────────────────────────────────
def corr_with_test(x, y, label=""):
    """Pearson correlation with t-stat and p-value."""
    mask = x.notna() & y.notna()
    x_, y_ = x[mask], y[mask]
    n = len(x_)
    if n < 10:
        return None
    r, p = stats.pearsonr(x_, y_)
    t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else np.inf
    sig = ""
    if p < 0.01:
        sig = "***"
    elif p < 0.05:
        sig = "**"
    elif p < 0.10:
        sig = "*"
    print(f"  {label:45s}  r={r:+.4f}  t={t_stat:+.3f}  p={p:.4f}  n={n}  {sig}")
    return r, p, n

def mean_test(group, label=""):
    """One-sample t-test: is mean significantly different from 0?"""
    x = group.dropna()
    n = len(x)
    if n < 5:
        return None
    t_stat, p = stats.ttest_1samp(x, 0)
    mean = x.mean()
    std = x.std()
    sig = ""
    if p < 0.01:
        sig = "***"
    elif p < 0.05:
        sig = "**"
    elif p < 0.10:
        sig = "*"
    print(f"  {label:45s}  mean={mean:+.4f}  std={std:.4f}  t={t_stat:+.3f}  p={p:.4f}  n={n}  {sig}")
    return mean, t_stat, p, n

def two_sample_test(g1, g2, l1="Group1", l2="Group2"):
    """Welch's t-test between two groups."""
    g1, g2 = g1.dropna(), g2.dropna()
    if len(g1) < 5 or len(g2) < 5:
        return None
    t_stat, p = stats.ttest_ind(g1, g2, equal_var=False)
    print(f"    {l1}: mean={g1.mean():+.4f} (n={len(g1)})  vs  {l2}: mean={g2.mean():+.4f} (n={len(g2)})")
    print(f"    Welch t={t_stat:+.3f}, p={p:.4f}{'  ***' if p<0.01 else '  **' if p<0.05 else '  *' if p<0.1 else ''}")
    return t_stat, p


# ═════════════════════════════════════════════════════════════
# ANALYSIS 1: ETF Volume vs BTC Returns
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS 1: ETF Dollar Volume vs BTC Returns (Correlations)")
print("=" * 80)

print("\n[1a] Same-day correlations:")
corr_with_test(merged['log_dvol'], merged['btc_ret'], "Log(DollarVolume) vs BTC Return")
corr_with_test(merged['combined_dvol'], merged['btc_ret'], "DollarVolume vs BTC Return")
corr_with_test(merged['log_dvol'], merged['btc_ret'].abs(), "Log(DollarVolume) vs |BTC Return|")

print("\n[1b] Next-day predictive correlations:")
corr_with_test(merged['log_dvol'], merged['btc_ret_next1d'], "Log(DollarVolume) vs BTC Ret(t+1)")
corr_with_test(merged['log_dvol'], merged['btc_ret_next2d'], "Log(DollarVolume) vs BTC Ret(t+1:t+2)")
corr_with_test(merged['log_dvol'], merged['btc_ret_next3d'], "Log(DollarVolume) vs BTC Ret(t+1:t+3)")
corr_with_test(merged['log_dvol'], merged['btc_ret_next5d'], "Log(DollarVolume) vs BTC Ret(t+1:t+5)")

print("\n[1c] Volume change vs next-day return:")
merged['dvol_chg'] = merged['log_dvol'].diff()
corr_with_test(merged['dvol_chg'], merged['btc_ret_next1d'], "DollarVol Change vs BTC Ret(t+1)")

print("\n[1d] Signed volume (volume * sign of return):")
merged['signed_dvol'] = merged['log_dvol'] * np.sign(merged['btc_ret'])
corr_with_test(merged['signed_dvol'], merged['btc_ret_next1d'], "SignedVolume vs BTC Ret(t+1)")
corr_with_test(merged['signed_dvol'], merged['btc_ret_next5d'], "SignedVolume vs BTC Ret(t+1:t+5)")


# ═════════════════════════════════════════════════════════════
# ANALYSIS 2: IBIT Return vs BTC Return (Tracking Quality)
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS 2: IBIT vs BTC Return (Tracking & Lead-Lag)")
print("=" * 80)

print("\n[2a] Same-day tracking:")
corr_with_test(merged['ibit_ret'], merged['btc_ret'], "IBIT Ret vs BTC Ret (same day)")

print("\n[2b] Lead-lag structure:")
for lag in [1, 2, 3, 5]:
    merged[f'ibit_ret_lag{lag}'] = merged['ibit_ret'].shift(lag)
    corr_with_test(merged[f'ibit_ret_lag{lag}'], merged['btc_ret'],
                   f"IBIT Ret(t-{lag}) vs BTC Ret(t)")

print("\n[2c] Does ETF lead BTC? (IBIT today vs BTC tomorrow)")
corr_with_test(merged['ibit_ret'], merged['btc_ret_next1d'], "IBIT Ret(t) vs BTC Ret(t+1)")


# ═════════════════════════════════════════════════════════════
# ANALYSIS 3: Volume Surge Analysis
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS 3: Volume Surge Analysis")
print("=" * 80)

for pctl_threshold in [0.75, 0.90, 0.95]:
    high_vol = merged[merged['dvol_pctl'] > pctl_threshold]
    low_vol = merged[merged['dvol_pctl'] <= pctl_threshold]

    print(f"\n[3a] Volume > {int(pctl_threshold*100)}th percentile ({len(high_vol)} days):")
    mean_test(high_vol['btc_ret_next1d'], f"Next-day BTC return (vol > p{int(pctl_threshold*100)})")
    mean_test(low_vol['btc_ret_next1d'], f"Next-day BTC return (vol <= p{int(pctl_threshold*100)})")
    two_sample_test(high_vol['btc_ret_next1d'], low_vol['btc_ret_next1d'],
                    f"HighVol(>{int(pctl_threshold*100)})", f"LowVol(<={int(pctl_threshold*100)})")

print("\n[3b] Volume surge + direction (continuation vs reversal):")
high_vol_90 = merged[merged['dvol_pctl'] > 0.90]

# Positive return + high volume
pos_high = high_vol_90[high_vol_90['btc_ret'] > 0]
neg_high = high_vol_90[high_vol_90['btc_ret'] < 0]

print(f"\n  High volume + UP day ({len(pos_high)} days):")
mean_test(pos_high['btc_ret_next1d'], "Next-day return after HighVol UP")
mean_test(pos_high['btc_ret_next5d'], "Next-5day return after HighVol UP")

print(f"\n  High volume + DOWN day ({len(neg_high)} days):")
mean_test(neg_high['btc_ret_next1d'], "Next-day return after HighVol DOWN")
mean_test(neg_high['btc_ret_next5d'], "Next-5day return after HighVol DOWN")

if len(pos_high) >= 5 and len(neg_high) >= 5:
    print(f"\n  Continuation test (HighVol UP vs HighVol DOWN):")
    two_sample_test(pos_high['btc_ret_next1d'], neg_high['btc_ret_next1d'],
                    "After UP+HighVol", "After DOWN+HighVol")


# ═════════════════════════════════════════════════════════════
# ANALYSIS 4: GBTC Outflow Proxy
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS 4: GBTC/IBIT Volume Ratio as Sentiment Indicator")
print("=" * 80)

if 'gbtc_ibit_ratio' in merged.columns:
    ratio = merged['gbtc_ibit_ratio'].dropna()
    print(f"\nGBTC/IBIT volume ratio stats:")
    print(f"  Mean:   {ratio.mean():.4f}")
    print(f"  Median: {ratio.median():.4f}")
    print(f"  Std:    {ratio.std():.4f}")
    print(f"  Min:    {ratio.min():.4f}  Max: {ratio.max():.4f}")

    # Monthly evolution
    print(f"\n  Monthly evolution of GBTC/IBIT ratio:")
    monthly = merged.groupby(merged.index.to_period('M'))['gbtc_ibit_ratio'].mean()
    for period, val in monthly.items():
        if not np.isnan(val):
            print(f"    {period}: {val:.4f}")

    print(f"\n[4a] GBTC/IBIT ratio vs BTC returns:")
    corr_with_test(merged['gbtc_ibit_ratio'], merged['btc_ret'],
                   "GBTC/IBIT Ratio vs BTC Ret (same day)")
    corr_with_test(merged['gbtc_ibit_ratio'], merged['btc_ret_next1d'],
                   "GBTC/IBIT Ratio vs BTC Ret (t+1)")
    corr_with_test(merged['gbtc_ibit_ratio'], merged['btc_ret_next5d'],
                   "GBTC/IBIT Ratio vs BTC Ret (t+1:t+5)")

    # High GBTC ratio = more GBTC selling pressure
    ratio_median = merged['gbtc_ibit_ratio'].median()
    high_gbtc = merged[merged['gbtc_ibit_ratio'] > ratio_median]
    low_gbtc = merged[merged['gbtc_ibit_ratio'] <= ratio_median]

    print(f"\n[4b] Split by GBTC/IBIT ratio median ({ratio_median:.4f}):")
    mean_test(high_gbtc['btc_ret_next1d'], "Next-day ret when GBTC ratio HIGH")
    mean_test(low_gbtc['btc_ret_next1d'], "Next-day ret when GBTC ratio LOW")
    two_sample_test(high_gbtc['btc_ret_next1d'], low_gbtc['btc_ret_next1d'],
                    "HighGBTC", "LowGBTC")


# ═════════════════════════════════════════════════════════════
# ANALYSIS 5: Predictive Analysis & Lag Structure
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS 5: Comprehensive Lag Correlation Matrix")
print("=" * 80)

print("\n[5a] ETF combined dollar volume → BTC return at various lags:")
for lag in [1, 2, 3, 5, 10]:
    lagged_ret = merged['btc_ret'].shift(-lag)
    corr_with_test(merged['log_dvol'], lagged_ret, f"Log(DollarVol)(t) vs BTC Ret(t+{lag})")

print("\n[5b] BTC return → ETF volume at various lags (reverse causation check):")
for lag in [1, 2, 3, 5]:
    lagged_vol = merged['log_dvol'].shift(-lag)
    corr_with_test(merged['btc_ret'], lagged_vol, f"BTC Ret(t) vs Log(DollarVol)(t+{lag})")

print("\n[5c] Volume z-score predictive power:")
merged['dvol_zscore'] = (merged['log_dvol'] - merged['log_dvol'].expanding().mean()) / merged['log_dvol'].expanding().std()
corr_with_test(merged['dvol_zscore'], merged['btc_ret_next1d'], "Vol Z-score vs BTC Ret(t+1)")
corr_with_test(merged['dvol_zscore'], merged['btc_ret_next5d'], "Vol Z-score vs BTC Ret(t+1:t+5)")

# Quintile analysis
print("\n[5d] Volume quintile analysis (next-day returns):")
merged['dvol_quintile'] = pd.qcut(merged['log_dvol'], 5, labels=['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High'], duplicates='drop')
quintile_stats = merged.groupby('dvol_quintile')['btc_ret_next1d'].agg(['mean', 'std', 'count'])
for q in quintile_stats.index:
    row = quintile_stats.loc[q]
    t_stat = row['mean'] / (row['std'] / np.sqrt(row['count'])) if row['std'] > 0 else 0
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), row['count'] - 1))
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
    print(f"  {str(q):10s}: mean={row['mean']:+.4f}  std={row['std']:.4f}  n={int(row['count'])}  t={t_stat:+.3f}  p={p_val:.4f}  {sig}")


# ═════════════════════════════════════════════════════════════
# ANALYSIS 6: US vs Non-US Session Analysis
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS 6: ETF Volume Impact on US vs Non-US Session Returns")
print("=" * 80)

# Load 1-minute data for session analysis
print("\nLoading 1-minute data for intraday session analysis...")
from rv_regime.data_loader import load_btc_1m

df_1m = load_btc_1m()

# Define sessions: US hours = 9:30-16:00 ET, Non-US = rest
# ETF trading during US hours could affect non-US session (overnight)
# US session: 9:30 ET to 16:00 ET (same day)
# Non-US session: 16:00 ET to 9:30 ET (next day)

idx = df_1m.index
us_mask = (idx.hour > 9) | ((idx.hour == 9) & (idx.minute >= 30))
us_mask = us_mask & (idx.hour < 16)

# Compute US session returns
us_data = df_1m[us_mask].copy()
nonus_data = df_1m[~us_mask].copy()

# Group by date for US session
us_data['date'] = us_data.index.normalize()
nonus_data['date'] = nonus_data.index.normalize()

us_session_ret = us_data.groupby('date')['close'].apply(lambda g: g.iloc[-1] / g.iloc[0] - 1 if len(g) > 1 else np.nan)
nonus_session_ret = nonus_data.groupby('date')['close'].apply(lambda g: g.iloc[-1] / g.iloc[0] - 1 if len(g) > 1 else np.nan)

# Make tz-naive
us_session_ret.index = pd.to_datetime(us_session_ret.index).tz_localize(None).normalize()
nonus_session_ret.index = pd.to_datetime(nonus_session_ret.index).tz_localize(None).normalize()

# Filter to ETF period
us_ret_etf = us_session_ret[us_session_ret.index >= etf_start_dt]
nonus_ret_etf = nonus_session_ret[nonus_session_ret.index >= etf_start_dt]

# Merge session returns
merged['us_ret'] = us_ret_etf
merged['nonus_ret'] = nonus_ret_etf
# Non-US session on day T+1 follows US session on day T
merged['nonus_ret_next'] = nonus_ret_etf.shift(-1)

print(f"\nUS session returns: {merged['us_ret'].notna().sum()} days")
print(f"Non-US session returns: {merged['nonus_ret'].notna().sum()} days")

print("\n[6a] ETF volume vs US session return (same day):")
corr_with_test(merged['log_dvol'], merged['us_ret'], "Log(DollarVol) vs US Session Ret (same day)")

print("\n[6b] ETF volume vs Non-US session return (next session):")
corr_with_test(merged['log_dvol'], merged['nonus_ret_next'], "Log(DollarVol) vs Non-US Ret (next session)")

print("\n[6c] US session return → Non-US session return:")
corr_with_test(merged['us_ret'], merged['nonus_ret_next'], "US Ret(t) vs Non-US Ret(t+1)")

print("\n[6d] High ETF volume + US direction → Non-US session:")
high_vol_90_session = merged[merged['dvol_pctl'] > 0.90]
pos_us = high_vol_90_session[high_vol_90_session['us_ret'] > 0]
neg_us = high_vol_90_session[high_vol_90_session['us_ret'] < 0]

print(f"\n  HighVol + US UP ({len(pos_us)} days):")
mean_test(pos_us['nonus_ret_next'], "Next Non-US ret after HighVol US-UP")

print(f"\n  HighVol + US DOWN ({len(neg_us)} days):")
mean_test(neg_us['nonus_ret_next'], "Next Non-US ret after HighVol US-DOWN")


# ═════════════════════════════════════════════════════════════
# ANALYSIS 7: Regime-Conditional Analysis
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS 7: Regime-Conditional ETF Volume Analysis")
print("=" * 80)

if 'rv' in merged.columns and merged['rv'].notna().sum() > 20:
    rv_median = merged['rv'].median()
    high_rv = merged[merged['rv'] > rv_median]
    low_rv = merged[merged['rv'] <= rv_median]

    print(f"\n[7a] RV regime split (median RV = {rv_median:.4f}):")
    print(f"  High RV: {len(high_rv)} days,  Low RV: {len(low_rv)} days")

    print(f"\n  In HIGH volatility regime:")
    corr_with_test(high_rv['log_dvol'], high_rv['btc_ret_next1d'],
                   "Log(DollarVol) vs BTC Ret(t+1) [HighVol]")

    print(f"\n  In LOW volatility regime:")
    corr_with_test(low_rv['log_dvol'], low_rv['btc_ret_next1d'],
                   "Log(DollarVol) vs BTC Ret(t+1) [LowVol]")

    print(f"\n[7b] Signed volume predictive power by regime:")
    print(f"  HIGH volatility regime:")
    corr_with_test(high_rv['signed_dvol'], high_rv['btc_ret_next1d'],
                   "SignedVol vs BTC Ret(t+1) [HighVol]")
    print(f"  LOW volatility regime:")
    corr_with_test(low_rv['signed_dvol'], low_rv['btc_ret_next1d'],
                   "SignedVol vs BTC Ret(t+1) [LowVol]")


# ═════════════════════════════════════════════════════════════
# ANALYSIS 8: Rolling Correlation (Stability Check)
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS 8: Rolling Correlation Stability")
print("=" * 80)

for window in [30, 60, 90]:
    roll_corr = merged['log_dvol'].rolling(window).corr(merged['btc_ret_next1d'])
    print(f"\n  Rolling {window}d corr(Log(DollarVol), BTC Ret(t+1)):")
    print(f"    Mean: {roll_corr.mean():.4f}  Std: {roll_corr.std():.4f}")
    print(f"    Min:  {roll_corr.min():.4f}  Max: {roll_corr.max():.4f}")
    pct_pos = (roll_corr > 0).sum() / roll_corr.notna().sum() * 100
    print(f"    % positive: {pct_pos:.1f}%")


# ═════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY OF KEY FINDINGS")
print("=" * 80)

print("""
Key findings are printed above with significance levels:
  *** = p < 0.01 (highly significant)
  **  = p < 0.05 (significant)
  *   = p < 0.10 (marginally significant)

Notes:
- ETF data is from yfinance (price * volume as dollar volume proxy)
- True fund flows (creation/redemption) not available via yfinance
- Dollar volume is a proxy for flow intensity, not direction
- GBTC/IBIT ratio captures the structural GBTC→IBIT rotation
- US session = 9:30-16:00 ET, Non-US = rest of 24h
- All predictive tests use t+1 or later to avoid lookahead
""")

print("Analysis complete.")
