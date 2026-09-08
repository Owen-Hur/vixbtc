"""
Fear & Greed Index — BTC Trading Signal Analysis
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from rv_regime.data_loader import load_and_prepare

# 저장소 루트 (data/ 가 있는 곳) — 실행 위치와 무관하게 파일 위치로부터 계산
BASE_DIR = Path(__file__).resolve().parents[2]

# ── Load data ────────────────────────────────────────────────────────
rv_daily, daily_ret = load_and_prepare()

fg = pd.read_parquet(BASE_DIR / 'data' / 'fear_greed_daily.parquet')
print(f"\n[F&G] shape={fg.shape}, columns={list(fg.columns)}")
print(fg.head())

# F&G has 'date' column, not index-based
fg = fg.set_index('date')
fg.index = pd.to_datetime(fg.index)
if hasattr(fg.index, 'tz') and fg.index.tz is not None:
    fg.index = fg.index.tz_localize(None)

# Strip tz from daily_ret
if hasattr(daily_ret.index, 'tz') and daily_ret.index.tz is not None:
    daily_ret = daily_ret.copy()
    daily_ret.index = daily_ret.index.tz_localize(None)

daily_ret.index = pd.to_datetime(daily_ret.index)

# Build combined df
df = pd.DataFrame({
    'fg': fg['fear_greed_value'].astype(float),
    'ret': daily_ret
}).dropna()

print(f"\n[Merged] {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
print(f"F&G range: {df['fg'].min():.0f} – {df['fg'].max():.0f}")

# =====================================================================
# 1. EXTREME LEVELS — CONTRARIAN ANALYSIS
# =====================================================================
print("\n" + "="*70)
print("1. EXTREME LEVELS — CONTRARIAN ANALYSIS")
print("="*70)

horizons = [1, 3, 5, 7, 14, 30]

# Compute forward returns
for h in horizons:
    df[f'fwd_{h}d'] = df['ret'].shift(-1).rolling(h).sum().shift(-(h-1))
    # Simpler: cumulative return over next h days
    # fwd_1d = ret[t+1], fwd_3d = ret[t+1]+ret[t+2]+ret[t+3]

# Actually let's be more precise with log-like cumulative
for h in horizons:
    fwd = df['ret'].copy()
    col = []
    for i in range(len(df)):
        if i + h < len(df):
            col.append(df['ret'].iloc[i+1:i+1+h].sum())
        else:
            col.append(np.nan)
    df[f'fwd_{h}d'] = col

mask_fear = df['fg'] < 20
mask_greed = df['fg'] > 80

print(f"\nExtreme Fear (F&G < 20): {mask_fear.sum()} days")
print(f"Extreme Greed (F&G > 80): {mask_greed.sum()} days")
print(f"Total days: {len(df)}")

print(f"\n{'Horizon':<10} {'Unconditional':>14} {'Extreme Fear':>14} {'Extreme Greed':>14} {'Fear-Greed':>14}")
print("-" * 70)
for h in horizons:
    col = f'fwd_{h}d'
    uncond = df[col].mean() * 100
    fear_ret = df.loc[mask_fear, col].mean() * 100
    greed_ret = df.loc[mask_greed, col].mean() * 100
    diff = (fear_ret - greed_ret)
    print(f"{h}d{'':<8} {uncond:>13.3f}% {fear_ret:>13.3f}% {greed_ret:>13.3f}% {diff:>13.3f}%")

# t-tests for extreme fear vs unconditional
print("\n  t-test: Extreme Fear fwd returns vs zero")
for h in [1, 7, 14, 30]:
    col = f'fwd_{h}d'
    vals = df.loc[mask_fear, col].dropna()
    t, p = stats.ttest_1samp(vals, 0)
    print(f"    {h}d: t={t:.2f}, p={p:.4f}, n={len(vals)}")

print("\n  t-test: Extreme Greed fwd returns vs zero")
for h in [1, 7, 14, 30]:
    col = f'fwd_{h}d'
    vals = df.loc[mask_greed, col].dropna()
    t, p = stats.ttest_1samp(vals, 0)
    print(f"    {h}d: t={t:.2f}, p={p:.4f}, n={len(vals)}")

# =====================================================================
# 2. F&G RATE OF CHANGE
# =====================================================================
print("\n" + "="*70)
print("2. F&G RATE OF CHANGE")
print("="*70)

df['fg_chg_7d'] = df['fg'] - df['fg'].shift(7)
df['fg_chg_14d'] = df['fg'] - df['fg'].shift(14)
df['next_ret'] = df['ret'].shift(-1)

for period in ['7d', '14d']:
    col = f'fg_chg_{period}'
    valid = df[[col, 'next_ret']].dropna()
    corr, pval = stats.pearsonr(valid[col], valid['next_ret'])
    print(f"\n  Corr(F&G {period} change, next-day ret): {corr:.4f}  (p={pval:.4f})")

# Fast drops / rises
for period in ['7d', '14d']:
    col = f'fg_chg_{period}'
    valid = df[[col, 'next_ret']].dropna()
    q10 = valid[col].quantile(0.10)
    q90 = valid[col].quantile(0.90)

    fast_drop = valid[valid[col] <= q10]
    fast_rise = valid[valid[col] >= q90]
    uncond = valid['next_ret'].mean()

    print(f"\n  F&G {period} change — Fast Drop (bottom 10%, <= {q10:.1f}):")
    print(f"    n={len(fast_drop)}, avg next-day ret: {fast_drop['next_ret'].mean()*100:.4f}%")
    print(f"    vs unconditional: {uncond*100:.4f}%")
    t, p = stats.ttest_1samp(fast_drop['next_ret'], uncond)
    print(f"    t-stat vs uncond: {t:.2f}, p={p:.4f}")

    print(f"  F&G {period} change — Fast Rise (top 10%, >= {q90:.1f}):")
    print(f"    n={len(fast_rise)}, avg next-day ret: {fast_rise['next_ret'].mean()*100:.4f}%")
    t, p = stats.ttest_1samp(fast_rise['next_ret'], uncond)
    print(f"    t-stat vs uncond: {t:.2f}, p={p:.4f}")

# =====================================================================
# 3. F&G REGIME DURATION
# =====================================================================
print("\n" + "="*70)
print("3. F&G REGIME DURATION")
print("="*70)

# Compute consecutive days in fear (<40) and greed (>60)
df['in_fear'] = (df['fg'] < 40).astype(int)
df['in_greed'] = (df['fg'] > 60).astype(int)

# Consecutive fear days
fear_consec = []
count = 0
for v in df['in_fear']:
    if v:
        count += 1
    else:
        count = 0
    fear_consec.append(count)
df['fear_streak'] = fear_consec

greed_consec = []
count = 0
for v in df['in_greed']:
    if v:
        count += 1
    else:
        count = 0
    greed_consec.append(count)
df['greed_streak'] = greed_consec

print(f"\n  Max consecutive fear days (<40): {df['fear_streak'].max()}")
print(f"  Max consecutive greed days (>60): {df['greed_streak'].max()}")

# After 7+ consecutive fear days
mask_7fear = df['fear_streak'] >= 7
mask_7greed = df['greed_streak'] >= 7

print(f"\n  Days with 7+ consecutive fear: {mask_7fear.sum()}")
if mask_7fear.sum() > 0:
    avg_ret = df.loc[mask_7fear, 'next_ret'].mean() * 100
    t, p = stats.ttest_1samp(df.loc[mask_7fear, 'next_ret'].dropna(), 0)
    print(f"  Avg next-day return after 7+ fear days: {avg_ret:.4f}%  (t={t:.2f}, p={p:.4f})")

print(f"\n  Days with 7+ consecutive greed: {mask_7greed.sum()}")
if mask_7greed.sum() > 0:
    avg_ret = df.loc[mask_7greed, 'next_ret'].mean() * 100
    t, p = stats.ttest_1samp(df.loc[mask_7greed, 'next_ret'].dropna(), 0)
    print(f"  Avg next-day return after 7+ greed days: {avg_ret:.4f}%  (t={t:.2f}, p={p:.4f})")

# Distribution of streak lengths
print("\n  Fear streak distribution:")
# Get streak lengths (count at end of each streak)
fear_ends = df['fear_streak'].copy()
fear_ends_shifted = fear_ends.shift(-1).fillna(0)
streak_ends = fear_ends[(fear_ends > 0) & (fear_ends_shifted == 0)]
if len(streak_ends) > 0:
    print(f"    # of fear streaks: {len(streak_ends)}")
    print(f"    Mean length: {streak_ends.mean():.1f}, Median: {streak_ends.median():.0f}, Max: {streak_ends.max()}")

print("\n  Greed streak distribution:")
greed_ends = df['greed_streak'].copy()
greed_ends_shifted = greed_ends.shift(-1).fillna(0)
streak_ends_g = greed_ends[(greed_ends > 0) & (greed_ends_shifted == 0)]
if len(streak_ends_g) > 0:
    print(f"    # of greed streaks: {len(streak_ends_g)}")
    print(f"    Mean length: {streak_ends_g.mean():.1f}, Median: {streak_ends_g.median():.0f}, Max: {streak_ends_g.max()}")

# =====================================================================
# 4. F&G LEVEL BINS ANALYSIS
# =====================================================================
print("\n" + "="*70)
print("4. F&G LEVEL BINS ANALYSIS")
print("="*70)

bins = [0, 20, 40, 60, 80, 100]
labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
df['fg_bin'] = pd.cut(df['fg'], bins=bins, labels=labels, include_lowest=True)

print(f"\n{'Bin':<10} {'Count':>8} {'Avg Next-Day Ret':>18} {'Std':>10} {'t-stat':>10} {'p-value':>10}")
print("-" * 70)
for label in labels:
    mask = df['fg_bin'] == label
    vals = df.loc[mask, 'next_ret'].dropna()
    n = len(vals)
    mean = vals.mean() * 100
    std = vals.std() * 100
    if n > 1:
        t, p = stats.ttest_1samp(vals, 0)
    else:
        t, p = np.nan, np.nan
    print(f"{label:<10} {n:>8} {mean:>17.4f}% {std:>9.3f}% {t:>10.2f} {p:>10.4f}")

# Unconditional
vals_all = df['next_ret'].dropna()
print(f"{'All':<10} {len(vals_all):>8} {vals_all.mean()*100:>17.4f}% {vals_all.std()*100:>9.3f}%")

# =====================================================================
# 5. PRACTICAL SIGNAL TEST — CONTRARIAN STRATEGY
# =====================================================================
print("\n" + "="*70)
print("5. PRACTICAL SIGNAL TEST — CONTRARIAN STRATEGY")
print("="*70)

# Signal: long when F&G < 25, short when F&G > 75, else long (BTC bias)
df['signal'] = 1  # default long
df.loc[df['fg'] < 25, 'signal'] = 1   # long on extreme fear
df.loc[df['fg'] > 75, 'signal'] = -1  # short on extreme greed

df['strat_ret'] = df['signal'] * df['next_ret']
df['bh_ret'] = df['next_ret']  # buy & hold = always long

valid = df[['strat_ret', 'bh_ret']].dropna()

ann_factor = np.sqrt(365)

# Strategy stats
strat_mean = valid['strat_ret'].mean()
strat_std = valid['strat_ret'].std()
strat_sharpe = strat_mean / strat_std * ann_factor
strat_cum = (1 + valid['strat_ret']).cumprod()

bh_mean = valid['bh_ret'].mean()
bh_std = valid['bh_ret'].std()
bh_sharpe = bh_mean / bh_std * ann_factor
bh_cum = (1 + valid['bh_ret']).cumprod()

excess_mean = (strat_mean - bh_mean) * 365 * 100

print(f"\n  Strategy: Long (F&G<25), Short (F&G>75), else Long")
print(f"  Period: {valid.index[0].date()} to {valid.index[-1].date()} ({len(valid)} days)")
print(f"\n  {'Metric':<25} {'Strategy':>12} {'Buy & Hold':>12}")
print(f"  {'-'*50}")
print(f"  {'Avg daily ret':<25} {strat_mean*100:>11.4f}% {bh_mean*100:>11.4f}%")
print(f"  {'Daily vol':<25} {strat_std*100:>11.4f}% {bh_std*100:>11.4f}%")
print(f"  {'Sharpe (ann.)':<25} {strat_sharpe:>12.3f} {bh_sharpe:>12.3f}")
print(f"  {'Cumul. return':<25} {(strat_cum.iloc[-1]-1)*100:>11.1f}% {(bh_cum.iloc[-1]-1)*100:>11.1f}%")
print(f"  {'Excess ann. ret':<25} {excess_mean:>11.2f}%")

# Signal breakdown
n_long = (df['signal'] == 1).sum()
n_short = (df['signal'] == -1).sum()
print(f"\n  Signal distribution: Long={n_long} ({n_long/len(df)*100:.1f}%), Short={n_short} ({n_short/len(df)*100:.1f}%)")

# Permutation test
print(f"\n  Shuffled test (2000 permutations)...")
np.random.seed(42)
actual_sharpe = strat_sharpe
signals = df['signal'].values
next_rets = df['next_ret'].values

# Remove NaN for permutation
valid_mask = ~np.isnan(next_rets)
signals_v = signals[valid_mask]
rets_v = next_rets[valid_mask]

n_perms = 2000
shuffled_sharpes = np.empty(n_perms)
for i in range(n_perms):
    perm_signals = np.random.permutation(signals_v)
    perm_ret = perm_signals * rets_v
    m = perm_ret.mean()
    s = perm_ret.std()
    shuffled_sharpes[i] = m / s * ann_factor

p_value = (shuffled_sharpes >= actual_sharpe).mean()
print(f"  Actual Sharpe: {actual_sharpe:.3f}")
print(f"  Shuffled Sharpe: mean={shuffled_sharpes.mean():.3f}, std={shuffled_sharpes.std():.3f}")
print(f"  p-value (fraction >= actual): {p_value:.4f}")
print(f"  Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "="*70)
print("SUMMARY OF KEY FINDINGS")
print("="*70)
fear_1d = df.loc[mask_fear, 'fwd_1d'].mean() * 100
fear_7d = df.loc[mask_fear, 'fwd_7d'].mean() * 100
greed_1d = df.loc[mask_greed, 'fwd_1d'].mean() * 100
greed_7d = df.loc[mask_greed, 'fwd_7d'].mean() * 100
uncond_1d = df['fwd_1d'].mean() * 100
print(f"  Unconditional avg 1d return:      {uncond_1d:.4f}%")
print(f"  After Extreme Fear (<20) 1d:      {fear_1d:.4f}%  (7d: {fear_7d:.4f}%)")
print(f"  After Extreme Greed (>80) 1d:     {greed_1d:.4f}%  (7d: {greed_7d:.4f}%)")
print(f"  Contrarian strategy Sharpe:        {strat_sharpe:.3f} vs B&H {bh_sharpe:.3f}")
print(f"  Permutation p-value:               {p_value:.4f}")
