"""
Final Combined Strategy: DVOL + ETF Volume + Risk Filters
Long-only with position sizing (0 or 1), NO optimization.

Signals used (all statistically significant):
1. DVOL level: High IV = bullish (Q5 +0.42%/d, t=2.04)
2. DVOL change lag2: IV rising 2d ago = bullish (r=0.092, p=0.004)
3. ETF volume surge: institutional buying = bullish overnight (p=0.016)
4. Google Trends: FOMO filter (search > P80 = reduce)
5. F&G: momentum confirmation (greed = bullish)

Strategy variants (all fixed-parameter, no optimization):
A. DVOL-only: long when DVOL > median, flat otherwise
B. DVOL momentum: long when DVOL 2d change > 0, flat otherwise
C. DVOL combined: long when DVOL > median OR DVOL rising
D. Full combo: C + Google Trends risk filter
E. Full combo + ETF (shorter period)
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from rv_regime.data_loader import load_and_prepare
from rv_regime.config import TAKER_FEE


def load_all_data():
    _, daily_ret = load_and_prepare()
    daily_ret = daily_ret[~daily_ret.index.duplicated(keep='first')]
    tz = daily_ret.index.tz

    # DVOL
    dvol_raw = pd.read_parquet('data/dvol_daily.parquet')
    dvol = dvol_raw['dvol'].copy()
    dvol.index = pd.DatetimeIndex(dvol.index).normalize().tz_localize(tz)
    dvol = dvol[~dvol.index.duplicated(keep='first')]

    # Google Trends
    gt = pd.read_parquet('data/google_trends_bitcoin.parquet')
    gt_col = [c for c in gt.columns if c != 'date'][0] if 'date' not in gt.index.name else gt.columns[0]
    if 'date' in gt.columns:
        gt_series = pd.Series(gt[gt_col].values if gt_col != 'date' else gt.iloc[:, 1].values,
                              index=pd.DatetimeIndex(gt['date']))
    else:
        gt_series = gt.iloc[:, 0]
    gt_series.index = gt_series.index.normalize().tz_localize(tz)
    gt_series = gt_series[~gt_series.index.duplicated(keep='first')]
    # Forward fill weekly to daily
    gt_daily = gt_series.resample('D').ffill()

    # F&G
    fg = pd.read_parquet('data/fear_greed_daily.parquet')
    fg_idx = pd.DatetimeIndex(pd.to_datetime(fg['date'])).normalize().tz_localize(tz)
    fear_greed = pd.Series(fg['fear_greed_value'].values, index=fg_idx)
    fear_greed = fear_greed[~fear_greed.index.duplicated(keep='first')]

    # ETF volume
    try:
        etf = pd.read_parquet('data/btc_etf_daily.parquet')
        if 'date' in etf.columns:
            etf_idx = pd.DatetimeIndex(pd.to_datetime(etf['date'])).normalize().tz_localize(tz)
        else:
            etf_idx = pd.DatetimeIndex(etf.index).normalize().tz_localize(tz)
        
        vol_col = [c for c in etf.columns if 'total_dollar' in c.lower() or 'dollar_vol' in c.lower()]
        if vol_col:
            etf_vol = pd.Series(etf[vol_col[0]].values, index=etf_idx)
        elif 'total_volume' in etf.columns:
            etf_vol = pd.Series(etf['total_volume'].values, index=etf_idx)
        else:
            # Try to find any volume column
            etf_vol = None
        if etf_vol is not None:
            etf_vol = etf_vol[~etf_vol.index.duplicated(keep='first')]
    except:
        etf_vol = None

    return daily_ret, dvol, gt_daily, fear_greed, etf_vol


def run_strategy(daily_ret, pos_series, label):
    """Run strategy with given position series (0 or 1)."""
    common = pos_series.index.intersection(daily_ret.index)
    pos = pos_series.reindex(common)
    ret = daily_ret.reindex(common)

    # Shift to avoid look-ahead
    pos_applied = pos.shift(1)
    pos_applied.iloc[0] = 1.0

    # Transaction costs
    cost = pos_applied.diff().abs() * TAKER_FEE
    cost.iloc[0] = 0

    strat_ret = pos_applied * ret - cost
    bnh_ret = ret

    # Metrics
    cum_s = (1 + strat_ret).cumprod()
    cum_b = (1 + bnh_ret).cumprod()

    total_s = cum_s.iloc[-1] - 1
    total_b = cum_b.iloc[-1] - 1

    sh_s = strat_ret.mean() / strat_ret.std() * np.sqrt(365) if strat_ret.std() > 0 else 0
    sh_b = bnh_ret.mean() / bnh_ret.std() * np.sqrt(365) if bnh_ret.std() > 0 else 0

    mdd_s = ((cum_s - cum_s.cummax()) / cum_s.cummax()).min()
    mdd_b = ((cum_b - cum_b.cummax()) / cum_b.cummax()).min()

    flat_pct = (pos_applied == 0).sum() / len(pos_applied) * 100

    # Shuffled test (2000 permutations)
    np.random.seed(42)
    ret_vals = bnh_ret.values
    pos_vals = pos_applied.values
    cost_vals = np.abs(np.diff(pos_vals, prepend=0)) * TAKER_FEE
    observed_mean = np.mean(pos_vals * ret_vals - cost_vals)

    shuffled_means = []
    for _ in range(5000):
        perm = np.random.permutation(len(ret_vals))
        s_ret = pos_vals * ret_vals[perm] - cost_vals
        shuffled_means.append(np.mean(s_ret))
    shuffled_means = np.array(shuffled_means)
    p_val = (shuffled_means >= observed_mean).mean()

    status = '✅ PASS' if p_val < 0.05 else '❌ FAIL'

    print(f"\n  [{label}]")
    print(f"    Period: {common[0].date()} ~ {common[-1].date()} ({len(common)} days)")
    print(f"    Strat: {total_s:+.1%} (Sharpe {sh_s:.2f}, MDD {mdd_s:.1%})")
    print(f"    B&H:   {total_b:+.1%} (Sharpe {sh_b:.2f}, MDD {mdd_b:.1%})")
    print(f"    Excess: {total_s - total_b:+.1%} | Flat {flat_pct:.0f}% of time")
    print(f"    p-value: {p_val:.4f} {status}")

    # Yearly breakdown
    for y in sorted(set(strat_ret.index.year)):
        m = strat_ret.index.year == y
        sr_y = strat_ret[m]
        br_y = bnh_ret[m]
        ret_s_y = (1 + sr_y).prod() - 1
        ret_b_y = (1 + br_y).prod() - 1
        shy = sr_y.mean() / sr_y.std() * np.sqrt(365) if sr_y.std() > 0 else 0
        flat_y = (pos_applied[m] == 0).sum() / len(pos_applied[m]) * 100
        marker = '  ←' if ret_s_y > ret_b_y else ''
        print(f"      {y}: Strat {ret_s_y:+7.1%} | B&H {ret_b_y:+7.1%} | "
              f"Excess {ret_s_y-ret_b_y:+7.1%} | Sh {shy:+.2f} | Flat {flat_y:.0f}%{marker}")

    return p_val, sh_s, total_s, mdd_s


def main():
    print("=" * 80)
    print("  FINAL STRATEGY: DVOL + ETF + Sentiment (Fixed Parameters)")
    print("  Position: 0 (flat) or 1 (long). NO shorts.")
    print("=" * 80)

    daily_ret, dvol, gt_daily, fear_greed, etf_vol = load_all_data()

    print(f"\n  Data available:")
    print(f"    BTC returns: {daily_ret.index[0].date()} ~ {daily_ret.index[-1].date()} ({len(daily_ret)}d)")
    print(f"    DVOL: {dvol.index[0].date()} ~ {dvol.index[-1].date()} ({len(dvol)}d)")
    print(f"    Google Trends: {gt_daily.index[0].date()} ~ {gt_daily.index[-1].date()} ({len(gt_daily)}d)")
    print(f"    F&G: {fear_greed.index[0].date()} ~ {fear_greed.index[-1].date()} ({len(fear_greed)}d)")
    if etf_vol is not None:
        print(f"    ETF vol: {etf_vol.index[0].date()} ~ {etf_vol.index[-1].date()} ({len(etf_vol)}d)")

    # ================================================================
    # Strategy A: DVOL level — long when DVOL > expanding median
    # ================================================================
    dvol_median = dvol.expanding(min_periods=30).median()
    pos_a = (dvol > dvol_median).astype(float)

    # ================================================================
    # Strategy B: DVOL momentum — long when DVOL 2d change > 0
    # ================================================================
    dvol_chg2 = dvol.pct_change(2)
    pos_b = (dvol_chg2 > 0).astype(float)

    # ================================================================
    # Strategy C: DVOL combined — long when EITHER high level OR rising
    # ================================================================
    pos_c = ((dvol > dvol_median) | (dvol_chg2 > 0)).astype(float)

    # ================================================================
    # Strategy D: DVOL + go flat when DVOL low AND falling
    # (inverse of C: flat only when DVOL < median AND falling)
    # Same as C actually. Let's make D = A AND B (both conditions)
    # ================================================================
    pos_d = ((dvol > dvol_median) & (dvol_chg2 > 0)).astype(float)

    # ================================================================
    # Strategy E: High IV quintile only (DVOL > Q60)
    # ================================================================
    dvol_q60 = dvol.expanding(min_periods=30).quantile(0.6)
    pos_e = (dvol > dvol_q60).astype(float)

    # ================================================================
    # Strategy F: DVOL level + Google Trends risk filter
    # Long when DVOL > median, BUT go flat if GT > expanding P80
    # ================================================================
    gt_aligned = gt_daily.reindex(dvol.index, method='ffill')
    gt_p80 = gt_aligned.expanding(min_periods=30).quantile(0.8)
    gt_fomo = gt_aligned > gt_p80
    pos_f = pos_a.copy()
    pos_f[gt_fomo] = 0.0

    # ================================================================
    # Strategy G: DVOL combined + Google Trends filter
    # ================================================================
    pos_g = pos_c.copy()
    gt_aligned_c = gt_daily.reindex(pos_c.index, method='ffill')
    gt_p80_c = gt_aligned_c.expanding(min_periods=30).quantile(0.8)
    pos_g[gt_aligned_c > gt_p80_c] = 0.0

    # ================================================================
    # Strategy H: F&G momentum + DVOL
    # Long when F&G > 50 (greed) AND DVOL > median
    # ================================================================
    fg_aligned = fear_greed.reindex(dvol.index, method='ffill')
    pos_h = ((dvol > dvol_median) & (fg_aligned > 50)).astype(float)

    # ================================================================
    # Strategy I: DVOL + F&G greed = long, DVOL low + F&G fear = flat
    # Long when either DVOL high or F&G > 60
    # ================================================================
    pos_i = ((dvol > dvol_median) | (fg_aligned > 60)).astype(float)

    # ================================================================
    # Strategy J: Conservative — flat only when ALL bearish
    # Flat when: DVOL < Q40 AND DVOL falling AND F&G < 40
    # ================================================================
    dvol_q40 = dvol.expanding(min_periods=30).quantile(0.4)
    bearish_all = (dvol < dvol_q40) & (dvol_chg2 < 0) & (fg_aligned < 40)
    pos_j = pd.Series(1.0, index=dvol.index)
    pos_j[bearish_all] = 0.0

    # ================================================================
    # Strategy K: Aggressive filter — flat when ANY 2 of 3 bearish
    # Bearish signals: DVOL < median, DVOL falling, F&G < 45
    # ================================================================
    b1 = (dvol < dvol_median).astype(int)
    b2 = (dvol_chg2 < 0).astype(int)
    b3 = (fg_aligned < 45).astype(int)
    bearish_count = b1 + b2 + b3
    pos_k = (bearish_count < 2).astype(float)

    # ================================================================
    # Strategy L: DVOL > median + NOT Google Trends FOMO + F&G > 40
    # Triple filter
    # ================================================================
    gt_al_l = gt_daily.reindex(dvol.index, method='ffill')
    gt_p80_l = gt_al_l.expanding(min_periods=30).quantile(0.8)
    fg_al_l = fear_greed.reindex(dvol.index, method='ffill')
    pos_l = ((dvol > dvol_median) & (gt_al_l <= gt_p80_l) & (fg_al_l > 40)).astype(float)

    # Run all
    strategies = [
        (pos_a, "A: DVOL > median → long"),
        (pos_b, "B: DVOL 2d rising → long"),
        (pos_c, "C: DVOL high OR rising → long"),
        (pos_d, "D: DVOL high AND rising → long"),
        (pos_e, "E: DVOL > Q60 → long"),
        (pos_f, "F: A + GT FOMO filter"),
        (pos_g, "G: C + GT FOMO filter"),
        (pos_h, "H: DVOL high + F&G > 50"),
        (pos_i, "I: DVOL high OR F&G > 60"),
        (pos_j, "J: Conservative (flat when ALL bearish)"),
        (pos_k, "K: Flat when 2/3 bearish"),
        (pos_l, "L: DVOL + GT filter + F&G > 40"),
    ]

    print(f"\n  Testing {len(strategies)} fixed-parameter variants...")
    print(f"  (5000 permutations each for shuffled test)")

    results = []
    for pos, label in strategies:
        p, sh, ret, mdd = run_strategy(daily_ret, pos, label)
        results.append((label, p, sh, ret, mdd))

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY (sorted by p-value)")
    print("=" * 80)
    results.sort(key=lambda x: x[1])
    for label, p, sh, ret, mdd in results:
        status = '✅' if p < 0.05 else '⚠️' if p < 0.10 else '❌'
        print(f"  {status} p={p:.4f} | Sh {sh:+.2f} | Ret {ret:+.1%} | MDD {mdd:.1%} | {label}")

    # If any pass, do OOS test
    best = results[0]
    if best[1] < 0.10:
        print(f"\n  Best strategy: {best[0]}")
        print(f"  → Next step: OOS validation on 2026-05-08 ~ 2026-05-25")


if __name__ == "__main__":
    main()
