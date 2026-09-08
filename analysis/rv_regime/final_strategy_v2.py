"""
Final Strategy v2: Look-ahead bias 수정 + OOS 검증

핵심 수정사항:
  - shift(1) → shift(2): DVOL 클로즈 시점 (T 20:00 ET)이 
    트레이드 데이 T+1 시작 (T 17:00 ET) 이후이므로,
    안전하게 2일 shift 필요.
  - expanding median/quantile도 동일하게 T-2 시점까지만 사용
  - OOS: 2026-05-08 ~ 2026-05-25 별도 검증
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from rv_regime.data_loader import load_and_prepare
from rv_regime.config import TAKER_FEE, IS_END


def load_all_data():
    _, daily_ret = load_and_prepare()
    daily_ret = daily_ret[~daily_ret.index.duplicated(keep='first')]
    tz = daily_ret.index.tz

    dvol_raw = pd.read_parquet('data/dvol_daily.parquet')
    dvol = dvol_raw['dvol'].copy()
    dvol.index = pd.DatetimeIndex(dvol.index).normalize().tz_localize(tz)
    dvol = dvol[~dvol.index.duplicated(keep='first')]

    fg = pd.read_parquet('data/fear_greed_daily.parquet')
    fg_idx = pd.DatetimeIndex(pd.to_datetime(fg['date'])).normalize().tz_localize(tz)
    fear_greed = pd.Series(fg['fear_greed_value'].values, index=fg_idx)
    fear_greed = fear_greed[~fear_greed.index.duplicated(keep='first')]

    return daily_ret, dvol, fear_greed


def generate_signals(dvol, fear_greed):
    """Generate all signal variants with PROPER lag (shift=2 for DVOL)."""
    
    # DVOL signals - 이미 시그널 자체에 shift(2) 적용
    # 이렇게 하면 run_strategy에서 추가 shift 불필요
    dvol_median = dvol.expanding(min_periods=30).median()
    dvol_q60 = dvol.expanding(min_periods=30).quantile(0.6)
    dvol_q40 = dvol.expanding(min_periods=30).quantile(0.4)
    dvol_chg2 = dvol.pct_change(2)
    
    # 모든 DVOL 기반 시그널을 2일 shift (look-ahead 방지)
    dvol_high = (dvol > dvol_median).shift(2)
    dvol_above_q60 = (dvol > dvol_q60).shift(2)
    dvol_below_q40 = (dvol < dvol_q40).shift(2)
    dvol_rising = (dvol_chg2 > 0).shift(2)
    dvol_falling = (dvol_chg2 < 0).shift(2)
    
    # F&G - 발표 시점: 매일 00:00 UTC 전후
    # Trade day T 시작 (T-1) 17:00 ET 기준, F&G[T-1]은 
    # (T-1) 00:00 UTC = (T-2) 19:00 ET에 발표 → 사용 가능
    # 하지만 안전하게 shift(1) 적용
    fg_aligned = fear_greed.reindex(dvol.index, method='ffill')
    fg_bullish = (fg_aligned > 50).shift(1)
    fg_greed = (fg_aligned > 60).shift(1)
    fg_fear = (fg_aligned < 40).shift(1)
    
    strategies = {}
    
    # A: DVOL > expanding median
    strategies['A: DVOL > median'] = dvol_high.astype(float)
    
    # B: DVOL 2d rising
    strategies['B: DVOL 2d rising'] = dvol_rising.astype(float)
    
    # C: DVOL high OR rising
    strategies['C: DVOL high OR rising'] = (dvol_high | dvol_rising).astype(float)
    
    # D: DVOL high AND rising (best in v1)
    strategies['D: DVOL high AND rising'] = (dvol_high & dvol_rising).astype(float)
    
    # E: DVOL > Q60
    strategies['E: DVOL > Q60'] = dvol_above_q60.astype(float)
    
    # H: DVOL high + F&G > 50
    strategies['H: DVOL high + F&G>50'] = (dvol_high & fg_bullish).astype(float)
    
    # K: Flat when 2/3 bearish
    b1 = (dvol_high.fillna(False) == False).astype(int)
    b2 = dvol_falling.fillna(False).astype(int)
    b3 = fg_fear.fillna(False).astype(int)
    bearish_count = b1 + b2 + b3
    strategies['K: Flat when 2/3 bearish'] = (bearish_count < 2).astype(float)
    
    return strategies


def run_strategy(daily_ret, pos_series, label, period_label="Full"):
    """Run strategy - NO additional shift needed (already in signal)."""
    common = pos_series.dropna().index.intersection(daily_ret.index)
    if len(common) < 30:
        print(f"  [{label}] ({period_label}) — insufficient data ({len(common)} days)")
        return None, None, None, None
    
    pos = pos_series.reindex(common)
    ret = daily_ret.reindex(common)

    # NO shift here - shift already applied in generate_signals
    cost = pos.diff().abs() * TAKER_FEE
    cost.iloc[0] = 0

    strat_ret = pos * ret - cost
    bnh_ret = ret

    cum_s = (1 + strat_ret).cumprod()
    cum_b = (1 + bnh_ret).cumprod()

    total_s = cum_s.iloc[-1] - 1
    total_b = cum_b.iloc[-1] - 1

    sh_s = strat_ret.mean() / strat_ret.std() * np.sqrt(365) if strat_ret.std() > 0 else 0
    sh_b = bnh_ret.mean() / bnh_ret.std() * np.sqrt(365) if bnh_ret.std() > 0 else 0

    mdd_s = ((cum_s - cum_s.cummax()) / cum_s.cummax()).min()
    mdd_b = ((cum_b - cum_b.cummax()) / cum_b.cummax()).min()

    flat_pct = (pos == 0).sum() / len(pos) * 100

    # Shuffled test
    np.random.seed(42)
    ret_vals = bnh_ret.values
    pos_vals = pos.values
    cost_vals = np.abs(np.diff(pos_vals, prepend=0)) * TAKER_FEE
    observed_mean = np.mean(pos_vals * ret_vals - cost_vals)

    shuffled_means = []
    for _ in range(5000):
        perm = np.random.permutation(len(ret_vals))
        s_ret = pos_vals * ret_vals[perm] - cost_vals
        shuffled_means.append(np.mean(s_ret))
    shuffled_means = np.array(shuffled_means)
    p_val = (shuffled_means >= observed_mean).mean()

    status = '✅ PASS' if p_val < 0.05 else '⚠️' if p_val < 0.10 else '❌ FAIL'

    print(f"\n  [{label}] ({period_label})")
    print(f"    Period: {common[0].date()} ~ {common[-1].date()} ({len(common)} days)")
    print(f"    Strat: {total_s:+.1%} (Sharpe {sh_s:.2f}, MDD {mdd_s:.1%})")
    print(f"    B&H:   {total_b:+.1%} (Sharpe {sh_b:.2f}, MDD {mdd_b:.1%})")
    print(f"    Excess: {total_s - total_b:+.1%} | Flat {flat_pct:.0f}% of time")
    print(f"    p-value: {p_val:.4f} {status}")

    # Yearly
    for y in sorted(set(strat_ret.index.year)):
        m = strat_ret.index.year == y
        sr_y = strat_ret[m]
        br_y = bnh_ret[m]
        ret_s_y = (1 + sr_y).prod() - 1
        ret_b_y = (1 + br_y).prod() - 1
        shy = sr_y.mean() / sr_y.std() * np.sqrt(365) if sr_y.std() > 0 else 0
        flat_y = (pos[m] == 0).sum() / len(pos[m]) * 100
        marker = '  ←' if ret_s_y > ret_b_y else ''
        print(f"      {y}: Strat {ret_s_y:+7.1%} | B&H {ret_b_y:+7.1%} | "
              f"Excess {ret_s_y-ret_b_y:+7.1%} | Sh {shy:+.2f} | Flat {flat_y:.0f}%{marker}")

    return p_val, sh_s, total_s, mdd_s


def main():
    print("=" * 80)
    print("  FINAL STRATEGY v2: Look-Ahead Bias 수정")
    print("  DVOL shift(2), F&G shift(1) — 시간축 완전 분리")
    print("=" * 80)

    daily_ret, dvol, fear_greed = load_all_data()
    
    # IS/OOS split
    is_end = pd.Timestamp(IS_END).tz_localize(daily_ret.index.tz)
    
    print(f"\n  IS period: ~ {IS_END}")
    print(f"  OOS period: {IS_END} ~ {daily_ret.index[-1].date()}")
    print(f"\n  === LOOK-AHEAD BIAS CHECK ===")
    print(f"  DVOL signal: shift(2) applied — 시그널 T-2 기준")
    print(f"    DVOL[T-2] closes at (T-2) 20:00 ET")
    print(f"    Trade day T starts at (T-1) 17:00 ET")
    print(f"    → 21시간 gap. ✅ 안전")
    print(f"  F&G signal: shift(1) applied — F&G[T-1] 기준")
    print(f"    F&G[T-1] published ~(T-2) 19:00 ET")  
    print(f"    Trade day T starts at (T-1) 17:00 ET")
    print(f"    → ~22시간 gap. ✅ 안전")
    print(f"  Expanding median/quantile: 시그널 shift에 포함. ✅")
    print(f"  Transaction cost: shift된 포지션 기준 diff. ✅")

    strategies = generate_signals(dvol, fear_greed)

    # === IN-SAMPLE ===
    print(f"\n{'='*80}")
    print(f"  IN-SAMPLE RESULTS (shift=2, bias-free)")
    print(f"{'='*80}")

    is_results = []
    for label, pos in strategies.items():
        pos_is = pos[pos.index <= is_end]
        ret_is = daily_ret[daily_ret.index <= is_end]
        p, sh, ret, mdd = run_strategy(ret_is, pos_is, label, "IS")
        if p is not None:
            is_results.append((label, p, sh, ret, mdd))

    # Summary IS
    print(f"\n  --- IS Summary (sorted by p-value) ---")
    is_results.sort(key=lambda x: x[1])
    for label, p, sh, ret, mdd in is_results:
        status = '✅' if p < 0.05 else '⚠️' if p < 0.10 else '❌'
        print(f"  {status} p={p:.4f} | Sh {sh:+.2f} | Ret {ret:+.1%} | MDD {mdd:.1%} | {label}")

    # === OUT-OF-SAMPLE ===
    print(f"\n{'='*80}")
    print(f"  OUT-OF-SAMPLE RESULTS (2026-05-08 ~ 2026-05-25)")
    print(f"{'='*80}")
    
    # For OOS, we use signals generated from ALL data (expanding, so no future leak)
    # but only evaluate on OOS period
    for label, pos in strategies.items():
        pos_oos = pos[pos.index > is_end]
        ret_oos = daily_ret[daily_ret.index > is_end]
        
        common = pos_oos.dropna().index.intersection(ret_oos.index)
        if len(common) < 5:
            print(f"\n  [{label}] (OOS) — insufficient data ({len(common)} days)")
            continue
            
        p_oos = pos_oos.reindex(common)
        r_oos = ret_oos.reindex(common)
        
        cost = p_oos.diff().abs() * TAKER_FEE
        cost.iloc[0] = 0
        
        strat_ret = p_oos * r_oos - cost
        bnh_ret = r_oos
        
        cum_s = (1 + strat_ret).cumprod()
        cum_b = (1 + bnh_ret).cumprod()
        
        total_s = cum_s.iloc[-1] - 1
        total_b = cum_b.iloc[-1] - 1
        
        flat_pct = (p_oos == 0).sum() / len(p_oos) * 100
        
        # Day-by-day OOS
        print(f"\n  [{label}] (OOS)")
        print(f"    Period: {common[0].date()} ~ {common[-1].date()} ({len(common)} days)")
        print(f"    Strat: {total_s:+.1%} | B&H: {total_b:+.1%} | Excess: {total_s-total_b:+.1%}")
        print(f"    Flat: {flat_pct:.0f}% | Position: {p_oos.value_counts().to_dict()}")
        
        # Day by day
        print(f"    Day-by-day:")
        for dt in common:
            pos_val = p_oos[dt]
            ret_val = r_oos[dt]
            s_ret = pos_val * ret_val
            print(f"      {dt.date()}: pos={pos_val:.0f} | BTC={ret_val:+.2%} | strat={s_ret:+.2%}")

    # === ADDITIONAL ROBUSTNESS CHECKS ===
    print(f"\n{'='*80}")
    print(f"  ROBUSTNESS: shift(1) vs shift(2) comparison")
    print(f"{'='*80}")
    
    # Re-generate with shift(1) for comparison (biased version)
    dvol_median = dvol.expanding(min_periods=30).median()
    dvol_chg2 = dvol.pct_change(2)
    
    for shift_val in [1, 2, 3]:
        dvol_high = (dvol > dvol_median).shift(shift_val)
        dvol_rising = (dvol_chg2 > 0).shift(shift_val)
        pos_test = (dvol_high & dvol_rising).astype(float)
        
        common = pos_test.dropna().index.intersection(daily_ret.index)
        pos_t = pos_test.reindex(common)
        ret_t = daily_ret.reindex(common)
        cost = pos_t.diff().abs() * TAKER_FEE
        cost.iloc[0] = 0
        strat = pos_t * ret_t - cost
        
        sh = strat.mean() / strat.std() * np.sqrt(365) if strat.std() > 0 else 0
        cum = (1 + strat).cumprod()
        total = cum.iloc[-1] - 1
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()
        flat = (pos_t == 0).sum() / len(pos_t) * 100
        
        # Quick shuffled test
        np.random.seed(42)
        obs = np.mean(pos_t.values * ret_t.values - np.abs(np.diff(pos_t.values, prepend=0)) * TAKER_FEE)
        shuf = []
        for _ in range(5000):
            perm = np.random.permutation(len(ret_t))
            sr = pos_t.values * ret_t.values[perm] - np.abs(np.diff(pos_t.values, prepend=0)) * TAKER_FEE
            shuf.append(np.mean(sr))
        p = (np.array(shuf) >= obs).mean()
        
        status = '✅' if p < 0.05 else '❌'
        print(f"  Strategy D (shift={shift_val}): Sharpe {sh:.2f} | Ret {total:+.1%} | "
              f"MDD {mdd:.1%} | Flat {flat:.0f}% | p={p:.4f} {status}")


if __name__ == "__main__":
    main()
