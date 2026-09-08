"""
Fixed-Parameter Funding Rate Strategy (NO optimization)
Run: python -m rv_regime.fixed_funding

Core idea: WF optimization consistently overfits and picks wrong direction.
Solution: NO grid search. Use theoretically motivated fixed rules.

From p=0.0065 discovery:
  High funding = trend continuation = LONG
  Low funding = trend weakening = SHORT

Multiple fixed rule variants tested simultaneously:
  1. Simple sign: funding 7d sum > 0 → long, < 0 → short
  2. Simple sign: funding 14d sum > 0 → long, < 0 → short
  3. Simple sign: funding 21d sum > 0 → long, < 0 → short
  4-6. Same but with OI confirmation (short only if OI also declining)
  7-9. Stricter: short only if funding < -threshold (fixed)
"""
import numpy as np
import pandas as pd

from rv_regime.data_loader import load_and_prepare
from rv_regime.config import TAKER_FEE


def load_signals():
    _, daily_ret = load_and_prepare()
    tz = daily_ret.index.tz

    def align_dates(dates, tz):
        dt = pd.DatetimeIndex(pd.to_datetime(dates))
        return dt.normalize().tz_localize(tz)

    fr = pd.read_parquet('data/funding_rate_daily.parquet')
    fr_idx = align_dates(fr['date'], tz)
    funding = pd.Series(fr['funding_rate_mean'].values, index=fr_idx, name='funding_rate')

    oi_change = None
    try:
        mt = pd.read_parquet('data/metrics_daily.parquet')
        mt_idx = align_dates(mt['date'], tz)
        if 'sum_open_interest_value' in mt.columns:
            oi_val = pd.Series(mt['sum_open_interest_value'].values, index=mt_idx)
            oi_change = oi_val.pct_change()
    except Exception:
        pass

    fg = pd.read_parquet('data/fear_greed_daily.parquet')
    fg_idx = align_dates(fg['date'], tz)
    fear_greed = pd.Series(fg['fear_greed_value'].values, index=fg_idx)

    return daily_ret, funding, oi_change, fear_greed


def run_fixed(daily_ret, funding, oi_change, fear_greed,
              fw, use_oi, oi_window, use_fg, fg_low, short_mode='sign'):
    """
    Fixed-parameter strategy. No expanding percentile, no optimization.

    short_mode:
      'sign': short when funding sum < 0
      'strict': short when funding sum < -0.001 (stricter threshold)
    """
    fr_sum = funding.rolling(fw, min_periods=fw).sum().dropna()

    # Default: LONG
    pos = pd.Series(1.0, index=fr_sum.index)

    if short_mode == 'sign':
        short_cond = fr_sum < 0
    elif short_mode == 'strict':
        short_cond = fr_sum < -0.001
    elif short_mode == 'very_strict':
        short_cond = fr_sum < -0.003
    else:
        short_cond = fr_sum < 0

    # OI confirmation
    if use_oi and oi_change is not None:
        oi_smooth = oi_change.reindex(fr_sum.index).rolling(oi_window, min_periods=oi_window).mean()
        oi_declining = oi_smooth < 0
        short_cond = short_cond & oi_declining

    # F&G confirmation (extreme fear = actually contrarian BUY, so don't short)
    if use_fg:
        fg_aligned = fear_greed.reindex(fr_sum.index, method='ffill')
        # Don't short during extreme fear (contrarian buy signal)
        not_extreme_fear = fg_aligned > fg_low
        short_cond = short_cond & not_extreme_fear

    pos[short_cond] = -1.0

    # Shift to avoid look-ahead
    pos_applied = pos.shift(1)
    pos_applied.iloc[0] = 1.0

    # Align
    common = pos_applied.index.intersection(daily_ret.index)
    pf = pos_applied.reindex(common)
    ret = daily_ret.reindex(common)

    cost = pf.diff().abs() * TAKER_FEE
    cost.iloc[0] = 0

    strat_ret = pf * ret - cost
    return strat_ret, ret, pf


def evaluate(strat_ret, bnh_ret, pos, label):
    """Compute metrics and shuffled test."""
    cum_s = (1 + strat_ret).cumprod()
    cum_b = (1 + bnh_ret).cumprod()

    sh = strat_ret.mean() / strat_ret.std() * np.sqrt(365) if strat_ret.std() > 0 else 0
    mdd = ((cum_s - cum_s.cummax()) / cum_s.cummax()).min()
    total_ret = cum_s.iloc[-1] - 1
    bnh_total = cum_b.iloc[-1] - 1
    short_pct = (pos == -1).sum() / len(pos) * 100

    # Shuffled test
    np.random.seed(42)
    ret_vals = bnh_ret.values
    pos_vals = pos.values
    cost_vals = np.abs(np.diff(pos_vals, prepend=0)) * TAKER_FEE
    shuffled = []
    for _ in range(2000):
        perm = np.random.permutation(len(ret_vals))
        s_ret = pos_vals * ret_vals[perm] - cost_vals
        s_std = np.std(s_ret)
        if s_std > 0:
            shuffled.append(np.mean(s_ret) / s_std * np.sqrt(365))
    shuffled = np.array(shuffled)
    p_val = (shuffled >= sh).mean()

    status = 'PASS ✓' if p_val < 0.05 else 'FAIL'

    print(
        f"  {label:40s} | Strat {total_ret:+7.1%} | B&H {bnh_total:+7.1%} | "
        f"Excess {total_ret-bnh_total:+7.1%} | Sharpe {sh:+.2f} | MDD {mdd:+.1%} | "
        f"Short {short_pct:.0f}% | p={p_val:.4f} {status}",
        flush=True,
    )

    # Yearly if promising
    if p_val < 0.10:
        for y in sorted(set(strat_ret.index.year)):
            m = strat_ret.index.year == y
            sr_y = strat_ret[m]
            br_y = bnh_ret[m]
            ret_s = (1 + sr_y).prod() - 1
            ret_b = (1 + br_y).prod() - 1
            shy = sr_y.mean() / sr_y.std() * np.sqrt(365) if sr_y.std() > 0 else 0
            short_y = (pos[m] == -1).sum() / len(pos[m]) * 100
            print(
                f"    {y}: Strat {ret_s:+7.1%} | B&H {ret_b:+7.1%} "
                f"| Excess {ret_s-ret_b:+7.1%} | Sharpe {shy:+.2f} | Short {short_y:.0f}%",
                flush=True,
            )

    return p_val, sh, total_ret


def main():
    print("=" * 80, flush=True)
    print("  Fixed-Parameter Funding Rate Strategy (NO Optimization)", flush=True)
    print("=" * 80, flush=True)

    daily_ret, funding, oi_change, fear_greed = load_signals()

    print(f"  Period: {daily_ret.index[0].date()} ~ {daily_ret.index[-1].date()}", flush=True)
    print(f"  Funding: {len(funding)} days", flush=True)
    print(flush=True)

    configs = [
        # (label, fw, use_oi, oi_window, use_fg, fg_low, short_mode)
        # Simple sign-based
        ("FR 7d sign",              7,  False, 7,  False, 25, 'sign'),
        ("FR 14d sign",             14, False, 7,  False, 25, 'sign'),
        ("FR 21d sign",             21, False, 7,  False, 25, 'sign'),

        # With OI confirmation
        ("FR 7d + OI 7d",           7,  True,  7,  False, 25, 'sign'),
        ("FR 14d + OI 7d",          14, True,  7,  False, 25, 'sign'),
        ("FR 21d + OI 7d",          21, True,  7,  False, 25, 'sign'),
        ("FR 7d + OI 14d",          7,  True,  14, False, 25, 'sign'),
        ("FR 14d + OI 14d",         14, True,  14, False, 25, 'sign'),
        ("FR 21d + OI 14d",         21, True,  14, False, 25, 'sign'),

        # Strict threshold
        ("FR 7d strict",            7,  False, 7,  False, 25, 'strict'),
        ("FR 14d strict",           14, False, 7,  False, 25, 'strict'),
        ("FR 21d strict",           21, False, 7,  False, 25, 'strict'),

        # Very strict
        ("FR 7d very_strict",       7,  False, 7,  False, 25, 'very_strict'),
        ("FR 14d very_strict",      14, False, 7,  False, 25, 'very_strict'),
        ("FR 21d very_strict",      21, False, 7,  False, 25, 'very_strict'),

        # Strict + OI
        ("FR 7d strict + OI 7d",    7,  True,  7,  False, 25, 'strict'),
        ("FR 14d strict + OI 7d",   14, True,  7,  False, 25, 'strict'),
        ("FR 21d strict + OI 7d",   21, True,  7,  False, 25, 'strict'),

        # With F&G filter (don't short during extreme fear)
        ("FR 7d + OI 7d + FG25",    7,  True,  7,  True,  25, 'sign'),
        ("FR 14d + OI 7d + FG25",   14, True,  7,  True,  25, 'sign'),
        ("FR 7d strict + OI + FG",   7, True,  7,  True,  25, 'strict'),
        ("FR 14d strict + OI + FG",  14, True,  7,  True,  25, 'strict'),

        # Very strict + OI + FG
        ("FR 7d vstrict+OI+FG",     7,  True,  7,  True,  25, 'very_strict'),
        ("FR 14d vstrict+OI+FG",    14, True,  7,  True,  25, 'very_strict'),
        ("FR 21d vstrict+OI+FG",    21, True,  7,  True,  25, 'very_strict'),
    ]

    print(f"  Testing {len(configs)} fixed-parameter variants...", flush=True)
    print(flush=True)

    best_p = 1.0
    best_label = ""

    for label, fw, use_oi, oi_window, use_fg, fg_low, short_mode in configs:
        sr, br, pf = run_fixed(daily_ret, funding, oi_change, fear_greed,
                               fw, use_oi, oi_window, use_fg, fg_low, short_mode)
        p, sh, ret = evaluate(sr, br, pf, label)
        if p < best_p:
            best_p = p
            best_label = label

    print(flush=True)
    print(f"  Best p-value: {best_p:.4f} ({best_label})", flush=True)


if __name__ == "__main__":
    main()
