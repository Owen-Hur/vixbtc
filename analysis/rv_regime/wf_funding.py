"""
Walk-Forward: Funding Rate Alpha Strategy
Run: python -m rv_regime.wf_funding

Core idea:
  Extreme positive funding = crowded long → mean revert → SHORT
  Extreme negative funding = crowded short → mean revert → LONG
  Default: LONG (+1) — BTC has structural upward bias

  Position is always +1 or -1 (max exposure, daily trading).

Signals:
  1. Funding Rate (primary): rolling sum of 8h funding rates
  2. OI change (confirmation): rapid OI increase + extreme funding = stronger signal
  3. Taker L/S ratio (confirmation): extreme taker buy/sell ratio

Parameters to optimize:
  - fw: funding rate rolling window (days)
  - qh: upper percentile for short signal
  - ql: lower percentile for long signal (overrides default)
  - signal_mode: 'funding_only', 'funding_oi', 'composite'
"""
import numpy as np
import pandas as pd
from itertools import product

from rv_regime.data_loader import load_and_prepare
from rv_regime.config import TAKER_FEE

WF_ROUNDS = [
    {"train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-03-31"},
    {"train_end": "2021-03-31", "test_start": "2021-04-01", "test_end": "2021-06-30"},
    {"train_end": "2021-06-30", "test_start": "2021-07-01", "test_end": "2021-09-30"},
    {"train_end": "2021-09-30", "test_start": "2021-10-01", "test_end": "2021-12-31"},
    {"train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-03-31"},
    {"train_end": "2022-03-31", "test_start": "2022-04-01", "test_end": "2022-06-30"},
    {"train_end": "2022-06-30", "test_start": "2022-07-01", "test_end": "2022-09-30"},
    {"train_end": "2022-09-30", "test_start": "2022-10-01", "test_end": "2022-12-31"},
    {"train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-03-31"},
    {"train_end": "2023-03-31", "test_start": "2023-04-01", "test_end": "2023-06-30"},
    {"train_end": "2023-06-30", "test_start": "2023-07-01", "test_end": "2023-09-30"},
    {"train_end": "2023-09-30", "test_start": "2023-10-01", "test_end": "2023-12-31"},
    {"train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-03-31"},
    {"train_end": "2024-03-31", "test_start": "2024-04-01", "test_end": "2024-06-30"},
    {"train_end": "2024-06-30", "test_start": "2024-07-01", "test_end": "2024-09-30"},
    {"train_end": "2024-09-30", "test_start": "2024-10-01", "test_end": "2024-12-31"},
    {"train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-03-31"},
    {"train_end": "2025-03-31", "test_start": "2025-04-01", "test_end": "2025-06-30"},
    {"train_end": "2025-06-30", "test_start": "2025-07-01", "test_end": "2025-09-30"},
    {"train_end": "2025-09-30", "test_start": "2025-10-01", "test_end": "2025-12-31"},
    {"train_end": "2025-12-31", "test_start": "2026-01-01", "test_end": "2026-03-31"},
    {"train_end": "2026-03-31", "test_start": "2026-04-01", "test_end": "2026-05-07"},
]

# Keep parameter space small to avoid overfitting
SEARCH_FW = [3, 5, 7, 14, 21]        # funding rolling window
SEARCH_QH = [0.65, 0.70, 0.75, 0.80, 0.85]  # upper percentile → short (less conservative)
SEARCH_QL = [0.10, 0.15, 0.20, 0.25, 0.30]  # lower percentile → long


def load_signals():
    """Load and align all signal data with daily returns."""
    # Daily returns from 1m bars
    _, daily_ret = load_and_prepare()

    # Funding rate
    fr = pd.read_parquet('data/funding_rate_daily.parquet')
    fr['date'] = pd.to_datetime(fr['date'])

    # Metrics (OI, taker ratio) — starts 2020-09
    try:
        mt = pd.read_parquet('data/metrics_daily.parquet')
        mt['date'] = pd.to_datetime(mt['date'])
    except Exception:
        mt = None

    # Align funding rate to daily_ret index
    # daily_ret has tz-aware index (America/New_York, midnight-aligned)
    # fr has naive date timestamps — we need to match them
    tz = daily_ret.index.tz

    def align_dates(dates, tz):
        """Convert naive dates to tz-aware midnight timestamps matching daily_ret."""
        dt = pd.DatetimeIndex(pd.to_datetime(dates))
        return dt.normalize().tz_localize(tz)

    fr_idx = align_dates(fr['date'], tz)
    funding = pd.Series(
        fr['funding_rate_mean'].values,
        index=fr_idx,
        name='funding_rate'
    )

    # OI change (pct)
    oi_change = None
    taker_ratio = None
    if mt is not None and len(mt) > 0:
        mt_idx = align_dates(mt['date'], tz)
        if 'sum_open_interest_value' in mt.columns:
            oi_val = pd.Series(
                mt['sum_open_interest_value'].values,
                index=mt_idx,
                name='oi_value'
            )
            oi_change = oi_val.pct_change()
            oi_change.name = 'oi_change'

        if 'sum_taker_long_short_vol_ratio' in mt.columns:
            taker_ratio = pd.Series(
                mt['sum_taker_long_short_vol_ratio'].values,
                index=mt_idx,
                name='taker_ls_ratio'
            )

    return daily_ret, funding, oi_change, taker_ratio


def expanding_percentile(series, q):
    """Expanding window percentile (bias-free, vectorized where possible)."""
    vals = series.values
    n = len(vals)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.percentile(vals[:i + 1], q * 100)
    return pd.Series(out, index=series.index)


def run_funding_strategy(daily_ret, funding, oi_change, taker_ratio,
                         fw, qh, ql, mode='funding_only'):
    """
    Run funding rate based strategy.

    Modes:
      funding_only: just funding rate signal
      funding_oi: funding + OI change confirmation
      composite: funding + OI + taker ratio score
    """
    # Rolling sum of funding rate (cumulative cost of holding long)
    fr_smooth = funding.rolling(fw, min_periods=fw).sum().dropna()

    # Expanding thresholds
    th_upper = expanding_percentile(fr_smooth, qh)
    th_lower = expanding_percentile(fr_smooth, ql)

    # Base signal from funding rate
    # High funding → longs are paying → crowded long → SHORT
    # Low funding → shorts are paying → crowded short → LONG
    pos = pd.Series(1.0, index=fr_smooth.index)  # default long
    pos[fr_smooth > th_upper] = -1.0  # extreme positive funding → short
    # extreme negative funding → long (confirms default, but still important
    # for returning from short)

    if mode == 'funding_oi' and oi_change is not None:
        # Only short when funding is extreme AND OI is rising (new positions)
        oi_aligned = oi_change.reindex(fr_smooth.index)
        oi_rising = oi_aligned > 0
        # Relax short signal: require both extreme funding AND rising OI
        short_mask = (fr_smooth > th_upper) & oi_rising
        pos = pd.Series(1.0, index=fr_smooth.index)
        pos[short_mask] = -1.0

    elif mode == 'composite' and oi_change is not None and taker_ratio is not None:
        # Score-based: each signal contributes
        oi_aligned = oi_change.reindex(fr_smooth.index).fillna(0)
        tr_aligned = taker_ratio.reindex(fr_smooth.index)

        # Funding score: > upper → -1, < lower → +1, else 0
        f_score = pd.Series(0.0, index=fr_smooth.index)
        f_score[fr_smooth > th_upper] = -1.0
        f_score[fr_smooth < th_lower] = 1.0

        # OI score: high OI change amplifies funding signal direction
        oi_th_up = expanding_percentile(oi_aligned.dropna(), 0.80)
        oi_th_up = oi_th_up.reindex(fr_smooth.index).ffill()
        oi_score = pd.Series(0.0, index=fr_smooth.index)
        # Rising OI + positive funding → more crowded → bearish
        oi_score[(oi_aligned > oi_th_up) & (fr_smooth > 0)] = -0.5
        # Rising OI + negative funding → shorts adding → bearish for shorts
        oi_score[(oi_aligned > oi_th_up) & (fr_smooth < 0)] = 0.5

        # Taker ratio score
        if tr_aligned.notna().sum() > 30:
            tr_th_up = expanding_percentile(tr_aligned.dropna(), 0.85)
            tr_th_lo = expanding_percentile(tr_aligned.dropna(), 0.15)
            tr_th_up = tr_th_up.reindex(fr_smooth.index).ffill()
            tr_th_lo = tr_th_lo.reindex(fr_smooth.index).ffill()
            tr_score = pd.Series(0.0, index=fr_smooth.index)
            tr_score[tr_aligned > tr_th_up] = -0.5  # extreme taker buying → overbought
            tr_score[tr_aligned < tr_th_lo] = 0.5   # extreme taker selling → oversold
        else:
            tr_score = pd.Series(0.0, index=fr_smooth.index)

        total_score = f_score + oi_score + tr_score
        pos = pd.Series(1.0, index=fr_smooth.index)  # default long
        pos[total_score <= -1.0] = -1.0  # enough bearish signals → short

    # Shift to avoid look-ahead
    pos_applied = pos.shift(1)
    pos_applied.iloc[0] = 1.0  # start long

    # Align with returns
    common = pos_applied.index.intersection(daily_ret.index)
    pos_final = pos_applied.reindex(common)
    ret = daily_ret.reindex(common)

    # Costs
    cost = pos_final.diff().abs() * TAKER_FEE
    cost.iloc[0] = 0  # already in position

    strat_ret = pos_final * ret - cost
    return strat_ret, ret, pos_final


def select_best(daily_ret, funding, oi_change, taker_ratio, train_end_ts):
    """Grid search: maximize Information Ratio (excess return Sharpe over B&H).

    Also requires minimum 5% short allocation to avoid degenerating to B&H.
    """
    modes = ['funding_only']
    if oi_change is not None:
        modes.append('funding_oi')
    if oi_change is not None and taker_ratio is not None:
        modes.append('composite')

    combos = [
        (fw, qh, ql, m)
        for fw, qh, ql, m in product(SEARCH_FW, SEARCH_QH, SEARCH_QL, modes)
    ]

    best_ir = -999
    best_params = None
    best_raw_sharpe = 0

    for fw, qh, ql, mode in combos:
        try:
            sr, br, pos = run_funding_strategy(
                daily_ret, funding, oi_change, taker_ratio,
                fw, qh, ql, mode
            )
            s = sr[sr.index <= train_end_ts]
            b = br.reindex(s.index)
            p = pos.reindex(s.index)

            if len(s) < 30:
                continue

            # Require minimum short allocation (avoid B&H degeneration)
            short_pct = (p == -1).sum() / len(p)
            if short_pct < 0.05:
                continue

            # Information Ratio: Sharpe of excess returns
            excess = s - b
            ir = excess.mean() / excess.std() * np.sqrt(365) if excess.std() > 0 else 0

            # Also compute raw sharpe for reporting
            raw_sh = s.mean() / s.std() * np.sqrt(365) if s.std() > 0 else 0

            if ir > best_ir:
                best_ir = ir
                best_params = (fw, qh, ql, mode)
                best_raw_sharpe = raw_sh
        except Exception:
            continue

    return best_params, best_raw_sharpe


def main():
    print("=" * 80)
    print("  Walk-Forward: Funding Rate Alpha Strategy")
    print("=" * 80)

    daily_ret, funding, oi_change, taker_ratio = load_signals()
    tz = daily_ret.index.tz

    print(f"  Funding: {len(funding)} days ({funding.index[0].date()} ~ {funding.index[-1].date()})")
    if oi_change is not None:
        print(f"  OI change: {oi_change.notna().sum()} days")
    if taker_ratio is not None:
        print(f"  Taker ratio: {taker_ratio.notna().sum()} days")

    n_modes = 1 + (1 if oi_change is not None else 0) + (1 if taker_ratio is not None else 0)
    n_combos = len(SEARCH_FW) * len(SEARCH_QH) * len(SEARCH_QL) * n_modes
    print(f"  Combos per round: {n_combos}")
    print()

    all_strat = []
    all_bnh = []
    all_pos = []
    round_info = []

    for ri, r in enumerate(WF_ROUNDS):
        train_end_ts = pd.Timestamp(r["train_end"]).tz_localize(tz)
        test_start_ts = pd.Timestamp(r["test_start"]).tz_localize(tz)
        test_end_ts = pd.Timestamp(r["test_end"]).tz_localize(tz)

        params, train_sh = select_best(
            daily_ret, funding, oi_change, taker_ratio, train_end_ts
        )
        if params is None:
            print(f"  R{ri+1:>2}: no valid params")
            continue

        fw, qh, ql, mode = params
        sr, br, pf = run_funding_strategy(
            daily_ret, funding, oi_change, taker_ratio,
            fw, qh, ql, mode
        )

        mask = (sr.index >= test_start_ts) & (sr.index <= test_end_ts)
        test_sr = sr[mask]
        test_br = br[mask]
        test_pf = pf[mask]

        if len(test_sr) == 0:
            print(f"  R{ri+1:>2}: no test data")
            continue

        test_sharpe = (
            test_sr.mean() / test_sr.std() * np.sqrt(365)
            if test_sr.std() > 0 else 0
        )
        test_ret = (1 + test_sr).cumprod().iloc[-1] - 1
        bnh_ret = (1 + test_br).cumprod().iloc[-1] - 1
        short_pct = (test_pf == -1).sum() / len(test_pf) * 100

        info = {
            "round": ri + 1,
            "test_period": f"{r['test_start']}~{r['test_end']}",
            "fw": fw, "qh": qh, "ql": ql, "mode": mode,
            "train_sharpe": train_sh, "test_sharpe": test_sharpe,
            "test_ret": test_ret, "bnh_ret": bnh_ret,
            "short_pct": short_pct, "test_days": len(test_sr),
        }
        round_info.append(info)

        mode_s = {'funding_only': 'FR', 'funding_oi': 'FR+OI', 'composite': 'COMP'}
        print(
            f"  R{ri+1:>2}: {r['test_start']}~{r['test_end']} | "
            f"fw={fw},qh={qh},ql={ql},{mode_s.get(mode,'?')} | "
            f"Train {train_sh:.2f} -> Test {test_sharpe:+.2f} | "
            f"Strat {test_ret:+.1%} B&H {bnh_ret:+.1%} | Short {short_pct:.0f}%"
        )

        all_strat.append(test_sr)
        all_bnh.append(test_br)
        all_pos.append(test_pf)

    if not all_strat:
        print("\nNo valid rounds.")
        return

    wf_strat = pd.concat(all_strat)
    wf_bnh = pd.concat(all_bnh)
    wf_pos = pd.concat(all_pos)

    cum_s = (1 + wf_strat).cumprod()
    cum_b = (1 + wf_bnh).cumprod()
    sh = wf_strat.mean() / wf_strat.std() * np.sqrt(365) if wf_strat.std() > 0 else 0
    mdd = ((cum_s - cum_s.cummax()) / cum_s.cummax()).min()

    test_sharpes = [r["test_sharpe"] for r in round_info]
    mode_counts = {}
    for r in round_info:
        m = r["mode"]
        mode_counts[m] = mode_counts.get(m, 0) + 1

    print(f"\n{'='*80}")
    print(f"  WF Summary (Funding Rate Strategy)")
    print(f"{'='*80}")
    print(
        f"  Period: {wf_strat.index[0].strftime('%Y-%m-%d')} ~ "
        f"{wf_strat.index[-1].strftime('%Y-%m-%d')} ({len(wf_strat)} days)"
    )
    print(f"  Strategy: {cum_s.iloc[-1]-1:>+.1%} (Sharpe {sh:.2f}, MDD {mdd:+.1%})")
    print(f"  B&H:      {cum_b.iloc[-1]-1:>+.1%}")
    print(f"  Excess:   {cum_s.iloc[-1]-cum_b.iloc[-1]:>+.1%}")
    print(
        f"  Rounds Sharpe > 0: "
        f"{sum(1 for s in test_sharpes if s > 0)}/{len(test_sharpes)}"
    )
    print(f"  Avg Test Sharpe: {np.mean(test_sharpes):.2f}")
    print(f"  Median Test Sharpe: {np.median(test_sharpes):.2f}")
    print(f"  Mode selection: {mode_counts}")

    # Shuffled test
    print(f"\n  Shuffled test (2000 iters)...", end="", flush=True)
    np.random.seed(42)
    ret_vals = wf_bnh.values
    pos_vals = wf_pos.values
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
    print(f" p={p_val:.4f} {'PASS' if p_val < 0.05 else 'FAIL'}")

    # Yearly breakdown
    print(f"\n{'='*80}")
    print(f"  WF Yearly Breakdown")
    print(f"{'='*80}")
    for y in sorted(set(wf_strat.index.year)):
        m = wf_strat.index.year == y
        sr_y = wf_strat[m]
        br_y = wf_bnh[m]
        ret_s = (1 + sr_y).prod() - 1
        ret_b = (1 + br_y).prod() - 1
        shy = sr_y.mean() / sr_y.std() * np.sqrt(365) if sr_y.std() > 0 else 0
        short_y = (wf_pos[m] == -1).sum() / len(wf_pos[m]) * 100
        print(
            f"  {y}: Strat {ret_s:+7.1%} | B&H {ret_b:+7.1%} "
            f"| Excess {ret_s-ret_b:+7.1%} | Sharpe {shy:+.2f} | Short {short_y:.0f}%"
        )

    # Per-mode comparison if multiple modes selected
    if len(mode_counts) > 1:
        print(f"\n  Mode breakdown: {mode_counts}")


if __name__ == "__main__":
    main()
