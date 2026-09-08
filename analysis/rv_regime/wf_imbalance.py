"""
Walk-Forward for Trade Imbalance Strategy
Run: python -m rv_regime.wf_imbalance

Strategy:
  1. Compute daily mean trade_imbalance from 1m bars (CME trade day convention)
  2. Smooth with rolling mean over `rw` days
  3. Expanding percentile thresholds (upper qh, lower ql)
  4. Two modes:
     - mean_reversion: overbought (high imbalance) → short, oversold → long
     - momentum: high imbalance → long, low imbalance → short
  5. Default position: +1 (long)
  6. shift(1) to avoid look-ahead bias
"""
import numpy as np
import pandas as pd
from itertools import product

from rv_regime.data_loader import load_btc_1m, assign_trade_date, compute_daily_ret
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

SEARCH_RW = [3, 5, 7, 14, 21]
SEARCH_QH = [0.80, 0.85, 0.90, 0.95]
SEARCH_QL = [0.05, 0.10, 0.15, 0.20]
SEARCH_MODE = ['mean_reversion', 'momentum']


def compute_daily_imbalance(df_1m, trade_date):
    """Daily mean trade_imbalance per trade day."""
    imb = df_1m['trade_imbalance'].groupby(trade_date).mean()
    imb.index = pd.to_datetime(imb.index)
    imb = imb.sort_index()
    imb.name = 'imbalance'
    return imb


def compute_expanding_thresholds(series, qh, ql):
    """
    Expanding window percentile thresholds (bias-free).
    Returns upper and lower threshold series.
    """
    values = series.values
    n = len(values)
    th_upper = np.empty(n)
    th_lower = np.empty(n)
    for i in range(n):
        th_upper[i] = np.percentile(values[:i + 1], qh * 100)
        th_lower[i] = np.percentile(values[:i + 1], ql * 100)
    upper = pd.Series(th_upper, index=series.index, name='th_upper')
    lower = pd.Series(th_lower, index=series.index, name='th_lower')
    return upper, lower


def run_imbalance(daily_imb, daily_ret, rw, qh, ql, mode):
    """
    Run the imbalance strategy with given params.

    Returns strat_ret, bnh_ret, pos series.
    """
    # Smooth imbalance with rolling mean
    smooth = daily_imb.rolling(rw, min_periods=rw).mean().dropna()

    # Expanding thresholds on the smoothed series
    th_upper, th_lower = compute_expanding_thresholds(smooth, qh, ql)

    # Generate signals
    if mode == 'mean_reversion':
        # High imbalance (overbought) → short, low imbalance (oversold) → long
        pos_decision = pd.Series(1.0, index=smooth.index)  # default long
        pos_decision[smooth > th_upper] = -1.0  # overbought → short
        pos_decision[smooth < th_lower] = 1.0   # oversold → long (same as default)
    else:  # momentum
        # High imbalance → long (momentum), low imbalance → short
        pos_decision = pd.Series(1.0, index=smooth.index)  # default long
        pos_decision[smooth > th_upper] = 1.0   # strong buy pressure → long
        pos_decision[smooth < th_lower] = -1.0  # strong sell pressure → short

    # Avoid look-ahead: shift(1)
    pos_applied = pos_decision.shift(1)
    pos_applied.iloc[0] = 0.0

    # Align with returns
    common = pos_applied.index.intersection(daily_ret.index)
    pos = pos_applied.reindex(common)
    ret = daily_ret.reindex(common)

    # Transaction costs
    cost = pos.diff().abs() * TAKER_FEE
    cost.iloc[0] = abs(pos.iloc[0]) * TAKER_FEE

    strat_ret = pos * ret - cost
    return strat_ret, ret, pos


def select_best(daily_imb, daily_ret, train_end_ts):
    """Grid search over params, pick best Sharpe on training set."""
    combos = [
        (rw, qh, ql, mode)
        for rw, qh, ql, mode in product(SEARCH_RW, SEARCH_QH, SEARCH_QL, SEARCH_MODE)
        if qh + ql <= 1.0
    ]
    best_sharpe = -999
    best_params = None
    for rw, qh, ql, mode in combos:
        try:
            sr, _, _ = run_imbalance(daily_imb, daily_ret, rw, qh, ql, mode)
            s = sr[sr.index <= train_end_ts]
            if len(s) < 30:
                continue
            sh = s.mean() / s.std() * np.sqrt(365) if s.std() > 0 else 0
            if sh > best_sharpe:
                best_sharpe = sh
                best_params = (rw, qh, ql, mode)
        except Exception:
            continue
    return best_params, best_sharpe


def main():
    print("[wf_imbalance] Loading BTC 1m data...")
    df_1m = load_btc_1m()
    print(f"  {len(df_1m):,} rows, {df_1m.index[0]} ~ {df_1m.index[-1]}")

    # Check that trade_imbalance column exists
    if 'trade_imbalance' not in df_1m.columns:
        raise ValueError(
            f"trade_imbalance column not found. Available: {list(df_1m.columns)}"
        )

    trade_date = assign_trade_date(df_1m)
    daily_imb = compute_daily_imbalance(df_1m, trade_date)
    daily_ret = compute_daily_ret(df_1m, trade_date)

    print(f"  Imbalance: {len(daily_imb)} days, Returns: {len(daily_ret)} days")
    print(f"  Imbalance range: [{daily_imb.min():.4f}, {daily_imb.max():.4f}]")
    print(f"  Combos per round: {len([1 for rw, qh, ql, mode in product(SEARCH_RW, SEARCH_QH, SEARCH_QL, SEARCH_MODE) if qh + ql <= 1.0])}")

    tz = daily_imb.index.tz

    print("\n" + "=" * 80)
    print("  Walk-Forward: Trade Imbalance Strategy (22 rounds)")
    print("=" * 80)

    all_strat = []
    all_bnh = []
    all_pos = []
    round_info = []

    for ri, r in enumerate(WF_ROUNDS):
        train_end_ts = pd.Timestamp(r["train_end"])
        test_start_ts = pd.Timestamp(r["test_start"])
        test_end_ts = pd.Timestamp(r["test_end"])
        if tz is not None:
            train_end_ts = train_end_ts.tz_localize(tz)
            test_start_ts = test_start_ts.tz_localize(tz)
            test_end_ts = test_end_ts.tz_localize(tz)

        params, train_sh = select_best(daily_imb, daily_ret, train_end_ts)
        if params is None:
            print(f"  R{ri+1:>2}: no valid params — skipping")
            continue

        rw, qh, ql, mode = params
        sr, br, pf = run_imbalance(daily_imb, daily_ret, rw, qh, ql, mode)

        mask = (sr.index >= test_start_ts) & (sr.index <= test_end_ts)
        test_sr = sr[mask]
        test_br = br[mask]
        test_pf = pf[mask]

        if len(test_sr) == 0:
            print(f"  R{ri+1:>2}: no test data — skipping")
            continue

        test_sharpe = (
            test_sr.mean() / test_sr.std() * np.sqrt(365)
            if test_sr.std() > 0 else 0
        )
        test_ret = (1 + test_sr).cumprod().iloc[-1] - 1
        bnh_ret = (1 + test_br).cumprod().iloc[-1] - 1
        short_pct = (test_pf == -1).sum() / len(test_pf) * 100

        mode_short = 'MR' if mode == 'mean_reversion' else 'MO'

        info = {
            "round": ri + 1,
            "test_period": f"{r['test_start']}~{r['test_end']}",
            "rw": rw, "qh": qh, "ql": ql, "mode": mode,
            "train_sharpe": train_sh, "test_sharpe": test_sharpe,
            "test_ret": test_ret, "bnh_ret": bnh_ret,
            "short_pct": short_pct, "test_days": len(test_sr),
        }
        round_info.append(info)

        print(
            f"  R{ri+1:>2}: {r['test_start']}~{r['test_end']} | "
            f"rw={rw},qh={qh},ql={ql},{mode_short} | "
            f"Train {train_sh:.2f} -> Test {test_sharpe:+.2f} | "
            f"Strat {test_ret:+.1%} B&H {bnh_ret:+.1%} | Short {short_pct:.0f}%"
        )

        all_strat.append(test_sr)
        all_bnh.append(test_br)
        all_pos.append(test_pf)

    if not all_strat:
        print("\nNo valid rounds — exiting.")
        return

    wf_strat = pd.concat(all_strat)
    wf_bnh = pd.concat(all_bnh)
    wf_pos = pd.concat(all_pos)

    cum_s = (1 + wf_strat).cumprod()
    cum_b = (1 + wf_bnh).cumprod()
    sh = wf_strat.mean() / wf_strat.std() * np.sqrt(365) if wf_strat.std() > 0 else 0
    mdd = ((cum_s - cum_s.cummax()) / cum_s.cummax()).min()

    test_sharpes = [r["test_sharpe"] for r in round_info]

    # Mode breakdown
    mode_counts = {}
    for r in round_info:
        m = r["mode"]
        mode_counts[m] = mode_counts.get(m, 0) + 1

    print(f"\n{'='*80}")
    print(f"  WF Summary (Trade Imbalance)")
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
    actual_sharpe = sh
    shuffled = []
    for _ in range(2000):
        perm = np.random.permutation(len(ret_vals))
        s_ret = pos_vals * ret_vals[perm] - cost_vals
        s_std = np.std(s_ret)
        if s_std > 0:
            shuffled.append(np.mean(s_ret) / s_std * np.sqrt(365))
    shuffled = np.array(shuffled)
    p_val = (shuffled >= actual_sharpe).mean()
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
        print(
            f"  {y}: Strat {ret_s:+7.1%} | B&H {ret_b:+7.1%} "
            f"| Excess {ret_s-ret_b:+7.1%} | Sharpe {shy:+.2f}"
        )


if __name__ == "__main__":
    main()
