"""
Walk-Forward: Funding Rate MOMENTUM Strategy
Run: python -m rv_regime.wf_momentum

Core insight (from p=0.0065 discovery):
  High funding + rising OI = STRONG TREND CONTINUATION (not reversal!)
  Low funding + falling OI = TREND WEAKENING → short opportunity

Strategy:
  Default: LONG (+1)
  SHORT (-1) when: funding is LOW/falling AND OI is declining
  LONG (+1) otherwise (including when funding is high = trend strong)

  Position always +1 or -1 (max exposure, daily trading).
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

# Parameters — keep small
SEARCH_FW = [3, 5, 7, 14, 21]           # funding rolling window
SEARCH_OW = [3, 5, 7, 14, 21]           # OI change rolling window
SEARCH_FR_Q = [0.15, 0.20, 0.25, 0.30, 0.35]  # funding LOW percentile → short trigger
SEARCH_MODE = ['fr_oi', 'fr_only']       # with/without OI confirmation


def load_signals():
    """Load and align all signal data with daily returns."""
    _, daily_ret = load_and_prepare()
    tz = daily_ret.index.tz

    def align_dates(dates, tz):
        dt = pd.DatetimeIndex(pd.to_datetime(dates))
        return dt.normalize().tz_localize(tz)

    # Funding rate
    fr = pd.read_parquet('data/funding_rate_daily.parquet')
    fr_idx = align_dates(fr['date'], tz)
    funding = pd.Series(fr['funding_rate_mean'].values, index=fr_idx, name='funding_rate')

    # Metrics (OI)
    try:
        mt = pd.read_parquet('data/metrics_daily.parquet')
        mt_idx = align_dates(mt['date'], tz)
        if 'sum_open_interest_value' in mt.columns:
            oi_val = pd.Series(mt['sum_open_interest_value'].values, index=mt_idx, name='oi_value')
            oi_change = oi_val.pct_change()
            oi_change.name = 'oi_change'
        else:
            oi_change = None
    except Exception:
        oi_change = None

    return daily_ret, funding, oi_change


def expanding_percentile(series, q):
    """Expanding window percentile (bias-free)."""
    vals = series.values
    n = len(vals)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.percentile(vals[:i + 1], q * 100)
    return pd.Series(out, index=series.index)


def run_momentum(daily_ret, funding, oi_change, fw, ow, fr_q, mode):
    """
    Momentum strategy based on funding rate direction.

    Logic:
      - Compute rolling sum of funding rate (fw days)
      - Compute rolling mean of OI change (ow days)
      - SHORT when: funding is below its expanding lower percentile (fr_q)
        AND (in fr_oi mode) OI is declining (rolling mean < 0)
      - LONG otherwise (default) — high funding = strong trend = stay long
    """
    # Rolling sum of funding rate
    fr_smooth = funding.rolling(fw, min_periods=fw).sum().dropna()

    # Expanding lower threshold for funding
    th_lower = expanding_percentile(fr_smooth, fr_q)

    # Funding is weak/negative
    fr_weak = fr_smooth < th_lower

    if mode == 'fr_oi' and oi_change is not None:
        # OI declining (rolling mean)
        oi_smooth = oi_change.reindex(fr_smooth.index).rolling(ow, min_periods=ow).mean()
        oi_declining = oi_smooth < 0

        # SHORT only when funding is weak AND OI is declining
        short_signal = fr_weak & oi_declining
    else:
        # SHORT when funding is weak (no OI confirmation)
        short_signal = fr_weak

    pos = pd.Series(1.0, index=fr_smooth.index)  # default LONG
    pos[short_signal] = -1.0

    # Shift to avoid look-ahead
    pos_applied = pos.shift(1)
    pos_applied.iloc[0] = 1.0

    # Align with returns
    common = pos_applied.index.intersection(daily_ret.index)
    pos_final = pos_applied.reindex(common)
    ret = daily_ret.reindex(common)

    # Costs
    cost = pos_final.diff().abs() * TAKER_FEE
    cost.iloc[0] = 0

    strat_ret = pos_final * ret - cost
    return strat_ret, ret, pos_final


def select_best(daily_ret, funding, oi_change, train_end_ts):
    """Grid search: maximize Information Ratio with min 5% short."""
    combos = [
        (fw, ow, fr_q, m)
        for fw, ow, fr_q, m in product(SEARCH_FW, SEARCH_OW, SEARCH_FR_Q, SEARCH_MODE)
    ]

    best_ir = -999
    best_params = None
    best_raw_sharpe = 0

    for fw, ow, fr_q, mode in combos:
        try:
            sr, br, pos = run_momentum(
                daily_ret, funding, oi_change, fw, ow, fr_q, mode
            )
            s = sr[sr.index <= train_end_ts]
            b = br.reindex(s.index)
            p = pos.reindex(s.index)

            if len(s) < 30:
                continue

            # Require minimum short allocation
            short_pct = (p == -1).sum() / len(p)
            if short_pct < 0.05:
                continue

            # Information Ratio
            excess = s - b
            ir = excess.mean() / excess.std() * np.sqrt(365) if excess.std() > 0 else 0

            raw_sh = s.mean() / s.std() * np.sqrt(365) if s.std() > 0 else 0

            if ir > best_ir:
                best_ir = ir
                best_params = (fw, ow, fr_q, mode)
                best_raw_sharpe = raw_sh
        except Exception:
            continue

    return best_params, best_raw_sharpe


def main():
    print("=" * 80)
    print("  Walk-Forward: Funding Rate MOMENTUM Strategy")
    print("=" * 80)

    daily_ret, funding, oi_change = load_signals()
    tz = daily_ret.index.tz

    print(f"  Funding: {len(funding)} days ({funding.index[0].date()} ~ {funding.index[-1].date()})")
    if oi_change is not None:
        print(f"  OI change: {oi_change.notna().sum()} days")

    n_combos = len(SEARCH_FW) * len(SEARCH_OW) * len(SEARCH_FR_Q) * len(SEARCH_MODE)
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

        params, train_sh = select_best(daily_ret, funding, oi_change, train_end_ts)
        if params is None:
            print(f"  R{ri+1:>2}: no valid params")
            continue

        fw, ow, fr_q, mode = params
        sr, br, pf = run_momentum(daily_ret, funding, oi_change, fw, ow, fr_q, mode)

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
            "fw": fw, "ow": ow, "fr_q": fr_q, "mode": mode,
            "train_sharpe": train_sh, "test_sharpe": test_sharpe,
            "test_ret": test_ret, "bnh_ret": bnh_ret,
            "short_pct": short_pct, "test_days": len(test_sr),
        }
        round_info.append(info)

        mode_s = 'FR+OI' if mode == 'fr_oi' else 'FR'
        print(
            f"  R{ri+1:>2}: {r['test_start']}~{r['test_end']} | "
            f"fw={fw},ow={ow},q={fr_q},{mode_s} | "
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
    print(f"  WF Summary (Funding Rate MOMENTUM)")
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


if __name__ == "__main__":
    main()
