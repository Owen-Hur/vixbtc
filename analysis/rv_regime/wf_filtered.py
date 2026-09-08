"""
Walk-Forward for Direction-Filtered RV Regime
Run: python -m rv_regime.wf_filtered
"""
import numpy as np
import pandas as pd
from itertools import product

from rv_regime.data_loader import load_and_prepare
from rv_regime.strategy import compute_rv_ratio, compute_expanding_threshold
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

SEARCH_SW = [2, 3, 4, 5, 7]
SEARCH_LW = [21, 45, 60, 70, 75, 80, 90]
SEARCH_QH = [0.70, 0.75, 0.80, 0.85]
SEARCH_RW = [3, 5, 7, 10, 14, 21]


def run_filtered(rv_daily, daily_ret, sw, lw, qh, rw):
    rv_ratio = compute_rv_ratio(rv_daily, sw, lw)
    thresholds = compute_expanding_threshold(rv_ratio, qh)
    is_expansion = rv_ratio > thresholds
    rolling_ret = daily_ret.rolling(rw, min_periods=rw).sum().reindex(rv_ratio.index)
    is_bearish = rolling_ret < 0
    short_signal = is_expansion & is_bearish
    pos_decision = pd.Series(
        np.where(short_signal, -1.0, 1.0), index=rv_ratio.index
    )
    pos_applied = pos_decision.shift(1)
    pos_applied.iloc[0] = 0.0
    common = pos_applied.index.intersection(daily_ret.index)
    pos = pos_applied.reindex(common)
    ret = daily_ret.reindex(common)
    cost = pos.diff().abs() * TAKER_FEE
    cost.iloc[0] = abs(pos.iloc[0]) * TAKER_FEE
    strat_ret = pos * ret - cost
    return strat_ret, ret, pos


def select_best(rv_daily, daily_ret, train_end_ts):
    combos = [
        (s, l, q, rw)
        for s, l, q, rw in product(SEARCH_SW, SEARCH_LW, SEARCH_QH, SEARCH_RW)
        if s < l
    ]
    best_sharpe = -999
    best_params = None
    for sw, lw, qh, rw in combos:
        try:
            sr, _, _ = run_filtered(rv_daily, daily_ret, sw, lw, qh, rw)
            s = sr[sr.index <= train_end_ts]
            if len(s) < 30:
                continue
            sh = s.mean() / s.std() * np.sqrt(365) if s.std() > 0 else 0
            if sh > best_sharpe:
                best_sharpe = sh
                best_params = (sw, lw, qh, rw)
        except Exception:
            continue
    return best_params, best_sharpe


def main():
    rv_daily, daily_ret = load_and_prepare()
    tz = rv_daily.index.tz

    print("=" * 80)
    print("  Walk-Forward: Direction-Filtered RV Regime (22 rounds)")
    print("=" * 80)

    all_strat = []
    all_bnh = []
    all_pos = []
    round_info = []

    for ri, r in enumerate(WF_ROUNDS):
        train_end_ts = pd.Timestamp(r["train_end"]).tz_localize(tz)
        test_start_ts = pd.Timestamp(r["test_start"]).tz_localize(tz)
        test_end_ts = pd.Timestamp(r["test_end"]).tz_localize(tz)

        params, train_sh = select_best(rv_daily, daily_ret, train_end_ts)
        if params is None:
            print(f"  R{ri+1}: no valid params")
            continue

        sw, lw, qh, rw = params
        sr, br, pf = run_filtered(rv_daily, daily_ret, sw, lw, qh, rw)

        mask = (sr.index >= test_start_ts) & (sr.index <= test_end_ts)
        test_sr = sr[mask]
        test_br = br[mask]
        test_pf = pf[mask]

        test_sharpe = (
            test_sr.mean() / test_sr.std() * np.sqrt(365)
            if test_sr.std() > 0
            else 0
        )
        test_ret = (1 + test_sr).cumprod().iloc[-1] - 1 if len(test_sr) > 0 else 0
        bnh_ret = (1 + test_br).cumprod().iloc[-1] - 1 if len(test_br) > 0 else 0
        short_pct = (
            (test_pf == -1).sum() / len(test_pf) * 100 if len(test_pf) > 0 else 0
        )

        info = {
            "round": ri + 1,
            "test_period": f"{r['test_start']}~{r['test_end']}",
            "sw": sw, "lw": lw, "qh": qh, "rw": rw,
            "train_sharpe": train_sh, "test_sharpe": test_sharpe,
            "test_ret": test_ret, "bnh_ret": bnh_ret,
            "short_pct": short_pct, "test_days": len(test_sr),
        }
        round_info.append(info)

        print(
            f"  R{ri+1:>2}: {r['test_start']}~{r['test_end']} | "
            f"sw={sw},lw={lw},qh={qh},rw={rw} | "
            f"Train {train_sh:.2f} -> Test {test_sharpe:+.2f} | "
            f"Strat {test_ret:+.1%} B&H {bnh_ret:+.1%} | Short {short_pct:.0f}%"
        )

        all_strat.append(test_sr)
        all_bnh.append(test_br)
        all_pos.append(test_pf)

    wf_strat = pd.concat(all_strat)
    wf_bnh = pd.concat(all_bnh)
    wf_pos = pd.concat(all_pos)

    cum_s = (1 + wf_strat).cumprod()
    cum_b = (1 + wf_bnh).cumprod()
    sh = wf_strat.mean() / wf_strat.std() * np.sqrt(365)
    mdd = ((cum_s - cum_s.cummax()) / cum_s.cummax()).min()

    test_sharpes = [r["test_sharpe"] for r in round_info]

    print(f"\n{'='*80}")
    print(f"  WF Summary (Direction-Filtered)")
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
        print(
            f"  {y}: Strat {ret_s:+7.1%} | B&H {ret_b:+7.1%} "
            f"| Excess {ret_s-ret_b:+7.1%} | Sharpe {shy:+.2f}"
        )


if __name__ == "__main__":
    main()
