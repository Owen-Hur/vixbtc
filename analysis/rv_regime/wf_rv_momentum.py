"""
Walk-Forward: RV Regime + Momentum Strategy (Parallel + Progress)
Run: python -m rv_regime.wf_rv_momentum

SHORT only when ALL conditions met:
  1. RV ratio > expanding threshold (volatility expansion)
  2. Price < MA (downtrend)
  3. Recent return < 0 (bearish momentum)
LONG otherwise.
"""
import sys
import time
import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count

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

SEARCH_SW = [2, 3, 4, 5]
SEARCH_LW = [21, 45, 60, 75, 90]
SEARCH_QH = [0.70, 0.75, 0.80, 0.85]
SEARCH_MA = [10, 20, 30, 50]
SEARCH_RW = [5, 7, 14, 21]


def load_data():
    from rv_regime.data_loader import load_btc_1m, assign_trade_date
    rv_daily, daily_ret = load_and_prepare()
    df_1m = load_btc_1m()
    trade_date = assign_trade_date(df_1m)
    daily_close = df_1m.groupby(trade_date)['close'].last()
    daily_close.index = pd.to_datetime(daily_close.index)
    daily_close = daily_close.sort_index()
    daily_close.name = 'close'
    return rv_daily, daily_ret, daily_close


# --- Precompute all RV ratios and thresholds (the slow part) ---
def precompute_rv(rv_daily):
    """Precompute RV ratio and expanding threshold for all (sw, lw, qh) combos."""
    cache = {}
    combos = [(sw, lw, qh) for sw, lw, qh in product(SEARCH_SW, SEARCH_LW, SEARCH_QH) if sw < lw]
    total = len(combos)
    print(f"  [precompute] {total} RV ratio+threshold 조합 계산 중...", flush=True)
    t0 = time.time()
    for i, (sw, lw, qh) in enumerate(combos):
        rv_ratio = compute_rv_ratio(rv_daily, sw, lw)
        thresholds = compute_expanding_threshold(rv_ratio, qh)
        is_expansion = rv_ratio > thresholds
        cache[(sw, lw, qh)] = (rv_ratio.index, is_expansion.values)
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"    {i+1}/{total} ({(i+1)/total*100:.0f}%) - {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)
    elapsed = time.time() - t0
    print(f"  [precompute] 완료! {elapsed:.1f}초, {len(cache)} 조합", flush=True)
    return cache


def run_rv_momentum_fast(rv_expansion_idx, rv_expansion_vals, daily_ret, daily_close, ma_period, rw):
    """Fast version using precomputed RV expansion."""
    idx = rv_expansion_idx
    is_expansion = pd.Series(rv_expansion_vals, index=idx)

    ma = daily_close.rolling(ma_period, min_periods=ma_period).mean()
    ma = ma.reindex(idx)
    close_aligned = daily_close.reindex(idx)
    is_downtrend = close_aligned < ma

    rolling_ret = daily_ret.rolling(rw, min_periods=rw).sum().reindex(idx)
    is_bearish = rolling_ret < 0

    short_signal = is_expansion & is_downtrend & is_bearish

    pos_decision = pd.Series(np.where(short_signal, -1.0, 1.0), index=idx)
    pos_applied = pos_decision.shift(1)
    pos_applied.iloc[0] = 1.0

    common = pos_applied.index.intersection(daily_ret.index)
    pos = pos_applied.reindex(common)
    ret = daily_ret.reindex(common)

    cost = pos.diff().abs() * TAKER_FEE
    cost.iloc[0] = 0

    strat_ret = pos * ret - cost
    return strat_ret, ret, pos


def _eval_combo(args):
    """Worker function for parallel grid search."""
    sw, lw, qh, ma, rw, rv_idx, rv_vals, daily_ret, daily_close, train_end_ts = args
    try:
        sr, br, pos = run_rv_momentum_fast(rv_idx, rv_vals, daily_ret, daily_close, ma, rw)
        s = sr[sr.index <= train_end_ts]
        b = br.reindex(s.index)
        p = pos.reindex(s.index)

        if len(s) < 30:
            return None

        short_pct = (p == -1).sum() / len(p)
        if short_pct < 0.05:
            return None

        excess = s - b
        ir = excess.mean() / excess.std() * np.sqrt(365) if excess.std() > 0 else 0
        raw_sh = s.mean() / s.std() * np.sqrt(365) if s.std() > 0 else 0
        return (ir, raw_sh, (sw, lw, qh, ma, rw))
    except Exception:
        return None


def select_best_parallel(rv_cache, daily_ret, daily_close, train_end_ts, n_workers=None):
    """Parallel grid search using precomputed RV data."""
    if n_workers is None:
        n_workers = min(cpu_count(), 10)

    tasks = []
    for (sw, lw, qh), (rv_idx, rv_vals) in rv_cache.items():
        for ma, rw in product(SEARCH_MA, SEARCH_RW):
            tasks.append((sw, lw, qh, ma, rw, rv_idx, rv_vals, daily_ret, daily_close, train_end_ts))

    with Pool(n_workers) as pool:
        results = pool.map(_eval_combo, tasks)

    best_ir = -999
    best_params = None
    best_raw_sharpe = 0
    for r in results:
        if r is not None:
            ir, raw_sh, params = r
            if ir > best_ir:
                best_ir = ir
                best_params = params
                best_raw_sharpe = raw_sh

    return best_params, best_raw_sharpe


def main():
    print("=" * 80, flush=True)
    print("  Walk-Forward: RV Regime + Momentum (Parallel)", flush=True)
    print("=" * 80, flush=True)

    rv_daily, daily_ret, daily_close = load_data()
    tz = rv_daily.index.tz

    n_rv = sum(1 for sw, lw, qh in product(SEARCH_SW, SEARCH_LW, SEARCH_QH) if sw < lw)
    n_mom = len(SEARCH_MA) * len(SEARCH_RW)
    n_total = n_rv * n_mom
    print(f"  RV 조합: {n_rv}, 모멘텀 조합: {n_mom}, 총 조합: {n_total}/라운드", flush=True)
    print(f"  CPU 코어: {cpu_count()}, 병렬 워커: {min(cpu_count(), 10)}", flush=True)
    print(flush=True)

    # Precompute RV (the expensive part — do it once)
    rv_cache = precompute_rv(rv_daily)

    all_strat = []
    all_bnh = []
    all_pos = []
    round_info = []

    t_total = time.time()
    for ri, r in enumerate(WF_ROUNDS):
        t_round = time.time()
        train_end_ts = pd.Timestamp(r["train_end"]).tz_localize(tz)
        test_start_ts = pd.Timestamp(r["test_start"]).tz_localize(tz)
        test_end_ts = pd.Timestamp(r["test_end"]).tz_localize(tz)

        params, train_sh = select_best_parallel(rv_cache, daily_ret, daily_close, train_end_ts)
        round_time = time.time() - t_round

        if params is None:
            print(f"  R{ri+1:>2}/22: no valid params ({round_time:.1f}s)", flush=True)
            continue

        sw, lw, qh, ma, rw = params
        rv_idx, rv_vals = rv_cache[(sw, lw, qh)]
        sr, br, pf = run_rv_momentum_fast(rv_idx, rv_vals, daily_ret, daily_close, ma, rw)

        mask = (sr.index >= test_start_ts) & (sr.index <= test_end_ts)
        test_sr = sr[mask]
        test_br = br[mask]
        test_pf = pf[mask]

        if len(test_sr) == 0:
            print(f"  R{ri+1:>2}/22: no test data ({round_time:.1f}s)", flush=True)
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
            "sw": sw, "lw": lw, "qh": qh, "ma": ma, "rw": rw,
            "train_sharpe": train_sh, "test_sharpe": test_sharpe,
            "test_ret": test_ret, "bnh_ret": bnh_ret,
            "short_pct": short_pct, "test_days": len(test_sr),
        }
        round_info.append(info)

        elapsed_total = time.time() - t_total
        eta_total = elapsed_total / (ri + 1) * (22 - ri - 1)

        print(
            f"  R{ri+1:>2}/22: {r['test_start']}~{r['test_end']} | "
            f"sw={sw},lw={lw},qh={qh},ma={ma},rw={rw} | "
            f"Train {train_sh:.2f} -> Test {test_sharpe:+.2f} | "
            f"Strat {test_ret:+.1%} B&H {bnh_ret:+.1%} | Short {short_pct:.0f}% | "
            f"{round_time:.1f}s (ETA {eta_total:.0f}s)",
            flush=True
        )

        all_strat.append(test_sr)
        all_bnh.append(test_br)
        all_pos.append(test_pf)

    total_time = time.time() - t_total
    print(f"\n  전체 WF 소요: {total_time:.1f}초", flush=True)

    if not all_strat:
        print("\nNo valid rounds.", flush=True)
        return

    wf_strat = pd.concat(all_strat)
    wf_bnh = pd.concat(all_bnh)
    wf_pos = pd.concat(all_pos)

    cum_s = (1 + wf_strat).cumprod()
    cum_b = (1 + wf_bnh).cumprod()
    sh = wf_strat.mean() / wf_strat.std() * np.sqrt(365) if wf_strat.std() > 0 else 0
    mdd = ((cum_s - cum_s.cummax()) / cum_s.cummax()).min()

    test_sharpes = [r["test_sharpe"] for r in round_info]

    print(f"\n{'='*80}", flush=True)
    print(f"  WF Summary (RV Regime + Momentum)", flush=True)
    print(f"{'='*80}", flush=True)
    print(
        f"  Period: {wf_strat.index[0].strftime('%Y-%m-%d')} ~ "
        f"{wf_strat.index[-1].strftime('%Y-%m-%d')} ({len(wf_strat)} days)",
        flush=True
    )
    print(f"  Strategy: {cum_s.iloc[-1]-1:>+.1%} (Sharpe {sh:.2f}, MDD {mdd:+.1%})", flush=True)
    print(f"  B&H:      {cum_b.iloc[-1]-1:>+.1%}", flush=True)
    print(f"  Excess:   {cum_s.iloc[-1]-cum_b.iloc[-1]:>+.1%}", flush=True)
    print(
        f"  Rounds Sharpe > 0: "
        f"{sum(1 for s in test_sharpes if s > 0)}/{len(test_sharpes)}",
        flush=True
    )
    print(f"  Avg Test Sharpe: {np.mean(test_sharpes):.2f}", flush=True)
    print(f"  Median Test Sharpe: {np.median(test_sharpes):.2f}", flush=True)

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
    print(f" p={p_val:.4f} {'PASS' if p_val < 0.05 else 'FAIL'}", flush=True)

    # Yearly breakdown
    print(f"\n{'='*80}", flush=True)
    print(f"  WF Yearly Breakdown", flush=True)
    print(f"{'='*80}", flush=True)
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
            f"| Excess {ret_s-ret_b:+7.1%} | Sharpe {shy:+.2f} | Short {short_y:.0f}%",
            flush=True
        )


if __name__ == "__main__":
    main()
