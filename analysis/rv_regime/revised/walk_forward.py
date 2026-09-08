"""
walk_forward.py — Walk-Forward Optimization

Expanding window 방식:
  각 라운드에서 Train 구간 IS Sharpe Top 1 파라미터 선택 후
  Test 구간에 적용. Test 일별 수익률을 연결하여 전체 성과 산출.

사용법:
  python -m rv_regime.walk_forward
"""
import numpy as np
import pandas as pd
from itertools import product

from rv_regime.config import TAKER_FEE
from rv_regime.data_loader import load_and_prepare, load_daily_funding
from rv_regime.strategy import (
    compute_rv_ratio, compute_expanding_threshold,
    generate_positions, compute_strategy_returns,
)

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
SEARCH_LW = [21, 45, 60, 70, 75, 80, 85, 90]
SEARCH_QH = [0.70, 0.75, 0.80, 0.85]


def _run_full(rv_daily, daily_ret, sw, lw, qh, daily_funding=None):
    """전체 기간 전략 실행, 일별 수익률/포지션 반환."""
    rv_ratio = compute_rv_ratio(rv_daily, sw, lw)
    thresholds = compute_expanding_threshold(rv_ratio, qh)
    pos_applied = generate_positions(rv_ratio, thresholds)
    strat_ret, bnh_ret, cost = compute_strategy_returns(
        pos_applied, daily_ret, TAKER_FEE, daily_funding
    )
    pos = pos_applied.reindex(strat_ret.index)
    return strat_ret, bnh_ret, pos


def _select_best_params(rv_daily, daily_ret, train_end_ts, daily_funding=None):
    """Train 구간에서 IS Sharpe Top 1 파라미터 선택."""
    combos = [(s, l, q) for s, l, q in product(SEARCH_SW, SEARCH_LW, SEARCH_QH)
              if s < l]

    best_sharpe = -999
    best_params = None

    for sw, lw, qh in combos:
        try:
            sr, _, _ = _run_full(rv_daily, daily_ret, sw, lw, qh, daily_funding)
            mask = sr.index <= train_end_ts
            s = sr[mask]
            if len(s) < 30:
                continue
            sh = s.mean() / s.std() * np.sqrt(365) if s.std() > 0 else 0
            if sh > best_sharpe:
                best_sharpe = sh
                best_params = (sw, lw, qh)
        except Exception:
            continue

    return best_params, best_sharpe


def run_walk_forward(rv_daily, daily_ret, rounds=None, daily_funding=None):
    """
    Walk-Forward Optimization 실행.

    Returns
    -------
    wf_strat : pd.Series — WF 연결 일별 전략 수익률
    wf_bnh : pd.Series — WF 연결 일별 B&H 수익률
    wf_pos : pd.Series — WF 연결 포지션
    round_info : list[dict] — 라운드별 정보
    """
    rounds = rounds or WF_ROUNDS
    tz = rv_daily.index.tz

    all_strat = []
    all_bnh = []
    all_pos = []
    round_info = []

    for ri, r in enumerate(rounds):
        train_end_ts = pd.Timestamp(r['train_end']).tz_localize(tz)
        test_start_ts = pd.Timestamp(r['test_start']).tz_localize(tz)
        test_end_ts = pd.Timestamp(r['test_end']).tz_localize(tz)

        params, train_sh = _select_best_params(rv_daily, daily_ret, train_end_ts, daily_funding)
        if params is None:
            print(f"  R{ri+1}: no valid params found")
            continue

        sw, lw, qh = params
        sr, br, pf = _run_full(rv_daily, daily_ret, sw, lw, qh, daily_funding)

        mask = (sr.index >= test_start_ts) & (sr.index <= test_end_ts)
        test_sr = sr[mask]
        test_br = br[mask]
        test_pf = pf[mask]

        test_cum = (1 + test_sr).cumprod()
        test_sharpe = (test_sr.mean() / test_sr.std() * np.sqrt(365)
                       if test_sr.std() > 0 else 0)
        test_ret = test_cum.iloc[-1] - 1 if len(test_cum) > 0 else 0

        info = {
            'round': ri + 1,
            'test_period': f"{r['test_start']} ~ {r['test_end']}",
            'sw': sw, 'lw': lw, 'qh': qh,
            'train_sharpe': train_sh,
            'test_sharpe': test_sharpe,
            'test_ret': test_ret,
            'test_days': len(test_sr),
        }
        round_info.append(info)
        print(f"  R{ri+1}: {info['test_period']} | "
              f"sw={sw},lw={lw},qh={qh} | "
              f"train Sh={train_sh:.2f} → test Sh={test_sharpe:+.2f}, "
              f"ret={test_ret:+.1%}")

        all_strat.append(test_sr)
        all_bnh.append(test_br)
        all_pos.append(test_pf)

    wf_strat = pd.concat(all_strat)
    wf_bnh = pd.concat(all_bnh)
    wf_pos = pd.concat(all_pos)

    return wf_strat, wf_bnh, wf_pos, round_info


def print_wf_summary(wf_strat, wf_bnh, round_info):
    """Walk-Forward 종합 결과 출력."""
    n = len(wf_strat)
    cum_s = (1 + wf_strat).cumprod()
    cum_b = (1 + wf_bnh).cumprod()
    sharpe = (wf_strat.mean() / wf_strat.std() * np.sqrt(365)
              if wf_strat.std() > 0 else 0)
    mdd = ((cum_s - cum_s.cummax()) / cum_s.cummax()).min()

    test_sharpes = [r['test_sharpe'] for r in round_info]

    print(f"\n{'='*70}")
    print(f"  Walk-Forward Summary")
    print(f"{'='*70}")
    print(f"  Period: {wf_strat.index[0].strftime('%Y-%m-%d')} ~ "
          f"{wf_strat.index[-1].strftime('%Y-%m-%d')} ({n} days)")
    print(f"  Strategy: {cum_s.iloc[-1]-1:>+.1%} (Sharpe {sharpe:.2f}, MDD {mdd:+.1%})")
    print(f"  B&H:      {cum_b.iloc[-1]-1:>+.1%}")
    print(f"  Excess:   {cum_s.iloc[-1]-cum_b.iloc[-1]:>+.1%}")
    print(f"\n  Rounds: {len(round_info)}")
    print(f"  Test Sharpe > 0: {sum(1 for s in test_sharpes if s > 0)}/{len(test_sharpes)}")
    print(f"  Avg Test Sharpe: {np.mean(test_sharpes):.2f}")


if __name__ == '__main__':
    print("=" * 70)
    print("  Walk-Forward Optimization (with funding rate)")
    print("=" * 70)

    rv_daily, daily_ret = load_and_prepare()
    daily_funding = load_daily_funding()
    print(f"  Funding rate: {len(daily_funding)} days, "
          f"mean {daily_funding.mean()*100:.4f}%/day")

    wf_strat, wf_bnh, wf_pos, info = run_walk_forward(
        rv_daily, daily_ret, daily_funding=daily_funding
    )
    print_wf_summary(wf_strat, wf_bnh, info)
