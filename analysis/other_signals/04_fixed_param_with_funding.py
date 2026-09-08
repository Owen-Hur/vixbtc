"""
04_fixed_param_with_funding.py

고정 파라미터 OOS 검증 (lookahead bias 없음, 펀딩 포함).

설계:
  - Train: 2020-01 ~ 2022-12 (3년)에서만 파라미터 선택
  - OOS: 2023-01 ~ 2026-05 (3.4년)에 고정 파라미터 적용
  - Train 데이터는 OOS 결정에 절대 사용 안 함

결과:
  - 선택 파라미터: sw=4, lw=14, qh=0.60 (Train Sharpe 1위)
  - OOS 전략: +89.4%, Sharpe 0.63
  - OOS B&H:  +263.2%
  - 초과수익: -173.9%p
  - 상승장(2023, 2024)에서 대패, 하락장(2025, 2026)에서만 승

결론: 전략 자체는 돈을 벌지만 B&H 대비 구조적 불리.
"""
import numpy as np
import pandas as pd
from itertools import product
from rv_regime.data_loader import load_and_prepare, load_daily_funding
from rv_regime.strategy import (
    compute_rv_ratio, compute_expanding_threshold,
    generate_positions, compute_strategy_returns,
)
from rv_regime.config import TAKER_FEE

rv_daily, daily_ret = load_and_prepare()
daily_funding = load_daily_funding()
tz = rv_daily.index.tz

TRAIN_END = pd.Timestamp("2022-12-31").tz_localize(tz)
OOS_START = pd.Timestamp("2023-01-01").tz_localize(tz)

SEARCH_SW = [2, 3, 4, 5, 7, 10]
SEARCH_LW = [14, 21, 30, 45, 60, 65, 70, 75, 80, 85, 90]
SEARCH_QH = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

combos = [(s, l, q) for s, l, q in product(SEARCH_SW, SEARCH_LW, SEARCH_QH) if s < l]

# Train 구간 그리드 서치
results = []
for sw, lw, qh in combos:
    rv_ratio = compute_rv_ratio(rv_daily, sw, lw)
    thresholds = compute_expanding_threshold(rv_ratio, qh)
    pos_applied = generate_positions(rv_ratio, thresholds)
    strat_ret, _, _ = compute_strategy_returns(
        pos_applied, daily_ret, TAKER_FEE, daily_funding
    )
    sr_train = strat_ret[strat_ret.index <= TRAIN_END]
    if len(sr_train) < 60:
        continue
    sh = sr_train.mean() / sr_train.std() * np.sqrt(365) if sr_train.std() > 0 else 0
    results.append({'sw': sw, 'lw': lw, 'qh': qh, 'train_sharpe': sh})

df = pd.DataFrame(results).sort_values('train_sharpe', ascending=False)
best = df.iloc[0]
sw, lw, qh = int(best['sw']), int(best['lw']), best['qh']
print(f"선택: sw={sw}, lw={lw}, qh={qh:.2f} (Train Sharpe {best['train_sharpe']:.2f})")

# OOS 검증
rv_ratio = compute_rv_ratio(rv_daily, sw, lw)
thresholds = compute_expanding_threshold(rv_ratio, qh)
pos_applied = generate_positions(rv_ratio, thresholds)
strat_ret, bnh_ret, _ = compute_strategy_returns(
    pos_applied, daily_ret, TAKER_FEE, daily_funding
)

sr_oos = strat_ret[strat_ret.index >= OOS_START]
br_oos = bnh_ret[strat_ret.index >= OOS_START]

cum_s = (1 + sr_oos).cumprod()
cum_b = (1 + br_oos).cumprod()
sh_oos = sr_oos.mean() / sr_oos.std() * np.sqrt(365)

print(f"\nOOS 결과:")
print(f"  전략: {cum_s.iloc[-1]-1:+.1%}, Sharpe {sh_oos:.2f}")
print(f"  B&H:  {cum_b.iloc[-1]-1:+.1%}")
print(f"  초과: {cum_s.iloc[-1]-cum_b.iloc[-1]:+.1%}")
