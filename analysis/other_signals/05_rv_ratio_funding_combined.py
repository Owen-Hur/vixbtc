"""
05_rv_ratio_funding_combined.py

rv_ratio + 펀딩레이트 결합 전략 (4,200 조합 그리드 서치).

가설:
  - rv_ratio 확대 AND 누적 펀딩 상위 → 과밀 Long, 조정 임박 → Short
  - 두 조건 동시 만족 시에만 Short, 그 외 Long

결과:
  - 선택: sw=4, lw=70, qh=0.80, fr_window=1, fr_qh=0.70
  - OOS 전략 Sharpe 0.88, +185.3% 보이지만 Short 비율 3.2%
  - 사실상 B&H와 동일 (Long 96.8%)
  - 40일 Short hit rate 45% (동전던지기보다 나쁨)

핵심: 펀딩레이트 단독 예측력 분석에서
  - vs 다음날 BTC 상관: -0.005 (사실상 0)
  - 고펀딩 후 다음날 하락 확률: 46% (오히려 상승이 약간 우세)
  - "고펀딩 = 과밀 Long" 가설이 데이터에서 지지되지 않음

결론: 펀딩레이트는 BTC 방향 예측 시그널로 활용 불가.
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


def run_combined(sw, lw, qh, fr_window, fr_qh):
    """rv_ratio expansion AND 누적 펀딩 > threshold → Short."""
    rv_ratio = compute_rv_ratio(rv_daily, sw, lw)
    thresholds = compute_expanding_threshold(rv_ratio, qh)
    cum_fr = daily_funding.rolling(fr_window, min_periods=fr_window).mean()

    common = rv_ratio.index.intersection(cum_fr.dropna().index).intersection(daily_ret.index)
    rr = rv_ratio.reindex(common)
    th = thresholds.reindex(common)
    cfr = cum_fr.reindex(common)

    # 펀딩 threshold: expanding (당일 제외)
    cfr_vals = cfr.values
    fr_th = np.empty(len(cfr_vals))
    for i in range(len(cfr_vals)):
        fr_th[i] = np.inf if i == 0 else np.percentile(cfr_vals[:i], fr_qh * 100)
    fr_threshold = pd.Series(fr_th, index=common)

    # 결합: 둘 다 충족 시 Short
    rv_expand = rr > th
    fr_high = cfr > fr_threshold
    is_short = rv_expand & fr_high
    pos_decision = pd.Series(np.where(is_short, -1.0, 1.0), index=common)
    pos_applied = pos_decision.shift(1)
    pos_applied.iloc[0] = 0.0

    strat_ret, bnh_ret, _ = compute_strategy_returns(
        pos_applied, daily_ret, TAKER_FEE, daily_funding
    )
    return strat_ret, bnh_ret, pos_applied.reindex(strat_ret.index)


SEARCH = {
    'sw': [2, 3, 4, 5, 7],
    'lw': [14, 21, 30, 45, 60, 70, 80],
    'qh': [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    'fr_window': [1, 3, 5, 7],
    'fr_qh': [0.60, 0.65, 0.70, 0.75, 0.80],
}
combos = [(s, l, q, fw, fq)
          for s in SEARCH['sw'] for l in SEARCH['lw']
          for q in SEARCH['qh'] for fw in SEARCH['fr_window']
          for fq in SEARCH['fr_qh']
          if s < l]

print(f"탐색: {len(combos)} 조합")
results = []
for sw, lw, qh, fw, fq in combos:
    try:
        sr, _, pos = run_combined(sw, lw, qh, fw, fq)
        sr_t = sr[sr.index <= TRAIN_END]
        if len(sr_t) < 60:
            continue
        sh = sr_t.mean() / sr_t.std() * np.sqrt(365) if sr_t.std() > 0 else 0
        results.append({'sw': sw, 'lw': lw, 'qh': qh, 'fr_w': fw, 'fr_qh': fq, 'train_sharpe': sh})
    except Exception:
        continue

df = pd.DataFrame(results).sort_values('train_sharpe', ascending=False)
best = df.iloc[0]
sw, lw, qh, fw, fq = int(best['sw']), int(best['lw']), best['qh'], int(best['fr_w']), best['fr_qh']
print(f"선택: sw={sw}, lw={lw}, qh={qh}, fr_w={fw}, fr_qh={fq}")

sr, br, pos = run_combined(sw, lw, qh, fw, fq)
sr_o = sr[sr.index >= OOS_START]
br_o = br[sr.index >= OOS_START]
pos_o = pos.reindex(sr_o.index)

cum_s = (1 + sr_o).cumprod()
cum_b = (1 + br_o).cumprod()
sh_o = sr_o.mean() / sr_o.std() * np.sqrt(365)
print(f"\nOOS 전략: {cum_s.iloc[-1]-1:+.1%}, Sharpe {sh_o:.2f}")
print(f"OOS B&H:  {cum_b.iloc[-1]-1:+.1%}")
print(f"Short 비율: {(pos_o==-1).mean()*100:.1f}%")
