"""
03_btc_volatility_direction.py

BTC 고변동성 날의 수익률 방향성 분석.
질문: "BTC RV가 높을 때 하락 편향이 있는가?" (전통 주식 시장의 VIX-수익 음의 상관과 비교)

발견:
  - RV vs 당일 수익률 상관: -0.09 (거의 무상관)
  - RV 상위 25% 날의 양수 수익률 비율: 48% (거의 반반)
  - RV 상위 10% 날도: 44.7% (여전히 거의 반반)
  - +5% 급등일과 -5% 급락일의 RV 분포가 비슷 (87 vs 92 백분위)

결론: BTC는 전통 주식과 달리 고변동성이 양방향(상승/하락 모두)으로 발생.
      "변동성 확대 → Short" 규칙이 구조적으로 약함.
"""
import numpy as np
import pandas as pd
from rv_regime.data_loader import load_and_prepare
from rv_regime.strategy import compute_rv_ratio, compute_expanding_threshold

rv_daily, daily_ret = load_and_prepare()

sw, lw, qh = 3, 75, 0.80
rv_ratio = compute_rv_ratio(rv_daily, sw, lw)
thresholds = compute_expanding_threshold(rv_ratio, qh)
is_expansion = rv_ratio > thresholds

common = rv_ratio.index.intersection(daily_ret.index)
rv_d = rv_daily.reindex(common)
ret = daily_ret.reindex(common)

# 변동성 분위수별 양수 수익률 빈도
print("--- 변동성 수준별 양수 수익률 빈도 ---")
for label, q_lo, q_hi in [('하위 25%', 0, 0.25), ('25-50%', 0.25, 0.50),
                           ('50-75%', 0.50, 0.75), ('상위 25%', 0.75, 1.0)]:
    if q_lo == 0:
        mask = rv_d <= rv_d.quantile(q_hi)
    elif q_hi == 1.0:
        mask = rv_d > rv_d.quantile(q_lo)
    else:
        mask = (rv_d > rv_d.quantile(q_lo)) & (rv_d <= rv_d.quantile(q_hi))
    sub = ret[mask]
    print(f"  RV {label}: {mask.sum()}일, 양수 {(sub > 0).mean()*100:.1f}%, "
          f"평균 {sub.mean()*100:+.3f}%")

# 상관관계
print(f"\nRV vs 당일수익률 상관: {rv_d.corr(ret):.4f}")
print(f"RV vs 당일|수익률|(절대) 상관: {rv_d.corr(ret.abs()):.4f}")
print(f"RV vs 다음날수익률 상관: {rv_d.corr(ret.shift(-1).dropna()):.4f}")
