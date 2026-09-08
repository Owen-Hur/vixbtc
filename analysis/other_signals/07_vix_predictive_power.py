"""
07_vix_predictive_power.py

VIX 시그널의 BTC 방향 예측력 종합 분석.

5가지 시그널 정의:
  1. VIX 절대 수준 (level)
  2. VIX 일간 변화량 (ΔV)
  3. VIX 변화율 (%ΔV)
  4. VIX 5일 변화 (5일 추세)
  5. VIX vs MA20 위치

발견:
  - VIX 수준 vs 다음날 BTC 상관: +0.063 (양의 상관!)
  - VIX > 40 (극단 공포) 후 다음날 양수 56.4%, 평균 +0.96%
  - VIX 급등(+10%) 후 다음날 양수 57.9%, 평균 +0.48%
  - 모든 VIX 시그널이 +상관 또는 약한 음 상관

해석:
  - "VIX 높으면 BTC 하락"이 아니라 "VIX 높으면 BTC 상승" 방향
  - 단, 상관 0.06~0.17로 매우 약함
  - VIX 높은 시기는 BTC가 이미 하락한 뒤 → 평균회귀 효과 포착

결론: VIX → BTC 방향 예측력 매우 약하고, 직관과 반대.
"""
import pandas as pd
import numpy as np
from rv_regime.data_loader import load_and_prepare

rv_daily, daily_ret = load_and_prepare()
vix = pd.read_parquet('/Users/macbook/btc_project/data/vix_daily.parquet')
vix.index = pd.to_datetime(vix.index).tz_localize('America/New_York')

common = daily_ret.index.intersection(vix.index)
ret = daily_ret.reindex(common)
v = vix['vix'].reindex(common)
next_ret = ret.shift(-1)

# 1. VIX 수준별
print("--- VIX 수준별 다음날 BTC 수익률 ---")
for lo, hi, label in [(0,15,"<15"), (15,20,"15-20"), (20,25,"20-25"),
                       (25,30,"25-30"), (30,40,"30-40"), (40,100,">40")]:
    mask = (v >= lo) & (v < hi)
    sub = next_ret[mask].dropna()
    if len(sub) == 0: continue
    print(f"  VIX {label}: {mask.sum()}일, 양수 {(sub>0).mean()*100:.1f}%, "
          f"평균 {sub.mean()*100:+.3f}%")

# 2. VIX 변화량 시그널들의 상관
print("\n--- 종합 상관 ---")
dv = v.diff()
dv_pct = v.pct_change()
dv5 = v - v.shift(5)
ma20 = v.rolling(20).mean()
ratio_ma = v / ma20

signals = {
    'VIX level': v, 'VIX Δ1일': dv, 'VIX %Δ1일': dv_pct,
    'VIX Δ5일': dv5, 'VIX/MA20': ratio_ma,
}
nr = next_ret.dropna()
ret5 = ret.rolling(5).sum().shift(-5)
print(f"  {'시그널':<15} {'corr(다음날)':>12} {'corr(5일후)':>12}")
for name, sig in signals.items():
    c1 = sig.corr(nr)
    c5 = sig.corr(ret5.reindex(sig.index))
    print(f"  {name:<15} {c1:>+12.4f} {c5:>+12.4f}")
