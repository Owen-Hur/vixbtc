"""
06_vix_ma_alignment.py

VIX 다중 MA 정배열/역배열 → BTC 영향 분석.

설계:
  - 5-MA (5, 10, 20, 40, 60일) 모두 정배열 → VIX 상승 추세 (공포 증가)
  - 모두 역배열 → VIX 하락 추세 (공포 감소)
  - 각 전환 순간 후 BTC 1/5/10일 수익률 측정

발견:
  - 정배열 진입: 10일 후 BTC +5.0%, 양수 61.5% (직관과 반대)
  - 역배열 진입: 10일 후 BTC +2.8%, 양수 54.8%
  - 정배열/역배열 모두 그 후 BTC 상승 → 시그널이 직관과 모순

전략 백테스트 (역배열→Long, 정배열→Short):
  - 전략 -60.5%, B&H +472.6%, 초과 -533%p

결론: VIX MA 정배열/역배열은 BTC 트레이딩 시그널로 사용 불가.
       정상 시장에서 VIX-BTC 연결 약하다는 Mensi(2023) 결론 재확인.
"""
import pandas as pd
import numpy as np
from rv_regime.data_loader import load_and_prepare, load_daily_funding

rv_daily, daily_ret = load_and_prepare()
daily_funding = load_daily_funding()
vix = pd.read_parquet('/Users/macbook/btc_project/data/vix_daily.parquet')
vix.index = pd.to_datetime(vix.index).tz_localize('America/New_York')

common = daily_ret.index.intersection(vix.index)
ret = daily_ret.reindex(common)
v = vix['vix'].reindex(common)

# 다중 MA
mas = {w: v.rolling(w, min_periods=w).mean() for w in [5, 10, 20, 40, 60]}
aligned = pd.DataFrame({f'ma{w}': mas[w] for w in [5,10,20,40,60]})
aligned['ret'] = ret
aligned = aligned.dropna()

# 정배열/역배열
bull = (aligned['ma5'] > aligned['ma10']) & (aligned['ma10'] > aligned['ma20']) & \
       (aligned['ma20'] > aligned['ma40']) & (aligned['ma40'] > aligned['ma60'])
bear = (aligned['ma5'] < aligned['ma10']) & (aligned['ma10'] < aligned['ma20']) & \
       (aligned['ma20'] < aligned['ma40']) & (aligned['ma40'] < aligned['ma60'])

print(f"정배열(VIX↑): {bull.sum()}일 ({bull.mean()*100:.1f}%)")
print(f"역배열(VIX↓): {bear.sum()}일 ({bear.mean()*100:.1f}%)")

# 전환 순간 → 이후 N일
bull_enter = bull & ~bull.shift(1).fillna(False)
bear_enter = bear & ~bear.shift(1).fillna(False)

for label, mask in [("정배열 진입", bull_enter), ("역배열 진입", bear_enter)]:
    print(f"\n{label}: {mask.sum()}회")
    for fwd in [1, 5, 10]:
        cum_ret = ret.rolling(fwd).sum().shift(-fwd)
        sub = cum_ret.reindex(aligned.index)[mask].dropna()
        if len(sub) > 0:
            print(f"  +{fwd}일: 양수 {(sub>0).mean()*100:.1f}%, 평균 {sub.mean()*100:+.3f}%")
