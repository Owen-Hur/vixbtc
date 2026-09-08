"""
08_vix_release_btc_response.py

VIX 일간 종가 발표(ET 16:15) 이후 BTC 시간별 반응 분석.

설계:
  - VIX 발표 시점: 매일 ET 16:15 (옵션 마감 후 종가 확정)
  - BTC 1분봉으로 16:15 이후 1분 ~ 24시간 누적 수익률 측정
  - VIX 변화(ΔV, %ΔV)와 BTC 누적수익의 상관 계산
  - 분석 도구: scipy.stats.pearsonr, ttest_ind (모델 학습 없음)
  - 기간: 2020-01-03 ~ 2026-05-25 (1,635일)

결과:
  [A] 상관 감쇠 곡선 (전체 기간)
    - 1m: r=+0.002 (무반응)
    - 5m: r=-0.062 (반응 시작)
    - 30m: r=-0.126 ***
    - 1h:  r=-0.111 ***
    - 4h:  r=-0.163 ***
    - 6h:  r=-0.176 *** (피크)
    - 8h:  r=-0.098 ***
    - 12h: r=-0.039 (소멸 시작)
    - 24h: r=-0.008 (완전 소멸)

  [C] 연도별 안정성 — 결정적 약점
    - 2020 r(4h) = -0.277 ***
    - 2021 r(4h) = -0.206 ***
    - 2022 r(4h) = -0.144 ***
    - 2023 r(4h) = -0.045 (효과 사라짐)
    - 2024 r(4h) = +0.012 (효과 없음)
    - 2025 r(4h) = -0.060 (효과 없음)
    - 2026 r(4h) = +0.128 (오히려 양 상관)

결론:
  - 전체 기간 통계: VIX → BTC 영향 6~8시간 유효 (피크 6h)
  - 그러나 효과는 2020~2022가 만든 것
  - 2023년 이후 사실상 소멸 → BTC ETF 승인(2024-01) 등 시장 구조 변화 추정
  - 현재 시점에서 트레이딩 시그널로 활용 불가
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# 저장소 루트 (data/ 가 있는 곳) — 실행 위치와 무관하게 파일 위치로부터 계산
BASE_DIR = Path(__file__).resolve().parents[2]

vix = pd.read_parquet(BASE_DIR / 'data' / 'vix_daily.parquet')
vix.index = pd.to_datetime(vix.index).tz_localize('America/New_York')
vix['vix_change'] = vix['vix'].diff()
vix['vix_pct'] = vix['vix'].pct_change()

btc_dir = BASE_DIR / 'data' / 'btc_1m_24h'
btc = pd.concat([pd.read_parquet(f) for f in sorted(btc_dir.glob('btc_1m_*.parquet'))]).sort_index()
btc = btc[~btc.index.duplicated(keep='last')]

horizons = [1, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 1440]
horizon_labels = ['1m','5m','10m','15m','20m','30m','45m','1h','90m','2h','3h','4h','6h','8h','12h','24h']

results = []
for date, row in vix.iterrows():
    if pd.isna(row['vix_change']):
        continue
    release_time = date.replace(hour=16, minute=15)
    future = btc.index[btc.index >= release_time]
    if len(future) == 0:
        continue
    release_idx = btc.index.get_loc(future[0])
    if (future[0] - release_time).total_seconds() > 300:
        continue
    start_price = btc.iloc[release_idx]['close']

    data = {'date': date.date(), 'vix': row['vix'], 'vix_change': row['vix_change'], 'vix_pct': row['vix_pct']}
    for h_min, h_label in zip(horizons, horizon_labels):
        end_idx = release_idx + h_min
        data[f'btc_{h_label}'] = (btc.iloc[end_idx]['close'] / start_price) - 1 if end_idx < len(btc) else np.nan
    results.append(data)

df = pd.DataFrame(results)
print(f"분석일: {len(df)}일, 기간: {df['date'].min()} ~ {df['date'].max()}")

# [A] 상관 감쇠 곡선
print(f"\n{'horizon':>8} {'r(ΔVIX)':>12} {'p':>8} {'n':>6}")
for h_label in horizon_labels:
    valid = df[df[f'btc_{h_label}'].notna() & df['vix_change'].notna()]
    if len(valid) < 30: continue
    r, p = stats.pearsonr(valid['vix_change'], valid[f'btc_{h_label}'])
    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
    print(f"{h_label:>8} {r:>+10.4f} {sig:>3} {p:>5.3f} {len(valid):>6}")

# [C] 연도별 안정성
print(f"\n--- 연도별 r(ΔVIX vs BTC) ---")
df['year'] = pd.to_datetime(df['date']).dt.year
print(f"{'year':>6}", end="")
for h in ['30m','1h','2h','4h','8h']:
    print(f"{'r('+h+')':>10}", end="")
print()
for yr in sorted(df['year'].unique()):
    sub = df[df['year'] == yr]
    print(f"{yr:>6}", end="")
    for h_label in ['30m','1h','2h','4h','8h']:
        v = sub[sub[f'btc_{h_label}'].notna() & sub['vix_change'].notna()]
        if len(v) < 15:
            print(f"{'-':>10}", end="")
        else:
            r, _ = stats.pearsonr(v['vix_change'], v[f'btc_{h_label}'])
            print(f"{r:>+10.3f}", end="")
    print()

df.to_parquet(BASE_DIR / 'data' / 'vix_btc_response_full.parquet')
