"""
VIX 1시간봉의 각 변동 시점에 1분~4시간 후 BTC 수익률을 붙여 반응 데이터셋을 만든다.
장중(is_market_hours) 플래그도 함께 기록한다.

입력: data/vix_1h.parquet, data/btc_1m_24h/
출력: data/vix_1h_btc_response.parquet
"""
import pandas as pd
import numpy as np
import os
import pytz

# ── 1. VIX 1시간봉 로드 ──
vix = pd.read_parquet('data/vix_1h.parquet')
print(f"VIX 1h: {vix.index.min()} ~ {vix.index.max()}, {len(vix)}건")

# VIX Change율 계산
vix['vix_change'] = vix['vix'].diff()
vix['vix_pct'] = vix['vix'].pct_change()
vix = vix.dropna()

# ── 2. BTC 1분봉 로드 (2024~) ──
btc_dir = 'data/btc_1m_24h/'
btc_files = sorted([f for f in os.listdir(btc_dir) if f >= 'btc_1m_2024'])
btc_list = [pd.read_parquet(os.path.join(btc_dir, f))[['close']] for f in btc_files]
btc = pd.concat(btc_list).sort_index()
btc = btc[~btc.index.duplicated(keep='first')]
print(f"BTC 1m: {btc.index.min()} ~ {btc.index.max()}, {len(btc):,}건")

# ── 3. VIX 변동 시점 → BTC 수익률 계산 ──
# BTC 응답 시간대: 1분~4시간
offsets = {
    'btc_1m': 1, 'btc_2m': 2, 'btc_3m': 3, 'btc_5m': 5,
    'btc_10m': 10, 'btc_15m': 15, 'btc_20m': 20, 'btc_30m': 30,
    'btc_45m': 45, 'btc_1h': 60, 'btc_90m': 90, 'btc_2h': 120,
    'btc_3h': 180, 'btc_4h': 240,
}

results = []
skipped = 0
for ts, row in vix.iterrows():
    # VIX 변동 시점의 BTC 가격
    if ts not in btc.index:
        skipped += 1
        continue
    
    base_price = btc.loc[ts, 'close']
    record = {
        'timestamp': ts,
        'vix': row['vix'],
        'vix_change': row['vix_change'],
        'vix_pct': row['vix_pct'],
    }
    
    has_data = False
    for col_name, minutes in offsets.items():
        target_time = ts + pd.Timedelta(minutes=minutes)
        if target_time in btc.index:
            target_price = btc.loc[target_time, 'close']
            record[col_name] = (target_price - base_price) / base_price
            has_data = True
        else:
            record[col_name] = np.nan
    
    if has_data:
        results.append(record)

print(f"\n매칭 성공: {len(results)}건, 스킵: {skipped}건")

df = pd.DataFrame(results)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')

# 장중 시간 추출
df['hour'] = df['timestamp'].dt.hour
df['is_market_hours'] = (df['hour'] >= 9) & (df['hour'] <= 16)

print(f"\n전체: {len(df)}건")
print(f"장중(9:30-16:00): {df['is_market_hours'].sum()}건")
print(f"장외: {(~df['is_market_hours']).sum()}건")
print(f"NaN 현황: {df[list(offsets.keys())].isna().sum().to_dict()}")

df.to_parquet('data/vix_1h_btc_response.parquet', index=False)
print(f"\n저장: data/vix_1h_btc_response.parquet")
print(df.head())
