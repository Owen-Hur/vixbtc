"""
VIX 종가(16:00 ET) 확정 시점을 기준으로, BTC 1분봉에서 1분~24시간 후 수익률을
시차별로 추출해 반응 데이터셋을 재구축한다.

입력: data/vix_daily.parquet, data/btc_1m_24h/
출력: data/vix_btc_response.parquet
"""
import pandas as pd
import numpy as np
import os

# ── 1. VIX 일봉 로드 ──
vix = pd.read_parquet('data/vix_daily.parquet')
vix.index = pd.to_datetime(vix.index)
vix = vix.sort_index()
vix['vix_change'] = vix['vix'].diff()
vix['vix_pct'] = vix['vix'].pct_change()
vix = vix.dropna()
print(f"VIX: {vix.index.min().date()} ~ {vix.index.max().date()}, {len(vix)}건")

# ── 2. BTC 1분봉 전체 로드 ──
btc_dir = 'data/btc_1m_24h/'
files = sorted([f for f in os.listdir(btc_dir) if f.endswith('.parquet')])
print(f"BTC 1분봉 파일: {len(files)}개")

btc_list = []
for f in files:
    chunk = pd.read_parquet(os.path.join(btc_dir, f))[['close']]
    btc_list.append(chunk)
btc = pd.concat(btc_list).sort_index()
btc = btc[~btc.index.duplicated(keep='first')]
print(f"BTC: {btc.index.min()} ~ {btc.index.max()}, {len(btc):,}건")

# ET 타임존 확인
if btc.index.tz is None:
    import pytz
    btc.index = btc.index.tz_localize('America/New_York')
print(f"BTC timezone: {btc.index.tz}")

# ── 3. VIX 확정 시점(16:00 ET) 기준 BTC 수익률 계산 ──
offsets = {
    'btc_1m': 1, 'btc_5m': 5, 'btc_10m': 10, 'btc_15m': 15,
    'btc_20m': 20, 'btc_30m': 30, 'btc_45m': 45, 'btc_1h': 60,
    'btc_90m': 90, 'btc_2h': 120, 'btc_3h': 180, 'btc_4h': 240,
    'btc_6h': 360, 'btc_8h': 480, 'btc_12h': 720, 'btc_24h': 1440,
}

import pytz
et = pytz.timezone('America/New_York')

results = []
skipped = 0
for date, row in vix.iterrows():
    # 16:00 ET on VIX date
    close_time = pd.Timestamp(date.year, date.month, date.day, 16, 0, tzinfo=et)
    
    # BTC price at 16:00 ET
    if close_time not in btc.index:
        skipped += 1
        continue
    
    base_price = btc.loc[close_time, 'close']
    
    record = {
        'date': date,
        'vix': row['vix'],
        'vix_change': row['vix_change'],
        'vix_pct': row['vix_pct'],
    }
    
    valid = True
    for col_name, minutes in offsets.items():
        target_time = close_time + pd.Timedelta(minutes=minutes)
        if target_time in btc.index:
            target_price = btc.loc[target_time, 'close']
            record[col_name] = (target_price - base_price) / base_price
        else:
            record[col_name] = np.nan
    
    # btc_1m이라도 있으면 포함
    if not np.isnan(record.get('btc_1m', np.nan)):
        record['year'] = date.year
        results.append(record)

print(f"\n스킵: {skipped}건 (16:00 ET에 BTC 데이터 없음)")

df = pd.DataFrame(results)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['year'].astype(int)

print(f"\n=== 최종 결과 ===")
print(f"  기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"  건수: {len(df)}")
print(f"  연도별: {df['year'].value_counts().sort_index().to_dict()}")
print(f"  NaN 현황:")
for col in offsets.keys():
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        print(f"    {col}: {nan_count}건 NaN")

# 기존 백업 후 저장
import shutil
if os.path.exists('data/vix_btc_response.parquet'):
    shutil.copy('data/vix_btc_response.parquet', 'data/vix_btc_response_backup_2023.parquet')
    print("\n  기존 파일 백업: vix_btc_response_backup_2023.parquet")

df.to_parquet('data/vix_btc_response.parquet', index=False)
print(f"  저장 완료: data/vix_btc_response.parquet")
