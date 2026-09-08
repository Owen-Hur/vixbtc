"""
01_download_funding_rate.py

Binance BTCUSDT 무기한 선물 펀딩레이트 히스토리 다운로드.
8시간마다 정산 (UTC 00:00, 08:00, 16:00).
2020-01-01 ~ 2026-05-28, 약 7,000 레코드.
"""
import ccxt
import pandas as pd
import time
from pathlib import Path

exchange = ccxt.binance({'options': {'defaultType': 'future'}})
symbol = 'BTC/USDT'

all_funding = []
since = exchange.parse8601('2020-01-01T00:00:00Z')
end = exchange.parse8601('2026-05-28T00:00:00Z')

print("Downloading funding rate history...")
while since < end:
    try:
        data = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not data:
            break
        for d in data:
            all_funding.append({
                'timestamp': d['timestamp'],
                'datetime': d['datetime'],
                'funding_rate': d['fundingRate'],
            })
        since = data[-1]['timestamp'] + 1
        if len(all_funding) % 5000 == 0:
            print(f"  {len(all_funding)} records, latest: {data[-1]['datetime'][:10]}")
        time.sleep(0.1)
    except Exception as e:
        print(f"  Error: {e}, retrying...")
        time.sleep(1)
        continue

df = pd.DataFrame(all_funding)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime').sort_index()
df = df[~df.index.duplicated(keep='last')]

out_path = Path(__file__).parent.parent / 'data' / 'funding_rate_history.parquet'
df.to_parquet(out_path)
print(f"\nSaved {len(df)} records to {out_path}")
