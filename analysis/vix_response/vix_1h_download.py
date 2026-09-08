import yfinance as yf
import pandas as pd
import os

# VIX 1시간봉 다운로드
vix_1h = yf.download('^VIX', period='max', interval='60m', progress=False)

# 멀티인덱스 정리
vix_1h.columns = vix_1h.columns.get_level_values(0)
vix_1h = vix_1h[['Close']].rename(columns={'Close': 'vix'})
vix_1h.index.name = 'timestamp'

# UTC → ET 변환
vix_1h.index = vix_1h.index.tz_convert('America/New_York')

print(f"VIX 1h: {vix_1h.index.min()} ~ {vix_1h.index.max()}, {len(vix_1h)}건")
print(vix_1h.head())

# 저장
vix_1h.to_parquet('data/vix_1h.parquet')
print("\n저장: data/vix_1h.parquet")

# BTC 1분봉 대응 기간 확인
btc_files = sorted([f for f in os.listdir('data/btc_1m_24h/') if '2024' in f or '2025' in f or '2026' in f])
print(f"\nBTC 1분봉 파일 (2024~): {len(btc_files)}개")
print(f"  {btc_files[0]} ~ {btc_files[-1]}")
