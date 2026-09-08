"""
02_download_vix_daily.py

FRED(St. Louis Fed)에서 VIX 일간 종가 다운로드.
2020-01-01 ~ 2026-05-28.
"""
import pandas as pd
import urllib.request
from pathlib import Path

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS&cosd=2020-01-01&coed=2026-05-28"
urllib.request.urlretrieve(url, '/tmp/vix_raw.csv')

vix = pd.read_csv('/tmp/vix_raw.csv')
vix.columns = ['date', 'vix']
vix['date'] = pd.to_datetime(vix['date'])
vix = vix.set_index('date')
vix = vix[vix['vix'] != '.']
vix['vix'] = vix['vix'].astype(float)
vix = vix.dropna()

out_path = Path(__file__).parent.parent / 'data' / 'vix_daily.parquet'
vix.to_parquet(out_path)
print(f"Saved {len(vix)} days to {out_path}")
print(f"Range: {vix.index[0].date()} ~ {vix.index[-1].date()}")
print(f"Mean: {vix['vix'].mean():.1f}, Min: {vix['vix'].min():.1f}, Max: {vix['vix'].max():.1f}")
