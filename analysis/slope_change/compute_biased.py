"""Lookahead 방지 여부에 따른 방향 적중률 대비 — 슬라이드 8용"""
import pandas as pd, numpy as np, os
from pathlib import Path
from scipy import stats

# 저장소 루트 (data/ 가 있는 곳) — 실행 위치와 무관하게 파일 위치로부터 계산
BASE = str(Path(__file__).resolve().parents[2]); ET = "America/New_York"
vix = pd.read_parquet(f"{BASE}/data/vix_slope_daily.parquet")
vix.index = pd.DatetimeIndex(vix.index); vix = vix.sort_index()
vix["slope_change"] = vix["slope"].diff()

btc_dir = f"{BASE}/data/btc_1m_24h/"
files = sorted(f for f in os.listdir(btc_dir) if f.endswith(".parquet"))
btc = pd.concat([pd.read_parquet(btc_dir+f)[["close"]] for f in files]).sort_index()
btc = btc[~btc.index.duplicated(keep="first")]
btc.index = btc.index.tz_localize(ET) if btc.index.tz is None else btc.index.tz_convert(ET)
bi = btc.index
def price_at(ts):
    p = bi.searchsorted(ts)
    if p >= len(bi): return np.nan
    if (bi[p]-ts) > pd.Timedelta(minutes=5): return np.nan
    return btc.iloc[p]["close"]

rows=[]
for d,row in vix.iterrows():
    sc=row["slope_change"]
    if pd.isna(sc) or sc==0: continue
    # 당일 장중 09:30→15:59 (BIASED: 당일 VIX 종가로 당일 장중을 '예측')
    o=price_at(pd.Timestamp(d.year,d.month,d.day,9,30,tz=ET))
    c=price_at(pd.Timestamp(d.year,d.month,d.day,15,59,tz=ET))
    same = (c-o)/o if not (pd.isna(o) or pd.isna(c)) else np.nan
    # 익일 장중 (UNBIASED)
    base=pd.Timestamp(d.year,d.month,d.day,16,15,tz=ET)
    no=price_at(base+pd.Timedelta(minutes=1035)); ncl=price_at(base+pd.Timedelta(minutes=1424))
    nxt = (ncl-no)/no if not (pd.isna(no) or pd.isna(ncl)) else np.nan
    rows.append({"date":d,"sc":sc,"same":same,"next":nxt})

df=pd.DataFrame(rows).set_index("date").sort_index()
def hit(sub,col):
    v=sub[["sc",col]].dropna(); v=v[v["sc"]!=0]
    pred=np.sign(v["sc"]); act=np.sign(v[col])
    return (pred==act).mean()*100, len(v)

for name,a,b in [("IS","2024-01-01","2025-10-31"),("OOS","2025-11-01","2026-04-30"),("전체","2024-01-01","2026-04-30")]:
    sub=df[(df.index>=a)&(df.index<=b)]
    hb,nb=hit(sub,"same"); hu,nu=hit(sub,"next")
    print(f"{name:4s} | 미방지(당일) {hb:.1f}% (n={nb}) | 방지(익일) {hu:.1f}% (n={nu})")
