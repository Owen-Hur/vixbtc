# -*- coding: utf-8 -*-
"""전체 기간 누적 방향 적중률 (IS/OOS 색 구분) — 덱 팔레트"""
import pandas as pd, numpy as np, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.rcParams['font.family']='AppleGothic'
matplotlib.rcParams['axes.unicode_minus']=False

# 저장소 루트 (data/ 가 있는 곳) — 실행 위치와 무관하게 파일 위치로부터 계산
BASE=str(Path(__file__).resolve().parents[2]); ET="America/New_York"
HERE=str(Path(__file__).resolve().parent)   # analysis/slope_change/
NAVY="#122B46"; STEEL="#7C97B3"; RED="#A93226"; MUTE="#6B7280"

vix=pd.read_parquet(f"{BASE}/data/vix_slope_daily.parquet")
vix.index=pd.DatetimeIndex(vix.index); vix=vix.sort_index(); vix["sc"]=vix["slope"].diff()
btc_dir=f"{BASE}/data/btc_1m_24h/"
files=sorted(f for f in os.listdir(btc_dir) if f.endswith(".parquet"))
btc=pd.concat([pd.read_parquet(btc_dir+f)[["close"]] for f in files]).sort_index()
btc=btc[~btc.index.duplicated(keep="first")]
btc.index=btc.index.tz_localize(ET) if btc.index.tz is None else btc.index.tz_convert(ET)
bi=btc.index
def price_at(ts):
    p=bi.searchsorted(ts)
    if p>=len(bi): return np.nan
    if (bi[p]-ts)>pd.Timedelta(minutes=5): return np.nan
    return btc.iloc[p]["close"]
rows=[]
for d,row in vix.iterrows():
    sc=row["sc"]
    if pd.isna(sc) or sc==0: continue
    base=pd.Timestamp(d.year,d.month,d.day,16,15,tz=ET)
    o=price_at(base+pd.Timedelta(minutes=1035)); c=price_at(base+pd.Timedelta(minutes=1424))
    if pd.isna(o) or pd.isna(c): continue
    rows.append({"date":d,"hit":1 if np.sign(sc)==np.sign((c-o)/o) else 0})
df=pd.DataFrame(rows).set_index("date").sort_index()
df=df[(df.index>="2024-01-01")&(df.index<="2026-04-30")]
df["cum"]=df["hit"].expanding().mean()*100
final=df["cum"].iloc[-1]

oos_start=pd.Timestamp("2025-11-01")
is_seg =df[df.index< oos_start]
oos_seg=df[df.index>=oos_start]
# 경계 연결: OOS 세그먼트 앞에 IS 마지막 점 추가
oos_plot=pd.concat([is_seg.iloc[[-1]], oos_seg])

fig,ax=plt.subplots(1,1,figsize=(8.2,3.5))
ax.axhline(50, color=RED, lw=1.3, ls="--", label="50% (무작위 기준)", zorder=2)
ax.plot(is_seg.index,  is_seg["cum"],  color=STEEL, lw=1.8, label="누적 적중률 · IS", zorder=3)
ax.plot(oos_plot.index, oos_plot["cum"], color=NAVY, lw=2.6, label="누적 적중률 · OOS", zorder=4)
ax.axvline(oos_start, color=MUTE, lw=1.0, ls=":", zorder=1)
ax.text(oos_start, 33.6, " OOS", fontsize=9, color=MUTE, va="bottom", ha="left")
ax.text(oos_start, 33.6, "IS ", fontsize=9, color=MUTE, va="bottom", ha="right")
ax.annotate(f"전 기간 {final:.1f}%", xy=(df.index[-1], final),
            xytext=(-6,9), textcoords="offset points",
            fontsize=10.5, color=NAVY, fontweight="bold", ha="right")

ax.set_ylabel("방향 적중률 (%)", fontsize=10, color=NAVY)
ax.set_ylim(33,66)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.legend(loc="upper right", fontsize=8.8, ncol=1, framealpha=0.92)
ax.grid(alpha=0.25)
ax.tick_params(colors=NAVY, labelsize=8)
for sp in ax.spines.values(): sp.set_color("#CCCCCC")
plt.tight_layout()
os.makedirs(f"{BASE}/_slide_assets", exist_ok=True)
out=f"{BASE}/_slide_assets/slide_period_hit.png"
plt.savefig(out,dpi=200,bbox_inches="tight")
plt.savefig(f"{HERE}/charts/slope_change_period_hit.png",dpi=200,bbox_inches="tight")
print(f"n={len(df)}, final={final:.1f}%, saved {out}")
