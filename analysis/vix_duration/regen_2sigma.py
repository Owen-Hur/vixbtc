"""2σ 이벤트 경로 차트 — 덱 팔레트로 재생성 (raw 재계산)"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/macbook/btc_project"
ET = "America/New_York"
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

NAVY = "#122B46"; STEEL = "#5B7B9A"; GOLD = "#C99A2E"; RED = "#A93226"

vix = pd.read_parquet(f"{BASE}/data/vix_daily.parquet").copy()
vix.columns = ["vix"]
vidx = pd.to_datetime(vix.index)
try: vidx = vidx.tz_localize(None)
except (TypeError, AttributeError): pass
vix.index = pd.DatetimeIndex(vidx).normalize()
vix = vix[~vix.index.duplicated(keep="first")].sort_index()
vix["dvix_pct"] = vix["vix"].pct_change()

btc_dir = f"{BASE}/data/btc_1m_24h/"
files = sorted(f for f in os.listdir(btc_dir) if f.endswith(".parquet"))
btc = pd.concat([pd.read_parquet(btc_dir + f)[["close"]] for f in files]).sort_index()
btc = btc[~btc.index.duplicated(keep="first")]
if btc.index.tz is None: btc.index = btc.index.tz_localize(ET)
else: btc.index = btc.index.tz_convert(ET)
btc_idx = btc.index

def price_at(ts):
    pos = btc_idx.searchsorted(ts)
    if pos >= len(btc_idx): return np.nan
    found = btc_idx[pos]
    if (found - ts) > pd.Timedelta(minutes=5): return np.nan
    return btc.iloc[pos]["close"]

NMIN = 1035
records = []
for d, row in vix.iterrows():
    if np.isnan(row["dvix_pct"]): continue
    base = pd.Timestamp(d.year, d.month, d.day, 16, 15, tz=ET)
    p0 = price_at(base)
    if np.isnan(p0): continue
    rec = {"date": d, "dvix_pct": row["dvix_pct"]}
    for m in range(1, NMIN + 1):
        pt = price_at(base + pd.Timedelta(minutes=m))
        rec[f"t{m-1}"] = (pt - p0)/p0 if not np.isnan(pt) else np.nan
    records.append(rec)

df = pd.DataFrame(records).set_index("date").sort_index()
primary = df[(df.index >= "2024-01-01") & (df.index <= "2026-04-30")]

sd = primary["dvix_pct"].std(); mu = primary["dvix_pct"].mean()
spike = primary[primary["dvix_pct"] > mu + 2*sd]
drop = primary[primary["dvix_pct"] < mu - 2*sd]
print(f"spike={len(spike)}, drop={len(drop)}")

hours = np.arange(0, NMIN)/60
spike_path = [spike[f"t{m}"].dropna().mean()*100 for m in range(NMIN)]
drop_path = [drop[f"t{m}"].dropna().mean()*100 for m in range(NMIN)]

fig, ax = plt.subplots(1, 1, figsize=(7.6, 3.5))
ax.plot(hours, spike_path, color=RED, lw=1.8, label=f"VIX 급등 후 (n={len(spike)})")
ax.plot(hours, drop_path, color=NAVY, lw=1.8, label=f"VIX 급락 후 (n={len(drop)})")
ax.axhline(0, color="#555555", lw=0.8)
ax.set_ylabel("BTC 평균 누적수익률 (%)", fontsize=10, color=NAVY)
ax.set_xlabel("VIX 확정 후 경과 시간 (시간)", fontsize=10, color=NAVY)
ax.set_xlim(0, 17.2)
ax.set_xticks(np.arange(0, 18, 2))
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax.grid(alpha=0.25)
ax.tick_params(colors=NAVY, labelsize=8)
for sp in ax.spines.values(): sp.set_color("#CCCCCC")
plt.tight_layout()
plt.savefig(f"{BASE}/vix_duration/charts/slide_2sigma.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved slide_2sigma.png")
