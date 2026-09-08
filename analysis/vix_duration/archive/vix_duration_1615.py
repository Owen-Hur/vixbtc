"""
VIX 종가(16:15 ET) → BTC 야간 지속성 분석 (수정판)
====================================================

⚠️ 중요 수정사항 (2026-06-02):
  VIX 종가 확정 시각이 16:00 ET가 아니라 16:15 ET임을 확인.
  기존 분석은 16:00부터 BTC를 측정했으므로 16:00~16:15 사이 15분간 lookahead bias 발생.
  본 스크립트는 base 시점을 16:15로 수정한 재분석.

핵심 질문:
  16:15에 확정된 VIX 종가가 그 이후 야간 BTC의 방향을 예측하는가?

윈도우 분해:
  A. Intraday(09:30~16:15): VIX↔BTC 동시 반응 (거래 불가, contemporaneous 벤치마크)
  B. 야간(16:15~익일 09:30, VIX 동결): 16:15 확정 VIX → 이후 BTC (예측적, 거래 후보)
"""
import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ────────────────── 1. 데이터 로드 ──────────────────
BASE_DIR = "/Users/macbook/btc_project"
vix = pd.read_parquet(f"{BASE_DIR}/data/vix_daily.parquet").copy()
vix.columns = ["vix"]
vidx = pd.to_datetime(vix.index)
try:
    vidx = vidx.tz_localize(None)
except (TypeError, AttributeError):
    pass
vix.index = pd.DatetimeIndex(vidx).normalize()
vix = vix[~vix.index.duplicated(keep="first")].sort_index()
vix["dvix"] = vix["vix"].diff()
vix["dvix_pct"] = vix["vix"].pct_change()

btc_dir = f"{BASE_DIR}/data/btc_1m_24h/"
files = sorted(f for f in os.listdir(btc_dir) if f.endswith(".parquet"))
btc = pd.concat([pd.read_parquet(btc_dir + f)[["close"]] for f in files]).sort_index()
btc = btc[~btc.index.duplicated(keep="first")]
if btc.index.tz is None:
    btc.index = btc.index.tz_localize("America/New_York")
else:
    btc.index = btc.index.tz_convert("America/New_York")

ET = "America/New_York"
btc_idx = btc.index

def price_at(ts):
    pos = btc_idx.searchsorted(ts)
    if pos >= len(btc_idx):
        return np.nan
    found = btc_idx[pos]
    if (found - ts) > pd.Timedelta(minutes=5):
        return np.nan
    return btc.iloc[pos]["close"]

# ────────────────── 2. 응답 테이블 ──────────────────
# 16:15부터 익일 09:30까지 = 17시간 15분 = 1035분
NMIN = 1035

# 호라이즌(분 단위)
night_offsets = {
    "15m": 15, "30m": 30, "1h": 60, "2h": 120, "3h": 180,
    "4h": 240, "6h": 360, "8h": 480, "12h": 720, "to_open": 1035,
}

rows = []
for d, r in vix.iterrows():
    if np.isnan(r["dvix"]):
        continue
    # ★ 핵심 변경: 16:00 → 16:15
    base_ts = pd.Timestamp(d.year, d.month, d.day, 16, 15, tz=ET)
    p0 = price_at(base_ts)
    if np.isnan(p0):
        continue
    rec = {"date": d, "vix": r["vix"], "dvix": r["dvix"], "dvix_pct": r["dvix_pct"]}
    # 윈도우 A: Intraday 09:30 → 16:15
    p_open = price_at(pd.Timestamp(d.year, d.month, d.day, 9, 30, tz=ET))
    rec["intraday"] = (p0 - p_open) / p_open if not np.isnan(p_open) else np.nan
    # 윈도우 B
    for name, mins in night_offsets.items():
        pt = price_at(base_ts + pd.Timedelta(minutes=mins))
        rec[f"n_{name}"] = (pt - p0) / p0 if not np.isnan(pt) else np.nan
    rows.append(rec)

df = pd.DataFrame(rows).set_index("date").sort_index()
df["year"] = df.index.year

# ────────────────── 3. 분 단위 상관 (1 ~ 1035분) ──────────────────
print("Computing minute-by-minute correlations (1 ~ 1035 min)...")
minute_records = []
for d, r in vix.iterrows():
    if np.isnan(r["dvix"]):
        continue
    base_ts = pd.Timestamp(d.year, d.month, d.day, 16, 15, tz=ET)
    p0 = price_at(base_ts)
    if np.isnan(p0):
        continue
    rec = {"date": d, "dvix_pct": r["dvix_pct"]}
    for m in range(1, NMIN + 1):
        pt = price_at(base_ts + pd.Timedelta(minutes=m))
        rec[f"m{m}"] = (pt - p0) / p0 if not np.isnan(pt) else np.nan
    minute_records.append(rec)

mdf = pd.DataFrame(minute_records).set_index("date").sort_index()
mdf["year"] = mdf.index.year

primary = mdf[(mdf.index >= "2024-01-01") & (mdf.index <= "2026-04-30")]
supp    = mdf[(mdf.index >= "2020-01-01") & (mdf.index <= "2023-12-31")]

print(f"Primary (2024-01~2026-04): {len(primary)} days")
print(f"Supplementary (2020-2023): {len(supp)} days")

minute_results = []
for m in range(1, NMIN + 1):
    col = f"m{m}"
    p_valid = primary[["dvix_pct", col]].dropna()
    if len(p_valid) > 20:
        r_p, p_p = stats.pearsonr(p_valid["dvix_pct"], p_valid[col])
        pred = -np.sign(p_valid["dvix_pct"])
        hit_p = (np.sign(p_valid[col]) == pred).mean() * 100
        n_p = len(p_valid)
    else:
        r_p, p_p, hit_p, n_p = np.nan, np.nan, np.nan, len(p_valid)

    s_valid = supp[["dvix_pct", col]].dropna()
    if len(s_valid) > 20:
        r_s, p_s = stats.pearsonr(s_valid["dvix_pct"], s_valid[col])
        pred = -np.sign(s_valid["dvix_pct"])
        hit_s = (np.sign(s_valid[col]) == pred).mean() * 100
        n_s = len(s_valid)
    else:
        r_s, p_s, hit_s, n_s = np.nan, np.nan, np.nan, len(s_valid)

    minute_results.append({
        "minute": m, "hours": m/60,
        "r_primary": r_p, "p_primary": p_p, "hit_primary": hit_p, "n_primary": n_p,
        "r_supp": r_s, "p_supp": p_s, "hit_supp": hit_s, "n_supp": n_s,
    })

mres = pd.DataFrame(minute_results)
mres.to_csv(f"{BASE_DIR}/vix_duration/data/vix_duration_1615_minute_corr.csv", index=False)

# ────────────────── 4. 호라이즌별 요약 ──────────────────
def corr_stats(a, b):
    m = pd.concat([a, b], axis=1).dropna()
    if len(m) < 20:
        return (np.nan, np.nan, len(m))
    r, p = stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])
    return (r, p, len(m))

primary_df = df[(df.index >= "2024-01-01") & (df.index <= "2026-04-30")]
supp_df    = df[(df.index >= "2020-01-01") & (df.index <= "2023-12-31")]

# Intraday 동시 벤치마크
rA_primary, pA_primary, nA_primary = corr_stats(primary_df["dvix_pct"], primary_df["intraday"])
rA_supp, pA_supp, nA_supp = corr_stats(supp_df["dvix_pct"], supp_df["intraday"])

print(f"\nIntraday concurrent (primary): r={rA_primary:+.4f}, p={pA_primary:.2e}, n={nA_primary}")
print(f"Intraday concurrent (supp):    r={rA_supp:+.4f}, p={pA_supp:.2e}, n={nA_supp}")

# 호라이즌별
horizon_results_p, horizon_results_s = [], []
COST = 0.0004
for name in night_offsets:
    col = f"n_{name}"

    rp, pp, np_ = corr_stats(primary_df["dvix_pct"], primary_df[col])
    mp = primary_df[["dvix_pct", col]].dropna()
    pred_p = -np.sign(mp["dvix_pct"])
    hit_p = (np.sign(mp[col]) == pred_p).mean() * 100
    gross_p = (pred_p * mp[col])
    net_p = gross_p - COST
    horizon_results_p.append({
        "horizon": name, "r": rp, "p": pp, "n": np_, "hit": hit_p,
        "gross_mean": gross_p.mean() * 100, "net_mean": net_p.mean() * 100,
        "cum_net": net_p.sum() * 100,
    })

    rs, ps, ns = corr_stats(supp_df["dvix_pct"], supp_df[col])
    ms = supp_df[["dvix_pct", col]].dropna()
    pred_s = -np.sign(ms["dvix_pct"])
    hit_s = (np.sign(ms[col]) == pred_s).mean() * 100
    gross_s = (pred_s * ms[col])
    net_s = gross_s - COST
    horizon_results_s.append({
        "horizon": name, "r": rs, "p": ps, "n": ns, "hit": hit_s,
        "gross_mean": gross_s.mean() * 100, "net_mean": net_s.mean() * 100,
        "cum_net": net_s.sum() * 100,
    })

print("\n=== Primary (2024-01~2026-04) — 16:15 기준 ===")
print(f"{'horizon':>8s} | {'r':>8s} | {'p':>8s} | {'hit%':>6s} | {'cum_net%':>9s}")
for h in horizon_results_p:
    print(f"{h['horizon']:>8s} | {h['r']:+.4f} | {h['p']:.2e} | {h['hit']:>5.1f}% | {h['cum_net']:+8.1f}%")

print("\n=== Supplementary (2020-2023) — 16:15 기준 ===")
print(f"{'horizon':>8s} | {'r':>8s} | {'p':>8s} | {'hit%':>6s} | {'cum_net%':>9s}")
for h in horizon_results_s:
    print(f"{h['horizon']:>8s} | {h['r']:+.4f} | {h['p']:.2e} | {h['hit']:>5.1f}% | {h['cum_net']:+8.1f}%")

# ────────────────── 5. 2σ 이벤트 ──────────────────
sd = primary_df["dvix_pct"].std()
mu = primary_df["dvix_pct"].mean()
spike = primary_df[primary_df["dvix_pct"] > mu + 2*sd]
drop = primary_df[primary_df["dvix_pct"] < mu - 2*sd]

print(f"\n2σ events (primary): spike={len(spike)}, drop={len(drop)}")
event_results = []
for name in night_offsets:
    col = f"n_{name}"
    s = spike[col].dropna()
    d2 = drop[col].dropna()
    ts_s = stats.ttest_1samp(s, 0) if len(s) > 2 else (np.nan, np.nan)
    ts_d = stats.ttest_1samp(d2, 0) if len(d2) > 2 else (np.nan, np.nan)
    event_results.append({
        "horizon": name,
        "spike_mean": s.mean() * 100, "spike_t": ts_s[0], "spike_p": ts_s[1],
        "drop_mean": d2.mean() * 100, "drop_t": ts_d[0], "drop_p": ts_d[1],
    })

# ────────────────── 6. 차트 ──────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

minutes = mres["minute"].values
hours = mres["hours"].values

ax = axes[0]
ax.plot(hours, mres["r_primary"], color="#A93226", lw=1.3, label=f"Primary 2024-2026 (n={len(primary)})")
ax.plot(hours, mres["r_supp"],    color="#7F8C8D", lw=1.0, alpha=0.7, label=f"Supp 2020-2023 (n={len(supp)})")
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("Pearson r")
ax.set_title("VIX 16:15 close change vs overnight BTC return — minute-by-minute correlation")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(hours, mres["hit_primary"], color="#A93226", lw=1.3, label="Primary 2024-2026")
ax.plot(hours, mres["hit_supp"],    color="#7F8C8D", lw=1.0, alpha=0.7, label="Supp 2020-2023")
ax.axhline(50, color="black", lw=0.8, ls="--", label="50% (random)")
ax.set_ylabel("Directional hit %")
ax.set_xlabel("Hours after 16:15 ET")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{BASE_DIR}/vix_duration/charts/vix_duration_1615_minute_corr.png", dpi=120, bbox_inches="tight")
print("\nChart saved: vix_duration_1615_minute_corr.png")

# ────────────────── 7. 결과 요약 저장 (JSON) ──────────────────
import json
summary = {
    "base_time": "16:15 ET",
    "n_primary": len(primary),
    "n_supp": len(supp),
    "intraday_concurrent": {
        "primary": {"r": rA_primary, "p": pA_primary, "n": nA_primary},
        "supp": {"r": rA_supp, "p": pA_supp, "n": nA_supp},
    },
    "horizon_primary": horizon_results_p,
    "horizon_supp": horizon_results_s,
    "events_2sigma_primary": event_results,
    "avg_hit_primary": float(mres["hit_primary"].mean()),
    "avg_hit_supp": float(mres["hit_supp"].mean()),
    "min_r_primary": {
        "minute": int(mres.loc[mres["r_primary"].idxmin(), "minute"]),
        "r": float(mres["r_primary"].min()),
    },
}
with open(f"{BASE_DIR}/vix_duration/data/vix_duration_1615_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nAvg hit rate (primary): {summary['avg_hit_primary']:.2f}%")
print(f"Avg hit rate (supp):    {summary['avg_hit_supp']:.2f}%")
print(f"Strongest negative r (primary): m{summary['min_r_primary']['minute']} → r={summary['min_r_primary']['r']:+.4f}")
print("\nDONE")
