"""
VIX → BTC Duration 분석 (최종 버전, T-day 시간축)
==================================================

시간축 정의 (Lookahead-free 구조의 명시적 표현):
  각 거래일을 "T일"로 인덱싱하고, T일의 시간 t는 0:00부터 시작한다.

  T일의 0:00  ≡  절대 시각 16:16 ET (VIX 종가 확정 직후 첫 BTC 측정)
  T일의 17:14 ≡  익일 09:30 ET (다음 미국 시장 개장 직전)

  예시:
    2024-01-02 16:16 ET = T=0의 0:00
    2024-01-02 17:16 ET = T=0의 1:00
    2024-01-03 09:30 ET = T=0의 17:14
    2024-01-03 16:16 ET = T=1의 0:00

  이 시간축에서 ΔVIX%(T)는 T일의 0:00 직전에 확정되고,
  BTC[T, t]는 t > 0:00에서만 관측된다.
  따라서 lookahead bias가 시간축 정의로부터 구조적으로 불가능하다.

핵심 질문:
  "VIX 변화 정보가 BTC에 얼마나 오래 남아 방향 예측에 쓸 수 있는가?"
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ────────────────── 1. 데이터 로드 ──────────────────
# 저장소 루트 (data/ 가 있는 곳) — 실행 위치와 무관하게 파일 위치로부터 계산
BASE_DIR = str(Path(__file__).resolve().parents[2])
OUT_DIR = str(Path(__file__).resolve().parent)   # analysis/vix_duration/ — 산출물(data/ · charts/) 위치
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

# ────────────────── 2. T-day 시간축 구성 ──────────────────
# Base 시점 (절대 시각 16:15 ET) ← BTC base_price를 잡는 곳, 측정은 그 이후부터
VIX_CLOSE_HOUR = 16
VIX_CLOSE_MIN = 15

# T일의 t:mm 분 단위로 측정 (T일의 0:00 ~ 17:14, 총 1035개 시점)
# 측정 인덱스 m=1, 2, ..., 1035 (base 후 m분)
#   m=1 ≡ T일 0:00 (= 절대 16:16)
#   m=1035 ≡ T일 17:14 (= 절대 익일 09:30)
NMIN = 1035

print(f"[Time axis] Each trading day T:")
print(f"  T-day 0:00  ≡ absolute 16:16 ET (first BTC measurement after VIX close)")
print(f"  T-day 17:14 ≡ absolute next-day 09:30 ET (US market reopen)")
print(f"  Total minutes measured: {NMIN}")
print()

# ────────────────── 3. 각 거래일 응답 데이터 ──────────────────
print("Computing T-day responses...")
records = []
for d, r in vix.iterrows():
    if np.isnan(r["dvix"]):
        continue
    base_ts = pd.Timestamp(d.year, d.month, d.day, VIX_CLOSE_HOUR, VIX_CLOSE_MIN, tz=ET)
    p0 = price_at(base_ts)  # 16:15 close (T-day 0:00 직전)
    if np.isnan(p0):
        continue
    rec = {"date": d, "dvix_pct": r["dvix_pct"]}
    # 동시 벤치마크: 그 날 09:30 → 16:15 (T-day 시작 전, 거래 불가)
    p_open = price_at(pd.Timestamp(d.year, d.month, d.day, 9, 30, tz=ET))
    rec["intraday"] = (p0 - p_open) / p_open if not np.isnan(p_open) else np.nan
    # T-day의 각 시점 t = 0:00 ~ 17:14 (분 단위)
    for m in range(1, NMIN + 1):
        # T-day 분 = m - 1 (m=1이 T-day 0:00이므로)
        # 절대 시각 = base_ts + m분 (= 16:15 + m분)
        pt = price_at(base_ts + pd.Timedelta(minutes=m))
        # 컬럼명을 T-day 분 단위로 정의 (0 ~ 1034)
        tday_min = m - 1
        rec[f"t{tday_min}"] = (pt - p0) / p0 if not np.isnan(pt) else np.nan
    records.append(rec)

df = pd.DataFrame(records).set_index("date").sort_index()
df["year"] = df.index.year

primary = df[(df.index >= "2024-01-01") & (df.index <= "2026-04-30")]
supp    = df[(df.index >= "2020-01-01") & (df.index <= "2023-12-31")]

print(f"Primary (2024-01~2026-04): {len(primary)} T-days")
print(f"Supplementary (2020-2023): {len(supp)} T-days")
print()

# ────────────────── 4. 분 단위 상관·적중률 ──────────────────
def compute_metrics(subset, tday_min):
    col = f"t{tday_min}"
    valid = subset[["dvix_pct", col]].dropna()
    if len(valid) < 20:
        return np.nan, np.nan, np.nan, len(valid)
    r, p = stats.pearsonr(valid["dvix_pct"], valid[col])
    pred = -np.sign(valid["dvix_pct"])
    hit = (np.sign(valid[col]) == pred).mean() * 100
    return r, p, hit, len(valid)

print("Computing minute-by-minute metrics across T-day...")
minute_data = []
for tday_min in range(0, NMIN):
    r_p, p_p, hit_p, n_p = compute_metrics(primary, tday_min)
    r_s, p_s, hit_s, n_s = compute_metrics(supp, tday_min)
    minute_data.append({
        "tday_min": tday_min,
        "tday_hour": tday_min / 60,
        "tday_clock": f"{tday_min//60:02d}:{tday_min%60:02d}",
        "r_primary": r_p, "p_primary": p_p, "hit_primary": hit_p, "n_primary": n_p,
        "r_supp": r_s, "p_supp": p_s, "hit_supp": hit_s, "n_supp": n_s,
    })

mres = pd.DataFrame(minute_data)
mres.to_csv(f"{OUT_DIR}/data/vix_duration_minute_metrics.csv", index=False)

# ────────────────── 5. 핵심 통계 출력 ──────────────────
def corr_stats(a, b):
    m = pd.concat([a, b], axis=1).dropna()
    if len(m) < 20:
        return (np.nan, np.nan, len(m))
    r, p = stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])
    return (r, p, len(m))

rA_p, pA_p, nA_p = corr_stats(primary["dvix_pct"], primary["intraday"])
rA_s, pA_s, nA_s = corr_stats(supp["dvix_pct"], supp["intraday"])

print("=" * 60)
print("RESULTS")
print("=" * 60)
print()
print("Concurrent benchmark (same day 09:30 → T-day 0:00; non-tradable):")
print(f"  Primary (2024-2026):       r = {rA_p:+.4f}, p = {pA_p:.2e}, n = {nA_p}")
print(f"  Supplementary (2020-2023): r = {rA_s:+.4f}, p = {pA_s:.2e}, n = {nA_s}")
print()

print(f"T-day window (t = 0:00 to 17:14, lookahead-free):")
print(f"  Primary avg hit rate:       {mres['hit_primary'].mean():.2f}%")
print(f"  Supplementary avg hit rate: {mres['hit_supp'].mean():.2f}%")
print()

idx_min_p = mres["r_primary"].idxmin()
idx_min_s = mres["r_supp"].idxmin()
print(f"  Strongest negative r (primary):")
print(f"    T-day time {mres.loc[idx_min_p, 'tday_clock']} (={mres.loc[idx_min_p, 'tday_hour']:.2f} h after VIX close)")
print(f"    r = {mres.loc[idx_min_p, 'r_primary']:+.4f}, p = {mres.loc[idx_min_p, 'p_primary']:.3f}")
print()
print(f"  Strongest negative r (supp):")
print(f"    T-day time {mres.loc[idx_min_s, 'tday_clock']} (={mres.loc[idx_min_s, 'tday_hour']:.2f} h after VIX close)")
print(f"    r = {mres.loc[idx_min_s, 'r_supp']:+.4f}, p = {mres.loc[idx_min_s, 'p_supp']:.3e}")
print()

sig_p = (mres["p_primary"] < 0.05).sum()
sig_s = (mres["p_supp"] < 0.05).sum()
print(f"  Minutes with p<0.05 (primary): {sig_p}/{NMIN} ({sig_p/NMIN*100:.1f}%)")
print(f"  Minutes with p<0.05 (supp):    {sig_s}/{NMIN} ({sig_s/NMIN*100:.1f}%)")
print()

# 주요 시점 표
print("Key T-day timepoints (primary):")
key_times = [15, 30, 60, 120, 240, 480, 720, 1034]  # 15분, 30분, 1h, 2h, 4h, 8h, 12h, 17:14
print(f"  {'T-day':>8s} | {'r':>8s} | {'p':>8s} | {'hit%':>6s}")
for tm in key_times:
    row = mres[mres["tday_min"] == tm].iloc[0]
    print(f"  {row['tday_clock']:>8s} | {row['r_primary']:+.4f} | {row['p_primary']:.2e} | {row['hit_primary']:>5.1f}%")
print()

# ────────────────── 6. 2σ 이벤트 ──────────────────
sd = primary["dvix_pct"].std()
mu = primary["dvix_pct"].mean()
spike = primary[primary["dvix_pct"] > mu + 2*sd]
drop = primary[primary["dvix_pct"] < mu - 2*sd]
print(f"2σ events (primary): spike={len(spike)}, drop={len(drop)}")

spike_path, drop_path = [], []
for tday_min in range(0, NMIN):
    col = f"t{tday_min}"
    s = spike[col].dropna()
    d2 = drop[col].dropna()
    spike_path.append(s.mean() * 100 if len(s) > 0 else np.nan)
    drop_path.append(d2.mean() * 100 if len(d2) > 0 else np.nan)

print("\n2σ event direction tests (primary):")
print(f"  {'T-day':>8s} | {'Spike μ':>9s} | {'p(spike)':>9s} | {'Drop μ':>9s} | {'p(drop)':>9s}")
for tm in [15, 60, 240, 480, 1034]:
    col = f"t{tm}"
    s = spike[col].dropna()
    d2 = drop[col].dropna()
    if len(s) > 2 and len(d2) > 2:
        ts_s = stats.ttest_1samp(s, 0)
        ts_d = stats.ttest_1samp(d2, 0)
        clock = f"{tm//60:02d}:{tm%60:02d}"
        print(f"  {clock:>8s} | {s.mean()*100:+8.3f}% | {ts_s[1]:>9.3f} | {d2.mean()*100:+8.3f}% | {ts_d[1]:>9.3f}")
print()

# ────────────────── 7. 차트 ──────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

tday_hours = mres["tday_hour"].values

ax = axes[0]
ax.plot(tday_hours, mres["r_primary"], color="#A93226", lw=1.4, label=f"Primary 2024-2026 (n={len(primary)})")
ax.plot(tday_hours, mres["r_supp"], color="#7F8C8D", lw=1.0, alpha=0.7, label=f"Supp 2020-2023 (n={len(supp)})")
ax.axhline(0, color="black", lw=0.5)
ax.axhline(rA_p, color="#A93226", lw=0.8, ls=":", alpha=0.6,
           label=f"Concurrent benchmark r={rA_p:+.3f} (non-tradable)")
ax.set_ylabel("Pearson r")
ax.set_title("VIX → BTC duration: minute-by-minute correlation across T-day")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(tday_hours, mres["hit_primary"], color="#A93226", lw=1.4, label="Primary 2024-2026")
ax.plot(tday_hours, mres["hit_supp"], color="#7F8C8D", lw=1.0, alpha=0.7, label="Supp 2020-2023")
ax.axhline(50, color="black", lw=0.8, ls="--", label="50% (random baseline)")
ax.set_ylabel("Directional hit rate (%)")
ax.set_xlabel("T-day time (hours from 0:00 ≡ VIX close)")
ax.set_xticks(np.arange(0, 18, 2))
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
chart_path = f"{OUT_DIR}/charts/vix_duration_minute_metrics.png"
plt.savefig(chart_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {chart_path}")

# 2σ 이벤트 경로
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(tday_hours, spike_path, color="#A93226", lw=1.4, label=f"After 2σ VIX Spike (n={len(spike)})")
ax.plot(tday_hours, drop_path, color="#2E86AB", lw=1.4, label=f"After 2σ VIX Drop (n={len(drop)})")
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("BTC mean cumulative return (%)")
ax.set_xlabel("T-day time (hours from 0:00)")
ax.set_xticks(np.arange(0, 18, 2))
ax.set_title("2σ event paths — BTC response across T-day (primary 2024-2026)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
chart_path2 = f"{OUT_DIR}/charts/vix_duration_2sigma_events.png"
plt.savefig(chart_path2, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {chart_path2}")

# ────────────────── 8. JSON 요약 ──────────────────
import json
summary = {
    "time_axis": {
        "definition": "T-day, 0:00 ≡ 16:16 ET (first BTC measurement after VIX close confirmation)",
        "tday_0000_absolute": "16:16 ET",
        "tday_1714_absolute": "next-day 09:30 ET",
        "total_minutes_per_tday": NMIN,
    },
    "n_primary": len(primary),
    "n_supp": len(supp),
    "concurrent_benchmark": {
        "primary": {"r": rA_p, "p": pA_p, "n": int(nA_p)},
        "supp": {"r": rA_s, "p": pA_s, "n": int(nA_s)},
    },
    "tday_summary": {
        "primary_avg_hit": float(mres["hit_primary"].mean()),
        "supp_avg_hit": float(mres["hit_supp"].mean()),
        "primary_min_r": {
            "tday_clock": mres.loc[idx_min_p, "tday_clock"],
            "tday_hour": float(mres.loc[idx_min_p, "tday_hour"]),
            "r": float(mres.loc[idx_min_p, "r_primary"]),
            "p": float(mres.loc[idx_min_p, "p_primary"]),
        },
        "supp_min_r": {
            "tday_clock": mres.loc[idx_min_s, "tday_clock"],
            "tday_hour": float(mres.loc[idx_min_s, "tday_hour"]),
            "r": float(mres.loc[idx_min_s, "r_supp"]),
            "p": float(mres.loc[idx_min_s, "p_supp"]),
        },
        "sig_minutes_primary": int(sig_p),
        "sig_minutes_supp": int(sig_s),
        "total_minutes": NMIN,
    },
    "two_sigma_events_primary": {
        "n_spike": len(spike),
        "n_drop": len(drop),
    },
}
with open(f"{OUT_DIR}/data/vix_duration_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print()
print("DONE")
