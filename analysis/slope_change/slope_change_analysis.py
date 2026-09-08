"""
VIX slope_change → BTC 방향 예측 분석 (T-day 시간축)
====================================================

시간축 정의 (Lookahead-free 구조의 명시적 표현):
  각 거래일을 "T일"로 인덱싱하고, T일의 시간 t는 0:00부터 시작한다.

  T일의 0:00  ≡  절대 시각 16:16 ET (VIX 종가 확정 직후 첫 BTC 측정)
  T일의 17:14 ≡  절대 시각 익일 09:30 ET (다음 미국 시장 개장)
  T일의 23:43 ≡  절대 시각 익일 15:59 ET (다음 미국 시장 마감 직전)

  예측 변수 slope_change(T) = slope[T] - slope[T-1]는
  T일 0:00 직전(절대 시각 T일 16:15)에 확정된다.

핵심 가설:
  slope_change(T) > 0 → BTC 상승 / slope_change(T) < 0 → BTC 하락

분석 구조:
  1. T-day 0:00 ~ 17:14 (야간 윈도우) 분 단위 적중률
  2. 다음 장중 (T-day 17:14 → 23:43) 적중률 — slope_change의 본래 검증 가설
  3. |slope_change| 크기별 적중률 분포
"""
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 저장소 루트 (data/ 가 있는 곳) — 실행 위치와 무관하게 파일 위치로부터 계산
BASE_DIR = str(Path(__file__).resolve().parents[2])
OUT_DIR = str(Path(__file__).resolve().parent)   # analysis/slope_change/ — 산출물(data/ · charts/) 위치
ET = "America/New_York"

# ─────────────────── 1. 데이터 로드 ───────────────────
vix_slope = pd.read_parquet(f"{BASE_DIR}/data/vix_slope_daily.parquet")
vix_slope.index = pd.DatetimeIndex(vix_slope.index)
vix_slope = vix_slope.sort_index()
# slope_change 계산
vix_slope["slope_change"] = vix_slope["slope"].diff()

print(f"VIX slope data: {vix_slope.index.min().date()} ~ {vix_slope.index.max().date()}, n={len(vix_slope)}")

btc_dir = f"{BASE_DIR}/data/btc_1m_24h/"
files = sorted(f for f in os.listdir(btc_dir) if f.endswith(".parquet"))
btc = pd.concat([pd.read_parquet(btc_dir + f)[["close"]] for f in files]).sort_index()
btc = btc[~btc.index.duplicated(keep="first")]
if btc.index.tz is None:
    btc.index = btc.index.tz_localize(ET)
else:
    btc.index = btc.index.tz_convert(ET)

btc_idx = btc.index

def price_at(ts):
    pos = btc_idx.searchsorted(ts)
    if pos >= len(btc_idx):
        return np.nan
    found = btc_idx[pos]
    if (found - ts) > pd.Timedelta(minutes=5):
        return np.nan
    return btc.iloc[pos]["close"]

# ─────────────────── 2. T-day 시간축 ───────────────────
VIX_CLOSE_HOUR = 16
VIX_CLOSE_MIN = 15  # base 절대 시각 16:15 (T-day 0:00 직전)

# T-day 0:00 ~ 17:14 (야간 윈도우, vix_duration과 동일)
NMIN_OVERNIGHT = 1035

# 다음 장중 종료 시점: T-day 23:43 (= 익일 15:59)
NEXT_INTRADAY_END_MIN = 1424

print(f"\n[Time axis] Each T-day:")
print(f"  T-day 0:00  ≡ absolute 16:16 ET (slope_change(T) just confirmed)")
print(f"  T-day 17:14 ≡ absolute next-day 09:30 ET (next US market open)")
print(f"  T-day 23:43 ≡ absolute next-day 15:59 ET (next US market close)")
print()

# ─────────────────── 3. 각 T-day 응답 계산 ───────────────────
print("Computing T-day responses...")
records = []
for d, row in vix_slope.iterrows():
    sc = row["slope_change"]
    if pd.isna(sc):
        continue
    base_ts = pd.Timestamp(d.year, d.month, d.day, VIX_CLOSE_HOUR, VIX_CLOSE_MIN, tz=ET)
    p0 = price_at(base_ts)
    if pd.isna(p0):
        continue
    rec = {"date": d, "slope": row["slope"], "slope_change": sc, "sc_abs": abs(sc)}

    # T-day 0:00 ~ 17:14 분 단위 누적 수익률
    for m in range(1, NMIN_OVERNIGHT + 1):
        pt = price_at(base_ts + pd.Timedelta(minutes=m))
        rec[f"t{m-1}"] = (pt - p0) / p0 if not pd.isna(pt) else np.nan

    # 다음 장중 BTC: T-day 17:14 (= 다음 09:30) → T-day 23:43 (= 다음 15:59)
    p_next_open = price_at(base_ts + pd.Timedelta(minutes=NMIN_OVERNIGHT))
    p_next_close = price_at(base_ts + pd.Timedelta(minutes=NEXT_INTRADAY_END_MIN))
    if not (pd.isna(p_next_open) or pd.isna(p_next_close)):
        rec["next_intraday"] = (p_next_close - p_next_open) / p_next_open
    else:
        rec["next_intraday"] = np.nan
    records.append(rec)

df = pd.DataFrame(records).set_index("date").sort_index()

# ─────────────────── 4. 구간 분리 ───────────────────
# vix_slope_daily는 2024-01부터 시작하므로 IS/OOS만 사용
IS_START, IS_END = "2024-01-01", "2025-10-31"
OOS_START, OOS_END = "2025-11-01", "2026-04-30"
ALL_START, ALL_END = "2024-01-01", "2026-04-30"

is_df = df[(df.index >= IS_START) & (df.index <= IS_END)]
oos_df = df[(df.index >= OOS_START) & (df.index <= OOS_END)]
all_df = df[(df.index >= ALL_START) & (df.index <= ALL_END)]

print(f"\nIS  (2024-01~2025-10): {len(is_df)} T-days")
print(f"OOS (2025-11~2026-04): {len(oos_df)} T-days")
print(f"ALL (2024-01~2026-04): {len(all_df)} T-days")

# ─────────────────── 5. 분 단위 적중률·상관 ───────────────────
def compute_metrics(subset, tday_min):
    col = f"t{tday_min}"
    valid = subset[["slope_change", col]].dropna()
    # slope_change == 0 제외
    valid = valid[valid["slope_change"] != 0]
    if len(valid) < 20:
        return np.nan, np.nan, np.nan, len(valid)
    r, p = stats.pearsonr(valid["slope_change"], valid[col])
    # 가설: slope_change > 0 → BTC 상승 → 예측 부호 = sign(slope_change)
    pred = np.sign(valid["slope_change"])
    hit = (np.sign(valid[col]) == pred).mean() * 100
    return r, p, hit, len(valid)

print("\nComputing minute-by-minute metrics across T-day window...")
minute_data = []
for tday_min in range(0, NMIN_OVERNIGHT):
    r_is, p_is, hit_is, n_is = compute_metrics(is_df, tday_min)
    r_oos, p_oos, hit_oos, n_oos = compute_metrics(oos_df, tday_min)
    r_all, p_all, hit_all, n_all = compute_metrics(all_df, tday_min)
    minute_data.append({
        "tday_min": tday_min,
        "tday_hour": tday_min / 60,
        "tday_clock": f"{tday_min//60:02d}:{tday_min%60:02d}",
        "r_is": r_is, "p_is": p_is, "hit_is": hit_is, "n_is": n_is,
        "r_oos": r_oos, "p_oos": p_oos, "hit_oos": hit_oos, "n_oos": n_oos,
        "r_all": r_all, "p_all": p_all, "hit_all": hit_all, "n_all": n_all,
    })

mres = pd.DataFrame(minute_data)
mres.to_csv(f"{OUT_DIR}/data/slope_change_minute_metrics.csv", index=False)
print(f"Saved: slope_change_minute_metrics.csv")

# ─────────────────── 6. 핵심 시점별 적중률 ───────────────────
def directional_test(subset, col_name):
    """slope_change 부호 기반 적중률 측정. col_name으로 BTC 측정 컬럼 지정."""
    valid = subset[["slope_change", col_name]].dropna()
    valid = valid[valid["slope_change"] != 0]
    if len(valid) < 5:
        return None
    pred = np.sign(valid["slope_change"])
    actual = np.sign(valid[col_name])
    hit = (pred == actual).mean() * 100
    n = len(valid)
    # Binomial test (H0: p = 0.5)
    correct = (pred == actual).sum()
    p_val = stats.binomtest(correct, n, 0.5).pvalue
    # Long/Short 분리
    long_mask = pred == 1
    short_mask = pred == -1
    long_hit = (actual[long_mask] == 1).mean() * 100 if long_mask.sum() > 0 else np.nan
    short_hit = (actual[short_mask] == -1).mean() * 100 if short_mask.sum() > 0 else np.nan
    return {
        "n": n,
        "hit": hit,
        "binom_p": p_val,
        "n_long": int(long_mask.sum()),
        "n_short": int(short_mask.sum()),
        "long_hit": long_hit,
        "short_hit": short_hit,
    }

print("\n" + "=" * 70)
print("RESULTS — Directional Accuracy")
print("=" * 70)

print("\n## 1. 다음 장중 적중률 (T-day 17:14 → 23:43, slope_change 본래 가설)")
print(f"{'구간':<6s} | {'n':>4s} | {'적중률':>7s} | {'binom p':>8s} | {'Long n':>6s} | {'Long%':>6s} | {'Short n':>7s} | {'Short%':>7s}")
results_next = {}
for name, sub in [("IS", is_df), ("OOS", oos_df), ("ALL", all_df)]:
    r = directional_test(sub, "next_intraday")
    results_next[name] = r
    if r:
        print(f"{name:<6s} | {r['n']:>4d} | {r['hit']:>6.1f}% | {r['binom_p']:>8.3f} | {r['n_long']:>6d} | {r['long_hit']:>5.1f}% | {r['n_short']:>7d} | {r['short_hit']:>6.1f}%")

# T-day 윈도우 내 핵심 시점들
key_times_min = [15, 30, 60, 120, 240, 480, 720, 1034]  # 15분, 30분, 1h, 2h, 4h, 8h, 12h, 17:14
print("\n## 2. T-day 윈도우 핵심 시점 적중률 (ALL 구간, 2024-01~2026-04)")
print(f"{'T-day':>8s} | {'n':>4s} | {'적중률':>7s} | {'binom p':>8s} | {'r':>8s}")
key_results = []
for tm in key_times_min:
    r = directional_test(all_df, f"t{tm}")
    valid = all_df[["slope_change", f"t{tm}"]].dropna()
    valid = valid[valid["slope_change"] != 0]
    rp, pp = stats.pearsonr(valid["slope_change"], valid[f"t{tm}"]) if len(valid) > 20 else (np.nan, np.nan)
    clock = f"{tm//60:02d}:{tm%60:02d}"
    if r:
        key_results.append({"tday_clock": clock, "tday_min": tm, **r, "r_pearson": rp, "p_pearson": pp})
        print(f"  {clock:>6s} | {r['n']:>4d} | {r['hit']:>6.1f}% | {r['binom_p']:>8.3f} | {rp:+.4f}")

# ─────────────────── 7. |slope_change| 크기별 적중률 ───────────────────
print("\n## 3. |slope_change| 크기별 적중률 (다음 장중)")

bins = [0, 0.3, 0.5, 1.0, 1.5, np.inf]
bin_labels = ["<0.3", "0.3~0.5", "0.5~1.0", "1.0~1.5", ">=1.5"]

magnitude_results = {}
for name, sub in [("IS", is_df), ("OOS", oos_df), ("ALL", all_df)]:
    valid = sub[["slope_change", "sc_abs", "next_intraday"]].dropna()
    valid = valid[valid["slope_change"] != 0]
    valid = valid.copy()
    valid["bin"] = pd.cut(valid["sc_abs"], bins=bins, labels=bin_labels)
    rows = []
    for label in bin_labels:
        b = valid[valid["bin"] == label]
        if len(b) == 0:
            rows.append({"bin": label, "n": 0, "hit": np.nan, "binom_p": np.nan})
            continue
        pred = np.sign(b["slope_change"])
        actual = np.sign(b["next_intraday"])
        hit = (pred == actual).mean() * 100
        correct = (pred == actual).sum()
        p_val = stats.binomtest(correct, len(b), 0.5).pvalue if len(b) > 0 else np.nan
        rows.append({"bin": label, "n": len(b), "hit": hit, "binom_p": p_val})
    magnitude_results[name] = rows
    print(f"\n[{name}]")
    print(f"{'|sc| 범위':<10s} | {'n':>4s} | {'적중률':>7s} | {'binom p':>8s}")
    for r in rows:
        if r["n"] > 0:
            print(f"  {r['bin']:<8s} | {r['n']:>4d} | {r['hit']:>6.1f}% | {r['binom_p']:>8.3f}")
        else:
            print(f"  {r['bin']:<8s} | {r['n']:>4d} | {'-':>7s} | {'-':>8s}")

# ─────────────────── 8. 차트 ───────────────────
print("\nGenerating charts...")

# Chart 1: 분 단위 적중률·상관 곡선
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
tday_hours = mres["tday_hour"].values

ax = axes[0]
ax.plot(tday_hours, mres["r_all"], color="#A93226", lw=1.4, label=f"ALL (n={len(all_df)})")
ax.plot(tday_hours, mres["r_is"], color="#5D7B9D", lw=1.0, alpha=0.7, label=f"IS (n={len(is_df)})")
ax.plot(tday_hours, mres["r_oos"], color="#C9A227", lw=1.0, alpha=0.7, label=f"OOS (n={len(oos_df)})")
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("Pearson r")
ax.set_title("slope_change → BTC: minute-by-minute correlation across T-day window")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(tday_hours, mres["hit_all"], color="#A93226", lw=1.4, label="ALL")
ax.plot(tday_hours, mres["hit_is"], color="#5D7B9D", lw=1.0, alpha=0.7, label="IS")
ax.plot(tday_hours, mres["hit_oos"], color="#C9A227", lw=1.0, alpha=0.7, label="OOS")
ax.axhline(50, color="black", lw=0.8, ls="--", label="50% (random baseline)")
ax.set_ylabel("Directional hit rate (%)")
ax.set_xlabel("T-day time (hours from 0:00)")
ax.set_xticks(np.arange(0, 18, 2))
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
chart1_path = f"{OUT_DIR}/charts/slope_change_minute_metrics.png"
plt.savefig(chart1_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {chart1_path}")

# Chart 2: |slope_change| 크기별 적중률
fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
x = np.arange(len(bin_labels))
width = 0.28

is_hits = [r["hit"] if r["n"] > 0 else 0 for r in magnitude_results["IS"]]
oos_hits = [r["hit"] if r["n"] > 0 else 0 for r in magnitude_results["OOS"]]
all_hits = [r["hit"] if r["n"] > 0 else 0 for r in magnitude_results["ALL"]]
is_ns = [r["n"] for r in magnitude_results["IS"]]
oos_ns = [r["n"] for r in magnitude_results["OOS"]]
all_ns = [r["n"] for r in magnitude_results["ALL"]]

ax.bar(x - width, is_hits, width, color="#5D7B9D", label="IS")
ax.bar(x, oos_hits, width, color="#C9A227", label="OOS")
ax.bar(x + width, all_hits, width, color="#A93226", label="ALL")
ax.axhline(50, color="black", lw=1.0, ls="--", label="50% (random)")

# 표본 수 표시
for i, (is_n, oos_n, all_n) in enumerate(zip(is_ns, oos_ns, all_ns)):
    if is_n > 0:
        ax.text(i - width, is_hits[i] + 1, f"n={is_n}", ha="center", fontsize=7, color="#5D7B9D")
    if oos_n > 0:
        ax.text(i, oos_hits[i] + 1, f"n={oos_n}", ha="center", fontsize=7, color="#C9A227")
    if all_n > 0:
        ax.text(i + width, all_hits[i] + 1, f"n={all_n}", ha="center", fontsize=7, color="#A93226")

ax.set_xticks(x)
ax.set_xticklabels(bin_labels)
ax.set_xlabel("|slope_change| range")
ax.set_ylabel("Directional hit rate (%)")
ax.set_title("Next-day intraday hit rate by |slope_change| magnitude")
ax.set_ylim(0, max(80, max(is_hits + oos_hits + all_hits) + 8))
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
chart2_path = f"{OUT_DIR}/charts/slope_change_magnitude_hit.png"
plt.savefig(chart2_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {chart2_path}")

# Chart 3: 다음 장중 수익률 분포 (slope_change > 0 vs < 0)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, sub) in zip(axes, [("IS", is_df), ("OOS", oos_df)]):
    valid = sub[["slope_change", "next_intraday"]].dropna()
    valid = valid[valid["slope_change"] != 0]
    pos = valid[valid["slope_change"] > 0]["next_intraday"] * 100
    neg = valid[valid["slope_change"] < 0]["next_intraday"] * 100
    ax.hist([pos, neg], bins=25, label=[f"sc>0 (n={len(pos)})", f"sc<0 (n={len(neg)})"],
            color=["#A93226", "#5D7B9D"], alpha=0.7)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axvline(pos.mean(), color="#A93226", lw=1.0, ls=":", label=f"sc>0 μ={pos.mean():+.2f}%")
    ax.axvline(neg.mean(), color="#5D7B9D", lw=1.0, ls=":", label=f"sc<0 μ={neg.mean():+.2f}%")
    ax.set_xlabel("Next-day intraday return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{name} — distribution of next-day return by slope_change sign")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
chart3_path = f"{OUT_DIR}/charts/slope_change_return_distribution.png"
plt.savefig(chart3_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {chart3_path}")

# ─────────────────── 9. JSON 요약 ───────────────────
summary = {
    "time_axis": {
        "definition": "T-day, 0:00 ≡ 16:16 ET (slope_change(T) confirmed just before)",
        "tday_0000_absolute": "16:16 ET",
        "tday_1714_absolute": "next-day 09:30 ET",
        "tday_2343_absolute": "next-day 15:59 ET",
    },
    "samples": {
        "IS": len(is_df),
        "OOS": len(oos_df),
        "ALL": len(all_df),
    },
    "next_intraday_directional": {
        name: results_next[name] for name in ["IS", "OOS", "ALL"] if results_next[name] is not None
    },
    "key_timepoints_all": key_results,
    "magnitude_bins": magnitude_results,
    "tday_window_avg": {
        "IS_avg_hit": float(mres["hit_is"].mean()),
        "OOS_avg_hit": float(mres["hit_oos"].mean()),
        "ALL_avg_hit": float(mres["hit_all"].mean()),
    },
}
with open(f"{OUT_DIR}/data/slope_change_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\nDONE")
