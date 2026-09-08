"""
VIX 종가(16:00 ET) → BTC 야간 지속성 분석 (P0)
================================================

핵심 질문:
  미국 주식/옵션 시장 마감(16:00 ET) 후 VIX는 동결된다. 그러나 BTC는 24시간 거래된다.
  16:00에 '확정된' VIX 종가/일중 변화가, 그 이후 야간(VIX 동결 구간) BTC 움직임을
  분 단위로 얼마나 오래 '예측'하는가?  (= lookahead 없는 예측적 관계)

윈도우 분해:
  A. Intraday(09:30~16:00): VIX↔BTC 동시 반응 (거래 불가, contemporaneous 벤치마크)
  B. 야간(16:00~익일 09:30, VIX 동결): 16:00 확정 VIX → 이후 BTC 누적수익률 (예측적, 거래 가능 후보)

산출물:
  docs/vix_overnight_persistence_report.md
  vix_overnight_persistence.png
"""
import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = []
def log(s=""):
    REPORT.append(str(s))

# ───────────────────────── 1. 데이터 로드 ─────────────────────────
vix = pd.read_parquet("data/vix_daily.parquet").copy()
vix.columns = ["vix"]
# 인덱스를 tz-naive 날짜로 정규화
vidx = pd.to_datetime(vix.index)
try:
    vidx = vidx.tz_localize(None)
except (TypeError, AttributeError):
    pass
vix.index = pd.DatetimeIndex(vidx).normalize()
vix = vix[~vix.index.duplicated(keep="first")].sort_index()
vix["dvix"] = vix["vix"].diff()
vix["dvix_pct"] = vix["vix"].pct_change()

# BTC 1분봉 (24시간, America/New_York)
btc_dir = "data/btc_1m_24h/"
files = sorted(f for f in os.listdir(btc_dir) if f.endswith(".parquet"))
btc = pd.concat([pd.read_parquet(btc_dir + f)[["close"]] for f in files]).sort_index()
btc = btc[~btc.index.duplicated(keep="first")]
if btc.index.tz is None:
    btc.index = btc.index.tz_localize("America/New_York")
else:
    btc.index = btc.index.tz_convert("America/New_York")

ET = "America/New_York"

# 가격 조회 헬퍼: 목표 시각 이후 가장 가까운 1분봉(최대 +5분 허용)
btc_idx = btc.index
def price_at(ts):
    pos = btc_idx.searchsorted(ts)
    if pos >= len(btc_idx):
        return np.nan
    found = btc_idx[pos]
    if (found - ts) > pd.Timedelta(minutes=5):
        return np.nan
    return btc.iloc[pos]["close"]

# ───────────────────────── 2. 응답 테이블 구성 ─────────────────────────
# 야간(예측) 오프셋
night_offsets = {
    "15m": 15, "30m": 30, "1h": 60, "2h": 120, "3h": 180,
    "4h": 240, "6h": 360, "8h": 480, "12h": 720, "to_open": 1050,  # ~17.5h → 익일 09:30
}

rows = []
for d, r in vix.iterrows():
    if np.isnan(r["dvix"]):
        continue
    base_ts = pd.Timestamp(d.year, d.month, d.day, 16, 0, tz=ET)
    p0 = price_at(base_ts)
    if np.isnan(p0):
        continue
    rec = {"date": d, "vix": r["vix"], "dvix": r["dvix"], "dvix_pct": r["dvix_pct"]}
    # 윈도우 A: 당일 Intraday(09:30→16:00) 동시 수익률 (contemporaneous 벤치마크)
    p_open = price_at(pd.Timestamp(d.year, d.month, d.day, 9, 30, tz=ET))
    rec["intraday"] = (p0 - p_open) / p_open if not np.isnan(p_open) else np.nan
    # 윈도우 B: 야간 Cumulative Return률 (예측적)
    for name, mins in night_offsets.items():
        pt = price_at(base_ts + pd.Timedelta(minutes=mins))
        rec[f"n_{name}"] = (pt - p0) / p0 if not np.isnan(pt) else np.nan
    rows.append(rec)

df = pd.DataFrame(rows).set_index("date").sort_index()
df["year"] = df.index.year

log("# VIX 종가(16:00 ET) → BTC 야간 지속성 분석 (P0)")
log("")
log(f"- 표본: {df.index.min().date()} ~ {df.index.max().date()}, 총 {len(df)} Trade Day")
log(f"- VIX 종가 변화(ΔVIX)는 16:00 ET에 확정 → 야간 BTC Return은 그 이후 측정 (**lookahead-free**)")
log(f"- 거래비용 가정: 왕복 0.04% (taker, 진입+청산 0.02%×2 가정은 보수적으로 0.04%)")
log("")

# ───────────────────────── 3. 핵심 비교: 동시 vs 예측 ─────────────────────────
def corr(a, b):
    m = pd.concat([a, b], axis=1).dropna()
    if len(m) < 20:
        return (np.nan, np.nan, len(m))
    r, p = stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])
    return (r, p, len(m))

log("## 1. 동시(Intraday) vs 예측(야간) 상관 — All 표본")
log("")
log("**윈도우 A (Intraday 동시, 거래 불가 벤치마크)**")
r, p, n = corr(df["dvix_pct"], df["intraday"])
log(f"- ΔVIX% vs 당일 Intraday(09:30→16:00) BTC: r={r:+.4f}, p={p:.2e}, n={n}")
log("")
log("**윈도우 B (야간 예측, VIX 동결 구간) — ΔVIX% vs 16:00 이후 BTC 누적수익률**")
log("")
log("| 호라이즌 | r | p-value | n | 방향성 hit% |")
log("|---|---|---|---|---|")
for name in night_offsets:
    col = f"n_{name}"
    r, p, n = corr(df["dvix_pct"], df[col])
    # 방향성: ΔVIX>0 → BTC<0 예측. hit = sign(-dvix)==sign(ret)
    m = df[["dvix_pct", col]].dropna()
    pred = -np.sign(m["dvix_pct"])
    hit = (np.sign(m[col]) == pred).mean() * 100
    log(f"| {name} | {r:+.4f} | {p:.2e} | {n} | {hit:.1f}% |")
log("")

# ───────────────────────── 4. Post-ETF (2024+) 별도 ─────────────────────────
post = df[df["year"] >= 2024]
log(f"## 2. Post-ETF (2024-01~) 별도 — n={len(post)}Trade Day")
log("(논문 H5b: ETF 이후에도 VIX→BTC 채널은 유지된다고 주장)")
log("")
log("| 호라이즌 | r | p-value | n | 방향성 hit% |")
log("|---|---|---|---|---|")
for name in night_offsets:
    col = f"n_{name}"
    r, p, n = corr(post["dvix_pct"], post[col])
    m = post[["dvix_pct", col]].dropna()
    pred = -np.sign(m["dvix_pct"])
    hit = (np.sign(m[col]) == pred).mean() * 100
    log(f"| {name} | {r:+.4f} | {p:.2e} | {n} | {hit:.1f}% |")
log("")

# ───────────────────────── 5. 2σ 이벤트 스터디 (야간) ─────────────────────────
log("## 3. VIX 2σ 이벤트 → 야간 BTC 반응 (All 표본)")
log("")
sd = df["dvix_pct"].std()
mu = df["dvix_pct"].mean()
spike = df[df["dvix_pct"] > mu + 2 * sd]
drop = df[df["dvix_pct"] < mu - 2 * sd]
log(f"- VIX Spike(2σ+): {len(spike)}건 / VIX Drop(2σ-): {len(drop)}건")
log("")
log("| 호라이즌 | 급등후 BTC Mean | t(p) | 급락후 BTC Mean | t(p) |")
log("|---|---|---|---|---|")
for name in night_offsets:
    col = f"n_{name}"
    s = spike[col].dropna(); d2 = drop[col].dropna()
    ts_s = stats.ttest_1samp(s, 0) if len(s) > 2 else (np.nan, np.nan)
    ts_d = stats.ttest_1samp(d2, 0) if len(d2) > 2 else (np.nan, np.nan)
    log(f"| {name} | {s.mean()*100:+.3f}% | {ts_s[0]:.2f}({ts_s[1]:.3f}) | {d2.mean()*100:+.3f}% | {ts_d[0]:.2f}({ts_d[1]:.3f}) |")
log("")

# ───────────────────────── 6. 거래 가능성 (롱숏, 비용 반영) ─────────────────────────
log("## 4. 단순 야간 Strategy 거래 가능성 (16:00 진입 → 호라이즌 청산)")
log("규칙: ΔVIX>0이면 BTC Short, ΔVIX<0이면 Long. 비용 왕복 0.04% 차감.")
log("")
log("| 호라이즌 | Mean 그로스 | Mean 넷(비용후) | 승률 | 누적 넷(단리합) |")
log("|---|---|---|---|---|")
COST = 0.0004
for name in night_offsets:
    col = f"n_{name}"
    m = df[["dvix_pct", col]].dropna()
    pos = -np.sign(m["dvix_pct"])  # short on spike, long on drop
    gross = pos * m[col]
    net = gross - COST
    log(f"| {name} | {gross.mean()*100:+.4f}% | {net.mean()*100:+.4f}% | {(gross>0).mean()*100:.1f}% | {net.sum()*100:+.1f}% |")
log("")

# ───────────────────────── 7. 결론 자동 판정 ─────────────────────────
# 야간 1h 상관과 to_open 상관으로 판정
r1, p1, _ = corr(df["dvix_pct"], df["n_1h"])
r_open, p_open2, _ = corr(df["dvix_pct"], df["n_to_open"])
rA, pA, _ = corr(df["dvix_pct"], df["intraday"])
log("## 5. 자동 판정")
log("")
log(f"- Intraday 동시 상관(A): r={rA:+.4f} (p={pA:.1e})")
log(f"- 야간 1h 예측 상관(B): r={r1:+.4f} (p={p1:.1e})")
log(f"- 야간 개장까지 예측 상관(B): r={r_open:+.4f} (p={p_open2:.1e})")
log("")
verdict = []
if pA < 0.05 and abs(rA) > 0.15:
    verdict.append("Intraday 동시 채널은 강하게 존재(예상대로, 거래 불가).")
if p1 < 0.05 and abs(r1) > 0.08:
    verdict.append("야간 예측 채널이 **유의** → 메커니즘 논거/거래 후보로 추가 검증 가치 있음(해석 A).")
else:
    verdict.append("야간 예측 채널은 **미약/비유의** → 16:00 발표 후 진입은 메커니즘 논거로 사용 어려움(해석 B). Granger 의존 권장.")
for v in verdict:
    log(f"- {v}")
log("")

# ───────────────────────── 8. 차트 ─────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
labels = list(night_offsets.keys())
xs = list(range(len(labels)))

# (1) 동시 vs 야간 상관 곡선
rs_all = [corr(df["dvix_pct"], df[f"n_{n}"])[0] for n in labels]
rs_post = [corr(post["dvix_pct"], post[f"n_{n}"])[0] for n in labels]
ax = axes[0, 0]
ax.axhline(rA, color="red", ls="--", label=f"Intraday Concurrent r={rA:.3f} (Non-tradable)")
ax.plot(xs, rs_all, "o-", label="Overnight Predictive r (All)")
ax.plot(xs, rs_post, "s-", label="Overnight Predictive r (post-ETF)")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45)
ax.set_title("ΔVIX% vs BTC: Concurrent (Intraday) vs Predictive (Overnight)")
ax.set_ylabel("Pearson r"); ax.legend(); ax.grid(alpha=0.3)

# (2) 2σ 이벤트 반응 경로
ax = axes[0, 1]
ax.plot(xs, [spike[f"n_{n}"].mean()*100 for n in labels], "r^-", label=f"After VIX Spike ({len(spike)})")
ax.plot(xs, [drop[f"n_{n}"].mean()*100 for n in labels], "gv-", label=f"After VIX Drop ({len(drop)})")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45)
ax.set_title("Overnight BTC Mean Cumulative Return After VIX 2σ Event")
ax.set_ylabel("%"); ax.legend(); ax.grid(alpha=0.3)

# (3) Strategy 넷수익 누적
ax = axes[1, 0]
nets = []
for n in labels:
    m = df[["dvix_pct", f"n_{n}"]].dropna()
    pos = -np.sign(m["dvix_pct"])
    nets.append(((pos*m[f"n_{n}"]) - COST).sum()*100)
ax.bar(xs, nets, color=["g" if v > 0 else "r" for v in nets])
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45)
ax.set_title("Overnight ΔVIX Contrarian Strategy Cumulative Net Return (after 0.04% cost)")
ax.set_ylabel("Cumulative %"); ax.grid(alpha=0.3)

# (4) 산점도: ΔVIX% vs 야간 1h
ax = axes[1, 1]
m = df[["dvix_pct", "n_1h"]].dropna()
ax.scatter(m["dvix_pct"]*100, m["n_1h"]*100, s=8, alpha=0.4)
ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
ax.set_title(f"ΔVIX% vs Overnight 1h BTC (r={r1:+.3f})")
ax.set_xlabel("ΔVIX %"); ax.set_ylabel("BTC 1h %"); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("vix_overnight_persistence.png", dpi=120)

# ───────────────────────── 저장 ─────────────────────────
os.makedirs("docs", exist_ok=True)
with open("docs/vix_overnight_persistence_report.md", "w") as f:
    f.write("\n".join(REPORT))

# 콘솔에도 핵심만 (1회)
print("ANALYSIS_COMPLETE rows=%d intradayR=%.4f night1hR=%.4f nightOpenR=%.4f" % (
    len(df), rA, r1, r_open))
