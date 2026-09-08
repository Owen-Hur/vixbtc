"""VIX Duration 슬라이드용 차트 재생성 — 덱 팔레트(네이비 #122B46) 매칭"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 이 스크립트가 있는 폴더(analysis/vix_duration/) 기준 — data/ · charts/ 하위 폴더 참조
BASE = str(Path(__file__).resolve().parent)
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

NAVY = "#122B46"
STEEL = "#5B7B9A"
GOLD = "#C99A2E"
RED = "#A93226"
GRAY = "#9AA5B1"

mres = pd.read_csv(f"{BASE}/data/vix_duration_minute_metrics.csv")
with open(f"{BASE}/data/vix_duration_summary.json") as f:
    summary = json.load(f)

h = mres["tday_hour"].values
rA = summary["concurrent_benchmark"]["primary"]["r"]

# ─────────── Chart A: 분 단위 적중률 (슬라이드 와이드) ───────────
fig, ax = plt.subplots(1, 1, figsize=(7.6, 3.3))
ax.plot(h, mres["hit_primary"], color=NAVY, lw=2.0, label="2024–2026 (n=601)", zorder=3)
ax.plot(h, mres["hit_supp"], color=GOLD, lw=1.1, alpha=0.8, label="2020–2023 (n=1017)")
ax.axhline(50, color=RED, lw=1.3, ls="--", label="50% (무작위 기준)", zorder=2)
ax.set_ylabel("방향 적중률 (%)", fontsize=10, color=NAVY)
ax.set_xlabel("VIX 확정 후 경과 시간 (시간)", fontsize=10, color=NAVY)
ax.set_xlim(0, 17.2); ax.set_ylim(40, 58)
ax.set_xticks(np.arange(0, 18, 2))
ax.legend(loc="upper right", fontsize=8, ncol=1, framealpha=0.9)
ax.grid(alpha=0.25)
ax.tick_params(colors=NAVY, labelsize=8)
for sp in ax.spines.values(): sp.set_color("#CCCCCC")
plt.tight_layout()
plt.savefig(f"{BASE}/charts/slide_hit.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved slide_hit.png")

# ─────────── Chart B: 분 단위 상관계수 (슬라이드 와이드) ───────────
fig, ax = plt.subplots(1, 1, figsize=(7.6, 3.3))
ax.plot(h, mres["r_primary"], color=NAVY, lw=2.0, label="2024–2026 (n=601)", zorder=3)
ax.plot(h, mres["r_supp"], color=GOLD, lw=1.1, alpha=0.8, label="2020–2023 (n=1017)")
ax.axhline(0, color="#555555", lw=0.8)
ax.axhline(rA, color=RED, lw=1.3, ls=":", label=f"동시 벤치마크 r={rA:.2f}\n(거래 불가)", zorder=2)
ax.set_ylabel("상관계수 r", fontsize=10, color=NAVY)
ax.set_xlabel("VIX 확정 후 경과 시간 (시간)", fontsize=10, color=NAVY)
ax.set_xlim(0, 17.2)
ax.set_xticks(np.arange(0, 18, 2))
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax.grid(alpha=0.25)
ax.tick_params(colors=NAVY, labelsize=8)
for sp in ax.spines.values(): sp.set_color("#CCCCCC")
plt.tight_layout()
plt.savefig(f"{BASE}/charts/slide_corr.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved slide_corr.png")

# ─────────── Chart C: 2σ 이벤트 경로 (재계산 불가 → 기존 데이터 없음) ───────────
# 2σ 경로는 raw가 없으므로, 기존 vix_duration_2sigma_events.png를 덱 색감으로 재생성하려면
# raw 필요. 대신 분석 스크립트의 spike/drop path를 재현하기 위해 원본 분석 재실행 필요.
# 여기서는 별도 스크립트(regen_2sigma.py)에서 처리.
print("done (hit, corr)")
