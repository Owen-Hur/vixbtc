"""슬라이드용 차트 재생성 — 덱 팔레트(네이비 #122B46) 매칭"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE = "/Users/macbook/btc_project/slope_change"

# 한글 폰트 (AppleGothic)
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

NAVY = "#122B46"
STEEL = "#5B7B9A"
GOLD = "#C99A2E"
RED = "#A93226"
GRAY = "#9AA5B1"

mres = pd.read_csv(f"{BASE}/data/slope_change_minute_metrics.csv")
with open(f"{BASE}/data/slope_change_summary.json") as f:
    summary = json.load(f)

# ─────────── Chart A: 분 단위 적중률 (단일 패널, 슬라이드용 와이드) ───────────
fig, ax = plt.subplots(1, 1, figsize=(7.6, 3.4))
h = mres["tday_hour"].values
ax.plot(h, mres["hit_all"], color=NAVY, lw=2.0, label="전체 (n=580)", zorder=3)
ax.plot(h, mres["hit_is"], color=STEEL, lw=1.1, alpha=0.85, label="IS (n=457)")
ax.plot(h, mres["hit_oos"], color=GOLD, lw=1.1, alpha=0.85, label="OOS (n=123)")
ax.axhline(50, color=RED, lw=1.3, ls="--", label="50% (무작위 기준)", zorder=2)
ax.set_ylabel("방향 적중률 (%)", fontsize=10, color=NAVY)
ax.set_xlabel("신호 확정 후 경과 시간 (시간)", fontsize=10, color=NAVY)
ax.set_xlim(0, 17.2)
ax.set_ylim(38, 62)
ax.set_xticks(np.arange(0, 18, 2))
ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
ax.grid(alpha=0.25)
ax.tick_params(colors=NAVY, labelsize=8)
for spine in ax.spines.values():
    spine.set_color("#CCCCCC")
plt.tight_layout()
plt.savefig(f"{BASE}/charts/slide_minute_hit.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved slide_minute_hit.png")

# ─────────── Chart B: |slope_change| 크기별 적중률 ───────────
bins = summary["magnitude_bins"]
labels = [r["bin"] for r in bins["ALL"]]
is_hits = [r["hit"] if r["n"] > 0 else 0 for r in bins["IS"]]
oos_hits = [r["hit"] if r["n"] > 0 else 0 for r in bins["OOS"]]
all_hits = [r["hit"] if r["n"] > 0 else 0 for r in bins["ALL"]]
all_ns = [r["n"] for r in bins["ALL"]]

fig, ax = plt.subplots(1, 1, figsize=(7.6, 3.6))
x = np.arange(len(labels))
w = 0.26
ax.bar(x - w, is_hits, w, color=STEEL, label="IS")
ax.bar(x, oos_hits, w, color=GOLD, label="OOS")
ax.bar(x + w, all_hits, w, color=NAVY, label="전체")
ax.axhline(50, color=RED, lw=1.3, ls="--", label="50% (무작위)")
for i, n in enumerate(all_ns):
    # 라벨을 50% 점선 위로 띄워 겹침 방지
    ly = max(all_hits[i] + 2.0, 53.5)
    ax.text(i + w, ly, f"n={n}", ha="center", fontsize=7, color=NAVY)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlabel("|slope_change| 크기 구간", fontsize=10, color=NAVY)
ax.set_ylabel("적중률 (%)", fontsize=10, color=NAVY)
ax.set_ylim(0, 72)
ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
ax.grid(alpha=0.25, axis="y")
ax.tick_params(colors=NAVY, labelsize=8)
for spine in ax.spines.values():
    spine.set_color("#CCCCCC")
plt.tight_layout()
plt.savefig(f"{BASE}/charts/slide_magnitude_hit.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved slide_magnitude_hit.png")

# ─────────── Chart C: 상관계수 분 단위 (슬라이드용) ───────────
fig, ax = plt.subplots(1, 1, figsize=(7.6, 3.0))
ax.plot(h, mres["r_all"], color=NAVY, lw=2.0, label="전체")
ax.plot(h, mres["r_is"], color=STEEL, lw=1.0, alpha=0.8, label="IS")
ax.plot(h, mres["r_oos"], color=GOLD, lw=1.0, alpha=0.8, label="OOS")
ax.axhline(0, color="#555555", lw=0.8)
ax.set_ylabel("상관계수 r", fontsize=10, color=NAVY)
ax.set_xlabel("신호 확정 후 경과 시간 (시간)", fontsize=10, color=NAVY)
ax.set_xlim(0, 17.2)
ax.set_xticks(np.arange(0, 18, 2))
ax.legend(loc="lower right", fontsize=8, ncol=3, framealpha=0.9)
ax.grid(alpha=0.25)
ax.tick_params(colors=NAVY, labelsize=8)
for spine in ax.spines.values():
    spine.set_color("#CCCCCC")
plt.tight_layout()
plt.savefig(f"{BASE}/charts/slide_minute_corr.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved slide_minute_corr.png")

# ─────────── Chart D: 다음날 수익률 분포 (IS, 단일) ───────────
import os
# 원자료가 필요하므로 분석 재현 대신 기존 분포 차트 재활용 불가 → summary로 대체 불가.
# 대신 IS 분포를 다시 계산하기 위해 raw 필요. 여기서는 기존 분포 차트가 이미 있으므로 패스.
print("done")
