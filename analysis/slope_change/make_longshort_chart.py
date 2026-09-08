"""양방향 분해(Long/Short) 적중률 차트 — 덱 팔레트"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']='AppleGothic'
matplotlib.rcParams['axes.unicode_minus']=False

# 저장소 루트 (data/ 가 있는 곳) — 실행 위치와 무관하게 파일 위치로부터 계산
BASE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent   # analysis/slope_change/

NAVY="#122B46"; STEEL="#5B7B9A"; RED="#A93226"; MUTE="#6B7280"

groups=["IS","OOS","전체"]
long_hit=[51.4,50.0,51.1]; long_n=[247,62,309]
short_hit=[49.5,50.8,49.8]; short_n=[210,61,271]

fig,ax=plt.subplots(1,1,figsize=(7.2,3.4))
x=np.arange(len(groups)); w=0.36
b1=ax.bar(x-w/2,long_hit,w,color=NAVY,label="Long (sc>0)")
b2=ax.bar(x+w/2,short_hit,w,color=STEEL,label="Short (sc<0)")
ax.axhline(50,color=RED,lw=1.3,ls="--",label="50% (무작위 기준)")

# 값·표본 라벨
for xi,(lh,ln_) in enumerate(zip(long_hit,long_n)):
    ax.text(xi-w/2,lh+0.4,f"{lh:.1f}%",ha="center",fontsize=10,color=NAVY,fontweight="bold")
    ax.text(xi-w/2,2,f"n={ln_}",ha="center",fontsize=8,color="white")
for xi,(sh,sn) in enumerate(zip(short_hit,short_n)):
    ax.text(xi+w/2,sh+0.4,f"{sh:.1f}%",ha="center",fontsize=10,color=NAVY,fontweight="bold")
    ax.text(xi+w/2,2,f"n={sn}",ha="center",fontsize=8,color="white")

ax.set_xticks(x); ax.set_xticklabels(groups,fontsize=11,color=NAVY)
ax.set_ylabel("방향 적중률 (%)",fontsize=10,color=NAVY)
ax.set_ylim(0,62)
ax.legend(loc="upper right",fontsize=8.5,ncol=3,framealpha=0.9)
ax.grid(alpha=0.25,axis="y")
ax.tick_params(colors=NAVY,labelsize=9)
for sp in ax.spines.values(): sp.set_color("#CCCCCC")
plt.tight_layout()
(BASE/"_slide_assets").mkdir(parents=True, exist_ok=True)
out=str(BASE/"_slide_assets"/"slide_longshort.png")
plt.savefig(out,dpi=200,bbox_inches="tight")
plt.savefig(str(HERE/"charts"/"slope_change_longshort.png"),dpi=200,bbox_inches="tight")
print("saved",out)
