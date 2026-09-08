"""
VIX → BTC 방향성 재검증: ① 임계 필터 + ⑤ 부호 데이터 결정
============================================================
기존 분석은 '예측부호 = -sign(ΔVIX%)' 단일 룰(All 표본 Mean)만 봤다.
본 분석은 두 가지를 바꾼다:

  ① 임계 필터(threshold filter):
     ΔVIX%가 ±임계를 넘는 '큰 변화'에만 베팅, 그 사이는 관망.
     → 미세 변동(노이즈)을 거래에서 제외.

  ⑤ 부호를 데이터에 맡김(sign learned from data, not assumed):
     'VIX↑→BTC↓'를 미리 가정하지 않는다.
     train 구간에서 어느 부호(정방향/역방향)가 맞는지 학습 →
     OOS 구간에서 그 부호로 검증. (lookahead-free)

타깃: 다음 Trade Day(16:00 ET close-to-close) BTC Return 부호
비용: 왕복 0.04% (taker), 옵션으로 펀딩 영향 주석
검정: 이항검정(binomial) + 다중검정 보정(Bonferroni)

산출물:
  new/vix_threshold_directional_report.md
  new/vix_threshold_grid.png
  new/vix_threshold_directional.csv
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ET = "America/New_York"
COST = 0.0004
SPLIT_DATE = pd.Timestamp("2023-01-01")  # train: ~2022, OOS: 2023~
REPORT = []
def w(s=""): REPORT.append(str(s))

# ───────── 1. 데이터 ─────────
vix = pd.read_parquet("data/vix_daily.parquet").copy(); vix.columns = ["vix"]
vidx = pd.to_datetime(vix.index)
try: vidx = vidx.tz_localize(None)
except Exception: pass
vix.index = pd.DatetimeIndex(vidx).normalize()
vix = vix[~vix.index.duplicated()].sort_index()

btc_dir = "data/btc_1m_24h/"
files = sorted(f for f in os.listdir(btc_dir) if f.endswith(".parquet"))
btc = pd.concat([pd.read_parquet(btc_dir+f)[["close"]] for f in files]).sort_index()
btc = btc[~btc.index.duplicated()]
btc.index = btc.index.tz_localize(ET) if btc.index.tz is None else btc.index.tz_convert(ET)
close = btc["close"]
def price_at(ts):
    return close.reindex([ts], method="pad", tolerance=pd.Timedelta(minutes=5)).iloc[0]

# 16:00 ET 기준 일별 BTC
recs = []
for d, r in vix.iterrows():
    p = price_at(pd.Timestamp(d.year, d.month, d.day, 16, 0, tz=ET))
    if not np.isnan(p):
        recs.append((d, r["vix"], p))
dd = pd.DataFrame(recs, columns=["date","vix","btc"]).set_index("date").sort_index()
dd["dvix_pct"] = dd["vix"].pct_change()
# 신호는 t시점(16:00)에 확정 → 다음 구간 수익률(t→t+1)을 예측 (lookahead-free)
dd["btc_fwd"] = dd["btc"].pct_change().shift(-1)
dd = dd.dropna(subset=["dvix_pct","btc_fwd"])
dd["year"] = dd.index.year

train = dd[dd.index < SPLIT_DATE]
oos   = dd[dd.index >= SPLIT_DATE]

# ───────── 2. 핵심 함수 ─────────
def binom_p(hits, n):
    """적중률이 50%를 유의하게 초과/미달하는지 양측 이항검정."""
    if n == 0: return np.nan
    return stats.binomtest(hits, n, 0.5).pvalue

def eval_rule(data, thr, sign):
    """
    임계 thr(%), 부호 sign(+1: VIX↑→BTC↓ 정방향 / -1: 역방향)로 평가.
    포지션: 정방향이면 pos = -sign(dvix)·(|dvix%|>thr), 역방향이면 +sign.
    반환: n(거래수), hit%, Mean넷수익%, 누적넷%, 이항 p
    """
    dv = data["dvix_pct"].to_numpy()*100  # %
    fwd = data["btc_fwd"].to_numpy()
    mask = np.abs(dv) > thr
    if mask.sum() == 0:
        return dict(n=0, hit=np.nan, net=np.nan, cum=np.nan, p=np.nan)
    # 정방향(sign=+1): VIX 오르면 숏 → pos = -sign(dv). 역방향(sign=-1): pos = +sign(dv)
    pos = (-np.sign(dv)) * sign
    pos = pos[mask]; ret = fwd[mask]
    gross = pos*ret
    net = gross - COST
    hits = int((gross > 0).sum())
    n = int(mask.sum())
    return dict(n=n, hit=hits/n*100, net=net.mean()*100, cum=net.sum()*100,
                p=binom_p(hits, n), hits=hits)

# ───────── 3. 격자 탐색 (train에서) ─────────
THRESHOLDS = [0, 1, 2, 3, 5, 7, 10]   # ΔVIX% 절대 임계
grid_rows = []
for thr in THRESHOLDS:
    for sign, sname in [(+1,"정방향(VIX↑→BTC↓)"), (-1,"역방향(VIX↑→BTC↑)")]:
        tr = eval_rule(train, thr, sign)
        grid_rows.append(dict(thr=thr, sign=sign, sname=sname, **{f"tr_{k}":v for k,v in tr.items()}))
grid = pd.DataFrame(grid_rows)

# train에서 '부호를 데이터로 결정': 각 임계별로 train 적중률이 높은 부호 선택
chosen = []
for thr in THRESHOLDS:
    sub = grid[grid.thr==thr]
    best = sub.loc[sub["tr_hit"].idxmax()]  # train 적중률 최대 부호
    chosen.append(dict(thr=thr, chosen_sign=int(best["sign"]),
                       chosen_sname=best["sname"],
                       tr_hit=best["tr_hit"], tr_n=int(best["tr_n"])))
chosen = pd.DataFrame(chosen)

# ───────── 4. OOS 검증 (train에서 고른 부호로) ─────────
oos_rows = []
for _, c in chosen.iterrows():
    thr = c["thr"]; sign = c["chosen_sign"]
    o = eval_rule(oos, thr, sign)
    # 비교용: All기간 결과도
    full = eval_rule(dd, thr, sign)
    oos_rows.append(dict(thr=thr, chosen_sname=c["chosen_sname"],
                         tr_hit=c["tr_hit"], tr_n=c["tr_n"],
                         oos_n=o["n"], oos_hit=o["hit"], oos_net=o["net"],
                         oos_cum=o["cum"], oos_p=o["p"],
                         full_n=full["n"], full_hit=full["hit"], full_cum=full["cum"]))
oosdf = pd.DataFrame(oos_rows)

# 다중검정 보정: OOS에서 검정한 임계 수 = len(THRESHOLDS)
bonf = 0.05/len(THRESHOLDS)

# 베이스라인: 기존 단일 룰(thr=0, 정방향) — All기간
base = eval_rule(dd, 0, +1)

# 결과 CSV
out = grid.merge(oosdf, on="thr", how="left")
out.to_csv("new/vix_threshold_directional.csv", index=False)

# ───────── 5. 차트 ─────────
fig, axes = plt.subplots(1, 3, figsize=(18,5))
# (1) train: 임계별 양 부호 적중률
for sign, c, lab in [(+1,"tab:red","forward (VIX^->BTC v)"), (-1,"tab:blue","reverse")]:
    sub = grid[grid.sign==sign]
    axes[0].plot(sub["thr"], sub["tr_hit"], "o-", color=c, label=lab)
axes[0].axhline(50, color="gray", ls="--", label="50% coin flip")
axes[0].set_title("TRAIN: hit% by threshold & sign"); axes[0].set_xlabel("|dVIX%| threshold")
axes[0].set_ylabel("hit %"); axes[0].legend(); axes[0].grid(alpha=0.3)
# (2) OOS: 선택 부호의 적중률 + 표본수
ax2 = axes[1]; ax2b = ax2.twinx()
ax2.plot(oosdf["thr"], oosdf["oos_hit"], "o-", color="green", label="OOS hit%")
ax2.axhline(50, color="gray", ls="--")
ax2b.bar(oosdf["thr"], oosdf["oos_n"], alpha=0.2, color="gray")
ax2.set_title("OOS: hit% (line) & trade count (bars)")
ax2.set_xlabel("|dVIX%| threshold"); ax2.set_ylabel("OOS hit %"); ax2b.set_ylabel("n trades")
ax2.legend(); ax2.grid(alpha=0.3)
# (3) OOS 누적넷
axes[2].bar(oosdf["thr"], oosdf["oos_cum"],
            color=["g" if v>0 else "r" for v in oosdf["oos_cum"].fillna(0)])
axes[2].set_title("OOS: cumulative NET return % (cost 0.04%)")
axes[2].set_xlabel("|dVIX%| threshold"); axes[2].set_ylabel("cum net %"); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("new/vix_threshold_grid.png", dpi=120); plt.close()

# ───────── 6. 보고서 ─────────
w("# VIX → BTC 방향성 재검증 — 임계 필터 + 부호 데이터 결정")
w("")
w("> 작성일 2026-06-02 · 코드 `vix_threshold_directional.py`")
w(f"> 표본: {dd.index.min().date()} ~ {dd.index.max().date()}, 총 {len(dd)} Trade Day")
w("")
w("## 0. 무엇을, 왜, 어떻게 바꿨나")
w("")
w("기존 야간/Granger 분석의 방향성 검증은 **단일 룰**이었다:")
w("```")
w("예측부호 = -sign(ΔVIX%)   # VIX 조금이라도 오르면 무조건 숏, All 표본 Mean")
w("```")
w("이 룰의 두 가지 약점을 본 분석에서 교정한다.")
w("")
w("| 약점 | 본 분석의 교정 |")
w("|---|---|")
w("| 미세 변화도 무조건 베팅(노이즈 흡수) | **① 임계 필터**: |ΔVIX%|>임계인 '큰 변화'에만 베팅, 그 사이 관망 |")
w("| 'VIX↑→BTC↓' 부호를 사람이 미리 가정 | **⑤ 부호 데이터 결정**: train에서 정/역 부호를 학습 → OOS 검증 |")
w("")
w("**핵심 식별**: 신호 ΔVIX%는 t시점 16:00에 확정 → 타깃은 t→t+1 수익률(`shift(-1)`) → lookahead-free.")
w("")
w("## 1. 데이터 · 정의")
w("| 항목 | 값 |")
w("|---|---|")
w(f"| 신호 | ΔVIX% = VIX 종가 전일 대비 변화율 (16:00 ET 확정) |")
w(f"| 타깃 | 다음 Trade Day BTC Return (16:00→16:00 close-to-close) |")
w(f"| 표본 | {len(dd)}일 ({dd.index.min().date()}~{dd.index.max().date()}) |")
w(f"| Train | {len(train)}일 (~2022-12) → 부호 학습 |")
w(f"| OOS | {len(oos)}일 (2023-01~) → 검증 (건드리지 않음) |")
w(f"| 비용 | 왕복 {COST*100:.2f}% (taker) |")
w(f"| 임계 격자 | {THRESHOLDS} (|ΔVIX%|, 단위 %) |")
w(f"| 부호 | +1 정방향(VIX↑→BTC↓) / −1 역방향(VIX↑→BTC↑) |")
w("")
w("## 2. 방법론")
w("1. **임계 필터**: |ΔVIX%|>thr인 날만 거래. thr=0이면 모든 날(=기존 단일 룰과 동일).")
w("2. **부호 학습(⑤)**: train 구간에서 각 임계별로 정방향/역방향 중 **적중률 높은 부호**를 선택.")
w("3. **OOS 검증**: train에서 고른 부호를 OOS에 적용. train 정보만으로 결정 → 미래 누수 없음.")
w("4. **이항검정**: 적중 횟수가 n회 중 50%를 유의하게 벗어나는지 양측 binomial test.")
w(f"5. **다중검정 보정**: 임계 {len(THRESHOLDS)}개 검정 → Bonferroni 임계 p<{bonf:.4f}.")
w("6. **거래 가능성**: 적중률뿐 아니라 비용 차감 후 누적 순수익으로 최종 판단.")
w("")
w("## 3. TRAIN 결과 — 임계별·부호별 적중률")
w("(train에서 어느 부호가 우세한지 관찰. 이 단계는 '학습'이며 성과 주장 아님.)")
w("")
w("| 임계 |ΔVIX%|> | 부호 | train n | train hit% | train 누적넷% |")
w("|---|---|---|---|---|")
for _, g in grid.iterrows():
    w(f"| {g['thr']} | {g['sname']} | {int(g['tr_n']) if not np.isnan(g['tr_n']) else 0} | "
      f"{g['tr_hit']:.1f}% | {g['tr_cum']:+.1f}% |" if not np.isnan(g['tr_hit']) else
      f"| {g['thr']} | {g['sname']} | 0 | - | - |")
w("")
w("**train에서 선택된 부호(임계별):**")
w("")
w("| 임계 | 선택 부호 | train hit% | train n |")
w("|---|---|---|---|")
for _, c in chosen.iterrows():
    w(f"| {c['thr']} | {c['chosen_sname']} | {c['tr_hit']:.1f}% | {int(c['tr_n'])} |")
w("")
w("## 4. OOS 결과 — 핵심 (train에서 고른 부호로 검증)")
w("")
w("| 임계 | 선택부호 | OOS n | OOS hit% | 이항 p | 비용후 누적넷% | 판정 |")
w("|---|---|---|---|---|---|---|")
for _, o in oosdf.iterrows():
    if o["oos_n"] == 0 or np.isnan(o["oos_hit"]):
        w(f"| {o['thr']} | {o['chosen_sname']} | 0 | - | - | - | 표본없음 |")
        continue
    verdict = "유의(보정통과)" if o["oos_p"]<bonf and o["oos_hit"]>50 else \
              ("경계(보정미달)" if o["oos_p"]<0.05 and o["oos_hit"]>50 else "무의미(≈50%)")
    w(f"| {o['thr']} | {o['chosen_sname']} | {int(o['oos_n'])} | {o['oos_hit']:.1f}% | "
      f"{o['oos_p']:.3f} | {o['oos_cum']:+.1f}% | {verdict} |")
w("")
w(f"- Bonferroni 임계: p < {bonf:.4f}")
w(f"- 베이스라인(기존 단일 룰, thr=0 정방향, All기간): hit {base['hit']:.1f}%, 누적넷 {base['cum']:+.1f}%")
w("")
w("![격자 결과](vix_threshold_grid.png)")
w("")
w("## 5. 해석")
# 자동 판정 로직
pass_rows = oosdf[(oosdf["oos_p"]<bonf) & (oosdf["oos_hit"]>50) & (oosdf["oos_n"]>=20)]
edge_rows = oosdf[(oosdf["oos_p"]<0.05) & (oosdf["oos_hit"]>50) & (oosdf["oos_n"]>=20)]
if len(pass_rows):
    w(f"- ✅ **다중검정 보정 통과 임계 {len(pass_rows)}건 발견**: "
      + ", ".join(f"임계{int(r.thr)}(hit {r.oos_hit:.1f}%, 넷 {r.oos_cum:+.1f}%)" for _,r in pass_rows.iterrows()))
    w("  → 추가 견고성 검증(다른 분할·펀딩 반영·표본 안정성) 권장.")
elif len(edge_rows):
    w(f"- ⚠️ 보정 전 p<0.05인 경계 사례 {len(edge_rows)}건 있으나 Bonferroni 미통과 → 우연 가능성 높음.")
else:
    w("- ❌ **OOS에서 50%를 유의하게 초과하는 임계 없음.** 부호를 데이터로 학습하고 큰 변화만 걸러도 방향 예측 불가.")
w("")
w("- 임계를 높일수록(큰 VIX Change만) 표본 수가 급감 → 적중률이 흔들려도 우연 구분 어려움(이항 p가 커짐).")
w("- train에서 좋아 보인 부호가 OOS에서 유지되는지가 관건. 유지 안 되면 train 적중률은 과적합.")
w("")
w("## 6. 한계")
w("- 일별 close-to-close 타깃만 검증(Intraday·야간 분리는 별도). 펀딩비 미반영(반영 시 순수익 하락 → 결론 강화 방향).")
w("- 고임계 구간 표본 부족 → 통계력 약함. 임계·부호 격자 탐색의 다중검정은 Bonferroni로 보정했으나 보수적.")
w("- 부호 학습을 train 적중률 기준으로 했음. 누적수익 기준 등 대안 학습규칙은 미검토.")
w("- 단일 train/OOS 분할(2023 경계). 워크포워드 다분할은 후속 과제.")
w("")
w("## 7. 결론")
if len(pass_rows):
    w("임계 필터 + 부호 데이터 학습 조합에서 **OOS 유의 사례가 관찰됨**. 단정 전 견고성 검증 필요(아래 후속).")
else:
    w("**임계 필터(①)와 부호 데이터 학습(⑤)을 적용해도 VIX→BTC 방향 예측은 OOS에서 동전 던지기를 넘지 못한다.** "
      "기존 단일 룰의 '방향 예측 불가' 결론은 더 정교한 결정규칙에서도 견고하게 유지된다.")
w("")
w("### 산출물")
w("- 보고서: `new/vix_threshold_directional_report.md`")
w("- 차트: `new/vix_threshold_grid.png`")
w("- 원자료(격자+OOS): `new/vix_threshold_directional.csv`")
w("- 코드: `new/vix_threshold_directional.py`")

with open("new/vix_threshold_directional_report.md","w") as f:
    f.write("\n".join(REPORT))

print("DONE n=%d train=%d oos=%d | base_hit=%.1f | oos_pass=%d edge=%d bonf=%.4f" % (
    len(dd), len(train), len(oos), base["hit"], len(pass_rows), len(edge_rows), bonf))
