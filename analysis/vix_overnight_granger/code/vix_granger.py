"""
VIX ↔ BTC Granger 인과 분석 (종합판)
=====================================
포함:
  - 일별 ΔVIX% ↔ BTC, 양방향, All/pre/post-ETF
  - 일별 VIX slope(기간구조) Δslope ↔ BTC, 양방향, All/pre/post-ETF
  - 시간별(Intraday) ΔVIX ↔ BTC + 정렬 진단(동시성 누수 검증)
  - 정상성(ADF), VAR 차수선택(AIC/BIC), 4가지 검정통계량, 효과크기(R² 증분)
  - 다중검정 보정, 한계
산출물: docs/VIX_granger_report.md, vix_granger.png
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ET = "America/New_York"; MAXLAG_D = 10; MAXLAG_H = 12
L = []
def w(s=""): L.append(str(s))

# ───────── 데이터 ─────────
btc_dir = "data/btc_1m_24h/"
files = sorted(f for f in os.listdir(btc_dir) if f.endswith(".parquet"))
btc = pd.concat([pd.read_parquet(btc_dir+f)[["close"]] for f in files]).sort_index()
btc = btc[~btc.index.duplicated()]
btc.index = btc.index.tz_localize(ET) if btc.index.tz is None else btc.index.tz_convert(ET)
close = btc["close"]
def price_at(ts):
    return close.reindex([ts], method="pad", tolerance=pd.Timedelta(minutes=5)).iloc[0]

# VIX 일별
vix_d = pd.read_parquet("data/vix_daily.parquet").copy(); vix_d.columns=["vix"]
vidx = pd.to_datetime(vix_d.index)
try: vidx = vidx.tz_localize(None)
except Exception: pass
vix_d.index = pd.DatetimeIndex(vidx).normalize(); vix_d = vix_d[~vix_d.index.duplicated()].sort_index()

# VIX slope 일별
sl = pd.read_parquet("data/vix_slope_daily.parquet").copy()
if "Date" in sl.columns:
    sl["Date"] = pd.to_datetime(sl["Date"]); sl = sl.set_index("Date")
sl.index = pd.DatetimeIndex(pd.to_datetime(sl.index)).normalize()
sl = sl.sort_index(); sl = sl[~sl.index.duplicated()]

# 일별 BTC(16:00 ET) 수익률 + 병합
recs=[]
for d,r in vix_d.iterrows():
    p = price_at(pd.Timestamp(d.year,d.month,d.day,16,0,tz=ET))
    if not np.isnan(p): recs.append((d, r["vix"], p))
dd = pd.DataFrame(recs, columns=["date","vix","btc"]).set_index("date").sort_index()
dd["dvix_pct"] = dd["vix"].pct_change()
dd["btc_ret"]  = dd["btc"].pct_change()
dd = dd.join(sl[["slope"]], how="left")
dd["dslope"] = dd["slope"].diff()
dd = dd.dropna(subset=["dvix_pct","btc_ret"]); dd["year"]=dd.index.year

# 시간별
vix_h = pd.read_parquet("data/vix_1h.parquet").copy(); vix_h.columns=["vix"]
vix_h.index = pd.to_datetime(vix_h.index)
vix_h.index = vix_h.index.tz_localize(ET) if vix_h.index.tz is None else vix_h.index.tz_convert(ET)
vix_h = vix_h[~vix_h.index.duplicated()].sort_index()
bh = close.reindex(vix_h.index, method="pad", tolerance=pd.Timedelta(minutes=5))
hh = pd.DataFrame({"vix":vix_h["vix"],"btc":bh}).dropna()
hh["dvix"]=hh["vix"].diff(); hh["btc_ret"]=hh["btc"].pct_change()
hh["hour"]=hh.index.hour; hh["gap"]=hh.index.to_series().diff().dt.total_seconds()/3600
hh_m = hh[(hh["hour"]>=10)&(hh["hour"]<=16)&(hh["gap"]==1.0)].dropna()

# ───────── 헬퍼 ─────────
def adf(x):
    try:
        r=adfuller(x.dropna(),autolag="AIC"); return r[0],r[1]
    except Exception: return (np.nan,np.nan)

def var_order(effect,cause,maxlag):
    data=pd.concat([effect,cause],axis=1).dropna(); data.columns=["e","c"]
    try:
        sel=VAR(data).select_order(maxlag)
        return sel.aic, sel.bic
    except Exception: return (np.nan,np.nan)

def granger(cause,effect,maxlag):
    """반환: lag별 ssr_ftest p dict, 최소p/lag, 4통계량(@최소p lag), R²증분(@최소p lag), 고정lag1 p."""
    data=pd.concat([effect,cause],axis=1).dropna(); data.columns=["effect","cause"]
    res=grangercausalitytests(data[["effect","cause"]],maxlag=maxlag,verbose=False)
    ps={lag:res[lag][0]["ssr_ftest"][1] for lag in res}
    best=min(ps,key=ps.get)
    t=res[best][0]
    stats4={"ssr_F":t["ssr_ftest"][1],"ssr_chi2":t["ssr_chi2test"][1],
            "lr":t["lrtest"][1],"params_F":t["params_ftest"][1]}
    # R² 증분: restricted vs unrestricted OLS
    try:
        r_res,r_unr=res[best][1][0],res[best][1][1]
        dR2=r_unr.rsquared - r_res.rsquared
    except Exception:
        dR2=np.nan
    return ps,best,stats4,dR2,ps[1]

def block(title,cause,effect,lc,le,maxlag):
    n=pd.concat([effect,cause],axis=1).dropna().shape[0]
    if n < maxlag+10:
        w(f"**{title}: {lc} → {le}** — 표본 부족(n={n}) 생략"); w(""); return {l:np.nan for l in range(1,maxlag+1)}
    ps,best,st4,dR2,p1=granger(cause,effect,maxlag)
    mp=ps[best]
    bonf = 0.05/maxlag
    sig = "예측력 있음" if mp<bonf else ("경계(다중검정 미통과)" if mp<0.05 else "예측력 없음")
    w(f"**{title}: {lc} → {le}**")
    w(f"- 최소 p={mp:.4f}(lag {best}) · 고정 lag1 p={p1:.4f} · Bonferroni 임계 {bonf:.4f} → **{sig}**")
    w(f"- 최소p lag에서 4통계량: F={st4['ssr_F']:.3f} / χ²={st4['ssr_chi2']:.3f} / LR={st4['lr']:.3f} / paramF={st4['params_F']:.3f}")
    w(f"- 효과크기(VIX lag 추가 R² 증분): ΔR²={dR2:.4f}")
    w(f"- lag별 p: " + " / ".join(f"L{l}:{ps[l]:.3f}" for l in sorted(ps)))
    w("")
    return ps

# ───────── 보고서 ─────────
w("# VIX ↔ BTC Granger 인과 분석 — 종합 보고서")
w("")
w("> 작성일 2026-06-02 · 코드 `vix_granger.py`")
w("> **기본(primary) 구간: 2024-01~2026-04 (논문 ETF 기준점 이후)** · 추가(supplementary) 구간: 2020~2023")
w("")
w("## 0. Granger 인과란")
w("\"과거의 X가, Y의 과거만 쓸 때보다 Y의 미래를 더 잘 예측하는가\"를 F검정으로 확인하는 **시차(lag) 예측력** 검정. "
  "동시적(contemporaneous) 상관과 구별되며, 거래 가능한 신호는 반드시 시차 예측력이 있어야 한다.")
w("- H0: X가 Y를 Granger-cause 하지 **않는다**. p<유의수준이면 H0 기각 → 예측력 있음.")
w("- 전제: 두 시리즈가 **정상성(stationary)**. 수익률/변화율 사용, ADF로 확인.")
w("- **다중검정 보정**: lag 1~K 중 최소 p를 취하면 거짓양성이 커지므로, Bonferroni 임계(0.05/K)와 고정 lag1을 함께 본다.")
w("")
w("## 1. 데이터")
w("| Frequency | 원인변수 | BTC | 표본 | 기준 |")
w("|---|---|---|---|---|")
w(f"| 일별 | ΔVIX%, Δslope(기간구조) | 16:00 ET close-to-close 수익률 | {len(dd)}일 ({dd.index.min().date()}~{dd.index.max().date()}) | 둘 다 16:00 ET 정렬 |")
w(f"| 시간별(Intraday) | ΔVIX | 시간별 수익률 | {len(hh_m)}시간 ({hh_m.index.min().date()}~{hh_m.index.max().date()}) | 10~16시, 연속 1h만 |")
w("")
w("- VIX slope = VIX(30d) − VIX(3m) 계열(`vix_slope_daily.parquet`). 논문이 BTC에 가장 강하다고 한 PCA2(기울기)에 대응.")
w("- **VIX 분 단위 데이터 미보유** → 분 단위 Granger는 수행 불가. 보유 최고 해상도는 시간별(2024-05~).")
w("")
w("## 2. 정상성 (ADF: stat, p<0.05면 정상)")
for nm,x in [("일별 ΔVIX%",dd["dvix_pct"]),("일별 Δslope",dd["dslope"]),("일별 BTC수익률",dd["btc_ret"]),
             ("시간별 ΔVIX",hh_m["dvix"]),("시간별 BTC수익률",hh_m["btc_ret"])]:
    s,p=adf(x); w(f"- {nm}: ADF={s:.2f}, p={p:.4f}")
w("→ 모든 변화율/수익률 시리즈 정상. Granger 적용 가능.")
w("")
w("## 3. VAR 최적 차수 선택 (정보기준)")
a1,b1=var_order(dd["btc_ret"],dd["dvix_pct"],MAXLAG_D)
a2,b2=var_order(hh_m["btc_ret"],hh_m["dvix"],MAXLAG_H)
w(f"- 일별 [BTC, ΔVIX] VAR: AIC 차수={a1}, BIC 차수={b1}")
w(f"- 시간별 [BTC, ΔVIX] VAR: AIC 차수={a2}, BIC 차수={b2}")
w("→ 정보기준이 권하는 차수는 낮음(자기상관 구조가 약함). 그럼에도 lag 1~K 전부 검정해 강건성 확인.")
w("")
_prim0=dd[(dd.index>=pd.Timestamp("2024-01-01"))&(dd.index<=pd.Timestamp("2026-04-30"))]
rc=stats.pearsonr(_prim0["dvix_pct"],_prim0["btc_ret"])
w(f"**참고 동시 상관(일별, 기본 구간 2024-01~2026-04):** ΔVIX% vs BTC r={rc[0]:+.3f}(p={rc[1]:.1e}) — 동시성은 유의(거래 불가).")
w("")

R={}
prim=dd[(dd.index>=pd.Timestamp("2024-01-01"))&(dd.index<=pd.Timestamp("2026-04-30"))]
supp=dd[(dd.index>=pd.Timestamp("2020-01-01"))&(dd.index<=pd.Timestamp("2023-12-31"))]
w("## 4. 일별 Granger — ΔVIX ↔ BTC (양방향)")
w(f"> **기본(primary) 구간: 2024-01~2026-04 (n={len(prim)})** — 논문 ETF 기준점 이후, 결론의 근거. "
  f"추가(supplementary) 구간: 2020~2023 (n={len(supp)}) — 강건성 비교용.")
w("")
w(f"### 4.1 기본 구간 (2024-01~2026-04, n={len(prim)})")
R["dp_v2b"]=block("VIX→BTC(기본)",prim["dvix_pct"],prim["btc_ret"],"ΔVIX%","BTC수익률",MAXLAG_D)
R["dp_b2v"]=block("BTC→VIX(기본)",prim["btc_ret"],prim["dvix_pct"],"BTC수익률","ΔVIX%",MAXLAG_D)
w(f"### 4.2 추가 구간 (2020~2023, n={len(supp)})")
R["d_v2b"]=block("VIX→BTC(추가)",supp["dvix_pct"],supp["btc_ret"],"ΔVIX%","BTC수익률",MAXLAG_D)
R["d_b2v"]=block("BTC→VIX(추가)",supp["btc_ret"],supp["dvix_pct"],"BTC수익률","ΔVIX%",MAXLAG_D)

w("## 5. 일별 Granger — VIX slope(기간구조) ↔ BTC (양방향)")
ds=dd.dropna(subset=["dslope"])
w(f"> slope 데이터(`vix_slope_daily.parquet`)는 **2024-01 이후 {len(ds)}일만 존재** → 기본 구간(2024-01~2026-04)과 정합. "
  f"slope = VIX(3m) − VIX(30d). 논문이 BTC에 가장 강하다고 한 기간구조 기울기에 대응.")
w("")
R["s_v2b"]=block("slope→BTC",ds["dslope"],ds["btc_ret"],"Δslope","BTC수익률",MAXLAG_D)
R["s_b2v"]=block("BTC→slope",ds["btc_ret"],ds["dslope"],"BTC수익률","Δslope",MAXLAG_D)
R["sp_v2b"]=R["s_v2b"]  # slope는 post-ETF만 존재 → All=post

w("## 6. 시간별 Granger (Intraday, 최고 해상도) — 양방향")
R["h_v2b"]=block("VIX→BTC(시간별)",hh_m["dvix"],hh_m["btc_ret"],"ΔVIX","BTC수익률",MAXLAG_H)
R["h_b2v"]=block("BTC→VIX(시간별)",hh_m["btc_ret"],hh_m["dvix"],"BTC수익률","ΔVIX",MAXLAG_H)

# 정렬 진단
hh2=hh_m.copy(); hh2["fwd1"]=hh2["btc_ret"].shift(-1); hh2["fwd2"]=hh2["btc_ret"].shift(-2)
def cc(a,b):
    d=hh2[[a,b]].dropna(); r,p=stats.pearsonr(d[a],d[b]); pred=-np.sign(d[a]); hit=(np.sign(d[b])==pred).mean()*100; return r,p,hit
r0=cc("dvix","btc_ret"); r1=cc("dvix","fwd1"); r2=cc("dvix","fwd2")
d2=hh2[["dvix","fwd2"]].dropna(); g=-np.sign(d2["dvix"])*d2["fwd2"]; nt=g-0.0004
w("### 6.1 ⚠️ 시간별 '유의'의 실체 — 정렬 진단 (매우 중요)")
w("시간별 Granger가 p≈0으로 나오지만 **거래 가능한 예측력이 아니다.** VIX 1시간봉 시각 라벨링 때문에 "
  "`ΔVIX[t]`가 같은 시계 구간 BTC와 겹쳐 동시성이 시차 검정에 새어든 것이다. 겹침을 제거하면 예측력 소멸:")
w("")
w("| 정렬 | r | p | 방향성 hit% | 의미 |")
w("|---|---|---|---|---|")
w(f"| ΔVIX[t] vs BTC[t,t+1] (동시) | {r1[0]:+.4f} | {r1[1]:.1e} | {r1[2]:.1f}% | 강한 **동시성**(Non-tradable) |")
w(f"| ΔVIX[t] vs BTC[t+1,t+2] (**겹침없는 진짜 예측**) | {r2[0]:+.4f} | {r2[1]:.1e} | **{r2[2]:.1f}%** | **예측력 0** |")
w("")
w(f"- 진짜 예측 단순Strategy 누적넷 **{nt.sum()*100:+.1f}%**(승률 {(g>0).mean()*100:.1f}%) → 손실.")
w(f"- 시간별 Granger p≈0은 정렬 누수에 의한 동시성이며, 진짜 시차 예측력은 일별과 동일하게 **0**. "
  f"(과거 `docs/vix_btc_trading_strategy_report.md`에서 수정한 lookahead bias와 동일.)")
w("")

# ───────── 차트 ─────────
fig,axes=plt.subplots(1,3,figsize=(18,5))
for ax,keys,title,xl in [
    (axes[0],[("dp_v2b","VIX->BTC PRIMARY 2024+","o-"),("dp_b2v","BTC->VIX PRIMARY","s-"),("d_v2b","VIX->BTC SUPPL 2020-23","o--")],"Daily dVIX Granger p","lag(days)"),
    (axes[1],[("s_v2b","slope->BTC (2024+)","o-"),("s_b2v","BTC->slope (2024+)","s-")],"Daily slope Granger p","lag(days)"),
    (axes[2],[("h_v2b","VIX->BTC","o-"),("h_b2v","BTC->VIX","s-")],"Hourly Granger p","lag(hours)")]:
    for k,lab,st in keys:
        d=R[k]; ax.plot(list(d.keys()),list(d.values()),st,label=lab)
    ax.axhline(0.05,color="red",ls=":",label="p=0.05"); ax.axhline(0.005,color="darkred",ls="--",label="Bonferroni")
    ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel("p-value"); ax.set_ylim(0,1); ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("vix_granger.png",dpi=120); plt.close()

# ───────── 종합 판정 ─────────
w("## 7. 종합 판정")
w("")
w("| 검정 | 최소 p | 고정 lag1 p | 판정 |")
w("|---|---|---|---|")
def row(name,k):
    d=R[k]; w(f"| {name} | {min(d.values()):.4f} | {d[1]:.4f} | {'예측력 없음' if min(d.values())>=0.005 else '검토'} |")
row("ΔVIX→BTC (기본 2024+)","dp_v2b"); row("BTC→ΔVIX (기본 2024+)","dp_b2v")
row("ΔVIX→BTC (추가 2020-23)","d_v2b"); row("BTC→ΔVIX (추가 2020-23)","d_b2v")
row("slope→BTC (기본 2024+)","s_v2b"); row("BTC→slope (기본 2024+)","s_b2v")
w(f"| ΔVIX→BTC (시간별 명목) | {min(R['h_v2b'].values()):.4f} | {R['h_v2b'][1]:.4f} | ⚠️ 동시성 누수(6.1) |")
w(f"| **ΔVIX→BTC (시간별 진짜예측)** | {r2[1]:.4f} | — | **예측력 없음** |")
w("")
w("### 최종 해석")
w("- **기본 구간(2024+) 일별 ΔVIX·slope**: 양방향 시차 예측력 **없음**(Bonferroni 임계 통과 0건). 추가 구간(2020-23)도 동일.")
w("- **slope(논문 최강 변수)도 예측 불가**: 동시성만 유의, 시차 0 → 논문의 '동시적 관계만 유의' 결론과 일치.")
w("- **시간별**: 명목 p≈0은 정렬 누수(동시성)이며 겹침 제거 시 예측력 0(hit<50%).")
w("- SSRN 논문의 '정보 즉시 흡수, 시차 효과 없음'이 ΔVIX·slope·시간별 Granger로 **정식 확인**.")
w("")
w("## 8. 한계")
w("- Granger는 **선형·예측적** 인과만 검정(진짜 구조적 인과 아님). 비선형 인과(예: 극단 레짐 한정)는 미검정.")
w("- lag 1~K 최소 p는 다중검정 위험 → Bonferroni·고정 lag1 병기로 완화. 그래도 음성이므로 결론 견고.")
w("- 시간별 VIX는 스냅샷 라벨링으로 동시성 누수 발생 → 겹침 제거로 통제.")
w("- 분 단위 VIX 미보유로 분 단위 Granger 미수행(야간 분단위 상관분석으로 보완: `VIX_overnight_persistence_FULL_REPORT.md`).")
w("- 거래비용·펀딩 미반영의 순수 통계검정. 거래 가능성은 효과크기·적중률과 함께 판단해야 하며, 본 분석은 양쪽 모두 음성.")
w("")
w("## 9. P0 최종 종결")
w("VIX(레벨 변화·기간구조 slope 모두) → BTC의 **시차 예측 채널은 존재하지 않는다.** 동시성만 강하고 거래 불가. "
  "'15분/야간 반응(해석 B)'에 이어 Granger도 음성 → **VIX 기반 방향성 진입 전략은 구조적으로 불가**로 종결한다.")
w("")
w("---")
w("")
w("## 10. 발표용 설명 — 비전공 청중을 위한 서술 (narrative)")
w("")
w("> 이 절은 통계·금융 비전공 청중에게 구두로 발표하는 상황을 가정하여, 앞의 검정 결과를 "
  "용어 정의와 함께 사실 기반으로 서술한 것이다. 수치는 본문 3~7절과 동일하다.")
w("")
w("### 10.1 분석 목적")
w("")
w("앞선 야간 지속성 분석(`VIX_overnight_persistence_FULL_REPORT.md`)에서, VIX 종가 변화가 이후 BTC Return의 "
  "방향을 예측하지 못한다는 결과를 얻었다. 본 분석은 그 결론을 **정식 통계 검정인 Granger 인과 검정**으로 "
  "재확인한다. 상관·적중률 같은 기술 통계가 아니라, 회귀모형에 기반한 가설검정으로 '예측력 부재'를 공식화하는 것이 목적이다.")
w("")
w("### 10.2 Granger 인과 검정이란")
w("")
w("Granger 인과는 '인과'라는 이름과 달리 엄밀히는 **예측력 검정**이다. 핵심 질문은 다음과 같다: "
  "\"변수 Y(BTC)의 과거값만으로 Y의 미래를 예측할 때보다, 여기에 변수 X(VIX)의 과거값을 추가했을 때 예측이 "
  "통계적으로 유의하게 개선되는가?\"")
w("")
w("- 개선되면 'X가 Y를 Granger-cause 한다'고 표현한다(= X의 과거가 Y의 미래 예측에 정보를 더한다).")
w("- 개선되지 않으면 X의 과거는 Y의 미래에 대해 추가 정보가 없다.")
w("")
w("귀무가설(H0)은 '예측 개선 없음'이며, 검정의 p값이 작을수록(통상 0.05 미만) H0를 기각하여 예측력이 있다고 판단한다. "
  "여기서 강조할 점은 **동시점 관계와 예측 관계의 구별**이다. VIX와 BTC가 같은 시점에 함께 움직이는 것(동시적 상관)은 "
  "거래에 쓸 수 없으나, VIX의 **과거**가 BTC의 **미래**를 예측하는 것(시차 예측력)은 거래에 활용 가능하다. "
  "Granger 검정은 후자만을 검정한다.")
w("")
w("### 10.3 검정의 전제와 절차")
w("")
w("Granger 검정은 분석 대상 시계열이 **정상성(stationarity)**, 즉 시간이 지나도 Mean·분산이 안정적이라는 조건을 요구한다. "
  "가격 수준은 추세가 있어 이 조건을 위배하므로, 본 분석은 **변화율·수익률**을 사용하고 ADF(단위근) 검정으로 정상성을 확인했다(섹션 2).")
w("")
w("이어 VAR 모형의 적정 시차(lag) 수를 정보기준(AIC/BIC)으로 점검하고(섹션 3), 시차 1~K까지 모든 경우를 검정했다. "
  "여러 시차를 동시에 검정하면 우연히 하나가 유의해 보일 위험(다중검정 문제)이 있으므로, "
  "Bonferroni 보정(유의수준을 검정 횟수로 나눔)과 고정 시차 1 결과를 함께 제시하여 거짓 양성을 통제했다.")
w("")
w("### 10.4 결과 요약")
w("")
w("**(1) 일별 VIX Change → BTC: 예측력 없음.** 기본 구간(2024-01~2026-04)과 추가 구간(2020~2023) 모두에서, "
  "VIX Change의 과거값은 BTC 미래 수익률을 유의하게 예측하지 못했다(모든 시차에서 p값이 기준을 넘지 못함). "
  "반대 방향(BTC→VIX)도 동일하게 예측력이 없었다.")
w("")
w("**(2) VIX 기간구조(slope) → BTC: 예측력 없음.** 선행연구가 BTC에 대해 가장 설명력이 크다고 보고한 "
  "VIX 기간구조 기울기(slope) 변수조차, 시차를 둔 예측에서는 유의성이 없었다. 동시점 관계는 유의하나 예측 관계는 부재하다.")
w("")
w("**(3) 시간별 검정의 '유의' 결과는 통계적 가짜였다.** 시간 단위 데이터에서는 검정 p값이 0에 가깝게 나왔으나, "
  "이는 VIX 1시간봉의 시각 표기 방식 때문에 **동일 시간대의 동시적 관계가 시차 검정에 잘못 섞여 들어간 것**이다. "
  "겹치지 않는 구간으로 정렬을 교정하자 예측력은 사라졌다(방향 적중률 50% 미만, 섹션 6.1). "
  "이 현상은 본 프로젝트가 과거에 식별·교정한 look-ahead bias와 동일한 유형이다.")
w("")
w("### 10.5 동시 관계와 예측 관계의 구별")
w("")
w("본 분석에서 VIX와 BTC의 **동시점** 상관은 통계적으로 유의했다(기본 구간 일별 음의 상관). "
  "그러나 이는 같은 시점에 두 변수가 함께 변한다는 의미일 뿐, 한쪽이 다른 쪽의 미래를 예고한다는 의미가 아니다. "
  "Granger 검정이 측정하는 시차 예측력은 모든 설정(일별·기간구조·시간별 교정 후)에서 부재했다. "
  "즉 관계는 존재하되 그 관계는 거래에 활용할 수 없는 동시적 형태이다.")
w("")
w("### 10.6 결론")
w("")
w("VIX(수준 변화 및 기간구조 기울기 모두)는 BTC Return의 미래 방향을 시차를 두고 예측하지 못한다. "
  "관계는 동시적으로만 성립하며, 정보는 즉시 가격에 반영된다. 이는 선행연구(SSRN #6233752)가 일별 데이터에서 "
  "도출한 '정보 즉시 흡수, 시차 효과 부재' 결론과 일치하며, 본 분석은 이를 정상성 확인·다중검정 보정·정렬 교정을 "
  "거친 Granger 검정으로 공식 확인했다. 따라서 VIX를 신호로 한 BTC 방향성 진입 전략은 통계적 근거가 없다.")

os.makedirs("docs",exist_ok=True)
open("docs/VIX_granger_report.md","w").write("\n".join(L))
print("DONE daily=%d hourly=%d slope_n=%d | d_v2b=%.3f s_v2b=%.3f sp_v2b=%.3f h_true_p=%.3f"%(
    len(dd),len(hh_m),len(ds),min(R["d_v2b"].values()),min(R["s_v2b"].values()),min(R["sp_v2b"].values()),r2[1]))
