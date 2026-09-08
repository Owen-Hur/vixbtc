# VIX → BTC : 변동성 기간구조의 BTC 예측력 검증

> **결론부터**: VIX 기간구조(term structure)는 BTC와 **동시적(contemporaneous)으로만** 연결되어 있으며,
> **거래 가능한 시차(lagged) 예측 채널은 존재하지 않는다.**
> Lookahead bias를 구조적으로 배제한 설계에서 재검증한 결과, 초기에 관측된 "슬로프 변화율 60%대 적중" 알파는 **소멸**했다.
>
> 이 README와 각 분석 문서가 제시하는 결론 수치는 **2026-09-08에 코드를 재실행해 확인했거나 커밋된 아티팩트에서 직접 읽은 값**입니다.
> 검증을 시도했으나 생성 코드·아티팩트가 남아 있지 않아 확인할 수 없었던 수치는 **각주로 남기지 않고 문서에서 제거**했습니다.
> → [재현 검증 결과표](#재현-검증-결과표-2026-09-08)

---

## 이 저장소에 대하여

Y-FoRM 26-1 2차 프로젝트 **"From VIX to Bitcoin — Testing the Predictive Power of Volatility Term Structure"** 의
작업물 중, **본인(Owen-Hur / wshh 브랜치)이 담당한 파트**를 정리한 개인 아카이브입니다.

- **팀 프로젝트**: 3인 공동 진행 (본인 / Luca / minsung). 팀 원본 저장소는 별도이며, 이 저장소에는 본인 브랜치의 산출물만 담았습니다.
- **본인 담당**: LSTM Autoencoder 이상탐지 모듈, slope_change 신호 검증, VIX Duration(유효시간) 분석,
  Overnight/Granger 인과 검정, RV Regime 대안 전략, 데이터 파이프라인.
- **팀원 담당(미포함)**: HMM 3-State 레짐 모듈 등.

### 기반 논문

> Jie Luo, Wei-Tse Tsai, Kuang-Chieh Yen (2026).
> *Volatility Transmission to Bitcoin: The Role of VIX Term Structure and Crypto Options Markets.*
> SSRN Preprint (ssrn-6233752).

논문의 핵심 주장은 **"VIX 기간구조 안에 그 날의 방향성 정보가 담겨 있다"** — 구체적으로는
기간구조의 **slope 성분(PCA2)** 이 **level 성분(PCA1)** 보다 BTC 수익률을 2.3배 잘 설명한다는 것입니다.
다만 논문 스스로 **"동시적 관계만 유의하고 시차 효과는 없다"** 고 명시합니다.

(논문 원문 PDF는 저작권 문제로 이 저장소에 포함하지 않습니다. 위 SSRN ID로 조회 가능합니다.)

---

## 프로젝트 배경 — 왜 연구 질문이 한 번 바뀌었는가

이 저장소의 가치는 "수익 나는 전략"이 아니라, **가설이 어떻게 세워지고 어떤 절차로 기각되었는가** 의 기록에 있습니다.
그리고 그 기록의 출발점에는, 연구 도중 **연구 질문 자체를 재정의한 전환점** 이 있습니다.

### ① 최초 설계 — "실시간 이상탐지로 선제 대응한다"

팀 전체의 최초 설계는 VIX 기간구조 4개 만기(9d/30d/3M/6M)를 PCA로 압축하고
HMM 3-State(공포 / 경계 / 정상)로 레짐을 진단해 BTC 선물 롱/숏/중립을 결정하는 2단 구조였습니다
(설계 문서: [`constitution.md`](constitution.md), [`docs/hmm_pipeline_draft.html`](docs/hmm_pipeline_draft.html)).
**Phase A(일별 레짐 판정) = HMM 모듈은 팀원 담당** 이라 이 저장소에 없고,
본인은 **Phase B(장중 이상탐지)** 를 맡았습니다.

논문의 주장("일일 VIX 기간구조가 그 날의 방향성 정보를 담고 있다")을 읽고, Phase B에서 세운 목표는
**VIX-BTC 관계를 실시간으로 감시하는 LSTM Autoencoder 이상탐지기** 였습니다.
BTC 체결 미시구조 + VIXY 1분봉으로 두 시장의 정상 관계를 학습해 두고,
그 관계가 깨지는 순간(anomaly)을 장중에 포착해 **포지션을 선제적으로 조정**한다는 구상이었습니다.

모듈 자체는 완성되었고, 아래 사양·결과는 커밋된 아티팩트로 그대로 재현됩니다.

| 항목 | 값 | 재실행 검증 |
|---|---|---|
| 입력 | 60분 sliding window × 7 features (BTC 4 + VIXY 3) | ✅ `config.py` |
| 구조 | Encoder LSTM(7→64→32) → latent 32 → Decoder LSTM(32→32→64) → Dense(7) | ✅ |
| 파라미터 | 65,223 | ✅ `model.pt` state_dict 실측 |
| 학습 | IS 2024-01~2025-04, 115,188 windows(main_60), 80/20 시간순 분할 | ✅ |
| 임계값 | IS anomaly score 92.5 percentile = **0.357368** | ✅ `artifacts/threshold.json` |
| anomaly rate | IS 7.50% (8,640) / OOS 7.10% (6,134) | ✅ 추론 재실행 |
| 방향 정보 | anomaly 구간 상승비율 **IS 49.8% / OOS 49.5%**, corr(score, 부호수익) ≈ 0.00 | ✅ 아티팩트에서 재계산 |
| 지속시간 | anomaly 에피소드의 **83.2%가 14분 이하** | ✅ 아티팩트에서 재계산 |

### ② 전환점 — 논문의 근거 수준과 최초 설계의 목표가 어긋나 있었다

진행 도중 확인한 사실은, **논문의 근거가 실시간 VIX가 아니라 "하루 단위 데이터로 그 날을 사후 평가"하는 수준**이라는 점이었습니다.
논문은 일별 종가 기준의 동시적 관계를 보고했을 뿐이며, 장중 어느 시점에 어떤 순서로 정보가 흘렀는지는 다루지 않았습니다.

즉, "실시간 이상탐지 → 장중 선제 대응"이라는 최초 설계는
**논문이 실제로 뒷받침할 수 있는 주장의 해상도(일별·사후)를 넘어서 있었습니다.**
LSTM-AE가 만들어낸 결과들도 같은 지점을 가리켰습니다 — 이상 구간을 잘 찾아내지만 **방향을 알려주지 않고**(상승 비율 IS 49.8% / OOS 49.5%),
탐지된 anomaly의 **83.2%가 14분 이하** 로 끝나 왕복 거래비용조차 회수하지 못했습니다.

> **이 단계는 "모델 실패"가 아니라 "질문의 재정의 계기"였습니다.**
> 실시간 개입이 통하지 않는다는 사실 자체보다 중요했던 것은,
> *근거 논문이 지지할 수 있는 시간 해상도가 어디까지인가* 를 확인한 것입니다.

### ③ 재정의된 연구 질문 — overnight 방향성

그래서 질문을 논문의 근거 수준에 맞춰 다시 세웠습니다.

> **"장마감 시점(16:15 ET)에 발표되는 그 날의 VIX 기간구조가,
> 장마감 이후 overnight 동안의 BTC 방향성에 영향을 주는가?"**

이 질문은 두 가지 이유로 자연스러웠습니다.

1. **VIX 기간구조는 장마감에 확정된다.** 그 시점 이후는 예측변수가 종속변수보다 항상 선행하므로 lookahead가 구조적으로 불가능합니다.
2. **BTC는 24시간 거래된다.** 미국 장이 닫혀 있는 17시간 동안 BTC만 홀로 움직이므로, 정보가 남아 있다면 이 구간에서 관측되어야 합니다.

이 재정의로부터 **T-day 시간축**이 나왔고, 이후의 모든 검증(slope_change, VIX Duration, Overnight/Granger)이 이 축 위에서 진행되었습니다.

```
T-day 정의 — VIX 종가는 절대시각 16:15 ET 에 확정된다.
  T-day  0:00 ≡ 16:16 ET (신호 확정 직후 첫 BTC 측정)
  T-day 17:14 ≡ 익일 09:30 ET (다음 장 개장)
  T-day 23:43 ≡ 익일 15:59 ET (다음 장 마감 직전)
→ 예측변수가 종속변수보다 항상 선행하므로 lookahead가 발생할 수 없다.
```

### ④ 결과 — 스스로 세운 알파를 스스로 반증하다

재정의된 질문에 대한 답은 **"아니오"** 였습니다. 그리고 그 과정에서,
**편향을 제거하기 전 매우 좋아 보였던 신호가 편향의 산물이었음** 을 확인했습니다.
아래 네 단계가 그 기록입니다.

---

## 검증 단계별 기록 — 재정의된 질문에 대한 답

### 1단계 · slope_change 일봉 신호 (허위 알파 → 기각) ⚠️ 이 저장소의 핵심 교훈

```
slope        = VIX3M - VIX
slope_change = slope[T] - slope[T-1]

slope_change > 0  →  Long BTC     (contango 심화 = risk-on)
slope_change < 0  →  Short BTC    (backwardation 방향 = risk-off)
```

**편향 제거 이전의 결과는 매우 좋아 보였습니다.**
당시(2026-05-24) 리포트는 이 신호를 확정 전략으로 채택했습니다
(→ [`docs/strategy_report_2026-05-24_superseded.md`](docs/strategy_report_2026-05-24_superseded.md), **폐기된 문서** —
그 리포트의 백테스트 성과 수치는 생성 코드가 남아 있지 않아 재현할 수 없었고, 이 저장소에서 제거했습니다).
재현 가능한 형태로 남아 있는 편향 대조군(당일 09:30→15:59 측정)은 OOS **61.8%** 입니다.

이후 시간축을 **T-day 체계**로 재정의해 lookahead bias를 구조적으로 배제하고 재검증했습니다.

| 측정 | IS | OOS | ALL |
|---|---|---|---|
| 다음 장중 방향 적중률 (T-day 17:14 → 23:43) | 50.5% (n=457) | 50.4% (n=123) | 50.5% (n=580, binomial p = 0.836) |
| T-day 1,035개 시점 평균 적중률 | 46.3% | 46.3% | 46.3% |
| \|slope_change\| 강도별 (5구간, ALL) | 48.1 ~ 51.4% — **H2(강도 가설) 기각** | | |
| **Lookahead 미제거 대조군(당일 09:30→15:59)** | 54.7% | **61.8%** | 56.2% ← 허위 성과 |

**알파는 lookahead bias였습니다.** 편향을 제거하자 동전 던지기로 수렴했습니다.
재현 코드: [`analysis/slope_change/compute_biased.py`](analysis/slope_change/compute_biased.py) (편향 대조군) vs
[`analysis/slope_change/slope_change_analysis.py`](analysis/slope_change/slope_change_analysis.py) (수정 버전).

→ 상세: [`analysis/slope_change/slope_change_report.md`](analysis/slope_change/slope_change_report.md)

### 2단계 · VIX Duration — "영향이 얼마나 오래 남는가"

VIX는 장 마감 후 동결되지만 BTC는 24시간 거래됩니다. 그 야간 17시간 동안 신호가 남아있는지
**1,035개 분 단위 시점** 에서 상관계수·적중률·Bonferroni 보정 검정을 수행했습니다.

| 구간 | n | 평균 적중률 | p<0.05 시점 | Bonferroni 통과 시점 |
|---|---|---|---|---|
| ETF 후 (2024-01~2026-04) | 601 | 47.48% | 10 / 1,035 | **0개** |
| ETF 전 (2020-01~2023-12) | 1,017 | 49.65% | 915 / 1,035 | 249개 (상관은 유의) |
| 2σ 극단 이벤트 (ETF 후) | 24 (spike 15 / drop 9) | 전 시점 p > 0.05 | — | — |

> **핵심 통찰: 변동성 동조 ≠ 방향 예측.**
> ETF 전 구간의 동시적 상관은 r = **-0.180**, p = **7.3×10⁻⁹** 로 극도로 유의하지만 야간 적중률은 49.7%입니다.
> 상관계수가 아무리 유의해도 방향이 맞지 않으면 거래 엣지가 아닙니다.

BTC 현물 ETF 승인(2024-01) 이후에는 그 미약한 흔적조차 사라졌습니다 — 기관 자금 유입으로 정보 처리 속도가 빨라진 것으로 해석됩니다.

→ 상세: [`analysis/vix_duration/vix_duration_report.md`](analysis/vix_duration/vix_duration_report.md)

### 3단계 · Granger 인과 검정으로 종결

야간 분석의 "예측 불가" 결론을 정식 통계검정으로 확정했습니다.
ΔVIX·기간구조 slope·시간대별로 **양방향** Granger 인과를 검정했고, **Bonferroni 임계를 통과한 시차 인과는 한 건도 없었습니다.**

| 검정 | 최소 p | 고정 lag1 p | 판정 |
|---|---|---|---|
| ΔVIX→BTC (일별, 2024+, n=601) | 0.3655 | 0.3655 | 예측력 없음 |
| BTC→ΔVIX (일별, 2024+) | 0.2601 | 0.5118 | 예측력 없음 |
| ΔVIX→BTC (일별, 2020-23, n=1,017) | 0.0914 | 0.8638 | 예측력 없음 |
| **Δslope→BTC (논문 최강 변수, n=566)** | **0.6100** | 0.8766 | **예측력 없음** |
| ΔVIX→BTC (시간별, 명목) | 0.0000 | 0.0000 | ⚠️ 정렬 누수에 의한 **동시성** |
| **ΔVIX→BTC (시간별, 겹침 제거한 진짜 예측)** | **0.9954** | — | **예측력 없음 (hit 47.2%)** |

시간별 Granger가 p≈0으로 나오는 것은 VIX 1시간봉의 시각 라벨링 때문에 동시성이 시차 검정에 새어든 것으로,
겹침을 제거하면 r = -0.0001 (p = 1.0), 적중률 47.2%, 단순전략 누적넷 -159.2%로 예측력이 소멸합니다.
동시적 상관은 r = -0.3902 (p = 7.7e-127)로 강하지만 거래 불가입니다.

→ [`analysis/vix_overnight_granger/reports/VIX_granger_report.md`](analysis/vix_overnight_granger/reports/VIX_granger_report.md),
  [`analysis/vix_overnight_granger/reports/VIX_overnight_persistence_FULL_REPORT.md`](analysis/vix_overnight_granger/reports/VIX_overnight_persistence_FULL_REPORT.md)

임계 필터 + 부호를 데이터로 학습시키는 대안 결정규칙도 시도했으나,
Bonferroni 보정(p < 0.0071) 후 OOS에서 50%를 유의하게 넘는 임계는 **0개** 였습니다
(베이스라인 단일 룰: 적중률 48.6%, 누적넷 -106.2%).
→ [`analysis/vix_threshold/`](analysis/vix_threshold/)

### 4단계 · 외부 신호(VIX) → 내부 신호(RV)로의 전환 시도

VIX 채널이 닫혔으므로 BTC 자체의 실현변동성(Realized Volatility) 레짐으로 방향을 틀었습니다.

- 초기(v1) RV Regime 전략은 유망해 보였으나 **3가지 편향이 내재** 해 있었습니다
  (① 당일 데이터를 포함한 expanding percentile ② 펀딩비 미반영 ③ 1.6년의 짧은 검증구간).
  v1의 원본 코드는 남아 있지 않아 당시 성과 수치는 이 저장소에 싣지 않습니다.
- 편향 3개를 모두 제거하자 → **누적 -57.4%, Sharpe 0.02, MDD -84.7%, B&H(+51.6%) 대비 -109.0%p. 알파 소멸.**
  (22 WF 라운드, 2021-01 ~ 2026-05, 1,964일 — **재실행으로 완전 재현 확인**)
- 고정 파라미터 검증(Train 2020-22 → OOS 2023-26, sw=4/lw=14/qh=0.60): OOS +89.4%, Sharpe 0.63이나 B&H 대비 -173.9%p
  (생성 코드: [`analysis/other_signals/04_fixed_param_with_funding.py`](analysis/other_signals/04_fixed_param_with_funding.py))
- 펀딩비 실측: 일평균 0.0328% → **연 ~12%**. 전략이 82.5% Long인 구조에서 얇은 엣지를 소멸시킵니다.

→ [`analysis/rv_regime/README_first_alpha_search.md`](analysis/rv_regime/README_first_alpha_search.md),
  [`docs/strategy_journal.md`](docs/strategy_journal.md)

---

## 재현 검증 결과표 (2026-09-08)

원본 프로젝트를 별도 사본으로 복사하고 Python 3.12 가상환경(pandas 3.0.5 / numpy 2.5.3 / torch 2.14 / statsmodels / scikit-learn)에서
**문서에 적힌 수치를 실제로 재실행해 대조**했습니다. 불일치가 있으면 **재실행 결과를 확정값으로 채택**했습니다.

| 항목 | 문서 수치 | 재실행 수치 | 판정 |
|---|---|---|---|
| slope_change 다음 장중 적중률 (IS/OOS/ALL) | 50.5 / 50.4 / 50.5%, p=0.836 | **동일** | ✅ 완전 일치 |
| slope_change 편향 대조군 (당일 장중) | OOS 61.8%, ALL 56.2% | **동일** | ✅ 완전 일치 |
| slope_change T-day 1,035시점 평균 적중률 | ~47% / ~49% / ~47% | **46.29 / 46.30 / 46.29%** | ⚠️ **문서 오기 → 정정** |
| VIX Duration 평균 적중률 (ETF 후/전) | 47.5% / 49.7% | **47.48% / 49.65%** | ✅ 일치 |
| VIX Duration Bonferroni 통과 시점 | 0개 / 249개 | **동일** | ✅ 완전 일치 |
| VIX Duration 동시 상관 (2020-23) | r ≈ -0.18, **p < 10⁻⁹** | r = -0.1801, **p = 7.29×10⁻⁹** | ⚠️ **자릿수 정정 (p < 10⁻⁸)** |
| Granger 인과 리포트 전문 | — | **바이트 단위 동일 재생성** | ✅ 완전 재현 |
| Overnight 지속성 리포트 2종 | — | **바이트 단위 동일 재생성** | ✅ 완전 재현 |
| vix_threshold 리포트 | OOS 통과 임계 0개, 베이스 48.6% | **바이트 단위 동일 재생성** | ✅ 완전 재현 |
| RV Regime 편향 제거 후 | -57.4%, Sharpe 0.02, MDD -84.7%, -109.0%p | **동일** | ✅ 완전 일치 |
| LSTM-AE anomaly signal (IS/OOS) | 130,848 / 98,136 windows | **동일 (score 최대 오차 2×10⁻⁷, 판정 불일치 0건)** | ✅ 완전 재현 |
| LSTM-AE 임계값 | 구 REPORT.md **0.361667** | artifacts **0.357368** | ⚠️ **구버전 수치 → artifacts 값으로 문서 정정** |
| LSTM-AE OOS anomaly rate | 구 REPORT.md 6.5% (5,616건) | **7.10% (6,134건)** | ⚠️ **구버전 수치 → 문서 정정** |
| LSTM-AE anomaly 지속시간 ≤14분 비중 | 구 REPORT.md 81% | **83.2% (IS·OOS 공통)** | ⚠️ **구버전 수치 → 문서 정정** |
| LSTM-AE 파라미터 수 | 65,223 | **65,223** (`model.pt` state_dict 실측) | ✅ 일치 |
| anomaly 구간 방향 (상승:하락) | 50:50 | **IS 49.8% / OOS 49.5% 상승** | ✅ 일치 |
| XGBoost 방향 분류기 OOS 정확도 | 50.8% | **50.78%** (`direction_meta.json`) | ✅ 일치 |
| 잔차 모멘텀 OOS 15분 적중률 | 54.3% | **54.3%** (`residual_analysis.py` 재실행, 로그와 숫자 완전 일치) | ✅ 완전 재현 |

### 이번 재실행으로 확정한 사항 (이전 정리에서 미해결로 남겨둔 각주)

1. **`STRATEGY_REPORT.md` vs `slope_change_report.md` 상충 → 후자가 확정.**
   전자(2026-05-24)의 백테스트 성과는 lookahead 미제거 상태의 결과이고, 생성 스크립트도 남아 있지 않아 재현할 수 없어 문서에서 제거했습니다.
   유효한 값은 후자(2026-06-02, T-day 축)의 **50.4%** 이며, **재현 가능한 편향 대조군의 최댓값은 61.8%** (`compute_biased.py`, 동일 구간·동일 표본 n=123)입니다.

2. **LSTM-AE 임계값 0.361667 vs 0.3574 → 0.357368 확정.**
   저장소에 커밋된 `artifacts/threshold.json` · `model.pt` · `scaler.pkl` 로 추론을 재실행하면
   `anomaly_signals_is/oos.parquet` 이 **판정 불일치 0건으로 재현**됩니다.
   `lstm_ae/REPORT.md` 에 적혀 있던 0.361667 / score mean 0.2596 / OOS anomaly 6.5% 는 **재학습 이전 버전의 수치**여서
   아티팩트 실측값(threshold **0.357368**, IS score mean **0.2573**, OOS anomaly rate **7.10%**)으로 교체했습니다.
   (`analysis/rv_regime/results/lstm_ae_residual_analysis.log` 의 "Anomaly windows: 6,134 (7.1%)" 도 아티팩트 쪽과 일치합니다.)

3. **`PIPELINE.md` vs `REPORT.md` feature 개수 불일치 → 7개 확정.**
   `PIPELINE.md` 는 VIXY 합류 **이전** 에 작성된 초안이라 피처를 4개 + `vxx_*` 로 기술하고 있었습니다.
   실제 코드(`lstm_ae/config.py`)의 확정값인 **7개 피처**
   (`btc_return, trade_imbalance, trade_count, avg_trade_size, vixy_return, vixy_rolling_std, vixy_btc_corr`)로 문서를 정정했습니다.

4. **문서에 없던 사실 — 추론 파이프라인은 모델 3개를 사용합니다.**
   메인 60분 모델 외에 장초반(09:45~10:28) 전용 **15분 / 30분 모델** 이 별도 임계값(0.574284 / 0.781801)으로 함께 동작합니다.
   그래서 signal 파일의 총 window 수(IS 130,848)가 메인 모델 window 수(115,188)보다 큽니다. REPORT/PIPELINE 어느 쪽에도 기술되어 있지 않았습니다.

### 재현 자료가 없어 문서에서 제거한 항목

아래 항목들은 재현할 코드·아티팩트가 남아 있지 않거나 측정 정의가 문서에 기록되어 있지 않아 확인할 수 없었습니다.
"참고용"으로 남기지 않고 해당 수치와 그에 의존하던 서술을 문서에서 **삭제** 했습니다.

- **`docs/strategy_report_2026-05-24_superseded.md` 의 확정 전략 백테스트 성과** — 생성 스크립트가 저장소·원본 프로젝트 어디에도 없습니다.
  해당 문서에서는 성과 요약·레버리지 스케일링·비교군 섹션을 삭제하고, 재현이 확인된 장중 개입 실패 실험과 잔차 분해만 남겼습니다.
- **LSTM-AE 학습 실행 기록** (val loss / epoch / 학습시간 / loss curve) — CPU 재학습을 수행하지 않아 확인하지 못했습니다.
  대신 커밋된 아티팩트로 **추론을 재실행** 해 임계값·score 분포·anomaly 판정을 검증했고, 파라미터 수 65,223 은 `model.pt` state_dict에서 직접 세었습니다.
- **LSTM-AE 변동성 배수(§6.1)와 BTC-only baseline 비교(§7)** — "고강도(2x+)" 기준과 변동성 측정창이 REPORT.md에 명시되어 있지 않고,
  BTC-only(4 features) 모델 아티팩트도 저장소에 없습니다. 두 섹션을 삭제했습니다.
  (같은 아티팩트에서 재계산한 "방향 정보 없음"(상승 49.5~49.8%, corr ≈ 0)은 확인되어 그대로 남겼습니다.)
- **RV Regime v1(편향 내재 버전)의 성과 수치** — 저장소에 편향 제거 버전만 있어 원본 수치를 재현할 수 없습니다. v1 수치는 전부 삭제하고,
  재현이 확인된 편향 제거판(Sharpe 0.02)만 남겼습니다. `rv_regime/config.py` 는 원본 저장소에 커밋되어 있지 않아
  문서 기재값(taker fee 0.04% 편도 등)으로 복원해 실행했고, 복원한 파일은 출처 주석과 함께
  [`analysis/rv_regime/revised/config.py`](analysis/rv_regime/revised/config.py) 로 포함했습니다.
- **`analysis/other_signals/` · `analysis/vix_response/`** 의 중간 단계 탐색 스크립트 — 최종 결론에 직접 기여하지 않아 재실행하지 않았습니다.

---

## 최종 결론

1. **VIX → BTC 시차 예측력은 존재하지 않는다.** slope_change(50.5%), VIX Duration(47.5%), Granger(Bonferroni 통과 0건),
   임계 필터 + 부호 데이터 학습([`analysis/vix_threshold/`](analysis/vix_threshold/)) — 네 가지 독립적 접근이 모두 같은 결론에 도달했습니다.
2. **동시적 관계와 시차 예측은 전혀 다른 것이다.** 논문이 발견한 것은 전자, 트레이딩이 요구하는 것은 후자입니다.
   시간별 Granger에서 명목 p≈0(동시성)과 진짜 예측 p=0.9954가 나란히 관측된 것이 그 대비입니다.
3. **상관계수의 유의성은 거래 엣지를 보장하지 않는다.** p = 7.3×10⁻⁹ 이면서 적중률 49.7%가 실제로 관측됩니다.
4. **미세한 편향들이 합쳐지면 가짜 알파를 만든다.** 같은 데이터에서 lookahead 제거 전 61.8% → 제거 후 50.4%,
   RV Regime은 편향 3개를 제거하자 Sharpe 0.02 · B&H 대비 -109.0%p로 알파가 사라졌습니다.
5. **연구 설계는 근거의 해상도를 넘어설 수 없다.** 논문의 근거가 일별·사후 평가 수준인데 실시간 장중 개입을 설계한 것이
   이 프로젝트의 첫 번째 구조적 오류였고, 이를 인지해 질문을 재정의한 것이 전환점이었습니다.
6. **BTC 24시간 거래 구조 자체가 정보를 즉시 흡수한다.** VIX 종가가 확정되는 시점에 반영은 이미 끝나 있습니다.
7. **크립토에서 B&H는 매우 강한 벤치마크다.** 숏을 포함한 전략은 구조적으로 불리하며, 펀딩비(실측 연 ~12%)가 얇은 엣지를 소멸시킵니다.

---

## 저장소 구조

```
.
├── README.md                        # 이 문서
├── constitution.md                  # 팀 설계 원칙 / IS-OOS 규약 / 확정·미결 사항
├── requirements.txt
│
├── lstm_ae/                         # ★ LSTM Autoencoder 이상탐지 모듈 (①단계 구현)
│   ├── config.py                    #   하이퍼파라미터 · 경로 · IS/OOS 구간 (7 features 확정)
│   ├── dataset.py                   #   로딩 · 피처 계산 · 60/30/15분 윈도우 생성
│   ├── model.py                     #   LSTMAutoencoder
│   ├── train.py                     #   학습 + 임계값 산출
│   ├── inference.py                 #   anomaly_signals 생성 (main_60 + early_15 + early_30)
│   ├── backtest*.py                 #   포지션 전환/청산/레버리지 백테스트 (전부 기각)
│   ├── direction_model.py           #   XGBoost 방향 분류기 (OOS 50.8% → 기각)
│   ├── residual_analysis.py         #   잔차 분해 (OOS 15분 모멘텀 54.3% — 약한 방향 정보)
│   ├── exit_analysis.py             #   청산 시점 + GARCH 분석 (기각)
│   ├── vol_trading_backtest.py      #   변동성 브레이크아웃 (기각)
│   ├── REPORT.md / PIPELINE.md      #   결과 리포트 / 파이프라인 명세 (artifacts 기준으로 정정 완료)
│   └── artifacts/                   #   학습된 모델 3종 · scaler · threshold · anomaly signals
│
├── analysis/
│   ├── slope_change/                # ★ slope_change 검증 (T-day, lookahead-free) — 50.5%
│   ├── vix_duration/                # ★ VIX 유효시간 1,035시점 분석 — 47.5%
│   ├── vix_overnight_granger/       #   야간 지속성 + Granger 인과 (P0 종결)
│   ├── vix_threshold/               #   임계 필터 + 부호 데이터 학습 재검증
│   ├── vix_response/                #   VIX 1h / 반응함수 / 전략 탐색 (중간 단계 코드·차트)
│   ├── rv_regime/                   #   RV Regime 전환 시도 + 편향 제거 버전(revised/)
│   └── other_signals/               #   펀딩비 · ETF flow · DVOL · Fear&Greed 등 부가 신호
│
├── data_collection/                 # 데이터 수집 스크립트 (전부 무료 공개 소스)
├── data/README.md                   # 데이터 스키마 · 출처 · 재수집 안내 (원본 데이터 미포함)
├── docs/                            # 전처리 가이드 · 전략 일지 · 문헌 리뷰 · 폐기된 전략 리포트
└── results/                         # IS 백테스트 리포트 (중간 단계, 편향 수정 이전)
```

---

## 실행 방법

### 1. 환경 구성

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

> **statsmodels 0.15+ 호환 메모**: `grangercausalitytests(..., verbose=False)` 인자는 최신 statsmodels에서 제거되었습니다.
> `analysis/vix_overnight_granger/code/vix_granger.py` 실행 시 `TypeError` 가 나면 해당 인자를 지우거나 `statsmodels<0.15` 를 설치하십시오.
> (이번 재검증은 인자를 제거한 상태로 실행했고, 리포트는 바이트 단위로 동일하게 재생성되었습니다.)

### 2. 데이터 수집

원본 시장 데이터는 용량 문제로 저장소에 포함하지 않았습니다. 전부 **무료 공개 소스**에서 재수집 가능합니다.
스키마·출처·수집 절차는 [`data/README.md`](data/README.md) 참조.

```bash
python data_collection/fetch_vix_daily.py           # FRED → VIX 일별
python data_collection/fetch_btc_1m_historical.py   # data.binance.vision → BTC 1분봉
python data_collection/fetch_vixy_1m.py             # VIXY 1분봉
# VIX3M은 data_collection/NOTE_vix3m.md 참조 (CBOE 수동 다운로드)
```

### 3. LSTM Autoencoder

```bash
python -m lstm_ae.train             # 학습 → artifacts/model.pt, scaler.pkl, threshold.json
python -m lstm_ae.inference         # OOS 추론 → anomaly_signals_oos.parquet (98,136 windows)
python -m lstm_ae.inference --is    # IS 추론  → anomaly_signals_is.parquet  (130,848 windows)
python -m lstm_ae.backtest_final    # 장중 anomaly 개입 vs 무개입 비교 백테스트 (개입은 전부 기각)
```

학습된 artifacts가 이미 포함되어 있으므로 재학습 없이 추론부터 실행할 수 있습니다.
(재실행 검증 완료: 커밋된 signal parquet과 판정 불일치 0건으로 재현됩니다.)

### 4. 핵심 검증 재현

```bash
cd analysis/slope_change
python slope_change_analysis.py     # lookahead-free → IS 50.5% / OOS 50.4% / ALL 50.5% (p=0.836)
python compute_biased.py            # 편향 대조군    → OOS 61.8% / ALL 56.2%

cd ../vix_duration
python vix_duration_analysis.py     # 1,035시점 r·적중률 (ETF후 47.48% / ETF전 49.65%)
python regen_2sigma.py              # 2σ 이벤트 평균 경로 (spike 15 / drop 9)

cd ../..
python analysis/vix_overnight_granger/code/vix_overnight_full_report.py
python analysis/vix_overnight_granger/code/vix_granger.py
python analysis/vix_threshold/vix_threshold_directional.py
```

> 분석 스크립트들은 저장소 루트의 `data/` 를 기준 경로로 참조합니다. 루트에서 실행하거나 스크립트 상단의 `BASE_DIR` 를 조정하세요.
> `analysis/rv_regime/revised/` 는 `rv_regime` 이라는 이름의 패키지로 import됩니다(디렉토리명 변경 또는 심볼릭 링크 필요).
> 포함된 `config.py` 는 원본 저장소에 없던 파일을 문서 기재값으로 복원한 것이며, 각 값의 출처는 파일 상단 주석에 적어 두었습니다.

---

## IS / OOS 분리 원칙

프로젝트 전 구간에 [`constitution.md`](constitution.md) 의 80/20 시간순 분할 규약을 적용했습니다.

```
전체 : 2024-01-15 ~ 2026-04-30   (~27.5개월, ~583 거래일)
IS   : 2024-01-15 ~ 2025-10-31   (460 T-days, 79%)  ← 모델 학습 · 파라미터 탐색
OOS  : 2025-11-01 ~ 2026-04-30   (123 T-days, 21%)  ← 최종 1회 평가 전용
```

- OOS는 파라미터 탐색 완료 후 **단 1회** 평가에만 사용합니다. 탐색 중 OOS 접근 시 해당 실험은 전면 무효 처리했습니다.
- LSTM-AE 모듈은 데이터 가용성 때문에 IS 2024-01~2025-04 (348 거래일) / OOS 2025-05~2026-04 (261 거래일) 를 사용합니다 (`lstm_ae/config.py`).
- `scaler`·`threshold` 는 IS에서만 fit하고 OOS에는 frozen 상태로 transform만 적용합니다.
- VIX Duration 분석은 ETF 도입(2024-01) 전후를 구조적 분기점으로 삼아 별도 구간으로 비교했습니다.

---

## 한계 및 주의사항

1. **본 저장소의 성과 수치 중 일부는 폐기되었습니다.**
   [`docs/strategy_report_2026-05-24_superseded.md`](docs/strategy_report_2026-05-24_superseded.md) 와
   [`results/is_backtest_report.html`](results/is_backtest_report.html) 는 lookahead bias 수정 **이전** 의 결과이며,
   기록 보존 목적으로만 포함했습니다. 유효한 결론은 `analysis/slope_change/` 와 `analysis/vix_duration/` 입니다.
2. **`lstm_ae/REPORT.md` · `PIPELINE.md` 의 수치는 커밋된 `artifacts/` 기준으로 정정했습니다.**
   확정값은 항상 `lstm_ae/artifacts/` 의 JSON·parquet이며, 아티팩트로 확인되지 않는 값은 두 문서에서 삭제했습니다.
3. **VIXY 데이터 희소성**: VIXY 장중 분봉 채움률이 14~41%에 불과해 forward fill로 보완했습니다.
   실거래가 없는 구간의 피처 품질은 제한적입니다.
4. **휴일 캘린더 미적용**: MLK Day, 신정 등 저유동성일에 LSTM-AE False Positive가 발생합니다
   (OOS 최상위 anomaly 일자 1·2위가 2026-01-19 MLK Day 84.0%, 2026-01-01 신년 81.9%).
5. **펀딩비 반영 범위**: RV Regime 분석에는 반영(실측 일평균 0.0328%)했으나 일부 초기 백테스트에는 미반영입니다.
6. **표본 크기**: OOS 123 거래일은 통계적으로 넉넉하지 않습니다. 다만 결론이 *"엣지 없음"* 이므로
   표본 부족은 결론을 약화시키는 방향이 아닌 보수적 방향으로 작용합니다.
7. **기반 논문은 peer review를 거치지 않은 SSRN preprint** 입니다.

---

## 라이선스 / 이용

Y-FoRM 26-1 2차 프로젝트의 학습·연구 목적 산출물입니다. 투자 조언이 아니며, 실거래 사용을 권장하지 않습니다.
