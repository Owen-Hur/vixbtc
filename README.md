# VIX → BTC : 변동성 기간구조의 BTC 예측력 검증

> **결론부터**: VIX 기간구조(term structure)는 BTC와 **동시적(contemporaneous)으로만** 연결되어 있으며,
> **거래 가능한 시차(lagged) 예측 채널은 존재하지 않는다.**
> Lookahead bias를 구조적으로 배제한 설계에서 재검증한 결과, 초기에 관측된 "슬로프 변화율 65% 적중" 알파는 **소멸**했다.

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

논문의 핵심 주장은 "VIX 기간구조의 **slope 성분(PCA2)** 이 **level 성분(PCA1)** 보다 BTC 수익률을 2.3배 잘 설명한다"입니다.
다만 논문 스스로 **"동시적 관계만 유의하고 시차 효과는 없다"** 고 명시합니다.
이 프로젝트는 그 간극 — *"동시적 관계를 거래 가능한 예측 신호로 바꿀 수 있는가?"* — 를 검증했고, **답은 '아니오'** 였습니다.
(논문 원문 PDF는 저작권 문제로 이 저장소에 포함하지 않습니다. 위 SSRN ID로 조회 가능합니다.)

---

## 프로젝트 서사 — 무엇을 시도했고 왜 실패했는가

이 저장소의 가치는 "수익 나는 전략"이 아니라, **가설이 어떻게 세워지고 어떤 절차로 기각되었는가** 의 기록에 있습니다.

### 1단계 · HMM 3-State 레짐 (초기 설계, 폐기)

최초 설계는 VIX 기간구조 4개 만기(9d/30d/3M/6M)를 PCA로 압축하고 HMM 3-State
(공포 / 경계 / 정상)로 레짐을 진단해 BTC 선물 롱/숏/중립을 결정하는 구조였습니다.

- 설계 문서: [`constitution.md`](constitution.md), [`docs/hmm_pipeline_draft.html`](docs/hmm_pipeline_draft.html)
- HMM 모듈 구현 자체는 **팀원 담당 파트**이며 이 저장소에 없습니다.
- 본인은 이 구조의 **Phase B(장중 이상탐지)** 를 맡아 LSTM Autoencoder를 구현했습니다.

### 2단계 · LSTM Autoencoder 이상탐지 (구현 완료, 알파 생성 실패)

BTC 체결 미시구조 + VIXY 1분봉으로 **BTC-VIXY 관계의 이상 구간**을 탐지하는 비지도 모델.

| 항목 | 값 |
|---|---|
| 입력 | 60분 sliding window × 7 features |
| 구조 | Encoder LSTM(7→64→32) → latent 32 → Decoder LSTM(32→32→64) → Dense(7) |
| 파라미터 | 65,223 |
| 학습 | IS 2024-01~2025-04 (115,188 windows), 80/20 시간순 분할, Early stop @ epoch 32 |
| 임계값 | IS anomaly score 92.5 percentile = 0.3617 |
| 결과 | val loss 0.2420 (BTC-only 대비 **-33%**), OOS 고강도 anomaly 변동성 **16.9배** |

**모델은 잘 작동합니다 — 변동성 탐지기로서.** 그러나 트레이딩에는 쓸 수 없었습니다.

| 시도 | 결과 |
|---|---|
| anomaly → 포지션 청산 | OOS Alpha **-114%p** (수익 구간을 버림) |
| anomaly → 레버리지 축소/증가 | OOS Alpha -96 ~ -104%p |
| anomaly score 기반 연속 사이징 | OOS Alpha -297%p (거래비용 폭증) |
| XGBoost 방향 분류기 | OOS 정확도 **50.8%** (랜덤) |
| 잔차 분해 모멘텀 | OOS 54.3% — 유의하나 단독 신호로 너무 약함 |

**근본 원인**: Autoencoder는 "이상 여부"만 알려주고 **방향을 알려주지 않습니다** (anomaly 구간 상승:하락 = 50:50).
게다가 anomaly의 **81%가 14분 이하** 로 지속되어 왕복 거래비용(0.8%)조차 회수하지 못합니다.

→ 상세: [`lstm_ae/REPORT.md`](lstm_ae/REPORT.md), [`lstm_ae/PIPELINE.md`](lstm_ae/PIPELINE.md)

### 3단계 · slope_change 일봉 신호 (허위 알파 → 기각) ⚠️ 이 저장소의 핵심 교훈

LSTM-AE의 장중 개입이 전부 실패하자, 신호를 **일봉 레벨** 로 끌어올렸습니다.

```
slope        = VIX3M - VIX
slope_change = slope[T] - slope[T-1]

slope_change > 0  →  Long BTC     (contango 심화 = risk-on)
slope_change < 0  →  Short BTC    (backwardation 방향 = risk-off)
```

**초기 결과는 매우 좋아 보였습니다**: OOS 적중률 65.0%, Net Return +457.6%, Sharpe 5.31
(→ [`docs/strategy_report_2026-05-24_superseded.md`](docs/strategy_report_2026-05-24_superseded.md), **폐기된 문서**).

이후 시간축을 **T-day 체계**로 재정의해 lookahead bias를 구조적으로 배제하고 재검증했습니다.

> **T-day 정의** — VIX 종가는 절대시각 **16:15 ET** 에 확정된다.
> T-day 0:00 ≡ 16:16 ET (신호 확정 직후) / T-day 17:14 ≡ 익일 09:30 ET (다음 장 개장) / T-day 23:43 ≡ 익일 15:59 ET.
> 이 시간축에서는 예측변수가 종속변수보다 **항상 선행** 하므로 lookahead가 발생할 수 없다.

| 측정 | IS | OOS | ALL |
|---|---|---|---|
| 다음 장중 방향 적중률 | 50.5% | 50.4% | 50.5% (binomial p = 0.836) |
| T-day 1,035개 시점 평균 적중률 | ~47% | ~49% | ~47% |
| \|slope_change\| 강도별 (5구간) | 48~51% — **H2(강도 가설) 기각** | | |
| **Lookahead 미제거 시(참고)** | | **61.8%** | 56.2% ← 허위 성과 |

**알파는 lookahead bias였습니다.** 편향을 제거하자 동전 던지기로 수렴했습니다.
재현 코드: [`analysis/slope_change/compute_biased.py`](analysis/slope_change/compute_biased.py) (편향 버전 61.8% 재현) vs
[`analysis/slope_change/slope_change_analysis.py`](analysis/slope_change/slope_change_analysis.py) (수정 버전 50.5%).

→ 상세: [`analysis/slope_change/slope_change_report.md`](analysis/slope_change/slope_change_report.md)

### 4단계 · VIX Duration — "영향이 얼마나 오래 남는가"

VIX는 장 마감 후 동결되지만 BTC는 24시간 거래됩니다. 그 야간 17시간 동안 신호가 남아있는지
**1,035개 분 단위 시점** 에서 상관계수·적중률·Bonferroni 보정 검정을 수행했습니다.

| 구간 | n | 평균 적중률 | Bonferroni 통과 시점 |
|---|---|---|---|
| ETF 후 (2024-01~2026-04) | 601 | 47.5% | **0개** |
| ETF 전 (2020-01~2023-12) | 1,017 | 49.7% | 249개 (상관은 유의) |
| 2σ 극단 이벤트 (ETF 후) | 24 | 전 시점 p > 0.05 | — |

> **핵심 통찰: 변동성 동조 ≠ 방향 예측.**
> ETF 전 구간은 r ≈ -0.18, p < 10⁻⁹ 로 상관이 **극도로 유의**하지만 적중률은 49.7%입니다.
> 상관계수가 아무리 유의해도 방향이 맞지 않으면 거래 엣지가 아닙니다.

BTC 현물 ETF 승인(2024-01) 이후에는 그 미약한 흔적조차 사라졌습니다 — 기관 자금 유입으로 정보 처리 속도가 빨라진 것으로 해석됩니다.

→ 상세: [`analysis/vix_duration/vix_duration_report.md`](analysis/vix_duration/vix_duration_report.md)

### 5단계 · Granger 인과 검정으로 종결

야간 분석의 "예측 불가" 결론을 정식 통계검정으로 확정했습니다.
ΔVIX·기간구조 slope·시간대별로 양방향 Granger 인과를 검정한 결과 유의한 시차 인과는 확인되지 않았습니다.

→ [`analysis/vix_overnight_granger/reports/VIX_granger_report.md`](analysis/vix_overnight_granger/reports/VIX_granger_report.md),
  [`analysis/vix_overnight_granger/reports/VIX_overnight_persistence_FULL_REPORT.md`](analysis/vix_overnight_granger/reports/VIX_overnight_persistence_FULL_REPORT.md)

### 6단계 · 외부 신호(VIX) → 내부 신호(RV)로의 전환 시도

VIX 채널이 닫혔으므로 BTC 자체의 실현변동성(Realized Volatility) 레짐으로 방향을 틀었습니다.

- 초기 RV Regime 전략: Walk-Forward Sharpe 0.69, B&H 대비 +20.7%p — **그러나 3가지 편향 내재**
  (① 당일 데이터를 포함한 expanding percentile ② 펀딩비 미반영 ③ 1.6년의 짧은 검증구간)
- 편향 3개를 모두 제거하자 → **Sharpe 0.02, B&H 대비 -109%p. 알파 소멸.**
- 고정 파라미터 검증(Train 2020-22 → OOS 2023-26): OOS Sharpe 0.63이나 B&H 대비 -173.9%p

→ [`analysis/rv_regime/README_first_alpha_search.md`](analysis/rv_regime/README_first_alpha_search.md),
  [`docs/strategy_journal.md`](docs/strategy_journal.md)

---

## 최종 결론

1. **VIX → BTC 시차 예측력은 존재하지 않는다.** slope_change(50.5%), VIX Duration(47.5%), Granger,
   임계 필터 + 부호 데이터 결정([`analysis/vix_threshold/`](analysis/vix_threshold/)) — 네 가지 독립적 접근이 모두 같은 결론에 도달했습니다.
2. **동시적 관계와 시차 예측은 전혀 다른 것이다.** 논문이 발견한 것은 전자, 트레이딩이 요구하는 것은 후자입니다.
3. **상관계수의 유의성은 거래 엣지를 보장하지 않는다.** p < 10⁻⁹ 이면서 적중률 49.7%가 실제로 관측됩니다.
4. **미세한 편향들이 합쳐지면 가짜 알파를 만든다.** 같은 데이터에서 lookahead 제거 전 61.8% → 제거 후 50.4%.
5. **BTC 24시간 거래 구조 자체가 정보를 즉시 흡수한다.** VIX 종가가 확정되는 시점에 반영은 이미 끝나 있습니다.
6. **크립토에서 B&H는 매우 강한 벤치마크다.** 숏을 포함한 전략은 구조적으로 불리하며, 펀딩비(연 ~12%)가 얇은 엣지를 소멸시킵니다.

---

## 저장소 구조

```
.
├── README.md                        # 이 문서
├── constitution.md                  # 팀 설계 원칙 / IS-OOS 규약 / 확정·미결 사항
├── requirements.txt
│
├── lstm_ae/                         # ★ LSTM Autoencoder 이상탐지 모듈 (본인 핵심 구현)
│   ├── config.py                    #   하이퍼파라미터 · 경로 · IS/OOS 구간
│   ├── dataset.py                   #   로딩 · 피처 계산 · 60분 윈도우 생성
│   ├── model.py                     #   LSTMAutoencoder
│   ├── train.py                     #   학습 + 임계값 산출
│   ├── inference.py                 #   anomaly_signals 생성
│   ├── backtest*.py                 #   포지션 전환/청산/레버리지 백테스트 (전부 기각)
│   ├── direction_model.py           #   XGBoost 방향 분류기 (기각)
│   ├── residual_analysis.py         #   잔차 분해 (약한 방향 정보 발견)
│   ├── exit_analysis.py             #   청산 시점 + GARCH 분석 (기각)
│   ├── vol_trading_backtest.py      #   변동성 브레이크아웃 (기각)
│   ├── REPORT.md / PIPELINE.md      #   결과 리포트 / 파이프라인 명세
│   └── artifacts/                   #   학습된 모델 · scaler · threshold · anomaly signals
│
├── analysis/
│   ├── slope_change/                # ★ slope_change 검증 (T-day, lookahead-free) — 50.5%
│   ├── vix_duration/                # ★ VIX 유효시간 1,035시점 분석 — 47.5%
│   ├── vix_overnight_granger/       #   야간 지속성 + Granger 인과 (P0 종결)
│   ├── vix_threshold/               #   임계 필터 + 부호 데이터 결정 재검증
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
python -m lstm_ae.inference         # OOS 추론 → anomaly_signals_oos.parquet
python -m lstm_ae.inference --is    # IS 추론
python -m lstm_ae.backtest_final    # 확정 전략 vs 비교군 백테스트
```

학습된 artifacts가 이미 포함되어 있으므로 재학습 없이 추론부터 실행할 수 있습니다.

### 4. 핵심 검증 재현

```bash
cd analysis/slope_change
python slope_change_analysis.py     # lookahead-free → IS 50.5% / OOS 50.4%
python compute_biased.py            # 편향 미제거 → OOS 61.8% (대조군)

cd ../vix_duration
python vix_duration_analysis.py     # 1,035시점 r·적중률 + Bonferroni
python regen_2sigma.py              # 2σ 이벤트 평균 경로

cd ../vix_overnight_granger
python code/vix_overnight_full_report.py
python code/vix_granger.py
```

> 분석 스크립트들은 저장소 루트의 `data/` 를 기준 경로로 참조합니다. 루트에서 실행하거나 스크립트 내 경로를 조정하세요.

---

## IS / OOS 분리 원칙

프로젝트 전 구간에 [`constitution.md`](constitution.md) 의 80/20 시간순 분할 규약을 적용했습니다.

```
전체 : 2024-01-15 ~ 2026-04-30   (~27.5개월, ~583 거래일)
IS   : 2024-01-15 ~ 2025-10-31   (~460일, 79%)  ← 모델 학습 · 파라미터 탐색
OOS  : 2025-11-01 ~ 2026-04-30   (~123일, 21%)  ← 최종 1회 평가 전용
```

- OOS는 파라미터 탐색 완료 후 **단 1회** 평가에만 사용합니다. 탐색 중 OOS 접근 시 해당 실험은 전면 무효 처리했습니다.
- LSTM-AE 모듈은 데이터 가용성 때문에 IS 2024-01~2025-04 / OOS 2025-05~2026-04 를 사용합니다 (`lstm_ae/config.py`).
- `scaler`·`threshold` 는 IS에서만 fit하고 OOS에는 frozen 상태로 transform만 적용합니다.
- VIX Duration 분석은 ETF 도입(2024-01) 전후를 구조적 분기점으로 삼아 별도 구간으로 비교했습니다.

---

## 한계 및 주의사항

1. **본 저장소의 성과 수치 중 일부는 폐기되었습니다.**
   [`docs/strategy_report_2026-05-24_superseded.md`](docs/strategy_report_2026-05-24_superseded.md) 와
   [`results/is_backtest_report.html`](results/is_backtest_report.html) 는 lookahead bias 수정 **이전** 의 결과이며,
   기록 보존 목적으로만 포함했습니다. 유효한 결론은 `analysis/slope_change/` 와 `analysis/vix_duration/` 입니다.
2. **VIXY 데이터 희소성**: VIXY 장중 분봉 채움률이 14~41%에 불과해 forward fill로 보완했습니다.
   실거래가 없는 구간의 피처 품질은 제한적입니다.
3. **휴일 캘린더 미적용**: MLK Day, 신정 등 저유동성일에 LSTM-AE False Positive가 발생합니다.
4. **펀딩비 반영 범위**: RV Regime 분석에는 반영했으나 일부 초기 백테스트에는 미반영입니다.
5. **표본 크기**: OOS 123 거래일은 통계적으로 넉넉하지 않습니다. 다만 결론이 *"엣지 없음"* 이므로
   표본 부족은 결론을 약화시키는 방향이 아닌 보수적 방향으로 작용합니다.
6. **기반 논문은 peer review를 거치지 않은 SSRN preprint** 입니다.

---

## 라이선스 / 이용

Y-FoRM 26-1 2차 프로젝트의 학습·연구 목적 산출물입니다. 투자 조언이 아니며, 실거래 사용을 권장하지 않습니다.
