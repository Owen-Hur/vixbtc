# LSTM Autoencoder Report

> ## 문서 기준 (2026-09-08 갱신)
>
> 이 문서의 모든 수치는 저장소에 커밋된 `artifacts/` 를 로드해 **추론을 재실행하여 확인한 값**입니다.
> 재실행으로 확인되지 않는 수치(재학습 로그, 변동성 배수, BTC-only baseline 비교)는 이 문서에서 제거했습니다.
> **확정값의 출처는 항상 `artifacts/threshold.json` 과 `artifacts/anomaly_signals_*.parquet` 입니다.**
>
> **재현 결과**: `python -m lstm_ae.inference [--is]` 를 커밋된 `model.pt`·`scaler.pkl`·`threshold.json` 으로 실행하면
> `anomaly_signals_is.parquet`(130,848 windows) / `anomaly_signals_oos.parquet`(98,136 windows)이
> **anomaly 판정 불일치 0건**(score 최대 오차 2×10⁻⁷)으로 재현됩니다.
> `residual_analysis.py` 출력도 커밋된 로그와 숫자 단위로 완전히 일치합니다.
>
> **추론 파이프라인은 모델 3개를 사용합니다.**
> 메인 60분 모델(`main_60`, threshold 0.357368) 외에 장초반(09:45~10:28) 전용
> **15분 모델**(threshold 0.574284)과 **30분 모델**(threshold 0.781801)이 함께 동작합니다.
> 그래서 signal 파일의 총 window 수(IS 130,848)가 메인 모델 window 수(115,188)보다 큽니다.
> 아래 5장의 통계는 모두 `main_60` 기준입니다.

## 1. Overview

BTC 체결 데이터 + VIXY 1분봉(7 features)으로 학습한 LSTM Autoencoder의 IS/OOS 결과를 정리한다.

### Model Purpose

- 장중(09:30~15:59 ET) BTC-VIXY 관계의 **이상 구간(anomaly)** 탐지
- HMM이 결정한 메인 포지션에 대해, anomaly 발생 시 포지션 교체 여부 판단에 활용
- Reconstruction error(MSE)가 임계값을 초과하면 anomaly로 판정

---

## 2. Data

| 항목 | 값 |
|---|---|
| BTC 원본 | Binance BTCUSDT 영구선물 틱 체결 (data.binance.vision) |
| VIXY 원본 | VIXY ETF 1분봉 (팀원 제공, 월별 parquet) |
| BTC 전처리 | 틱 → 1분봉 (OHLCV + 체결 통계), UTC → ET 변환, 장중 필터링 |
| VIXY 전처리 | UTC → ET 변환, BTC index에 left join + forward fill (빈 분봉 채움) |
| 총 BTC 1분봉 | 237,510행 (28개월) |
| IS 구간 | 2024-01 ~ 2025-04 (135,720행, 348 거래일) |
| OOS 구간 | 2025-05 ~ 2026-04 (101,790행, 261 거래일) |

### VIXY Data Note

VIXY는 유동성이 낮아 장중 390분 중 14~41%만 실거래가 존재한다.
빈 분봉은 직전 체결가로 forward fill 처리했다 (거래 없으면 마지막 체결가 = 현재가).

### Input Features (7)

| Feature | Description | Source |
|---|---|---|
| `btc_return` | BTC 1분 수익률 | 1분봉 close |
| `trade_imbalance` | 매수/매도 비율 (-1 ~ +1) | 체결 is_buyer_maker |
| `trade_count` | 1분간 체결 건수 | 체결 count |
| `avg_trade_size` | 1분간 평균 체결 규모 | 체결 qty 평균 |
| `vixy_return` | VIXY 1분 수익률 | VIXY 분봉 close |
| `vixy_rolling_std` | VIXY 20분 rolling 표준편차 | VIXY vixy_return |
| `vixy_btc_corr` | VIXY-BTC 20분 rolling 상관계수 | vixy_return × btc_return |

---

## 3. Model Architecture

```
Encoder:
  LSTM(7 → 64)  →  LSTM(64 → 32)  →  latent vector (32-dim)
                                        ↑ last timestep only

Decoder:
  RepeatVector(32 → 60×32)  →  LSTM(32 → 32)  →  LSTM(32 → 64)  →  Dense(64 → 7)

Input/Output shape: (batch, 60, 7)
Loss: MSE
Anomaly score = reconstruction error (MSE per window)
```

| Hyperparameter | Value |
|---|---|
| Window size | 60 (minutes) |
| Encoder hidden | [64, 32] |
| Decoder hidden | [32, 64] |
| Latent dim | 32 |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping patience | 5 |
| Total parameters | 65,223 |

---

## 4. Training

| 항목 | 값 |
|---|---|
| 학습/검증 분할 | 80/20 (시간순, 랜덤 아님) |
| 학습 windows | 92,150 |
| 검증 windows | 23,038 |
| 정규화 | StandardScaler (IS에서 fit, OOS에 frozen 적용) |
| 학습 종료 조건 | Early stopping (patience=5, val_loss 기준), 최고 val_loss 모델 저장 |

> 학습 실행 로그(epoch별 loss, 학습 시간)는 이번 검증에서 재학습을 수행하지 않아 확인할 수 없었으므로 문서에 싣지 않는다.
> 저장된 `model.pt` 로부터 확인 가능한 값(파라미터 수 65,223)만 §3에 기재한다.

### Threshold

- 방법: IS 전체 window의 anomaly score 분포에서 **상위 7.5% (92.5th percentile)**
- 임계값: **0.357368** (`artifacts/threshold.json`)

```
Score distribution (IS main_60, 115,188 windows) — artifacts 재실행 실측:
  Mean:    0.2573
  Std:     1.0166
  92.5%:   0.357368  ← threshold
  Max:   335.6274
```

---

## 5. IS vs OOS Results

모두 커밋된 `artifacts/anomaly_signals_*.parquet` 에서 재계산한 값이다 (`main_60` 기준).

### 5.1 Anomaly Rate

| | IS | OOS |
|---|---|---|
| Total windows | 115,188 | 86,391 |
| Anomaly count | 8,640 | 6,134 |
| **Anomaly rate** | **7.50%** | **7.10%** |
| Trading days | 348 | 261 |
| Daily anomaly % (max) | 77.6% | 84.0% |

### 5.2 Score Distribution

| | IS | OOS |
|---|---|---|
| Mean | 0.2573 | 0.2450 |
| Std | 1.0166 | 0.0862 |
| Max | 335.6274 | 3.7425 |

### 5.3 Anomaly Duration

| | IS | OOS |
|---|---|---|
| **Total episodes** | **800** | **548** |
| **≤ 14분 지속 비중** | **83.2%** | **83.2%** |

단기 에피소드가 지배적이다 — 탐지된 이상 구간의 83.2%가 14분 안에 해소된다.

### 5.4 Top Anomaly Dates

일별 anomaly 비율의 최대치는 IS 77.6%, OOS 84.0%이며, OOS 상위 2일은
**2026-01-19 (MLK Day, 84.0%)** 와 **2026-01-01 (신년, 81.9%)** 로 둘 다 저유동성 휴일이다.
휴일 캘린더를 적용하지 않은 데서 오는 False Positive이다 (§7-5).

---

## 6. Key Findings

### 6.1 Direction Prediction — Not Possible

| | Up | Down |
|---|---|---|
| IS anomaly 구간 | 49.8% | 50.2% |
| OOS anomaly 구간 | 49.5% | 50.5% |

anomaly score와 수익률 부호의 상관계수는 IS/OOS 모두 ≈ 0.00.
VIXY 피처를 추가해도 **Autoencoder 구조로는 anomaly의 방향(상승/하락)을 예측할 수 없다.**
(커밋된 signal parquet + BTC 1분봉으로 재계산한 값이다.)

### 6.2 Position Switch Cost

왕복 거래비용: 수수료 0.10% + 슬리피지 0.30% = **0.80%**.
anomaly 에피소드의 **83.2%가 14분 이하**(§5.3)로, 이 구간에서 포지션을 전환하면 왕복 비용을 회수할 만한
가격 변화가 발생하기 어렵다. 방향 정보가 없으므로(§6.1) 전환 방향을 고를 근거도 없다.

---

## 7. Limitations

1. **방향 예측 불가**: VIXY 추가 후에도 anomaly 방향은 49.5~49.8% 상승. Autoencoder는 구조적으로 "이상 여부"만 판단하며 방향을 알려주지 않음.
2. **단기 노이즈 과다**: Anomaly 에피소드의 83.2%가 14분 이하 지속. 이 구간에서의 포지션 전환은 거래비용만 소모.
3. **VIXY 데이터 희소성**: VIXY 장중 분봉 채움률 14~41%. Forward fill로 보완했으나 실거래 없는 구간의 피처 품질은 제한적.
4. **장 초반 공백**: 09:30~10:28 (59분)은 window 축적 기간으로 anomaly score 없음.
5. **휴일 오탐**: MLK Day, 새해 등 저유동성 날에 False Positive 발생. 휴일 캘린더 미적용.

---

## 8. Strategy Implication

Anomaly 감지 시 **포지션 전환(롱↔숏)**은 다음 이유로 부적합:
- 방향 예측 불가 (상승 49.5~49.8%)
- 83.2%의 anomaly가 14분 이하 → 거래비용 회수 불가

**포지션 청산(flat) 전략**이 현실적:
- 방향을 맞출 필요 없음 (불확실할 때 빠지는 것)
- LSTM-AE의 가치 = "Alpha 생성"이 아닌 "Drawdown 감소"

```
실험군: 메인 포지션 + anomaly 시 청산 + 해소 시 재진입
대조군: 메인 포지션 하루 종일 유지
차이 = LSTM-AE가 "위험 구간을 피한" 효과
```

---

## 9. Artifacts

```
lstm_ae/artifacts/
├── model.pt                      # Trained model weights (main_60, 7 features)
├── model_early_15.pt             # 장초반 15분 모델
├── model_early_30.pt             # 장초반 30분 모델
├── scaler.pkl                    # StandardScaler (fit on IS, 7 features)
├── threshold.json                # main_60 threshold: 0.357368 (92.5th pctl)
├── threshold_early_15.json       # 0.574284
├── threshold_early_30.json       # 0.781801
├── anomaly_signals_is.parquet    # IS signals (130,848 windows, 모델 3종 합산)
└── anomaly_signals_oos.parquet   # OOS signals (98,136 windows, 모델 3종 합산)
```

### Signal Schema (`anomaly_signals_*.parquet`)

| Column | Type | Description |
|---|---|---|
| timestamp (index) | datetime | 1-min bar timestamp (ET) |
| anomaly_score | float64 | Reconstruction error (MSE) |
| is_anomaly | bool | score > threshold (main_60 = 0.357368) |
| trade_date | date | Trading date |

---

## 10. How to Run

```bash
cd btc_project

# Train (BTC + VIXY, 7 features)
python3 -m lstm_ae.train

# Inference (OOS)
python3 -m lstm_ae.inference

# Inference (IS)
python3 -m lstm_ae.inference --is
```

---

## 11. Conclusion

BTC+VIXY LSTM Autoencoder는 장중 이상 구간을 안정적으로 탐지한다(IS 7.50% / OOS 7.10%). 그러나 Autoencoder 구조의 본질적 한계로 **방향 예측은 불가능**하며(상승 49.5~49.8%, corr ≈ 0), 에피소드의 83.2%가 14분 이하로 끝나 포지션 전환 전략은 거래비용 대비 수익성이 없다.

LSTM-AE 모듈의 실전 역할은 **포지션 전환이 아닌 위험 구간 회피(청산)**이며, Trading 모듈에서 anomaly 감지 시 포지션 청산 → 해소 시 재진입 전략으로 Drawdown 감소 효과를 검증해야 한다.
