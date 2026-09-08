# LSTM Autoencoder Report

## 1. Overview

BTC 체결 데이터 + VIXY 1분봉으로 학습한 LSTM Autoencoder의 IS/OOS 결과를 정리한다.
BTC-only baseline(4 features)과 BTC+VIXY(7 features) 두 차례 실험을 수행했다.

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
| 실행 epochs | 37/50 (Early stopping) |
| 최종 train loss | 0.2644 |
| 최고 val loss | 0.2420 (Epoch 32) |
| 학습 시간 | 1,719초 (~29분, CPU) |

### Loss Curve

```
Epoch  1: train=0.5076  val=0.4233
Epoch  5: train=0.3742  val=0.3172
Epoch 10: train=0.3389  val=0.2859
Epoch 15: train=0.3073  val=0.2665
Epoch 20: train=0.2920  val=0.2661
Epoch 25: train=0.2829  val=0.2510
Epoch 30: train=0.2727  val=0.2483
Epoch 32: train=0.2691  val=0.2420  ← best
Epoch 37: train=0.2645  val=0.2455  ← early stop
```

### Threshold

- 방법: IS 전체 window의 anomaly score 분포에서 **상위 7.5% (92.5th percentile)**
- 임계값: **0.361667**

```
Score distribution (IS, 115,188 windows):
  Mean:    0.2596
  Std:     1.0064
  50%:     0.2345
  90%:     0.3405
  92.5%:   0.3617  ← threshold
  95%:     0.3956
  99%:     0.6276
  Max:   330.7151
```

---

## 5. IS vs OOS Results

### 5.1 Anomaly Rate

| | IS | OOS |
|---|---|---|
| Total windows | 115,188 | 86,391 |
| Anomaly count | 8,640 | 5,616 |
| **Anomaly rate** | **7.5%** | **6.5%** |
| Trading days | 348 | 261 |
| Daily anomaly % (mean) | 7.5% | 6.5% |
| Daily anomaly % (std) | 12.3% | 12.8% |
| Daily anomaly % (max) | 80.1% | 83.4% |
| Days with 0% anomaly | 66 | 54 |

### 5.2 Score Distribution

| Percentile | IS | OOS |
|---|---|---|
| Mean | 0.2596 | 0.2441 |
| Median | 0.2345 | 0.2275 |
| Std | 1.0064 | 0.0849 |
| 90th | 0.3405 | 0.3337 |
| 95th | 0.3956 | 0.3817 |
| 99th | 0.6276 | 0.5214 |
| Max | 330.7151 | 3.9886 |

### 5.3 Anomaly Duration

| Duration | IS | OOS |
|---|---|---|
| 1 min | 332건 (40.3%) | 215건 (41.7%) |
| 2~4 min | 216건 (26.2%) | 132건 (25.6%) |
| 5~14 min | 138건 (16.8%) | 71건 (13.8%) |
| 15~59 min | 96건 (11.7%) | 70건 (13.6%) |
| 60+ min | 41건 (5.0%) | 28건 (5.4%) |
| **Total episodes** | **823** | **516** |
| Mean duration | 10.5 min | 10.9 min |
| Median duration | 2 min | 2 min |

### 5.4 Top Anomaly Dates

**IS (2024-01 ~ 2025-04)**

| Date | Anomaly % | Max Score | Event |
|---|---|---|---|
| 2024-03-05 | 80.1% | 1.36 | BTC $69K → $60K 급락 (ETF 후 첫 대형 조정) |
| 2024-08-05 | 56.8% | 40.01 | 엔캐리 청산 + 글로벌 증시 급락 |
| 2024-03-12 | 49.2% | 0.79 | CPI 발표일 |
| 2024-06-10 | 48.9% | 0.48 | - |
| 2024-08-02 | 48.6% | 1.29 | 고용지표 악화 → 경기 침체 우려 |

**OOS (2025-05 ~ 2026-04)**

| Date | Anomaly % | Max Score | Event |
|---|---|---|---|
| 2026-01-19 | 83.4% | 0.88 | MLK Day 전후 |
| 2026-01-01 | 82.8% | 0.57 | 새해 첫 거래일 |
| 2026-02-05 | 77.9% | 2.73 | 비농업 고용지표 급등 → BTC 급락 |
| 2025-09-19 | 69.2% | 0.69 | Triple Witching |
| 2025-09-18 | 55.6% | 0.50 | FOMC 결정일 |

---

## 6. Key Findings

### 6.1 Volatility Detection

| | Normal | Anomaly | High Intensity (2x+) |
|---|---|---|---|
| **IS** avg \|return\| | 0.066% | 0.189% (**2.9x**) | 0.708% (**10.8x**) |
| **OOS** avg \|return\| | 0.054% | 0.122% (**2.3x**) | 0.916% (**16.9x**) |

| Correlation | IS | OOS |
|---|---|---|
| Score ↔ Volatility | r = 0.10 | r = 0.33 |
| Score ↔ Direction | r = -0.03 | r = 0.01 |

> 고강도(2x+) anomaly에서 변동성이 정상 대비 **10~17배** — 변동성 탐지기로서 유효하다.

### 6.2 Direction Prediction — Not Possible

| | Up | Down |
|---|---|---|
| IS anomaly 구간 | 50% | 50% |
| OOS anomaly 구간 | 50% | 50% |

Score↔방향 상관계수가 IS/OOS 모두 ≈ 0.
VIXY 피처를 추가해도 **Autoencoder 구조로는 anomaly의 방향(상승/하락)을 예측할 수 없다.**

### 6.3 Position Switch Cost Analysis (OOS)

왕복 거래비용: 수수료 0.10% + 슬리피지 0.30% = **0.80%**

| Duration | Episodes | Cost-exceeding | Rate |
|---|---|---|---|
| 1 min | 215 | 0 | 0.0% |
| 2~4 min | 132 | 1 | 0.8% |
| 5~14 min | 71 | 0 | 0.0% |
| 15~59 min | 70 | 4 | 5.7% |
| 60+ min | 28 | 7 | 25.0% |

> 14분 이하 anomaly(전체의 81%)에서 포지션 전환 시 거래비용 회수 불가.

---

## 7. BTC-Only Baseline Comparison

BTC-only(4 features) → BTC+VIXY(7 features) 변화:

| | BTC-Only (4) | BTC+VIXY (7) |
|---|---|---|
| Val loss | 0.3619 | **0.2420 (-33%)** |
| Parameters | 64,260 | 65,223 |
| IS anomaly rate | 7.5% | 7.5% |
| OOS anomaly rate | 7.9% | 6.5% |
| OOS volatility ratio (anomaly/normal) | 1.9x | **2.3x** |
| OOS high-intensity ratio (2x+) | 11.4x | **16.9x** |
| OOS score↔volatility | r = 0.23 | **r = 0.33** |
| OOS score↔direction | r = -0.003 | r = 0.009 |

> VIXY 추가로 복원 성능과 변동성 탐지가 개선되었으나, 방향 예측은 여전히 불가능.

---

## 8. Limitations

1. **방향 예측 불가**: VIXY 추가 후에도 anomaly 방향은 50:50. Autoencoder는 구조적으로 "이상 여부"만 판단하며 방향을 알려주지 않음.
2. **단기 노이즈 과다**: Anomaly의 67%가 4분 이하 지속. 이 구간에서의 포지션 전환은 거래비용만 소모.
3. **VIXY 데이터 희소성**: VIXY 장중 분봉 채움률 14~41%. Forward fill로 보완했으나 실거래 없는 구간의 피처 품질은 제한적.
4. **장 초반 공백**: 09:30~10:28 (59분)은 window 축적 기간으로 anomaly score 없음.
5. **휴일 오탐**: MLK Day, 새해 등 저유동성 날에 False Positive 발생. 휴일 캘린더 미적용.

---

## 9. Strategy Implication

Anomaly 감지 시 **포지션 전환(롱↔숏)**은 다음 이유로 부적합:
- 방향 예측 불가 (50:50)
- 81%의 anomaly가 14분 이하 → 거래비용 회수 불가

**포지션 청산(flat) 전략**이 현실적:
- 방향을 맞출 필요 없음 (불확실할 때 빠지는 것)
- LSTM-AE의 가치 = "Alpha 생성"이 아닌 "Drawdown 감소"

```
실험군: 메인 포지션 + anomaly 시 청산 + 해소 시 재진입
대조군: 메인 포지션 하루 종일 유지
차이 = LSTM-AE가 "위험 구간을 피한" 효과
```

---

## 10. Artifacts

```
lstm_ae/artifacts/
├── model.pt                      # Trained model weights (Epoch 32, 7 features)
├── scaler.pkl                    # StandardScaler (fit on IS, 7 features)
├── threshold.json                # Threshold: 0.361667 (92.5th pctl)
├── anomaly_signals_is.parquet    # IS signals (115,188 windows)
└── anomaly_signals_oos.parquet   # OOS signals (86,391 windows)
```

### Signal Schema (`anomaly_signals_*.parquet`)

| Column | Type | Description |
|---|---|---|
| timestamp (index) | datetime | 1-min bar timestamp (ET) |
| anomaly_score | float64 | Reconstruction error (MSE) |
| is_anomaly | bool | score > 0.361667 |
| trade_date | date | Trading date |

---

## 11. How to Run

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

## 12. Conclusion

BTC+VIXY LSTM Autoencoder는 BTC-only 대비 **복원 성능 33% 개선**, **고강도 변동성 탐지 16.9배**로 향상되었다. 그러나 Autoencoder 구조의 본질적 한계로 **방향 예측은 불가능**하며, 포지션 전환 전략은 거래비용 대비 수익성이 없다.

LSTM-AE 모듈의 실전 역할은 **포지션 전환이 아닌 위험 구간 회피(청산)**이며, Trading 모듈에서 anomaly 감지 시 포지션 청산 → 해소 시 재진입 전략으로 Drawdown 감소 효과를 검증해야 한다.
