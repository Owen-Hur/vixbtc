# LSTM Autoencoder Pipeline

## 1. 개요

LSTM Autoencoder는 장중(09:30~15:59 ET) BTC-VIXY 관계의 이상 구간을 탐지한다.
HMM이 결정한 메인 포지션에 대해, anomaly 발생 시 포지션 교체 여부 판단에 활용한다.

```
Input: BTC 1분봉 + VIXY 1분봉 → 60분 sliding window
Model: LSTM Autoencoder (reconstruction error = anomaly score)
Output: anomaly_signals.parquet (timestamp, score, is_anomaly, trade_date)
```

---

## 2. 데이터 흐름

### 2.1 전체 파이프라인

```
[1] 원본 데이터
    ├── IS/OOS: BTC 과거 체결 (data.binance.vision) + VIXY 1분봉
    └── OS:     BTC 실시간 체결 (btc_collector.py)   + VIXY 1분봉

        ↓

[2] 전처리 (preprocess_historical.py)
    ├── 틱 → 1분봉 resample (OHLCV + 체결 통계)
    ├── UTC → ET 변환 (서머타임 자동 처리)
    ├── 장중 필터링 (09:30~15:59 ET, 주중)
    └── 월별 parquet 저장

        ↓

[3] 피처 계산 (lstm_ae/dataset.py)
    ├── BTC: btc_return, trade_imbalance, trade_count, avg_trade_size
    ├── VIXY: vxx_return, vxx_rolling_std, vxx_btc_corr (합류 예정)
    └── StandardScaler 정규화

        ↓

[4] 윈도우 생성 (lstm_ae/dataset.py)
    ├── 거래일별로 60분 sliding window 생성
    ├── 교차일 window 없음 (일 경계에서 끊김)
    └── 08:30 시작 시 09:30에 첫 window 완성 (검토 중)

        ↓

[5] 학습 또는 추론
    ├── 학습: lstm_ae/train.py → model.pt, scaler.pkl, threshold.json
    └── 추론: lstm_ae/inference.py → anomaly_signals.parquet

        ↓

[6] Trading 모듈에 전달
    └── anomaly_signals.parquet을 Trading이 읽어서 포지션 판단
```

### 2.2 데이터 구간

| 구간 | 기간 | 소스 | 용도 |
|---|---|---|---|
| IS | 2024.01 ~ 2025.04 | 과거 체결 (체결만, LOB 없음) | 모델 학습 |
| OOS | 2025.05 ~ 2026.04 | 과거 체결 (체결만, LOB 없음) | 1차 검증 |
| OS | 2026.05.10~ | 실시간 수집 (체결 + LOB) | 2차 검증 |

### 2.3 과거 vs 실시간 데이터 차이

| 항목 | 과거 (historical) | 실시간 (btc_collector) |
|---|---|---|
| 타임스탬프 컬럼 | `time` | `timestamp_ms` |
| 수량 컬럼 | `qty` | `quantity` |
| boolean 값 | 소문자 `true/false` | 대문자 `True/False` |
| 추가 컬럼 | `id`, `quote_qty` 있음 | 없음 |
| 이상 행 | 없음 | `price=0` 간헐 발생 → 필터링 필요 |
| LOB 데이터 | ❌ 없음 | ✅ 있음 (LSTM-AE에서는 미사용) |

> OS 전처리 시 컬럼명 매핑 필요. 전처리 후 출력 형식은 동일.

---

## 3. 모델 구조

### 3.1 아키텍처

```
Encoder:
  LSTM(n_features → 64)  →  LSTM(64 → 32)  →  latent vector (32-dim)
                                                  ↑ last timestep only

Decoder:
  RepeatVector(32 → 60×32)  →  LSTM(32 → 32)  →  LSTM(32 → 64)  →  Dense(64 → n_features)

Input/Output: (batch, 60, n_features)
Loss: MSE
Anomaly score = reconstruction error (MSE per window)
```

### 3.2 하이퍼파라미터 (config.py)

| Parameter | Value | 비고 |
|---|---|---|
| Window size | 60 min | |
| Encoder hidden | [64, 32] | |
| Decoder hidden | [32, 64] | |
| Latent dim | 32 | |
| Learning rate | 1e-3 | Adam |
| Batch size | 64 | |
| Max epochs | 50 | |
| Early stopping patience | 5 | val_loss 기준 |
| Threshold percentile | 92.5 | 상위 7.5% |
| Train/Val split | 80/20 | 시간순 (랜덤 아님) |

### 3.3 Input Features

**현재 (BTC-only, 4개)**

| Feature | Description | Source |
|---|---|---|
| `btc_return` | BTC 1분 수익률 | 1분봉 close |
| `trade_imbalance` | 매수/매도 비율 (-1~+1) | 체결 is_buyer_maker |
| `trade_count` | 1분간 체결 건수 | 체결 count |
| `avg_trade_size` | 1분간 평균 체결 규모 | 체결 qty 평균 |

**VIXY 합류 후 (7개)** — 추가 예정

| Feature | Description | Source |
|---|---|---|
| `vxx_return` | VIXY 1분 수익률 | VIXY 분봉 |
| `vxx_rolling_std` | VIXY 20분 rolling 표준편차 | VIXY 분봉 |
| `vxx_btc_corr` | VIXY-BTC 20분 rolling 상관계수 | 계산 |

---

## 4. 학습 프로세스

```bash
cd btc_project
python3 -m lstm_ae.train
```

### 4.1 순서

```
1. 데이터 준비
   - IS 구간 parquet 로딩
   - 피처 계산 (btc_return, trade_imbalance, ...)
   - StandardScaler fit & transform → scaler.pkl 저장
   - 60분 sliding window 생성 (거래일 단위)
   - 80/20 시간순 분할 (학습/검증)

2. 모델 학습
   - MSE loss, Adam optimizer
   - Early stopping (patience=5, val_loss 기준)
   - 최고 val_loss 모델 저장 → model.pt

3. 임계값 계산
   - 최고 모델로 IS 전체 window의 anomaly score 계산
   - 92.5th percentile = threshold → threshold.json
```

### 4.2 출력 Artifacts

```
lstm_ae/artifacts/
├── model.pt           # 학습된 모델 가중치
├── scaler.pkl         # StandardScaler (IS에서 fit)
└── threshold.json     # 임계값 및 score 분포 통계
```

---

## 5. 추론 프로세스

```bash
python3 -m lstm_ae.inference        # OOS 구간
python3 -m lstm_ae.inference --is   # IS 구간
```

### 5.1 순서

```
1. 학습된 model, scaler, threshold 로드
2. 대상 구간 parquet 로딩 + 피처 계산
3. IS에서 fit한 scaler로 정규화 (transform only, fit 안 함)
4. 60분 sliding window 생성
5. 각 window의 anomaly score 계산 (reconstruction error)
6. score > threshold → is_anomaly = True
7. anomaly_signals.parquet 저장
```

### 5.2 Output Schema (anomaly_signals.parquet)

| Column | Type | Description |
|---|---|---|
| timestamp (index) | datetime | 1분봉 시각 (ET) |
| anomaly_score | float64 | Reconstruction error (MSE) |
| is_anomaly | bool | score > threshold |
| trade_date | date | 거래일 |

---

## 6. Trading 모듈 연동

### 6.1 인터페이스

Trading 모듈은 `anomaly_signals.parquet`을 읽어서 다음을 수행:

```
매분 anomaly_score와 is_anomaly 확인
  → is_anomaly == True: 포지션 교체 또는 청산 판단
  → is_anomaly == False: 메인 포지션 유지
```

### 6.2 Trading에 전달하는 정보

| 항목 | 값 | 비고 |
|---|---|---|
| anomaly 여부 | `is_anomaly` (bool) | 매분 제공 |
| anomaly 강도 | `anomaly_score` (float) | 임계값 대비 배수로 활용 가능 |
| 임계값 | `threshold.json` | 진입/해소 기준 |

### 6.3 미확정 사항 (VIXY 합류 후 결정)

- **포지션 전환 vs 청산**: BTC-only 상태에서는 방향 예측 불가 → VIXY 합류 후 재평가
- **최소 보유 시간**: 현재 5분 설정, 최적값은 VIXY 합류 후 재검토
- **히스테리시스 임계값**: 진입/해소 임계값 분리 여부도 재검토 대상

---

## 7. Walk-forward 설계

```
1차: IS (2024.01~2025.04) → OOS (2025.05~2026.04)
     관세 전 학습 모델이 관세 후에도 작동하는가?

2차: OOS를 학습으로 전환 → OS (2026.05.10~, ~3주)
     최근 시장 학습 모델의 실전 적용 가능성
```

| 결과 | 해석 |
|---|---|
| 둘 다 성공 | 전략이 시장 구조 변화에 강건 |
| 1차만 실패 | 관세가 VIX-BTC 관계를 바꿨다는 증거 |
| 2차만 실패 | OS 기간 부족 (통계적 한계) |

---

## 8. 파일 구조

```
lstm_ae/
├── __init__.py
├── config.py          # 하이퍼파라미터 중앙 관리
├── dataset.py         # 데이터 로딩, 피처 계산, 윈도우 생성, Dataset
├── model.py           # LSTMAutoencoder 클래스
├── train.py           # 학습 루프 + 임계값 계산
├── inference.py       # 추론 + anomaly_signals 생성
├── REPORT.md          # BTC-only baseline 결과 리포트
├── PIPELINE.md        # 이 문서
└── artifacts/
    ├── model.pt
    ├── scaler.pkl
    ├── threshold.json
    ├── anomaly_signals_is.parquet
    └── anomaly_signals_oos.parquet
```

---

## 9. 주의사항

1. **scaler는 IS에서만 fit**: OOS/OS에서는 `transform`만 사용. 재학습 시에만 새로 fit.
2. **window는 거래일 단위**: 교차일 window 없음. 전일 마지막 데이터와 당일 첫 데이터를 섞지 않음.
3. **VIXY 합류 시 전체 재학습 필요**: n_features 변경 → scaler, model, threshold 모두 새로 생성.
4. **OS 전처리 시 컬럼명 매핑**: 실시간 데이터는 과거와 컬럼명이 다름 (Section 2.3 참고).
