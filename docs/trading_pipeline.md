# Trading Module Pipeline

> ⚠️ **설계 문서입니다 — 이 문서가 기술하는 Trading 모듈은 구현되지 않았습니다.**
> HMM 방향 신호(타 팀원 담당)가 이 저장소에 없어 2단계 통합 백테스트를 실행하지 못했습니다.
> 이 문서에 성과 수치는 없으며, 실제로 실행된 것은 `lstm_ae/` 의 단독 개입 실험뿐입니다
> (결과: 12종 전부 기각 — [`strategy_report_2026-05-24_superseded.md`](strategy_report_2026-05-24_superseded.md)).

## 1. 개요

Trading 모듈은 HMM과 LSTM-AE의 출력을 받아 실제 포지션을 관리하고 백테스트를 수행한다.
이 모듈의 목적은 **이상 탐지 기반 포지션 교체가 단순 보유 대비 Alpha를 만드는지** 검증하는 것이다.

```
Input:
  ├── HMM → 일별 메인 포지션 (Long/Short)
  ├── LSTM-AE → 분별 anomaly signal (score, is_anomaly)
  ├── BTC 1분봉 (가격 데이터)
  └── BTC 펀딩비 (8시간 단위)

Output:
  ├── 백테스트 성과 (수익률, Sharpe, Drawdown 등)
  ├── 거래 로그 (진입/청산 시점, 비용)
  └── 실험군 vs 대조군 비교 결과
```

---

## 2. 전략 구조

### 2.1 2단계 의사결정

```
[1단계] 장 시작 전 — HMM 메인 포지션 결정
    VIX slope(PCA2) + HMM 3-State regime 분류
    → 오늘의 방향: Long 또는 Short

[2단계] 장중 (09:30~15:59 ET) — LSTM-AE 이상 탐지
    매분 anomaly_score 확인
    → anomaly 발생: 포지션 교체 또는 청산 (VIXY 합류 후 확정)
    → anomaly 해소: 메인 포지션 복귀
    → 15:59: 전량 청산 (오버나이트 없음)
```

### 2.2 비교 실험

```
실험군: 메인 포지션 + anomaly 시 포지션 교체/청산 + 해소 시 복귀
대조군: 메인 포지션 하루 종일 유지 (교체 없음)

성과 차이 = LSTM-AE 모델의 순수 기여도
```

---

## 3. 데이터 입력

### 3.1 HMM 출력 (일별)

| Column | Type | Description |
|---|---|---|
| trade_date | date | 거래일 |
| regime | int | HMM state (0: Normal, 1: Alert, 2: Fear) |
| direction | str | "Long" 또는 "Short" |
| vix_slope | float | VIX PCA2 값 |

### 3.2 LSTM-AE 출력 (분별)

| Column | Type | Description |
|---|---|---|
| timestamp | datetime | 1분봉 시각 (ET) |
| anomaly_score | float64 | Reconstruction error |
| is_anomaly | bool | score > threshold |
| trade_date | date | 거래일 |

> 파일 위치: `lstm_ae/artifacts/anomaly_signals_*.parquet`

### 3.3 BTC 가격 데이터 (분별)

| Column | Type | Description |
|---|---|---|
| timestamp | datetime | 1분봉 시각 (ET) |
| open, high, low, close | float | OHLC |
| volume | float | 거래량 |

> 파일 위치: `data/processed/btc_1m_*.parquet`

### 3.4 BTC 펀딩비

| Column | Type | Description |
|---|---|---|
| timestamp | datetime | 정산 시각 |
| funding_rate | float | 펀딩비율 |

> 소스: ccxt (Binance), 8시간 단위, 12:00 ET (= 16:00 UTC) 정산분이 장중에 해당
> 상태: 미수집 — ccxt로 수집 필요

---

## 4. 거래 규칙

### 4.1 확정된 설정

| 항목 | 값 | 비고 |
|---|---|---|
| 거래 대상 | BTCUSDT 영구선물 (Binance) | |
| 거래 시간 | 09:30 ~ 15:59 ET (주중) | |
| 레버리지 | 2x | PoC 보수적 설정 |
| 진입 | 09:30 시장가 | HMM 방향대로 |
| 청산 | 15:59 시장가 | 오버나이트 없음 |
| 최소 보유 시간 | 5분 | 과도한 교체 방지 (재검토 예정) |

### 4.2 거래 비용

| 항목 | 값 | 적용 방식 |
|---|---|---|
| 수수료 | Taker 0.05% | 왕복 적용 (진입 + 청산) |
| 슬리피지 | 고정 15bp | IS/OOS 백테스트 시 |
| 슬리피지 (OS) | 호가창 기반 실측 | OS에서만 가능 (아래 참고) |
| 펀딩비 | 실제값 반영 | 12:00 ET 정산, 장중 1회 |

### 4.3 일일 거래 흐름

```
09:30  HMM 방향 확인 → 메인 포지션 진입 (2x 레버리지)
         ├── 수수료: 0.05%
         └── 슬리피지: 15bp

09:30~15:58  매분 anomaly signal 확인
         ├── anomaly 발생 → 포지션 교체/청산 (비용 발생)
         ├── anomaly 해소 → 메인 포지션 복귀 (비용 발생)
         └── 12:00 ET → 펀딩비 정산 (보유 포지션에 적용)

15:59  전량 청산
         ├── 수수료: 0.05%
         └── 슬리피지: 15bp
```

---

## 5. 슬리피지 처리

### 5.1 구간별 차이

| 구간 | 호가창 데이터 | 슬리피지 방식 |
|---|---|---|
| IS (24.01~25.04) | ❌ 없음 | 고정 15bp |
| OOS (25.05~26.04) | ❌ 없음 | 고정 15bp |
| OS (26.05.10~) | ✅ 있음 | 호가창 기반 실측 가능 |

### 5.2 호가창 기반 슬리피지 계산 (OS)

```
LOB 데이터: bid_p1~20, bid_q1~20, ask_p1~20, ask_q1~20 (1초 스냅샷)

매수 주문 시:
  주문량을 ask_p1부터 순차 체결 시뮬레이션
  실제 평균 체결가 - mid price = 슬리피지

매도 주문 시:
  주문량을 bid_p1부터 순차 체결 시뮬레이션
  mid price - 실제 평균 체결가 = 슬리피지
```

### 5.3 슬리피지 검증

OS 구간에서 실측 슬리피지를 IS/OOS의 고정 15bp와 비교:
- 15bp > 실측 → 백테스트가 보수적이었음 (실제 성과가 더 좋을 수 있음)
- 15bp < 실측 → 백테스트가 낙관적이었음 (실제 성과가 더 나쁠 수 있음)

---

## 6. 백테스트 구현

### 6.1 기본 구조

```python
for each trading_day:
    # 1. HMM 방향 확인
    direction = hmm_signals[date]  # "Long" or "Short"
    
    # 2. 09:30 진입
    entry_price = btc_1m[date, "09:30"].open
    position = direction
    apply_cost(entry_price, slippage=15bp, fee=0.05%)
    
    # 3. 장중 anomaly 체크 (매분)
    for minute in 09:31 ~ 15:58:
        signal = anomaly_signals[date, minute]
        
        if signal.is_anomaly and can_switch():  # 최소 보유시간 체크
            # 포지션 교체 (비용 발생)
            close_position()  # 비용: 15bp + 0.05%
            open_opposite()   # 비용: 15bp + 0.05%
            
        elif not signal.is_anomaly and position != direction:
            # 메인 방향 복귀 (비용 발생)
            close_position()
            open_main_direction()
    
    # 4. 12:00 ET 펀딩비 정산
    if holding_at_noon:
        apply_funding_rate()
    
    # 5. 15:59 청산
    close_all(slippage=15bp, fee=0.05%)
    
    record_daily_pnl()
```

### 6.2 대조군 (비교 전략)

```python
for each trading_day:
    direction = hmm_signals[date]
    entry at 09:30
    hold all day (no anomaly-based switching)
    funding at 12:00
    exit at 15:59
    record_daily_pnl()
```

---

## 7. 평가 지표

| 지표 | 목표 | 설명 |
|---|---|---|
| Net Return | 대조군 대비 양(+) 초과수익 | 거래비용 차감 후 |
| Sharpe Ratio | 대조군 대비 개선 | 일별 수익률 기준 |
| Max Drawdown | 대조군 대비 감소 | 최대 누적 손실폭 |
| Hit Rate | >= 55% | 수익 거래일 / 전체 거래일 |
| Anomaly Precision | >= 60% | 유효한 anomaly / 전체 anomaly |
| Avg Slippage (OS) | <= 15bp | 호가창 기반 실측값 |
| Daily Trade Count | 모니터링 | 과도한 교체 여부 체크 |

---

## 8. 구간별 데이터 가용성

OS(실시간 수집) 구간은 실제로 진행하지 않았으므로 IS/OOS 두 구간만 정리한다.

| 데이터 | IS | OOS |
|---|---|---|
| BTC 1분봉 (체결 기반) | ✅ | ✅ |
| BTC LOB (호가창) | ❌ | ❌ |
| VIXY 1분봉 | ✅ | ✅ |
| VIX 일봉 | ✅ | ✅ |
| HMM 방향 signal | 타 팀원 제공 | 타 팀원 제공 |
| LSTM-AE anomaly signal | ✅ | ✅ |
| BTC 펀딩비 | ✅ (`data/funding_rate_history.parquet`) | ✅ |

---

## 9. 설계 시점의 미결 항목과 그 결말

이 문서는 VIXY 피처 합류 **이전**에 작성된 설계안이다. 당시 미결로 두었던 항목의 결말은 다음과 같다.

| 항목 | 설계 시점 | 결말 |
|---|---|---|
| anomaly 시 행동 | 미정 (전환 vs 청산) | 7 features 학습 후에도 방향 예측 불가(상승 49.5~49.8%, `lstm_ae/REPORT.md` §6.1) → **청산으로 확정** |
| 최소 보유 시간 | 5분 | 에피소드의 83.2%가 14분 이하로 끝나 전환 자체가 기각되어 미적용 |
| 임계값 방식 | 단일 (92.5%ile) | 단일 임계값 그대로 확정 (`artifacts/threshold.json` = 0.357368) |
| BTC 펀딩비 | 미수집 | 수집 완료 — `data/funding_rate_history.parquet` (Binance, 8시간 단위) |

---

## 10. 기술 스택 (권장)

| 용도 | 도구 |
|---|---|
| 백테스트 프레임워크 | vectorbt |
| 데이터 처리 | pandas, numpy |
| 시각화 | plotly, Streamlit |
| 성과 분석 | scipy (통계 검정) |
