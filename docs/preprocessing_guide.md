# BTC 선물 데이터 전처리 가이드

## 개요

과거 체결(tick) 데이터를 미국 장중 1분봉으로 집계하고, VIX/VIXY 데이터와 병합 가능한 형태로 저장합니다.

```
수집/다운로드 (raw tick 데이터, 월별 CSV)
        ↓
전처리: 로딩 → UTC→ET 변환 → 장중 필터링 → 1분봉 집계
        ↓
월별 parquet 저장 (data/processed/btc_1m_YYYY-MM.parquet)
        ↓
VIX 일봉 / VIXY 분봉 병합 (별도 담당자 데이터와 합류)
        ↓
모델 학습 / 백테스트
```

---

## Raw 데이터 구조

### 소스

`data.binance.vision` — BTCUSDT 영구 선물 체결 데이터

### 파일 구조

```
data/historical/trades/
  BTCUSDT-trades-2024-01.csv
  BTCUSDT-trades-2024-02.csv
  ...
  BTCUSDT-trades-2026-04.csv   (28개월)
```

### 규모

- 총 용량: **159GB**
- 파일당: ~1억 행, ~5.7GB
- 한 번에 메모리에 전부 올릴 수 없음 → **월별 순차 처리 필수**

### 컬럼 구조

첫 행에 헤더가 포함되어 있음 (`header=0`으로 로딩).

| 컬럼명 | 타입 | 설명 |
|---|---|---|
| id | int | 체결 ID |
| price | float | 체결 가격 (USDT) |
| qty | float | 체결 수량 (BTC) |
| quote_qty | float | 체결 금액 (USDT) |
| time | int | 체결 시각 (UTC 밀리초 타임스탬프) |
| is_buyer_maker | bool | `true`=매도 공격, `false`=매수 공격 |

> **is_buyer_maker 해석:**
> - `false` → 매수자가 시장가 주문 = **매수 공격** (가격 올리는 방향)
> - `true` → 매도자가 시장가 주문 = **매도 공격** (가격 내리는 방향)

---

## 전처리 파이프라인

### Step 1: 파일 로딩

파일당 ~1억 행이므로 chunk 처리가 필요할 수 있음.

```python
import pandas as pd

# 단일 월 파일 로딩
df = pd.read_csv(
    'data/historical/trades/BTCUSDT-trades-2024-01.csv',
    header=0,  # 첫 행이 헤더
    dtype={
        'id': 'int64',
        'price': 'float64',
        'qty': 'float64',
        'quote_qty': 'float64',
        'time': 'int64',
        'is_buyer_maker': 'object'  # true/false 문자열
    }
)

# boolean 변환 (소문자 문자열 → bool)
df['is_buyer_maker'] = df['is_buyer_maker'].map({'true': True, 'false': False})
```

> **메모리 주의:** 파일 하나가 ~5.7GB이므로, 16GB RAM 기준 한 파일 로딩만으로
> 메모리의 상당 부분을 차지합니다. 필요 시 `chunksize` 파라미터를 사용합니다.

### Step 2: UTC → ET 변환

```python
df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True)
df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
```

`America/New_York` 사용 시 서머타임 자동 처리:
- EDT (UTC-4): 3월 둘째 일요일 ~ 11월 첫째 일요일
- EST (UTC-5): 나머지 기간

> **주의:** UTC±4, UTC±5를 수동 계산하면 서머타임 전환일 경계에서
> 데이터가 1시간씩 어긋납니다. 반드시 `America/New_York`으로 변환합니다.

### Step 3: 장중 필터링 (09:30 ~ 15:59 ET, 주중만)

```python
df = df.set_index('timestamp')

market_mask = (
    (df.index.time >= pd.Timestamp("09:30").time()) &
    (df.index.time < pd.Timestamp("16:00").time()) &   # 15:59:59.999까지 포함
    (df.index.dayofweek < 5)                            # 월~금
)

df_market = df[market_mask]
```

- 전략 청산 시각이 15:59 ET이므로, 16:00:00 이후 데이터는 제외
- 24시간 중 6.5시간만 사용 → 약 **21~22%** 만 남음
- 주말(토/일) 제외

### Step 4: 1분봉 집계

틱 데이터를 1분 단위로 집계합니다. 1초봉 단계를 거치지 않고 **틱 → 1분봉 직행**합니다.

```python
df_1m = df_market.resample('1min').agg(
    open=('price', 'first'),
    high=('price', 'max'),
    low=('price', 'min'),
    close=('price', 'last'),
    volume=('qty', 'sum'),
    quote_volume=('quote_qty', 'sum'),
    trade_count=('price', 'count'),
    avg_trade_size=('qty', 'mean'),
    buy_count=('is_buyer_maker', lambda x: (~x).sum()),   # 매수 공격 건수
    sell_count=('is_buyer_maker', lambda x: x.sum()),      # 매도 공격 건수
)

# trade_imbalance: -1(완전 매도) ~ +1(완전 매수)
total = df_1m['buy_count'] + df_1m['sell_count']
df_1m['trade_imbalance'] = (
    (df_1m['buy_count'] - df_1m['sell_count']) / total
).fillna(0)

# 보조 컬럼 정리
df_1m = df_1m.drop(columns=['buy_count', 'sell_count'])
```

**산출 컬럼 (BTC 1분봉):**

| 컬럼 | 설명 | 용도 |
|---|---|---|
| open | 시가 | OHLCV |
| high | 고가 | OHLCV |
| low | 저가 | OHLCV |
| close | 종가 | btc_return 계산 기반 |
| volume | 거래량 (BTC) | OHLCV |
| quote_volume | 거래대금 (USDT) | 참고용 |
| trade_count | 1분간 체결 건수 | LSTM AE 입력 피처 |
| avg_trade_size | 1분간 평균 체결 규모 | LSTM AE 입력 피처 |
| trade_imbalance | 매수/매도 비율 (-1~+1) | LSTM AE 입력 피처 |

> **trade_imbalance 해석:**
> - `+1에 가까울수록` → 매수 압력 강함
> - `-1에 가까울수록` → 매도 압력 강함
> - `0` → 균형 또는 체결 없음

### Step 5: 빈 1분봉 처리

장중이라도 특정 1분 구간에 체결이 없을 수 있습니다 (BTCUSDT라 극히 드묾).

```python
# close → 직전 close로 ffill (가격 연속성 유지)
df_1m['close'] = df_1m['close'].ffill()
df_1m['open'] = df_1m['open'].ffill()
df_1m['high'] = df_1m['high'].ffill()
df_1m['low'] = df_1m['low'].ffill()

# 수량 관련 → 0으로 채움 (거래 없음 = 0)
df_1m['volume'] = df_1m['volume'].fillna(0)
df_1m['quote_volume'] = df_1m['quote_volume'].fillna(0)
df_1m['trade_count'] = df_1m['trade_count'].fillna(0)
df_1m['avg_trade_size'] = df_1m['avg_trade_size'].fillna(0)
df_1m['trade_imbalance'] = df_1m['trade_imbalance'].fillna(0)
```

### Step 6: 날짜 컬럼 추가 + parquet 저장

```python
# ET 기준 거래일 (VIX 일봉 병합 키)
df_1m['trade_date'] = df_1m.index.date

# 월별 parquet 저장
df_1m.to_parquet(f'data/processed/btc_1m_2024-01.parquet')
```

---

## 출력 파일 구조

```
data/processed/
  btc_1m_2024-01.parquet
  btc_1m_2024-02.parquet
  ...
  btc_1m_2026-04.parquet   (28개 파일)
```

### 출력 스키마

| 컬럼 | 타입 | 설명 |
|---|---|---|
| (index) | DatetimeIndex (ET, tz-aware) | 1분봉 시각 (예: `2024-01-02 09:30:00-05:00`) |
| open | float64 | 시가 |
| high | float64 | 고가 |
| low | float64 | 저가 |
| close | float64 | 종가 |
| volume | float64 | 거래량 (BTC) |
| quote_volume | float64 | 거래대금 (USDT) |
| trade_count | int64 | 체결 건수 |
| avg_trade_size | float64 | 평균 체결 규모 |
| trade_imbalance | float64 | 매수/매도 비율 (-1~+1) |
| trade_date | date | ET 기준 거래일 |

### 예상 규모

- 1거래일당: 390분 (09:30~15:59, 390개 행)
- 1개월당: ~8,580행 (약 22거래일)
- 28개월 합계: ~240,000행
- parquet 용량: 월별 수백 KB 수준 (틱 대비 극적으로 축소)

### 왜 parquet인가

- **용량**: CSV 대비 1/5 ~ 1/10 압축
- **속도**: 컬럼 단위 읽기 → 특정 컬럼만 로딩 시 매우 빠름
- **타입 보존**: datetime, float 등 타입이 저장 시점 그대로 유지 (CSV는 문자열화됨)

---

## VIX / VIXY 데이터 병합 가이드 (타 팀원 참고용)

BTC 1분봉 전처리는 위에서 완료됩니다. VIX/VIXY 데이터는 별도 담당자가 수집하며,
아래 사양에 맞춰 전처리하면 BTC 데이터와 병합할 수 있습니다.

### 데이터 용도 구분

| 데이터 | 해상도 | 사용처 | 병합 키 |
|---|---|---|---|
| VIX 4개 만기 (9d, 30d, 3m, 6m) | **일봉** | HMM 장초 포지션 결정 | `trade_date` |
| VIXY | **1분봉** | LSTM AE 장중 이상탐지 | 1분봉 timestamp (ET) |

### A. VIX 일봉 — HMM용

**소스:** yfinance (`^VIX9D`, `^VIX`, `^VIX3M`, `^VIX6M`) 또는 CBOE

**필요 기간:** 2024-01-01 ~ 2026-04-30

**전처리 후 필요한 형태:**

```
파일: data/external/vix_daily.parquet

| 컬럼         | 타입    | 설명                    |
|-------------|---------|------------------------|
| trade_date  | date    | 거래일 (미국 기준)        |
| vix_9d      | float64 | VIX 9일                 |
| vix_30d     | float64 | VIX 30일                |
| vix_3m      | float64 | VIX 3개월               |
| vix_6m      | float64 | VIX 6개월               |
```

**BTC와 병합:** `trade_date` 기준 left join
```

### B. VIXY 1분봉 — LSTM AE용

**소스:** yfinance (`VIXY`, interval=`1m`)

**주의:** yfinance 1분봉은 **최근 7일만 제공**합니다.
과거 28개월 분봉 데이터가 필요하므로 별도 유료 소스(Polygon.io 등)가 필요할 수 있습니다.

**필요 기간:** 2024-01-01 ~ 2026-04-30 (BTC와 동일)

**전처리 후 필요한 형태:**

```
파일: data/external/vixy_1m.parquet

| 컬럼              | 타입                          | 설명                |
|------------------|-------------------------------|-------------------|
| (index)          | DatetimeIndex (ET, tz-aware)  | 1분봉 시각 (ET)     |
| vixy_close       | float64                       | VIXY 종가           |
| vixy_volume      | float64                       | VIXY 거래량          |
```

**전처리 시 주의사항:**
- 시간대를 반드시 ET(America/New_York)로 변환해야 BTC와 정렬됨
- VIXY는 미국 장중(09:30~16:00)에만 거래되므로 별도 필터링 불필요
- 레벨(절대 가격) 비교 금지 — contango 롤오버 비용으로 장기 하락 편향 있음
- **수익률(pct_change)과 방향성 신호로만 활용**

**BTC와 병합:** 1분봉 timestamp (ET) 기준 left join

---

## 실시간 수집 데이터 (btc_collector.py) 전처리

### 과거 vs 실시간 데이터 차이

| 항목 | 과거 (historical) | 실시간 (btc_collector) |
|---|---|---|
| 타임스탬프 컬럼명 | `time` | `timestamp_ms` |
| 수량 컬럼명 | `qty` | `quantity` |
| boolean 값 | 소문자 `true/false` | 대문자 `True/False` |
| 추가 컬럼 | `id`, `quote_qty` 있음 | 없음 |
| 이상 행 | 없음 | `price=0` 간헐 발생 → 필터링 필요 |
| 헤더 | 있음 (header=0) | 있음 (header=0) |

실시간 데이터에 동일한 1분봉 집계를 적용할 때 컬럼명을 먼저 통일합니다:

```python
# 실시간 데이터 컬럼명 → 과거 형식으로 통일
df_rt = df_rt.rename(columns={
    'timestamp_ms': 'time',
    'quantity': 'qty'
})

# price=0 이상 행 제거
df_rt = df_rt[df_rt['price'] > 0]

# 이후 동일한 Step 2~6 적용
```

### 결측 구간 탐지

```python
# 체결 데이터: BTCUSDT 기준 10초 이상 공백이면 수집 중단으로 판단
df['gap'] = df['timestamp'].diff()
gaps = df[df['gap'] > pd.Timedelta(seconds=10)]
print(f"의심 구간 {len(gaps)}건")
```

### 결측 처리

- 1분봉 집계 시 해당 구간은 NaN 발생
- volume=0 → 거래 없음 (정상) / volume=NaN → 수집 중단 (결측)
- LSTM AE: 입력 window(60분) 안에 결측 포함 시 해당 window 제외
- 백테스트: 결측 구간은 포지션 변경 없이 유지 (보수적 처리)
- 결측 비율이 5% 이상이면 해당 일자 전체 제외 검토

---

## 전체 순서 요약

```
1. 월별 CSV 로딩
   pd.read_csv(header=0) → 체결 데이터

2. UTC → ET 변환
   .dt.tz_convert('America/New_York')

3. 장중 필터링
   09:30 ~ 15:59 ET, 주중(월~금)만

4. 1분봉 집계
   .resample('1min').agg(...)
   → OHLCV + trade_count + avg_trade_size + trade_imbalance

5. 빈 1분봉 처리
   close ffill, 수량 컬럼 0

6. trade_date 추가 + parquet 저장
   → data/processed/btc_1m_YYYY-MM.parquet

--- 이하 별도 담당자 데이터 합류 후 ---

7. VIX 일봉 병합 (trade_date 기준)
   → HMM용 feature engineering

8. VIXY 1분봉 병합 (timestamp 기준)
   → LSTM AE용 feature engineering
```

---

## 주의사항

| 항목 | 내용 |
|---|---|
| 헤더 | 과거 CSV에 헤더 있음 — `header=0`으로 로딩 |
| 서머타임 | UTC±4/5 수동 계산 금지 — `America/New_York` 사용 |
| 장 종료 시각 | 15:59까지 포함 (`< 16:00`), 16:00 이후 제외 |
| boolean 처리 | 과거 데이터는 소문자 문자열 (`true`/`false`) → 명시적 변환 필요 |
| 빈 1분봉 | close는 ffill, 수량 컬럼은 0 |
| VIXY 레벨 비교 금지 | contango 드리프트로 장기 하락 편향 — 수익률/방향성만 사용 |
| parquet 저장 | 전처리는 월별 1회 실행, 이후 parquet 재사용 |
