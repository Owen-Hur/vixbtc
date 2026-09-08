# ⚠ VIX3M 데이터 수집 안내

`data/vix_slope_daily.parquet`는 **VIX(30일)·VIX3M(3개월)·slope·direction** 4컬럼을 가진 일별 데이터로, 2024-01-02 ~ 2026-04-30 (584 영업일) 구간을 커버합니다.

## ⚠ 현재 상태
이 파일을 **생성한 스크립트는 현재 프로젝트에 보존되지 않았습니다.** 데이터 파일 자체(`data/vix_slope_daily.parquet`)는 그대로 사용 가능하지만, 다른 기간으로 확장하거나 재현하려면 아래 방법 중 하나로 새로 수집해야 합니다.

## 수집 방법 (권장 순서)

### 옵션 A — CBOE 공식 (가장 정확)
CBOE 공식 사이트에서 VIX3M 종가를 다운로드:
```
https://www.cboe.com/tradable_products/vix/term_structure/
```
또는 historical data 페이지에서 VIX3M (3-month VIX) CSV 다운로드.

### 옵션 B — Yahoo Finance (^VIX3M 티커)
```python
import yfinance as yf
vix3m = yf.download("^VIX3M", start="2024-01-01", end="2026-05-01")
# CBOE에서 미러링하므로 사실상 동일
```

### 옵션 C — 별도 데이터 벤더
Bloomberg, Refinitiv 등에서 VIX3M Index 일별 종가.

## 생성 예시 코드 (참고용)

데이터를 받은 뒤 `vix_slope_daily.parquet`를 재구성하는 예시:

```python
import pandas as pd

# vix_daily.parquet은 01_download_vix_daily.py로 이미 받음
vix = pd.read_parquet("data/vix_daily.parquet")    # VIX(30d)

# VIX3M은 위 옵션 A/B/C로 받아서:
vix3m = pd.read_csv("vix3m_raw.csv", parse_dates=["Date"])
vix3m = vix3m.set_index("Date")[["Close"]].rename(columns={"Close":"vix3m"})

# 병합
df = vix.join(vix3m, how="inner")

# 기간구조 slope (단기 - 장기 부호 컨벤션)
df["slope"] = df["vix"] - df["vix3m"]

# 방향 라벨 (Slope Change 분석용)
df["direction"] = df["slope"].diff().apply(
    lambda x: "Long" if x < 0 else ("Short" if x > 0 else "Neutral")
)

df.index.name = "Date"
df.to_parquet("data/vix_slope_daily.parquet")
```

## 슬라이드 부호 컨벤션 (중요)

8페이지 정의: **`slope = VIX − VIX3M`** (단기 − 장기)
- `slope > 0` → backwardation (역전, 단기 스트레스)
- `slope < 0` → contango (정상)
- LONG 규칙: `slope_change < 0` (contango 심화 → BTC 매수)
- SHORT 규칙: `slope_change > 0` (backwardation 전환 → BTC 매도)

→ 부호 반전 시 결과 수치는 변하지 않음 (LONG/SHORT 라벨만 swap). 검증 완료.

## 참고
- 논문(SSRN 6233752)도 VIX 4종 만기(9d, 30d, 3m, 6m) 모두 사용. 우리는 그중 30d·3m으로 강건성 확보 (논문 보고 만기 간 ΔVIX 상관 > 0.90).
- VIX3M = CBOE가 같은 방식으로 산출한 3개월 기대변동성 지수.
