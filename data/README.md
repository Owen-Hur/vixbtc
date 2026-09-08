# 데이터 — 스키마 · 출처 · 재수집 안내

원본 시장 데이터(총 ~270MB)는 저장소 용량 문제로 **포함하지 않았습니다.**
아래 데이터는 **전부 무료 공개 소스**에서 재수집 가능하며, `data_collection/` 의 스크립트로 재현할 수 있습니다.

이 디렉토리에 아래 구조대로 파일을 채우면 `lstm_ae/` 와 `analysis/` 의 모든 스크립트가 그대로 동작합니다.

```
data/
├── vix_daily.parquet                    # VIX 30일 일별 종가
├── vix_slope_daily.parquet              # VIX · VIX3M · slope · direction  ★ 핵심 신호
├── vix_1h.parquet                       # VIX 1시간봉 (시간대별 Granger 검정용)
├── funding_rate_history.parquet         # Binance 펀딩레이트 (8h)
├── btc_1m_24h/                          # BTC 1분봉 24시간 (2020-01 ~ 2026-05, 77 파일, ~220MB)
│   └── btc_1m_YYYY-MM.parquet
├── btc_historical_processed/            # BTC 1분봉 미국 장중만 필터링 (2024-01 ~ 2026-04, ~17MB)
│   └── btc_1m_YYYY-MM.parquet
└── vixy_1m/                             # VIXY 1분봉 월별 (~8MB)
    ├── YYYYMM.parquet
    ├── VIXY_1m_IS_20240115_20251031.parquet
    └── VIXY_1m_OOS_20251101_20260430.parquet
```

---

## 1. VIX 계열

### `vix_daily.parquet`

| 항목 | 값 |
|---|---|
| 출처 | **FRED** (St. Louis Fed) — `VIXCLS` 시리즈, 무료 |
| 수집 | `python data_collection/fetch_vix_daily.py` |
| 기간 | 2020-01-02 ~ 2026-05-27 (1,638행) |
| 인덱스 | `date` (일별, tz-naive) |
| 컬럼 | `vix` (float) — VIX 30일 종가 |

### `vix_slope_daily.parquet` ★ 핵심 신호

| 항목 | 값 |
|---|---|
| 출처 | **CBOE** 공식 히스토리컬 데이터 (VIX + VIX3M), 무료 |
| 수집 | `data_collection/NOTE_vix3m.md` 참조 (VIX3M은 CBOE 웹에서 수동 다운로드 필요) |
| 기간 | 2024-01-02 ~ 2026-04-30 (584행) |
| 인덱스 | `date` (일별) |
| 컬럼 | `vix`, `vix3m`, `slope` (= vix3m − vix), `direction` (= sign(slope 변화)) |

> **타이밍 규약**: VIX/VIX3M 종가는 절대시각 **16:15 ET** 에 확정됩니다.
> 모든 분석은 이 확정 시각 이후의 BTC만 종속변수로 사용합니다 (T-day 시간축). 자세한 내용은
> `analysis/slope_change/slope_change_report.md` 5장 참조.

### `vix_1h.parquet`

| 항목 | 값 |
|---|---|
| 출처 | yfinance (`^VIX`, interval=1h) |
| 기간 | 2024-05-30 ~ 2026-05-29 (7,103행) |
| 컬럼 | `vix` |

---

## 2. BTC 계열

### `btc_1m_24h/btc_1m_YYYY-MM.parquet` (~220MB)

| 항목 | 값 |
|---|---|
| 출처 | **data.binance.vision** (Binance 공개 아카이브, 무료) — BTCUSDT 영구선물 aggTrade |
| 보조 | 2024년 이후 구간은 WebSocket 실시간 수집분 병합 (`data_collection/btc_collector.py`) |
| 수집 | `python data_collection/fetch_btc_1m_historical.py` |
| 기간 | 2020-01 ~ 2026-05 (77개 월별 파일) |
| 인덱스 | `timestamp` (1분, tz = `America/New_York`) |
| 컬럼 | `open` `high` `low` `close` `volume` `quote_volume` `trade_count` `avg_trade_size` `trade_imbalance` |

- `trade_imbalance` = (매수체결량 − 매도체결량) / 총체결량, 범위 −1 ~ +1 (`is_buyer_maker` 기준)
- `avg_trade_size` = 해당 분의 체결 수량 평균

### `btc_historical_processed/btc_1m_YYYY-MM.parquet` (~17MB)

위 데이터를 **미국 장중(09:30~15:59 ET, 주중)** 만 필터링한 버전. LSTM-AE 학습·추론의 직접 입력입니다.

| 컬럼 | 위와 동일 + `trade_date` (거래일 date) |
|---|---|
| 크기 | 월당 ~8,970행 |

전처리 상세(틱 → 1분봉 resample, UTC → ET 변환, 서머타임 처리, 이상행 필터링)는
[`../docs/preprocessing_guide.md`](../docs/preprocessing_guide.md) 참조.

### `funding_rate_history.parquet`

| 항목 | 값 |
|---|---|
| 출처 | Binance Futures API (`fapi/v1/fundingRate`), 무료 — `ccxt` 사용 |
| 수집 | `python data_collection/download_signals.py` |
| 기간 | 2020-01-01 ~ 2026-05-28 (7,018행, 8시간 간격) |
| 컬럼 | `timestamp`, `funding_rate` |

> 펀딩비는 연 환산 시 롱 포지션에 ~12% 수준의 비용이 되며, 얇은 엣지를 소멸시키는 핵심 변수입니다.

---

## 3. VIXY 1분봉

| 항목 | 값 |
|---|---|
| 대상 | **VIXY** (ProShares VIX Short-Term Futures ETF) |
| 출처 | Alpaca Market Data API (`data_collection/fetch_vixy_1m.py`) — 무료 티어로 수집 가능 |
| 대안 | yfinance `VIXY` (1분봉은 최근 30일 제한) |
| 컬럼 | `open` `high` `low` `close` `volume` `trade_count` `vwap` |
| 인덱스 | `timestamp` (1분, ET) |

> **VXX(Barclays ETN)는 발행 구조 리스크 때문에 사용하지 않습니다.** VIXY(ETF)를 사용합니다.
> VIXY는 유동성이 낮아 장중 390분 중 **14~41%만 실거래가 존재** 합니다.
> LSTM-AE 파이프라인에서는 BTC 인덱스에 left join 후 forward fill로 빈 분봉을 채웁니다.

---

## 4. 부가 신호 데이터 (선택)

`analysis/other_signals/` 의 스크립트가 사용하는 데이터입니다. 핵심 결론에는 필요하지 않습니다.

| 파일 | 출처 |
|---|---|
| `btc_etf_daily.parquet` | 미국 BTC 현물 ETF 일별 순유입 |
| `fear_greed_daily.parquet` | alternative.me Fear & Greed Index (무료 API) |
| `dvol_daily.parquet` | Deribit DVOL (크립토 내재변동성) |
| `onchain_daily.parquet`, `metrics_daily.parquet` | 온체인 지표 |
| `google_trends_*.parquet` | Google Trends "bitcoin" 검색량 |
| `global_markets.parquet`, `external_daily.parquet` | 글로벌 지수 (yfinance) |

---

## 5. 재수집 순서 (권장)

```bash
# 1) VIX 일별
python data_collection/fetch_vix_daily.py
#    VIX3M은 CBOE에서 수동 다운로드 → data_collection/NOTE_vix3m.md 참조
#    → vix_slope_daily.parquet 생성

# 2) BTC 1분봉 (오래 걸립니다, ~220MB)
python data_collection/fetch_btc_1m_historical.py
python data_collection/download_klines_extended.py   # 구간 확장

# 3) BTC 장중 필터링본 생성
#    docs/preprocessing_guide.md 절차에 따라 btc_historical_processed/ 생성

# 4) VIXY 1분봉
python data_collection/fetch_vixy_1m.py

# 5) 펀딩비 및 부가 신호
python data_collection/download_signals.py
```

## 6. 실시간 수집 (선택)

`data_collection/btc_collector.py` 는 Binance WebSocket으로 체결(aggTrade) + 호가(LOB)를 실시간 수집합니다.
과거 아카이브 데이터와 컬럼명이 다르므로 전처리 시 매핑이 필요합니다.

| 항목 | 과거 (archive) | 실시간 (collector) |
|---|---|---|
| 타임스탬프 | `time` | `timestamp_ms` |
| 수량 | `qty` | `quantity` |
| boolean | 소문자 `true/false` | 대문자 `True/False` |
| 추가 컬럼 | `id`, `quote_qty` | — |
| 이상 행 | 없음 | `price=0` 간헐 발생 → 필터링 필요 |
| LOB | 없음 | 있음 (LSTM-AE에서는 미사용) |
