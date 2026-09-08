# 1차 알파 탐색 아카이브 (원본 `first/` 폴더)

BTC 알파 전략 탐색 과정에서 진행한 1차 작업물의 기록이다.
**모든 시도가 실패했지만**, 실패 과정 자체가 다음 단계 설계의 근거가 된다.

## 산출물 위치

원본 프로젝트의 `first/` 폴더에 있던 산출물이며, 이 저장소에서는 아래 위치에 흩어져 있다.

```
analysis/rv_regime/
├── README_first_alpha_search.md       # 본 문서
├── revised/                           # 편향 제거판 RV Regime 전략 코드
│   ├── config.py                      # 경로 · TAKER_FEE · SW/LW/QH · IS_END
│   ├── strategy.py                    # percentile 수정 (values[:i])
│   ├── data_loader.py                 # load_daily_funding 포함
│   └── walk_forward.py                # 펀딩 반영 + 22 라운드
└── results/
    ├── backtest_result.txt            # sw=3/lw=75/qh=0.8 IS·OOS·FULL 성과
    └── lstm_ae_residual_analysis.log  # LSTM-AE 잔차 분석 출력

analysis/other_signals/                # 아래 3~7단계를 생성한 스크립트
├── 01_download_funding_rate.py
├── 02_download_vix_daily.py
├── 03_btc_volatility_direction.py
├── 04_fixed_param_with_funding.py     # ← 3단계
├── 05_rv_ratio_funding_combined.py    # ← 4단계
├── 06_vix_ma_alignment.py             # ← 5단계
├── 07_vix_predictive_power.py         # ← 6단계
└── 08_vix_release_btc_response.py     # ← 7단계

docs/strategy_journal.md               # 6단계 실패 여정 일지 (원 STRATEGY_JOURNAL.md)
data/                                  # vix_daily · funding_rate_history 등 (재수집 안내: data/README.md)
```

## 작업 흐름 요약

### 1단계: RV Regime v1 (편향 내재)
BTC 자체 실현변동성(rv_ratio) 레짐으로 방향을 잡는 전략이 시작점.
3가지 편향(당일 데이터 포함 percentile, 펀딩 미반영, 1.6년 짧은 검증)을 발견.

> **v1의 성과 수치는 이 저장소에 싣지 않습니다.** 초기 버전은 lookahead(자기참조) 편향으로 부풀려진 결과였고,
> 원본 코드가 저장소에 남아 있지 않아(`revised/` 만 보존) 정확한 수치를 확인할 수 없습니다.
> **확정값은 아래 2단계(편향 제거판)** 입니다.

### 2단계: 편향 수정 → 알파 소멸
- `strategy.py`에서 expanding percentile을 `values[:i+1]` → `values[:i]`로 수정
- 펀딩레이트 다운로드 후 전략 비용에 반영
- 데이터 기간 확장 (2021-01 ~ 2026-05, 22 WF 라운드)
- 결과: Sharpe 0.02, B&H -109%p (편향 제거 시 알파 사라짐)

> **재실행 검증 (2026-09-08) — 완전 재현 ✅**
> `python -m rv_regime.walk_forward` 실행 결과:
> 기간 2021-01-01 ~ 2026-05-07 (1,964일), 전략 **-57.4% (Sharpe 0.02, MDD -84.7%)**,
> B&H **+51.6%**, 초과 **-109.0%p**, 22라운드 중 test Sharpe > 0 은 12/22, 평균 test Sharpe 0.32.
> 펀딩레이트 실측 일평균 **0.0328%** (연 ~12%). 문서 수치와 소수점까지 일치합니다.
>
> 실행에 필요한 `rv_regime/config.py` 는 원본 저장소에 커밋되어 있지 않아 문서 기재값으로 복원했고,
> 값마다 출처를 주석으로 달아 `revised/config.py` 로 포함했습니다:
> `TRADE_DAY_CUTOFF_HOUR=17`, `TAKER_FEE=0.0004`(taker 0.04% 편도), `SW/LW/QH=3/75/0.80`, `IS_END="2022-12-31"`,
> `DATA_DIR`/`BTC_1M_DIR` 는 저장소 `data/` 기준.

### 3단계: 고정 파라미터 검증 (`analysis/other_signals/04_fixed_param_with_funding.py`)
- Train 2020-2022 → OOS 2023-2026 (lookahead bias 없음)
- 선택: sw=4, lw=14, qh=0.60 (Train Sharpe 1위)
- OOS +89.4%, Sharpe 0.63이지만 B&H 대비 -173.9%p

### 4단계: rv_ratio + 펀딩 결합 (`analysis/other_signals/05_rv_ratio_funding_combined.py`)
- 4,200 조합 그리드 서치
- 펀딩레이트 단독 예측력 분석: vs 다음날 BTC 상관 -0.005 (사실상 0)
- 두 조건 동시 충족 시 Short → Short 비율 3.2% → 사실상 B&H

### 5단계: VIX MA 정배열/역배열
- 5-MA (5,10,20,40,60일) 정배열/역배열
- 정배열 진입 후 BTC 오히려 상승 (직관과 반대)
- 백테스트: B&H 대비 -533%p 대패

### 6단계: VIX 방향 예측력 종합
- 5가지 시그널(level, ΔV, %ΔV, 5일변화, MA비율)
- VIX vs 다음날 BTC 상관: +0.063 (양 상관)
- 직관과 반대 + 매우 약함

### 7단계: VIX 발표 후 BTC 시간별 반응 (핵심 발견)
- VIX 종가 발표(ET 16:15) 후 1분~24시간 누적 수익률 측정
- 전체 기간(1,635일): 6시간이 피크 (r=-0.176, p<0.001)
- **그러나 2023년 이후 효과 완전 소멸**
- 2020~2022가 만든 통계, 현재 시점에서는 트레이딩 활용 불가

## 결정적 발견

1. **편향 제거가 결정적**: 미세한 편향 3개가 합쳐지면 가짜 알파를 만든다
2. **BTC 변동성은 양방향**: 전통 주식(VIX↑≈하락)과 다름 → "RV↑→Short" 약함
3. **펀딩 비용은 거대**: 연 ~12% Long 비용이 얇은 에지를 소멸
4. **VIX→BTC 영향력 시간변화**: 2020-2022에는 6h 유효, 2023+ 소멸
5. **크립토 시장은 B&H가 강한 벤치마크**: Short 포함 전략은 구조적 불리

## 다음 단계

이 1차 탐색은 **무엇이 작동하지 않는지**를 정확히 보여주었다.
여기서 얻은 "외부 시그널(VIX)로는 시차 예측이 안 된다"는 결론이
`analysis/slope_change/` · `analysis/vix_duration/` 의 T-day 재검증으로 이어졌다.
