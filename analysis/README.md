# analysis/ — 분석 모듈 인덱스

연구가 진행된 순서대로 정리했습니다. **★ 표시가 최종 결론을 뒷받침하는 핵심 분석** 입니다.

| 폴더 | 질문 | 결론 | 상태 | 재실행 검증 (2026-09-08) |
|---|---|---|---|---|
| `rv_regime/` | BTC 자체 RV 레짐으로 방향을 잡을 수 있는가? | 편향 제거 시 Sharpe 0.02, B&H 대비 -109.0%p — 알파 소멸 | 기각 | ✅ 편향 제거판 완전 재현 (-57.4%, Sharpe 0.02, -109.0%p) |
| `vix_response/` | VIX 발표 후 BTC는 언제·어떻게 반응하는가? | 2020~2022엔 6h 피크, 2023+ 소멸 | 중간 단계 | 재실행 안 함 (최종 결론에 미기여하는 중간 단계 코드) |
| `vix_overnight_granger/` | 야간 구간에 시차 예측력이 남아 있는가? | Granger 인과 불성립 | P0 종결 | ✅ 리포트 3종 바이트 단위 동일 재생성 |
| `vix_threshold/` | 임계 필터 + 부호를 데이터로 학습하면? | Bonferroni 보정 후 유의 임계 없음 | 기각 | ✅ 리포트·CSV 바이트 단위 동일 재생성 |
| **`slope_change/` ★** | slope 변화율이 익일 장중 방향을 예측하는가? | **IS 50.5% / OOS 50.4%, p=0.836** | **기각** | ✅ 완전 재현 (편향 대조군 OOS 61.8% 포함) |
| **`vix_duration/` ★** | VIX 영향이 얼마나 오래 남아 예측에 쓰이는가? | **1,035시점 전부 ≈50%, Bonferroni 통과 0개** | **기각** | ✅ minute_metrics.csv 완전 재현 (0개 / 249개) |
| `other_signals/` | 펀딩비 · ETF flow · DVOL · F&G는? | 단독 예측력 없음 (펀딩 r = −0.005) | 부가 | 재실행 안 함 (최종 결론에 미기여하는 부가 신호) |

> **재현 방법 메모**
> - 분석 스크립트는 저장소 루트의 `data/` 를 기준 경로로 참조합니다. 스크립트 상단의 `BASE_DIR` 를 로컬 경로로 바꿔 실행하십시오.
> - `vix_overnight_granger/code/vix_granger.py` 는 `grangercausalitytests(..., verbose=False)` 를 사용합니다.
>   statsmodels 0.15+ 에서는 이 인자가 제거되었으므로 삭제하거나 `statsmodels<0.15` 를 설치하십시오(결과는 동일).
> - `rv_regime/revised/` 는 `rv_regime` 패키지로 import되며 `config.py` 가 별도로 필요합니다
>   (`DATA_DIR`, `BTC_1M_DIR`, `TRADE_DAY_CUTOFF_HOUR=17`, `TAKER_FEE=0.0004`, `SW/LW/QH=3/75/0.80`, `IS_END="2022-12-31"`).

## 읽는 순서 (권장)

1. `slope_change/slope_change_report.md` — 아이디어 탄생부터 기각까지의 전 과정 (입문자용 서술)
2. `vix_duration/vix_duration_report.md` — "변동성 동조 ≠ 방향 예측" 핵심 통찰
3. `vix_overnight_granger/reports/VIX_granger_report.md` — 정식 통계검정으로 종결
4. `../lstm_ae/REPORT.md` — 장중 이상탐지 모듈의 IS/OOS 결과

## 공통 설계 규약

- **T-day 시간축**: VIX 종가 확정(16:15 ET) 직후를 0:00으로 두어 lookahead를 구조적으로 배제.
  `slope_change/` 와 `vix_duration/` 이 동일한 1,035시점 축을 공유하므로 결과를 직접 비교할 수 있습니다.
- **IS/OOS 80/20 시간순 분할** (`../constitution.md`), OOS는 최종 1회 평가 전용.
- **다중검정 보정**: 분 단위 다중 시점 검정에는 Bonferroni 보정 적용.
- **50% 기준선**: 방향 적중률의 귀무가설은 항상 50%(무작위)이며 binomial test로 검정.
