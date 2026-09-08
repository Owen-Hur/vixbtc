> ## ⚠️ 폐기된 문서 (SUPERSEDED) — 재현되지 않은 성과 수치는 삭제되었습니다
>
> 본 리포트(2026-05-24)는 slope_change 를 확정 전략으로 채택했으나,
> 이후 시간축을 T-day 체계로 재정의해 **lookahead bias를 구조적으로 제거하고 재검증한 결과
> 적중률은 IS 50.5% / OOS 50.4% (binomial p = 0.836)로 무작위 수준에 수렴** 했습니다.
>
> **2026-09-08 정리**: 이 문서가 보고했던 확정 전략의 백테스트 성과(수익률·Sharpe·적중률·MDD),
> 레버리지 스케일링 근거, 비교군 성과는 **생성 백테스트 스크립트가 저장소·원본 프로젝트 어디에도 남아 있지 않아 재현할 수 없습니다.**
> 따라서 해당 수치와 그에 의존하는 섹션(원 1장 요약, 2.7 필터 비교, 4장 확정 전략 상세, 3.2 강건성 검증, 5~7장)을 **모두 삭제** 했습니다.
> 남긴 것은 lookahead와 무관하고 재현이 확인된 두 부분뿐입니다 —
> **장중 개입 실패 실험 기록**과 **LSTM-AE 잔차 분해**(OOS 15분 적중률 54.3%, `lstm_ae/residual_analysis.py` 재실행으로 로그와 완전 일치).
>
> 유효한 결론은 다음 문서를 참조하세요:
> - [`analysis/slope_change/slope_change_report.md`](../analysis/slope_change/slope_change_report.md) — 재검증 최종 보고서
> - [`analysis/slope_change/compute_biased.py`](../analysis/slope_change/compute_biased.py) — 재현 가능한 편향 대조군(OOS 61.8%)
> - [`analysis/vix_duration/vix_duration_report.md`](../analysis/vix_duration/vix_duration_report.md)

---

# BTC 장중 트레이딩 전략 — 장중 개입 실험 기록 (2026-05-24)

> 담당: LSTM-AE 모듈 / 트레이딩 모델
> 이 문서에 남은 내용: 장중 개입 12종의 실패 기록 + LSTM-AE 잔차 분해.

## 1. 실험 결과 — 시도한 것과 실패한 것

총 12가지 장중 개입 방법을 시도했으며, **전부 실패**했다. 아래는 각 실험의 요약이다.

> **읽는 법**: 아래 표의 Alpha(%p)는 당시 기준 전략(일봉 slope_change 매일 거래) 대비 초과성과입니다.
> 그 기준 전략 자체가 이후 lookahead bias로 기각되었으므로 절대 수치로 읽지 말고,
> **"장중 개입은 어떤 형태로도 기여하지 못했다"** 는 방향성만 유효한 결과로 보십시오.

### 1.1 LSTM Autoencoder 이상 탐지 기반 포지션 조절

**목표**: VXX-BTC 관계의 이상을 탐지하여 포지션 전환/청산/레버리지 조절

**모델 구조**:

- 인코더: LSTM(64) -> LSTM(32), 디코더: LSTM(32) -> LSTM(64) -> Dense(7)
- 입력: 60분 x 7피처 (btc_return, trade_imbalance, trade_count, avg_trade_size, vixy_return, vixy_rolling_std, vixy_btc_corr)
- 장초반: 15분/30분 별도 모델, 임계값: 상위 7.5%
- 메인 모델: 65,223 파라미터 (`model.pt` state_dict 실측)

| 활용 방식                          | IS Alpha | OOS Alpha | 실패 원인 |
| anomaly -> 포지션 청산              | -100%p   | -114%p    | 수익 구간을 버림 |
| anomaly -> 레버리지 축소 (2x->1x)   | -101%p   | -104%p    | 동일 |
| anomaly -> 레버리지 증가 (2x->3x)   | -        | -96%p     | 고변동성 양날의 검 |
| anomaly -> 진입 필터               | +/-0%p   | +/-0%p    | 장 시작 시 anomaly 드뭄 |
| anomaly score 기반 연속 사이징       | -258%p   | -297%p    | 과다 거래 비용 |
| Score multiplier (2x~5x)         | 음수     | 음수      | multiplier 무관하게 실패 |

**근본 원인**: LSTM-AE가 잡아내는 VIXY-BTC 관계 이탈과 일봉 신호의 성패는 서로 독립적인 현상이었다.

### 1.2 XGBoost 방향 분류기

**목표**: anomaly 발생 시 향후 방향 예측. XGBoost (depth=4, n_est=300, 13개 피처).

| 조건                         | OOS 정확도 | 샘플 수 | 판정       |
| 기본 (fwd=15min)             | 50.8%      | 6,810   | 랜덤       |
| fwd=60min + +/-0.3% 필터    | 55.3%      | 4,315   | 소표본     |
| fwd=15min + 2x score        | 55.8%      | 792     | 소표본     |
| fwd=60min + 3x score        | 57.7%      | 300     | 매우 소표본 |

**결론: 분봉 미시구조 기반 방향 예측 불가. 기각.**

### 1.3 장중 최적 청산

**목표**: 조기 청산/trailing stop으로 수익 보존

| 규칙                 | IS Alpha        | OOS Alpha       |
| Target Profit 1~4%  | -214 ~ -365%p   | -113 ~ -245%p   |
| Trailing Stop 0.5~2% | -82 ~ -118%p   | -68 ~ -112%p    |
| Stop Loss 1~3%      | -32 ~ -72%p     | -45 ~ -70%p     |
| TS + SL 조합         | -113 ~ -153%p   | -122 ~ -135%p   |

**실패 원인**: 장중 최고점의 30%가 15시(마감 직전)에 발생. 조기 청산 = 수익 포기.

### 1.4 GARCH 변동성 기반 포지션 사이징

**목표**: 예측 변동성으로 레버리지 동적 조절

- BTC 장중 변동성의 일별 자기상관: r=0.067 (IS), r=0.139 (OOS) -- **통계적으로 0**
- 전일/5일 변동성 기반 vol-targeting: 모두 음의 Alpha (-268 ~ -376%p)

**실패 원인**: BTC 장중 변동성에 일별 지속성이 없음. GARCH 가정 불성립. **기각.**

### 1.5 변동성 브레이크아웃 (Anomaly 기반)

**목표**: anomaly 발생 시 방향 예측 없이 양방향 브레이크아웃으로 변동성 수익 포착

**전략 A: Slope + 브레이크아웃 오버레이** — IS/OOS 일관성 없음. **기각.**

**전략 B: Anomaly 기반 Vol 타겟팅** — 12종 중 유일하게 양의 Alpha가 나왔으나 크기가 미미. **기각.**

**전략 C: 순수 브레이크아웃 (Slope 무관)** — 소표본, 낮은 SR. **기각.**

### 1.6 일봉 피처 기반 신뢰도 분석

**목표**: slope 신호 성공/실패를 예측하는 피처 탐색

7개 피처(slope_change 크기, slope 수준, VIX 수준, 연속 방향 일수, anomaly 비율, BTC 변동성, 전일 수익률) 전부 IS t-test 비유의미 (p > 0.3).

> 당시에는 분위수별 승률 패턴을 근거로 레버리지 스케일링을 도입했으나, 그 승률은 lookahead가 제거되지 않은 측정이었다.
> T-day 축에서 재검증한 |slope_change| 강도별 적중률은 48.1~51.4%로 강도 가설이 기각되었다
> ([`analysis/slope_change/slope_change_report.md`](../analysis/slope_change/slope_change_report.md)).

### 1.7 장중 개입 실패의 근본 원인

12가지 장중 개입이 모두 실패한 이유를 구조적으로 정리한다.

| 개입 유형          | 시도한 방법                        | 실패 원인 |

| 포지션 전환/청산   | LSTM-AE anomaly                    | anomaly와 slope 실패가 독립적 |
| 레버리지 동적 조절 | anomaly score, GARCH, vol-targeting | BTC 장중 변동성에 일별 지속성 없음 |
| 조기 청산          | TP, TS, SL, 조합                   | 최고점 시간 균등 분포, 마감 직전 30% |
| 방향 예측          | XGBoost, 잔차 분해                 | 분봉 미시구조의 방향 예측력 ~50% |
| 변동성 트레이딩    | 브레이크아웃, vol 타겟팅           | IS/OOS 불일관, 비용 대비 수익 부족 |

**구조적 결론**: VIX slope은 일봉 신호이다. 일봉 신호가 지배하는 전략에서 분봉 레벨의 추가 정보는 노이즈일 뿐이다. 장중 개입은 가치를 추가하지 못하고, 거래 비용만 증가시킨다.

---

## 2. LSTM-AE 잔차 분해 — 약한 방향 정보

> 이 절의 수치는 `lstm_ae/residual_analysis.py` 를 재실행해 커밋된 로그와 숫자 단위로 일치함을 확인했습니다.

anomaly가 발생한 시점에서, 각 피처의 signed residual을 계산:

```
signed_residual = actual - reconstructed
양수: 실제 값이 모델 예측보다 높음
음수: 실제 값이 모델 예측보다 낮음
```

**피처별 상관 결과:**

| 피처                  | IS r (15분) | OOS r (15분) | 해석 |
| **btc_return**        | **+0.247** | **+0.081** | BTC가 예상 외 상승 -> 계속 상승 (모멘텀) |
| **trade_imbalance**   | **+0.108** | **+0.087**  | 매수 우위 -> BTC 상승 |
| vixy_return           | -0.036   | -0.046    | VXX 예상 외 상승 -> BTC 하락 |
| avg_trade_size        | -0.034   | -0.020     | 대형 거래 -> BTC 하락 |

**방향 예측 정확도:**

| 예측 규칙                        | IS 정확도  | OOS 정확도 | 판정 |
| BTC 잔차 모멘텀 (양수->Long)     | **57.1%**  | **54.3%**  | 약하지만 IS/OOS 일관 |
| VXX 잔차 반전 (양수->Short)      | 51.2%      | 53.1%      | 약함 |

**판정**: BTC 잔차 모멘텀(54.3% OOS)은 통계적으로 유의미하나, 단독 트레이딩 신호로 사용하기에는 **너무 약하다.** 향후 다른 신호와의 결합 가능성은 열려 있으나, 단독 신호로는 사용하지 않는다.
