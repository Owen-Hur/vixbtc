# VIX-BTC 선물 레버리지 전략 프로젝트 헌법 (Constitution)

> **문서 위상**: 이 문서는 프로젝트의 모든 설계 결정의 최상위 기준입니다.  
> 코드 구현, 백테스트 설계, 보고서 작성은 반드시 이 문서와 정합해야 합니다.  
> 변경이 필요한 경우 반드시 이 문서를 먼저 수정하고 팀 확인을 받습니다.

> **데이터 구간 정책**: 프로젝트 실행 시점의 최신 데이터까지 포함한다.  
> 구간 종료일은 고정값이 아니며, 학습 재실행 시 당일 기준으로 갱신한다.

---

## 0. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **전략 명칭** | VIX Regime-Based BTC Futures L/S Strategy |
| **거래 수단** | BTC/USDT 무기한 선물 (Binance Futures Perpetual) |
| **포지션 방향** | 롱(Long) / 숏(Short) / 중립(Neutral) |
| **신호 생성 주기** | 일 1회 (매일 장 시작 전, 한국 시간 밤 10:30 이전) |
| **기반 논문** | VIX to BTC — VIX 기간구조(slope)와 BTC 수익률의 관계 |
| **핵심 모듈** | Phase A: HMM 장초 포지션 / Phase B: LSTM AE 장중 이상탐지 |

---

## 1. 확정 결정 사항 (Confirmed Decisions)

> 아래 항목들은 팀 회의를 통해 최종 확정된 사항입니다. 재논의 없이 변경 불가.

---

### 1-1. HMM 포지션 결정 방식 — 두 가지 전략 병렬 실험

HMM 3-State 모델의 출력을 포지션으로 변환하는 방식으로 **두 가지를 동시에 실험**하고 성과를 비교한다.

#### 방식 A: 확률 벡터 기반 연속 포지션 (Soft Allocation)

HMM posterior 확률 벡터 `[P(Normal), P(Alert), P(Fear)]`를 그대로 가중합산하여  
레버리지를 **연속적(continuous)** 으로 결정한다.

```python
# 각 State에 사전 배정된 레버리지 값
LEV_NORMAL = +2.0   # 정상: 2배 롱
LEV_ALERT  =  0.0   # 경계: 중립
LEV_FEAR   = -1.0   # 공포: 1배 숏

def soft_leverage(posterior):
    """posterior = [P_normal, P_alert, P_fear]"""
    lev = (posterior[0] * LEV_NORMAL
         + posterior[1] * LEV_ALERT
         + posterior[2] * LEV_FEAR)
    return np.clip(lev, -MAX_LEVERAGE, +MAX_LEVERAGE)
```

- **특징**: 확률 분포 전체를 활용, 레버리지가 부드럽게 변동, 전환 비용 절감
- **리스크**:
  - **Dead Zone 문제**: Normal·Fear posterior가 균등하게 갈릴 때 레버리지가 극소값으로 수렴. 예시 posterior `[0.2, 0.5, 0.3]` → lev = 0.2×2.0 + 0.5×0.0 + 0.3×(−1.0) = **+0.10x**. 가능한 최대 포지션(+2.0x)의 5%에 불과하며, 자본 $10K 기준 실제 익스포저 $1,000. BTC ±5% 움직여도 손익 ±$50인 반면 왕복 수수료 $0.80 즉시 발생 — 비용 대비 기대수익 음수 구간.
  - **Dead Zone 표류**: 모델 불확실 상태(Alert 50%+)가 며칠 지속되면 레버리지가 dead zone 내에서 계속 진동. 매일 포지션을 소폭 조정하나 경제적 실익 없이 거래비용만 누적.
  - **리밸런싱 누적 비용**: posterior가 매일 소폭 변동하면 레버리지도 매일 달라져 불필요한 리밸런싱이 반복 발생. t-1 캐리 구조로 완화 가능하나, dead zone 수렴 자체는 해결 불가.

#### 방식 B: State 매핑 기반 이산 포지션 (Hard Allocation)

가장 높은 posterior 확률의 State로 판정 후 **사전 결정된 레버리지**를 그대로 매핑한다.

```python
# Grid Search로 최적화할 파라미터 (θ)
param_grid = {
    'p_fear_min':   [0.55, 0.65, 0.75, 0.85],  # 공포 확정 임계값
    'p_alert_min':  [0.50, 0.60, 0.70],          # 경계 확정 임계값
    'lev_fear':     [-2.0, -1.0, -0.5, 0.0],     # 공포 레버리지 (숏 허용)
    'lev_alert':    [-0.5, 0.0, 0.5, 1.0],       # 경계 레버리지
    'lev_normal':   [1.0, 1.5, 2.0],             # 정상 레버리지 (롱)
    'max_leverage': [1.0, 2.0, 3.0],             # 절대 상한
}

def hard_leverage(posterior, theta):
    if posterior[2] > theta['p_fear_min']:
        return theta['lev_fear']
    elif posterior[1] > theta['p_alert_min']:
        return theta['lev_alert']
    else:
        return theta['lev_normal']
```

- **특징**: 해석 명확, Walk-Forward Grid Search로 최적화 가능
- **리스크**:
  - **Cliff 전환**: `p_fear_min=0.65` 설정 시 P_fear=0.60인 날 +2x 롱 유지 → 다음날 P_fear=0.85로 돌변하면 즉시 −1x 숏 전환. 자본 $10K 기준 하루 거래 규모 **$30,000**(+2x 청산 $20K + −1x 진입 $10K), 수수료 **$12** 즉시 손실. 당일 BTC −5% 하락 중이었다면 포지션 손실 $1,000 추가.
  - **임계값 근방 소음**: posterior가 임계값 ±0.05 범위 내에서 진동하면 며칠 단위로 `+2x ↔ 0.0x ↔ −1x` 왕복 가능. 각 전환마다 수수료 발생 + 시장 방향 오판 위험.
  - **Grid Search 과적합**: 1,728가지 파라미터 조합이 IS에 과적합되면 OOS에서 cliff 전환 빈도 급증 가능. θ Round 간 표준편차가 클수록 파라미터 불안정 신호로 해석.
  - **마진 부족 위험**: `lev_fear=−2.0, lev_normal=+3.0` 조합 선택 시 단일 전환에서 마진율 급락 가능 → `risk_manager.py` 청산 방지 로직 필수.

#### 비교 평가 기준

| 지표 | 설명 |
|------|------|
| OOS Sharpe | IS/OOS 비율이 1.5 이하인 경우에만 유효 간주 |
| Max Drawdown | 두 방식 모두 동일 기간 비교 |
| 거래 횟수 / 전환 비용 | Soft가 Hard 대비 전환 빈도 감소 여부 확인 |
| θ Round 간 분산 | Hard 방식의 파라미터 안정성 진단 |

---

### 1-2. 실험 조건 정의 — 1 기준선 + 2×2 교차 설계

총 **5가지 실험 조건**을 병렬 비교한다.  
**실행 방식 (2)** × **레버리지 할당 방식 (2)** + **기준선 (1)**

```
[기준선]  B1: Buy & Hold Long (신호 없음, 전 기간 1x 롱)

[실험군]  실행 방식              레버리지 할당      실험 조건
          ─────────────    ×    ─────────────   =  ─────────
          일일 청산 (B2)        Soft (방식 A)       B2-Soft
                                Hard (방식 B)       B2-Hard
          오버나이트 (B3)       Soft (방식 A)       B3-Soft
                                Hard (방식 B)       B3-Hard
```

| 실험 조건 | 실행 방식 | 레버리지 할당 | 오버나이트 포지션 | 펀딩비 적용 |
|----------|---------|-------------|--------------|-----------|
| **B1** | 전기간 보유 | 1x 고정 (없음) | ✅ | ✅ |
| **B2-Soft** | 일일 청산 | Soft (방식 A) | ❌ | ❌ |
| **B2-Hard** | 일일 청산 | Hard (방식 B) | ❌ | ❌ |
| **B3-Soft** | 오버나이트 캐리 | Soft (방식 A) | ✅ | ✅ |
| **B3-Hard** | 오버나이트 캐리 | Hard (방식 B) | ✅ | ✅ |

---

#### 기준선 B1: Buy & Hold Long

- 아무 조작 없이 전체 기간 동안 **BTC 무기한 선물 1배 롱 유지**
- 펀딩비 실제 발생치 반영, 거래비용 없음 (최초 진입 1회만)
- **목적**: 레버리지 시장 자체의 raw 수익 기준선

```
lev_signal = +1.0  # 전 기간 고정
```

---

#### Layer 1 — 실행 방식: 일일 청산 (B2) vs 오버나이트 캐리 (B3)

**B2: 일일 청산 후 HMM 신호 재진입 (Daily Reset)**

- 매일 장 종료 직전 포지션 **전량 청산**, 오버나이트 포지션 없음
- 다음 날 장 시작 시 HMM 신호 기반 레버리지 재진입
- 펀딩비·오버나이트 리스크 미적용

```python
# 매일 반복 (B2-Soft 또는 B2-Hard)
장 종료 전: 전량 청산                              # 오버나이트 없음
장 시작 시: lev = leverage_fn(posterior)          # Soft or Hard 선택
```

**B3: 오버나이트 캐리 + HMM 신호 점진 조정 (Overnight Carry)**

- t-1 포지션 그대로 보유하되, t일 장 시작 전 HMM 신호를 **목표로 점진 이동**
- 펀딩비·오버나이트 리스크 적용

```python
def smooth_rebalance(prev_lev, target_lev, max_step=0.5):
    """
    prev_lev  : t-1 종가 기준 보유 레버리지
    target_lev: HMM 신호 기반 목표 레버리지 (Soft 또는 Hard 출력)
    max_step  : 1회 조정 최대 변동폭 (기본 0.5배, §3 Q1 미결사항)
    """
    delta = target_lev - prev_lev
    delta = np.clip(delta, -max_step, +max_step)
    return prev_lev + delta

# 매일 반복 (B3-Soft 또는 B3-Hard)
target_lev = leverage_fn(posterior)               # Soft or Hard 선택
actual_lev = smooth_rebalance(prev_lev, target_lev)
```

---

#### Layer 2 — 레버리지 할당 방식: Soft (방식 A) vs Hard (방식 B)

B2·B3 각각에 §1-1의 방식 A(Soft) 또는 방식 B(Hard)를 적용한다.

```python
# Soft 적용 시 (B2-Soft, B3-Soft)
target_lev = soft_leverage(posterior)             # §1-1 방식 A

# Hard 적용 시 (B2-Hard, B3-Hard)
target_lev = hard_leverage(posterior, theta)      # §1-1 방식 B
```

---

#### 비교 분석 구조 (Attribution)

| 비교 쌍 | 통제 변수 | 측정 대상 |
|--------|---------|---------|
| B1 vs 나머지 4개 전체 | — | HMM 신호의 전체 기여도 |
| B2-Soft vs B2-Hard | 일일 청산 고정 | Soft vs Hard (오버나이트 없는 환경) |
| B3-Soft vs B3-Hard | 오버나이트 캐리 고정 | Soft vs Hard (오버나이트 있는 환경) |
| B2-Soft vs B3-Soft | Soft 고정 | 일일 청산 vs 오버나이트 캐리의 가치 |
| B2-Hard vs B3-Hard | Hard 고정 | 일일 청산 vs 오버나이트 캐리의 가치 |

---

### 1-3. Dead Zone 필터 실험 계획

Soft 방식(방식 A)에서 발생하는 경제적 무의미 포지션 구간(Dead Zone, `|lev| < threshold`)을 필터링하는 로직의 유효성을 검증하기 위해 아래 세 가지 변형을 병렬 실험한다.

#### 구현

```python
DEAD_ZONE_THRESHOLD = 0.3  # Q7 미결사항 — 1차 실험 후 조정

def apply_dead_zone(target_lev, prev_lev, mode='neutral'):
    """
    target_lev : soft_leverage()가 계산한 레버리지
    prev_lev   : t-1 실제 보유 레버리지
    mode       : 'none' | 'neutral' | 'hold'
    """
    if mode == 'none':
        return target_lev                          # 필터 없음
    if abs(target_lev) < DEAD_ZONE_THRESHOLD:
        if mode == 'neutral':
            return 0.0                             # 강제 중립
        elif mode == 'hold':
            return prev_lev                        # 이전 포지션 유지
    return target_lev
```

#### 실험 변형 A-1: Dead Zone 없음 (기본 Soft)

- Soft posterior 가중합 결과를 그대로 적용. 아무리 작은 레버리지도 포지션 진입.
- **가설**: Dead Zone 구간 비용 누적이 OOS Sharpe를 유의미하게 낮춤.

#### 실험 변형 A-2: Dead Zone → 강제 중립 (`mode='neutral'`)

- `|lev| < threshold` 구간에서 레버리지를 0.0으로 강제 설정. 모델 불확실 구간 완전 대기.
- **가설**: 비용 절감 효과가 포지션 기회 손실보다 큼.

#### 실험 변형 A-3: Dead Zone → 이전 포지션 유지 (`mode='hold'`)

- `|lev| < threshold` 구간에서 t-1 레버리지를 그대로 유지. 시그널 확정 시까지 홀드.
- **가설**: 추세 지속 구간에서 수익 유지, 불필요한 전환 비용 최소화.

#### 비교 평가 기준

| 지표 | 설명 |
|------|------|
| OOS Sharpe | Dead Zone 처리 방식에 따른 리스크 조정 수익 비교 |
| 거래 횟수 | Dead Zone 필터가 리밸런싱 빈도를 얼마나 줄이는가 |
| Dead Zone 발동 비율 | 전체 거래일 중 `|lev| < threshold` 비율 |
| Max Drawdown | 강제 중립 vs 이전 포지션 유지 시 손실 패턴 비교 |
| 수수료 절감액 | A-1 대비 A-2·A-3의 실제 비용 차이 |

> `dead_zone_threshold` 기본값 0.3x는 §3 Q7로 등록된 미결사항. 1차 실험 후 조정.

---

## 2. 아키텍처 원칙 (Architectural Principles)

### 2-1. 데이터 수집

| 데이터 | 소스 | 해상도 | 용도 |
|--------|------|--------|------|
| VIX_9d, 30d, 3m, 6m | CBOE | 일봉 | HMM feature 계산 기반 |
| BTC/USDT 무기한 선물 OHLCV | Binance Futures | 일봉 | 백테스트 수익률 계산 |
| 펀딩비율 (Funding Rate) | Binance Futures | 8시간봉 → 일 합산 | 목적함수 비용 반영 |
| VIXY (ETF) 1분봉 | alpaca api | 1분봉 | Phase B LSTM AE 입력 |

> **⚠️ VXX 사용 금지**: ETN 구조적 괴리율 리스크 (2022년 Barclays 발행 중단 → 30%+ 프리미엄)  
> VIXY(ProShares ETF)로 대체 확정. 코드 내 VXX 참조 전면 교체.

### 2-2. HMM Feature 벡터 (확정)

```python
FEATURES = ['pca1', 'pca2', 'delta_slope', 'vix_30d_level']
```

| Feature | 역할 | 포함 근거 |
|---------|------|----------|
| `pca1` | Level 변화 요인 | 4개 만기 공통 움직임 압축 (분산 96.17%) |
| `pca2` | Slope 변화 요인 | 논문 핵심 변수. slope와 중복이므로 slope 대체 |
| `delta_slope` | 기울기 가속도 | 단기 패닉 가속/둔화 탐지 |
| `vix_30d_level` | 절대 공포 수준 | pca1/2가 놓치는 역사적 극단값 |

> `slope = VIX_9d - VIX_30d` 는 pca2와 심각한 다중공선성(VIF > 5) → **Feature에서 제외**

### 2-3. HMM State 레이블 고정 (필수 원칙)

HMM은 매 학습마다 State 번호가 무작위로 배정됨.  
**VIX_30d 평균 오름차순 정렬**로 강제 재정렬 후 레이블 고정.

```python
state_means = [is_data.loc[hmm.predict(is_feat) == k, 'vix_30d_level'].mean()
               for k in range(3)]
order = np.argsort(state_means)
STATE_MAP = {order[0]: 'normal', order[1]: 'alert', order[2]: 'fear'}
# → state_map.pkl 저장
```

> **재학습 규칙**: `hmm.fit()` 실행 시 `STATE_MAP` 반드시 재계산 → `state_map.pkl` 갱신  
> 매일 추론(`hmm.predict`)에서는 저장된 `STATE_MAP` 그대로 사용.

### 2-4. 목적함수 (선물 전용)

```python
def objective(returns, n_trades, leverage_series, funding_series, param):
    sharpe       = returns.mean() / returns.std() * np.sqrt(252)
    mdd          = max_drawdown(returns)
    trade_cost   = n_trades * 0.0004                          # 선물 taker fee 4bp
    funding_cost = (leverage_series * funding_series).sum()   # 롱: 지불, 숏: 수취
    liq_penalty  = (leverage_series > param['max_leverage']).mean() * 2.0
    return sharpe - 0.5 * abs(mdd) - 0.3 * trade_cost - 0.4 * funding_cost - liq_penalty
```

### 2-5. Walk-Forward 검증 구조

- **3A. Expanding Window**: 학습 구간 매 라운드 확장
- **3B. Rolling Window**: 학습 구간 고정 (6개월)
- 두 방식 모두 실행 후 OOS Sharpe 높은 쪽 선택
- `theta_final = median_params(results)` — 라운드 중앙값 사용

### 2-6. IS/OOS 분리 원칙

#### 구간 설계 정책 (ML 표준 방식)

- **최신 데이터 포함 원칙**: 전체 구간 종료일 = 학습 실행 시점의 가장 최근 거래일
- **분할 비율**: IS 80% / OOS 20% (시간순 분할, 무작위 분할 절대 금지)
- **OOS 기준일 정렬**: 분기 또는 월 초로 맞춰 해석 편의성 확보

#### 확정 구간 (2026-04-30 기준)

```
전체 : 2024-01-15 ~ 2026-04-30  (약 27.5개월, ~580 거래일)
IS   : 2024-01-15 ~ 2025-10-31  (약 460 거래일, ~79%)  ← HMM 학습 및 Grid Search
OOS  : 2025-11-01 ~ 2026-04-30  (약 125 거래일, ~21%) ← 최종 1회 평가에만 사용
```

| 구간 | 기간 | 거래일 수 | 비율 | 포함된 주요 시장 이벤트 |
|------|------|-----------|------|------------------------|
| **IS** | 2024-01-15 ~ 2025-10-31 | ~460일 | 79% | BTC 반감기(2024-04), ATH 돌파(2024-11), 사후 조정 국면 |
| **OOS** | 2025-11-01 ~ 2026-04-30 | ~125일 | 21% | Post-ATH 변동성 구간, 2026년 초 조정 국면 |

> **OOS 데이터는 최종 1회 평가에만 사용.** 파라미터 탐색·하이퍼파라미터 튜닝 중 OOS 접근 시 해당 실험 결과 전면 무효 처리.

#### 데이터 구간 연장 타당성 근거 (ML 전문가 표준)

- **López de Prado** 원칙: 가용한 최신 데이터를 최대한 활용. 데이터 연장 자체는 Look-ahead bias가 아님.
- **Look-ahead bias 발생 조건**: OOS가 IS보다 시간상 앞에 있을 때. 이 구간 설계는 해당 없음.
- **OOS 품질**: 2025-11 ~ 2026-04는 IS에 없는 시장 국면(소타 하락기, BTC 회복기) 포함 → 실질적 OOS 향습도 높음.
- **6개월 OOS**: 단순 Sharpe 개산이 아닌 실제 운용에서도 평가할 수 있는 충분한 기간.

#### Walk-Forward 라운드 구성 (IS 내부 검증)

```
Round 1 — Expanding:  train [2024-01 ~ 2024-06]  val [2024-07 ~ 2024-09]
Round 2 — Expanding:  train [2024-01 ~ 2024-09]  val [2024-10 ~ 2025-01]
Round 3 — Expanding:  train [2024-01 ~ 2025-01]  val [2025-02 ~ 2025-05]
Round 4 — Expanding:  train [2024-01 ~ 2025-05]  val [2025-06 ~ 2025-10]

# López de Prado: val set 앞뒤 embargo 5거래일 적용 (serial leakage 방지)
```

#### 구간 갱신 규칙

- OOS 기간(6개월)은 고정, IS는 데이터 새로 수집 시 확장
- 갱신 내역은 §5 변경 이력에 날짜와 함께 기록

---

## 3. 미결 사항 (Open Questions)

> 결정 전까지 기본값으로 진행. 결정 시 §1 또는 §2 해당 항목 업데이트.

| # | 항목 | 현재 기본값 | 결정 필요 시점 |
|---|------|------------|--------------|
| Q1 | Benchmark 3의 `max_step` (스무딩 속도 상한) | 0.5배 | Walk-Forward 완료 후 |
| Q2 | 공포 State 숏 허용 여부 (`lev_fear` 음수 가능?) | 허용 (Grid Search에 포함) | 백테스트 결과 확인 후 |
| Q3 | 방식 A Soft 레버리지의 State별 기준값 (`LEV_NORMAL`, `LEV_ALERT`, `LEV_FEAR`) | +2.0 / 0.0 / -1.0 | 1차 실험 후 조정 |
| Q4 | HMM 재학습 주기 | 월 1회 전체 재학습 | 온라인 학습 대안 비교 후 |
| Q5 | 파라미터 탐색 방식 | 전수 Grid Search (1,728가지) | 계산 예산 확정 후 (Random Search 대안 검토) |
| Q6 | LSTM AE Phase B와의 연동 | 현재 분리 운영 | HMM 모듈 완료 후 |
| Q7 | Soft 방식 Dead Zone 임계값 (`dead_zone_threshold`) — `|lev| < threshold` 시 처리 모드(강제 중립 vs 이전 포지션 유지) 및 임계값 수치 결정 | 0.3x / 미설정 (§1-3 실험 대상) | §1-3 Dead Zone 실험 완료 후 |
| Q8 | Walk-Forward 라운드별 best θ 통합 방식 — **중앙값(median) vs 추론 시점 앙상블(Val OOS Sharpe 가중)** 선택. 앙상블 선택 시 음수 Sharpe 라운드 처리 방식(0 클리핑 vs softmax vs 제외) 추가 결정 필요 | 중앙값 (현행) | Walk-Forward 완료 후 비교 실험 |

---

## 4. 모듈 구조

```
hmm_module/
├── config.py            # FEATURES, param_grid, 타임라인, MAX_LEVERAGE
├── data_loader.py       # VIX 일봉, BTC 선물 OHLCV, 펀딩비율 수집
├── feature_engineer.py  # ΔVIX, PCA, slope, delta_slope 계산
├── hmm_trainer.py       # HMM fit + STATE_MAP 고정
├── grid_search.py       # 방식 A·B 목적함수 + Grid Search 루프
├── walk_forward.py      # Expanding / Rolling 라운드 실행
├── benchmark.py         # Benchmark 1·2·3 포트폴리오 계산
├── oos_evaluator.py     # OOS 성과 비교 (전략 vs 3개 비교군)
├── risk_manager.py      # 레버리지 클리핑, 마진 체크, 청산 방지
├── daily_inference.py   # 실전 추론 → 레버리지 신호 출력
└── artifacts/
    ├── scaler.pkl        # IS fit 고정 (StandardScaler)
    ├── pca.pkl           # IS fit 고정 (PCA n=2)
    ├── hmm.pkl           # 학습된 GaussianHMM
    ├── state_map.pkl     # STATE_MAP (재학습 시 갱신)
    └── theta_final.pkl   # 최적 θ (방식 B용)
```

---

## 5. 변경 이력

| 날짜 | 변경 내용 | 사유 |
|------|----------|------|
| 2026-05-19 | 최초 작성 | 팀 회의 결정 사항 문서화 |
| 2026-05-19 | 현물 → 선물 전환 | 레버리지 롱/숏 전략 채택 |
| 2026-05-19 | 비교군 3종 확정 | B1(전체 롱), B2(일일 청산), B3(HMM 스무딩) |
| 2026-05-19 | HMM 방식 A·B 병렬 실험 채택 | Soft vs Hard 비교 실험 결정 |
| 2026-05-19 | IS/OOS 구간 2026-04-30까지 연장 | ML 표준(최대 가용 데이터 활용) 적용, OOS 6개월 확보 |
| 2026-05-19 | IS/OOS 구간 재설계 | 고정 종료일(2025-05) → 최신 데이터 포함(2026-05-16), IS 80%/OOS 20% ML 표준 적용 |
| 2026-05-24 | §1-1 Soft·Hard 리스크 상세화 | Dead Zone 수치 예시, Cliff 전환 비용, Grid Search 과적합·마진 리스크 구체화 |
| 2026-05-24 | §1-3 Dead Zone 실험 계획 추가 | Soft 방식 A-1(필터 없음) / A-2(강제 중립) / A-3(이전 포지션 유지) 병렬 실험 설계 |
| 2026-05-24 | §3 Q7 추가 | Dead Zone 임계값 및 처리 모드 결정 항목 미결사항 등록 |
| 2026-05-24 | §3 Q8 추가 | 라운드별 θ 통합 방식(중앙값 vs 추론 시점 앙상블) 미결사항 등록 |
| 2026-05-24 | §1-2 전면 재설계 | 비교군 3종 → 1 기준선 + 2×2 교차 설계(5가지 조건)로 재구조화. Layer 1(실행 방식: B2 일일청산 vs B3 오버나이트) × Layer 2(레버리지: Soft vs Hard) 명시 |
