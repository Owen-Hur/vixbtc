# 참고 논문 목록

## 핵심 논문 (프로젝트 기반)

### 0. Luo, Tsai & Yen (2026) — 프로젝트 출발점
- 제목: Volatility Transmission to Bitcoin: The Role of VIX Term Structure and Crypto Options Markets
- 저널: SSRN preprint (미출판, 피어 리뷰 없음)
- 파일: `../ssrn-6233752.pdf`
- 핵심 발견: VIX slope(PCA2)이 level(PCA1)보다 BTC 수익률에 2.3배 강한 설명력. 단, 동시적 관계만 유의하고 래그 효과는 없음.
- 프로젝트 적용: Slope-HMM 전략의 근거로 사용 → look-ahead bias 수정 후 전략 실패 → 폐기
- 비판: 동시적 관계만 존재하므로 트레이딩 시그널로 사용 불가. PCA 스케일 차이 미고려, R²=0.118로 설명력 미미, 시간대 불일치 미언급.


## 관련 논문 (리서치 완료)

### 1. Chiu, Hung & Yen (2025)
- 제목: SVIX, VIX, and cryptocurrency market return
- 저널: The Quarterly Review of Economics and Finance (피어 리뷰)
- 피인용: 1회
- 데이터: 2014~2022
- 핵심: VMS(= VIX² - SVIX²)가 높을수록 이후 크립토 수익률이 높다. 래그 예측력 존재를 주장.
- 비고: Luo 논문 공저자(Yen) 동일. SVIX 데이터 접근이 필요.

### 2. Wang, Ma, Bouri & Guo (2023)
- 제목: Which factors drive Bitcoin volatility: macroeconomic, technical, or both?
- 저널: Journal of Forecasting (피어 리뷰, Wiley)
- 피인용: 108회
- 데이터: 2011.12 ~ 2021.04
- 핵심: 17개 거시경제 + 18개 기술적 지표로 BTC 변동성 예측. S&P 500 RV, 글로벌 실물경제 활동지수, 무역가중 달러 인덱스가 가장 유효. 저변동성 구간에서는 기술적 지표가 우위.
- 방법론: Elastic Net, LASSO
- 비고: 예측(forecasting) 프레임워크 사용. 가장 견고한 논문. RV 기반 접근의 학술적 근거.

### 3. Shahrour, Lemand & Mourey (2025)
- 제목: Cross-market volatility dynamics in crypto and traditional financial instruments
- 저널: The Journal of Risk Finance
- 피인용: 15회
- 핵심: BTC, NEAR이 변동성 spillover의 주요 송신자. Treasury Bills, GBTC는 수신자. 양방향 전이 존재.
- 비고: spillover 방향성 분석. 직접적 예측력 검증은 아님.

### 4. Mensi, Gubareva, Ko, Vo & Kang (2023)
- 제목: Tail spillover effects between cryptocurrencies and uncertainty in gold, oil, and stock markets
- 저널: Financial Innovation (Springer, 피어 리뷰)
- 피인용: 115회
- 핵심: 정상 시장에서 크립토-전통자산 연결 약함, 극단적 상황에서 급격히 강해짐. 크립토가 변동성 지수에 리더십 영향력.
- 비고: "극단 상황에서만 관계 강화" → 레짐 전환 전략과 연결 가능. VIX-BTC → RV 전환의 근거.

### 5. García-Medina & Aguayo-Moreno (2024)
- 제목: LSTM-GARCH hybrid model for the prediction of volatility in cryptocurrency portfolios
- 저널: Computational Economics (Springer)
- 피인용: 114회
- 핵심: LSTM + GARCH 하이브리드가 단독 모델보다 변동성 예측 정확도 높음.
- 비고: 변동성 예측 방법론 참고용.


## 프로젝트 서사 (VIX-BTC → RV Regime)

1. Luo et al. (2026)의 "VIX slope → BTC" 발견을 기반으로 Slope-HMM 전략 시도
2. Look-ahead bias 수정 후 래그 예측력 소멸 확인 (hit rate 47.4%)
3. 논문 재분석: "동시적 관계만 유의, 래그 효과 없음" — 전략 도출 구조적 불가
4. Wang et al. (2023): BTC 변동성 예측에 S&P 500 RV가 핵심 → 외부 IV가 아닌 RV가 유효
5. Mensi et al. (2023): 정상 시장에서 크립토-전통자산 연결 약함 → 외부 시그널 한계
6. 결론: 외부 시그널(VIX)에서 내부 시그널(BTC 자체 RV)로 전환
7. RV Regime 전략 구축 → 편향(당일 데이터 포함 percentile·펀딩 미반영·짧은 검증구간) 제거 후 Sharpe 0.02, B&H 대비 -109.0%p로 알파 소멸
