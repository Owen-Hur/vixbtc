"""
LSTM Autoencoder 하이퍼파라미터 및 경로 설정

모든 설정값을 여기서 관리합니다.
수정이 필요하면 이 파일만 변경하면 됩니다.
"""

from pathlib import Path

# ─── 경로 ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "btc_historical_processed"
VIXY_DIR = BASE_DIR / "data" / "vixy_1m"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# ─── 데이터 구간 ──────────────────────────────────────────
# IS: 모델 학습용 (과거)
# OOS: 성능 검증용 (미래, 학습에 절대 사용하지 않음)
IS_START = "2024-01"
IS_END = "2025-04"
OOS_START = "2025-05"
OOS_END = "2026-04"

# IS 내부에서 학습/검증 분할 비율
TRAIN_RATIO = 0.8  # 80% 학습, 20% 검증 (시간순 분할)

# ─── 피처 ─────────────────────────────────────────────────
# BTC 4개 + VIXY 3개 = 7개 피처
FEATURES = [
    "btc_return", "trade_imbalance", "trade_count", "avg_trade_size",
    "vixy_return", "vixy_rolling_std", "vixy_btc_corr",
]
N_FEATURES = len(FEATURES)

# ─── 모델 구조 ────────────────────────────────────────────
WINDOW_SIZE = 60        # 60분 = 1시간 window (메인)
ENCODER_HIDDEN = [64, 32]  # 인코더 LSTM 레이어 크기
LATENT_DIM = 32            # 잠재 벡터 차원 (= 인코더 마지막 레이어)
DECODER_HIDDEN = [32, 64]  # 디코더 LSTM 레이어 크기

# 장초반 모델: 09:45~10:28 (메인 60분 window 축적 전)
EARLY_WINDOW_SIZES = [15, 30]  # 09:45~09:59 → 15분, 10:00~10:28 → 30분
EARLY_ENCODER_HIDDEN = [32, 16]
EARLY_DECODER_HIDDEN = [16, 32]
EARLY_LATENT_DIM = 16

# ─── 학습 설정 ────────────────────────────────────────────
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 50
EARLY_STOP_PATIENCE = 5  # 검증 loss가 5 epoch 연속 안 줄면 중단

# ─── 이상 탐지 ────────────────────────────────────────────
THRESHOLD_PERCENTILE = 92.5  # 상위 7.5% = 92.5 백분위수
# 장 초반 (09:45~10:29): window가 짧으므로 별도 임계값
EARLY_SESSION_END = "10:30"

# ─── 장중 설정 ────────────────────────────────────────────
MARKET_OPEN = "09:30"
MARKET_CLOSE = "15:59"
WINDOW_ACCUMULATION_END = "09:44"  # 09:30~09:44는 window 축적, score 없음
