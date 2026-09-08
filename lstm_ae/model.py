"""
LSTM Autoencoder 모델 정의

구조:
  인코더: LSTM(64) → LSTM(32) → 잠재 벡터 (32차원)
  디코더: RepeatVector → LSTM(32) → LSTM(64) → Dense(n_features)

입력/출력: (batch, 60, n_features)
손실함수: MSE (입력과 출력의 차이)
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM 기반 Autoencoder.

    Autoencoder의 핵심 아이디어:
      - 인코더: 60분 × n_features 데이터를 32차원으로 압축
      - 디코더: 32차원에서 다시 60분 × n_features로 복원
      - 정상 패턴은 잘 복원됨 (error 낮음)
      - 이상 패턴은 복원 못 함 (error 높음)
      - error = anomaly score

    Args:
        n_features: 입력 피처 수 (현재 4, VIXY 합류 시 7)
        encoder_hidden: 인코더 LSTM 레이어 크기 리스트 [64, 32]
        decoder_hidden: 디코더 LSTM 레이어 크기 리스트 [32, 64]
        window_size: 시퀀스 길이 (60)
    """

    def __init__(self, n_features, encoder_hidden, decoder_hidden, window_size):
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = encoder_hidden[-1]  # 32

        # ── 인코더 ────────────────────────────────
        # LSTM(64): 입력 (batch, 60, n_features) → 출력 (batch, 60, 64)
        self.encoder_lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=encoder_hidden[0],  # 64
            batch_first=True,
        )

        # LSTM(32): 입력 (batch, 60, 64) → 출력 (batch, 60, 32)
        # 마지막 시점만 사용 → (batch, 32) = 잠재 벡터
        self.encoder_lstm2 = nn.LSTM(
            input_size=encoder_hidden[0],   # 64
            hidden_size=encoder_hidden[1],  # 32
            batch_first=True,
        )

        # ── 디코더 ────────────────────────────────
        # 잠재 벡터를 60번 복제한 뒤 LSTM으로 복원
        # LSTM(32): 입력 (batch, 60, 32) → 출력 (batch, 60, 32)
        self.decoder_lstm1 = nn.LSTM(
            input_size=decoder_hidden[0],   # 32
            hidden_size=decoder_hidden[0],  # 32
            batch_first=True,
        )

        # LSTM(64): 입력 (batch, 60, 32) → 출력 (batch, 60, 64)
        self.decoder_lstm2 = nn.LSTM(
            input_size=decoder_hidden[0],   # 32
            hidden_size=decoder_hidden[1],  # 64
            batch_first=True,
        )

        # Dense: 각 시점마다 64차원 → n_features 차원으로 변환
        self.output_layer = nn.Linear(decoder_hidden[1], n_features)

    def encode(self, x):
        """
        입력을 잠재 벡터로 압축

        Args:
            x: (batch, 60, n_features)
        Returns:
            latent: (batch, 32) — 60분 데이터의 압축 표현
        """
        # 첫 번째 LSTM: 모든 시점의 출력 사용
        out, _ = self.encoder_lstm1(x)          # (batch, 60, 64)

        # 두 번째 LSTM: 마지막 시점의 출력만 사용 (압축)
        out, _ = self.encoder_lstm2(out)         # (batch, 60, 32)
        latent = out[:, -1, :]                   # (batch, 32) ← 마지막 시점만

        return latent

    def decode(self, latent):
        """
        잠재 벡터에서 원본 복원

        Args:
            latent: (batch, 32)
        Returns:
            reconstructed: (batch, 60, n_features)
        """
        # 32차원 벡터를 60번 복제 → (batch, 60, 32)
        repeated = latent.unsqueeze(1).repeat(1, self.window_size, 1)

        # 디코더 LSTM 통과
        out, _ = self.decoder_lstm1(repeated)    # (batch, 60, 32)
        out, _ = self.decoder_lstm2(out)         # (batch, 60, 64)

        # 각 시점마다 n_features로 변환
        reconstructed = self.output_layer(out)   # (batch, 60, n_features)

        return reconstructed

    def forward(self, x):
        """
        전체 순전파: 인코딩 → 디코딩

        Args:
            x: (batch, 60, n_features)
        Returns:
            reconstructed: (batch, 60, n_features)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed


def compute_anomaly_score(model, x, device="cpu"):
    """
    단일 window의 anomaly score (reconstruction error) 계산.

    Args:
        model: 학습된 LSTMAutoencoder
        x: (1, 60, n_features) 또는 (60, n_features)
        device: "cpu" 또는 "cuda"

    Returns:
        score: float (MSE 값, 높을수록 이상)
    """
    model.eval()
    if x.dim() == 2:
        x = x.unsqueeze(0)  # (60, n_features) → (1, 60, n_features)

    x = x.to(device)

    with torch.no_grad():
        x_hat = model(x)
        # 각 피처, 각 시점의 오차 제곱 평균 = MSE
        score = torch.mean((x - x_hat) ** 2).item()

    return score
