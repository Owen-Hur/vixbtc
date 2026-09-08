"""
LSTM Autoencoder 학습 스크립트

실행:
    cd btc_project
    python -m lstm_ae.train

출력:
    artifacts/model.pt         — 학습된 모델 가중치
    artifacts/scaler.pkl       — StandardScaler (dataset.py에서 저장)
    artifacts/threshold.json   — 이상 탐지 임계값
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .dataset import prepare_data, TimeSeriesDataset
from .model import LSTMAutoencoder


def train_epoch(model, dataloader, criterion, optimizer, device):
    """1 epoch 학습, 평균 loss 반환"""
    model.train()
    total_loss = 0
    n_batches = 0

    for x, target in dataloader:
        x, target = x.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, dataloader, criterion, device):
    """검증 loss 계산"""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for x, target in dataloader:
            x, target = x.to(device), target.to(device)
            output = model(x)
            loss = criterion(output, target)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def compute_all_scores(model, windows, device):
    """
    모든 window의 anomaly score를 한 번에 계산.
    임계값(threshold) 결정에 사용합니다.
    """
    model.eval()
    scores = []

    dataset = TimeSeriesDataset(windows)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_hat = model(x)
            # window별 MSE (batch 내 각 샘플별로)
            batch_scores = torch.mean((x - x_hat) ** 2, dim=(1, 2))
            scores.extend(batch_scores.cpu().numpy())

    return np.array(scores)


def compute_threshold(scores):
    """
    anomaly score 분포에서 임계값 결정.

    상위 7.5% 지점 = 92.5 백분위수
    이 값을 초과하면 이상(anomaly)으로 판정합니다.
    """
    threshold = np.percentile(scores, config.THRESHOLD_PERCENTILE)

    print(f"\n  Score 분포:")
    print(f"    평균:    {scores.mean():.6f}")
    print(f"    표준편차: {scores.std():.6f}")
    print(f"    50%:     {np.percentile(scores, 50):.6f}")
    print(f"    90%:     {np.percentile(scores, 90):.6f}")
    print(f"    92.5%:   {threshold:.6f}  ← 임계값")
    print(f"    95%:     {np.percentile(scores, 95):.6f}")
    print(f"    99%:     {np.percentile(scores, 99):.6f}")
    print(f"    최대:    {scores.max():.6f}")

    return threshold


def train_model(model, train_loader, val_loader, device, model_name, save_path):
    """모델 학습 공통 로직. 메인/장초반 모델 모두 이 함수로 학습."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n모델 파라미터 수: {total_params:,}")

    print(f"\n{'='*50}")
    print(f"[{model_name}] 학습 시작 (max {config.EPOCHS} epochs, patience={config.EARLY_STOP_PATIENCE})")
    print(f"{'='*50}")

    best_val_loss = float("inf")
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        t_epoch = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t_epoch

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            marker = " ★ 저장"
        else:
            patience_counter += 1
            marker = f" (patience {patience_counter}/{config.EARLY_STOP_PATIENCE})"

        print(
            f"  Epoch {epoch:3d}/{config.EPOCHS} | "
            f"train_loss: {train_loss:.6f} | "
            f"val_loss: {val_loss:.6f} | "
            f"{elapsed:.1f}초{marker}"
        )

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping! (검증 loss가 {config.EARLY_STOP_PATIENCE} epoch 연속 미개선)")
            break

    total_time = time.time() - t_start
    print(f"\n학습 완료! 총 {total_time:.0f}초, 최고 val_loss: {best_val_loss:.6f}")

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. 데이터 준비 ──────────────────────────────
    train_dataset, val_dataset, scaler, is_windows_all = prepare_data()

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
    )

    # ── 2. 메인 모델 학습 (60분 window) ─────────────
    model = LSTMAutoencoder(
        n_features=config.N_FEATURES,
        encoder_hidden=config.ENCODER_HIDDEN,
        decoder_hidden=config.DECODER_HIDDEN,
        window_size=config.WINDOW_SIZE,
    ).to(device)

    model = train_model(
        model, train_loader, val_loader, device,
        "메인 60분", config.ARTIFACT_DIR / "model.pt",
    )

    # ── 3. 메인 임계값 계산 ──────────────────────────
    print(f"\n{'='*50}")
    print("메인 임계값 계산")
    print(f"{'='*50}")

    print("\n  IS 전체 window에 대해 anomaly score 계산 중...")
    all_scores = compute_all_scores(model, is_windows_all, device)
    threshold = compute_threshold(all_scores)

    threshold_data = {
        "threshold": float(threshold),
        "percentile": config.THRESHOLD_PERCENTILE,
        "n_windows": len(all_scores),
        "score_mean": float(all_scores.mean()),
        "score_std": float(all_scores.std()),
    }

    threshold_path = config.ARTIFACT_DIR / "threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"\n  임계값 저장: {threshold_path}")

    # ── 4. 장초반 모델 학습 (15분, 30분) ─────────────
    from .dataset import load_parquets, compute_features, create_early_windows

    print(f"\n{'='*50}")
    print("장초반 데이터 준비")
    print(f"{'='*50}")

    is_df = load_parquets(config.IS_START, config.IS_END)
    is_df = compute_features(is_df)
    is_df[config.FEATURES] = scaler.transform(is_df[config.FEATURES].values)

    for ws in config.EARLY_WINDOW_SIZES:
        print(f"\n{'='*50}")
        print(f"장초반 {ws}분 모델")
        print(f"{'='*50}")

        early_windows, _ = create_early_windows(is_df, config.FEATURES, ws)

        split_idx = int(len(early_windows) * config.TRAIN_RATIO)
        early_train = TimeSeriesDataset(early_windows[:split_idx])
        early_val = TimeSeriesDataset(early_windows[split_idx:])

        print(f"  학습: {len(early_train):,}개 | 검증: {len(early_val):,}개")

        early_train_loader = DataLoader(early_train, batch_size=config.BATCH_SIZE, shuffle=True)
        early_val_loader = DataLoader(early_val, batch_size=config.BATCH_SIZE, shuffle=False)

        early_model = LSTMAutoencoder(
            n_features=config.N_FEATURES,
            encoder_hidden=config.EARLY_ENCODER_HIDDEN,
            decoder_hidden=config.EARLY_DECODER_HIDDEN,
            window_size=ws,
        ).to(device)

        save_path = config.ARTIFACT_DIR / f"model_early_{ws}.pt"
        early_model = train_model(
            early_model, early_train_loader, early_val_loader, device,
            f"장초반 {ws}분", save_path,
        )

        # 장초반 임계값
        early_scores = compute_all_scores(early_model, early_windows, device)
        early_threshold = compute_threshold(early_scores)

        early_th_data = {
            "threshold": float(early_threshold),
            "percentile": config.THRESHOLD_PERCENTILE,
            "window_size": ws,
            "n_windows": len(early_scores),
            "score_mean": float(early_scores.mean()),
            "score_std": float(early_scores.std()),
        }

        early_th_path = config.ARTIFACT_DIR / f"threshold_early_{ws}.json"
        with open(early_th_path, "w") as f:
            json.dump(early_th_data, f, indent=2)
        print(f"\n  임계값 저장: {early_th_path}")

    # ── 5. 최종 요약 ──────────────────────────────
    print(f"\n{'='*50}")
    print("전체 학습 완료 — 저장된 파일:")
    print(f"{'='*50}")
    print(f"  메인 모델 (60분):    {config.ARTIFACT_DIR / 'model.pt'}")
    print(f"  메인 임계값:         {config.ARTIFACT_DIR / 'threshold.json'}")
    for ws in config.EARLY_WINDOW_SIZES:
        print(f"  장초반 모델 ({ws}분):  {config.ARTIFACT_DIR / f'model_early_{ws}.pt'}")
        print(f"  장초반 임계값 ({ws}분): {config.ARTIFACT_DIR / f'threshold_early_{ws}.json'}")
    print(f"  스케일러:            {config.ARTIFACT_DIR / 'scaler.pkl'}")


if __name__ == "__main__":
    main()
