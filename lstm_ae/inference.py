"""
LSTM Autoencoder 추론 — anomaly signal 생성

학습된 모델로 IS 또는 OOS 구간의 anomaly score를 계산하고,
Trading 팀원이 사용할 anomaly_signals.parquet을 생성합니다.

장초반(09:45~10:28)은 15분/30분 모델, 이후(10:29~15:59)는 60분 메인 모델 사용.

실행:
    cd btc_project
    python -m lstm_ae.inference              # OOS 구간 (기본)
    python -m lstm_ae.inference --is         # IS 구간
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import joblib

from . import config
from .dataset import load_parquets, compute_features, create_windows, create_early_windows
from .model import LSTMAutoencoder


def load_main_model(device):
    """메인 60분 모델 로드"""
    model = LSTMAutoencoder(
        n_features=config.N_FEATURES,
        encoder_hidden=config.ENCODER_HIDDEN,
        decoder_hidden=config.DECODER_HIDDEN,
        window_size=config.WINDOW_SIZE,
    ).to(device)
    model.load_state_dict(
        torch.load(config.ARTIFACT_DIR / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()

    with open(config.ARTIFACT_DIR / "threshold.json") as f:
        threshold_data = json.load(f)

    print(f"  메인 모델 로드 (60분, threshold={threshold_data['threshold']:.6f})")
    return model, threshold_data


def load_early_model(device, window_size):
    """장초반 모델 로드"""
    model_path = config.ARTIFACT_DIR / f"model_early_{window_size}.pt"
    th_path = config.ARTIFACT_DIR / f"threshold_early_{window_size}.json"

    if not model_path.exists():
        print(f"  장초반 {window_size}분 모델 없음 — 건너뜀")
        return None, None

    model = LSTMAutoencoder(
        n_features=config.N_FEATURES,
        encoder_hidden=config.EARLY_ENCODER_HIDDEN,
        decoder_hidden=config.EARLY_DECODER_HIDDEN,
        window_size=window_size,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    with open(th_path) as f:
        threshold_data = json.load(f)

    print(f"  장초반 모델 로드 ({window_size}분, threshold={threshold_data['threshold']:.6f})")
    return model, threshold_data


def compute_scores_batch(model, windows, device):
    """배치 단위로 anomaly score 계산"""
    scores = []
    with torch.no_grad():
        for i in range(0, len(windows), config.BATCH_SIZE):
            batch = torch.FloatTensor(windows[i:i + config.BATCH_SIZE]).to(device)
            reconstructed = model(batch)
            batch_scores = torch.mean((batch - reconstructed) ** 2, dim=(1, 2))
            scores.extend(batch_scores.cpu().numpy())
    return np.array(scores)


def generate_signals(start_month, end_month, output_name):
    """
    메인 + 장초반 모델을 사용해 통합 anomaly signal 생성.

    출력 시간 범위:
      09:44 (15분 window 첫 출력) ~ 15:59
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*50}")
    print(f"Anomaly Signal 생성 ({output_name})")
    print(f"구간: {start_month} ~ {end_month}")
    print(f"{'='*50}")

    # 1. 모델 로드
    print("\n[1/4] 모델 로드...")
    main_model, main_th = load_main_model(device)
    scaler = joblib.load(config.ARTIFACT_DIR / "scaler.pkl")

    early_models = {}
    for ws in config.EARLY_WINDOW_SIZES:
        m, th = load_early_model(device, ws)
        if m is not None:
            early_models[ws] = (m, th)

    # 2. 데이터 로딩 + 피처 계산
    print("\n[2/4] 데이터 로딩...")
    df = load_parquets(start_month, end_month)
    df = compute_features(df)
    df[config.FEATURES] = scaler.transform(df[config.FEATURES].values)

    # 3. Score 계산
    print("\n[3/4] Anomaly score 계산...")

    all_signals = []

    # 3a. 장초반 모델
    for ws in sorted(early_models.keys()):
        model, th_data = early_models[ws]
        windows, timestamps = create_early_windows(df, config.FEATURES, ws)
        if len(windows) == 0:
            continue
        scores = compute_scores_batch(model, windows, device)
        threshold = th_data["threshold"]

        early_df = pd.DataFrame({
            "timestamp": timestamps,
            "anomaly_score": scores,
            "is_anomaly": scores > threshold,
            "model": f"early_{ws}",
        })
        all_signals.append(early_df)

    # 3b. 메인 모델
    windows, timestamps = create_windows(df, config.FEATURES, config.WINDOW_SIZE)
    scores = compute_scores_batch(main_model, windows, device)
    threshold = main_th["threshold"]

    main_df = pd.DataFrame({
        "timestamp": timestamps,
        "anomaly_score": scores,
        "is_anomaly": scores > threshold,
        "model": "main_60",
    })
    all_signals.append(main_df)

    # 4. 통합 — 같은 timestamp에 여러 모델 출력이 있으면 작은 window 우선
    # (장초반에는 early만, 10:29 이후에는 main만 존재하므로 중복 없음)
    print("\n[4/4] Signal 통합 및 저장...")
    signals = pd.concat(all_signals, ignore_index=True)
    signals = signals.sort_values("timestamp")
    signals = signals.drop_duplicates(subset="timestamp", keep="first")

    signals["trade_date"] = pd.to_datetime(signals["timestamp"]).dt.date
    signals.set_index("timestamp", inplace=True)

    # 저장
    out_path = config.ARTIFACT_DIR / f"anomaly_signals_{output_name}.parquet"
    signals.to_parquet(out_path)

    # 요약
    n_total = len(signals)
    n_anomaly = signals["is_anomaly"].sum()
    pct = n_anomaly / n_total * 100

    n_early = (signals["model"] != "main_60").sum()
    n_main = (signals["model"] == "main_60").sum()

    print(f"\n{'='*50}")
    print(f"완료!")
    print(f"  총 window: {n_total:,}")
    print(f"    장초반: {n_early:,} (15분+30분 모델)")
    print(f"    메인:   {n_main:,} (60분 모델)")
    print(f"  이상 탐지: {n_anomaly:,} ({pct:.1f}%)")
    print(f"  저장: {out_path}")
    print(f"{'='*50}")

    return signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is", dest="run_is", action="store_true",
                        help="IS 구간 signal 생성")
    args = parser.parse_args()

    if args.run_is:
        generate_signals(config.IS_START, config.IS_END, "is")
    else:
        generate_signals(config.OOS_START, config.OOS_END, "oos")


if __name__ == "__main__":
    main()
