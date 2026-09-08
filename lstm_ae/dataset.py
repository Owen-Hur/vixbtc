"""
데이터 로딩 + 슬라이딩 윈도우 생성

역할:
  1. 월별 parquet 로딩 → 전체 이어붙이기
  2. 피처 계산 (btc_return 등)
  3. IS/OOS 분할
  4. StandardScaler 정규화 (IS에서 fit, 고정)
  5. 일별로 끊어서 60분 window 생성
  6. PyTorch Dataset으로 변환
"""

import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib

from . import config


def load_parquets(start_month, end_month):
    """
    지정 기간의 BTC + VIXY 월별 parquet을 이어붙여 반환합니다.

    Args:
        start_month: "2024-01" 형태
        end_month: "2025-04" 형태

    Returns:
        pd.DataFrame (1분봉, ET timezone index, VIXY 컬럼 포함)
    """
    # ── BTC 로딩 ──
    all_files = sorted(glob.glob(str(config.DATA_DIR / "btc_1m_*.parquet")))
    selected = []
    for f in all_files:
        month = f.split("btc_1m_")[1].replace(".parquet", "")
        if start_month <= month <= end_month:
            selected.append(f)

    if not selected:
        raise FileNotFoundError(f"{start_month}~{end_month} 범위의 BTC parquet 파일이 없습니다.")

    dfs = [pd.read_parquet(f) for f in selected]
    df = pd.concat(dfs)
    df.sort_index(inplace=True)
    print(f"  BTC 로딩: {len(selected)}개 파일, {len(df):,}행")

    # ── VIXY 로딩 + 병합 ──
    vixy_files = sorted(glob.glob(str(config.VIXY_DIR / "20*.parquet")))
    # start_month="2024-01" → "202401" 형태로 변환
    start_compact = start_month.replace("-", "")
    end_compact = end_month.replace("-", "")
    vixy_selected = []
    for f in vixy_files:
        fname = f.split("/")[-1].replace(".parquet", "")  # "202401"
        if start_compact <= fname <= end_compact:
            vixy_selected.append(f)

    if vixy_selected:
        vixy_dfs = [pd.read_parquet(f) for f in vixy_selected]
        vixy = pd.concat(vixy_dfs)
        vixy.sort_index(inplace=True)

        # UTC → ET 변환 후 tz 통일
        vixy.index = vixy.index.tz_convert("America/New_York")

        # VIXY 컬럼 이름 충돌 방지 (close → vixy_close 등)
        vixy = vixy.rename(columns={
            "close": "vixy_close",
            "open": "vixy_open",
            "high": "vixy_high",
            "low": "vixy_low",
            "volume": "vixy_volume",
            "trade_count": "vixy_trade_count",
            "vwap": "vixy_vwap",
        })

        # BTC index에 맞춰 병합 (left join) + forward fill
        df = df.join(vixy[["vixy_close"]], how="left")
        df["vixy_close"] = df["vixy_close"].ffill()

        # 장 시작 첫 행에 NaN 남을 수 있음 → 해당 날의 첫 유효값으로 bfill
        df["vixy_close"] = df.groupby("trade_date")["vixy_close"].transform(
            lambda x: x.bfill()
        )

        print(f"  VIXY 로딩: {len(vixy_selected)}개 파일, 병합 완료 (ffill 적용)")
    else:
        print(f"  VIXY 파일 없음 — BTC only 모드")
        df["vixy_close"] = np.nan

    return df


def compute_features(df):
    """
    raw 1분봉에서 모델 입력 피처를 계산합니다.

    BTC 피처 (4개):
      - btc_return: close의 1분 수익률
      - trade_imbalance: 이미 존재
      - trade_count: 이미 존재
      - avg_trade_size: 이미 존재

    VIXY 피처 (3개):
      - vixy_return: VIXY 1분 수익률
      - vixy_rolling_std: VIXY 20분 rolling 표준편차
      - vixy_btc_corr: VIXY-BTC 20분 rolling 상관계수
    """
    df = df.copy()

    # ── BTC 피처 ──
    df["btc_return"] = df["close"].pct_change()
    df["btc_return"] = df.groupby("trade_date")["btc_return"].transform(
        lambda x: x.fillna(0)
    )

    # ── VIXY 피처 ──
    if "vixy_close" in df.columns and df["vixy_close"].notna().any():
        # vixy_return: 1분 수익률
        df["vixy_return"] = df["vixy_close"].pct_change()
        df["vixy_return"] = df.groupby("trade_date")["vixy_return"].transform(
            lambda x: x.fillna(0)
        )

        # vixy_rolling_std: 20분 rolling 표준편차
        df["vixy_rolling_std"] = df.groupby("trade_date")["vixy_return"].transform(
            lambda x: x.rolling(20, min_periods=1).std()
        )
        df["vixy_rolling_std"] = df["vixy_rolling_std"].fillna(0)

        # vixy_btc_corr: 20분 rolling 상관계수
        df["vixy_btc_corr"] = df.groupby("trade_date").apply(
            lambda g: g["vixy_return"].rolling(20, min_periods=5).corr(g["btc_return"])
        ).reset_index(level=0, drop=True)
        df["vixy_btc_corr"] = df["vixy_btc_corr"].fillna(0)

        print("  VIXY 피처 계산 완료: vixy_return, vixy_rolling_std, vixy_btc_corr")
    else:
        print("  VIXY 데이터 없음 — VIXY 피처 0으로 채움")
        df["vixy_return"] = 0.0
        df["vixy_rolling_std"] = 0.0
        df["vixy_btc_corr"] = 0.0

    return df


def create_windows(df, features, window_size):
    """
    일별로 끊어서 슬라이딩 윈도우를 생성합니다.

    왜 일별로 끊는가:
      전날 15:59 → 오늘 09:30 사이 16시간+ 공백이 있으므로
      날짜를 넘어가는 window는 만들면 안 됩니다.

    Args:
        df: 피처 계산 완료된 DataFrame
        features: 사용할 피처 컬럼명 리스트
        window_size: window 크기 (분)

    Returns:
        windows: np.array (n_windows, window_size, n_features)
        timestamps: 각 window의 마지막 시점 리스트 (추론 시 사용)
    """
    windows = []
    timestamps = []

    for date, day_df in df.groupby("trade_date"):
        day_values = day_df[features].values  # (390, n_features)

        if len(day_values) < window_size:
            continue

        for i in range(len(day_values) - window_size + 1):
            window = day_values[i : i + window_size]  # (60, n_features)
            windows.append(window)
            timestamps.append(day_df.index[i + window_size - 1])

    windows = np.array(windows, dtype=np.float32)
    print(f"  윈도우 생성: {len(windows):,}개 (window_size={window_size})")
    return windows, timestamps


def create_early_windows(df, features, window_size):
    """
    장초반 전용 윈도우 생성.

    09:30부터 시작하는 데이터에서 window_size(15 or 30)분 윈도우를 만들되,
    메인 60분 윈도우가 생성되기 전 구간만 대상으로 합니다.

    window_size=15: 장 시작 15분째(09:44)부터 60분째 직전(10:28)까지
    window_size=30: 장 시작 30분째(09:59)부터 60분째 직전(10:28)까지
    """
    windows = []
    timestamps = []
    main_ws = config.WINDOW_SIZE  # 60

    for date, day_df in df.groupby("trade_date"):
        day_values = day_df[features].values

        if len(day_values) < main_ws:
            continue

        # window_size분째부터 시작, 60분째 직전까지
        for i in range(0, main_ws - window_size):
            end_idx = i + window_size
            if end_idx > len(day_values):
                break
            window = day_values[i:end_idx]
            windows.append(window)
            timestamps.append(day_df.index[end_idx - 1])

    windows = np.array(windows, dtype=np.float32)
    print(f"  장초반 윈도우 생성: {len(windows):,}개 (window_size={window_size})")
    return windows, timestamps


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset — Autoencoder이므로 입력=타겟 (자기 자신을 복원)
    """

    def __init__(self, windows):
        self.windows = torch.FloatTensor(windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        return x, x  # 입력과 타겟이 동일


def prepare_data():
    """
    전체 데이터 준비 파이프라인.

    Returns:
        train_dataset: 학습용 PyTorch Dataset
        val_dataset: 검증용 PyTorch Dataset
        scaler: fit된 StandardScaler (저장해서 추론 시 재사용)
        is_windows_all: IS 전체 window (임계값 계산 시 사용)
    """
    print("=" * 50)
    print("데이터 준비 시작")
    print("=" * 50)

    # 1. IS 구간 parquet 로딩
    print("\n[1/5] IS 데이터 로딩...")
    is_df = load_parquets(config.IS_START, config.IS_END)

    # 2. 피처 계산
    print("\n[2/5] 피처 계산...")
    is_df = compute_features(is_df)

    # 3. 정규화 (IS 전체에서 fit)
    print("\n[3/5] 정규화 (StandardScaler)...")
    scaler = StandardScaler()
    scaler.fit(is_df[config.FEATURES].values)

    # scaler 저장 (추론 시 동일 scaler 재사용)
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = config.ARTIFACT_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  scaler 저장: {scaler_path}")

    # 정규화 적용
    is_df[config.FEATURES] = scaler.transform(is_df[config.FEATURES].values)

    # 4. 윈도우 생성
    print("\n[4/5] 슬라이딩 윈도우 생성...")
    is_windows, is_timestamps = create_windows(
        is_df, config.FEATURES, config.WINDOW_SIZE
    )

    # 5. 학습/검증 분할 (시간순 — 앞 80% 학습, 뒤 20% 검증)
    print("\n[5/5] 학습/검증 분할...")
    split_idx = int(len(is_windows) * config.TRAIN_RATIO)
    train_windows = is_windows[:split_idx]
    val_windows = is_windows[split_idx:]

    print(f"  학습: {len(train_windows):,}개 윈도우")
    print(f"  검증: {len(val_windows):,}개 윈도우")

    train_dataset = TimeSeriesDataset(train_windows)
    val_dataset = TimeSeriesDataset(val_windows)

    print("\n데이터 준비 완료!")
    return train_dataset, val_dataset, scaler, is_windows
