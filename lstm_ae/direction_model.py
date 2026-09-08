"""
XGBoost 방향 분류기 — LSTM-AE anomaly 발생 시 향후 방향 예측

역할:
  LSTM-AE가 "지금 이상하다" (when)를 판정하면,
  이 모델이 "어느 방향으로 갈 것인가" (which way)를 예측한다.

입력 피처:
  - LSTM-AE anomaly score (강도)
  - BTC 미시구조 (trade_imbalance, 최근 수익률, 변동성)
  - VIXY 상태 (vixy_return, rolling_std, btc_corr)
  - VIX slope 정보 (당일 slope, 변화율)
  - 시간대 정보

타겟:
  향후 N분 수익률 방향 (1 = 상승, 0 = 하락)

실행:
    cd btc_project
    python3 -m lstm_ae.direction_model              # 학습 + 검증
    python3 -m lstm_ae.direction_model --predict     # signal 생성
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

from . import config

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── 설정 ─────────────────────────────────────────
FORWARD_MINUTES = 15      # 향후 15분 방향 예측
MIN_RETURN_THRESHOLD = 0  # 0% 기준 (상승/하락)
DIRECTION_FEATURES = [
    # LSTM-AE 출력
    "anomaly_score",
    # BTC 미시구조 (현재 시점)
    "trade_imbalance",
    "trade_count",
    "avg_trade_size",
    "btc_return",
    # BTC 최근 추세
    "btc_return_5m",      # 최근 5분 누적 수익률
    "btc_return_15m",     # 최근 15분 누적 수익률
    "btc_volatility_15m", # 최근 15분 변동성
    "btc_momentum",       # 최근 15분 방향성 (양/음)
    # VIXY 상태
    "vixy_return",
    "vixy_rolling_std",
    "vixy_btc_corr",
    # 시간대
    "minutes_from_open",  # 장 시작 후 경과 분
]


def load_btc_data(start_month, end_month):
    """BTC 1분봉 로딩"""
    import glob
    all_files = sorted(glob.glob(str(config.DATA_DIR / "btc_1m_*.parquet")))
    selected = []
    for f in all_files:
        month = f.split("btc_1m_")[1].replace(".parquet", "")
        if start_month <= month <= end_month:
            selected.append(f)
    dfs = [pd.read_parquet(f) for f in selected]
    df = pd.concat(dfs)
    df.sort_index(inplace=True)
    return df


def prepare_direction_features(btc_df, ae_signals, vix_df=None):
    """
    anomaly 발생 시점의 방향 예측 피처를 구성합니다.

    Args:
        btc_df: BTC 1분봉 DataFrame (trade_date, close, trade_imbalance 등)
        ae_signals: LSTM-AE anomaly signals DataFrame
        vix_df: VIX slope 일별 DataFrame (optional)

    Returns:
        features_df: 피처 DataFrame (anomaly 시점만)
        targets: 향후 N분 방향 (1=상승, 0=하락)
    """
    # BTC에 추가 피처 계산
    btc = btc_df.copy()
    btc["btc_return"] = btc["close"].pct_change()

    # 최근 N분 누적 수익률
    btc["btc_return_5m"] = btc["close"].pct_change(5)
    btc["btc_return_15m"] = btc["close"].pct_change(15)

    # 최근 15분 변동성
    btc["btc_volatility_15m"] = btc["btc_return"].rolling(15).std()

    # 모멘텀: 최근 15분 수익률의 부호 (-1 ~ +1)
    btc["btc_momentum"] = btc["btc_return"].rolling(15).mean()

    # 향후 N분 수익률 (타겟)
    btc["future_return"] = btc["close"].shift(-FORWARD_MINUTES) / btc["close"] - 1

    # 장 시작 후 경과 분
    from datetime import time as dtime
    btc["minutes_from_open"] = btc.index.map(
        lambda x: (x.hour * 60 + x.minute) - (9 * 60 + 30)
    )

    # VIXY 피처 계산
    vixy_files = sorted(
        Path(config.VIXY_DIR).glob("20*.parquet")
    )
    if vixy_files:
        start_compact = btc["trade_date"].min().strftime("%Y%m")
        end_compact = btc["trade_date"].max().strftime("%Y%m")
        vixy_selected = [
            f for f in vixy_files
            if start_compact <= f.stem <= end_compact
        ]
        if vixy_selected:
            vixy = pd.concat([pd.read_parquet(f) for f in vixy_selected])
            vixy.sort_index(inplace=True)
            vixy.index = vixy.index.tz_convert("America/New_York")
            vixy = vixy.rename(columns={"close": "vixy_close"})
            btc = btc.join(vixy[["vixy_close"]], how="left")
            btc["vixy_close"] = btc["vixy_close"].ffill()
            btc["vixy_close"] = btc.groupby("trade_date")["vixy_close"].transform(
                lambda x: x.bfill()
            )
            btc["vixy_return"] = btc["vixy_close"].pct_change().fillna(0)
            btc["vixy_rolling_std"] = btc.groupby("trade_date")["vixy_return"].transform(
                lambda x: x.rolling(20, min_periods=1).std()
            ).fillna(0)
            btc["vixy_btc_corr"] = btc.groupby("trade_date").apply(
                lambda g: g["vixy_return"].rolling(20, min_periods=5).corr(g["btc_return"])
            ).reset_index(level=0, drop=True).fillna(0)
        else:
            btc["vixy_return"] = 0.0
            btc["vixy_rolling_std"] = 0.0
            btc["vixy_btc_corr"] = 0.0
    else:
        btc["vixy_return"] = 0.0
        btc["vixy_rolling_std"] = 0.0
        btc["vixy_btc_corr"] = 0.0

    # anomaly signal 병합
    ae = ae_signals.copy()
    merged = btc.join(ae[["anomaly_score", "is_anomaly"]], how="inner")

    # anomaly 시점만 추출
    anomaly_rows = merged[merged["is_anomaly"] == True].copy()

    # 향후 수익률이 NaN인 행 제거 (장 마감 근처)
    anomaly_rows = anomaly_rows.dropna(subset=["future_return"])

    # 타겟: 향후 방향
    targets = (anomaly_rows["future_return"] > MIN_RETURN_THRESHOLD).astype(int)

    # 피처 선택
    available_features = [f for f in DIRECTION_FEATURES if f in anomaly_rows.columns]
    features = anomaly_rows[available_features].copy()

    # NaN 처리
    features = features.fillna(0)

    return features, targets, anomaly_rows.index


def train_direction_model():
    """IS 데이터로 학습, OOS로 검증"""
    print("=" * 60)
    print("XGBoost 방향 분류기 학습")
    print("=" * 60)

    # 1. 데이터 로딩
    print("\n[1/5] 데이터 로딩...")
    is_btc = load_btc_data(config.IS_START, config.IS_END)
    oos_btc = load_btc_data(config.OOS_START, config.OOS_END)
    print(f"  IS BTC: {len(is_btc):,}행")
    print(f"  OOS BTC: {len(oos_btc):,}행")

    ae_is = pd.read_parquet(config.ARTIFACT_DIR / "anomaly_signals_is.parquet")
    ae_oos = pd.read_parquet(config.ARTIFACT_DIR / "anomaly_signals_oos.parquet")
    print(f"  IS anomaly signals: {len(ae_is):,}")
    print(f"  OOS anomaly signals: {len(ae_oos):,}")

    # 2. 피처 구성
    print("\n[2/5] 피처 구성...")
    X_is, y_is, idx_is = prepare_direction_features(is_btc, ae_is)
    X_oos, y_oos, idx_oos = prepare_direction_features(oos_btc, ae_oos)

    print(f"  IS anomaly 샘플: {len(X_is):,} (Up: {y_is.sum():,}, Down: {(1-y_is).sum():,})")
    print(f"  OOS anomaly 샘플: {len(X_oos):,} (Up: {y_oos.sum():,}, Down: {(1-y_oos).sum():,})")
    print(f"  IS 방향 비율: Up {y_is.mean()*100:.1f}% / Down {(1-y_is.mean())*100:.1f}%")
    print(f"  OOS 방향 비율: Up {y_oos.mean()*100:.1f}% / Down {(1-y_oos.mean())*100:.1f}%")
    print(f"  사용 피처 ({len(X_is.columns)}개): {list(X_is.columns)}")

    # 3. 모델 학습
    print("\n[3/5] XGBoost 학습...")

    # IS 내에서 시간순 학습/검증 분할 (80/20)
    split_idx = int(len(X_is) * 0.8)
    X_train, X_val = X_is.iloc[:split_idx], X_is.iloc[split_idx:]
    y_train, y_val = y_is.iloc[:split_idx], y_is.iloc[split_idx:]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 4. 평가
    print("\n[4/5] 평가...")

    # IS 전체
    is_pred = model.predict(X_is)
    is_prob = model.predict_proba(X_is)[:, 1]
    is_acc = accuracy_score(y_is, is_pred)

    # IS 검증
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    # OOS
    oos_pred = model.predict(X_oos)
    oos_prob = model.predict_proba(X_oos)[:, 1]
    oos_acc = accuracy_score(y_oos, oos_pred)

    print(f"\n  정확도:")
    print(f"    IS 학습:   {accuracy_score(y_train, model.predict(X_train))*100:.1f}%")
    print(f"    IS 검증:   {val_acc*100:.1f}%")
    print(f"    IS 전체:   {is_acc*100:.1f}%")
    print(f"    OOS:       {oos_acc*100:.1f}%")

    # Confidence별 정확도
    print(f"\n  Confidence별 정확도 (OOS):")
    for conf_th in [0.50, 0.55, 0.60, 0.65, 0.70]:
        high_conf = np.maximum(oos_prob, 1 - oos_prob) >= conf_th
        if high_conf.sum() > 0:
            acc = accuracy_score(y_oos[high_conf], oos_pred[high_conf])
            pct = high_conf.mean() * 100
            print(f"    conf >= {conf_th:.0%}: {acc*100:.1f}% ({high_conf.sum():,}건, 전체의 {pct:.1f}%)")

    # 피처 중요도
    print(f"\n  피처 중요도 (Top 10):")
    importances = pd.Series(
        model.feature_importances_, index=X_is.columns
    ).sort_values(ascending=False)
    for feat, imp in importances.head(10).items():
        print(f"    {feat:25s}: {imp:.4f}")

    # 5. 저장
    print("\n[5/5] 저장...")
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = config.ARTIFACT_DIR / "direction_model.json"
    model.save_model(str(model_path))
    print(f"  모델: {model_path}")

    meta = {
        "forward_minutes": FORWARD_MINUTES,
        "features": list(X_is.columns),
        "is_accuracy": float(is_acc),
        "is_val_accuracy": float(val_acc),
        "oos_accuracy": float(oos_acc),
        "is_samples": len(X_is),
        "oos_samples": len(X_oos),
        "is_up_ratio": float(y_is.mean()),
        "oos_up_ratio": float(y_oos.mean()),
    }
    meta_path = config.ARTIFACT_DIR / "direction_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  메타: {meta_path}")

    # 6. 요약
    print(f"\n{'='*60}")
    print("학습 완료!")
    print(f"{'='*60}")
    print(f"  모델: XGBoost (depth=4, n_est=300)")
    print(f"  피처: {len(X_is.columns)}개")
    print(f"  IS 정확도: {is_acc*100:.1f}% (검증 {val_acc*100:.1f}%)")
    print(f"  OOS 정확도: {oos_acc*100:.1f}%")

    if oos_acc > 0.55:
        print(f"\n  ★ OOS 55% 이상 — 방향 예측에 의미 있음!")
    elif oos_acc > 0.52:
        print(f"\n  △ OOS 52~55% — 약한 신호, 추가 검증 필요")
    else:
        print(f"\n  ✗ OOS 52% 미만 — 방향 예측 불가, 레버리지 조절로 전환 권장")

    return model, meta


def generate_direction_signals():
    """학습된 모델로 IS/OOS 방향 signal 생성"""
    print("=" * 60)
    print("방향 Signal 생성")
    print("=" * 60)

    model = xgb.XGBClassifier()
    model.load_model(str(config.ARTIFACT_DIR / "direction_model.json"))

    with open(config.ARTIFACT_DIR / "direction_meta.json") as f:
        meta = json.load(f)

    for period, start, end, name in [
        ("IS", config.IS_START, config.IS_END, "is"),
        ("OOS", config.OOS_START, config.OOS_END, "oos"),
    ]:
        print(f"\n--- {period} ({start} ~ {end}) ---")
        btc = load_btc_data(start, end)
        ae = pd.read_parquet(config.ARTIFACT_DIR / f"anomaly_signals_{name}.parquet")

        X, y, idx = prepare_direction_features(btc, ae)

        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

        # 기존 anomaly signal에 방향 정보 추가
        ae_out = ae.copy()
        ae_out["direction_prob"] = np.nan   # Up 확률
        ae_out["predicted_direction"] = ""  # Up/Down
        ae_out["direction_confidence"] = np.nan  # 확신도

        ae_out.loc[idx, "direction_prob"] = probs
        ae_out.loc[idx, "predicted_direction"] = np.where(preds == 1, "Up", "Down")
        ae_out.loc[idx, "direction_confidence"] = np.maximum(probs, 1 - probs)

        out_path = config.ARTIFACT_DIR / f"anomaly_signals_{name}.parquet"
        ae_out.to_parquet(out_path)
        print(f"  저장: {out_path}")
        print(f"  방향 예측 추가: {len(idx):,}건")

    print(f"\n완료!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true",
                        help="학습된 모델로 signal 생성")
    args = parser.parse_args()

    if args.predict:
        generate_direction_signals()
    else:
        train_direction_model()


if __name__ == "__main__":
    main()
