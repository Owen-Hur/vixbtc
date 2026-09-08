"""
LSTM-AE 복원 오차 분해 분석

anomaly 발생 시 "어떤 피처가, 어느 방향으로" 이상한지 분석하고,
이 정보가 BTC 향후 방향을 예측하는지 검증.

핵심 가설:
  - vixy_return 실제 >> 복원 → VXX 예상 외 급등 → 공포 → BTC 하락
  - btc_return 실제 << 복원 → BTC 예상 외 급락 → 과매도 반등
  - 복원 오차의 "부호"가 방향 정보를 내포
"""
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from scipy import stats

from . import config
from .model import LSTMAutoencoder
from .dataset import load_parquets, compute_features, create_windows


FEATURES = config.FEATURES  # 7개


def load_model(device):
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
    return model


def compute_residuals(model, windows, device, batch_size=64):
    """
    피처별 signed residual 계산.

    Returns:
        signed_residuals: (n_windows, n_features) — 마지막 시점의 (actual - reconstructed)
        mse_per_feature: (n_windows, n_features) — 피처별 MSE
    """
    all_signed = []
    all_mse = []

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.FloatTensor(windows[i:i+batch_size]).to(device)
            reconstructed = model(batch)

            # 마지막 시점 signed residual (방향 정보)
            # actual - reconstructed: 양수면 실제가 더 큼
            signed = (batch[:, -1, :] - reconstructed[:, -1, :]).cpu().numpy()
            all_signed.append(signed)

            # 피처별 MSE (전체 window)
            mse = torch.mean((batch - reconstructed) ** 2, dim=1).cpu().numpy()
            all_mse.append(mse)

    return np.vstack(all_signed), np.vstack(all_mse)


def main():
    device = torch.device("cpu")
    model = load_model(device)
    scaler = joblib.load(config.ARTIFACT_DIR / "scaler.pkl")

    with open(config.ARTIFACT_DIR / "threshold.json") as f:
        th_data = json.load(f)
    threshold = th_data["threshold"]

    for period, start, end, label in [
        ("IS", config.IS_START, config.IS_END, "is"),
        ("OOS", config.OOS_START, config.OOS_END, "oos"),
    ]:
        print(f"\n{'='*70}")
        print(f"  {period} ({start} ~ {end})")
        print(f"{'='*70}")

        # 데이터 로딩
        df = load_parquets(start, end)
        df = compute_features(df)
        df[FEATURES] = scaler.transform(df[FEATURES].values)

        windows, timestamps = create_windows(df, FEATURES, config.WINDOW_SIZE)
        print(f"  Windows: {len(windows):,}")

        # 전체 anomaly score
        scores = []
        with torch.no_grad():
            for i in range(0, len(windows), 64):
                batch = torch.FloatTensor(windows[i:i+64]).to(device)
                recon = model(batch)
                s = torch.mean((batch - recon) ** 2, dim=(1, 2)).cpu().numpy()
                scores.extend(s)
        scores = np.array(scores)

        is_anomaly = scores > threshold

        # anomaly window만 잔차 분해
        anom_idx = np.where(is_anomaly)[0]
        anom_windows = windows[anom_idx]
        anom_timestamps = [timestamps[i] for i in anom_idx]
        print(f"  Anomaly windows: {len(anom_idx):,} ({len(anom_idx)/len(windows)*100:.1f}%)")

        if len(anom_idx) == 0:
            continue

        # 잔차 계산
        signed_res, mse_res = compute_residuals(model, anom_windows, device)

        # BTC 1분봉 로딩 (향후 수익률 계산용)
        btc_files = sorted(glob.glob(str(config.DATA_DIR / "btc_1m_*.parquet")))
        start_c = start.replace("-", "")
        end_c = end.replace("-", "")
        sel = [f for f in btc_files
               if start_c <= Path(f).stem.replace("btc_1m_", "").replace("-", "") <= end_c]
        btc = pd.concat([pd.read_parquet(f) for f in sel]).sort_index()
        btc["ret_1m"] = btc["close"].pct_change().fillna(0)

        # anomaly 시점의 향후 N분 수익률
        results = []
        for i, ts in enumerate(anom_timestamps):
            if ts not in btc.index:
                continue
            loc = btc.index.get_loc(ts)

            for fwd in [5, 15, 30, 60]:
                if loc + fwd >= len(btc):
                    continue
                fwd_slice = btc.iloc[loc:loc+fwd]
                if fwd_slice.iloc[-1]["trade_date"] != fwd_slice.iloc[0]["trade_date"]:
                    continue
                fwd_ret = float(fwd_slice["ret_1m"].sum())

                row = {"timestamp": ts, "fwd": fwd, "fwd_ret": fwd_ret}
                for j, feat in enumerate(FEATURES):
                    row[f"signed_{feat}"] = float(signed_res[i, j])
                    row[f"mse_{feat}"] = float(mse_res[i, j])
                results.append(row)

        res_df = pd.DataFrame(results)
        print(f"  분석 가능 샘플: {len(res_df):,}")

        # === 분석 1: 피처별 signed residual과 향후 수익의 상관 ===
        print(f"\n  === 피처별 signed residual → 향후 수익 상관 ===")
        for fwd in [5, 15, 30, 60]:
            sub = res_df[res_df["fwd"] == fwd]
            if len(sub) < 30:
                continue
            print(f"\n  [{fwd}분 후]")
            for feat in FEATURES:
                col = f"signed_{feat}"
                r, p = stats.pearsonr(sub[col], sub["fwd_ret"])
                sig = " ***" if p < 0.01 else (" **" if p < 0.05 else (" *" if p < 0.1 else ""))
                print(f"    {feat:<20s}: r={r:+.4f}, p={p:.4f}{sig}")

        # === 분석 2: vixy_return residual 방향별 BTC 수익 ===
        print(f"\n  === vixy_return 잔차 방향별 BTC 향후 수익 ===")
        for fwd in [15, 30, 60]:
            sub = res_df[res_df["fwd"] == fwd]
            if len(sub) < 30:
                continue

            # VXX가 예상보다 급등 (signed > 0) vs 급락 (signed < 0)
            vxx_up = sub[sub["signed_vixy_return"] > 0]
            vxx_down = sub[sub["signed_vixy_return"] < 0]

            print(f"  [{fwd}분 후]")
            print(f"    VXX 예상 외 상승 (n={len(vxx_up):,}): BTC 평균 {vxx_up['fwd_ret'].mean()*10000:+.2f}bp")
            print(f"    VXX 예상 외 하락 (n={len(vxx_down):,}): BTC 평균 {vxx_down['fwd_ret'].mean()*10000:+.2f}bp")

            if len(vxx_up) > 5 and len(vxx_down) > 5:
                t, p = stats.ttest_ind(vxx_up["fwd_ret"], vxx_down["fwd_ret"])
                print(f"    t={t:+.3f}, p={p:.4f}")

        # === 분석 3: btc_return residual 방향별 ===
        print(f"\n  === btc_return 잔차 방향별 BTC 향후 수익 ===")
        for fwd in [15, 30, 60]:
            sub = res_df[res_df["fwd"] == fwd]
            if len(sub) < 30:
                continue

            btc_over = sub[sub["signed_btc_return"] > 0]  # BTC가 예상보다 상승
            btc_under = sub[sub["signed_btc_return"] < 0]  # BTC가 예상보다 하락

            print(f"  [{fwd}분 후]")
            print(f"    BTC 예상 외 상승 (n={len(btc_over):,}): 이후 {btc_over['fwd_ret'].mean()*10000:+.2f}bp")
            print(f"    BTC 예상 외 하락 (n={len(btc_under):,}): 이후 {btc_under['fwd_ret'].mean()*10000:+.2f}bp")

            if len(btc_over) > 5 and len(btc_under) > 5:
                t, p = stats.ttest_ind(btc_over["fwd_ret"], btc_under["fwd_ret"])
                print(f"    t={t:+.3f}, p={p:.4f}")

        # === 분석 4: 복합 신호 — VXX↑ + BTC↓ = 공포 확산 ===
        print(f"\n  === 복합 신호: VXX↑ & BTC↓ (공포 확산) vs VXX↓ & BTC↑ (낙관) ===")
        for fwd in [15, 30, 60]:
            sub = res_df[res_df["fwd"] == fwd]
            if len(sub) < 30:
                continue

            fear = sub[(sub["signed_vixy_return"] > 0) & (sub["signed_btc_return"] < 0)]
            optimism = sub[(sub["signed_vixy_return"] < 0) & (sub["signed_btc_return"] > 0)]

            print(f"  [{fwd}분 후]")
            if len(fear) > 0:
                print(f"    공포 (VXX↑,BTC↓) n={len(fear):,}: BTC 이후 {fear['fwd_ret'].mean()*10000:+.2f}bp")
            if len(optimism) > 0:
                print(f"    낙관 (VXX↓,BTC↑) n={len(optimism):,}: BTC 이후 {optimism['fwd_ret'].mean()*10000:+.2f}bp")

            if len(fear) > 5 and len(optimism) > 5:
                t, p = stats.ttest_ind(fear["fwd_ret"], optimism["fwd_ret"])
                print(f"    t={t:+.3f}, p={p:.4f}")

        # === 분석 5: signed residual 기반 방향 예측 정확도 ===
        print(f"\n  === signed residual 기반 방향 예측 정확도 ===")
        for fwd in [15, 30, 60]:
            sub = res_df[res_df["fwd"] == fwd]
            if len(sub) < 30:
                continue

            # 가설: vixy_return 잔차 양수 → BTC 하락, 음수 → BTC 상승
            pred_dir = -np.sign(sub["signed_vixy_return"].values)  # VXX↑ → Short
            actual_dir = np.sign(sub["fwd_ret"].values)
            valid = actual_dir != 0
            if valid.sum() > 0:
                acc_vxx = (pred_dir[valid] == actual_dir[valid]).mean()
                print(f"  [{fwd}분] VXX 잔차 반대 방향: {acc_vxx*100:.1f}% (n={valid.sum()})")

            # 가설: btc_return 잔차 양수 → 모멘텀 (계속 상승)
            pred_dir2 = np.sign(sub["signed_btc_return"].values)  # 모멘텀
            if valid.sum() > 0:
                acc_btc = (pred_dir2[valid] == actual_dir[valid]).mean()
                print(f"  [{fwd}분] BTC 잔차 모멘텀:     {acc_btc*100:.1f}% (n={valid.sum()})")

            # 가설: btc_return 잔차 음수 → 반전 (과매도 반등)
            pred_dir3 = -np.sign(sub["signed_btc_return"].values)  # 반전
            if valid.sum() > 0:
                acc_btc_rev = (pred_dir3[valid] == actual_dir[valid]).mean()
                print(f"  [{fwd}분] BTC 잔차 반전:       {acc_btc_rev*100:.1f}% (n={valid.sum()})")


if __name__ == "__main__":
    main()
