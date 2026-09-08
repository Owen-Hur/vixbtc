"""장중 최적 청산 + GARCH 분석"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from . import config


def load_btc(start, end):
    btc_files = sorted(glob.glob(str(config.DATA_DIR / "btc_1m_*.parquet")))
    sel = [f for f in btc_files
           if start <= Path(f).stem.replace("btc_1m_", "").replace("-", "") <= end]
    return pd.concat([pd.read_parquet(f) for f in sel]).sort_index()


def get_directions(vix_df):
    vix_df = vix_df.copy()
    vix_df["slope_change"] = vix_df["slope"].diff()
    dirs = {}
    for date, row in vix_df.iterrows():
        sc = row["slope_change"]
        if pd.isna(sc) or abs(sc) < 0.5:
            dirs[date.strftime("%Y-%m-%d")] = (0, 0, row["vix"])
        elif sc > 0:
            dirs[date.strftime("%Y-%m-%d")] = (1, abs(sc), row["vix"])
        else:
            dirs[date.strftime("%Y-%m-%d")] = (-1, abs(sc), row["vix"])
    return dirs


COST = 2 * (0.0005 + 0.0005) * 2  # 왕복 비용 × 레버리지


def analyze_exit(btc_df, dirs):
    btc = btc_df.copy()
    btc["ret_1m"] = btc["close"].pct_change().fillna(0)
    btc["direction"] = btc["trade_date"].astype(str).map(
        lambda x: dirs.get(x, (0, 0, 0))[0]
    )
    trading = btc[btc["direction"] != 0].copy()
    trading["signed_ret"] = trading["direction"] * trading["ret_1m"] * 2

    # 일별 누적 수익 곡선
    max_rets = []
    final_rets = []
    max_hours = []
    for td in trading["trade_date"].unique():
        day = trading[trading["trade_date"] == td]
        cum = day["signed_ret"].cumsum().values
        max_ret = float(cum.max())
        final_ret = float(cum[-1])
        max_idx = int(cum.argmax())
        max_hour = day.index[max_idx].hour
        max_rets.append(max_ret)
        final_rets.append(final_ret)
        max_hours.append(max_hour)

    max_rets = np.array(max_rets)
    final_rets = np.array(final_rets)

    print(f"\n  거래일: {len(final_rets)}일")
    print(f"  평균 최고점: {max_rets.mean()*100:+.2f}%")
    print(f"  평균 마감:   {final_rets.mean()*100:+.2f}%")
    print(f"  놓친 수익: {(max_rets-final_rets).mean()*100:.2f}%")
    print(f"  최고점 합: {max_rets.sum()*100:+.1f}% vs 마감 합: {final_rets.sum()*100:+.1f}%")

    print(f"\n  최고점 시간대:")
    for h in range(9, 16):
        cnt = sum(1 for x in max_hours if x == h)
        print(f"    {h:02d}시: {cnt}일 ({cnt/len(max_hours)*100:.0f}%)")

    return trading


def test_exit_rules(trading):
    trade_dates = trading["trade_date"].unique()

    # Baseline
    bl = []
    for td in trade_dates:
        day = trading[trading["trade_date"] == td]
        cum = day["signed_ret"].cumsum().values
        bl.append(float(cum[-1]) - COST)
    bl = np.array(bl)
    bl_sr = bl.mean() / bl.std() * np.sqrt(252) if bl.std() > 0 else 0
    bl_mdd = float((np.cumsum(bl) - np.maximum.accumulate(np.cumsum(bl))).min())
    print(f"\n  Baseline (마감 청산): Net {bl.sum()*100:+.1f}%, SR {bl_sr:+.2f}, "
          f"HR {(bl>0).mean()*100:.0f}%, MDD {bl_mdd*100:+.1f}%")

    # --- Target Profit ---
    print(f"\n  === Target Profit ===")
    for target in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04]:
        rets = []
        for td in trade_dates:
            day = trading[trading["trade_date"] == td]
            cum = day["signed_ret"].cumsum().values
            hit = np.where(cum >= target)[0]
            if len(hit) > 0:
                rets.append(target - COST)
            else:
                rets.append(float(cum[-1]) - COST)
        rets = np.array(rets)
        hit_n = int((np.abs(rets - (target - COST)) < 1e-9).sum())
        sr = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        alpha = rets.sum() - bl.sum()
        print(f"    TP {target*100:4.1f}%: Net {rets.sum()*100:+7.1f}%, "
              f"Alpha {alpha*100:+5.1f}%p, SR {sr:+.2f}, "
              f"hit {hit_n}/{len(rets)} ({hit_n/len(rets)*100:.0f}%)")

    # --- Trailing Stop ---
    print(f"\n  === Trailing Stop ===")
    for trail in [0.005, 0.008, 0.01, 0.012, 0.015, 0.02]:
        rets = []
        for td in trade_dates:
            day = trading[trading["trade_date"] == td]
            cum = day["signed_ret"].cumsum().values
            peak = cum[0]
            exit_val = float(cum[-1])
            for i in range(len(cum)):
                if cum[i] > peak:
                    peak = cum[i]
                if peak > trail and cum[i] < peak - trail:
                    exit_val = float(cum[i])
                    break
            rets.append(exit_val - COST)
        rets = np.array(rets)
        sr = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        mdd = float((np.cumsum(rets) - np.maximum.accumulate(np.cumsum(rets))).min())
        alpha = rets.sum() - bl.sum()
        print(f"    TS {trail*100:4.1f}%: Net {rets.sum()*100:+7.1f}%, "
              f"Alpha {alpha*100:+5.1f}%p, SR {sr:+.2f}, "
              f"HR {(rets>0).mean()*100:.0f}%, MDD {mdd*100:+.1f}%")

    # --- Stop Loss ---
    print(f"\n  === Stop Loss ===")
    for sl in [0.01, 0.015, 0.02, 0.025, 0.03]:
        rets = []
        for td in trade_dates:
            day = trading[trading["trade_date"] == td]
            cum = day["signed_ret"].cumsum().values
            exit_val = float(cum[-1])
            for i in range(len(cum)):
                if cum[i] < -sl:
                    exit_val = float(cum[i])
                    break
            rets.append(exit_val - COST)
        rets = np.array(rets)
        sr = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        mdd = float((np.cumsum(rets) - np.maximum.accumulate(np.cumsum(rets))).min())
        alpha = rets.sum() - bl.sum()
        print(f"    SL {sl*100:4.1f}%: Net {rets.sum()*100:+7.1f}%, "
              f"Alpha {alpha*100:+5.1f}%p, SR {sr:+.2f}, "
              f"HR {(rets>0).mean()*100:.0f}%, MDD {mdd*100:+.1f}%")

    # --- Trailing + Stop Loss ---
    print(f"\n  === Trailing Stop + Stop Loss 조합 ===")
    best_alpha = -999
    best_combo = ""
    for trail in [0.008, 0.01, 0.012, 0.015]:
        for sl in [0.015, 0.02, 0.025, 0.03]:
            rets = []
            for td in trade_dates:
                day = trading[trading["trade_date"] == td]
                cum = day["signed_ret"].cumsum().values
                peak = cum[0]
                exit_val = float(cum[-1])
                for i in range(len(cum)):
                    if cum[i] > peak:
                        peak = cum[i]
                    if peak > trail and cum[i] < peak - trail:
                        exit_val = float(cum[i])
                        break
                    if cum[i] < -sl:
                        exit_val = float(cum[i])
                        break
                rets.append(exit_val - COST)
            rets = np.array(rets)
            sr = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
            mdd = float((np.cumsum(rets) - np.maximum.accumulate(np.cumsum(rets))).min())
            alpha = rets.sum() - bl.sum()
            if alpha > best_alpha:
                best_alpha = alpha
                best_combo = f"TS {trail*100:.1f}% + SL {sl*100:.1f}%"
            print(f"    TS {trail*100:.1f}%+SL {sl*100:.1f}%: Net {rets.sum()*100:+7.1f}%, "
                  f"Alpha {alpha*100:+5.1f}%p, SR {sr:+.2f}, "
                  f"HR {(rets>0).mean()*100:.0f}%, MDD {mdd*100:+.1f}%")

    print(f"\n  Best combo: {best_combo} (Alpha {best_alpha*100:+.1f}%p)")


def analyze_garch(btc_df, dirs):
    """GARCH 기반 변동성 예측 분석"""
    btc = btc_df.copy()
    btc["ret_1m"] = btc["close"].pct_change().fillna(0)
    btc["direction"] = btc["trade_date"].astype(str).map(
        lambda x: dirs.get(x, (0, 0, 0))[0]
    )
    btc["sc_abs"] = btc["trade_date"].astype(str).map(
        lambda x: dirs.get(x, (0, 0, 0))[1]
    )

    trading = btc[btc["direction"] != 0].copy()
    trading["signed_ret"] = trading["direction"] * trading["ret_1m"] * 2

    # 일별 실현 변동성
    daily_vol = trading.groupby("trade_date").agg(
        realized_vol=("ret_1m", "std"),
        day_ret=("signed_ret", "sum"),
        direction=("direction", "first"),
        sc_abs=("sc_abs", "first"),
    ).reset_index()
    daily_vol = daily_vol.sort_values("trade_date").reset_index(drop=True)

    # 전일 변동성 → 오늘 예측 (naive GARCH: 전일 실현변동성 사용)
    daily_vol["prev_vol"] = daily_vol["realized_vol"].shift(1)
    daily_vol["prev_vol_5d"] = daily_vol["realized_vol"].rolling(5).mean().shift(1)
    daily_vol = daily_vol.dropna()

    # 변동성 자기상관 확인
    from scipy.stats import pearsonr
    corr, p = pearsonr(daily_vol["prev_vol"], daily_vol["realized_vol"])
    print(f"\n  변동성 자기상관 (1일): r={corr:.3f}, p={p:.4f}")
    corr5, p5 = pearsonr(daily_vol["prev_vol_5d"], daily_vol["realized_vol"])
    print(f"  변동성 자기상관 (5일 평균): r={corr5:.3f}, p={p5:.4f}")

    # 변동성 예측 기반 포지션 사이징
    # target_risk / predicted_vol = leverage
    # target_risk = median(realized_vol) * 2 (baseline leverage)
    target_risk = daily_vol["realized_vol"].median() * 2

    print(f"\n  target_risk: {target_risk*10000:.2f}bp")
    print(f"  실현변동성 분포: median {daily_vol['realized_vol'].median()*10000:.2f}bp, "
          f"mean {daily_vol['realized_vol'].mean()*10000:.2f}bp")

    # 전략: vol-targeting
    print(f"\n  === Vol-Targeting 포지션 사이징 ===")
    for pred_col, pred_name in [("prev_vol", "전일"), ("prev_vol_5d", "5일평균")]:
        for max_lev in [2.5, 3.0, 4.0]:
            rets = []
            levs = []
            for _, row in daily_vol.iterrows():
                pred_vol = row[pred_col]
                if pred_vol <= 0:
                    lev = 2.0
                else:
                    lev = min(max_lev, max(0.5, target_risk / pred_vol))
                levs.append(lev)

                gross = row["direction"] * row["day_ret"] / 2 * lev  # day_ret is already 2x
                cost_adj = 2 * (0.0005 + 0.0005) * lev
                rets.append(gross - cost_adj)

            rets = np.array(rets)
            levs = np.array(levs)
            sr = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
            mdd = float((np.cumsum(rets) - np.maximum.accumulate(np.cumsum(rets))).min())

            # baseline 비교
            bl_rets = (daily_vol["day_ret"] - COST).values
            alpha = rets.sum() - bl_rets.sum()

            print(f"    {pred_name} vol, max_lev {max_lev:.1f}x: "
                  f"Net {rets.sum()*100:+7.1f}%, Alpha {alpha*100:+5.1f}%p, "
                  f"SR {sr:+.2f}, avg_lev {levs.mean():.2f}x, MDD {mdd*100:+.1f}%")

    # sc_abs + vol 결합
    print(f"\n  === sc_abs + Vol-Targeting 결합 ===")
    for max_lev in [3.0, 4.0]:
        rets = []
        for _, row in daily_vol.iterrows():
            pred_vol = row["prev_vol_5d"]
            sc_a = row["sc_abs"]

            # sc_abs가 크면 aggressive, 작으면 conservative
            sc_boost = min(1.5, max(0.8, sc_a))

            if pred_vol <= 0:
                lev = 2.0 * sc_boost
            else:
                lev = min(max_lev, max(0.5, target_risk / pred_vol * sc_boost))

            gross = row["direction"] * row["day_ret"] / 2 * lev
            cost_adj = 2 * (0.0005 + 0.0005) * lev
            rets.append(gross - cost_adj)

        rets = np.array(rets)
        sr = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        mdd = float((np.cumsum(rets) - np.maximum.accumulate(np.cumsum(rets))).min())
        bl_rets = (daily_vol["day_ret"] - COST).values
        alpha = rets.sum() - bl_rets.sum()
        print(f"    sc_abs*vol, max {max_lev:.1f}x: Net {rets.sum()*100:+7.1f}%, "
              f"Alpha {alpha*100:+5.1f}%p, SR {sr:+.2f}, MDD {mdd*100:+.1f}%")


def main():
    vix = pd.read_parquet(config.BASE_DIR / "data" / "vix_slope_daily.parquet")
    dirs = get_directions(vix)

    for period, start, end in [
        ("IS (2024-01 ~ 2025-04)", "202401", "202504"),
        ("OOS (2025-05 ~ 2026-04)", "202505", "202604"),
    ]:
        print(f"\n{'='*70}")
        print(f"  {period}")
        print(f"{'='*70}")

        btc = load_btc(start, end)

        print(f"\n--- [모델1] 장중 최적 청산 ---")
        trading = analyze_exit(btc, dirs)
        test_exit_rules(trading)

        print(f"\n--- [모델2] GARCH 변동성 기반 포지션 사이징 ---")
        analyze_garch(btc, dirs)


if __name__ == "__main__":
    main()
