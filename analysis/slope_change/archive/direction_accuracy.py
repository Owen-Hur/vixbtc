"""
VIX slope_change 방향 예측 정확도 검증

논문 기준: slope = VIX3M - VIX (일별 종가)
slope_change = slope[t] - slope[t-1]

핵심 검증:
  1) slope 계산이 논문과 동일한지
  2) lookahead bias 존재 여부
     - t 기준 (biased): 당일 slope_change로 당일 BTC 방향 예측
     - t-1 기준 (unbiased): 전일 slope_change로 당일 BTC 방향 예측
  3) 방향 정확도만 측정 (레버리지/수익률 무관)
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "btc_historical_processed"


def load_btc(start, end):
    files = sorted(glob.glob(str(DATA_DIR / "btc_1m_*.parquet")))
    sel = [f for f in files
           if start <= Path(f).stem.replace("btc_1m_", "").replace("-", "") <= end]
    return pd.concat([pd.read_parquet(f) for f in sel]).sort_index()


def compute_daily_btc_return(btc_df):
    """trade_date별 장중 BTC 수익률 (open-to-close)"""
    results = {}
    for td in btc_df["trade_date"].unique():
        day = btc_df[btc_df["trade_date"] == td].sort_index()
        if len(day) < 10:
            continue
        open_price = day["close"].iloc[0]
        close_price = day["close"].iloc[-1]
        ret = (close_price - open_price) / open_price
        results[str(td)] = ret
    return results


def main():
    vix = pd.read_parquet(BASE_DIR / "data" / "vix_slope_daily.parquet")

    # ── 1. slope 계산 검증 ──
    print("=" * 70)
    print("1. slope 계산 검증 (논문: slope = VIX3M - VIX)")
    print("=" * 70)
    vix["slope_check"] = vix["vix3m"] - vix["vix"]
    diff = (vix["slope"] - vix["slope_check"]).abs()
    print(f"  parquet의 slope vs (VIX3M - VIX) 최대 차이: {diff.max():.10f}")
    print(f"  평균 차이: {diff.mean():.10f}")
    if diff.max() < 1e-6:
        print("  ✅ slope = VIX3M - VIX 정확히 일치")
    else:
        print("  ⚠️ slope 계산에 차이 존재!")
    print()

    # ── 2. slope_change 계산 ──
    vix["slope_change"] = vix["slope"].diff()

    # t 기준 (biased): 당일 slope_change
    # t-1 기준 (unbiased): 전일 slope_change = shift(1)
    vix["sc_t0"] = vix["slope_change"]       # 당일 (lookahead!)
    vix["sc_t1"] = vix["slope_change"].shift(1)  # 전일 (no lookahead)

    print("=" * 70)
    print("2. Lookahead Bias 검증")
    print("=" * 70)
    print()
    print("  slope_change[t] = slope[t] - slope[t-1]")
    print("  slope[t] = VIX3M_close[t] - VIX_close[t]")
    print()
    print("  VIX/VIX3M 종가 확정: 16:15 ET")
    print("  BTC 거래 시간 (backtest): 09:30 ~ 15:59 ET")
    print()
    print("  ⚠️ slope_change[t]를 t일 BTC 방향 결정에 사용하면")
    print("     아직 확정되지 않은 당일 VIX 종가를 사용하는 것 → LOOKAHEAD BIAS")
    print()
    print("  ✅ 올바른 방법: slope_change[t-1] (전일 확정값)으로 t일 방향 결정")
    print()

    # ── 3. 방향 정확도 측정 ──
    for period_name, start, end in [
        ("IS (2024-01 ~ 2025-10)", "202401", "202510"),
        ("OOS (2025-11 ~ 2026-04)", "202511", "202604"),
        ("전체 (2024-01 ~ 2026-04)", "202401", "202604"),
    ]:
        print("=" * 70)
        print(f"3. 방향 정확도: {period_name}")
        print("=" * 70)

        btc = load_btc(start, end)
        daily_ret = compute_daily_btc_return(btc)

        for label, sc_col in [("t 기준 (BIASED)", "sc_t0"),
                               ("t-1 기준 (UNBIASED)", "sc_t1")]:
            correct = 0
            total = 0
            correct_long = 0
            total_long = 0
            correct_short = 0
            total_short = 0

            for date, row in vix.iterrows():
                sc = row[sc_col]
                if pd.isna(sc) or sc == 0:
                    continue
                date_str = date.strftime("%Y-%m-%d")
                if date_str not in daily_ret:
                    continue

                btc_ret = daily_ret[date_str]
                predicted_dir = 1 if sc > 0 else -1
                actual_dir = 1 if btc_ret > 0 else -1

                total += 1
                if predicted_dir == actual_dir:
                    correct += 1

                if predicted_dir == 1:
                    total_long += 1
                    if actual_dir == 1:
                        correct_long += 1
                else:
                    total_short += 1
                    if actual_dir == 1:
                        pass
                    else:
                        correct_short += 1

            acc = correct / total * 100 if total > 0 else 0
            acc_long = correct_long / total_long * 100 if total_long > 0 else 0
            acc_short = correct_short / total_short * 100 if total_short > 0 else 0

            print(f"\n  [{label}]")
            print(f"    전체 거래일: {total}일")
            print(f"    방향 정확도: {correct}/{total} = {acc:.1f}%")
            print(f"    Long  정확도: {correct_long}/{total_long} = {acc_long:.1f}%")
            print(f"    Short 정확도: {correct_short}/{total_short} = {acc_short:.1f}%")

        print()

    # ── 4. slope 수준(논문 방식) vs slope 변화율 비교 ──
    print("=" * 70)
    print("4. 논문 방식(slope 수준) vs slope_change 방향 정확도 비교")
    print("=" * 70)

    btc_all = load_btc("202401", "202604")
    daily_ret_all = compute_daily_btc_return(btc_all)

    for period_name, start, end in [
        ("IS", "2024-01", "2025-10"),
        ("OOS", "2025-11", "2026-04"),
    ]:
        period_vix = vix[(vix.index >= start) & (vix.index <= end)]

        methods = {
            "slope 수준 (논문: slope>0→Long)": lambda row: 1 if row["slope"] > 0 else -1,
            "slope_change t (BIASED)": lambda row: (1 if row["sc_t0"] > 0 else -1) if not pd.isna(row["sc_t0"]) and row["sc_t0"] != 0 else None,
            "slope_change t-1 (UNBIASED)": lambda row: (1 if row["sc_t1"] > 0 else -1) if not pd.isna(row["sc_t1"]) and row["sc_t1"] != 0 else None,
        }

        print(f"\n  [{period_name}]")
        for method_name, pred_func in methods.items():
            correct = 0
            total = 0
            for date, row in period_vix.iterrows():
                date_str = date.strftime("%Y-%m-%d")
                if date_str not in daily_ret_all:
                    continue
                pred = pred_func(row)
                if pred is None:
                    continue
                actual = 1 if daily_ret_all[date_str] > 0 else -1
                total += 1
                if pred == actual:
                    correct += 1
            acc = correct / total * 100 if total > 0 else 0
            print(f"    {method_name:<35s}: {correct}/{total} = {acc:.1f}%")

    # ── 5. |slope_change| 크기별 정확도 (unbiased) ──
    print()
    print("=" * 70)
    print("5. |slope_change(t-1)| 크기별 방향 정확도 (UNBIASED)")
    print("=" * 70)

    for period_name, start, end in [
        ("IS", "2024-01", "2025-10"),
        ("OOS", "2025-11", "2026-04"),
    ]:
        period_vix = vix[(vix.index >= start) & (vix.index <= end)]
        records = []
        for date, row in period_vix.iterrows():
            sc = row["sc_t1"]
            if pd.isna(sc) or sc == 0:
                continue
            date_str = date.strftime("%Y-%m-%d")
            if date_str not in daily_ret_all:
                continue
            pred = 1 if sc > 0 else -1
            actual = 1 if daily_ret_all[date_str] > 0 else -1
            records.append({"sc_abs": abs(sc), "correct": pred == actual})

        df = pd.DataFrame(records)
        if len(df) == 0:
            continue

        bins = [0, 0.3, 0.5, 1.0, 1.5, float("inf")]
        labels_bin = ["<0.3", "0.3~0.5", "0.5~1.0", "1.0~1.5", ">=1.5"]
        df["bin"] = pd.cut(df["sc_abs"], bins=bins, labels=labels_bin)

        print(f"\n  [{period_name}]")
        print(f"  {'|sc(t-1)| 범위':<15s} | {'건수':>4s} | {'정확도':>6s}")
        print(f"  {'-'*15}-+-{'-'*4}-+-{'-'*6}")
        for b in labels_bin:
            sub = df[df["bin"] == b]
            if len(sub) == 0:
                continue
            acc = sub["correct"].mean() * 100
            print(f"  {b:<15s} | {len(sub):>4d} | {acc:.1f}%")

        total_acc = df["correct"].mean() * 100
        print(f"  {'전체':<15s} | {len(df):>4d} | {total_acc:.1f}%")


if __name__ == "__main__":
    main()
