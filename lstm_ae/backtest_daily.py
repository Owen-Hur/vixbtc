"""매일 거래 백테스트 — |slope_change| 필터 제거"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from . import config

FEE = 0.0005
SLIP = 0.0005

def load_btc(start, end):
    btc_files = sorted(glob.glob(str(config.DATA_DIR / "btc_1m_*.parquet")))
    sel = [f for f in btc_files
           if start <= Path(f).stem.replace("btc_1m_","").replace("-","") <= end]
    return pd.concat([pd.read_parquet(f) for f in sel]).sort_index()

def run_bt(btc_df, vix_df, name, lev_func, use_filter):
    btc = btc_df.copy()
    btc["ret_1m"] = btc["close"].pct_change().fillna(0)

    dirs = {}
    for date, row in vix_df.iterrows():
        sc = row["slope_change"]
        if pd.isna(sc):
            dirs[date.strftime("%Y-%m-%d")] = (0, 0, row["vix"], 0)
            continue
        if use_filter and abs(sc) < 0.5:
            dirs[date.strftime("%Y-%m-%d")] = (0, sc, row["vix"], abs(sc))
        elif sc > 0:
            dirs[date.strftime("%Y-%m-%d")] = (1, sc, row["vix"], abs(sc))
        elif sc < 0:
            dirs[date.strftime("%Y-%m-%d")] = (-1, sc, row["vix"], abs(sc))
        else:
            dirs[date.strftime("%Y-%m-%d")] = (0, 0, row["vix"], 0)

    daily_pnls = []
    for td in btc["trade_date"].unique():
        info = dirs.get(str(td))
        if info is None or info[0] == 0:
            continue
        direction, sc, vix_level, sc_abs = info
        lev = lev_func(sc, vix_level, sc_abs)
        day = btc[btc["trade_date"] == td]
        if len(day) < 10:
            continue
        gross = (direction * day["ret_1m"] * lev).sum()
        cost = 2 * (FEE + SLIP) * lev
        daily_pnls.append(gross - cost)

    daily_pnls = np.array(daily_pnls)
    n = len(daily_pnls)
    if n == 0:
        return dict(strategy=name, n_days=0, total_net=0, sharpe=0, hit_rate=0, max_dd=0)
    total = daily_pnls.sum()
    sr = daily_pnls.mean() / daily_pnls.std() * np.sqrt(252) if daily_pnls.std() > 0 else 0
    hr = (daily_pnls > 0).mean()
    cum = np.cumsum(daily_pnls)
    mdd = (cum - np.maximum.accumulate(cum)).min()
    return dict(strategy=name, n_days=n, total_net=total, sharpe=sr, hit_rate=hr, max_dd=mdd)


def main():
    vix = pd.read_parquet(config.BASE_DIR / "data" / "vix_slope_daily.parquet")
    vix["slope_change"] = vix["slope"].diff()
    vix["sc_abs"] = vix["slope_change"].abs()

    strategies = {
        # 필터 없이 매일 거래
        "daily_2x":         (False, lambda sc, v, a: 2.0),
        "daily_agg":        (False, lambda sc, v, a: 3.0 if a >= 1.2 else 2.0),
        "daily_agg_0.8":    (False, lambda sc, v, a: 3.0 if a >= 0.8 else 2.0),
        "daily_3step":      (False, lambda sc, v, a: 3.0 if a >= 1.2 else (2.0 if a >= 0.5 else 1.0)),
        "daily_3step_v2":   (False, lambda sc, v, a: 3.0 if a >= 1.0 else (2.0 if a >= 0.3 else 1.0)),
        "daily_sc_scale":   (False, lambda sc, v, a: min(3.0, max(1.0, a * 2.0))),
        # 기존 필터 있는 버전 (비교용)
        "filter_2x":        (True,  lambda sc, v, a: 2.0),
        "filter_agg":       (True,  lambda sc, v, a: 3.0 if a >= 1.2 else 2.0),
    }

    for period, start, end in [
        ("8:2 IS (2024-01 ~ 2025-10)", "202401", "202510"),
        ("8:2 OOS (2025-11 ~ 2026-04)", "202511", "202604"),
    ]:
        print(f"\n{'='*70}")
        print(f"  {period}")
        print(f"{'='*70}")

        btc = load_btc(start, end)

        results = {}
        for name, (use_filt, func) in strategies.items():
            results[name] = run_bt(btc, vix, name, func, use_filt)

        hdr = f"  {'전략':<20s} | {'거래일':>5s} | {'Net':>8s} | {'Sharpe':>6s} | {'HR':>5s} | {'MDD':>7s}"
        sep = f"  {'-'*20}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*7}"
        print(hdr)
        print(sep)
        for name in strategies:
            r = results[name]
            print(f"  {name:<20s} | {r['n_days']:>5d} | {r['total_net']*100:+7.1f}% | {r['sharpe']:+5.2f} | {r['hit_rate']*100:4.1f}% | {r['max_dd']*100:+6.2f}%")

        # daily vs filter 비교
        print(f"\n  --- 매일 거래 vs 필터 비교 ---")
        d2 = results["daily_2x"]
        f2 = results["filter_2x"]
        da = results["daily_agg"]
        fa = results["filter_agg"]
        print(f"  2x 고정:   매일 {d2['total_net']*100:+.1f}% ({d2['n_days']}일) vs 필터 {f2['total_net']*100:+.1f}% ({f2['n_days']}일)")
        print(f"  Aggressive: 매일 {da['total_net']*100:+.1f}% ({da['n_days']}일) vs 필터 {fa['total_net']*100:+.1f}% ({fa['n_days']}일)")


if __name__ == "__main__":
    main()
