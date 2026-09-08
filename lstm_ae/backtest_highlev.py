"""고레버리지 매일 거래 백테스트"""
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

def run_bt(btc_df, vix_df, name, lev_func):
    btc = btc_df.copy()
    btc["ret_1m"] = btc["close"].pct_change().fillna(0)

    dirs = {}
    for date, row in vix_df.iterrows():
        sc = row["slope_change"]
        if pd.isna(sc) or sc == 0:
            dirs[date.strftime("%Y-%m-%d")] = (0, 0, row["vix"], 0)
        elif sc > 0:
            dirs[date.strftime("%Y-%m-%d")] = (1, sc, row["vix"], abs(sc))
        else:
            dirs[date.strftime("%Y-%m-%d")] = (-1, sc, row["vix"], abs(sc))

    daily_pnls = []
    daily_levs = []
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
        daily_levs.append(lev)

    daily_pnls = np.array(daily_pnls)
    daily_levs = np.array(daily_levs)
    n = len(daily_pnls)
    if n == 0:
        return dict(strategy=name, n_days=0, total_net=0, sharpe=0, hit_rate=0, max_dd=0, avg_lev=0)
    total = daily_pnls.sum()
    sr = daily_pnls.mean() / daily_pnls.std() * np.sqrt(252) if daily_pnls.std() > 0 else 0
    hr = (daily_pnls > 0).mean()
    cum = np.cumsum(daily_pnls)
    mdd = (cum - np.maximum.accumulate(cum)).min()
    return dict(strategy=name, n_days=n, total_net=total, sharpe=sr, 
                hit_rate=hr, max_dd=mdd, avg_lev=daily_levs.mean())


def main():
    vix = pd.read_parquet(config.BASE_DIR / "data" / "vix_slope_daily.parquet")
    vix["slope_change"] = vix["slope"].diff()
    vix["sc_abs"] = vix["slope_change"].abs()

    strategies = {
        # 기준: 현재 daily_3step_v2
        "v2_base(3/2/1)":     lambda sc, v, a: 3.0 if a >= 1.0 else (2.0 if a >= 0.3 else 1.0),
        
        # max 4x
        "4/2/1":              lambda sc, v, a: 4.0 if a >= 1.0 else (2.0 if a >= 0.3 else 1.0),
        "4/3/1":              lambda sc, v, a: 4.0 if a >= 1.0 else (3.0 if a >= 0.3 else 1.0),
        "4/3/2":              lambda sc, v, a: 4.0 if a >= 1.0 else (3.0 if a >= 0.3 else 2.0),
        "4/2.5/1.5":          lambda sc, v, a: 4.0 if a >= 1.0 else (2.5 if a >= 0.3 else 1.5),
        
        # max 5x
        "5/3/1":              lambda sc, v, a: 5.0 if a >= 1.0 else (3.0 if a >= 0.3 else 1.0),
        "5/3/2":              lambda sc, v, a: 5.0 if a >= 1.0 else (3.0 if a >= 0.3 else 2.0),
        "5/4/2":              lambda sc, v, a: 5.0 if a >= 1.0 else (4.0 if a >= 0.3 else 2.0),
        
        # max 6x~8x
        "6/4/2":              lambda sc, v, a: 6.0 if a >= 1.0 else (4.0 if a >= 0.3 else 2.0),
        "8/4/2":              lambda sc, v, a: 8.0 if a >= 1.0 else (4.0 if a >= 0.3 else 2.0),
        "8/5/2":              lambda sc, v, a: 8.0 if a >= 1.0 else (5.0 if a >= 0.3 else 2.0),
        "10/5/2":             lambda sc, v, a: 10.0 if a >= 1.0 else (5.0 if a >= 0.3 else 2.0),
        
        # 4단계 (sc 구간 세분화)
        "5/4/3/1":            lambda sc, v, a: 5.0 if a >= 1.5 else (4.0 if a >= 1.0 else (3.0 if a >= 0.3 else 1.0)),
        "6/5/3/1":            lambda sc, v, a: 6.0 if a >= 1.5 else (5.0 if a >= 1.0 else (3.0 if a >= 0.3 else 1.0)),
        "8/5/3/1":            lambda sc, v, a: 8.0 if a >= 1.5 else (5.0 if a >= 1.0 else (3.0 if a >= 0.3 else 1.0)),
        "10/6/3/1":           lambda sc, v, a: 10.0 if a >= 1.5 else (6.0 if a >= 1.0 else (3.0 if a >= 0.3 else 1.0)),
        
        # 연속 스케일링 (상한 조절)
        "linear_max5":        lambda sc, v, a: min(5.0, max(1.0, a * 3.0)),
        "linear_max8":        lambda sc, v, a: min(8.0, max(1.0, a * 4.0)),
        "linear_max10":       lambda sc, v, a: min(10.0, max(1.0, a * 5.0)),
    }

    for period, start, end in [
        ("8:2 IS (2024-01 ~ 2025-10)", "202401", "202510"),
        ("8:2 OOS (2025-11 ~ 2026-04)", "202511", "202604"),
    ]:
        print(f"\n{'='*75}")
        print(f"  {period}")
        print(f"{'='*75}")

        btc = load_btc(start, end)

        results = {}
        for name, func in strategies.items():
            results[name] = run_bt(btc, vix, name, func)

        hdr = f"  {'전략':<20s} | {'거래일':>4s} | {'Net':>9s} | {'SR':>5s} | {'HR':>5s} | {'MDD':>8s} | {'avg_lev':>7s}"
        sep = f"  {'-'*20}-+-{'-'*4}-+-{'-'*9}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*7}"
        print(hdr)
        print(sep)
        for name in strategies:
            r = results[name]
            print(f"  {name:<20s} | {r['n_days']:>4d} | {r['total_net']*100:+8.1f}% | {r['sharpe']:+4.2f} | {r['hit_rate']*100:4.1f}% | {r['max_dd']*100:+7.2f}% | {r['avg_lev']:6.2f}x")


if __name__ == "__main__":
    main()
