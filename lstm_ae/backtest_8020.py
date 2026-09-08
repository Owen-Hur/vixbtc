"""확정 전략 백테스트 — Constitution 80/20 구간 적용"""
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
        if pd.isna(sc) or abs(sc) < 0.5:
            dirs[date.strftime("%Y-%m-%d")] = (0, sc, row["vix"], row["sc_abs"])
        elif sc > 0:
            dirs[date.strftime("%Y-%m-%d")] = (1, sc, row["vix"], row["sc_abs"])
        else:
            dirs[date.strftime("%Y-%m-%d")] = (-1, sc, row["vix"], row["sc_abs"])

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
    total = daily_pnls.sum()
    sr = daily_pnls.mean() / daily_pnls.std() * np.sqrt(252) if daily_pnls.std() > 0 else 0
    hr = (daily_pnls > 0).mean()
    cum = np.cumsum(daily_pnls)
    mdd = (cum - np.maximum.accumulate(cum)).min()
    return dict(strategy=name, n_days=n, total_net=total,
                sharpe=sr, hit_rate=hr, max_dd=mdd)


def main():
    vix = pd.read_parquet(config.BASE_DIR / "data" / "vix_slope_daily.parquet")
    vix["slope_change"] = vix["slope"].diff()
    vix["sc_abs"] = vix["slope_change"].abs()

    strategies = {
        "baseline_2x":    lambda sc, v, a: 2.0,
        "aggressive":     lambda sc, v, a: 3.0 if a >= 1.2 else 2.0,
        "sc_2step":       lambda sc, v, a: 2.5 if a >= 0.9 else 1.5,
        "combined":       lambda sc, v, a: 2.5 if (a >= 0.9 and v >= 18) else (2.0 if a >= 0.7 else 1.5),
        "conservative":   lambda sc, v, a: 2.0 if a >= 0.7 else 1.0,
    }

    # 기존 구간 vs Constitution 구간
    splits = [
        ("기존 IS",  "202401", "202504", "기존 OOS",  "202505", "202604"),
        ("8:2 IS",   "202401", "202510", "8:2 OOS",   "202511", "202604"),
    ]

    for is_name, is_start, is_end, oos_name, oos_start, oos_end in splits:
        for period, start, end in [
            (is_name, is_start, is_end),
            (oos_name, oos_start, oos_end),
        ]:
            print(f"\n{'='*70}")
            print(f"  {period} ({start[:4]}-{start[4:]} ~ {end[:4]}-{end[4:]})")
            print(f"{'='*70}")

            btc = load_btc(start, end)

            results = {}
            for name, func in strategies.items():
                results[name] = run_bt(btc, vix, name, func)

            baseline = results["baseline_2x"]["total_net"]

            hdr = f"  {'전략':<18s} | {'거래일':>5s} | {'Net':>8s} | {'Alpha':>8s} | {'Sharpe':>6s} | {'HR':>5s} | {'MDD':>7s}"
            sep = f"  {'-'*18}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*7}"
            print(hdr)
            print(sep)
            for name in strategies:
                r = results[name]
                alpha = r["total_net"] - baseline
                print(f"  {name:<18s} | {r['n_days']:>5d} | {r['total_net']*100:+7.1f}% | {alpha*100:+7.1f}% | {r['sharpe']:+5.2f} | {r['hit_rate']*100:4.1f}% | {r['max_dd']*100:+6.2f}%")

            best = max((n for n in strategies if n != "baseline_2x"),
                       key=lambda n: results[n]["total_net"] - baseline)
            ba = (results[best]["total_net"] - baseline) * 100
            print(f"\n  Best: {best} -> Alpha {ba:+.1f}%p")


if __name__ == "__main__":
    main()
