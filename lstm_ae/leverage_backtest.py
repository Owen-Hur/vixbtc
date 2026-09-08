"""레버리지 조절 백테스트 — 일별 신뢰도 기반"""
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

def run_lev_bt(btc_df, ae_df, vix_df, name, lev_func):
    btc = btc_df.copy()
    btc = btc.join(ae_df[["anomaly_score","is_anomaly"]], how="left")
    btc["is_anomaly"] = btc["is_anomaly"].fillna(False)
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
                annual=total/n*252 if n else 0,
                sharpe=sr, hit_rate=hr, max_dd=mdd)


def main():
    vix = pd.read_parquet(config.BASE_DIR / "data" / "vix_slope_daily.parquet")
    vix["slope_change"] = vix["slope"].diff()
    vix["sc_abs"] = vix["slope_change"].abs()

    strategies = {
        "baseline_2x":    lambda sc, v, a: 2.0,
        "sc_scale":       lambda sc, v, a: min(3.0, max(1.0, a * 1.5)),
        "sc_2step":       lambda sc, v, a: 2.5 if a >= 0.9 else 1.5,
        "sc_3step":       lambda sc, v, a: 3.0 if a >= 1.5 else (2.0 if a >= 0.8 else 1.0),
        "vix_scale":      lambda sc, v, a: 2.5 if v >= 22 else (2.0 if v >= 16 else 1.5),
        "combined":       lambda sc, v, a: 2.5 if (a >= 0.9 and v >= 18) else (2.0 if a >= 0.7 else 1.5),
        "conservative":   lambda sc, v, a: 2.0 if a >= 0.7 else 1.0,
        "aggressive":     lambda sc, v, a: 3.0 if a >= 1.2 else 2.0,
        "linear":         lambda sc, v, a: min(3.0, 0.5 + a * 1.5),
    }

    for period, start, end, label in [
        ("IS", "202401", "202504", "is"),
        ("OOS", "202505", "202604", "oos"),
    ]:
        print(f"\n{'='*70}")
        print(f"  {period} ({start[:4]}-{start[4:]} ~ {end[:4]}-{end[4:]})")
        print(f"{'='*70}")

        btc = load_btc(start, end)
        ae = pd.read_parquet(config.ARTIFACT_DIR / f"anomaly_signals_{label}.parquet")

        results = {}
        for name, func in strategies.items():
            results[name] = run_lev_bt(btc, ae, vix, name, func)

        baseline = results["baseline_2x"]["total_net"]

        hdr = f"  {'전략':<18s} | {'Net':>8s} | {'Alpha':>8s} | {'Sharpe':>6s} | {'HR':>5s} | {'MDD':>7s}"
        sep = f"  {'-'*18}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*7}"
        print(hdr)
        print(sep)
        for name in strategies:
            r = results[name]
            alpha = r["total_net"] - baseline
            tn = r["total_net"]*100
            al = alpha*100
            sh = r["sharpe"]
            h = r["hit_rate"]*100
            md = r["max_dd"]*100
            print(f"  {name:<18s} | {tn:+7.1f}% | {al:+7.1f}% | {sh:+5.2f} | {h:4.1f}% | {md:+6.2f}%")

        # best alpha
        best = max((n for n in strategies if n != "baseline_2x"),
                   key=lambda n: results[n]["total_net"] - baseline)
        ba = (results[best]["total_net"] - baseline) * 100
        print(f"\n  Best: {best} → Alpha {ba:+.1f}%p")


if __name__ == "__main__":
    main()
