"""확정 전략 vs 비교군 — 공정 비교"""
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
        return dict(strategy=name, n_days=0, total_net=0, sharpe=0, hit_rate=0, max_dd=0, avg_lev=0, daily_pnls=np.array([]))
    total = daily_pnls.sum()
    sr = daily_pnls.mean() / daily_pnls.std() * np.sqrt(252) if daily_pnls.std() > 0 else 0
    hr = (daily_pnls > 0).mean()
    cum = np.cumsum(daily_pnls)
    mdd = (cum - np.maximum.accumulate(cum)).min()
    win_avg = daily_pnls[daily_pnls > 0].mean() if (daily_pnls > 0).any() else 0
    loss_avg = daily_pnls[daily_pnls <= 0].mean() if (daily_pnls <= 0).any() else 0
    return dict(strategy=name, n_days=n, total_net=total, sharpe=sr,
                hit_rate=hr, max_dd=mdd, avg_lev=daily_levs.mean(),
                daily_pnls=daily_pnls, win_avg=win_avg, loss_avg=loss_avg)

def main():
    vix = pd.read_parquet(config.BASE_DIR / "data" / "vix_slope_daily.parquet")
    vix["slope_change"] = vix["slope"].diff()
    vix["sc_abs"] = vix["slope_change"].abs()

    strategies = {
        # === 실험군 ===
        "10/5/3/1 (실험군)":  lambda sc, v, a: 10.0 if a >= 1.5 else (5.0 if a >= 1.0 else (3.0 if a >= 0.3 else 1.0)),

        # === 비교군 ===
        # B1: 고정 3x Long (slope 방향 무시, 매일 롱)
        "B1: 3x Long고정":    lambda sc, v, a: 3.0,  # direction 강제 +1 필요
        # B2: slope 방향 + 고정 3x (스케일링 없음)
        "B2: slope+3x고정":   lambda sc, v, a: 3.0,
        # B3: slope 방향 + 고정 1x (레버리지 없음)
        "B3: slope+1x고정":   lambda sc, v, a: 1.0,
    }

    for period, start, end in [
        ("8:2 IS (2024-01 ~ 2025-10)", "202401", "202510"),
        ("8:2 OOS (2025-11 ~ 2026-04)", "202511", "202604"),
    ]:
        print(f"\n{'='*80}")
        print(f"  {period}")
        print(f"{'='*80}")

        btc = load_btc(start, end)

        # B1은 방향 무시하고 무조건 Long이므로 별도 계산
        btc_c = btc.copy()
        btc_c["ret_1m"] = btc_c["close"].pct_change().fillna(0)
        b1_pnls = []
        for td in btc_c["trade_date"].unique():
            day = btc_c[btc_c["trade_date"] == td]
            if len(day) < 10:
                continue
            gross = (day["ret_1m"] * 3.0).sum()  # 무조건 Long 3x
            cost = 2 * (FEE + SLIP) * 3.0
            b1_pnls.append(gross - cost)
        b1_pnls = np.array(b1_pnls)
        b1_n = len(b1_pnls)
        b1_total = b1_pnls.sum()
        b1_sr = b1_pnls.mean() / b1_pnls.std() * np.sqrt(252) if b1_pnls.std() > 0 else 0
        b1_hr = (b1_pnls > 0).mean()
        b1_cum = np.cumsum(b1_pnls)
        b1_mdd = (b1_cum - np.maximum.accumulate(b1_cum)).min()

        # 나머지 전략 (slope 방향 사용)
        results = {}
        for name, func in strategies.items():
            if "Long고정" in name:
                continue
            results[name] = run_bt(btc, vix, name, func)

        # 출력
        print(f"\n  {'전략':<20s} | {'거래일':>4s} | {'Net':>10s} | {'Alpha':>8s} | {'SR':>5s} | {'HR':>5s} | {'MDD':>8s} | {'avg_lev':>7s}")
        print(f"  {'-'*20}-+-{'-'*4}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*7}")

        # B1
        baseline_net = results["B2: slope+3x고정"]["total_net"]
        print(f"  {'B1: 3x Long고정':<20s} | {b1_n:>4d} | {b1_total*100:+9.1f}% | {'(기준)':>8s} | {b1_sr:+4.2f} | {b1_hr*100:4.1f}% | {b1_mdd*100:+7.2f}% | {'3.00':>6s}x")

        # B2, B3
        for name in ["B3: slope+1x고정", "B2: slope+3x고정"]:
            r = results[name]
            alpha = r["total_net"] - baseline_net
            print(f"  {name:<20s} | {r['n_days']:>4d} | {r['total_net']*100:+9.1f}% | {alpha*100:+7.1f}%p | {r['sharpe']:+4.2f} | {r['hit_rate']*100:4.1f}% | {r['max_dd']*100:+7.2f}% | {r['avg_lev']:6.2f}x")

        # 실험군
        r = results["10/5/3/1 (실험군)"]
        alpha = r["total_net"] - baseline_net
        print(f"  {'-'*20}-+-{'-'*4}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*7}")
        print(f"  {'10/5/3/1 (실험군)':<20s} | {r['n_days']:>4d} | {r['total_net']*100:+9.1f}% | {alpha*100:+7.1f}%p | {r['sharpe']:+4.2f} | {r['hit_rate']*100:4.1f}% | {r['max_dd']*100:+7.2f}% | {r['avg_lev']:6.2f}x")

        # 상세 비교
        exp = results["10/5/3/1 (실험군)"]
        b2 = results["B2: slope+3x고정"]
        print(f"\n  --- 실험군 vs B2(slope+3x) 상세 비교 ---")
        print(f"  Net:    {exp['total_net']*100:+.1f}% vs {b2['total_net']*100:+.1f}% → Alpha {(exp['total_net']-b2['total_net'])*100:+.1f}%p")
        print(f"  Sharpe: {exp['sharpe']:+.2f} vs {b2['sharpe']:+.2f} → {exp['sharpe']-b2['sharpe']:+.2f}")
        print(f"  HR:     {exp['hit_rate']*100:.1f}% vs {b2['hit_rate']*100:.1f}% → {(exp['hit_rate']-b2['hit_rate'])*100:+.1f}%p")
        print(f"  MDD:    {exp['max_dd']*100:.1f}% vs {b2['max_dd']*100:.1f}% → {(exp['max_dd']-b2['max_dd'])*100:+.1f}%p")
        print(f"  avg_lev: {exp['avg_lev']:.2f}x vs {b2['avg_lev']:.2f}x")
        print(f"  승 평균: {exp['win_avg']*100:+.3f}% vs {b2['win_avg']*100:+.3f}%")
        print(f"  패 평균: {exp['loss_avg']*100:+.3f}% vs {b2['loss_avg']*100:+.3f}%")

        # B1 vs 실험군
        print(f"\n  --- 실험군 vs B1(3x Long 고정) ---")
        print(f"  Net:    {exp['total_net']*100:+.1f}% vs {b1_total*100:+.1f}%")
        print(f"  Sharpe: {exp['sharpe']:+.2f} vs {b1_sr:+.2f}")
        print(f"  MDD:    {exp['max_dd']*100:.1f}% vs {b1_mdd*100:.1f}%")


if __name__ == "__main__":
    main()
