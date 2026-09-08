"""
LSTM-AE 활용 전략 백테스트

대조군: VIX slope 방향으로 종일 보유 (논문 베이스라인)
실험군: 대조군 + LSTM-AE anomaly 기반 리스크 관리

4가지 LSTM-AE 활용 전략:
  A. 청산형: anomaly → 포지션 청산, 해소 → 재진입
  B. 레버리지 축소형: anomaly → 2x→1x, 해소 → 2x 복귀
  C. 진입 필터형: anomaly 중에는 진입하지 않음
  D. 연속 사이징: score 크기에 비례해 포지션 축소

비용 구조:
  수수료: 0.05% per side (taker)
  슬리피지: 5bp per side
  레버리지: 2x
  펀딩비: 미반영 (별도 분석)

실행:
    cd btc_project
    python3 -m lstm_ae.backtest           # OOS 백테스트
    python3 -m lstm_ae.backtest --is      # IS 백테스트
    python3 -m lstm_ae.backtest --both    # IS + OOS 모두
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# ─── 비용 설정 ───────────────────────────────────────────
FEE_RATE = 0.0005       # 0.05% per side (taker)
SLIP_RATE = 0.0005      # 5bp per side
LEVERAGE = 2.0
MIN_HOLD_MINUTES = 5    # 최소 보유 시간

# 1회 거래 비용 = (fee + slip) × 레버리지
COST_PER_TRADE = (FEE_RATE + SLIP_RATE) * LEVERAGE  # 0.20% per side

# Score multiplier 실험: 기본 threshold × multiplier
SCORE_MULTIPLIERS = [1.0, 2.0, 3.0, 5.0]


def load_btc_1m(start_month, end_month):
    """BTC 1분봉 로딩"""
    all_files = sorted(glob.glob(str(config.DATA_DIR / "btc_1m_*.parquet")))
    selected = []
    for f in all_files:
        month = Path(f).stem.replace("btc_1m_", "")  # "2024-01"
        month_compact = month.replace("-", "")         # "202401"
        start_compact = start_month.replace("-", "")
        end_compact = end_month.replace("-", "")
        if start_compact <= month_compact <= end_compact:
            selected.append(f)

    dfs = [pd.read_parquet(f) for f in selected]
    df = pd.concat(dfs)
    df.sort_index(inplace=True)
    print(f"  BTC 1분봉: {len(df):,}행 ({len(selected)}개 파일)")
    return df


def load_vix_slope():
    """VIX slope daily + slope_change 계산"""
    vix = pd.read_parquet(config.BASE_DIR / "data" / "vix_slope_daily.parquet")
    vix["slope_change"] = vix["slope"].diff()
    return vix


def build_daily_direction(vix_df, slope_change_threshold=0.5):
    """
    VIX slope 변화율 기반 일별 방향 결정.
    |slope_change| >= threshold인 날만 거래.

    Returns:
        dict: {date_str: direction} where direction is 1 (Long) or -1 (Short)
    """
    directions = {}
    for date, row in vix_df.iterrows():
        sc = row["slope_change"]
        if pd.isna(sc):
            continue
        if abs(sc) < slope_change_threshold:
            # 필터링 — 이 날은 거래하지 않음
            directions[date.strftime("%Y-%m-%d")] = 0
        elif sc > 0:
            # slope 증가 = 정상화(contango 심화) = risk-on → Long
            directions[date.strftime("%Y-%m-%d")] = 1
        else:
            # slope 감소 = 역전 방향(backwardation) = risk-off → Short
            directions[date.strftime("%Y-%m-%d")] = -1
    return directions


def run_backtest(btc_df, ae_signals, directions, strategy, score_multiplier=1.0):
    """
    단일 전략 백테스트 실행.

    Args:
        btc_df: BTC 1분봉
        ae_signals: anomaly signal DataFrame
        directions: {date_str: direction} dict
        strategy: "baseline" | "exit" | "leverage" | "filter" | "sizing"
        score_multiplier: threshold에 곱할 배수

    Returns:
        result dict with performance metrics
    """
    # anomaly threshold 조정
    ae = ae_signals.copy()
    if score_multiplier != 1.0:
        # 원래 is_anomaly는 1x 기준 → multiplier 적용
        # threshold_json에서 가져오는 대신, score 분포 기반으로 재계산
        # 원래 7.5%가 1x → 5x면 훨씬 적은 비율만 anomaly
        for model_type in ae["model"].unique():
            mask = ae["model"] == model_type
            scores = ae.loc[mask, "anomaly_score"]
            # 1x threshold = 92.5 percentile
            base_th = scores.quantile(0.925)
            new_th = base_th * score_multiplier
            ae.loc[mask, "is_anomaly"] = scores > new_th

    # BTC에 anomaly 정보 병합
    btc = btc_df.copy()
    btc = btc.join(ae[["anomaly_score", "is_anomaly"]], how="left")
    btc["is_anomaly"] = btc["is_anomaly"].fillna(False)
    btc["anomaly_score"] = btc["anomaly_score"].fillna(0)

    # score 정규화 (연속 사이징용)
    max_score = btc["anomaly_score"].quantile(0.99)
    if max_score > 0:
        btc["norm_score"] = (btc["anomaly_score"] / max_score).clip(0, 1)
    else:
        btc["norm_score"] = 0

    # 1분봉 수익률
    btc["ret_1m"] = btc["close"].pct_change().fillna(0)

    # 일별 그룹으로 처리
    daily_pnls = []
    daily_trades = []
    daily_details = []

    trade_dates = btc["trade_date"].unique()

    for td in trade_dates:
        td_str = str(td)
        direction = directions.get(td_str, 0)

        if direction == 0:
            # 이 날은 거래 안 함
            continue

        day = btc[btc["trade_date"] == td].copy()
        if len(day) < 10:
            continue

        # 전략별 포지션 시퀀스 계산
        positions, n_trades = compute_positions(
            day, direction, strategy, score_multiplier
        )

        # PnL 계산 (레버리지 적용)
        gross_pnl = (positions * day["ret_1m"].values * LEVERAGE).sum()

        # 거래 비용
        trade_cost = n_trades * COST_PER_TRADE

        net_pnl = gross_pnl - trade_cost

        daily_pnls.append({
            "date": td,
            "direction": direction,
            "gross_pnl": gross_pnl,
            "trade_cost": trade_cost,
            "net_pnl": net_pnl,
            "n_trades": n_trades,
            "n_anomalies": day["is_anomaly"].sum(),
        })

    if not daily_pnls:
        return None

    results_df = pd.DataFrame(daily_pnls)

    # 성과 지표 계산
    total_net = results_df["net_pnl"].sum()
    total_gross = results_df["gross_pnl"].sum()
    total_cost = results_df["trade_cost"].sum()
    total_trades = results_df["n_trades"].sum()
    n_days = len(results_df)

    # 일별 수익률 → Sharpe
    daily_returns = results_df["net_pnl"]
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0)

    # Max Drawdown
    cum_pnl = daily_returns.cumsum()
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    max_dd = drawdown.min()

    # Hit rate
    hit_rate = (daily_returns > 0).mean()

    # 연환산 수익률
    annual_return = total_net / n_days * 252

    return {
        "strategy": strategy,
        "score_multiplier": score_multiplier,
        "n_days": n_days,
        "total_net": total_net,
        "total_gross": total_gross,
        "total_cost": total_cost,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "hit_rate": hit_rate,
        "total_trades": total_trades,
        "avg_trades_per_day": total_trades / n_days,
        "daily_returns": daily_returns,
    }


def compute_positions(day_df, direction, strategy, score_multiplier):
    """
    분 단위 포지션 시퀀스 계산.

    Returns:
        positions: np.array of position sizes (-1 ~ +1, before leverage)
        n_trades: 거래 횟수 (진입/청산/재진입 각 1회)
    """
    n = len(day_df)
    is_anomaly = day_df["is_anomaly"].values
    norm_score = day_df["norm_score"].values

    if strategy == "baseline":
        # 대조군: 종일 direction 방향 보유
        positions = np.full(n, float(direction))
        n_trades = 2  # 진입 + 청산
        return positions, n_trades

    elif strategy == "exit":
        # A. 청산형: anomaly → 청산, 해소 → 재진입
        positions = np.full(n, float(direction))
        in_anomaly = False
        exit_time = -999
        n_trades = 2  # 진입 + 최종 청산

        for i in range(n):
            if is_anomaly[i] and not in_anomaly:
                # anomaly 시작 → 청산
                in_anomaly = True
                exit_time = i
                n_trades += 1  # 청산 거래
            elif not is_anomaly[i] and in_anomaly:
                # anomaly 해소 → 최소 보유 체크 후 재진입
                if i - exit_time >= MIN_HOLD_MINUTES:
                    in_anomaly = False
                    n_trades += 1  # 재진입 거래

            if in_anomaly:
                positions[i] = 0  # 포지션 없음

        return positions, n_trades

    elif strategy == "leverage":
        # B. 레버리지 축소: anomaly → 절반으로 축소
        positions = np.full(n, float(direction))
        in_anomaly = False
        exit_time = -999
        n_trades = 2

        for i in range(n):
            if is_anomaly[i] and not in_anomaly:
                in_anomaly = True
                exit_time = i
                n_trades += 1  # 축소 거래
            elif not is_anomaly[i] and in_anomaly:
                if i - exit_time >= MIN_HOLD_MINUTES:
                    in_anomaly = False
                    n_trades += 1  # 복원 거래

            if in_anomaly:
                positions[i] = direction * 0.5  # 50%로 축소

        return positions, n_trades

    elif strategy == "filter":
        # C. 진입 필터: 장 시작 시 anomaly면 진입 보류
        positions = np.zeros(n)
        entered = False
        n_trades = 0

        for i in range(n):
            if not entered:
                if not is_anomaly[i]:
                    # anomaly 아닐 때 진입
                    entered = True
                    n_trades += 1
                    positions[i] = float(direction)
                # anomaly면 계속 대기
            else:
                positions[i] = float(direction)

        if entered:
            n_trades += 1  # 최종 청산

        return positions, n_trades

    elif strategy == "sizing":
        # D. 연속 사이징: score에 비례해 축소
        # position = direction × (1 - norm_score)
        scale = 1.0 - norm_score
        scale = np.clip(scale, 0, 1)
        positions = direction * scale

        # 거래 횟수: 포지션 변화가 10% 이상인 경우만 카운트
        pos_changes = np.abs(np.diff(positions))
        n_trades = 2 + int((pos_changes > 0.1).sum())

        return positions, n_trades

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def run_all_strategies(start_month, end_month, period_name):
    """모든 전략 + score_multiplier 조합 백테스트"""
    print(f"\n{'='*70}")
    print(f"  백테스트: {period_name} ({start_month} ~ {end_month})")
    print(f"{'='*70}")

    # 데이터 로딩
    print("\n[1/3] 데이터 로딩...")
    btc = load_btc_1m(start_month, end_month)
    ae = pd.read_parquet(config.ARTIFACT_DIR / f"anomaly_signals_{period_name}.parquet")
    vix = load_vix_slope()
    directions = build_daily_direction(vix, slope_change_threshold=0.5)

    print(f"  VIX slope 거래일: {sum(1 for v in directions.values() if v != 0)}일")
    print(f"  VIX slope 필터링: {sum(1 for v in directions.values() if v == 0)}일")
    print(f"  Anomaly 비율: {ae['is_anomaly'].mean()*100:.1f}%")

    # 전략 실행
    print("\n[2/3] 전략 실행...")
    strategies = ["baseline", "exit", "leverage", "filter", "sizing"]
    all_results = []

    for mult in SCORE_MULTIPLIERS:
        print(f"\n  --- Score Multiplier: {mult}x ---")
        for strat in strategies:
            if strat == "baseline" and mult != 1.0:
                continue  # baseline은 multiplier 무관

            result = run_backtest(btc, ae, directions, strat, mult)
            if result:
                all_results.append(result)
                net = result["total_net"]
                sr = result["sharpe"]
                hr = result["hit_rate"]
                trades = result["avg_trades_per_day"]
                ann = result["annual_return"]
                label = f"{strat}" if mult == 1.0 else f"{strat}_{mult}x"
                print(f"    {label:20s} | Net: {net*100:+7.2f}% | "
                      f"Annual: {ann*100:+7.1f}% | Sharpe: {sr:+5.2f} | "
                      f"HR: {hr*100:.1f}% | MDD: {result['max_dd']*100:+6.2f}% | "
                      f"Trades/day: {trades:.1f}")

    # 결과 요약
    print(f"\n{'='*70}")
    print(f"  결과 요약 ({period_name})")
    print(f"{'='*70}")

    baseline = [r for r in all_results if r["strategy"] == "baseline"][0]
    baseline_net = baseline["total_net"]
    baseline_sr = baseline["sharpe"]

    print(f"\n  대조군 (baseline): Net {baseline_net*100:+.2f}%, "
          f"Sharpe {baseline_sr:+.2f}, HR {baseline['hit_rate']*100:.1f}%")
    print()

    print(f"  {'전략':<22s} | {'mult':>4s} | {'Net':>8s} | {'Alpha':>8s} | "
          f"{'Sharpe':>6s} | {'HR':>5s} | {'MDD':>7s} | {'T/day':>5s}")
    print(f"  {'-'*22}-+-{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-"
          f"{'-'*6}-+-{'-'*5}-+-{'-'*7}-+-{'-'*5}")

    for r in all_results:
        if r["strategy"] == "baseline":
            continue
        alpha = r["total_net"] - baseline_net
        label = r["strategy"]
        mult = r["score_multiplier"]
        print(f"  {label:<22s} | {mult:>3.0f}x | {r['total_net']*100:+7.2f}% | "
              f"{alpha*100:+7.2f}% | {r['sharpe']:+5.2f} | "
              f"{r['hit_rate']*100:4.1f}% | {r['max_dd']*100:+6.2f}% | "
              f"{r['avg_trades_per_day']:4.1f}")

    # Best alpha 찾기
    non_baseline = [r for r in all_results if r["strategy"] != "baseline"]
    if non_baseline:
        best = max(non_baseline, key=lambda r: r["total_net"] - baseline_net)
        best_alpha = best["total_net"] - baseline_net
        print(f"\n  ★ Best Alpha: {best['strategy']} ({best['score_multiplier']}x) "
              f"→ Alpha {best_alpha*100:+.2f}%p, "
              f"Net {best['total_net']*100:+.2f}%")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is", dest="run_is", action="store_true",
                        help="IS 구간 백테스트")
    parser.add_argument("--both", action="store_true",
                        help="IS + OOS 모두")
    args = parser.parse_args()

    if args.both:
        is_results = run_all_strategies(config.IS_START, config.IS_END, "is")
        oos_results = run_all_strategies(config.OOS_START, config.OOS_END, "oos")
    elif args.run_is:
        run_all_strategies(config.IS_START, config.IS_END, "is")
    else:
        run_all_strategies(config.OOS_START, config.OOS_END, "oos")


if __name__ == "__main__":
    main()
