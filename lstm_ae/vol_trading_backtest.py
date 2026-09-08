"""
변동성 트레이딩 백테스트

전략 A: Anomaly 브레이크아웃
  - 기존 slope 전략 위에 추가 레이어
  - anomaly 발생 시 현재가 대비 ±X% 브레이크아웃 주문
  - 돌파 방향으로 추가 포지션 진입
  - 방향 예측 불필요

전략 B: Anomaly 기반 Vol 타겟팅
  - anomaly 빈도/강도로 변동성 예측
  - 목표 리스크 대비 레버리지 조절
  - 장중 실시간 업데이트

전략 C: 순수 변동성 전략 (slope 독립)
  - slope 방향과 무관하게
  - anomaly 발생 시 브레이크아웃만으로 거래
  - 필터링된 185일에서도 거래 가능
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

FEE = 0.0005
SLIP = 0.0005


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


def cost(leverage):
    return 2 * (FEE + SLIP) * leverage


# ═══════════════════════════════════════════════════════
#  전략 A: Anomaly 브레이크아웃 (slope 전략 위에 추가)
# ═══════════════════════════════════════════════════════

def strategy_a_breakout(btc_df, ae_df, dirs, breakout_pct, add_lev, hold_minutes):
    """
    기존 slope 2x + anomaly 시 브레이크아웃 추가 포지션.

    breakout_pct: 브레이크아웃 진입 기준 (예: 0.003 = 0.3%)
    add_lev: 추가 포지션 레버리지 (예: 1.0)
    hold_minutes: 브레이크아웃 포지션 보유 시간
    """
    btc = btc_df.copy()
    btc = btc.join(ae_df[["anomaly_score", "is_anomaly"]], how="left")
    btc["is_anomaly"] = btc["is_anomaly"].fillna(False)
    btc["ret_1m"] = btc["close"].pct_change().fillna(0)

    daily_pnls_base = []
    daily_pnls_strat = []

    for td in btc["trade_date"].unique():
        td_str = str(td)
        info = dirs.get(td_str)
        if info is None:
            continue
        direction, sc_abs, vix_level = info

        day = btc[btc["trade_date"] == td]
        if len(day) < 30:
            continue

        prices = day["close"].values
        rets = day["ret_1m"].values
        is_anom = day["is_anomaly"].values

        # baseline: slope 방향으로 보유 (거래일만)
        if direction != 0:
            base_lev = 3.0 if sc_abs >= 1.2 else 2.0
            base_pnl = float((direction * rets * base_lev).sum()) - cost(base_lev)
            daily_pnls_base.append(base_pnl)
        else:
            base_pnl = 0
            daily_pnls_base.append(0)

        # 브레이크아웃 추가 수익
        breakout_pnl = 0
        n_breakout_trades = 0
        i = 0
        while i < len(day):
            if not is_anom[i]:
                i += 1
                continue

            # anomaly 발생 → 현재가 기준 브레이크아웃 대기
            trigger_price = prices[i]
            up_trigger = trigger_price * (1 + breakout_pct)
            down_trigger = trigger_price * (1 - breakout_pct)

            # 향후 분봉에서 돌파 확인 (최대 30분 대기)
            triggered = False
            for j in range(i + 1, min(i + 31, len(day))):
                if prices[j] >= up_trigger:
                    # 상방 돌파 → Long 추가
                    bo_dir = 1
                    triggered = True
                    break
                elif prices[j] <= down_trigger:
                    # 하방 돌파 → Short 추가
                    bo_dir = -1
                    triggered = True
                    break

            if triggered:
                # hold_minutes 동안 보유
                entry_idx = j
                exit_idx = min(entry_idx + hold_minutes, len(day) - 1)
                bo_ret = float((bo_dir * rets[entry_idx:exit_idx + 1] * add_lev).sum())
                bo_ret -= cost(add_lev)  # 추가 거래 비용
                breakout_pnl += bo_ret
                n_breakout_trades += 1
                i = exit_idx + 1  # 다음 anomaly 탐색
            else:
                i += 1

        daily_pnls_strat.append(base_pnl + breakout_pnl)

    base = np.array(daily_pnls_base)
    strat = np.array(daily_pnls_strat)
    return base, strat


# ═══════════════════════════════════════════════════════
#  전략 B: Anomaly 기반 Vol 타겟팅
# ═══════════════════════════════════════════════════════

def strategy_b_vol_target(btc_df, ae_df, dirs, target_vol_bp, lookback, max_lev):
    """
    장중 anomaly 빈도로 실시간 변동성 추정 → 레버리지 조절.

    - anomaly 많으면 → 변동성 높다고 판단 → 레버리지 축소 (리스크 일정)
    - anomaly 없으면 → 변동성 낮다고 판단 → 레버리지 유지/확대

    target_vol_bp: 목표 분당 변동성 (bp)
    lookback: anomaly 추정에 사용할 과거 분 수
    max_lev: 최대 레버리지
    """
    btc = btc_df.copy()
    btc = btc.join(ae_df[["anomaly_score", "is_anomaly"]], how="left")
    btc["is_anomaly"] = btc["is_anomaly"].fillna(False)
    btc["anomaly_score"] = btc["anomaly_score"].fillna(0)
    btc["ret_1m"] = btc["close"].pct_change().fillna(0)

    daily_pnls_base = []
    daily_pnls_strat = []

    for td in btc["trade_date"].unique():
        td_str = str(td)
        info = dirs.get(td_str)
        if info is None or info[0] == 0:
            continue
        direction, sc_abs, vix_level = info

        day = btc[btc["trade_date"] == td]
        if len(day) < 30:
            continue

        rets = day["ret_1m"].values
        scores = day["anomaly_score"].values
        is_anom = day["is_anomaly"].values

        base_lev = 3.0 if sc_abs >= 1.2 else 2.0
        base_pnl = float((direction * rets * base_lev).sum()) - cost(base_lev)
        daily_pnls_base.append(base_pnl)

        # 장중 실시간 레버리지 조절
        strat_pnl = 0
        n_lev_changes = 0
        prev_lev = base_lev

        for i in range(len(day)):
            # lookback 분간 실현 변동성 계산
            start_idx = max(0, i - lookback)
            if i - start_idx >= 5:
                recent_vol = np.std(rets[start_idx:i]) * 10000  # bp 단위
            else:
                recent_vol = target_vol_bp  # 초기에는 목표 변동성 사용

            # anomaly 보정: anomaly 구간이면 변동성 1.35x 예상
            recent_anom_rate = np.mean(is_anom[start_idx:max(1, i)])
            vol_estimate = recent_vol * (1 + recent_anom_rate * 0.35)

            # 레버리지 = 목표 변동성 / 추정 변동성 × 기본 레버리지
            if vol_estimate > 0:
                lev = min(max_lev, max(0.5, target_vol_bp / vol_estimate * base_lev))
            else:
                lev = base_lev

            # 레버리지 변경 횟수 (30분마다만 조절)
            if i % 30 == 0 and abs(lev - prev_lev) > 0.3:
                n_lev_changes += 1
                prev_lev = lev

            strat_pnl += direction * rets[i] * lev

        # 비용: 기본 진입/청산 + 레버리지 변경 횟수
        avg_lev = base_lev  # 근사
        strat_pnl -= cost(avg_lev) + n_lev_changes * cost(1.0)
        daily_pnls_strat.append(strat_pnl)

    base = np.array(daily_pnls_base)
    strat = np.array(daily_pnls_strat)
    return base, strat


# ═══════════════════════════════════════════════════════
#  전략 C: 순수 변동성 (slope 무관, 필터일 포함)
# ═══════════════════════════════════════════════════════

def strategy_c_pure_vol(btc_df, ae_df, breakout_pct, add_lev, hold_minutes):
    """
    slope 방향 없이, anomaly 발생 시 브레이크아웃만으로 거래.
    모든 거래일 대상 (필터링된 185일 포함).
    """
    btc = btc_df.copy()
    btc = btc.join(ae_df[["anomaly_score", "is_anomaly"]], how="left")
    btc["is_anomaly"] = btc["is_anomaly"].fillna(False)
    btc["ret_1m"] = btc["close"].pct_change().fillna(0)

    daily_pnls = []
    daily_trades = []

    for td in btc["trade_date"].unique():
        day = btc[btc["trade_date"] == td]
        if len(day) < 30:
            continue

        prices = day["close"].values
        rets = day["ret_1m"].values
        is_anom = day["is_anomaly"].values

        day_pnl = 0
        n_trades = 0
        i = 0

        while i < len(day):
            if not is_anom[i]:
                i += 1
                continue

            trigger_price = prices[i]
            up_trigger = trigger_price * (1 + breakout_pct)
            down_trigger = trigger_price * (1 - breakout_pct)

            triggered = False
            for j in range(i + 1, min(i + 31, len(day))):
                if prices[j] >= up_trigger:
                    bo_dir = 1
                    triggered = True
                    break
                elif prices[j] <= down_trigger:
                    bo_dir = -1
                    triggered = True
                    break

            if triggered:
                entry_idx = j
                exit_idx = min(entry_idx + hold_minutes, len(day) - 1)
                bo_ret = float((bo_dir * rets[entry_idx:exit_idx + 1] * add_lev).sum())
                bo_ret -= cost(add_lev)
                day_pnl += bo_ret
                n_trades += 1
                i = exit_idx + 1
            else:
                i += 1

        daily_pnls.append(day_pnl)
        daily_trades.append(n_trades)

    return np.array(daily_pnls), np.array(daily_trades)


def print_metrics(name, pnls, baseline=None):
    n = len(pnls)
    total = pnls.sum()
    sr = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    hr = (pnls > 0).mean()
    cum = np.cumsum(pnls)
    mdd = float((cum - np.maximum.accumulate(cum)).min()) if len(cum) > 0 else 0
    annual = total / n * 252 if n > 0 else 0

    alpha_str = ""
    if baseline is not None:
        alpha = total - baseline.sum()
        alpha_str = f", Alpha {alpha*100:+.1f}%p"

    print(f"    {name:<35s}: Net {total*100:+7.1f}%{alpha_str}, "
          f"SR {sr:+.2f}, HR {hr*100:.0f}%, MDD {mdd*100:+.1f}%")


def main():
    vix = pd.read_parquet(config.BASE_DIR / "data" / "vix_slope_daily.parquet")
    dirs = get_directions(vix)

    for period, start, end, label in [
        ("IS (2024-01 ~ 2025-04)", "202401", "202504", "is"),
        ("OOS (2025-05 ~ 2026-04)", "202505", "202604", "oos"),
    ]:
        print(f"\n{'='*70}")
        print(f"  {period}")
        print(f"{'='*70}")

        btc = load_btc(start, end)
        ae = pd.read_parquet(config.ARTIFACT_DIR / f"anomaly_signals_{label}.parquet")

        # ─── 전략 A: 브레이크아웃 ───
        print(f"\n  [A] Slope 전략 + Anomaly 브레이크아웃 추가")
        print(f"  {'─'*60}")

        for bo_pct in [0.002, 0.003, 0.005]:
            for hold in [15, 30, 60]:
                for add_l in [1.0, 2.0]:
                    base, strat = strategy_a_breakout(
                        btc, ae, dirs, bo_pct, add_l, hold
                    )
                    alpha = strat.sum() - base.sum()
                    sr = strat.mean() / strat.std() * np.sqrt(252) if strat.std() > 0 else 0
                    # 간략 출력: alpha > 0인 것만
                    if alpha > 0:
                        print(f"    BO {bo_pct*100:.1f}% hold {hold:2d}m lev {add_l:.0f}x: "
                              f"Alpha {alpha*100:+.1f}%p, "
                              f"Net {strat.sum()*100:+.1f}%, SR {sr:+.2f}")

        # 주요 설정 상세 출력
        print(f"\n    --- 주요 설정 비교 ---")
        for bo_pct, hold, add_l in [
            (0.002, 15, 1.0), (0.003, 15, 1.0), (0.003, 30, 1.0),
            (0.005, 30, 1.0), (0.003, 15, 2.0), (0.005, 30, 2.0),
        ]:
            base, strat = strategy_a_breakout(btc, ae, dirs, bo_pct, add_l, hold)
            print_metrics(
                f"BO {bo_pct*100:.1f}% h{hold}m {add_l:.0f}x",
                strat, base
            )

        # baseline
        base, _ = strategy_a_breakout(btc, ae, dirs, 0.003, 1.0, 15)
        print_metrics("Baseline (slope only)", base)

        # ─── 전략 B: Vol 타겟팅 ───
        print(f"\n  [B] Anomaly 기반 Vol 타겟팅")
        print(f"  {'─'*60}")

        base_ref, _ = strategy_a_breakout(btc, ae, dirs, 0.003, 1.0, 15)

        for target_vol in [10, 13, 15, 18, 20]:
            for lookback in [30, 60]:
                for max_l in [3.0, 4.0]:
                    _, strat = strategy_b_vol_target(
                        btc, ae, dirs, target_vol, lookback, max_l
                    )
                    print_metrics(
                        f"tgt {target_vol}bp lb {lookback}m max {max_l:.0f}x",
                        strat, base_ref
                    )

        # ─── 전략 C: 순수 변동성 (slope 무관) ───
        print(f"\n  [C] 순수 Anomaly 브레이크아웃 (slope 무관, 전 거래일)")
        print(f"  {'─'*60}")

        for bo_pct in [0.002, 0.003, 0.005, 0.008]:
            for hold in [10, 15, 30, 60]:
                for add_l in [1.0, 2.0]:
                    pnls, trades = strategy_c_pure_vol(
                        btc, ae, bo_pct, add_l, hold
                    )
                    total_trades = trades.sum()
                    if total_trades > 0:
                        sr = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
                        trading_days = (trades > 0).sum()
                        print(f"    BO {bo_pct*100:.1f}% h{hold:2d}m {add_l:.0f}x: "
                              f"Net {pnls.sum()*100:+7.1f}%, SR {sr:+.2f}, "
                              f"거래일 {trading_days}, 총거래 {int(total_trades)}")


if __name__ == "__main__":
    main()
