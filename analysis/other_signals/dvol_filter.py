"""
DVOL Risk Filter — BTC 포지션 관리 도구
========================================
용도: BTC 롱 포지션을 잡고 있을 때, DVOL 기반으로 
      "유지(LONG)" vs "회피(FLAT)" 판단.

근거:
  - DVOL(Deribit BTC IV) > expanding Q60 → 롱 유지
  - DVOL < Q60 → flat (현금)
  - IS 검증: Sharpe 1.70, MDD -16.9% (B&H -48.5%), p=0.0074
  - shift(2) 적용으로 look-ahead bias 없음

사용법:
  python dvol_filter.py           # 현재 시그널 확인
  python dvol_filter.py --history # 최근 30일 시그널 이력
"""
import argparse
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests


def fetch_dvol(days=400):
    """Deribit DVOL 일봉 데이터 가져오기."""
    end_ts = int(datetime.utcnow().timestamp() * 1000)
    start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    params = {
        "currency": "BTC",
        "resolution": "1D",
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
    }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()["result"]["data"]

    # [timestamp_ms, open, high, low, close]
    dates = [pd.Timestamp(row[0], unit="ms").normalize() for row in data]
    closes = [row[4] for row in data]

    dvol = pd.Series(closes, index=pd.DatetimeIndex(dates), name="dvol")
    dvol = dvol[~dvol.index.duplicated(keep="last")]
    return dvol


def compute_signal(dvol, quantile=0.6, min_periods=30):
    """
    시그널 계산.
    
    Returns:
        signal: Series of 'LONG' or 'FLAT'
        details: dict with current state info
    """
    q_threshold = dvol.expanding(min_periods=min_periods).quantile(quantile)
    dvol_chg2 = dvol.pct_change(2)

    # shift(2): T-2 시점 DVOL로 T 시점 판단
    above_q = (dvol > q_threshold).shift(2)
    rising = (dvol_chg2 > 0).shift(2)

    # 전략 E: DVOL > Q60 (가장 높은 Sharpe, 가장 낮은 MDD)
    signal_e = above_q.map({True: "LONG", False: "FLAT"})

    # 전략 D: DVOL high AND rising (가장 낮은 p-value)
    signal_d = (above_q & rising).map({True: "LONG", False: "FLAT"})

    # 현재 상태 정보
    latest_dvol = dvol.iloc[-1]
    latest_date = dvol.index[-1]
    current_q60 = q_threshold.iloc[-1]
    current_chg2 = dvol_chg2.iloc[-1]

    details = {
        "date": str(latest_date.date()),
        "dvol": round(float(latest_dvol), 2),
        "q60_threshold": round(float(current_q60), 2),
        "dvol_2d_change": f"{current_chg2:+.2%}" if not np.isnan(current_chg2) else "N/A",
        "above_q60": bool(latest_dvol > current_q60),
        "rising_2d": bool(current_chg2 > 0) if not np.isnan(current_chg2) else None,
        "signal_conservative": signal_e.iloc[-1],  # E: Q60만
        "signal_strict": signal_d.iloc[-1],         # D: Q60 + rising
        "note": "shift(2) 적용 — 오늘 시그널은 2일 전 DVOL 기준",
    }

    return signal_e, signal_d, details


def print_current(details):
    """현재 시그널 출력."""
    print("=" * 55)
    print("  DVOL Risk Filter — BTC Position Signal")
    print("=" * 55)
    print(f"  Date:           {details['date']}")
    print(f"  DVOL:           {details['dvol']}")
    print(f"  Q60 Threshold:  {details['q60_threshold']}")
    print(f"  2d Change:      {details['dvol_2d_change']}")
    print(f"  Above Q60:      {'✅' if details['above_q60'] else '❌'}")
    print(f"  Rising 2d:      {'✅' if details['rising_2d'] else '❌'}")
    print()
    
    sig_e = details["signal_conservative"]
    sig_d = details["signal_strict"]
    
    if sig_e == "LONG":
        print(f"  📈 Conservative (E): LONG")
    else:
        print(f"  💤 Conservative (E): FLAT")
        
    if sig_d == "LONG":
        print(f"  📈 Strict (D):       LONG")
    else:
        print(f"  💤 Strict (D):       FLAT")
    
    print()
    print("  Conservative: DVOL > Q60 → LONG (MDD -17%)")
    print("  Strict:       DVOL > Q60 AND rising → LONG (p=0.011)")
    print("=" * 55)


def print_history(dvol, signal_e, signal_d, n=30):
    """최근 n일 시그널 이력."""
    print(f"\n  Recent {n}-day history:")
    print(f"  {'Date':>12s}  {'DVOL':>7s}  {'Cons':>6s}  {'Strict':>6s}")
    print("  " + "-" * 38)
    for i in range(-min(n, len(dvol)), 0):
        dt = dvol.index[i]
        dv = dvol.iloc[i]
        se = signal_e.iloc[i] if i < len(signal_e) and pd.notna(signal_e.iloc[i]) else "N/A"
        sd = signal_d.iloc[i] if i < len(signal_d) and pd.notna(signal_d.iloc[i]) else "N/A"
        marker_e = "📈" if se == "LONG" else "💤" if se == "FLAT" else "  "
        marker_d = "📈" if sd == "LONG" else "💤" if sd == "FLAT" else "  "
        print(f"  {dt.date()!s:>12s}  {dv:7.1f}  {marker_e} {se:>4s}  {marker_d} {sd:>4s}")


def main():
    parser = argparse.ArgumentParser(description="DVOL Risk Filter")
    parser.add_argument("--history", action="store_true", help="최근 30일 시그널 이력")
    parser.add_argument("--days", type=int, default=30, help="이력 일수 (default: 30)")
    parser.add_argument("--json", action="store_true", help="JSON 출력 (자동화용)")
    args = parser.parse_args()

    dvol = fetch_dvol(days=400)
    signal_e, signal_d, details = compute_signal(dvol)

    if args.json:
        print(json.dumps(details, indent=2))
        return

    print_current(details)

    if args.history:
        print_history(dvol, signal_e, signal_d, n=args.days)


if __name__ == "__main__":
    main()
