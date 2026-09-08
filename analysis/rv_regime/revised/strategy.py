"""
strategy.py — RV Regime 전략 핵심 로직

전략:
  1. rv_ratio = RV_short / RV_long (rolling mean 비율)
  2. expanding percentile로 Expansion 판정 (과거 데이터만 사용)
  3. Expansion → Short(-1), Non-Expansion → Long(+1)
  4. T 17:00 판단 → T+1 수익에 적용 (shift 1회)

Lookahead bias 방지:
  - RV[T]: T 16:59까지의 데이터 → T 17:00에 확정 (shift 불필요)
  - Expanding percentile: rv_ratio[:i+1]만 사용 (미래 배제)
  - Position: shift(1)로 T 판단 → T+1 적용
"""
import numpy as np
import pandas as pd

from rv_regime.config import SW, LW, QH, TAKER_FEE


def compute_rv_ratio(
    rv_daily: pd.Series,
    sw: int = SW,
    lw: int = LW,
) -> pd.Series:
    """
    RV ratio = RV_short_window / RV_long_window.

    Parameters
    ----------
    rv_daily : 거래일별 RV
    sw : 단기 rolling window
    lw : 장기 rolling window

    Returns
    -------
    rv_ratio : pd.Series (NaN 제거 후)
    """
    rv_s = rv_daily.rolling(sw, min_periods=sw).mean()
    rv_l = rv_daily.rolling(lw, min_periods=lw).mean()
    rv_ratio = (rv_s / rv_l).dropna()
    rv_ratio.name = 'rv_ratio'
    return rv_ratio


def compute_expanding_threshold(
    rv_ratio: pd.Series,
    qh: float = QH,
) -> pd.Series:
    """
    Expanding window percentile threshold (bias-free).

    threshold[i] = np.percentile(rv_ratio[0:i+1], qh * 100)
    → 자기 자신까지만 포함, 미래 데이터 미사용.

    Parameters
    ----------
    rv_ratio : rv_ratio 시계열
    qh : 백분위 임계값 (0~1)

    Returns
    -------
    thresholds : pd.Series (같은 인덱스)
    """
    values = rv_ratio.values
    n = len(values)
    th = np.empty(n)
    for i in range(n):
        if i == 0:
            th[i] = np.inf
        else:
            th[i] = np.percentile(values[:i], qh * 100)
    return pd.Series(th, index=rv_ratio.index, name='threshold')


def generate_positions(
    rv_ratio: pd.Series,
    thresholds: pd.Series,
) -> pd.Series:
    """
    포지션 생성.

    규칙:
      - rv_ratio > threshold → Expansion → Short(-1)
      - else → Non-Expansion → Long(+1)
      - shift(1): T 판단 → T+1 적용

    Returns
    -------
    pos_applied : pd.Series (index=rv_ratio.index)
        T+1에 적용될 포지션. pos_applied[0] = 0 (첫날 미정).
    """
    is_expansion = rv_ratio > thresholds
    pos_decision = pd.Series(
        np.where(is_expansion, -1.0, 1.0),
        index=rv_ratio.index,
        name='pos_decision',
    )
    # T 판단 → T+1 적용
    pos_applied = pos_decision.shift(1)
    pos_applied.iloc[0] = 0.0
    pos_applied.name = 'pos_applied'
    return pos_applied


def compute_strategy_returns(
    pos_applied: pd.Series,
    daily_ret: pd.Series,
    taker_fee: float = TAKER_FEE,
    daily_funding: pd.Series = None,
) -> tuple:
    """
    전략 수익률 및 거래비용 계산.

    strat_ret[T] = pos_applied[T] × daily_ret[T] - tx_cost[T] - pos_applied[T] × funding[T]
    tx_cost[T] = |pos변화[T]| × taker_fee
    funding: Long이면 양수 펀딩 시 비용, Short이면 양수 펀딩 시 수익

    Returns
    -------
    strat_ret : pd.Series — 전략 일일 수익률
    bnh_ret : pd.Series — Buy & Hold 일일 수익률
    cost : pd.Series — 일일 거래비용 (tx + funding)
    """
    common = pos_applied.index.intersection(daily_ret.index)
    pos = pos_applied.reindex(common)
    ret = daily_ret.reindex(common)

    tx_cost = pos.diff().abs() * taker_fee
    tx_cost.iloc[0] = abs(pos.iloc[0]) * taker_fee

    if daily_funding is not None:
        fr = daily_funding.reindex(common).fillna(0)
        funding_cost = pos * fr
        bnh_funding = fr.copy()
    else:
        funding_cost = pd.Series(0.0, index=common)
        bnh_funding = pd.Series(0.0, index=common)

    strat_ret = pos * ret - tx_cost - funding_cost
    strat_ret.name = 'strat_ret'

    bnh_ret = ret - bnh_funding
    bnh_ret.name = 'bnh_ret'

    cost = tx_cost + funding_cost.abs()
    cost.name = 'cost'

    return strat_ret, bnh_ret, cost


def run_strategy(
    rv_daily: pd.Series,
    daily_ret: pd.Series,
    sw: int = SW,
    lw: int = LW,
    qh: float = QH,
    taker_fee: float = TAKER_FEE,
    daily_funding: pd.Series = None,
) -> dict:
    """
    전체 전략 파이프라인 실행.

    Parameters
    ----------
    rv_daily : 거래일별 RV
    daily_ret : 거래일별 수익률
    sw, lw, qh : 전략 파라미터
    taker_fee : 거래 비용

    Returns
    -------
    result : dict
        rv_ratio, thresholds, is_expansion, pos_applied,
        strat_ret, bnh_ret, cost, common_index
    """
    rv_ratio = compute_rv_ratio(rv_daily, sw, lw)
    thresholds = compute_expanding_threshold(rv_ratio, qh)
    is_expansion = rv_ratio > thresholds
    pos_applied = generate_positions(rv_ratio, thresholds)
    strat_ret, bnh_ret, cost = compute_strategy_returns(
        pos_applied, daily_ret, taker_fee, daily_funding
    )

    common = strat_ret.index

    return {
        'rv_ratio': rv_ratio,
        'thresholds': thresholds,
        'is_expansion': is_expansion,
        'pos_applied': pos_applied.reindex(common),
        'strat_ret': strat_ret,
        'bnh_ret': bnh_ret,
        'cost': cost,
        'common_index': common,
        'params': {'sw': sw, 'lw': lw, 'qh': qh},
    }
