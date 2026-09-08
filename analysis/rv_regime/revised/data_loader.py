"""
data_loader.py — BTC 1분봉 로드 및 거래일 기준 RV/수익률 계산

거래일 정의: (T-1) 17:00 ET ~ T 16:59 ET
  → 17:00 이후 봉은 다음 거래일로 분류
  → RV[T]는 T 17:00에 확정 가능 (마지막 봉 = T 16:59)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from rv_regime.config import BTC_1M_DIR, DATA_DIR, TRADE_DAY_CUTOFF_HOUR


def load_btc_1m(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    BTC 24시간 1분봉 parquet 로드.

    Returns
    -------
    df : pd.DataFrame
        columns=['open','high','low','close','volume'],
        index=DatetimeIndex (America/New_York, tz-aware)
    """
    base_dir = Path(base_dir or BTC_1M_DIR)
    files = sorted(base_dir.glob("btc_1m_*.parquet"))
    if not files:
        raise FileNotFoundError(f"parquet 파일 없음: {base_dir}")

    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


def assign_trade_date(df: pd.DataFrame) -> pd.Series:
    """
    각 1분봉에 거래일 라벨을 부여.

    규칙:
      - 17:00 이전 봉 → 해당 날짜가 거래일
      - 17:00 이후 봉 → 다음 날짜가 거래일

    Returns
    -------
    trade_date : pd.Series (같은 인덱스, 값=tz-aware Timestamp)
    """
    idx = df.index
    trade_date = pd.Series(idx.normalize(), index=idx)
    mask_after_cutoff = idx.hour >= TRADE_DAY_CUTOFF_HOUR
    trade_date[mask_after_cutoff] = (
        trade_date[mask_after_cutoff] + pd.Timedelta(days=1)
    )
    return trade_date


def compute_daily_rv(df_1m: pd.DataFrame, trade_date: pd.Series) -> pd.Series:
    """
    거래일별 Realized Volatility 계산.

    RV[T] = sqrt(sum(r_1m²))  where r_1m = log(close/close_prev)
    거래일 T의 봉: (T-1) 17:00 ~ T 16:59

    Returns
    -------
    rv : pd.Series, index=DatetimeIndex, name='rv'
    """
    log_ret = np.log(df_1m['close'] / df_1m['close'].shift(1))
    rv = log_ret.groupby(trade_date).apply(
        lambda g: np.sqrt((g ** 2).sum())
    )
    rv.index = pd.to_datetime(rv.index)
    rv.name = 'rv'
    rv = rv.sort_index()
    rv = rv[rv > 0]  # 거래 없는 날 제거
    return rv


def compute_daily_ret(df_1m: pd.DataFrame, trade_date: pd.Series) -> pd.Series:
    """
    거래일별 수익률 계산 (close-to-close).

    daily_ret[T] = close[T 16:59] / close[(T-1) 16:59] - 1
    = (T-1) 17:00 ~ T 17:00 구간 수익률

    Returns
    -------
    ret : pd.Series, index=DatetimeIndex, name='ret'
    """
    close_5pm = df_1m.groupby(trade_date)['close'].last()
    close_5pm.index = pd.to_datetime(close_5pm.index)
    close_5pm = close_5pm.sort_index()
    ret = close_5pm.pct_change()
    ret.name = 'ret'
    return ret


def load_daily_funding() -> pd.Series:
    """
    Binance BTCUSDT 펀딩레이트를 거래일별로 집계.

    펀딩은 8시간마다 3회(UTC 00:00, 08:00, 16:00) 정산.
    거래일 기준(17:00 ET cutoff)으로 합산하여 일별 펀딩레이트 반환.

    Returns
    -------
    daily_fr : pd.Series (tz-aware, America/New_York)
    """
    fr_path = DATA_DIR / "funding_rate_history.parquet"
    if not fr_path.exists():
        raise FileNotFoundError(f"펀딩레이트 파일 없음: {fr_path}")

    fr = pd.read_parquet(fr_path)
    fr_et = fr.copy()
    fr_et.index = fr_et.index.tz_convert('America/New_York')

    trade_date = fr_et.index.normalize()
    mask_after = fr_et.index.hour >= TRADE_DAY_CUTOFF_HOUR
    td_series = pd.Series(trade_date, index=fr_et.index)
    td_series[mask_after] = td_series[mask_after] + pd.Timedelta(days=1)

    daily_fr = fr_et['funding_rate'].groupby(td_series).sum()
    daily_fr.index = pd.to_datetime(daily_fr.index)
    daily_fr.name = 'daily_funding_rate'
    daily_fr = daily_fr.sort_index()
    return daily_fr


def load_and_prepare() -> tuple:
    """
    전체 파이프라인용 데이터 준비.

    Returns
    -------
    rv_daily : pd.Series — 거래일별 RV
    daily_ret : pd.Series — 거래일별 수익률
    """
    print("[data_loader] BTC 1분봉 로드 중...")
    df_1m = load_btc_1m()
    print(f"  {len(df_1m):,}행, {df_1m.index[0]} ~ {df_1m.index[-1]}")

    trade_date = assign_trade_date(df_1m)
    rv_daily = compute_daily_rv(df_1m, trade_date)
    daily_ret = compute_daily_ret(df_1m, trade_date)

    print(f"  RV: {len(rv_daily)}일, 수익률: {len(daily_ret)}일")
    return rv_daily, daily_ret
