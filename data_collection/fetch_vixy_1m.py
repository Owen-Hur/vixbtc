"""
VIXY 1분봉 수집 파이프라인 (Alpaca Markets)
=============================================
- Alpaca 무료 계좌 API 키 필요 (paper trading 계좌 OK)
- IS 구간 전체 1분봉을 월 단위로 분할 수집 후 Parquet 저장
- 재실행 시 이미 수집된 월은 스킵 (멱등성 보장)

API 키 발급: https://app.alpaca.markets/paper/dashboard/overview
"""

import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

# ── Alpaca SDK ──────────────────────────────────────────────────────────────
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    raise ImportError("alpaca-py 패키지가 필요합니다: pip install alpaca-py")

# ── 설정 ────────────────────────────────────────────────────────────────────
SYMBOL      = "VIXY"
# constitution.md §2-6 확정 구간
# 전체: 2024-01-15 ~ 2026-04-30
# IS  : 2024-01-15 ~ 2025-10-31
# OOS : 2025-11-01 ~ 2026-04-30
DATA_START  = date(2024, 1, 15)    # 전체 구간 시작 (IS 시작)
DATA_END    = date(2026, 4, 30)    # 전체 구간 종료 (OOS 종료)
OUT_DIR     = Path(__file__).parent.parent / "data" / "vixy_1m"

# API 키: 환경변수 또는 아래 직접 입력
API_KEY     = os.getenv("ALPACA_API_KEY", "여기에_API_KEY_입력")
API_SECRET  = os.getenv("ALPACA_SECRET_KEY", "여기에_SECRET_KEY_입력")


def month_range(start: date, end: date):
    """start ~ end 사이의 월 시작일 리스트 반환."""
    cur = start.replace(day=1)
    while cur <= end:
        yield cur
        cur += relativedelta(months=1)


def fetch_month(client: StockHistoricalDataClient,
                symbol: str,
                year: int,
                month: int) -> pd.DataFrame:
    """특정 월의 1분봉 데이터를 가져옴."""
    start_dt = datetime(year, month, 1)
    end_dt   = (start_dt + relativedelta(months=1))  # 다음달 1일 = 이번달 마지막

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        feed="iex",          # 무료 계좌는 'iex' 사용 (sip는 유료)
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    if df.empty:
        return df

    # MultiIndex(symbol, timestamp) → 단일 인덱스
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    return df


def save_month(df: pd.DataFrame, out_dir: Path, year: int, month: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{year:04d}{month:02d}.parquet"
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    return path


def load_existing() -> set:
    """이미 수집된 year-month 집합 반환."""
    if not OUT_DIR.exists():
        return set()
    return {
        (int(p.stem[:4]), int(p.stem[4:6]))
        for p in OUT_DIR.glob("*.parquet")
        if len(p.stem) == 6 and p.stem.isdigit()
    }


def concat_all(out_dir: Path) -> pd.DataFrame:
    """수집된 모든 parquet 파일을 합쳐 단일 DataFrame 반환."""
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files]).sort_index()


def main():
    # API 키 확인
    if "입력" in API_KEY or "입력" in API_SECRET:
        print("❌ API 키를 설정하세요:")
        print("   1) fetch_vixy_1m.py 파일에서 API_KEY / API_SECRET 직접 입력")
        print("   2) 또는 환경변수: export ALPACA_API_KEY=... && export ALPACA_SECRET_KEY=...")
        print()
        print("   API 키 발급: https://app.alpaca.markets/paper/dashboard/overview")
        return

    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    existing = load_existing()

    months = list(month_range(DATA_START, DATA_END))
    pending = [(m.year, m.month) for m in months
               if (m.year, m.month) not in existing]

    print(f"[VIXY 1분봉 수집]")
    print(f"  대상 기간 : {DATA_START} ~ {DATA_END} (IS+OOS 전체)")
    print(f"  저장 경로 : {OUT_DIR}")
    print(f"  전체 월수 : {len(months)}개월")
    print(f"  기수집    : {len(existing)}개월 (스킵)")
    print(f"  수집 예정 : {len(pending)}개월")
    print()

    if not pending:
        print("✅ 모든 월 데이터가 이미 수집되어 있습니다.")
        _show_summary()
        return

    failed = []
    for year, month in tqdm(pending, desc="월별 수집"):
        try:
            df = fetch_month(client, SYMBOL, year, month)
            if df.empty:
                tqdm.write(f"  ⚠️  {year}-{month:02d}: 데이터 없음 (거래일 없음?)")
            else:
                path = save_month(df, OUT_DIR, year, month)
                tqdm.write(f"  ✅ {year}-{month:02d}: {len(df):,}봉 → {path.name}")
        except Exception as e:
            tqdm.write(f"  ❌ {year}-{month:02d}: {e}")
            failed.append((year, month))
        time.sleep(0.3)  # rate limit 방지

    print()
    print("─" * 50)
    print(f"완료: {len(pending) - len(failed)}개월 성공, {len(failed)}개월 실패")
    if failed:
        print(f"실패 목록: {failed}")

    _show_summary()


def _show_summary():
    df_all = concat_all(OUT_DIR)
    if df_all.empty:
        print("수집된 데이터 없음.")
        return
    print(f"\n[수집 데이터 요약]")
    print(f"  총 봉 수  : {len(df_all):,}")
    print(f"  기간      : {df_all.index[0]} ~ {df_all.index[-1]}")
    print(f"  컬럼      : {list(df_all.columns)}")
    print(f"  저장 위치 : {OUT_DIR}")
    print()
    print(df_all.tail(3))


if __name__ == "__main__":
    main()
