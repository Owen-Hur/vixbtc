"""
BTC 선물 1분봉 확장 다운로드 (data.binance.vision)
2020-01 ~ 2023-12 구간을 btc_1m_24h/ 포맷에 맞춰 저장

사용법:
  python data/download_klines_extended.py
"""
import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m"
OUT_DIR = Path(__file__).parent / "btc_1m_24h"
TMP_DIR = Path(__file__).parent / "_tmp_klines"

START_YEAR, START_MONTH = 2020, 1
END_YEAR, END_MONTH = 2023, 12


def get_months():
    months = []
    y, m = START_YEAR, START_MONTH
    while (y, m) <= (END_YEAR, END_MONTH):
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def download_and_extract(month_str):
    fname = f"BTCUSDT-1m-{month_str}.zip"
    url = f"{BASE_URL}/{fname}"
    zip_path = TMP_DIR / fname
    csv_name = fname.replace(".zip", ".csv")
    csv_path = TMP_DIR / csv_name

    if csv_path.exists():
        return csv_path

    print(f"  downloading {fname}...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, str(zip_path))
        size = os.path.getsize(zip_path) / 1e6
        print(f" {size:.1f}MB", end="")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(TMP_DIR))
        os.remove(zip_path)
        print(" ok")
        return csv_path
    except Exception as e:
        print(f" FAILED: {e}")
        for p in [zip_path, csv_path]:
            if p.exists():
                os.remove(p)
        return None


def process_to_parquet(csv_path, month_str):
    out_path = OUT_DIR / f"btc_1m_{month_str}.parquet"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return out_path

    no_header_cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ]
    first_val = open(csv_path).readline().split(",")[0].strip()
    if first_val.isdigit():
        df = pd.read_csv(csv_path, header=None, names=no_header_cols)
    else:
        df = pd.read_csv(csv_path)
        df.columns = [c if c != "count" else "trade_count" for c in df.columns]

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    df.set_index("timestamp", inplace=True)

    total_count = df["trade_count"]
    taker_buy_vol = df["taker_buy_volume"]
    taker_sell_vol = df["volume"] - taker_buy_vol

    out = pd.DataFrame({
        "open": df["open"].astype("float64"),
        "high": df["high"].astype("float64"),
        "low": df["low"].astype("float64"),
        "close": df["close"].astype("float64"),
        "volume": df["volume"].astype("float64"),
        "quote_volume": df["quote_volume"].astype("float64"),
        "trade_count": df["trade_count"].astype("int64"),
        "avg_trade_size": np.where(
            total_count > 0,
            df["volume"] / total_count,
            0.0,
        ),
        "trade_imbalance": np.where(
            df["volume"] > 0,
            (taker_buy_vol - taker_sell_vol) / df["volume"],
            0.0,
        ),
    }, index=df.index)

    out.to_parquet(out_path, engine="pyarrow")
    print(f"  [saved] {out_path.name} — {len(out):,} rows")
    return out_path


def main():
    OUT_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    months = get_months()
    print(f"=== BTC 1m klines download: {months[0]} ~ {months[-1]} ({len(months)} months) ===\n")

    for i, m in enumerate(months, 1):
        print(f"[{i}/{len(months)}] {m}")
        csv_path = download_and_extract(m)
        if csv_path and csv_path.exists():
            process_to_parquet(csv_path, m)

    # cleanup tmp
    import shutil
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
        print(f"\ntmp cleaned up")

    total = sum(f.stat().st_size for f in OUT_DIR.glob("btc_1m_*.parquet"))
    count = len(list(OUT_DIR.glob("btc_1m_*.parquet")))
    print(f"\nDone! {count} files, {total/1e9:.2f} GB total")


if __name__ == "__main__":
    main()
