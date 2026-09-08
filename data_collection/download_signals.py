"""
Download supplementary signal data for BTC strategy:
1. Binance Funding Rate -> funding_rate_daily.parquet
2. Binance Metrics -> metrics_daily.parquet
3. Fear & Greed Index -> fear_greed_daily.parquet
4. Binance Liquidation Snapshot -> liquidation_daily.parquet
"""

import urllib.request
import zipfile
import io
import os
import json
import time
import csv
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://data.binance.vision/data/futures/um"

# Date ranges
START_DATE = date(2020, 1, 1)
END_DATE = date(2026, 5, 7)
MONTHLY_END = date(2026, 4, 1)  # last full month for monthly files
DAILY_START_EXTRA = date(2026, 5, 1)  # daily files for partial month
DAILY_END_EXTRA = date(2026, 5, 7)


def download_url(url, retries=2, timeout=30):
    """Download URL, return bytes or None on failure."""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                return None
    return None


def extract_csv_from_zip(data_bytes):
    """Extract first CSV from zip bytes, return list of rows."""
    try:
        with zipfile.ZipFile(io.BytesIO(data_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return None
            with zf.open(csv_names[0]) as f:
                text = f.read().decode("utf-8")
                reader = csv.reader(io.StringIO(text))
                rows = list(reader)
                return rows
    except Exception:
        return None


def generate_months(start, end):
    """Generate (year, month) tuples."""
    current = date(start.year, start.month, 1)
    end_m = date(end.year, end.month, 1)
    while current <= end_m:
        yield current.year, current.month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def generate_days(start, end):
    """Generate date objects."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def ts_to_datetime(ts):
    """Convert millisecond timestamp to datetime."""
    if pd.isna(ts):
        return pd.NaT
    if isinstance(ts, (int, float)):
        if ts > 1e12:
            return datetime.utcfromtimestamp(ts / 1000)
        else:
            return datetime.utcfromtimestamp(ts)
    return pd.NaT


def assign_trade_day(dt_utc):
    """
    CME trade day convention: day T = previous day 22:00 UTC to T 21:59:59 UTC
    (17:00 ET = 22:00 UTC for EST, 21:00 UTC for EDT, using 22:00 UTC as approximation)
    """
    if pd.isna(dt_utc):
        return pd.NaT
    # If before 22:00 UTC, it's the same calendar day's trade day
    # If at or after 22:00 UTC, it's the next calendar day's trade day
    if dt_utc.hour >= 22:
        return (dt_utc + timedelta(days=1)).date()
    else:
        return dt_utc.date()


# ============================================================
# 1. FUNDING RATE
# ============================================================
def download_funding_rate():
    print("=" * 60)
    print("1. DOWNLOADING FUNDING RATE")
    print("=" * 60)

    all_rows = []

    # Monthly files: 2020-01 to 2026-04
    months = list(generate_months(date(2020, 1, 1), MONTHLY_END))
    print(f"  Downloading {len(months)} monthly files...")
    header = None

    for i, (year, month) in enumerate(months):
        ym = f"{year}-{month:02d}"
        url = f"{BASE_URL}/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{ym}.zip"
        data = download_url(url)
        if data:
            rows = extract_csv_from_zip(data)
            if rows and len(rows) > 1:
                # Check if first row is header
                if header is None:
                    first = rows[0]
                    # If first row has non-numeric values that look like headers
                    try:
                        float(first[-1])
                        # no header
                        all_rows.extend(rows)
                    except (ValueError, IndexError):
                        header = first
                        all_rows.extend(rows[1:])
                else:
                    # Skip header row if present
                    if rows[0] == header:
                        all_rows.extend(rows[1:])
                    else:
                        try:
                            float(rows[0][-1])
                            all_rows.extend(rows)
                        except (ValueError, IndexError):
                            all_rows.extend(rows[1:])
                print(f"    {ym}: {len(rows)-1} records", end="\r")
            else:
                print(f"    {ym}: empty", end="\r")
        else:
            print(f"    {ym}: 404/fail", end="\r")

    print(f"\n  Monthly done. Total rows so far: {len(all_rows)}")

    # Daily files for 2026-05-01 to 2026-05-07
    print("  Downloading daily files for 2026-05...")
    for d in generate_days(DAILY_START_EXTRA, DAILY_END_EXTRA):
        ds = d.strftime("%Y-%m-%d")
        url = f"{BASE_URL}/daily/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{ds}.zip"
        data = download_url(url, retries=1, timeout=15)
        if data:
            rows = extract_csv_from_zip(data)
            if rows and len(rows) > 1:
                if header and rows[0] == header:
                    all_rows.extend(rows[1:])
                else:
                    try:
                        float(rows[0][-1])
                        all_rows.extend(rows)
                    except (ValueError, IndexError):
                        if header is None:
                            header = rows[0]
                        all_rows.extend(rows[1:])
                print(f"    {ds}: OK", end="\r")
        else:
            print(f"    {ds}: 404/fail", end="\r")

    print(f"\n  Total raw rows: {len(all_rows)}")
    if header:
        print(f"  Header: {header}")

    if not all_rows:
        print("  WARNING: No funding rate data downloaded!")
        return

    # Build DataFrame
    if header:
        df = pd.DataFrame(all_rows, columns=header)
    else:
        # Try to guess columns
        ncols = len(all_rows[0])
        if ncols >= 3:
            cols = ["symbol", "fundingRate", "fundingRateTimestamp"]
            if ncols > 3:
                cols += [f"col_{i}" for i in range(3, ncols)]
            df = pd.DataFrame(all_rows, columns=cols[:ncols])
        else:
            df = pd.DataFrame(all_rows)

    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample:\n{df.head(3)}")

    # Find timestamp column
    ts_col = None
    for c in df.columns:
        if "time" in c.lower() or "timestamp" in c.lower():
            ts_col = c
            break
    if ts_col is None:
        # Try last numeric column
        for c in df.columns:
            try:
                val = float(df[c].iloc[0])
                if val > 1e9:
                    ts_col = c
            except:
                pass

    # Find funding rate column
    fr_col = None
    for c in df.columns:
        if "fundingrate" in c.lower().replace("_", "") or "funding_rate" in c.lower():
            if "time" not in c.lower():
                fr_col = c
                break

    if ts_col is None or fr_col is None:
        print(f"  ERROR: Could not identify columns. ts_col={ts_col}, fr_col={fr_col}")
        # Try positional: assume col 1 = rate, col 2 = timestamp
        if len(df.columns) >= 3:
            fr_col = df.columns[1]
            ts_col = df.columns[2]
            print(f"  Falling back to positional: fr={fr_col}, ts={ts_col}")

    df["funding_rate"] = pd.to_numeric(df[fr_col], errors="coerce")
    df["timestamp_ms"] = pd.to_numeric(df[ts_col], errors="coerce")
    df["datetime_utc"] = df["timestamp_ms"].apply(ts_to_datetime)
    df["trade_day"] = df["datetime_utc"].apply(assign_trade_day)

    # Aggregate to daily mean
    daily = df.groupby("trade_day").agg(
        funding_rate_mean=("funding_rate", "mean"),
        funding_rate_sum=("funding_rate", "sum"),
        funding_rate_count=("funding_rate", "count"),
    ).reset_index()
    daily.rename(columns={"trade_day": "date"}, inplace=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    out_path = os.path.join(DATA_DIR, "funding_rate_daily.parquet")
    daily.to_parquet(out_path, index=False)
    print(f"  Saved {out_path}: {len(daily)} days, {daily['date'].min()} to {daily['date'].max()}")
    return daily


# ============================================================
# 2. METRICS
# ============================================================
def download_metrics():
    print("\n" + "=" * 60)
    print("2. DOWNLOADING METRICS")
    print("=" * 60)

    # Try monthly first
    all_rows = []
    header = None

    months = list(generate_months(START_DATE, MONTHLY_END))
    print(f"  Trying monthly files first ({len(months)} months)...")

    monthly_ok = 0
    monthly_fail = 0
    monthly_covered_months = set()

    for year, month in months:
        ym = f"{year}-{month:02d}"
        url = f"{BASE_URL}/monthly/metrics/BTCUSDT/BTCUSDT-metrics-{ym}.zip"
        data = download_url(url, retries=1, timeout=30)
        if data:
            rows = extract_csv_from_zip(data)
            if rows and len(rows) > 1:
                if header is None:
                    try:
                        float(rows[0][-1])
                        all_rows.extend(rows)
                    except (ValueError, IndexError):
                        header = rows[0]
                        all_rows.extend(rows[1:])
                else:
                    if rows[0] == header:
                        all_rows.extend(rows[1:])
                    else:
                        try:
                            float(rows[0][-1])
                            all_rows.extend(rows)
                        except (ValueError, IndexError):
                            all_rows.extend(rows[1:])
                monthly_ok += 1
                monthly_covered_months.add((year, month))
                print(f"    {ym}: {len(rows)-1} records", end="\r")
            else:
                monthly_fail += 1
        else:
            monthly_fail += 1

    print(f"\n  Monthly: {monthly_ok} OK, {monthly_fail} failed, {len(all_rows)} rows")

    # Fill gaps with daily downloads
    # Find which months we're missing
    all_months = set(months)
    missing_months = all_months - monthly_covered_months

    # Also need daily for partial month (2026-05)
    daily_dates = list(generate_days(DAILY_START_EXTRA, DAILY_END_EXTRA))

    # Add missing months' days
    for year, month in sorted(missing_months):
        m_start = date(year, month, 1)
        if month == 12:
            m_end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            m_end = date(year, month + 1, 1) - timedelta(days=1)
        m_end = min(m_end, END_DATE)
        daily_dates.extend(generate_days(m_start, m_end))

    daily_dates = sorted(set(daily_dates))

    if daily_dates:
        print(f"  Downloading {len(daily_dates)} daily files for gaps...")
        batch_ok = 0
        for i, d in enumerate(daily_dates):
            ds = d.strftime("%Y-%m-%d")
            url = f"{BASE_URL}/daily/metrics/BTCUSDT/BTCUSDT-metrics-{ds}.zip"
            data = download_url(url, retries=1, timeout=15)
            if data:
                rows = extract_csv_from_zip(data)
                if rows and len(rows) > 1:
                    if header is None:
                        try:
                            float(rows[0][-1])
                            all_rows.extend(rows)
                        except (ValueError, IndexError):
                            header = rows[0]
                            all_rows.extend(rows[1:])
                    else:
                        if rows[0] == header:
                            all_rows.extend(rows[1:])
                        else:
                            try:
                                float(rows[0][-1])
                                all_rows.extend(rows)
                            except (ValueError, IndexError):
                                all_rows.extend(rows[1:])
                    batch_ok += 1
            if (i + 1) % 50 == 0:
                print(f"    Progress: {i+1}/{len(daily_dates)} ({batch_ok} OK)", end="\r")

        print(f"\n  Daily gap-fill: {batch_ok} OK out of {len(daily_dates)}")

    print(f"  Total raw rows: {len(all_rows)}")
    if header:
        print(f"  Header: {header}")

    if not all_rows:
        print("  WARNING: No metrics data downloaded!")
        return

    if header:
        df = pd.DataFrame(all_rows, columns=header)
    else:
        df = pd.DataFrame(all_rows)

    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample:\n{df.head(3)}")

    # Convert numeric columns
    for c in df.columns:
        if c.lower() not in ("symbol", "create_time", "createtime"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Find timestamp column
    ts_col = None
    for c in df.columns:
        if "time" in c.lower():
            ts_col = c
            break

    if ts_col:
        # Check if timestamp is a datetime string or numeric ms
        sample_val = str(df[ts_col].iloc[0]).strip()
        if "-" in sample_val and ":" in sample_val:
            # It's a datetime string like "2020-09-01 00:00:00"
            df["datetime_utc"] = pd.to_datetime(df[ts_col], errors="coerce")
        else:
            df["timestamp_ms"] = pd.to_numeric(df[ts_col], errors="coerce")
            df["datetime_utc"] = df["timestamp_ms"].apply(ts_to_datetime)
        df["trade_day"] = df["datetime_utc"].apply(assign_trade_day)
    else:
        print("  WARNING: No timestamp column found, using row index")
        df["trade_day"] = pd.NaT

    # Keep useful columns
    keep_cols = []
    for c in df.columns:
        cl = c.lower().replace("_", "").replace(" ", "")
        if any(k in cl for k in ["openinterest", "longshort", "longratio", "shortratio",
                                   "takerbuy", "takersell", "buyvol", "sellvol",
                                   "longaccount", "shortaccount", "toptrader"]):
            keep_cols.append(c)

    if not keep_cols:
        # Keep all numeric columns
        keep_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64, float, int]
                     and c not in ("timestamp_ms",)]

    result_cols = ["trade_day"] + keep_cols
    daily = df[result_cols].copy()
    daily.rename(columns={"trade_day": "date"}, inplace=True)

    # If multiple rows per day, take mean
    if daily["date"].duplicated().any():
        daily = daily.groupby("date").mean().reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    out_path = os.path.join(DATA_DIR, "metrics_daily.parquet")
    daily.to_parquet(out_path, index=False)
    print(f"  Saved {out_path}: {len(daily)} days, {daily['date'].min()} to {daily['date'].max()}")
    return daily


# ============================================================
# 3. FEAR & GREED INDEX
# ============================================================
def download_fear_greed():
    print("\n" + "=" * 60)
    print("3. DOWNLOADING FEAR & GREED INDEX")
    print("=" * 60)

    url = "https://api.alternative.me/fng/?limit=0&format=json"
    print(f"  Fetching {url}")
    raw = download_url(url, retries=3, timeout=60)
    if raw is None:
        print("  ERROR: Failed to download Fear & Greed data!")
        return

    data = json.loads(raw)
    records = data.get("data", [])
    print(f"  Got {len(records)} records")

    if not records:
        print("  WARNING: No records in response!")
        return

    print(f"  Sample: {records[0]}")

    rows = []
    for r in records:
        rows.append({
            "date": datetime.utcfromtimestamp(int(r["timestamp"])).date(),
            "fear_greed_value": int(r["value"]),
            "fear_greed_class": r["value_classification"],
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    out_path = os.path.join(DATA_DIR, "fear_greed_daily.parquet")
    df.to_parquet(out_path, index=False)
    print(f"  Saved {out_path}: {len(df)} days, {df['date'].min()} to {df['date'].max()}")
    return df


# ============================================================
# 4. LIQUIDATION SNAPSHOT
# ============================================================
def download_liquidation():
    print("\n" + "=" * 60)
    print("4. DOWNLOADING LIQUIDATION SNAPSHOT")
    print("=" * 60)

    all_rows = []
    header = None

    # Try monthly first
    months = list(generate_months(START_DATE, MONTHLY_END))
    print(f"  Trying monthly files first ({len(months)} months)...")

    monthly_ok = 0
    monthly_covered = set()

    for year, month in months:
        ym = f"{year}-{month:02d}"
        url = f"{BASE_URL}/monthly/liquidationSnapshot/BTCUSDT/BTCUSDT-liquidationSnapshot-{ym}.zip"
        data = download_url(url, retries=1, timeout=30)
        if data:
            rows = extract_csv_from_zip(data)
            if rows and len(rows) > 1:
                if header is None:
                    try:
                        float(rows[0][-1])
                        all_rows.extend(rows)
                    except (ValueError, IndexError):
                        header = rows[0]
                        all_rows.extend(rows[1:])
                else:
                    if rows[0] == header:
                        all_rows.extend(rows[1:])
                    else:
                        try:
                            float(rows[0][-1])
                            all_rows.extend(rows)
                        except (ValueError, IndexError):
                            all_rows.extend(rows[1:])
                monthly_ok += 1
                monthly_covered.add((year, month))
                print(f"    {ym}: {len(rows)-1} records", end="\r")
        # else: silently skip

    print(f"\n  Monthly: {monthly_ok} OK, {len(all_rows)} rows")

    # Daily for gaps + partial month
    all_months_set = set(months)
    missing = all_months_set - monthly_covered
    daily_dates = list(generate_days(DAILY_START_EXTRA, DAILY_END_EXTRA))
    for year, month in sorted(missing):
        m_start = date(year, month, 1)
        if month == 12:
            m_end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            m_end = date(year, month + 1, 1) - timedelta(days=1)
        m_end = min(m_end, END_DATE)
        daily_dates.extend(generate_days(m_start, m_end))

    daily_dates = sorted(set(daily_dates))

    if daily_dates:
        print(f"  Downloading {len(daily_dates)} daily files for gaps...")
        batch_ok = 0
        for i, d in enumerate(daily_dates):
            ds = d.strftime("%Y-%m-%d")
            url = f"{BASE_URL}/daily/liquidationSnapshot/BTCUSDT/BTCUSDT-liquidationSnapshot-{ds}.zip"
            data = download_url(url, retries=1, timeout=15)
            if data:
                rows = extract_csv_from_zip(data)
                if rows and len(rows) > 1:
                    if header is None:
                        try:
                            float(rows[0][-1])
                            all_rows.extend(rows)
                        except (ValueError, IndexError):
                            header = rows[0]
                            all_rows.extend(rows[1:])
                    else:
                        if rows[0] == header:
                            all_rows.extend(rows[1:])
                        else:
                            try:
                                float(rows[0][-1])
                                all_rows.extend(rows)
                            except (ValueError, IndexError):
                                all_rows.extend(rows[1:])
                    batch_ok += 1
            if (i + 1) % 50 == 0:
                print(f"    Progress: {i+1}/{len(daily_dates)} ({batch_ok} OK)", end="\r")

        print(f"\n  Daily gap-fill: {batch_ok} OK out of {len(daily_dates)}")

    print(f"  Total raw rows: {len(all_rows)}")
    if header:
        print(f"  Header: {header}")

    if not all_rows:
        print("  WARNING: No liquidation data downloaded!")
        return

    if header:
        df = pd.DataFrame(all_rows, columns=header)
    else:
        df = pd.DataFrame(all_rows)

    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample:\n{df.head(3)}")

    # Find timestamp, side, quantity columns
    ts_col = None
    for c in df.columns:
        if "time" in c.lower():
            ts_col = c
            break

    side_col = None
    for c in df.columns:
        if "side" in c.lower():
            side_col = c
            break

    qty_col = None
    for c in df.columns:
        cl = c.lower()
        if "qty" in cl or "quantity" in cl or "original_quantity" in cl:
            qty_col = c
            break

    price_col = None
    for c in df.columns:
        cl = c.lower()
        if "price" in cl and "average" in cl:
            price_col = c
            break
    if price_col is None:
        for c in df.columns:
            if "price" in c.lower():
                price_col = c
                break

    print(f"  Identified: ts={ts_col}, side={side_col}, qty={qty_col}, price={price_col}")

    if ts_col:
        df["timestamp_ms"] = pd.to_numeric(df[ts_col], errors="coerce")
        df["datetime_utc"] = df["timestamp_ms"].apply(ts_to_datetime)
        df["trade_day"] = df["datetime_utc"].apply(assign_trade_day)

    if qty_col:
        df["qty"] = pd.to_numeric(df[qty_col], errors="coerce")
    if price_col:
        df["price"] = pd.to_numeric(df[price_col], errors="coerce")

    # Calculate volume = qty * price
    if "qty" in df.columns and "price" in df.columns:
        df["volume"] = df["qty"] * df["price"]
    elif "qty" in df.columns:
        df["volume"] = df["qty"]
    else:
        # Try to find a volume/notional column
        for c in df.columns:
            if "vol" in c.lower() or "notional" in c.lower():
                df["volume"] = pd.to_numeric(df[c], errors="coerce")
                break

    if "volume" not in df.columns:
        print("  WARNING: Could not determine volume column")
        return

    if side_col:
        # Determine long vs short liquidation
        # SELL side = long liquidation (longs getting liquidated)
        # BUY side = short liquidation (shorts getting liquidated)
        df["side_norm"] = df[side_col].astype(str).str.upper().str.strip()
        df["long_liq_vol"] = np.where(df["side_norm"] == "SELL", df["volume"], 0)
        df["short_liq_vol"] = np.where(df["side_norm"] == "BUY", df["volume"], 0)
    else:
        df["long_liq_vol"] = df["volume"]
        df["short_liq_vol"] = 0

    daily = df.groupby("trade_day").agg(
        long_liq_volume=("long_liq_vol", "sum"),
        short_liq_volume=("short_liq_vol", "sum"),
        liq_count=("volume", "count"),
    ).reset_index()
    daily["net_liq"] = daily["long_liq_volume"] - daily["short_liq_volume"]
    daily.rename(columns={"trade_day": "date"}, inplace=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    out_path = os.path.join(DATA_DIR, "liquidation_daily.parquet")
    daily.to_parquet(out_path, index=False)
    print(f"  Saved {out_path}: {len(daily)} days, {daily['date'].min()} to {daily['date'].max()}")
    return daily


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import sys

    print(f"Data directory: {DATA_DIR}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print()

    t0 = time.time()

    skip_existing = "--skip-existing" in sys.argv

    # Download in order of importance/speed
    fg_path = os.path.join(DATA_DIR, "fear_greed_daily.parquet")
    fr_path = os.path.join(DATA_DIR, "funding_rate_daily.parquet")

    if skip_existing and os.path.exists(fg_path):
        print("Skipping Fear & Greed (already exists)")
        fg = pd.read_parquet(fg_path)
    else:
        fg = download_fear_greed()

    if skip_existing and os.path.exists(fr_path):
        print("Skipping Funding Rate (already exists)")
        fr = pd.read_parquet(fr_path)
    else:
        fr = download_funding_rate()

    met = download_metrics()
    liq = download_liquidation()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    for name, df in [("Fear & Greed", fg), ("Funding Rate", fr),
                     ("Metrics", met), ("Liquidation", liq)]:
        if df is not None and len(df) > 0:
            print(f"  {name}: {len(df)} days ({df['date'].min().date()} to {df['date'].max().date()})")
        else:
            print(f"  {name}: FAILED or empty")

    print("\nDone!")
