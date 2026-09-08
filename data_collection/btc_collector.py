"""
BTC 선물 실시간 데이터 수집기 (24시간)
- LOB (호가창): 1초 간격 스냅샷
- 체결 데이터:  틱 단위 WebSocket

설치:
  pip install aiohttp "websockets==10.4"

실행:
  python btc_collector.py

저장:
  ./data/lob/YYYY-MM-DD.csv     (LOB 호가창)
  ./data/trades/YYYY-MM-DD.csv  (체결 데이터)
"""

import asyncio
import json
import csv
import os
import time
from datetime import datetime, timezone
import aiohttp
import websockets

# websockets 버전에 따라 connect 함수 선택 (10.x 구 API / 11+ 새 API)
_ws_ver = tuple(int(x) for x in websockets.__version__.split(".")[:2])
if _ws_ver >= (11, 0):
    try:
        from websockets.asyncio.client import connect as _ws_connect
    except ImportError:
        from websockets.legacy.client import connect as _ws_connect
else:
    _ws_connect = websockets.connect

# ── 설정 ──────────────────────────────────────────────────
LOB_DEPTH        = 20
LOB_INTERVAL_SEC = 1
DATA_DIR         = "./data"

LOB_URL = f"https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit={LOB_DEPTH}"
WS_URL  = "wss://fstream.binance.com/ws/btcusdt@trade"

os.makedirs(f"{DATA_DIR}/lob",    exist_ok=True)
os.makedirs(f"{DATA_DIR}/trades", exist_ok=True)


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


# ── LOB 수집 ──────────────────────────────────────────────
async def collect_lob():
    print(f"[LOB] 시작 — {LOB_DEPTH}호가, {LOB_INTERVAL_SEC}초 간격")

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(
                    LOB_URL,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status != 200:
                        print(f"[LOB] HTTP {resp.status} — 재시도")
                        await asyncio.sleep(LOB_INTERVAL_SEC)
                        continue

                    data  = await resp.json()
                    ts_ms = int(time.time() * 1000)

                    bids = data.get("bids", [])[:LOB_DEPTH]
                    asks = data.get("asks", [])[:LOB_DEPTH]

                    # 호가 부족 시 빈값으로 패딩
                    while len(bids) < LOB_DEPTH: bids.append(["", ""])
                    while len(asks) < LOB_DEPTH: asks.append(["", ""])

                    filepath    = f"{DATA_DIR}/lob/{today()}.csv"
                    file_exists = os.path.exists(filepath)

                    with open(filepath, "a", newline="") as f:
                        w = csv.writer(f)
                        if not file_exists:
                            bid_h = [f"bid_p{i+1}" for i in range(LOB_DEPTH)] \
                                  + [f"bid_q{i+1}" for i in range(LOB_DEPTH)]
                            ask_h = [f"ask_p{i+1}" for i in range(LOB_DEPTH)] \
                                  + [f"ask_q{i+1}" for i in range(LOB_DEPTH)]
                            w.writerow(["timestamp_ms"] + bid_h + ask_h)
                            print(f"[LOB] 파일 생성: {filepath}")
                        row = [ts_ms] \
                            + [b[0] for b in bids] + [b[1] for b in bids] \
                            + [a[0] for a in asks] + [a[1] for a in asks]
                        w.writerow(row)

            except asyncio.TimeoutError:
                pass  # 타임아웃은 무시하고 재시도
            except Exception as e:
                print(f"[LOB] 오류: {type(e).__name__}: {e}")

            await asyncio.sleep(LOB_INTERVAL_SEC)


# ── 체결 데이터 수집 ───────────────────────────────────────
async def collect_trades():
    print(f"[TRADES] WebSocket 연결 중...")
    msg_count = 0

    while True:
        f = None
        try:
            # websockets 버전 자동 호환
            # ping_interval=None: Binance 서버는 자체 keepalive 사용,
            #   클라이언트 ping 보내면 pong 안 해줘서 강제 종료됨
            async with _ws_connect(
                WS_URL,
                ping_interval=None,
            ) as ws:
                print(f"[TRADES] 연결 완료")

                filepath    = f"{DATA_DIR}/trades/{today()}.csv"
                file_exists = os.path.exists(filepath)

                f = open(filepath, "a", newline="")
                w = csv.writer(f)

                if not file_exists:
                    w.writerow([
                        "timestamp_ms",
                        "price",
                        "quantity",
                        "is_buyer_maker",
                    ])
                    f.flush()
                    print(f"[TRADES] 파일 생성: {filepath}")

                async for msg in ws:
                    raw = json.loads(msg)
                    # 단일스트림: T,p,q,m 바로 접근
                    # 멀티스트림: data 키 안에 있음
                    d = raw.get("data", raw)

                    msg_count += 1
                    if msg_count <= 3:
                        print(f"[TRADES] 수신 #{msg_count}: "
                              f"price={d.get('p')}, qty={d.get('q')}")

                    # 날짜 바뀌면 새 파일로 전환
                    new_path = f"{DATA_DIR}/trades/{today()}.csv"
                    if new_path != filepath:
                        f.close()
                        f = None
                        print(f"[TRADES] 날짜 변경 → 새 파일")
                        break

                    w.writerow([d["T"], d["p"], d["q"], d["m"]])
                    f.flush()

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"[TRADES] 연결 끊김 (code={e.code}) — 5초 후 재연결")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[TRADES] 오류: {type(e).__name__}: {e} — 5초 후 재연결")
            await asyncio.sleep(5)
        finally:
            if f is not None:
                try:
                    f.close()
                except:
                    pass


# ── 상태 모니터 ────────────────────────────────────────────
async def monitor():
    while True:
        await asyncio.sleep(60)

        lob_path   = f"{DATA_DIR}/lob/{today()}.csv"
        trade_path = f"{DATA_DIR}/trades/{today()}.csv"

        lob_mb   = os.path.getsize(lob_path)   / 1e6 \
                   if os.path.exists(lob_path)   else 0
        trade_mb = os.path.getsize(trade_path) / 1e6 \
                   if os.path.exists(trade_path) else 0

        print(f"[{now_str()} UTC] LOB: {lob_mb:.1f}MB | Trades: {trade_mb:.2f}MB")


# ── 메인 ──────────────────────────────────────────────────
async def main():
    print("=" * 50)
    print(" BTC 선물 수집기 시작 (24시간)")
    print(f" LOB:    ./data/lob/YYYY-MM-DD.csv")
    print(f" Trades: ./data/trades/YYYY-MM-DD.csv")
    print(f" 종료:   Ctrl+C")
    print("=" * 50)

    await asyncio.gather(
        collect_lob(),
        collect_trades(),
        monitor(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n수집 종료")
