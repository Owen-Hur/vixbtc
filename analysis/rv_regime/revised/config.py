"""
config.py — RV Regime(편향 제거판) 실행 설정

⚠️ 이 파일은 **원본 저장소에 커밋되어 있지 않아 2026-09-08 재현 검증 때 복원한 것** 입니다.
   값의 출처는 아래와 같으며, 이 설정으로 `python -m rv_regime.walk_forward` 를 실행하면
   문서에 기록된 결과(-57.4%, Sharpe 0.02, MDD -84.7%, B&H +51.6%, 초과 -109.0%p, 22라운드)가
   소수점까지 재현됩니다.

   - TAKER_FEE = 0.0004 (taker 0.04% 편도)
       ← docs/strategy_journal.md "비용: Taker fee 0.04% (편도)"
   - SW, LW, QH = 3, 75, 0.80
       ← analysis/rv_regime/results/backtest_result.txt 헤더 "params: sw=3, lw=75, qh=0.8"
         (walk_forward 는 라운드마다 파라미터를 재탐색하므로 이 기본값은 결과에 영향을 주지 않습니다)
   - IS_END = "2022-12-31"
       ← 동 파일 "IS (1028일, 2020-03-14 ~ 2022-12-31)"
   - TRADE_DAY_CUTOFF_HOUR = 17
       ← data_loader.py docstring "거래일 정의: (T-1) 17:00 ET ~ T 16:59 ET"

사용법:
    이 디렉토리(revised/)를 `rv_regime` 이라는 이름의 패키지로 두고 (또는 심볼릭 링크),
    저장소 루트에서 `python -m rv_regime.walk_forward` 를 실행합니다.
    `__init__.py` 가 없다면 빈 파일로 하나 만들어 주십시오.
"""

from pathlib import Path

# 저장소 루트 기준 경로 (revised/ → rv_regime/ → analysis/ → repo root)
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
BTC_1M_DIR = DATA_DIR / "btc_1m_24h"

# 거래일 정의: (T-1) 17:00 ET ~ T 16:59 ET
TRADE_DAY_CUTOFF_HOUR = 17

# Binance 무기한 선물 taker 수수료 (편도)
TAKER_FEE = 0.0004

# 전략 기본 파라미터 (walk_forward 는 라운드별로 재탐색)
SW, LW, QH = 3, 75, 0.80

# IS/OOS 경계 (고정 파라미터 검증용)
IS_END = "2022-12-31"
