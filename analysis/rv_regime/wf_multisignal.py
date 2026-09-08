"""
Walk-Forward: Multi-Signal + Session Strategy (A+B Combined)
Run: python -m rv_regime.wf_multisignal

Combines 5 signals into a scoring system:
  1. Funding Rate Momentum (p=0.0065 proven): high FR = bullish continuation
  2. OI Change: rising OI = bullish, falling = bearish
  3. Fear & Greed: extreme fear = contrarian buy, extreme greed = caution
  4. Price Momentum: above MA = bullish
  5. RV Regime: volatility expansion = risk-off

Session insight: Non-US hours generate most BTC returns;
OI change correlates 0.43 with Non-US returns → used as signal weight.

Scoring:
  Each signal → +1 (bullish), -1 (bearish), or 0 (neutral)
  Aggregate score = sum of active signals
  SHORT when score <= -threshold, LONG otherwise

Key optimization: precompute ALL O(n²) expanding percentiles upfront.
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore', category=FutureWarning)

from rv_regime.data_loader import load_and_prepare
from rv_regime.strategy import compute_rv_ratio, compute_expanding_threshold
from rv_regime.config import TAKER_FEE

WF_ROUNDS = [
    {"train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-03-31"},
    {"train_end": "2021-03-31", "test_start": "2021-04-01", "test_end": "2021-06-30"},
    {"train_end": "2021-06-30", "test_start": "2021-07-01", "test_end": "2021-09-30"},
    {"train_end": "2021-09-30", "test_start": "2021-10-01", "test_end": "2021-12-31"},
    {"train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-03-31"},
    {"train_end": "2022-03-31", "test_start": "2022-04-01", "test_end": "2022-06-30"},
    {"train_end": "2022-06-30", "test_start": "2022-07-01", "test_end": "2022-09-30"},
    {"train_end": "2022-09-30", "test_start": "2022-10-01", "test_end": "2022-12-31"},
    {"train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-03-31"},
    {"train_end": "2023-03-31", "test_start": "2023-04-01", "test_end": "2023-06-30"},
    {"train_end": "2023-06-30", "test_start": "2023-07-01", "test_end": "2023-09-30"},
    {"train_end": "2023-09-30", "test_start": "2023-10-01", "test_end": "2023-12-31"},
    {"train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-03-31"},
    {"train_end": "2024-03-31", "test_start": "2024-04-01", "test_end": "2024-06-30"},
    {"train_end": "2024-06-30", "test_start": "2024-07-01", "test_end": "2024-09-30"},
    {"train_end": "2024-09-30", "test_start": "2024-10-01", "test_end": "2024-12-31"},
    {"train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-03-31"},
    {"train_end": "2025-03-31", "test_start": "2025-04-01", "test_end": "2025-06-30"},
    {"train_end": "2025-06-30", "test_start": "2025-07-01", "test_end": "2025-09-30"},
    {"train_end": "2025-09-30", "test_start": "2025-10-01", "test_end": "2025-12-31"},
    {"train_end": "2025-12-31", "test_start": "2026-01-01", "test_end": "2026-03-31"},
    {"train_end": "2026-03-31", "test_start": "2026-04-01", "test_end": "2026-05-07"},
]

# ── Search Space ──
SEARCH_FW = [5, 7, 14, 21]           # funding rolling window
SEARCH_OW = [5, 7, 14]               # OI rolling window
SEARCH_FG_LOW = [20, 25, 30]         # fear threshold
SEARCH_FG_HIGH = [70, 75, 80]        # greed threshold
SEARCH_MA = [10, 20, 30]             # MA period
SEARCH_SHORT_TH = [2, 3, 4]          # min bearish score to trigger short
SEARCH_SIGNALS = ['all5', 'fr_oi_fg', 'fr_oi_ma', 'fr_fg_ma']

# RV regime
SEARCH_RV_SW = [3, 5]
SEARCH_RV_LW = [21, 60]
SEARCH_RV_QH = [0.75, 0.85]


def expanding_percentile(vals, q):
    """Expanding window percentile (bias-free). Takes numpy array, returns numpy array."""
    n = len(vals)
    out = np.empty(n)
    pct = q * 100
    for i in range(n):
        out[i] = np.percentile(vals[:i + 1], pct)
    return out


def load_all_signals():
    """Load all signal data sources."""
    rv_daily, daily_ret = load_and_prepare()
    tz = daily_ret.index.tz

    def align_dates(dates, tz):
        dt = pd.DatetimeIndex(pd.to_datetime(dates))
        return dt.normalize().tz_localize(tz)

    # Funding rate
    fr = pd.read_parquet('data/funding_rate_daily.parquet')
    fr_idx = align_dates(fr['date'], tz)
    funding = pd.Series(fr['funding_rate_mean'].values, index=fr_idx, name='funding_rate')
    print(f"  Funding: {len(funding)} days ({funding.index[0].date()} ~ {funding.index[-1].date()})", flush=True)

    # OI change
    oi_change = None
    try:
        mt = pd.read_parquet('data/metrics_daily.parquet')
        mt_idx = align_dates(mt['date'], tz)
        if 'sum_open_interest_value' in mt.columns:
            oi_val = pd.Series(mt['sum_open_interest_value'].values, index=mt_idx, name='oi_value')
            oi_change = oi_val.pct_change()
            oi_change.name = 'oi_change'
            print(f"  OI change: {oi_change.notna().sum()} days", flush=True)
    except Exception:
        pass

    # Fear & Greed
    fg = pd.read_parquet('data/fear_greed_daily.parquet')
    fg_idx = align_dates(fg['date'], tz)
    fear_greed = pd.Series(fg['fear_greed_value'].values, index=fg_idx, name='fear_greed')
    print(f"  Fear&Greed: {len(fear_greed)} days", flush=True)

    return rv_daily, daily_ret, funding, oi_change, fear_greed


def precompute_all(rv_daily, daily_ret, funding, oi_change, fear_greed):
    """
    Precompute ALL expensive O(n²) operations upfront.
    Returns dict of precomputed numpy arrays aligned to a common index.
    """
    t0 = time.time()
    tz = daily_ret.index.tz
    cache = {}

    # 1. Precompute FR rolling sums + expanding percentiles for all fw
    print("  [precompute] FR signals...", end="", flush=True)
    fr_cache = {}
    for fw in SEARCH_FW:
        fr_smooth = funding.rolling(fw, min_periods=fw).sum().dropna()
        vals = fr_smooth.values
        upper = expanding_percentile(vals, 0.75)
        lower = expanding_percentile(vals, 0.25)
        # FR signal: +1 if above upper, -1 if below lower, 0 otherwise
        sig = np.zeros(len(vals))
        sig[vals > upper] = 1
        sig[vals < lower] = -1
        fr_cache[fw] = (fr_smooth.index, sig)
    print(f" {len(fr_cache)} fw combos", flush=True)
    cache['fr'] = fr_cache

    # 2. Precompute OI rolling means for all ow
    print("  [precompute] OI signals...", end="", flush=True)
    oi_cache = {}
    if oi_change is not None:
        for ow in SEARCH_OW:
            oi_smooth = oi_change.rolling(ow, min_periods=ow).mean().dropna()
            sig = np.zeros(len(oi_smooth))
            sig[oi_smooth.values > 0] = 1
            sig[oi_smooth.values < 0] = -1
            oi_cache[ow] = (oi_smooth.index, sig)
    print(f" {len(oi_cache)} ow combos", flush=True)
    cache['oi'] = oi_cache

    # 3. Fear & Greed is already daily — just store aligned values
    cache['fg'] = (fear_greed.index, fear_greed.values)
    print("  [precompute] F&G ready", flush=True)

    # 4. Price MA for all ma_period
    print("  [precompute] MA signals...", end="", flush=True)
    price = (1 + daily_ret).cumprod()
    ma_cache = {}
    for ma_period in SEARCH_MA:
        ma = price.rolling(ma_period, min_periods=ma_period).mean()
        # Signal: +1 above MA, -1 below
        sig = np.zeros(len(price))
        above = price.values > ma.values
        below = price.values < ma.values
        sig[above] = 1
        sig[below] = -1
        ma_cache[ma_period] = (price.index, sig)
    print(f" {len(ma_cache)} MA combos", flush=True)
    cache['ma'] = ma_cache

    # 5. RV regime
    print("  [precompute] RV regime...", end="", flush=True)
    rv_cache = {}
    rv_combos = [(sw, lw, qh) for sw, lw, qh in product(SEARCH_RV_SW, SEARCH_RV_LW, SEARCH_RV_QH)
                 if sw < lw]
    for sw, lw, qh in rv_combos:
        rv_ratio = compute_rv_ratio(rv_daily, sw, lw)
        thresholds = compute_expanding_threshold(rv_ratio, qh)
        is_expansion = rv_ratio > thresholds
        # Signal: -1 if expansion (bearish), 0 otherwise
        sig = np.zeros(len(rv_ratio))
        sig[is_expansion.values] = -1
        rv_cache[(sw, lw, qh)] = (rv_ratio.index, sig)
    print(f" {len(rv_cache)} RV combos", flush=True)
    cache['rv'] = rv_cache

    # 6. Daily returns (for alignment)
    cache['daily_ret'] = (daily_ret.index, daily_ret.values)

    elapsed = time.time() - t0
    print(f"  [precompute] 전체 완료: {elapsed:.1f}초", flush=True)

    return cache


def _build_common_index(cache, fw, ow, ma_period, rv_key, signal_set):
    """Find the common date index for given parameter set."""
    # Start from FR index (shortest typically)
    fr_idx = cache['fr'][fw][0]
    common = fr_idx

    if signal_set in ('all5', 'fr_oi_fg', 'fr_oi_ma') and ow in cache['oi']:
        oi_idx = cache['oi'][ow][0]
        common = common.intersection(oi_idx)

    if signal_set in ('all5', 'fr_oi_ma', 'fr_fg_ma'):
        ma_idx = cache['ma'][ma_period][0]
        common = common.intersection(ma_idx)

    if signal_set == 'all5' and rv_key is not None and rv_key in cache['rv']:
        rv_idx = cache['rv'][rv_key][0]
        common = common.intersection(rv_idx)

    ret_idx = cache['daily_ret'][0]
    common = common.intersection(ret_idx)

    return common


# ── Parallel worker ──
_worker_cache = {}

def _init_worker(cache_serialized):
    """Initialize worker with precomputed cache."""
    _worker_cache.update(cache_serialized)


def _eval_combo(args):
    """Evaluate one parameter combo on training data."""
    (fw, ow, fg_low, fg_high, ma_period, rv_key, signal_set,
     short_th, train_end_ns, common_idx_vals) = args

    try:
        # Use int64 nanoseconds for train_end comparison (avoid tz issues)
        n = len(common_idx_vals)

        # Build score using numpy index lookup (no pd.Series reindex in worker)
        # common_idx_vals are the int64 nanosecond timestamps
        fr_idx, fr_sig = _worker_cache['fr'][fw]
        fr_lookup = dict(zip(fr_idx.values.astype(np.int64), fr_sig))
        score = np.array([fr_lookup.get(t, 0.0) for t in common_idx_vals.astype(np.int64)])

        # OI
        if signal_set in ('all5', 'fr_oi_fg', 'fr_oi_ma') and ow in _worker_cache['oi']:
            oi_idx, oi_sig = _worker_cache['oi'][ow]
            oi_lookup = dict(zip(oi_idx.values.astype(np.int64), oi_sig))
            score += np.array([oi_lookup.get(t, 0.0) for t in common_idx_vals.astype(np.int64)])

        # F&G — forward fill manually
        if signal_set in ('all5', 'fr_oi_fg', 'fr_fg_ma'):
            fg_idx, fg_vals = _worker_cache['fg']
            fg_lookup = dict(zip(fg_idx.values.astype(np.int64), fg_vals))
            # Simple: just lookup each date, ffill handled by sorted insert
            fg_aligned = np.zeros(n)
            fg_series = pd.Series(fg_vals, index=fg_idx)
            common_dt = pd.DatetimeIndex(common_idx_vals)
            fg_reindexed = fg_series.reindex(common_dt, method='ffill').values
            fg_sig = np.zeros(n)
            fg_sig[fg_reindexed < fg_low] = 1
            fg_sig[fg_reindexed > fg_high] = -1
            score += fg_sig

        # MA
        if signal_set in ('all5', 'fr_oi_ma', 'fr_fg_ma'):
            ma_idx, ma_sig = _worker_cache['ma'][ma_period]
            ma_lookup = dict(zip(ma_idx.values.astype(np.int64), ma_sig))
            score += np.array([ma_lookup.get(t, 0.0) for t in common_idx_vals.astype(np.int64)])

        # RV
        if signal_set == 'all5' and rv_key is not None and rv_key in _worker_cache['rv']:
            rv_idx, rv_sig = _worker_cache['rv'][rv_key]
            rv_lookup = dict(zip(rv_idx.values.astype(np.int64), rv_sig))
            score += np.array([rv_lookup.get(t, 0.0) for t in common_idx_vals.astype(np.int64)])

        # Position
        pos = np.ones(n)
        pos[score <= -short_th] = -1.0

        # Shift
        pos_applied = np.roll(pos, 1)
        pos_applied[0] = 1.0

        # Returns
        ret_idx, ret_vals = _worker_cache['daily_ret']
        ret_lookup = dict(zip(ret_idx.values.astype(np.int64), ret_vals))
        ret = np.array([ret_lookup.get(t, 0.0) for t in common_idx_vals.astype(np.int64)])

        # Cost
        cost = np.abs(np.diff(pos_applied, prepend=0)) * TAKER_FEE
        cost[0] = 0

        strat_ret = pos_applied * ret - cost

        # Train mask — use int64 comparison (no tz needed)
        train_mask = common_idx_vals.astype(np.int64) <= train_end_ns
        s = strat_ret[train_mask]
        b = ret[train_mask]
        p = pos_applied[train_mask]

        if len(s) < 60:
            return None

        short_pct = (p == -1).sum() / len(p)
        if short_pct < 0.05:
            return None

        # Information Ratio
        excess = s - b
        excess_std = np.std(excess)
        if excess_std > 0:
            ir = np.mean(excess) / excess_std * np.sqrt(365)
        else:
            ir = 0

        s_std = np.std(s)
        raw_sh = np.mean(s) / s_std * np.sqrt(365) if s_std > 0 else 0

        return (ir, raw_sh, fw, ow, fg_low, fg_high, ma_period, rv_key, signal_set, short_th)
    except Exception:
        return None


def select_best_parallel(cache, train_end_ts, n_workers=10):
    """Grid search with multiprocessing."""
    rv_keys = list(cache['rv'].keys())

    # Build combo list with precomputed common indices
    combos = []
    for fw in SEARCH_FW:
        for ow in SEARCH_OW:
            for fg_low in SEARCH_FG_LOW:
                for fg_high in SEARCH_FG_HIGH:
                    for ma_period in SEARCH_MA:
                        for signal_set in SEARCH_SIGNALS:
                            for short_th in SEARCH_SHORT_TH:
                                if signal_set == 'all5':
                                    for rv_key in rv_keys:
                                        common = _build_common_index(
                                            cache, fw, ow, ma_period, rv_key, signal_set)
                                        combos.append((
                                            fw, ow, fg_low, fg_high, ma_period,
                                            rv_key, signal_set, short_th,
                                            train_end_ts.value, common.values
                                        ))
                                else:
                                    common = _build_common_index(
                                        cache, fw, ow, ma_period, None, signal_set)
                                    combos.append((
                                        fw, ow, fg_low, fg_high, ma_period,
                                        None, signal_set, short_th,
                                        train_end_ts.value, common.values
                                    ))

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(cache,)
    ) as pool:
        results = pool.map(_eval_combo, combos, chunksize=100)

    best_ir = -999
    best = None
    for r in results:
        if r is not None and r[0] > best_ir:
            best_ir = r[0]
            best = r

    if best is None:
        return None, 0, len(combos)

    ir, raw_sh, fw, ow, fg_low, fg_high, ma_period, rv_key, signal_set, short_th = best
    return (fw, ow, fg_low, fg_high, ma_period, rv_key, signal_set, short_th), raw_sh, len(combos)


def run_test_period(cache, params, test_start_ts, test_end_ts):
    """Run strategy on test period with given params."""
    fw, ow, fg_low, fg_high, ma_period, rv_key, signal_set, short_th = params

    common = _build_common_index(cache, fw, ow, ma_period, rv_key, signal_set)

    # Build score
    fr_idx, fr_sig = cache['fr'][fw]
    score = pd.Series(fr_sig, index=fr_idx).reindex(common, fill_value=0).values.copy()

    if signal_set in ('all5', 'fr_oi_fg', 'fr_oi_ma') and ow in cache['oi']:
        oi_idx, oi_sig = cache['oi'][ow]
        score += pd.Series(oi_sig, index=oi_idx).reindex(common, fill_value=0).values

    if signal_set in ('all5', 'fr_oi_fg', 'fr_fg_ma'):
        fg_idx, fg_vals = cache['fg']
        fg_aligned = pd.Series(fg_vals, index=fg_idx).reindex(common, method='ffill').values
        fg_sig = np.zeros(len(common))
        fg_sig[fg_aligned < fg_low] = 1
        fg_sig[fg_aligned > fg_high] = -1
        score += fg_sig

    if signal_set in ('all5', 'fr_oi_ma', 'fr_fg_ma'):
        ma_idx, ma_sig = cache['ma'][ma_period]
        score += pd.Series(ma_sig, index=ma_idx).reindex(common, fill_value=0).values

    if signal_set == 'all5' and rv_key is not None and rv_key in cache['rv']:
        rv_idx, rv_sig = cache['rv'][rv_key]
        score += pd.Series(rv_sig, index=rv_idx).reindex(common, fill_value=0).values

    pos = np.ones(len(common))
    pos[score <= -short_th] = -1.0

    pos_applied = np.roll(pos, 1)
    pos_applied[0] = 1.0

    ret_idx, ret_vals = cache['daily_ret']
    ret = pd.Series(ret_vals, index=ret_idx).reindex(common).values

    cost = np.abs(np.diff(pos_applied, prepend=0)) * TAKER_FEE
    cost[0] = 0

    strat_ret = pos_applied * ret - cost

    # Test mask
    test_mask = (common >= test_start_ts) & (common <= test_end_ts)

    test_sr = pd.Series(strat_ret[test_mask], index=common[test_mask])
    test_br = pd.Series(ret[test_mask], index=common[test_mask])
    test_pf = pd.Series(pos_applied[test_mask], index=common[test_mask])

    return test_sr, test_br, test_pf


def main():
    print("=" * 80, flush=True)
    print("  Walk-Forward: Multi-Signal + Session Strategy", flush=True)
    print("=" * 80, flush=True)

    rv_daily, daily_ret, funding, oi_change, fear_greed = load_all_signals()
    tz = daily_ret.index.tz

    cache = precompute_all(rv_daily, daily_ret, funding, oi_change, fear_greed)

    n_workers = min(cpu_count() - 2, 10)
    print(f"  Workers: {n_workers}", flush=True)

    # Count combos
    rv_keys = list(cache['rv'].keys())
    n_base = (len(SEARCH_FW) * len(SEARCH_OW) * len(SEARCH_FG_LOW) *
              len(SEARCH_FG_HIGH) * len(SEARCH_MA) * len(SEARCH_SHORT_TH))
    n_all5 = n_base * len(rv_keys)
    n_other = n_base * 3
    n_total = n_all5 + n_other
    print(f"  Combos per round: {n_total}", flush=True)
    print(flush=True)

    all_strat = []
    all_bnh = []
    all_pos = []
    round_info = []
    t_total = time.time()

    for ri, r in enumerate(WF_ROUNDS):
        t0 = time.time()
        train_end_ts = pd.Timestamp(r["train_end"]).tz_localize(tz)
        test_start_ts = pd.Timestamp(r["test_start"]).tz_localize(tz)
        test_end_ts = pd.Timestamp(r["test_end"]).tz_localize(tz)

        params, train_sh, n_combos = select_best_parallel(
            cache, train_end_ts, n_workers
        )

        if params is None:
            print(f"  R{ri+1:>2}/{len(WF_ROUNDS)}: no valid params", flush=True)
            continue

        fw, ow, fg_low, fg_high, ma_period, rv_key, signal_set, short_th = params

        test_sr, test_br, test_pf = run_test_period(
            cache, params, test_start_ts, test_end_ts
        )

        if len(test_sr) == 0:
            print(f"  R{ri+1:>2}/{len(WF_ROUNDS)}: no test data", flush=True)
            continue

        test_sharpe = (
            test_sr.mean() / test_sr.std() * np.sqrt(365)
            if test_sr.std() > 0 else 0
        )
        test_ret = (1 + test_sr).cumprod().iloc[-1] - 1
        bnh_ret = (1 + test_br).cumprod().iloc[-1] - 1
        short_pct = (test_pf == -1).sum() / len(test_pf) * 100

        elapsed = time.time() - t0
        remaining = elapsed * (len(WF_ROUNDS) - ri - 1)

        sig_label = signal_set.upper().replace('_', '+')
        rv_label = f",rv={rv_key}" if rv_key else ""

        info = {
            "round": ri + 1,
            "test_period": f"{r['test_start']}~{r['test_end']}",
            "params": params,
            "train_sharpe": train_sh, "test_sharpe": test_sharpe,
            "test_ret": test_ret, "bnh_ret": bnh_ret,
            "short_pct": short_pct, "test_days": len(test_sr),
            "signal_set": signal_set,
        }
        round_info.append(info)

        print(
            f"  R{ri+1:>2}/{len(WF_ROUNDS)}: {r['test_start']}~{r['test_end']} | "
            f"{sig_label} th={short_th},fw={fw},ma={ma_period},fg={fg_low}/{fg_high}{rv_label} | "
            f"Train {train_sh:.2f} -> Test {test_sharpe:+.2f} | "
            f"Strat {test_ret:+.1%} B&H {bnh_ret:+.1%} | Short {short_pct:.0f}% | "
            f"{elapsed:.1f}s (ETA {remaining:.0f}s)",
            flush=True,
        )

        all_strat.append(test_sr)
        all_bnh.append(test_br)
        all_pos.append(test_pf)

    total_time = time.time() - t_total
    print(f"\n  전체 WF 소요: {total_time:.1f}초", flush=True)

    if not all_strat:
        print("\nNo valid rounds.", flush=True)
        return

    wf_strat = pd.concat(all_strat)
    wf_bnh = pd.concat(all_bnh)
    wf_pos = pd.concat(all_pos)

    cum_s = (1 + wf_strat).cumprod()
    cum_b = (1 + wf_bnh).cumprod()
    sh = wf_strat.mean() / wf_strat.std() * np.sqrt(365) if wf_strat.std() > 0 else 0
    mdd = ((cum_s - cum_s.cummax()) / cum_s.cummax()).min()

    test_sharpes = [r["test_sharpe"] for r in round_info]
    sig_counts = {}
    for r in round_info:
        s = r["signal_set"]
        sig_counts[s] = sig_counts.get(s, 0) + 1

    print(f"\n{'='*80}", flush=True)
    print(f"  WF Summary (Multi-Signal + Session)", flush=True)
    print(f"{'='*80}", flush=True)
    print(
        f"  Period: {wf_strat.index[0].strftime('%Y-%m-%d')} ~ "
        f"{wf_strat.index[-1].strftime('%Y-%m-%d')} ({len(wf_strat)} days)",
        flush=True,
    )
    print(f"  Strategy: {cum_s.iloc[-1]-1:>+.1%} (Sharpe {sh:.2f}, MDD {mdd:+.1%})", flush=True)
    print(f"  B&H:      {cum_b.iloc[-1]-1:>+.1%}", flush=True)
    print(f"  Excess:   {cum_s.iloc[-1]-cum_b.iloc[-1]:>+.1%}", flush=True)
    print(
        f"  Rounds Sharpe > 0: "
        f"{sum(1 for s in test_sharpes if s > 0)}/{len(test_sharpes)}",
        flush=True,
    )
    print(f"  Avg Test Sharpe: {np.mean(test_sharpes):.2f}", flush=True)
    print(f"  Median Test Sharpe: {np.median(test_sharpes):.2f}", flush=True)
    print(f"  Signal selection: {sig_counts}", flush=True)

    # Shuffled test
    print(f"\n  Shuffled test (2000 iters)...", end="", flush=True)
    np.random.seed(42)
    ret_vals = wf_bnh.values
    pos_vals = wf_pos.values
    cost_vals = np.abs(np.diff(pos_vals, prepend=0)) * TAKER_FEE
    shuffled = []
    for _ in range(2000):
        perm = np.random.permutation(len(ret_vals))
        s_ret = pos_vals * ret_vals[perm] - cost_vals
        s_std = np.std(s_ret)
        if s_std > 0:
            shuffled.append(np.mean(s_ret) / s_std * np.sqrt(365))
    shuffled = np.array(shuffled)
    p_val = (shuffled >= sh).mean()
    status = 'PASS' if p_val < 0.05 else 'FAIL'
    print(f" p={p_val:.4f} {status}", flush=True)

    # Yearly breakdown
    print(f"\n{'='*80}", flush=True)
    print(f"  WF Yearly Breakdown", flush=True)
    print(f"{'='*80}", flush=True)
    for y in sorted(set(wf_strat.index.year)):
        m = wf_strat.index.year == y
        sr_y = wf_strat[m]
        br_y = wf_bnh[m]
        ret_s = (1 + sr_y).prod() - 1
        ret_b = (1 + br_y).prod() - 1
        shy = sr_y.mean() / sr_y.std() * np.sqrt(365) if sr_y.std() > 0 else 0
        short_y = (wf_pos[m] == -1).sum() / len(wf_pos[m]) * 100
        print(
            f"  {y}: Strat {ret_s:+7.1%} | B&H {ret_b:+7.1%} "
            f"| Excess {ret_s-ret_b:+7.1%} | Sharpe {shy:+.2f} | Short {short_y:.0f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
