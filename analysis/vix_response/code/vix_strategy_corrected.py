"""
yfinance 1시간봉의 timestamp 가 bar '시작' 시각이라는 점 때문에 생긴 정렬 오류를
교정한 전략 재계산. VIX 변화율 시그널을 1 bar shift 해 확인 가능 시점(t+1h)부터
BTC 수익률을 측정하도록 맞춘 뒤 성과를 다시 평가한다.

입력: data/vix_1h.parquet, data/btc_1m_24h/
출력: vix_strategy_corrected.png
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════
# 수정된 로직:
#
# yfinance 1h bar: timestamp = bar 시작, close = bar 끝 가격
# bar[i] timestamp=t → close = VIX(t+1h)
#
# vix_pct[i] = (close[i] - close[i-1]) / close[i-1]
#            = VIX Change from t to t+1h
#
# 이 변화를 알 수 있는 시점: t+1h (bar 종료 후)
#
# 따라서 시그널로 사용하려면:
#   시그널: vix_pct[i] (t ~ t+1h 변화, t+1h에 확인)
#   진입: t+1h 시점
#   BTC 수익: t+1h → t+1h+Xm
#
# 즉 vix_pct를 1칸 shift해서 다음 bar의 BTC와 매칭해야 함
# ══════════════════════════════════════════════════════

vix = pd.read_parquet('data/vix_1h.parquet')
vix['vix_change'] = vix['vix'].diff()
vix['vix_pct'] = vix['vix'].pct_change()
vix = vix.dropna()

# BTC 1분봉 로드
btc_dir = 'data/btc_1m_24h/'
btc_files = sorted([f for f in os.listdir(btc_dir) if f >= 'btc_1m_2024'])
btc = pd.concat([pd.read_parquet(os.path.join(btc_dir, f))[['close']] for f in btc_files]).sort_index()
btc = btc[~btc.index.duplicated(keep='first')]

# 수정된 데이터 생성
# vix_pct[i]의 시그널로 → bar[i+1]의 시작 시점(= bar[i] 시작 + 1h)부터 BTC 측정
offsets = {
    'btc_1m': 1, 'btc_2m': 2, 'btc_3m': 3, 'btc_5m': 5,
    'btc_10m': 10, 'btc_15m': 15, 'btc_20m': 20, 'btc_30m': 30,
    'btc_45m': 45, 'btc_1h': 60, 'btc_90m': 90, 'btc_2h': 120,
    'btc_3h': 180, 'btc_4h': 240,
}

vix_list = list(vix.iterrows())
records = []

for i in range(len(vix_list) - 1):
    ts_signal, row_signal = vix_list[i]      # 시그널 bar
    ts_next, row_next = vix_list[i + 1]      # 다음 bar (진입 시점)
    
    # 진입 시점 = 시그널 bar 종료 = 다음 bar 시작 = ts_next
    entry_time = ts_next
    
    if entry_time not in btc.index:
        continue
    
    base_price = btc.loc[entry_time, 'close']
    
    # Intraday 여부 (진입 시점 기준)
    hour = entry_time.hour
    is_market = (hour >= 9) and (hour <= 16)
    
    record = {
        'signal_time': ts_signal,
        'entry_time': entry_time,
        'vix': row_signal['vix'],
        'vix_pct': row_signal['vix_pct'],  # 이전 1시간 VIX Change (이미 확정)
        'btc_entry_price': base_price,
        'hour': hour,
        'is_market_hours': is_market,
    }
    
    for col_name, minutes in offsets.items():
        target_time = entry_time + pd.Timedelta(minutes=minutes)
        if target_time in btc.index:
            target_price = btc.loc[target_time, 'close']
            record[col_name] = (target_price - base_price) / base_price
        else:
            record[col_name] = np.nan
    
    records.append(record)

df = pd.DataFrame(records)
df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True).dt.tz_convert('America/New_York')
df['signal_time'] = pd.to_datetime(df['signal_time'], utc=True).dt.tz_convert('America/New_York')

mkt = df[df['is_market_hours']].copy()
sigma = mkt['vix_pct'].std()

print(f"All: {len(df)}건, Intraday: {len(mkt)}건")
print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
print(f"VIX 1h 변화율 std: {sigma*100:.2f}%")
print(f"\n로직 확인:")
print(f"  시그널: VIX가 직전 1시간 동안 변한 정도 (이미 확정된 과거 정보)")
print(f"  진입: 시그널 확인 후 즉시 (bar 종료 시점)")
print(f"  BTC 수익: 진입 시점부터 Xm 후")

time_cols = ['btc_1m','btc_2m','btc_3m','btc_5m','btc_10m','btc_15m','btc_20m','btc_30m',
             'btc_45m','btc_1h','btc_90m','btc_2h','btc_3h','btc_4h']
time_labels = ['1m','2m','3m','5m','10m','15m','20m','30m','45m','1h','90m','2h','3h','4h']

# ══════════════════════════════════════════════════════
# 1. 수정된 상관 분석 (Intraday)
# ══════════════════════════════════════════════════════
print("\n" + "="*70)
print("1. 수정된 상관 분석 — 이전 1h VIX Change → 이후 BTC (Intraday)")
print("="*70)
corrs_corrected = []
for col, label in zip(time_cols, time_labels):
    valid = mkt[['vix_pct', col]].dropna()
    r, p = stats.pearsonr(valid['vix_pct'], valid[col])
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {label:>4s}  r={r:+.4f} p={p:.6f}{sig}")
    corrs_corrected.append({'label': label, 'r': r, 'p': p})

# ══════════════════════════════════════════════════════
# 2. 다양한 기준별 Strategy 성과 (1h Hold)
# ══════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. By Entry Criteria Strategy 성과 (1h 홀딩, Intraday)")
print("="*70)

thresholds = [
    ('0.5σ', 0.5*sigma),
    ('0.75σ', 0.75*sigma),
    ('1.0σ', 1.0*sigma),
    ('1.5σ', 1.5*sigma),
    ('2.0σ', 2.0*sigma),
]

hold_col = 'btc_1h'

# B&H
btc_start = mkt.iloc[0]['btc_entry_price']
btc_end = mkt.iloc[-1]['btc_entry_price']
bh_ret = (btc_end - btc_start) / btc_start * 100
days = (mkt['entry_time'].max() - mkt['entry_time'].min()).days
years = days / 365.25

print(f"\n  기간: {mkt['entry_time'].min().date()} ~ {mkt['entry_time'].max().date()} ({days}일)")
print(f"  BTC: ${btc_start:,.0f} → ${btc_end:,.0f}, B&H = {bh_ret:+.1f}%\n")

print(f"{'기준':>6s}  {'거래':>5s}  {'Total Return':>8s}  {'연환산':>8s}  {'Mean/건':>9s}  {'승률':>5s}  {'PF':>6s}  {'최대DD':>7s}")
results_thresh = {}
for name, thresh in thresholds:
    trades = []
    times = []
    for _, row in mkt.iterrows():
        v = row[hold_col]
        if pd.isna(v):
            continue
        if row['vix_pct'] > thresh:
            trades.append(-v)
            times.append(row['entry_time'])
        elif row['vix_pct'] < -thresh:
            trades.append(v)
            times.append(row['entry_time'])
    
    if len(trades) == 0:
        continue
    arr = np.array(trades)
    cum = np.cumsum(arr)
    total = cum[-1]*100
    annual = total / years
    avg = arr.mean()*100
    wr = (arr>0).mean()*100
    gp = arr[arr>0].sum()
    gl = abs(arr[arr<0].sum())
    pf = gp/gl if gl>0 else np.inf
    dd = np.min(cum - np.maximum.accumulate(cum))*100
    
    results_thresh[name] = {'arr': arr, 'times': times, 'total': total, 'n': len(arr),
                             'annual': annual, 'avg': avg, 'wr': wr, 'pf': pf, 'dd': dd}
    print(f"{name:>6s}  {len(arr):>5d}  {total:>+7.1f}%  {annual:>+7.1f}%  {avg:>+8.4f}%  {wr:>4.1f}%  {pf:>5.2f}  {dd:>+6.2f}%")

# ══════════════════════════════════════════════════════
# 3. By Hold Duration (1σ)
# ══════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. By Hold Duration 성과 (1σ, Intraday)")
print("="*70)
thresh_1s = 1.0 * sigma

print(f"\n{'홀딩':>4s}  {'거래':>5s}  {'Total Return':>8s}  {'Mean/건':>9s}  {'승률':>5s}  {'PF':>6s}  {'최대DD':>7s}")
hold_results = {}
for col, label in zip(time_cols, time_labels):
    trades = []
    for _, row in mkt.iterrows():
        v = row[col]
        if pd.isna(v):
            continue
        if row['vix_pct'] > thresh_1s:
            trades.append(-v)
        elif row['vix_pct'] < -thresh_1s:
            trades.append(v)
    if len(trades) == 0:
        continue
    arr = np.array(trades)
    cum = np.cumsum(arr)
    total = cum[-1]*100
    avg = arr.mean()*100
    wr = (arr>0).mean()*100
    gp = arr[arr>0].sum(); gl = abs(arr[arr<0].sum())
    pf = gp/gl if gl>0 else np.inf
    dd = np.min(cum - np.maximum.accumulate(cum))*100
    hold_results[label] = {'total': total, 'avg': avg, 'wr': wr, 'pf': pf, 'dd': dd, 'arr': arr}
    print(f"{label:>4s}  {len(arr):>5d}  {total:>+7.1f}%  {avg:>+8.4f}%  {wr:>4.1f}%  {pf:>5.2f}  {dd:>+6.2f}%")

# ══════════════════════════════════════════════════════
# 4. 급변 이벤트
# ══════════════════════════════════════════════════════
print("\n" + "="*70)
print("4. VIX 급변 이벤트 (Intraday, 2σ)")
print("="*70)
spike = mkt[mkt['vix_pct'] > 2*sigma]
drop = mkt[mkt['vix_pct'] < -2*sigma]
print(f"  급등:{len(spike)}건, 급락:{len(drop)}건")

for label_ev, sub_ev, direction in [('VIX Spike→숏', spike, -1), ('VIX Drop→롱', drop, 1)]:
    print(f"\n  [{label_ev}]")
    for col, label in zip(time_cols, time_labels):
        s = sub_ev[col].dropna() * direction
        if len(s) < 5:
            continue
        m = s.mean()
        wr = (s>0).mean()*100
        t, p = stats.ttest_1samp(s, 0)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        print(f"    {label:>4s}  mean={m*100:+.4f}%  승률={wr:.0f}%  t={t:+.2f} p={p:.4f}{sig}")

# ══════════════════════════════════════════════════════
# 5. 롱/숏 분리 (1σ, 1h)
# ══════════════════════════════════════════════════════
print("\n" + "="*70)
print("5. 롱/숏 분리 (1σ, 1h)")
print("="*70)
long_t = []; short_t = []
for _, row in mkt.iterrows():
    v = row['btc_1h']
    if pd.isna(v): continue
    if row['vix_pct'] > thresh_1s:
        short_t.append(-v)
    elif row['vix_pct'] < -thresh_1s:
        long_t.append(v)

for side, arr in [('롱', np.array(long_t)), ('숏', np.array(short_t))]:
    if len(arr) == 0: continue
    cum = np.cumsum(arr)
    total = cum[-1]*100
    avg = arr.mean()*100
    wr = (arr>0).mean()*100
    gp = arr[arr>0].sum(); gl = abs(arr[arr<0].sum())
    pf = gp/gl if gl>0 else np.inf
    dd = np.min(cum - np.maximum.accumulate(cum))*100
    print(f"  [{side}] n={len(arr)}  Total Return={total:+.1f}%  Mean={avg:+.4f}%  승률={wr:.1f}%  PF={pf:.2f}  DD={dd:+.2f}%")

# ══════════════════════════════════════════════════════
# 6. 월별 (1σ, 최적 홀딩)
# ══════════════════════════════════════════════════════
# 최적 홀딩 시간 결정
best_hold = max(hold_results.keys(), key=lambda k: hold_results[k]['total'])
print(f"\n  최적 홀딩: {best_hold}")

# ══════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. 상관계수
ax = axes[0, 0]
corr_df = pd.DataFrame(corrs_corrected)
colors = ['#e74c3c' if p < 0.05 else '#bdc3c7' for p in corr_df['p']]
ax.bar(range(len(time_labels)), corr_df['r'], color=colors)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels))); ax.set_xticklabels(time_labels, rotation=45)
ax.set_ylabel('Pearson r'); ax.set_title('Corrected Correlation: Previous 1h VIX to Subsequent BTC (red=p<0.05)')

# 2. 기준별 Total Return
ax = axes[0, 1]
names = list(results_thresh.keys())
totals = [results_thresh[n]['total'] for n in names]
ns = [results_thresh[n]['n'] for n in names]
bars = ax.bar(names, totals, color=['#3498db','#2ecc71','#e74c3c','#f39c12','#9b59b6'])
for bar, n in zip(bars, ns):
    ax.text(bar.get_x()+bar.get_width()/2, max(bar.get_height(),0)+1, f'n={n}', ha='center', fontsize=8)
ax.axhline(bh_ret, color='red', linestyle='--', label=f'B&H {bh_ret:.0f}%')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Total Return (%)'); ax.set_title('By Entry Criteria Total Return (1h Hold)')
ax.legend()

# 3. 홀딩시간별
ax = axes[0, 2]
hlabels = list(hold_results.keys())
htotals = [hold_results[h]['total'] for h in hlabels]
ax.bar(hlabels, htotals, color=['#2ecc71' if t>0 else '#e74c3c' for t in htotals])
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Total Return (%)'); ax.set_title('By Hold Duration (1σ)')
ax.tick_params(axis='x', rotation=45)

# 4. 누적수익 (1σ, best hold)
ax = axes[1, 0]
if '1.0σ' in results_thresh:
    cum = np.cumsum(results_thresh['1.0σ']['arr'])*100
    ax.plot(cum, 'k-', linewidth=0.8)
    ax.fill_between(range(len(cum)), cum, alpha=0.1)
    ax.axhline(0, color='red', linewidth=0.5)
    ax.set_xlabel('Trade Number'); ax.set_ylabel('Cumulative Return (%)')
    ax.set_title(f'Cumulative Return (1σ, 1h, n={len(cum)})')

# 5. Return Distribution
ax = axes[1, 1]
if '1.0σ' in results_thresh:
    arr = results_thresh['1.0σ']['arr']
    ax.hist(arr*100, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linewidth=1)
    ax.axvline(arr.mean()*100, color='green', linewidth=2, label=f'Mean {arr.mean()*100:+.3f}%')
    ax.set_xlabel('Return per Trade (%)'); ax.set_ylabel('Frequency')
    ax.set_title('Return Distribution'); ax.legend()

# 6. Drawdown
ax = axes[1, 2]
if '1.0σ' in results_thresh:
    arr = results_thresh['1.0σ']['arr']
    cum = np.cumsum(arr)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max)*100
    ax.fill_between(range(len(dd)), dd, color='red', alpha=0.3)
    ax.set_xlabel('Trade Number'); ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown')

plt.suptitle('VIX to BTC Strategy (After Lookahead Bias Fix)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('vix_strategy_corrected.png', dpi=150, bbox_inches='tight')
print("\n저장: vix_strategy_corrected.png")

# 데이터 저장
df.to_parquet('data/vix_1h_btc_response_corrected.parquet', index=False)
print("저장: data/vix_1h_btc_response_corrected.parquet")
