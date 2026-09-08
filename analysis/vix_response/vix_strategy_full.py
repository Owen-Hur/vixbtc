import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_parquet('data/vix_1h_btc_response.parquet')
mkt = df[df['is_market_hours']].copy()

# BTC 1분봉 로드 (Buy&Hold 비교용)
btc_dir = 'data/btc_1m_24h/'
btc_files = sorted([f for f in os.listdir(btc_dir) if f >= 'btc_1m_2024'])
btc = pd.concat([pd.read_parquet(os.path.join(btc_dir, f))[['close']] for f in btc_files]).sort_index()
btc = btc[~btc.index.duplicated(keep='first')]

# BTC 동시 변화 (t-1h → t) 추가
records = []
for _, row in mkt.iterrows():
    ts = row['timestamp']
    ts_1h_ago = ts - pd.Timedelta(hours=1)
    if ts not in btc.index or ts_1h_ago not in btc.index:
        continue
    btc_before = btc.loc[ts_1h_ago, 'close']
    btc_at = btc.loc[ts, 'close']
    rec = row.to_dict()
    rec['btc_during'] = (btc_at - btc_before) / btc_before
    rec['btc_price'] = btc_at
    records.append(rec)

td = pd.DataFrame(records)
td['timestamp'] = pd.to_datetime(td['timestamp'], utc=True).dt.tz_convert('America/New_York')
td = td.sort_values('timestamp').reset_index(drop=True)

sigma = td['vix_pct'].std()
print(f"데이터: {len(td)}건, {td['timestamp'].min()} ~ {td['timestamp'].max()}")
print(f"VIX 1h 변화율 std: {sigma*100:.2f}%")

# ══════════════════════════════════════════════════════
# 1. 다양한 By Entry Criteria 성과 비교
# ══════════════════════════════════════════════════════
print("\n" + "="*80)
print("1. By Entry Criteria 성과 비교 (Hold 1h)")
print("="*80)

thresholds = [
    ('0.5σ', 0.5*sigma),
    ('0.75σ', 0.75*sigma),
    ('1.0σ', 1.0*sigma),
    ('1.5σ', 1.5*sigma),
    ('2.0σ', 2.0*sigma),
]

hold_col = 'btc_1h'
results_by_thresh = {}

print(f"\n{'기준':>6s}  {'거래수':>6s}  {'Total Return':>8s}  {'Mean/거래':>10s}  {'승률':>6s}  {'최대낙폭':>8s}  {'최대연승':>8s}  {'최대연패':>8s}  {'수익표준편차':>12s}")
for name, thresh in thresholds:
    trades = []
    trade_times = []
    for _, row in td.iterrows():
        if row['vix_pct'] > thresh:
            trades.append(-row[hold_col])  # 숏
            trade_times.append(row['timestamp'])
        elif row['vix_pct'] < -thresh:
            trades.append(row[hold_col])   # 롱
            trade_times.append(row['timestamp'])
    
    trades = np.array(trades)
    if len(trades) == 0:
        continue
    
    cum = np.cumsum(trades)
    total_ret = cum[-1] * 100
    avg_ret = trades.mean() * 100
    win_rate = (trades > 0).mean() * 100
    max_dd = np.min(cum - np.maximum.accumulate(cum)) * 100
    std_ret = trades.std() * 100
    
    # 최대 연승/연패
    wins = (trades > 0).astype(int)
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    for w in wins:
        if w == 1:
            cur_win += 1; cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
        else:
            cur_loss += 1; cur_win = 0
            max_loss_streak = max(max_loss_streak, cur_loss)
    
    results_by_thresh[name] = {
        'trades': trades, 'times': trade_times, 'n': len(trades),
        'total': total_ret, 'avg': avg_ret, 'winrate': win_rate,
        'maxdd': max_dd, 'std': std_ret
    }
    
    print(f"{name:>6s}  {len(trades):>6d}  {total_ret:>+7.1f}%  {avg_ret:>+9.4f}%  {win_rate:>5.1f}%  {max_dd:>+7.2f}%  {max_win_streak:>8d}  {max_loss_streak:>8d}  {std_ret:>11.4f}%")

# ══════════════════════════════════════════════════════
# 2. By Hold Duration 성과 (1σ 기준)
# ══════════════════════════════════════════════════════
print("\n" + "="*80)
print("2. By Hold Duration 성과 (1σ 기준)")
print("="*80)

hold_cols = ['btc_1m','btc_5m','btc_10m','btc_15m','btc_20m','btc_30m',
             'btc_45m','btc_1h','btc_90m','btc_2h','btc_3h','btc_4h']
hold_labels = ['1m','5m','10m','15m','20m','30m','45m','1h','90m','2h','3h','4h']
thresh_1s = 1.0 * sigma

print(f"\n{'홀딩':>4s}  {'거래수':>6s}  {'Total Return':>8s}  {'Mean/거래':>10s}  {'승률':>6s}  {'최대낙폭':>8s}  {'Profit Factor':>14s}")
hold_results = {}
for col, label in zip(hold_cols, hold_labels):
    trades = []
    for _, row in td.iterrows():
        if row['vix_pct'] > thresh_1s:
            trades.append(-row[col])
        elif row['vix_pct'] < -thresh_1s:
            trades.append(row[col])
    trades = np.array(trades)
    if len(trades) == 0:
        continue
    cum = np.cumsum(trades)
    total = cum[-1]*100
    avg = trades.mean()*100
    wr = (trades>0).mean()*100
    dd = np.min(cum - np.maximum.accumulate(cum))*100
    gross_profit = trades[trades>0].sum()
    gross_loss = abs(trades[trades<0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    hold_results[label] = {'trades': trades, 'total': total, 'avg': avg, 'wr': wr, 'dd': dd, 'pf': pf}
    print(f"{label:>4s}  {len(trades):>6d}  {total:>+7.1f}%  {avg:>+9.4f}%  {wr:>5.1f}%  {dd:>+7.2f}%  {pf:>13.2f}")

# ══════════════════════════════════════════════════════
# 3. Buy & Hold 비교
# ══════════════════════════════════════════════════════
print("\n" + "="*80)
print("3. Buy & Hold 비교")
print("="*80)

# Strategy 기간의 BTC 가격
start_date = td['timestamp'].min()
end_date = td['timestamp'].max()
btc_start = td.iloc[0]['btc_price']
btc_end = td.iloc[-1]['btc_price']
bh_ret = (btc_end - btc_start) / btc_start * 100

# Strategy 수익 (1σ, 1h 홀딩)
strat_trades = []
strat_times = []
for _, row in td.iterrows():
    if row['vix_pct'] > thresh_1s:
        strat_trades.append(-row['btc_1h'])
        strat_times.append(row['timestamp'])
    elif row['vix_pct'] < -thresh_1s:
        strat_trades.append(row['btc_1h'])
        strat_times.append(row['timestamp'])

strat_trades = np.array(strat_trades)
strat_cum = np.cumsum(strat_trades)

# 연환산
days = (end_date - start_date).days
years = days / 365.25
strat_annual = (strat_cum[-1]) / years * 100
bh_annual = bh_ret / years

print(f"\n  기간: {start_date.date()} ~ {end_date.date()} ({days}일, {years:.1f}년)")
print(f"  BTC 가격: ${btc_start:,.0f} → ${btc_end:,.0f}")
print(f"\n  {'':>20s}  {'Total Return':>8s}  {'연환산':>8s}  {'거래수':>6s}  {'승률':>6s}  {'최대낙폭':>8s}")
print(f"  {'Buy & Hold':>20s}  {bh_ret:>+7.1f}%  {bh_annual:>+7.1f}%  {'1':>6s}  {'-':>6s}  {'-':>8s}")

# 다양한 기준 비교
for name in ['0.5σ','0.75σ','1.0σ','1.5σ','2.0σ']:
    if name in results_by_thresh:
        r = results_by_thresh[name]
        annual = r['total'] / years
        print(f"  {'Strategy '+name:>20s}  {r['total']:>+7.1f}%  {annual:>+7.1f}%  {r['n']:>6d}  {r['winrate']:>5.1f}%  {r['maxdd']:>+7.2f}%")

# ══════════════════════════════════════════════════════
# 4. Monthly Returns 분해 (1σ, 1h)
# ══════════════════════════════════════════════════════
print("\n" + "="*80)
print("4. Monthly Returns 분해 (1σ, 1h 홀딩)")
print("="*80)

strat_df = pd.DataFrame({'timestamp': strat_times, 'ret': strat_trades})
strat_df['month'] = strat_df['timestamp'].dt.to_period('M')
monthly = strat_df.groupby('month').agg(
    n_trades=('ret', 'count'),
    total_ret=('ret', 'sum'),
    avg_ret=('ret', 'mean'),
    win_rate=('ret', lambda x: (x>0).mean())
)
monthly['total_ret_pct'] = monthly['total_ret'] * 100
monthly['avg_ret_pct'] = monthly['avg_ret'] * 100
monthly['win_rate_pct'] = monthly['win_rate'] * 100

print(f"\n{'월':>10s}  {'거래수':>6s}  {'Total Return':>8s}  {'Mean/거래':>10s}  {'승률':>6s}")
for period, row in monthly.iterrows():
    print(f"  {str(period):>8s}  {row['n_trades']:>6.0f}  {row['total_ret_pct']:>+7.2f}%  {row['avg_ret_pct']:>+9.4f}%  {row['win_rate_pct']:>5.1f}%")

win_months = (monthly['total_ret'] > 0).sum()
total_months = len(monthly)
print(f"\n  수익 월: {win_months}/{total_months} ({win_months/total_months*100:.1f}%)")

# ══════════════════════════════════════════════════════
# 5. 롱/숏 분리 성과
# ══════════════════════════════════════════════════════
print("\n" + "="*80)
print("5. 롱/숏 분리 성과 (1σ, 1h)")
print("="*80)

long_trades = []
short_trades = []
for _, row in td.iterrows():
    if row['vix_pct'] > thresh_1s:
        short_trades.append(-row['btc_1h'])
    elif row['vix_pct'] < -thresh_1s:
        long_trades.append(row['btc_1h'])

long_arr = np.array(long_trades)
short_arr = np.array(short_trades)

for side, arr in [('Long (VIX급락)', long_arr), ('Short (VIX급등)', short_arr)]:
    cum = np.cumsum(arr)
    total = cum[-1]*100
    avg = arr.mean()*100
    wr = (arr>0).mean()*100
    dd = np.min(cum - np.maximum.accumulate(cum))*100
    gp = arr[arr>0].sum()
    gl = abs(arr[arr<0].sum())
    pf = gp/gl if gl>0 else np.inf
    print(f"\n  [{side}] n={len(arr)}")
    print(f"    Total Return: {total:+.1f}%")
    print(f"    Mean/거래: {avg:+.4f}%")
    print(f"    승률: {wr:.1f}%")
    print(f"    Profit Factor: {pf:.2f}")
    print(f"    최대낙폭: {dd:+.2f}%")
    print(f"    최대 단일 수익: {arr.max()*100:+.4f}%")
    print(f"    최대 단일 손실: {arr.min()*100:+.4f}%")

# ══════════════════════════════════════════════════════
# 6. By VIX Level Strategy 성과
# ══════════════════════════════════════════════════════
print("\n" + "="*80)
print("6. By VIX Level Strategy 성과 (1σ, 1h)")
print("="*80)

for regime, cond in [("VIX<15", td['vix']<15),
                      ("VIX 15-18", (td['vix']>=15)&(td['vix']<18)),
                      ("VIX 18-25", (td['vix']>=18)&(td['vix']<25)),
                      ("VIX>=25", td['vix']>=25)]:
    sub = td[cond]
    trades = []
    for _, row in sub.iterrows():
        if row['vix_pct'] > thresh_1s:
            trades.append(-row['btc_1h'])
        elif row['vix_pct'] < -thresh_1s:
            trades.append(row['btc_1h'])
    if len(trades) == 0:
        print(f"  [{regime}] 거래 없음")
        continue
    arr = np.array(trades)
    cum = np.cumsum(arr)
    total = cum[-1]*100
    avg = arr.mean()*100
    wr = (arr>0).mean()*100
    dd = np.min(cum - np.maximum.accumulate(cum))*100
    print(f"  [{regime:>8s}] n={len(arr):>4d}  Total Return={total:>+7.1f}%  Mean={avg:>+8.4f}%  승률={wr:>5.1f}%  최대낙폭={dd:>+6.2f}%")

# ══════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 3, figsize=(22, 18))

# 1. 진입기준별 Total Return 비교
ax = axes[0, 0]
names = list(results_by_thresh.keys())
totals = [results_by_thresh[n]['total'] for n in names]
ns = [results_by_thresh[n]['n'] for n in names]
bars = ax.bar(names, totals, color=['#3498db','#2ecc71','#e74c3c','#f39c12','#9b59b6'])
for bar, n in zip(bars, ns):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'n={n}', ha='center', fontsize=8)
ax.set_ylabel('Total Return (%)'); ax.set_title('By Entry Criteria Total Return (1h Hold)')
ax.axhline(bh_ret, color='red', linestyle='--', label=f'B&H {bh_ret:.0f}%')
ax.legend()

# 2. 홀딩시간별 Total Return
ax = axes[0, 1]
hlabels = list(hold_results.keys())
htotals = [hold_results[h]['total'] for h in hlabels]
ax.bar(hlabels, htotals, color=['#e74c3c' if t > 0 else '#bdc3c7' for t in htotals])
ax.set_ylabel('Total Return (%)'); ax.set_title('By Hold Duration Total Return (1σ)')
ax.tick_params(axis='x', rotation=45)

# 3. Buy&Hold vs Strategy 누적수익
ax = axes[0, 2]
# B&H 곡선 (일별 기준)
bh_prices = td.groupby(td['timestamp'].dt.date)['btc_price'].last()
bh_cum = (bh_prices / bh_prices.iloc[0] - 1) * 100
ax.plot(bh_cum.values, 'b-', label=f'Buy & Hold ({bh_ret:.0f}%)', linewidth=1)
# Strategy 곡선
strat_cum_pct = np.cumsum(strat_trades) * 100
# 시간축 맞추기: Trade Day 기준으로 재매핑
strat_x = np.linspace(0, len(bh_cum)-1, len(strat_cum_pct))
ax.plot(strat_x, strat_cum_pct, 'r-', label=f'Strategy 1σ/1h ({strat_cum_pct[-1]:.0f}%)', linewidth=1)
ax.set_xlabel('Trade Day'); ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Buy & Hold vs Strategy'); ax.legend()

# 4. Monthly Returns 막대
ax = axes[1, 0]
months_str = [str(p) for p in monthly.index]
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in monthly['total_ret_pct']]
ax.bar(range(len(months_str)), monthly['total_ret_pct'], color=colors)
ax.set_xticks(range(0, len(months_str), 3))
ax.set_xticklabels([months_str[i] for i in range(0, len(months_str), 3)], rotation=45, fontsize=7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Monthly Return (%)'); ax.set_title('Monthly Returns')

# 5. 롱/숏 누적
ax = axes[1, 1]
long_cum = np.cumsum(long_arr) * 100
short_cum = np.cumsum(short_arr) * 100
ax.plot(long_cum, 'b-', label=f'Long ({long_cum[-1]:+.0f}%, n={len(long_arr)})', linewidth=1)
ax.plot(short_cum, 'r-', label=f'Short ({short_cum[-1]:+.0f}%, n={len(short_arr)})', linewidth=1)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Trade Number'); ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Long vs Short Cumulative'); ax.legend()

# 6. Return Distribution
ax = axes[1, 2]
ax.hist(strat_trades*100, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linewidth=1)
ax.axvline(strat_trades.mean()*100, color='green', linewidth=2, label=f'Mean {strat_trades.mean()*100:+.3f}%')
ax.set_xlabel('Return per Trade (%)'); ax.set_ylabel('Frequency')
ax.set_title('Return Distribution (1σ, 1h)'); ax.legend()

# 7. 1σ 1h Cumulative Return 곡선
ax = axes[2, 0]
ax.plot(strat_cum_pct, 'k-', linewidth=0.8)
ax.fill_between(range(len(strat_cum_pct)), strat_cum_pct, alpha=0.1)
ax.axhline(0, color='red', linewidth=0.5)
ax.set_xlabel('Trade Number'); ax.set_ylabel('Cumulative Return (%)')
ax.set_title(f'Strategy Cumulative Return (1σ, 1h, n={len(strat_trades)})')

# 8. 승률 by 기준
ax = axes[2, 1]
wrs = [results_by_thresh[n]['winrate'] for n in names]
ax.bar(names, wrs, color='#f39c12')
ax.set_ylabel('Win Rate (%)'); ax.set_title('Win Rate by Entry Criteria')
ax.axhline(50, color='red', linestyle='--')
for i, (wr, n) in enumerate(zip(wrs, ns)):
    ax.text(i, wr+0.5, f'{wr:.1f}%', ha='center', fontsize=9)

# 9. Drawdown
ax = axes[2, 2]
cum_line = np.cumsum(strat_trades)
running_max = np.maximum.accumulate(cum_line)
drawdown = (cum_line - running_max) * 100
ax.fill_between(range(len(drawdown)), drawdown, color='red', alpha=0.3)
ax.plot(drawdown, 'r-', linewidth=0.5)
ax.set_xlabel('Trade Number'); ax.set_ylabel('Drawdown (%)')
ax.set_title('Strategy Drawdown')

plt.suptitle(f'VIX to BTC Trading Strategy Analysis ({start_date.date()} ~ {end_date.date()})', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('vix_strategy_full.png', dpi=150, bbox_inches='tight')
print("\n저장: vix_strategy_full.png")
