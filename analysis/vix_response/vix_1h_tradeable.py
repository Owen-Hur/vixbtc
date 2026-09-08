import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_parquet('data/vix_1h_btc_response.parquet')
mkt = df[df['is_market_hours']].copy()

# ══════════════════════════════════════════════════════
# 핵심: VIX가 변하는 1시간 동안 BTC도 이미 움직였다.
# "VIX Change 확인 후 진입"하면 남은 수익은 얼마인가?
#
# VIX Change 구간: t-1h → t (예: 9:00→10:00)
# BTC 동시 변화: t-1h → t (이미 발생, 진입 불가)
# BTC 잔여 반응: t → t+Xm (이것만 트레이딩 가능)
#
# 지금 btc_Xm이 바로 t → t+Xm이므로,
# 사실 이미 "진입 후 수익"을 측정하고 있다.
# 
# 하지만 VIX는 1시간 단위로 관찰하는데,
# BTC 반응의 일부는 VIX가 변하는 그 1시간 안에
# 이미 일어났을 수 있다.
#
# 이를 분리하기 위해:
# btc_during = t-1h → t (VIX 변동 중 BTC 변화)
# btc_after_Xm = t → t+Xm (VIX 확인 후 BTC 변화)
# 
# btc_during이 이미 크다면, btc_after는 작을 것
# ══════════════════════════════════════════════════════

# BTC 1분봉 로드
import os
btc_dir = 'data/btc_1m_24h/'
btc_files = sorted([f for f in os.listdir(btc_dir) if f >= 'btc_1m_2024'])
btc = pd.concat([pd.read_parquet(os.path.join(btc_dir, f))[['close']] for f in btc_files]).sort_index()
btc = btc[~btc.index.duplicated(keep='first')]

# VIX 변동 중 BTC 변화 (t-1h → t) 계산
import pytz
records = []
for _, row in mkt.iterrows():
    ts = row['timestamp']
    ts_1h_ago = ts - pd.Timedelta(hours=1)
    
    if ts not in btc.index or ts_1h_ago not in btc.index:
        continue
    
    btc_before = btc.loc[ts_1h_ago, 'close']
    btc_at = btc.loc[ts, 'close']
    btc_during = (btc_at - btc_before) / btc_before  # VIX 변동 중 BTC 변화
    
    records.append({
        'timestamp': ts,
        'vix': row['vix'],
        'vix_pct': row['vix_pct'],
        'btc_during': btc_during,  # 이미 발생 (진입 불가)
        'btc_1m': row['btc_1m'],   # 이후 (트레이딩 가능)
        'btc_5m': row['btc_5m'],
        'btc_10m': row['btc_10m'],
        'btc_15m': row['btc_15m'],
        'btc_20m': row['btc_20m'],
        'btc_30m': row['btc_30m'],
        'btc_45m': row['btc_45m'],
        'btc_1h': row['btc_1h'],
        'btc_90m': row.get('btc_90m', np.nan),
        'btc_2h': row.get('btc_2h', np.nan),
        'btc_3h': row.get('btc_3h', np.nan),
        'btc_4h': row.get('btc_4h', np.nan),
    })

td = pd.DataFrame(records)
print(f"트레이딩 분석 데이터: {len(td)}건")

# ══════════════════════════════════════════════════════
# 1. VIX 변동 중 vs 이후 — 상관 분해
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. VIX Change율 vs BTC 반응 — 동시 vs 이후 분해")
print("="*60)

r_during, p_during = stats.pearsonr(td['vix_pct'], td['btc_during'])
print(f"  [동시] VIX 변동 중 BTC (t-1h→t):  r={r_during:+.4f} p={p_during:.6f}")

after_cols = ['btc_1m','btc_5m','btc_10m','btc_15m','btc_20m','btc_30m',
              'btc_45m','btc_1h','btc_90m','btc_2h','btc_3h','btc_4h']
after_labels = ['1m','5m','10m','15m','20m','30m','45m','1h','90m','2h','3h','4h']

print(f"\n  [이후] VIX 확인 후 BTC (t→t+Xm):")
for col, label in zip(after_cols, after_labels):
    valid = td[['vix_pct', col]].dropna()
    r, p = stats.pearsonr(valid['vix_pct'], valid[col])
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"    {label:>4s}  r={r:+.4f} p={p:.6f}{sig}")

# ══════════════════════════════════════════════════════
# 2. 동시 BTC를 통제한 편상관
#    VIX가 BTC에 미치는 "추가" 영향 (이미 반영된 부분 제거)
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. 동시 BTC를 통제한 편상관 (순수 잔여 효과)")
print("="*60)

def partial_corr(x, y, z):
    from numpy.linalg import lstsq
    z_arr = np.column_stack([z, np.ones(len(z))])
    x_resid = x - z_arr @ lstsq(z_arr, x, rcond=None)[0]
    y_resid = y - z_arr @ lstsq(z_arr, y, rcond=None)[0]
    return stats.pearsonr(x_resid, y_resid)

for col, label in zip(after_cols, after_labels):
    valid = td[['vix_pct', col, 'btc_during']].dropna()
    raw_r, _ = stats.pearsonr(valid['vix_pct'], valid[col])
    part_r, part_p = partial_corr(valid['vix_pct'].values, valid[col].values, valid['btc_during'].values)
    sig = "***" if part_p<0.001 else "**" if part_p<0.01 else "*" if part_p<0.05 else ""
    print(f"  {label:>4s}  raw={raw_r:+.4f}  partial={part_r:+.4f} p={part_p:.6f}{sig}")

# ══════════════════════════════════════════════════════
# 3. VIX 급변 이벤트 — Expected Return on Entry After Confirmation
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. VIX 급변 이벤트 — Expected Return on Entry After Confirmation")
print("="*60)

sigma = td['vix_pct'].std()
spike = td[td['vix_pct'] > 2*sigma]
drop = td[td['vix_pct'] < -2*sigma]

print(f"  2σ = {sigma*100:.2f}%, 급등:{len(spike)}건, 급락:{len(drop)}건")

print(f"\n  [VIX Spike → BTC 숏 진입]")
print(f"  {'시간대':>4s}  {'Mean수익률':>10s}  {'승률':>6s}  {'t값':>6s}  {'p':>8s}  {'샤프(연)':>8s}")
for col, label in zip(after_cols, after_labels):
    s = spike[col].dropna()
    m = s.mean()
    win_rate = (s < 0).mean()  # 숏이므로 음수가 이익
    t, p = stats.ttest_1samp(s, 0)
    # 단순 샤프: mean/std * sqrt(연간 거래 횟수 추정)
    if s.std() > 0:
        sharpe = (m / s.std()) * np.sqrt(252)  # 대략적
    else:
        sharpe = 0
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {label:>4s}  {m*100:+.4f}%  {win_rate*100:.1f}%  {t:+.2f}  {p:.6f}{sig}  {sharpe:+.2f}")

print(f"\n  [VIX Drop → BTC 롱 진입]")
print(f"  {'시간대':>4s}  {'Mean수익률':>10s}  {'승률':>6s}  {'t값':>6s}  {'p':>8s}  {'샤프(연)':>8s}")
for col, label in zip(after_cols, after_labels):
    s = drop[col].dropna()
    m = s.mean()
    win_rate = (s > 0).mean()  # 롱이므로 양수가 이익
    t, p = stats.ttest_1samp(s, 0)
    if s.std() > 0:
        sharpe = (m / s.std()) * np.sqrt(252)
    else:
        sharpe = 0
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {label:>4s}  {m*100:+.4f}%  {win_rate*100:.1f}%  {t:+.2f}  {p:.6f}{sig}  {sharpe:+.2f}")

# ══════════════════════════════════════════════════════
# 4. Cumulative Return 시뮬레이션 (단순)
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. 단순 Strategy 시뮬레이션 (VIX 2σ 급변 시 진입)")
print("="*60)

for hold_col, hold_label in [('btc_30m','30m'), ('btc_1h','1h'), ('btc_2h','2h')]:
    # VIX Spike → 숏, VIX Drop → 롱
    trades = []
    for _, row in td.iterrows():
        if row['vix_pct'] > 2*sigma:
            trades.append(-row[hold_col])  # 숏: 부호 반전
        elif row['vix_pct'] < -2*sigma:
            trades.append(row[hold_col])   # 롱
    
    trades = np.array(trades)
    cum_ret = np.cumsum(trades)
    n_trades = len(trades)
    avg_ret = trades.mean() * 100
    win_rate = (trades > 0).mean() * 100
    max_dd = np.min(cum_ret - np.maximum.accumulate(cum_ret)) * 100
    total_ret = cum_ret[-1] * 100
    
    print(f"\n  [홀딩 {hold_label}] 거래 {n_trades}건")
    print(f"    총 수익: {total_ret:+.2f}%")
    print(f"    Mean 수익/거래: {avg_ret:+.4f}%")
    print(f"    승률: {win_rate:.1f}%")
    print(f"    최대 낙폭: {max_dd:.2f}%")

# ══════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Concurrent vs Subsequent Correlation
ax = axes[0, 0]
r_afters = [stats.pearsonr(td['vix_pct'], td[c].dropna())[0] for c in after_cols]
ax.bar(-1, r_during, color='gray', label='Concurrent (t-1h to t)')
ax.bar(range(len(after_labels)), r_afters, color='#e74c3c', label='Subsequent (t to t+X)')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks([-1]+list(range(len(after_labels))))
ax.set_xticklabels(['동시']+after_labels, rotation=45, fontsize=8)
ax.set_ylabel('Pearson r'); ax.set_title('Concurrent vs Subsequent Correlation')
ax.legend(fontsize=8)

# 2. Raw vs Partial
ax = axes[0, 1]
raws = [stats.pearsonr(td['vix_pct'], td[c])[0] for c in after_cols]
parts = []
for c in after_cols:
    v = td[['vix_pct',c,'btc_during']].dropna()
    pr, _ = partial_corr(v['vix_pct'].values, v[c].values, v['btc_during'].values)
    parts.append(pr)
ax.plot(range(len(after_labels)), raws, 'ro-', label='Raw', markersize=4)
ax.plot(range(len(after_labels)), parts, 'bs-', label='Partial (Concurrent Controlled)', markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(after_labels))); ax.set_xticklabels(after_labels, rotation=45)
ax.set_title('Raw vs Concurrent-Controlled Partial Correlation'); ax.legend(fontsize=8)

# 3. 급변 이벤트 후 기대수익
ax = axes[0, 2]
spike_m = [spike[c].mean()*100 for c in after_cols]
drop_m = [drop[c].mean()*100 for c in after_cols]
ax.plot(range(len(after_labels)), spike_m, 'r-o', label=f'VIX Spike (n={len(spike)})', markersize=4)
ax.plot(range(len(after_labels)), drop_m, 'b-o', label=f'VIX Drop (n={len(drop)})', markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(after_labels))); ax.set_xticklabels(after_labels, rotation=45)
ax.set_ylabel('BTC Mean Return (%)'); ax.set_title('Expected Return on Entry After Confirmation')
ax.legend(fontsize=8)

# 4-6. Cumulative Return 곡선
for idx, (hold_col, hold_label) in enumerate([('btc_30m','30m'), ('btc_1h','1h'), ('btc_2h','2h')]):
    ax = axes[1, idx]
    trades = []
    for _, row in td.iterrows():
        if row['vix_pct'] > 2*sigma:
            trades.append(-row[hold_col])
        elif row['vix_pct'] < -2*sigma:
            trades.append(row[hold_col])
    trades = np.array(trades)
    cum = np.cumsum(trades) * 100
    ax.plot(cum, 'k-', linewidth=0.8)
    ax.axhline(0, color='red', linewidth=0.5)
    ax.set_xlabel('Trade Number'); ax.set_ylabel('Cumulative Return (%)')
    ax.set_title(f'Cumulative Return (Hold {hold_label}, n={len(trades)})')

plt.suptitle('VIX Shock Detection to BTC Tradability Analysis', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('vix_1h_tradeable.png', dpi=150, bbox_inches='tight')
print("\n저장: vix_1h_tradeable.png")
