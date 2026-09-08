"""
VIX 대신 실제 거래 가능한 VIXY(VIX ETF) 1분봉을 사용해, 직전 5분간 2σ 이상 급변한
장중 시점 이후 BTC 가 60분간 어떻게 움직이는지 측정한다(감지 시점에 이미 확정된
변화만 사용하므로 lookahead-free).

입력: data/vixy_1m/, data/btc_1m_24h/
출력: vixy_btc_analysis.png
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 1. VIXY 1분봉 로드 ──
vixy_is = pd.read_parquet('data/vixy_1m/VIXY_1m_IS_20240115_20251031.parquet')
vixy_oos = pd.read_parquet('data/vixy_1m/VIXY_1m_OOS_20251101_20260430.parquet')
vixy = pd.concat([vixy_is, vixy_oos]).sort_index()
vixy = vixy[~vixy.index.duplicated(keep='first')]
vixy.index = vixy.index.tz_convert('America/New_York')
print(f"VIXY: {vixy.index.min()} ~ {vixy.index.max()}, {len(vixy):,}건")

# ── 2. BTC 1분봉 로드 ──
btc_dir = 'data/btc_1m_24h/'
btc_files = sorted([f for f in os.listdir(btc_dir) if f >= 'btc_1m_2024'])
btc = pd.concat([pd.read_parquet(os.path.join(btc_dir, f))[['close']] for f in btc_files]).sort_index()
btc = btc[~btc.index.duplicated(keep='first')]
print(f"BTC:  {btc.index.min()} ~ {btc.index.max()}, {len(btc):,}건")

# ── 3. VIXY 급변 이벤트 감지 ──
# VIXY 5분 수익률 계산 (1분은 노이즈 심함, 5분이 현실적 감지 단위)
vixy_close = vixy[['close']].rename(columns={'close': 'vixy'})

# 5분 변화율
vixy_close['vixy_5m_pct'] = vixy_close['vixy'].pct_change(5)
# 10분 변화율
vixy_close['vixy_10m_pct'] = vixy_close['vixy'].pct_change(10)
# 15분 변화율
vixy_close['vixy_15m_pct'] = vixy_close['vixy'].pct_change(15)

vixy_close = vixy_close.dropna()

# Intraday만 (9:30~16:00 ET)
vixy_close['hour'] = vixy_close.index.hour
vixy_close['minute'] = vixy_close.index.minute
mkt = vixy_close[(vixy_close['hour'] >= 10) & (vixy_close['hour'] <= 15)].copy()  # 10~15시 (양쪽 마진)

print(f"Intraday VIXY: {len(mkt):,}건")
sigma_5m = mkt['vixy_5m_pct'].std()
sigma_10m = mkt['vixy_10m_pct'].std()
print(f"VIXY 5분 변화율 std: {sigma_5m*100:.3f}%")
print(f"VIXY 10분 변화율 std: {sigma_10m*100:.3f}%")

# ── 4. VIXY 급변 시점에서 BTC Return 측정 ──
# 로직: VIXY가 직전 5분간 2σ 이상 급변 → 그 시점부터 BTC Return 측정
# 
# 시간축:
#   t-5m ────── t (지금) ────── t+1m, t+2m, ... t+60m
#   [VIXY 급변]   [감지]         [BTC 반응 측정]
#
# VIXY 5분 변화는 t 시점에 이미 확정 → lookahead-free

btc_offsets = {
    'btc_1m': 1, 'btc_2m': 2, 'btc_3m': 3, 'btc_5m': 5,
    'btc_10m': 10, 'btc_15m': 15, 'btc_20m': 20, 'btc_30m': 30,
    'btc_45m': 45, 'btc_1h': 60, 'btc_90m': 90, 'btc_2h': 120,
}

# 2σ 이벤트 감지 (5분 기준)
thresh = 2 * sigma_5m
spike_times = mkt[mkt['vixy_5m_pct'] > thresh].index
drop_times = mkt[mkt['vixy_5m_pct'] < -thresh].index

# 연속 이벤트 제거 (30분 이내 중복 제거)
def deduplicate(times, gap_minutes=30):
    if len(times) == 0:
        return times
    result = [times[0]]
    for t in times[1:]:
        if (t - result[-1]).total_seconds() > gap_minutes * 60:
            result.append(t)
    return pd.DatetimeIndex(result)

spike_times = deduplicate(spike_times)
drop_times = deduplicate(drop_times)

print(f"\nVIXY Spike(>2σ, 중복제거): {len(spike_times)}건")
print(f"VIXY Drop(<-2σ, 중복제거): {len(drop_times)}건")

# BTC Return 측정
def measure_btc_response(event_times, vixy_df):
    records = []
    for ts in event_times:
        if ts not in btc.index:
            continue
        base = btc.loc[ts, 'close']
        rec = {
            'timestamp': ts,
            'vixy': vixy_df.loc[ts, 'vixy'] if ts in vixy_df.index else np.nan,
            'vixy_5m_pct': vixy_df.loc[ts, 'vixy_5m_pct'] if ts in vixy_df.index else np.nan,
        }
        for col, mins in btc_offsets.items():
            target = ts + pd.Timedelta(minutes=mins)
            if target in btc.index:
                rec[col] = (btc.loc[target, 'close'] - base) / base
            else:
                rec[col] = np.nan
        records.append(rec)
    return pd.DataFrame(records)

spike_df = measure_btc_response(spike_times, vixy_close)
drop_df = measure_btc_response(drop_times, vixy_close)

print(f"매칭 — 급등: {len(spike_df)}건, 급락: {len(drop_df)}건")

# ── 5. 분석 ──
btc_cols = list(btc_offsets.keys())
btc_labels = ['1m','2m','3m','5m','10m','15m','20m','30m','45m','1h','90m','2h']

print("\n" + "="*70)
print("1. After VIXY Spike BTC 반응 (VIX Spike = 공포 증가 → BTC 하락?)")
print("="*70)
for col, label in zip(btc_cols, btc_labels):
    s = spike_df[col].dropna()
    if len(s) < 5: continue
    m = s.mean()
    t, p = stats.ttest_1samp(s, 0)
    wr = (s < 0).mean() * 100  # 하락 비율
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {label:>4s}  mean={m*100:+.4f}%  하락률={wr:.0f}%  t={t:+.2f} p={p:.4f}{sig}")

print("\n" + "="*70)
print("2. After VIXY Drop BTC 반응 (VIX Drop = 공포 감소 → BTC 상승?)")
print("="*70)
for col, label in zip(btc_cols, btc_labels):
    s = drop_df[col].dropna()
    if len(s) < 5: continue
    m = s.mean()
    t, p = stats.ttest_1samp(s, 0)
    wr = (s > 0).mean() * 100  # 상승 비율
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {label:>4s}  mean={m*100:+.4f}%  상승률={wr:.0f}%  t={t:+.2f} p={p:.4f}{sig}")

# ── 6. All 상관 (연속 데이터) ──
print("\n" + "="*70)
print("3. 연속 상관: 직전 5분 VIXY 변화 → 이후 BTC Return (Intraday)")
print("="*70)

# 매 분마다 계산하면 너무 많으므로 5분 간격으로 샘플링
sample_times = mkt.index[::5]  # 5분 간격
cont_records = []
for ts in sample_times:
    if ts not in btc.index:
        continue
    base = btc.loc[ts, 'close']
    rec = {'vixy_5m_pct': mkt.loc[ts, 'vixy_5m_pct']}
    for col, mins in btc_offsets.items():
        target = ts + pd.Timedelta(minutes=mins)
        if target in btc.index:
            rec[col] = (btc.loc[target, 'close'] - base) / base
        else:
            rec[col] = np.nan
    cont_records.append(rec)

cont_df = pd.DataFrame(cont_records)
print(f"  샘플: {len(cont_df):,}건")

for col, label in zip(btc_cols, btc_labels):
    valid = cont_df[['vixy_5m_pct', col]].dropna()
    r, p = stats.pearsonr(valid['vixy_5m_pct'], valid[col])
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {label:>4s}  r={r:+.4f} p={p:.6f}{sig}")

# ── 7. 10min Detection 기준으로도 ──
print("\n" + "="*70)
print("4. 직전 10분 VIXY 변화 → 이후 BTC (Intraday)")
print("="*70)
sample_times_10 = mkt.index[::10]
cont10 = []
for ts in sample_times_10:
    if ts not in btc.index:
        continue
    base = btc.loc[ts, 'close']
    rec = {'vixy_10m_pct': mkt.loc[ts, 'vixy_10m_pct']}
    for col, mins in btc_offsets.items():
        target = ts + pd.Timedelta(minutes=mins)
        if target in btc.index:
            rec[col] = (btc.loc[target, 'close'] - base) / base
        else:
            rec[col] = np.nan
    cont10.append(rec)

cont10_df = pd.DataFrame(cont10)
print(f"  샘플: {len(cont10_df):,}건")

for col, label in zip(btc_cols, btc_labels):
    valid = cont10_df[['vixy_10m_pct', col]].dropna()
    r, p = stats.pearsonr(valid['vixy_10m_pct'], valid[col])
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {label:>4s}  r={r:+.4f} p={p:.6f}{sig}")

# ── 시각화 ──
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. After VIXY Spike BTC Path
ax = axes[0, 0]
spike_means = [spike_df[c].mean()*100 for c in btc_cols]
drop_means = [drop_df[c].mean()*100 for c in btc_cols]
ax.plot(range(len(btc_labels)), spike_means, 'r-o', label=f'VIXY Spike (n={len(spike_df)})', markersize=4)
ax.plot(range(len(btc_labels)), drop_means, 'b-o', label=f'VIXY Drop (n={len(drop_df)})', markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(btc_labels))); ax.set_xticklabels(btc_labels, rotation=45)
ax.set_ylabel('BTC Mean Return (%)'); ax.set_title('BTC After VIXY Shock (2σ, 5min)')
ax.legend(fontsize=8)

# 2. 연속 상관 (5분)
ax = axes[0, 1]
r5 = [stats.pearsonr(cont_df[['vixy_5m_pct',c]].dropna()['vixy_5m_pct'], cont_df[['vixy_5m_pct',c]].dropna()[c])[0] for c in btc_cols]
p5 = [stats.pearsonr(cont_df[['vixy_5m_pct',c]].dropna()['vixy_5m_pct'], cont_df[['vixy_5m_pct',c]].dropna()[c])[1] for c in btc_cols]
colors = ['#e74c3c' if p<0.05 else '#bdc3c7' for p in p5]
ax.bar(range(len(btc_labels)), r5, color=colors)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(btc_labels))); ax.set_xticklabels(btc_labels, rotation=45)
ax.set_ylabel('Pearson r'); ax.set_title('Previous 5min VIXY to Subsequent BTC (red=p<0.05)')

# 3. 연속 상관 (10분)
ax = axes[0, 2]
r10 = [stats.pearsonr(cont10_df[['vixy_10m_pct',c]].dropna()['vixy_10m_pct'], cont10_df[['vixy_10m_pct',c]].dropna()[c])[0] for c in btc_cols]
p10 = [stats.pearsonr(cont10_df[['vixy_10m_pct',c]].dropna()['vixy_10m_pct'], cont10_df[['vixy_10m_pct',c]].dropna()[c])[1] for c in btc_cols]
colors10 = ['#e74c3c' if p<0.05 else '#bdc3c7' for p in p10]
ax.bar(range(len(btc_labels)), r10, color=colors10)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(btc_labels))); ax.set_xticklabels(btc_labels, rotation=45)
ax.set_ylabel('Pearson r'); ax.set_title('Previous 10min VIXY to Subsequent BTC (red=p<0.05)')

# 4. VIXY Spike 이벤트 Individual Paths
ax = axes[1, 0]
for _, row in spike_df.iterrows():
    path = [row[c]*100 for c in btc_cols]
    ax.plot(range(len(btc_labels)), path, alpha=0.15, linewidth=0.5, color='red')
ax.plot(range(len(btc_labels)), spike_means, 'k-o', markersize=4, linewidth=2, label='Mean')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(btc_labels))); ax.set_xticklabels(btc_labels, rotation=45)
ax.set_title(f'After VIXY Spike BTC Individual Paths (n={len(spike_df)})'); ax.legend()

# 5. VIXY Drop 이벤트 Individual Paths
ax = axes[1, 1]
for _, row in drop_df.iterrows():
    path = [row[c]*100 for c in btc_cols]
    ax.plot(range(len(btc_labels)), path, alpha=0.15, linewidth=0.5, color='blue')
ax.plot(range(len(btc_labels)), drop_means, 'k-o', markersize=4, linewidth=2, label='Mean')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(btc_labels))); ax.set_xticklabels(btc_labels, rotation=45)
ax.set_title(f'After VIXY Drop BTC Individual Paths (n={len(drop_df)})'); ax.legend()

# 6. p-value Trend 비교
ax = axes[1, 2]
ax.semilogy(range(len(btc_labels)), p5, 'ro-', label='5min Detection', markersize=4)
ax.semilogy(range(len(btc_labels)), p10, 'bs-', label='10min Detection', markersize=4)
ax.axhline(0.05, color='red', linestyle='--', alpha=0.5)
ax.axhline(0.01, color='orange', linestyle='--', alpha=0.5)
ax.set_xticks(range(len(btc_labels))); ax.set_xticklabels(btc_labels, rotation=45)
ax.set_ylabel('p-value (log)'); ax.set_title('p-value Trend')
ax.legend()

plt.suptitle('VIXY 1m to BTC Effective Duration Analysis (2024.01~2026.04, lookahead-free)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('vixy_btc_analysis.png', dpi=150, bbox_inches='tight')
print("\n저장: vixy_btc_analysis.png")
