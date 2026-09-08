"""
VIX 1시간봉 기준 반응 데이터로 (1) 전체 표본 (2) 장중(9~16시) (3) |변화율|>2σ 급변
이벤트의 시차별 상관을 비교한다.

입력: data/vix_1h_btc_response.parquet
출력: vix_1h_analysis.png
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_parquet('data/vix_1h_btc_response.parquet')

time_cols = ['btc_1m','btc_2m','btc_3m','btc_5m','btc_10m','btc_15m','btc_20m','btc_30m',
             'btc_45m','btc_1h','btc_90m','btc_2h','btc_3h','btc_4h']
time_labels = ['1m','2m','3m','5m','10m','15m','20m','30m','45m','1h','90m','2h','3h','4h']

print(f"데이터: {len(df)}건, {df['timestamp'].min()} ~ {df['timestamp'].max()}")

# ══════════════════════════════════════════════════════
# 1. All 상관
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. All — VIX 1시간 변화율 vs BTC Return")
print("="*60)
for col, label in zip(time_cols, time_labels):
    valid = df[['vix_pct', col]].dropna()
    r, p = stats.pearsonr(valid['vix_pct'], valid[col])
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {label:>4s}  r={r:+.4f} p={p:.4f}{sig}")

# ══════════════════════════════════════════════════════
# 2. Intraday만
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. Intraday(9~16시)만")
print("="*60)
mkt = df[df['is_market_hours']]
print(f"  n={len(mkt)}")
for col, label in zip(time_cols, time_labels):
    valid = mkt[['vix_pct', col]].dropna()
    r, p = stats.pearsonr(valid['vix_pct'], valid[col])
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {label:>4s}  r={r:+.4f} p={p:.4f}{sig}")

# ══════════════════════════════════════════════════════
# 3. VIX 급변 이벤트 (|변화율| > 2σ)
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. VIX 급변 이벤트 (Intraday, |변화율| > 2σ)")
print("="*60)
mkt_sigma = mkt['vix_pct'].std()
spike = mkt[mkt['vix_pct'] > 2*mkt_sigma]
drop = mkt[mkt['vix_pct'] < -2*mkt_sigma]
print(f"  2σ = {mkt_sigma*100:.2f}%, 급등:{len(spike)}건, 급락:{len(drop)}건")

print("\n  [VIX Spike → BTC]")
for col, label in zip(time_cols, time_labels):
    m = spike[col].mean()
    t, p = stats.ttest_1samp(spike[col].dropna(), 0)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"    {label:>4s}  mean={m*100:+.4f}%  t={t:+.2f} p={p:.4f}{sig}")

print("\n  [VIX Drop → BTC]")
for col, label in zip(time_cols, time_labels):
    m = drop[col].mean()
    t, p = stats.ttest_1samp(drop[col].dropna(), 0)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"    {label:>4s}  mean={m*100:+.4f}%  t={t:+.2f} p={p:.4f}{sig}")

# ══════════════════════════════════════════════════════
# 4. Permutation (Intraday)
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. Permutation Test — Intraday (5,000회)")
print("="*60)
np.random.seed(42)
for col, label in zip(time_cols, time_labels):
    valid = mkt[['vix_pct', col]].dropna()
    real_r, _ = stats.pearsonr(valid['vix_pct'], valid[col])
    perm_rs = np.array([stats.pearsonr(np.random.permutation(valid['vix_pct'].values), valid[col].values)[0] for _ in range(5000)])
    perm_p = np.mean(np.abs(perm_rs) >= np.abs(real_r))
    sig = "***" if perm_p<0.001 else "**" if perm_p<0.01 else "*" if perm_p<0.05 else "NS"
    print(f"  {label:>4s}  r={real_r:+.4f}  perm_p={perm_p:.4f} [{sig}]")

# ══════════════════════════════════════════════════════
# 5. By VIX Level (Intraday)
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. By VIX Level (Intraday)")
print("="*60)
for regime, cond in [("VIX<18", mkt['vix']<18),
                      ("VIX 18-25", (mkt['vix']>=18)&(mkt['vix']<25)),
                      ("VIX>=25", mkt['vix']>=25)]:
    sub = mkt[cond]
    print(f"\n  [{regime}] n={len(sub)}")
    for col, label in zip(time_cols, time_labels):
        valid = sub[['vix_pct', col]].dropna()
        if len(valid) < 10:
            print(f"    {label:>4s}  샘플 부족")
            continue
        r, p = stats.pearsonr(valid['vix_pct'], valid[col])
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        print(f"    {label:>4s}  r={r:+.4f} p={p:.4f}{sig}")

# ══════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. All vs Intraday Correlation
ax = axes[0, 0]
r_all = [stats.pearsonr(df[['vix_pct',c]].dropna()['vix_pct'], df[['vix_pct',c]].dropna()[c])[0] for c in time_cols]
r_mkt = [stats.pearsonr(mkt[['vix_pct',c]].dropna()['vix_pct'], mkt[['vix_pct',c]].dropna()[c])[0] for c in time_cols]
ax.plot(range(len(time_labels)), r_all, 'ko-', label='All', markersize=4)
ax.plot(range(len(time_labels)), r_mkt, 'rs-', label='Intraday', markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels))); ax.set_xticklabels(time_labels, rotation=45)
ax.set_ylabel('Pearson r'); ax.set_title('VIX 1h Change to BTC Correlation')
ax.legend()

# 2. Event Study
ax = axes[0, 1]
spike_m = [spike[c].mean()*100 for c in time_cols]
drop_m = [drop[c].mean()*100 for c in time_cols]
ax.plot(range(len(time_labels)), spike_m, 'r-o', label=f'VIX Spike (n={len(spike)})', markersize=4)
ax.plot(range(len(time_labels)), drop_m, 'b-o', label=f'VIX Drop (n={len(drop)})', markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels))); ax.set_xticklabels(time_labels, rotation=45)
ax.set_ylabel('BTC Mean Return (%)'); ax.set_title('BTC After VIX Shock (2σ)')
ax.legend(fontsize=8)

# 3. By VIX Level (Intraday)
ax = axes[0, 2]
for regime, cond, color in [("VIX<18", mkt['vix']<18, '#2ecc71'),
                             ("VIX 18-25", (mkt['vix']>=18)&(mkt['vix']<25), '#f39c12'),
                             ("VIX>=25", mkt['vix']>=25, '#e74c3c')]:
    sub = mkt[cond]
    rs = []
    for c in time_cols:
        v = sub[['vix_pct',c]].dropna()
        if len(v) >= 10:
            rs.append(stats.pearsonr(v['vix_pct'], v[c])[0])
        else:
            rs.append(np.nan)
    ax.plot(range(len(time_labels)), rs, 'o-', color=color, label=f'{regime} (n={len(sub)})', markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels))); ax.set_xticklabels(time_labels, rotation=45)
ax.set_title('By VIX Level (Intraday)'); ax.legend(fontsize=8)

# 4. 시간대별 상관 히트맵
ax = axes[1, 0]
hours = sorted(mkt['hour'].unique())
heatmap = []
for c in time_cols:
    row = []
    for h in hours:
        sub = mkt[mkt['hour']==h]
        v = sub[['vix_pct',c]].dropna()
        if len(v) >= 10:
            r, _ = stats.pearsonr(v['vix_pct'], v[c])
            row.append(r)
        else:
            row.append(np.nan)
    heatmap.append(row)
im = ax.imshow(np.array(heatmap), cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)
ax.set_xticks(range(len(hours))); ax.set_xticklabels(hours, fontsize=8)
ax.set_yticks(range(len(time_labels))); ax.set_yticklabels(time_labels, fontsize=8)
ax.set_xlabel('VIX Change Time (ET)'); ax.set_title('Time × BTC Response Time Correlation Heatmap')
plt.colorbar(im, ax=ax, shrink=0.8)

# 5. VIX 급변 크기 vs BTC 반응 산점도 (Intraday, btc_30m)
ax = axes[1, 1]
ax.scatter(mkt['vix_pct']*100, mkt['btc_30m']*100, alpha=0.1, s=5, color='black')
ax.axhline(0, color='red', linewidth=0.5); ax.axvline(0, color='red', linewidth=0.5)
ax.set_xlabel('VIX 1h Change (%)'); ax.set_ylabel('BTC 30m Return (%)')
ax.set_title('VIX Change vs BTC 30m (Intraday)')
ax.set_xlim(-15, 15); ax.set_ylim(-5, 5)

# 6. p-value Trend (Intraday)
ax = axes[1, 2]
ps = [stats.pearsonr(mkt[['vix_pct',c]].dropna()['vix_pct'], mkt[['vix_pct',c]].dropna()[c])[1] for c in time_cols]
ax.semilogy(range(len(time_labels)), ps, 'ko-', markersize=5)
ax.axhline(0.05, color='red', linestyle='--', label='p=0.05')
ax.axhline(0.01, color='orange', linestyle='--', label='p=0.01')
ax.set_xticks(range(len(time_labels))); ax.set_xticklabels(time_labels, rotation=45)
ax.set_ylabel('p-value (log)'); ax.set_title('Intraday Correlation p-value Trend')
ax.legend()

plt.suptitle(f'VIX 1h to BTC Response Analysis (2024.06~2026.05, Intraday n={len(mkt)})', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('vix_1h_analysis.png', dpi=150, bbox_inches='tight')
print("\n저장: vix_1h_analysis.png")
