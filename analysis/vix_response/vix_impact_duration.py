import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_parquet('data/vix_btc_response.parquet')

time_cols = ['btc_1m','btc_5m','btc_10m','btc_15m','btc_20m','btc_30m',
             'btc_45m','btc_1h','btc_90m','btc_2h','btc_3h','btc_4h',
             'btc_6h','btc_8h','btc_12h','btc_24h']
time_labels = ['1m','5m','10m','15m','20m','30m','45m','1h','90m','2h','3h','4h','6h','8h','12h','24h']
time_minutes = [1,5,10,15,20,30,45,60,90,120,180,240,360,480,720,1440]

# ── 1. 시차별 상관계수 + p-value ──
print("="*70)
print("1. VIX Change율 vs BTC Return — 시간대별 상관분석")
print("="*70)
corrs = []
for col, label in zip(time_cols, time_labels):
    r, p = stats.pearsonr(df['vix_pct'].dropna(), df[col].dropna())
    rs, ps = stats.spearmanr(df['vix_pct'].dropna(), df[col].dropna())
    corrs.append({'시간대': label, 'Pearson_r': r, 'p_value': p, 'Spearman_r': rs, 'sp_p': ps})
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {label:>4s}  Pearson r={r:+.4f} (p={p:.4f}){sig}   Spearman r={rs:+.4f} (p={ps:.4f})")

corr_df = pd.DataFrame(corrs)

# ── 2. VIX Spike/급락 이벤트 분석 ──
print("\n" + "="*70)
print("2. Event Study — VIX 급변동 시 BTC 반응 (|vix_pct| > 1σ)")
print("="*70)

threshold = df['vix_pct'].std()
vix_spike = df[df['vix_pct'] > threshold]   # VIX Spike
vix_drop = df[df['vix_pct'] < -threshold]   # VIX Drop

print(f"\n  VIX Spike 이벤트: {len(vix_spike)}건, VIX Drop 이벤트: {len(vix_drop)}건")
print(f"  threshold: ±{threshold:.4f} ({threshold*100:.2f}%)")

print("\n  [VIX Spike → BTC Mean 수익률]")
for col, label in zip(time_cols, time_labels):
    mean_ret = vix_spike[col].mean()
    t_stat, p_val = stats.ttest_1samp(vix_spike[col].dropna(), 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"    {label:>4s}  mean={mean_ret*100:+.4f}%  t={t_stat:+.2f}  p={p_val:.4f}{sig}")

print("\n  [VIX Drop → BTC Mean 수익률]")
for col, label in zip(time_cols, time_labels):
    mean_ret = vix_drop[col].mean()
    t_stat, p_val = stats.ttest_1samp(vix_drop[col].dropna(), 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"    {label:>4s}  mean={mean_ret*100:+.4f}%  t={t_stat:+.2f}  p={p_val:.4f}{sig}")

# ── 3. By VIX Level 분석 ──
print("\n" + "="*70)
print("3. Correlation by VIX Level 분석")
print("="*70)

for regime, cond in [("Low VIX (<20)", df['vix']<20), 
                      ("Mid VIX (20-30)", (df['vix']>=20)&(df['vix']<30)),
                      ("High VIX (≥30)", df['vix']>=30)]:
    sub = df[cond]
    print(f"\n  [{regime}] n={len(sub)}")
    for col, label in zip(time_cols[:8], time_labels[:8]):  # 1m~90m
        r, p = stats.pearsonr(sub['vix_pct'], sub[col])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {label:>4s}  r={r:+.4f} (p={p:.4f}){sig}")

# ── 4. 시각화 ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4-1. 상관계수 decay
ax = axes[0, 0]
ax.bar(range(len(time_labels)), corr_df['Pearson_r'], color=['#e74c3c' if p < 0.05 else '#bdc3c7' for p in corr_df['p_value']])
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45)
ax.set_ylabel('Pearson r')
ax.set_title('VIX Change vs BTC Return Correlation (red=p<0.05)')

# 4-2. Event study — VIX Spike 시 BTC 누적반응
ax = axes[0, 1]
spike_means = [vix_spike[col].mean()*100 for col in time_cols]
drop_means = [vix_drop[col].mean()*100 for col in time_cols]
ax.plot(range(len(time_labels)), spike_means, 'r-o', label='VIX Spike', markersize=4)
ax.plot(range(len(time_labels)), drop_means, 'b-o', label='VIX Drop', markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45)
ax.set_ylabel('BTC Mean Return (%)')
ax.set_title('BTC Response Path After VIX Event')
ax.legend()

# 4-3. p-value decay (log scale)
ax = axes[1, 0]
ax.semilogy(range(len(time_labels)), corr_df['p_value'], 'ko-', markersize=5)
ax.axhline(0.05, color='red', linestyle='--', label='p=0.05')
ax.axhline(0.01, color='orange', linestyle='--', label='p=0.01')
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45)
ax.set_ylabel('p-value (log)')
ax.set_title('Correlation p-value Trend')
ax.legend()

# 4-4. VIX Spike event — Individual Paths 산점
ax = axes[1, 1]
# 2σ 이상 극단 이벤트
extreme = df[df['vix_pct'] > 2*threshold]
for _, row in extreme.iterrows():
    path = [row[col]*100 for col in time_cols]
    ax.plot(range(len(time_labels)), path, alpha=0.3, linewidth=0.8)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45)
ax.set_ylabel('BTC Return (%)')
ax.set_title(f'VIX Extreme Spike (>2σ, n={len(extreme)}) Individual Paths')

plt.tight_layout()
plt.savefig('vix_impact_duration_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ 차트 저장: vix_impact_duration_analysis.png")
