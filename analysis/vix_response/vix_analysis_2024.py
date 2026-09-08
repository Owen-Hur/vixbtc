import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

df_all = pd.read_parquet('data/vix_btc_response.parquet')
df_all['date'] = pd.to_datetime(df_all['date'])
df = df_all[df_all['date'] >= '2024-01-01'].copy()

time_cols = ['btc_1m','btc_5m','btc_10m','btc_15m','btc_20m','btc_30m',
             'btc_45m','btc_1h','btc_90m','btc_2h','btc_3h','btc_4h',
             'btc_6h','btc_8h','btc_12h','btc_24h']
time_labels = ['1m','5m','10m','15m','20m','30m','45m','1h','90m','2h','3h','4h','6h','8h','12h','24h']

print(f"데이터: {df['date'].min().date()} ~ {df['date'].max().date()}, {len(df)}건")
print(f"연도별: {df['year'].value_counts().sort_index().to_dict()}")
print(f"VIX Mean: {df['vix'].mean():.2f}, std: {df['vix'].std():.2f}")
print()

# ══════════════════════════════════════════════════════════════
# 1. 시차별 상관분석
# ══════════════════════════════════════════════════════════════
print("="*70)
print("1. VIX Change율 vs BTC Return — 시간대별 상관")
print("="*70)
corrs = []
for col, label in zip(time_cols, time_labels):
    r, p = stats.pearsonr(df['vix_pct'], df[col])
    rs, ps = stats.spearmanr(df['vix_pct'], df[col])
    corrs.append({'label': label, 'r': r, 'p': p, 'rs': rs, 'ps': ps})
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {label:>4s}  Pearson r={r:+.4f} (p={p:.4f}){sig}   Spearman r={rs:+.4f} (p={ps:.4f})")
corr_df = pd.DataFrame(corrs)

# ══════════════════════════════════════════════════════════════
# 2. Permutation Test
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. Permutation Test (10,000회)")
print("="*70)
np.random.seed(42)
N_PERM = 10000
perm_pvals = []
for col, label in zip(time_cols, time_labels):
    real_r, _ = stats.pearsonr(df['vix_pct'], df[col])
    perm_rs = np.array([stats.pearsonr(np.random.permutation(df['vix_pct'].values), df[col].values)[0] for _ in range(N_PERM)])
    perm_p = np.mean(np.abs(perm_rs) >= np.abs(real_r))
    perm_pvals.append(perm_p)
    sig = "***" if perm_p < 0.001 else "**" if perm_p < 0.01 else "*" if perm_p < 0.05 else "NS"
    print(f"  {label:>4s}  real_r={real_r:+.4f}  perm_p={perm_p:.4f}  [{sig}]")

# ══════════════════════════════════════════════════════════════
# 3. Dose-Response
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. Dose-Response — VIX 분위별 BTC 반응")
print("="*70)
df['vix_quintile'] = pd.qcut(df['vix_pct'], 5, labels=['Q1(급락)','Q2(소락)','Q3(보합)','Q4(소등)','Q5(급등)'])
quintiles = ['Q1(급락)','Q2(소락)','Q3(보합)','Q4(소등)','Q5(급등)']

target_cols = ['btc_10m','btc_30m','btc_1h','btc_2h','btc_4h','btc_8h','btc_12h']
target_labels = ['10m','30m','1h','2h','4h','8h','12h']

for col, label in zip(target_cols, target_labels):
    print(f"\n  [BTC {label}]")
    for q in quintiles:
        sub = df[df['vix_quintile']==q][col]
        print(f"    {q}: mean={sub.mean()*100:+.4f}%  (n={len(sub)})")
    groups = [df[df['vix_quintile']==q][col].values for q in quintiles]
    f_stat, anova_p = stats.f_oneway(*groups)
    sig = "***" if anova_p < 0.001 else "**" if anova_p < 0.01 else "*" if anova_p < 0.05 else "NS"
    print(f"    ANOVA F={f_stat:.2f}, p={anova_p:.4f} [{sig}]")

# ══════════════════════════════════════════════════════════════
# 4. Asymmetry
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("4. Asymmetry — VIX Up vs 하락")
print("="*70)
vix_up = df[df['vix_pct'] > 0]
vix_down = df[df['vix_pct'] < 0]
print(f"  VIX Up Day: {len(vix_up)}건, VIX Down Day: {len(vix_down)}건")

asym_data = []
print(f"\n  {'시간대':>4s}  {'상승 r':>8s} {'p':>8s}  {'하락 r':>8s} {'p':>8s}")
for col, label in zip(time_cols, time_labels):
    r_up, p_up = stats.pearsonr(vix_up['vix_pct'], vix_up[col])
    r_dn, p_dn = stats.pearsonr(vix_down['vix_pct'], vix_down[col])
    sig_up = "*" if p_up < 0.05 else " "
    sig_dn = "*" if p_dn < 0.05 else " "
    print(f"  {label:>4s}  {r_up:+.4f}{sig_up} {p_up:.4f}  {r_dn:+.4f}{sig_dn} {p_dn:.4f}")
    asym_data.append({'label': label, 'r_up': r_up, 'r_down': r_dn})

# ══════════════════════════════════════════════════════════════
# 5. By VIX Level
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("5. Correlation by VIX Level")
print("="*70)
for regime, cond in [("Low VIX (<18)", df['vix']<18),
                      ("Mid VIX (18-25)", (df['vix']>=18)&(df['vix']<25)),
                      ("High VIX (≥25)", df['vix']>=25)]:
    sub = df[cond]
    print(f"\n  [{regime}] n={len(sub)}")
    for col, label in zip(time_cols, time_labels):
        r, p = stats.pearsonr(sub['vix_pct'], sub[col])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {label:>4s}  r={r:+.4f} (p={p:.4f}){sig}")

# ══════════════════════════════════════════════════════════════
# 6. Event Study: VIX Spike(>1σ) 시 BTC
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("6. Event Study — VIX Spike/급락 시 BTC 반응")
print("="*70)
sigma = df['vix_pct'].std()
vix_spike = df[df['vix_pct'] > sigma]
vix_drop = df[df['vix_pct'] < -sigma]
print(f"  1σ = {sigma*100:.2f}%, 급등: {len(vix_spike)}건, 급락: {len(vix_drop)}건")

print("\n  [VIX Spike → BTC]")
for col, label in zip(time_cols, time_labels):
    mean_ret = vix_spike[col].mean()
    t_stat, p_val = stats.ttest_1samp(vix_spike[col], 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"    {label:>4s}  mean={mean_ret*100:+.4f}%  t={t_stat:+.2f}  p={p_val:.4f}{sig}")

print("\n  [VIX Drop → BTC]")
for col, label in zip(time_cols, time_labels):
    mean_ret = vix_drop[col].mean()
    t_stat, p_val = stats.ttest_1samp(vix_drop[col], 0)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"    {label:>4s}  mean={mean_ret*100:+.4f}%  t={t_stat:+.2f}  p={p_val:.4f}{sig}")

# ══════════════════════════════════════════════════════════════
# 7. Bootstrap CI
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("7. Bootstrap 95% CI")
print("="*70)
np.random.seed(42)
boot_results = []
for col, label in zip(time_cols, time_labels):
    real_r, _ = stats.pearsonr(df['vix_pct'], df[col])
    bs = [stats.pearsonr(df['vix_pct'].sample(len(df), replace=True).values, 
                          df[col].sample(len(df), replace=True).values)[0] for _ in range(N_PERM)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    excl = "✅" if hi < 0 or lo > 0 else "❌"
    print(f"  {label:>4s}  r={real_r:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {excl}")
    boot_results.append({'label': label, 'r': real_r, 'ci_lo': lo, 'ci_hi': hi})

# ══════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. 상관계수
ax = axes[0, 0]
colors = ['#e74c3c' if p < 0.05 else '#bdc3c7' for p in corr_df['p']]
ax.bar(range(len(time_labels)), corr_df['r'], color=colors)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, fontsize=8)
ax.set_ylabel('Pearson r')
ax.set_title('VIX to BTC Correlation (red=p<0.05)')

# 2. Dose-response
ax = axes[0, 1]
for col, label in zip(target_cols, target_labels):
    means = [df[df['vix_quintile']==q][col].mean()*100 for q in quintiles]
    ax.plot(quintiles, means, 'o-', label=f'BTC {label}', markersize=5)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('BTC Mean Return (%)')
ax.set_title('Dose-Response')
ax.legend(fontsize=7)
ax.tick_params(axis='x', rotation=30)

# 3. 비대칭
ax = axes[0, 2]
asym_df = pd.DataFrame(asym_data)
ax.plot(range(len(time_labels)), asym_df['r_up'], 'r-o', label='VIX Up', markersize=4)
ax.plot(range(len(time_labels)), asym_df['r_down'], 'b-s', label='VIX Down', markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, fontsize=8)
ax.set_title('Asymmetry')
ax.legend(fontsize=8)

# 4. Event study
ax = axes[1, 0]
spike_means = [vix_spike[col].mean()*100 for col in time_cols]
drop_means = [vix_drop[col].mean()*100 for col in time_cols]
ax.plot(range(len(time_labels)), spike_means, 'r-o', label='VIX Spike', markersize=4)
ax.plot(range(len(time_labels)), drop_means, 'b-o', label='VIX Drop', markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, fontsize=8)
ax.set_ylabel('BTC Mean Return (%)')
ax.set_title('Event Study: BTC After VIX Shock')
ax.legend()

# 5. By VIX Level
ax = axes[1, 1]
for regime, cond, color in [("VIX<18", df['vix']<18, '#2ecc71'),
                             ("VIX 18-25", (df['vix']>=18)&(df['vix']<25), '#f39c12'),
                             ("VIX≥25", df['vix']>=25, '#e74c3c')]:
    sub = df[cond]
    rs = [stats.pearsonr(sub['vix_pct'], sub[col])[0] for col in time_cols]
    ax.plot(range(len(time_labels)), rs, 'o-', color=color, label=f'{regime} (n={len(sub)})', markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, fontsize=8)
ax.set_title('Correlation by VIX Level')
ax.legend(fontsize=8)

# 6. Bootstrap CI
ax = axes[1, 2]
boot_df = pd.DataFrame(boot_results)
yerr_lo = np.clip(boot_df['r']-boot_df['ci_lo'], 0, None)
yerr_hi = np.clip(boot_df['ci_hi']-boot_df['r'], 0, None)
ax.errorbar(range(len(time_labels)), boot_df['r'],
            yerr=[yerr_lo, yerr_hi],
            fmt='ko-', markersize=5, capsize=3)
ax.axhline(0, color='red', linestyle='--')
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, fontsize=8)
ax.set_title('Bootstrap 95% CI')

plt.suptitle(f'VIX to BTC Impact Analysis (2024.01 ~ 2026.05, ETF Era, n={len(df)})', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('vix_analysis_2024.png', dpi=150, bbox_inches='tight')
print("\n✅ 차트 저장: vix_analysis_2024.png")
