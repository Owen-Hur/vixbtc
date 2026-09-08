import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_parquet('data/vix_btc_response.parquet')
df['date'] = pd.to_datetime(df['date'])

time_cols = ['btc_1m','btc_5m','btc_10m','btc_15m','btc_20m','btc_30m',
             'btc_45m','btc_1h','btc_90m','btc_2h','btc_3h','btc_4h',
             'btc_6h','btc_8h','btc_12h','btc_24h']
time_labels = ['1m','5m','10m','15m','20m','30m','45m','1h','90m','2h','3h','4h','6h','8h','12h','24h']

print(f"데이터: {df['date'].min().date()} ~ {df['date'].max().date()}, {len(df)}건")
print()

# ══════════════════════════════════════════════════════════════
# 1. 기본 시차별 상관분석
# ══════════════════════════════════════════════════════════════
print("="*70)
print("1. VIX Change율 vs BTC Return — 시간대별 상관분석")
print("="*70)
corrs = []
for col, label in zip(time_cols, time_labels):
    valid = df[['vix_pct', col]].dropna()
    r, p = stats.pearsonr(valid['vix_pct'], valid[col])
    rs, ps = stats.spearmanr(valid['vix_pct'], valid[col])
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
    valid = df[['vix_pct', col]].dropna()
    real_r, _ = stats.pearsonr(valid['vix_pct'], valid[col])
    perm_rs = []
    for _ in range(N_PERM):
        shuffled = np.random.permutation(valid['vix_pct'].values)
        r, _ = stats.pearsonr(shuffled, valid[col].values)
        perm_rs.append(r)
    perm_p = np.mean(np.abs(np.array(perm_rs)) >= np.abs(real_r))
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

target_cols = ['btc_30m','btc_1h','btc_2h','btc_4h','btc_8h']
target_labels = ['30m','1h','2h','4h','8h']
quintiles = ['Q1(급락)','Q2(소락)','Q3(보합)','Q4(소등)','Q5(급등)']

for col, label in zip(target_cols, target_labels):
    print(f"\n  [BTC {label}]")
    for q in quintiles:
        sub = df[df['vix_quintile']==q][col].dropna()
        print(f"    {q}: mean={sub.mean()*100:+.4f}%  (n={len(sub)})")
    groups = [df[df['vix_quintile']==q][col].dropna().values for q in quintiles]
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
    r_up, p_up = stats.pearsonr(vix_up['vix_pct'], vix_up[col].fillna(0))
    r_dn, p_dn = stats.pearsonr(vix_down['vix_pct'], vix_down[col].fillna(0))
    sig_up = "*" if p_up < 0.05 else " "
    sig_dn = "*" if p_dn < 0.05 else " "
    print(f"  {label:>4s}  {r_up:+.4f}{sig_up} {p_up:.4f}  {r_dn:+.4f}{sig_dn} {p_dn:.4f}")
    asym_data.append({'label': label, 'r_up': r_up, 'r_down': r_dn})

# ══════════════════════════════════════════════════════════════
# 5. 연도별 안정성
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("5. 연도별 상관계수")
print("="*70)
years = sorted(df['year'].unique())
print(f"\n  {'시간대':>4s}", end="")
for y in years:
    print(f"  {y:>7d}", end="")
print(f"  {'All':>7s}")

yearly_heatmap = []
for col, label in zip(time_cols, time_labels):
    print(f"  {label:>4s}", end="")
    row = []
    for y in years:
        sub = df[df['year']==y]
        valid = sub[['vix_pct',col]].dropna()
        r, p = stats.pearsonr(valid['vix_pct'], valid[col])
        marker = "*" if p < 0.05 else " "
        print(f"  {r:+.3f}{marker}", end="")
        row.append(r)
    valid_all = df[["vix_pct",col]].dropna(); r_all, _ = stats.pearsonr(valid_all["vix_pct"], valid_all[col])
    print(f"  {r_all:+.4f}")
    yearly_heatmap.append(row)

# ══════════════════════════════════════════════════════════════
# 6. Out-of-sample: 전반(2020-2022) vs 후반(2023-2026)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("6. Out-of-sample — 전반(2020-2022) vs 후반(2023-2026)")
print("="*70)
df_early = df[df['year'] <= 2022]
df_late = df[df['year'] >= 2023]
print(f"  전반: {len(df_early)}건, 후반: {len(df_late)}건")

oos_data = []
print(f"\n  {'시간대':>4s}  {'전반 r':>8s} {'p':>8s}  {'후반 r':>8s} {'p':>8s}")
for col, label in zip(time_cols, time_labels):
    e = df_early[['vix_pct',col]].dropna()
    l = df_late[['vix_pct',col]].dropna()
    r_e, p_e = stats.pearsonr(e['vix_pct'], e[col])
    r_l, p_l = stats.pearsonr(l['vix_pct'], l[col])
    sig_e = "*" if p_e < 0.05 else " "
    sig_l = "*" if p_l < 0.05 else " "
    print(f"  {label:>4s}  {r_e:+.4f}{sig_e} {p_e:.4f}  {r_l:+.4f}{sig_l} {p_l:.4f}")
    oos_data.append({'label': label, 'r_early': r_e, 'r_late': r_l})

# ══════════════════════════════════════════════════════════════
# 7. FOMC Removed
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("7. FOMC Removed 후 상관")
print("="*70)
fomc_dates = pd.to_datetime([
    '2020-01-29','2020-03-03','2020-03-15','2020-04-29','2020-06-10',
    '2020-07-29','2020-09-16','2020-11-05','2020-12-16',
    '2021-01-27','2021-03-17','2021-04-28','2021-06-16',
    '2021-07-28','2021-09-22','2021-11-03','2021-12-15',
    '2022-01-26','2022-03-16','2022-05-04','2022-06-15',
    '2022-07-27','2022-09-21','2022-11-02','2022-12-14',
    '2023-02-01','2023-03-22','2023-05-03','2023-06-14',
    '2023-07-26','2023-09-20','2023-11-01','2023-12-13',
    '2024-01-31','2024-03-20','2024-05-01','2024-06-12',
    '2024-07-31','2024-09-18','2024-11-07','2024-12-18',
    '2025-01-29','2025-03-19','2025-05-07',
])
fomc_window = set()
for d in fomc_dates:
    for offset in [-1, 0, 1]:
        fomc_window.add(d + pd.Timedelta(days=offset))

df_no_fomc = df[~df['date'].isin(fomc_window)]
print(f"  All: {len(df)}건 → FOMC Removed: {len(df_no_fomc)}건")

fomc_corrs = []
print(f"\n  {'시간대':>4s}  {'All r':>8s}  {'FOMC제거 r':>10s}  {'차이':>8s}")
for col, label in zip(time_cols, time_labels):
    valid_all = df[['vix_pct',col]].dropna()
    valid_no = df_no_fomc[['vix_pct',col]].dropna()
    r_all, _ = stats.pearsonr(valid_all['vix_pct'], valid_all[col])
    r_no, p_no = stats.pearsonr(valid_no['vix_pct'], valid_no[col])
    sig = "***" if p_no < 0.001 else "**" if p_no < 0.01 else "*" if p_no < 0.05 else ""
    print(f"  {label:>4s}  {r_all:+.4f}  {r_no:+.4f}    {r_no-r_all:+.4f}  {sig}")
    fomc_corrs.append({'label': label, 'r_all': r_all, 'r_no_fomc': r_no})

# ══════════════════════════════════════════════════════════════
# 8. Bootstrap 95% CI
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("8. Bootstrap 95% 신뢰구간")
print("="*70)
np.random.seed(42)
N_BOOT = 10000
boot_results = []
print(f"\n  {'시간대':>4s}  {'r':>8s}  {'CI_lo':>8s}  {'CI_hi':>8s}  {'0 미포함':>8s}")
for col, label in zip(time_cols, time_labels):
    valid = df[['vix_pct',col]].dropna()
    real_r, _ = stats.pearsonr(valid['vix_pct'], valid[col])
    bs = []
    for _ in range(N_BOOT):
        idx = np.random.choice(len(valid), len(valid), replace=True)
        r, _ = stats.pearsonr(valid['vix_pct'].iloc[idx].values, valid[col].iloc[idx].values)
        bs.append(r)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    excl = "✅" if hi < 0 or lo > 0 else "❌"
    print(f"  {label:>4s}  {real_r:+.4f}  {lo:+.4f}  {hi:+.4f}      {excl}")
    boot_results.append({'label': label, 'r': real_r, 'ci_lo': lo, 'ci_hi': hi})

# ══════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 3, figsize=(22, 18))

# 1. 기본 상관계수
ax = axes[0, 0]
colors = ['#e74c3c' if p < 0.05 else '#bdc3c7' for p in corr_df['p']]
ax.bar(range(len(time_labels)), corr_df['r'], color=colors)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, fontsize=8)
ax.set_ylabel('Pearson r')
ax.set_title('VIX to BTC Correlation (red=p<0.05)')

# 2. Permutation test
ax = axes[0, 1]
colors_p = ['#e74c3c' if p < 0.05 else '#bdc3c7' for p in perm_pvals]
ax.bar(range(len(time_labels)), corr_df['r'], color=colors_p)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, fontsize=8)
ax.set_title('Permutation Test (red=shuffle p<0.05)')

# 3. Dose-response
ax = axes[0, 2]
for col, label in zip(target_cols, target_labels):
    means = [df[df['vix_quintile']==q][col].mean()*100 for q in quintiles]
    ax.plot(quintiles, means, 'o-', label=f'BTC {label}', markersize=5)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('BTC Mean Return (%)')
ax.set_title('Dose-Response: VIX Quantile to BTC')
ax.legend(fontsize=7)
ax.tick_params(axis='x', rotation=30)

# 4. Asymmetry
ax = axes[1, 0]
asym_df = pd.DataFrame(asym_data)
ax.plot(range(len(time_labels)), asym_df['r_up'], 'r-o', label='VIX Up Day', markersize=4)
ax.plot(range(len(time_labels)), asym_df['r_down'], 'b-s', label='VIX Down Day', markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, fontsize=8)
ax.set_title('Asymmetry: VIX Up vs Down')
ax.legend(fontsize=8)

# 5. 연도별 히트맵
ax = axes[1, 1]
hm = np.array(yearly_heatmap)
im = ax.imshow(hm, cmap='RdBu_r', aspect='auto', vmin=-0.4, vmax=0.4)
ax.set_xticks(range(len(years)))
ax.set_xticklabels(years, rotation=45, fontsize=8)
ax.set_yticks(range(len(time_labels)))
ax.set_yticklabels(time_labels, fontsize=8)
ax.set_title('Yearly Correlation Heatmap')
plt.colorbar(im, ax=ax, shrink=0.8)

# 6. Out-of-sample
ax = axes[1, 2]
oos_df = pd.DataFrame(oos_data)
ax.plot(range(len(time_labels)), oos_df['r_early'], 'g-o', label='Early (2020-22)', markersize=4)
ax.plot(range(len(time_labels)), oos_df['r_late'], 'm-s', label='Late (2023-26)', markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, fontsize=8)
ax.set_title('Out-of-sample: Early vs Late')
ax.legend(fontsize=8)

# 7. FOMC Removed
ax = axes[2, 0]
fomc_df = pd.DataFrame(fomc_corrs)
x = np.arange(len(time_labels))
ax.bar(x - 0.2, fomc_df['r_all'], 0.4, label='All', color='#e74c3c', alpha=0.7)
ax.bar(x + 0.2, fomc_df['r_no_fomc'], 0.4, label='FOMC Removed', color='#3498db', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(time_labels, rotation=45, fontsize=8)
ax.set_title('FOMC Removed Before/After')
ax.legend(fontsize=8)

# 8. Bootstrap CI
ax = axes[2, 1]
boot_df = pd.DataFrame(boot_results)
ax.errorbar(range(len(time_labels)), boot_df['r'],
            yerr=[boot_df['r']-boot_df['ci_lo'], boot_df['ci_hi']-boot_df['r']],
            fmt='ko-', markersize=5, capsize=3)
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, fontsize=8)
ax.set_title('Bootstrap 95% CI')

# 9. By VIX Level
ax = axes[2, 2]
for regime, cond, color in [("VIX<20", df['vix']<20, '#2ecc71'), 
                             ("VIX 20-30", (df['vix']>=20)&(df['vix']<30), '#f39c12'),
                             ("VIX≥30", df['vix']>=30, '#e74c3c')]:
    sub = df[cond]
    rs = []
    for col in time_cols:
        valid = sub[['vix_pct',col]].dropna()
        r, _ = stats.pearsonr(valid['vix_pct'], valid[col])
        rs.append(r)
    ax.plot(range(len(time_labels)), rs, 'o-', color=color, label=f'{regime} (n={len(sub)})', markersize=4)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(range(len(time_labels)))
ax.set_xticklabels(time_labels, rotation=45, fontsize=8)
ax.set_title('Correlation by VIX Level')
ax.legend(fontsize=8)

plt.suptitle(f'VIX to BTC Impact Analysis (2020.01 ~ 2026.05, n={len(df)})', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('vix_full_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ 차트 저장: vix_full_analysis.png")
