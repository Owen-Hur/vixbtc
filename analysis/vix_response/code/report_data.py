"""
vix_btc_response.parquet 에서 보고서 인용용 수치를 콘솔로 출력한다.
전기/후기 상관 비교, VIX 레벨별 상관, permutation test, event study,
상승/하락 비대칭, dose-response 를 차례로 계산한다(파일 산출물 없음).

입력: data/vix_btc_response.parquet, data/btc_1m_24h/
"""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('data/vix_btc_response.parquet')
df['date'] = pd.to_datetime(df['date'])

time_cols = ['btc_1m','btc_5m','btc_10m','btc_15m','btc_20m','btc_30m',
             'btc_45m','btc_1h','btc_90m','btc_2h','btc_3h','btc_4h',
             'btc_6h','btc_8h','btc_12h','btc_24h']
time_labels = ['1m','5m','10m','15m','20m','30m','45m','1h','90m','2h','3h','4h','6h','8h','12h','24h']

df_pre = df[df['date'] < '2024-01-01']
df_post = df[df['date'] >= '2024-01-01']

print("=== 데이터 기본 정보 ===")
print(f"전체: {len(df)}건, {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"전기: {len(df_pre)}건, {df_pre['date'].min().date()} ~ {df_pre['date'].max().date()}")
print(f"후기: {len(df_post)}건, {df_post['date'].min().date()} ~ {df_post['date'].max().date()}")
print(f"\n전기 VIX: mean={df_pre['vix'].mean():.2f}, std={df_pre['vix'].std():.2f}, min={df_pre['vix'].min():.2f}, max={df_pre['vix'].max():.2f}")
print(f"후기 VIX: mean={df_post['vix'].mean():.2f}, std={df_post['vix'].std():.2f}, min={df_post['vix'].min():.2f}, max={df_post['vix'].max():.2f}")

# BTC 가격 범위 (close 기준)
import os
btc_files = sorted(os.listdir('data/btc_1m_24h/'))
btc_pre = []
btc_post = []
for f in btc_files:
    chunk = pd.read_parquet(f'data/btc_1m_24h/{f}')[['close']]
    if '2020' in f or '2021' in f or '2022' in f or '2023' in f:
        btc_pre.append(chunk)
    elif '2024' in f or '2025' in f or '2026' in f:
        btc_post.append(chunk)
btc_pre_df = pd.concat(btc_pre)
btc_post_df = pd.concat(btc_post)
print(f"\n전기 BTC: mean=${btc_pre_df['close'].mean():,.0f}, min=${btc_pre_df['close'].min():,.0f}, max=${btc_pre_df['close'].max():,.0f}")
print(f"후기 BTC: mean=${btc_post_df['close'].mean():,.0f}, min=${btc_post_df['close'].min():,.0f}, max=${btc_post_df['close'].max():,.0f}")

# 전기 vs 후기 상관계수 비교
print("\n=== 전기 vs 후기 상관계수 ===")
print(f"{'시간대':>4s}  {'전기 r':>8s} {'p':>8s}  {'후기 r':>8s} {'p':>8s}")
for col, label in zip(time_cols, time_labels):
    r1, p1 = stats.pearsonr(df_pre['vix_pct'], df_pre[col])
    r2, p2 = stats.pearsonr(df_post['vix_pct'], df_post[col])
    sig1 = "***" if p1<0.001 else "**" if p1<0.01 else "*" if p1<0.05 else ""
    sig2 = "***" if p2<0.001 else "**" if p2<0.01 else "*" if p2<0.05 else ""
    print(f"  {label:>4s}  {r1:+.4f}{sig1:3s} {p1:.4f}  {r2:+.4f}{sig2:3s} {p2:.4f}")

# 후기 VIX 레벨별 상세
print("\n=== 후기 VIX 레벨별 ===")
for regime, cond in [("VIX<18", df_post['vix']<18),
                      ("VIX 18-25", (df_post['vix']>=18)&(df_post['vix']<25)),
                      ("VIX>=25", df_post['vix']>=25)]:
    sub = df_post[cond]
    print(f"\n[{regime}] n={len(sub)}")
    for col, label in zip(time_cols, time_labels):
        r, p = stats.pearsonr(sub['vix_pct'], sub[col])
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        print(f"  {label:>4s}  r={r:+.4f} p={p:.4f}{sig}")

# 후기 Permutation
print("\n=== 후기 Permutation (VIX 18-25 구간) ===")
np.random.seed(42)
sub = df_post[(df_post['vix']>=18)&(df_post['vix']<25)]
for col, label in zip(time_cols, time_labels):
    real_r, _ = stats.pearsonr(sub['vix_pct'], sub[col])
    perm_rs = np.array([stats.pearsonr(np.random.permutation(sub['vix_pct'].values), sub[col].values)[0] for _ in range(10000)])
    perm_p = np.mean(np.abs(perm_rs) >= np.abs(real_r))
    sig = "***" if perm_p<0.001 else "**" if perm_p<0.01 else "*" if perm_p<0.05 else "NS"
    print(f"  {label:>4s}  r={real_r:+.4f}  perm_p={perm_p:.4f} [{sig}]")

# 후기 Event Study
print("\n=== 후기 Event Study ===")
sigma = df_post['vix_pct'].std()
spike = df_post[df_post['vix_pct'] > sigma]
drop = df_post[df_post['vix_pct'] < -sigma]
print(f"1σ={sigma*100:.2f}%, 급등:{len(spike)}건, 급락:{len(drop)}건")
print("\n[급등]")
for col, label in zip(time_cols, time_labels):
    m = spike[col].mean()
    t, p = stats.ttest_1samp(spike[col], 0)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {label:>4s}  mean={m*100:+.4f}%  t={t:+.2f} p={p:.4f}{sig}")
print("\n[급락]")
for col, label in zip(time_cols, time_labels):
    m = drop[col].mean()
    t, p = stats.ttest_1samp(drop[col], 0)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {label:>4s}  mean={m*100:+.4f}%  t={t:+.2f} p={p:.4f}{sig}")

# 후기 비대칭
print("\n=== 후기 비대칭 ===")
up = df_post[df_post['vix_pct']>0]
dn = df_post[df_post['vix_pct']<0]
print(f"상승:{len(up)}, 하락:{len(dn)}")
for col, label in zip(time_cols, time_labels):
    ru, pu = stats.pearsonr(up['vix_pct'], up[col])
    rd, pd_ = stats.pearsonr(dn['vix_pct'], dn[col])
    su = "*" if pu<0.05 else ""
    sd = "*" if pd_<0.05 else ""
    print(f"  {label:>4s}  up={ru:+.4f}{su} down={rd:+.4f}{sd}")

# 후기 Dose-Response
print("\n=== 후기 Dose-Response ===")
df_post_c = df_post.copy()
df_post_c['vix_q'] = pd.qcut(df_post_c['vix_pct'], 5, labels=['Q1','Q2','Q3','Q4','Q5'])
for col, label in zip(['btc_30m','btc_1h','btc_2h','btc_4h','btc_8h'],['30m','1h','2h','4h','8h']):
    print(f"\n[BTC {label}]")
    for q in ['Q1','Q2','Q3','Q4','Q5']:
        s = df_post_c[df_post_c['vix_q']==q][col]
        print(f"  {q}: mean={s.mean()*100:+.4f}% n={len(s)}")
    groups = [df_post_c[df_post_c['vix_q']==q][col].values for q in ['Q1','Q2','Q3','Q4','Q5']]
    f, p = stats.f_oneway(*groups)
    print(f"  ANOVA F={f:.2f} p={p:.4f}")
