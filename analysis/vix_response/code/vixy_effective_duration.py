"""
VIXY 급변 이벤트(5분 2σ, 30분 중복 제거) 전후 -5분~+120분 구간의 BTC 평균 경로를
추적해, 신호가 방향성을 유지하는 '유효 지속시간'을 시점별 t-test 로 추정한다.

입력: data/vixy_1m/, data/btc_1m_24h/
출력: vixy_effective_duration.png
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 데이터 로드
vixy_is = pd.read_parquet('data/vixy_1m/VIXY_1m_IS_20240115_20251031.parquet')
vixy_oos = pd.read_parquet('data/vixy_1m/VIXY_1m_OOS_20251101_20260430.parquet')
vixy = pd.concat([vixy_is, vixy_oos]).sort_index()
vixy = vixy[~vixy.index.duplicated(keep='first')]
vixy.index = vixy.index.tz_convert('America/New_York')

btc_dir = 'data/btc_1m_24h/'
btc_files = sorted([f for f in os.listdir(btc_dir) if f >= 'btc_1m_2024'])
btc = pd.concat([pd.read_parquet(os.path.join(btc_dir, f))[['close']] for f in btc_files]).sort_index()
btc = btc[~btc.index.duplicated(keep='first')]

# VIXY 5분 변화율
vixy_c = vixy[['close']].rename(columns={'close': 'vixy'})
vixy_c['vixy_5m_pct'] = vixy_c['vixy'].pct_change(5)
vixy_c = vixy_c.dropna()
vixy_c['hour'] = vixy_c.index.hour
mkt = vixy_c[(vixy_c['hour'] >= 10) & (vixy_c['hour'] <= 15)].copy()

sigma = mkt['vixy_5m_pct'].std()

# ══════════════════════════════════════════════════════
# 유효시간 측정:
#
# VIXY가 급변한 시점을 t로 잡고,
# t-5분(VIXY 움직임 시작)부터 t+120분까지
# BTC가 어떻게 움직이는지 경로를 추적한다.
#
# BTC Path가 계속 한 방향으로 가면 = 아직 유효
# BTC Path가 멈추거나 반전하면 = 유효시간 끝
# ══════════════════════════════════════════════════════

thresh = 2 * sigma

# 이벤트 감지 + 중복 제거 (30분)
def deduplicate(times, gap=30):
    if len(times) == 0: return times
    result = [times[0]]
    for t in times[1:]:
        if (t - result[-1]).total_seconds() > gap * 60:
            result.append(t)
    return pd.DatetimeIndex(result)

spike_times = deduplicate(mkt[mkt['vixy_5m_pct'] > thresh].index)
drop_times = deduplicate(mkt[mkt['vixy_5m_pct'] < -thresh].index)

print(f"VIXY Spike: {len(spike_times)}건, 급락: {len(drop_times)}건")

# BTC Path 추출: t-5분(VIXY 움직임 시작) 기준, -5분 ~ +120분
offsets_full = list(range(-5, 121))  # -5분 ~ +120분

def extract_btc_paths(event_times):
    paths = []
    for t in event_times:
        t_start = t - pd.Timedelta(minutes=5)  # VIXY 움직임 시작점
        if t_start not in btc.index:
            continue
        base = btc.loc[t_start, 'close']
        path = []
        valid = True
        for offset in offsets_full:
            target = t_start + pd.Timedelta(minutes=offset)
            if target in btc.index:
                path.append((btc.loc[target, 'close'] - base) / base * 100)
            else:
                path.append(np.nan)
        paths.append(path)
    return np.array(paths)

spike_paths = extract_btc_paths(spike_times)
drop_paths = extract_btc_paths(drop_times)

print(f"경로 추출 — 급등: {len(spike_paths)}건, 급락: {len(drop_paths)}건")

# Mean 경로
spike_mean = np.nanmean(spike_paths, axis=0)
drop_mean = np.nanmean(drop_paths, axis=0)

# 각 시점별 t-test (0과 다른지)
print("\n" + "="*70)
print("After VIXY Spike BTC Mean 경로 (t-5=VIXY 움직임 시작, t=감지)")
print("="*70)
key_offsets = [-5, 0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120]
for off in key_offsets:
    idx = offsets_full.index(off)
    vals = spike_paths[:, idx]
    vals = vals[~np.isnan(vals)]
    m = np.mean(vals)
    t_stat, p_val = stats.ttest_1samp(vals, 0)
    sig = "***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else ""
    label = f"t{off:+d}m" if off != 0 else "t=0(감지)"
    print(f"  {label:>12s}  mean={m:+.4f}%  t={t_stat:+.2f}  p={p_val:.4f}{sig}")

print("\n" + "="*70)
print("After VIXY Drop BTC Mean 경로")
print("="*70)
for off in key_offsets:
    idx = offsets_full.index(off)
    vals = drop_paths[:, idx]
    vals = vals[~np.isnan(vals)]
    m = np.mean(vals)
    t_stat, p_val = stats.ttest_1samp(vals, 0)
    sig = "***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else ""
    label = f"t{off:+d}m" if off != 0 else "t=0(감지)"
    print(f"  {label:>12s}  mean={m:+.4f}%  t={t_stat:+.2f}  p={p_val:.4f}{sig}")

# ══════════════════════════════════════════════════════
# 유효시간 판정:
# t=0(감지 시점)부터 BTC Mean 경로의 기울기가 0에 가까워지는 시점
# = 이동 Mean 기울기(5분 윈도우)가 처음으로 0을 교차하는 시점
# ══════════════════════════════════════════════════════

# Detection Point (t=0) 이후만
t0_idx = offsets_full.index(0)

for event_name, mean_path in [("VIXY Spike(BTC 하락)", spike_mean), ("VIXY Drop(BTC 상승)", drop_mean)]:
    after = mean_path[t0_idx:]  # t=0부터
    # 5분 이동 기울기
    slopes = np.diff(after)
    
    # 기울기 부호 변화 = 반전 시점
    if event_name.startswith("VIXY Spike"):
        # BTC가 하락하다가 멈추는 시점 = 기울기가 음→양으로
        sign_changes = np.where(np.diff(np.sign(slopes)) > 0)[0]
    else:
        # BTC가 상승하다가 멈추는 시점 = 기울기가 양→음으로
        sign_changes = np.where(np.diff(np.sign(slopes)) < 0)[0]
    
    if len(sign_changes) > 0:
        # 첫 번째 반전이 너무 이르면(노이즈) 다음 것 사용
        for sc in sign_changes:
            if sc >= 3:  # 최소 3분 이후
                print(f"\n  [{event_name}] 반전 시점: 감지 후 약 {sc}분")
                break
    
    # 피크 시점
    if event_name.startswith("VIXY Spike"):
        peak_idx = np.nanargmin(after)  # BTC 최저점
    else:
        peak_idx = np.nanargmax(after)  # BTC 최고점
    print(f"  [{event_name}] 피크: 감지 후 {peak_idx}분 (BTC {after[peak_idx]:+.4f}%)")

# ══════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. After VIXY Spike BTC Mean 경로 (±신뢰구간)
ax = axes[0, 0]
spike_se = np.nanstd(spike_paths, axis=0) / np.sqrt(np.sum(~np.isnan(spike_paths), axis=0))
ax.plot(offsets_full, spike_mean, 'r-', linewidth=2, label='Mean')
ax.fill_between(offsets_full, spike_mean - 1.96*spike_se, spike_mean + 1.96*spike_se, alpha=0.2, color='red')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='blue', linewidth=1, linestyle='--', label='Detection Point (t=0)')
ax.axvline(-5, color='gray', linewidth=1, linestyle=':', label='VIXY Movement Start (t-5)')
ax.set_xlabel('Time (min, from VIXY Movement Start)'); ax.set_ylabel('BTC Return (%)')
ax.set_title(f'After VIXY Spike BTC Path (n={len(spike_paths)})'); ax.legend(fontsize=8)

# 2. After VIXY Drop BTC Mean 경로
ax = axes[0, 1]
drop_se = np.nanstd(drop_paths, axis=0) / np.sqrt(np.sum(~np.isnan(drop_paths), axis=0))
ax.plot(offsets_full, drop_mean, 'b-', linewidth=2, label='Mean')
ax.fill_between(offsets_full, drop_mean - 1.96*drop_se, drop_mean + 1.96*drop_se, alpha=0.2, color='blue')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='blue', linewidth=1, linestyle='--', label='Detection Point (t=0)')
ax.axvline(-5, color='gray', linewidth=1, linestyle=':', label='VIXY Movement Start (t-5)')
ax.set_xlabel('Time (min)'); ax.set_ylabel('BTC Return (%)')
ax.set_title(f'After VIXY Drop BTC Path (n={len(drop_paths)})'); ax.legend(fontsize=8)

# 3. 양쪽 합쳐서 비교
ax = axes[1, 0]
ax.plot(offsets_full, spike_mean, 'r-', linewidth=2, label='VIXY Spike to BTC')
ax.plot(offsets_full, drop_mean, 'b-', linewidth=2, label='VIXY Drop to BTC')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='green', linewidth=1, linestyle='--', label='Detection (t=0)')
ax.axvline(-5, color='gray', linewidth=1, linestyle=':')
ax.set_xlabel('Time (min)'); ax.set_ylabel('BTC Return (%)')
ax.set_title('BTC Path Comparison: VIXY Spike vs Drop'); ax.legend(fontsize=8)

# 4. 각 분별 p-value (0과 다른지)
ax = axes[1, 1]
check_range = range(0, 121, 1)
spike_ps = []
drop_ps = []
for off in check_range:
    idx = offsets_full.index(off)
    sv = spike_paths[:, idx]; sv = sv[~np.isnan(sv)]
    dv = drop_paths[:, idx]; dv = dv[~np.isnan(dv)]
    _, sp = stats.ttest_1samp(sv, 0) if len(sv)>5 else (0, 1)
    _, dp = stats.ttest_1samp(dv, 0) if len(dv)>5 else (0, 1)
    spike_ps.append(sp)
    drop_ps.append(dp)

ax.semilogy(list(check_range), spike_ps, 'r-', label='VIXY Spike', linewidth=0.8)
ax.semilogy(list(check_range), drop_ps, 'b-', label='VIXY Drop', linewidth=0.8)
ax.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='p=0.05')
ax.set_xlabel('Time After Detection (min)'); ax.set_ylabel('p-value (log)')
ax.set_title('BTC Response Significance Trend'); ax.legend(fontsize=8)

plt.suptitle('Effective Duration of BTC Concurrent Response to VIXY Shock (2024.01~2026.04)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('vixy_effective_duration.png', dpi=150, bbox_inches='tight')
print("\n저장: vixy_effective_duration.png")
