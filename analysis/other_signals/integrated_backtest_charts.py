"""
integrated_backtest_charts.py — IS 백테스트 결과 시각화 (단일 HTML)

실행: python3 integrated_backtest_charts.py
결과: backtest_results/is_backtest_report.html
"""

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "HMM"))
from data_loader import load_all_daily
from feature_engineer import build_features
from hmm_trainer import fit_hmm, build_state_map, predict_states
from leverage.soft_leverage import soft_leverage_series
from config import IS_START, IS_END, TAKER_FEE, MAX_LEVERAGE

OUTPUT_DIR = Path(__file__).parent / "backtest_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 준비
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("데이터 로딩 중...")
print("=" * 60)

df = load_all_daily(IS_START, IS_END)
df['slope'] = df['vix_9d'] - df['vix_30d']
df['slope_change'] = df['slope'].diff()

feat, scaler, pca = build_features(df, fit=True)

print("[HMM] 모델 학습 중...")
hmm_model = fit_hmm(feat.values)
state_map = build_state_map(hmm_model, feat.values, df['vix_30d'].reindex(feat.index))
state_labels, posteriors = predict_states(hmm_model, feat.values, state_map)

bt = pd.DataFrame(index=feat.index)
bt['btc_ret'] = df['btc_ret'].reindex(feat.index)
bt['btc_price'] = df['close'].reindex(feat.index)
bt['slope'] = df['slope'].reindex(feat.index)
bt['slope_change'] = df['slope_change'].reindex(feat.index)
bt['funding_rate'] = df['funding_rate'].reindex(feat.index).fillna(0)
bt['hmm_state'] = state_labels
bt['p_normal'] = posteriors[:, 0]
bt['p_alert'] = posteriors[:, 1]
bt['p_fear'] = posteriors[:, 2]
bt['vix_30d'] = df['vix_30d'].reindex(feat.index)
bt = bt.dropna(subset=['btc_ret', 'slope_change'])

abs_sc = bt['slope_change'].abs()
Q25 = abs_sc.quantile(0.25)
Q50 = abs_sc.quantile(0.50)
Q75 = abs_sc.quantile(0.75)


def slope_change_leverage(sc):
    a = abs(sc)
    if a >= Q75:     mag = 2.0
    elif a >= Q50:   mag = 1.5
    elif a >= Q25:   mag = 1.0
    else:            mag = 0.5
    return (-1.0 if sc > 0 else 1.0) * mag


bt['lev_slope'] = bt['slope_change'].apply(slope_change_leverage)

CHAMPION_THETA = {'lev_normal': 1.5, 'lev_alert': 0.5, 'lev_fear': -0.5}
bt['lev_hmm'] = soft_leverage_series(
    posteriors[feat.index.isin(bt.index)], **CHAMPION_THETA, max_lev=MAX_LEVERAGE
)


def integrated_leverage_cap(row):
    cap = {'normal': 2.0, 'alert': 1.0, 'fear': 0.5}.get(row['hmm_state'], 2.0)
    return np.clip(row['lev_slope'], -cap, cap)


bt['lev_integrated_cap'] = bt.apply(integrated_leverage_cap, axis=1)


def compute_returns(bt_df, lev_col):
    lev = bt_df[lev_col]
    gross = lev * bt_df['btc_ret']
    cost = lev.diff().abs().fillna(lev.abs()) * TAKER_FEE
    funding = lev * bt_df['funding_rate']
    return gross - cost - funding


strategies = {
    'Slope 단독': 'lev_slope',
    'HMM 단독': 'lev_hmm',
    'Slope+HMM Cap': 'lev_integrated_cap',
}

net_returns, cum_returns = {}, {}
for name, col in strategies.items():
    net = compute_returns(bt, col)
    net_returns[name] = net
    cum_returns[name] = (1 + net).cumprod()

net_returns['Buy & Hold'] = bt['btc_ret']
cum_returns['Buy & Hold'] = (1 + bt['btc_ret']).cumprod()

colors = {
    'Slope 단독': '#FF6B35',
    'HMM 단독': '#004E89',
    'Slope+HMM Cap': '#2EC4B6',
    'Buy & Hold': '#888888',
}

print("데이터 준비 완료. 차트 생성 중...\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 1: 성과 대시보드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

strat_names = ['Buy & Hold', 'Slope 단독', 'HMM 단독', 'Slope+HMM Cap']
bar_colors = ['#888888', '#FF6B35', '#004E89', '#2EC4B6']
metrics = {}
for name in strat_names:
    net = net_returns[name]
    cum = cum_returns[name]
    total = cum.iloc[-1] - 1
    ann_ret = (1 + total) ** (252 / len(bt)) - 1
    ann_vol = net.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    peak = cum.cummax()
    mdd = ((cum - peak) / peak).min()
    metrics[name] = dict(total=total*100, ann=ann_ret*100, vol=ann_vol*100,
                         sharpe=sharpe, mdd=mdd*100)

fig1 = make_subplots(
    rows=2, cols=2,
    subplot_titles=("총수익률 (%)", "Sharpe Ratio", "Max Drawdown (%)", "연간 변동성 (%)"),
    vertical_spacing=0.22, horizontal_spacing=0.12,
)
metric_keys = [('total', '{:.0f}%'), ('sharpe', '{:.2f}'), ('mdd', '{:.0f}%'), ('vol', '{:.0f}%')]
for i, (key, fmt) in enumerate(metric_keys):
    r, c = divmod(i, 2)
    fig1.add_trace(go.Bar(
        x=strat_names,
        y=[metrics[s][key] for s in strat_names],
        marker_color=bar_colors,
        text=[fmt.format(metrics[s][key]) for s in strat_names],
        textposition='outside', showlegend=False,
    ), row=r+1, col=c+1)
fig1.update_layout(height=580, template='plotly_white', showlegend=False,
                   margin=dict(t=50, b=30, l=60, r=30))
print("  [1/7] 대시보드")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 2: 누적 수익률 + BTC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                     row_heights=[0.65, 0.35],
                     subplot_titles=("누적 수익률 비교", "BTC 선물 가격 (USD)"))

# 명시적 순서로 추가 — 각 trace에 고유 legendgroup
draw_order = ['Buy & Hold', 'HMM 단독', 'Slope+HMM Cap', 'Slope 단독']
for name in draw_order:
    cum = cum_returns[name]
    fig2.add_trace(go.Scatter(
        x=cum.index, y=cum.values,
        name=name,
        legendgroup=f'cum_{name}',
        line=dict(color=colors[name],
                  width=3.0 if name == 'Slope 단독' else 2.0,
                  dash='dot' if name == 'Buy & Hold' else 'solid'),
        visible=True,
    ), row=1, col=1)

fig2.add_trace(go.Scatter(
    x=bt.index, y=bt['btc_price'], name='BTC Price',
    legendgroup='btc_price',
    line=dict(color='#FFD166', width=1.8),
    showlegend=True,
), row=2, col=1)

fig2.update_layout(
    height=650, template='plotly_white', hovermode='x unified',
    legend=dict(x=0.01, y=0.98, bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#ddd', borderwidth=1, font_size=12),
    margin=dict(t=50, b=30, l=60, r=30),
)
fig2.update_yaxes(title_text="누적 배수", row=1, col=1)
fig2.update_yaxes(title_text="USD", row=2, col=1)
print("  [2/7] 누적 수익률")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 3: Drawdown (fill 제거 → 모든 선 보이게)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig3 = go.Figure()
# Buy & Hold를 먼저 (뒤에), 나머지를 위에
for name in ['Buy & Hold', 'HMM 단독', 'Slope+HMM Cap', 'Slope 단독']:
    cum = cum_returns[name]
    peak = cum.cummax()
    dd = (cum - peak) / peak * 100
    fig3.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name=name,
        legendgroup=f'dd_{name}',
        line=dict(color=colors[name],
                  width=2.5 if name == 'Slope 단독' else 1.8,
                  dash='dot' if name == 'Buy & Hold' else 'solid'),
        visible=True,
    ))

fig3.update_layout(
    height=450, template='plotly_white',
    yaxis_title="Drawdown (%)",
    hovermode='x unified',
    legend=dict(x=0.01, y=-0.15, orientation='h', font_size=12),
    margin=dict(t=30, b=80, l=60, r=30),
)
print("  [3/7] Drawdown")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 4: 월별 히트맵 (라벨 수정)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

monthly_data = {}
for name in ['Slope 단독', 'HMM 단독', 'Slope+HMM Cap', 'Buy & Hold']:
    monthly_data[name] = net_returns[name].resample('ME').apply(
        lambda x: (1+x).prod()-1) * 100

all_months = sorted(set().union(*[set(m.index) for m in monthly_data.values()]))
month_labels = [m.strftime('%y.%m') for m in all_months]

heatmap_z = []
strat_order = ['Buy & Hold', 'Slope+HMM Cap', 'HMM 단독', 'Slope 단독']
for name in strat_order:
    heatmap_z.append([round(monthly_data[name].get(m, 0), 1) for m in all_months])

fig4 = go.Figure(data=go.Heatmap(
    z=heatmap_z,
    x=month_labels,
    y=strat_order,
    colorscale=[
        [0, '#d73027'], [0.25, '#fc8d59'], [0.45, '#fee08b'],
        [0.5, '#ffffbf'],
        [0.55, '#d9ef8b'], [0.75, '#91cf60'], [1, '#1a9850'],
    ],
    zmid=0,
    text=[[f"{v:+.1f}" for v in row] for row in heatmap_z],
    texttemplate="%{text}",
    textfont={"size": 11, "color": "#333"},
    hovertemplate='%{y}<br>%{x}: %{z:.1f}%<extra></extra>',
    colorbar=dict(title="수익률(%)", len=0.8),
    xgap=2, ygap=2,
))

fig4.update_layout(
    height=350,
    template='plotly_white',
    xaxis=dict(
        tickmode='array',
        tickvals=month_labels,
        ticktext=month_labels,
        tickangle=-45,
        tickfont=dict(size=11),
        side='bottom',
    ),
    yaxis=dict(
        tickfont=dict(size=12),
        autorange='reversed',
    ),
    margin=dict(t=30, b=80, l=130, r=80),
)
print("  [4/7] 월별 히트맵")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 5: 타임라인 (vrect 대신 scatter marker로 HMM 상태 표시)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

state_color_map = {'normal': '#2EC4B6', 'alert': '#FFD166', 'fear': '#EF476F'}
state_num = bt['hmm_state'].map({'normal': 0, 'alert': 1, 'fear': 2})

fig5 = make_subplots(
    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    row_heights=[0.30, 0.22, 0.25, 0.23],
    subplot_titles=("BTC 가격 + HMM 상태", "HMM Posterior 확률",
                    "레버리지 비교", "VIX Slope & Slope Change"),
)

# Row 1: BTC 가격 (선) + HMM 상태 (컬러 마커로 표시)
fig5.add_trace(go.Scatter(
    x=bt.index, y=bt['btc_price'], name='BTC Price',
    mode='lines', line=dict(color='#333', width=1.5),
    legendgroup='tl_btc', showlegend=True,
), row=1, col=1)

# HMM 상태를 bar 대신 scatter marker로 — 렌더링 가벼움
for state, color, label in [('normal','#2EC4B6','Normal'),
                             ('alert','#FFD166','Alert'),
                             ('fear','#EF476F','Fear')]:
    mask = bt['hmm_state'] == state
    fig5.add_trace(go.Scatter(
        x=bt.index[mask], y=bt['btc_price'][mask],
        mode='markers', name=f'HMM: {label}',
        marker=dict(color=color, size=5, opacity=0.6),
        legendgroup=f'tl_hmm_{state}', showlegend=True,
    ), row=1, col=1)

# Row 2: Posterior — stackgroup 방식으로 정확한 영역 차트
for col_name, label, color, fcolor in [
    ('p_normal', 'P(Normal)', '#2EC4B6', 'rgba(46,196,182,0.6)'),
    ('p_alert',  'P(Alert)',  '#FFD166', 'rgba(255,209,102,0.6)'),
    ('p_fear',   'P(Fear)',   '#EF476F', 'rgba(239,71,111,0.6)'),
]:
    fig5.add_trace(go.Scatter(
        x=bt.index, y=bt[col_name], name=label,
        mode='lines', line=dict(color=color, width=0.3),
        stackgroup='posterior',           # 올바른 stacking
        fillcolor=fcolor,
        legendgroup=f'tl_post_{col_name}', showlegend=True,
        hovertemplate=f'{label}: %{{y:.3f}}<extra></extra>',
    ), row=2, col=1)

# Row 3: 레버리지 — 원본(투명) + 5일 이동평균(진하게)
# 원본은 연하게 배경으로, rolling mean으로 추세를 보여줌
lev_configs = [
    ('Slope',     'lev_slope',           '#FF6B35'),
    ('HMM',       'lev_hmm',             '#004E89'),
    ('Slope+HMM', 'lev_integrated_cap',  '#2EC4B6'),
]
for nm, col_name, clr in lev_configs:
    # 원본 — 연한 step 라인
    fig5.add_trace(go.Scatter(
        x=bt.index, y=bt[col_name],
        name=f'{nm} (raw)',
        mode='lines',
        line=dict(color=clr, width=0.7),
        opacity=0.25,
        legendgroup=f'tl_lev_{col_name}',
        showlegend=False,
        hoverinfo='skip',
    ), row=3, col=1)
    # 5일 이동평균 — 진한 선
    rolling = bt[col_name].rolling(5, min_periods=1).mean()
    fig5.add_trace(go.Scatter(
        x=bt.index, y=rolling,
        name=f'Lev: {nm} (5d avg)',
        mode='lines',
        line=dict(color=clr, width=2.2),
        legendgroup=f'tl_lev_{col_name}',
        showlegend=True,
        hovertemplate=f'{nm}: %{{y:.2f}}<extra></extra>',
    ), row=3, col=1)
fig5.add_hline(y=0, line_dash='dot', line_color='gray', line_width=0.8, row=3, col=1)

# Row 4: Slope (선) + Slope Change (막대)
fig5.add_trace(go.Scatter(
    x=bt.index, y=bt['slope'], name='VIX Slope (Level)',
    mode='lines', line=dict(color='#7B2D8E', width=1.5),
    legendgroup='tl_slope', showlegend=True,
), row=4, col=1)
fig5.add_trace(go.Bar(
    x=bt.index, y=bt['slope_change'], name='Slope Change',
    marker_color=[('#EF476F' if v > 0 else '#2EC4B6') for v in bt['slope_change']],
    opacity=0.5,
    legendgroup='tl_sc', showlegend=True,
), row=4, col=1)
fig5.add_hline(y=0, line_dash='dot', line_color='gray', line_width=0.8, row=4, col=1)

fig5.update_layout(
    height=1050, template='plotly_white', hovermode='x unified',
    legend=dict(x=1.01, y=1, font_size=10, bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#ddd', borderwidth=1),
    margin=dict(t=50, b=30, l=60, r=150),
)
fig5.update_yaxes(title_text="USD", row=1, col=1)
fig5.update_yaxes(title_text="확률", range=[0, 1], row=2, col=1)
fig5.update_yaxes(title_text="레버리지", row=3, col=1)
fig5.update_yaxes(title_text="Slope", row=4, col=1)
print("  [5/7] 타임라인")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 6: Hit Rate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig6 = make_subplots(rows=1, cols=2,
    subplot_titles=("|slope_change| 크기별 Hit Rate", "HMM 상태별 Hit Rate"),
    horizontal_spacing=0.18)

direction = np.sign(bt['lev_slope'])
correct = (direction * bt['btc_ret']) > 0

q_labels, q_hits, q_ns, q_colors = [], [], [], []
for label, lo, hi, clr in [
    ('Q1 (하위25%)', 0, Q25, '#fc8d59'),
    ('Q2 (25~50%)', Q25, Q50, '#fee08b'),
    ('Q3 (50~75%)', Q50, Q75, '#91cf60'),
    ('Q4 (상위25%)', Q75, 999, '#1a9850'),
]:
    mask = (abs_sc >= lo) & (abs_sc < hi)
    if mask.sum() > 0:
        q_labels.append(label)
        q_hits.append(correct[mask].mean() * 100)
        q_ns.append(mask.sum())
        q_colors.append(clr)

fig6.add_trace(go.Bar(
    x=q_labels, y=q_hits,
    text=[f"<b>{h:.1f}%</b><br>n={n}" for h, n in zip(q_hits, q_ns)],
    textposition='outside', marker_color=q_colors, width=0.55,
    showlegend=False,
), row=1, col=1)
fig6.add_hline(y=50, line_dash='dot', line_color='#e53935', line_width=1.5,
               annotation_text='50% (랜덤)', annotation_font_color='#e53935',
               row=1, col=1)

s_labels, s_hits, s_ns, s_colors = [], [], [], []
for state, color in [('normal','#2EC4B6'),('alert','#FFD166'),('fear','#EF476F')]:
    mask = bt['hmm_state'] == state
    if mask.sum() > 0:
        s_labels.append(state.upper())
        s_hits.append(correct[mask].mean() * 100)
        s_ns.append(mask.sum())
        s_colors.append(color)

fig6.add_trace(go.Bar(
    x=s_labels, y=s_hits,
    text=[f"<b>{h:.1f}%</b><br>n={n}" for h, n in zip(s_hits, s_ns)],
    textposition='outside', marker_color=s_colors, width=0.45,
    showlegend=False,
), row=1, col=2)
fig6.add_hline(y=50, line_dash='dot', line_color='#e53935', line_width=1.5, row=1, col=2)

fig6.update_layout(height=420, template='plotly_white', showlegend=False,
                   margin=dict(t=50, b=30, l=60, r=30))
fig6.update_yaxes(title_text="Hit Rate (%)", range=[38, 68], row=1, col=1)
fig6.update_yaxes(range=[38, 68], row=1, col=2)
print("  [6/7] Hit Rate")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 7: 수익률 분포 Violin
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig7 = go.Figure()
for name, color in [('Buy & Hold','#888888'),('Slope 단독','#FF6B35'),
                     ('HMM 단독','#004E89'),('Slope+HMM Cap','#2EC4B6')]:
    fig7.add_trace(go.Violin(
        y=net_returns[name].values * 100, name=name,
        box_visible=True, meanline_visible=True,
        line_color=color, fillcolor=color, opacity=0.5,
        points='outliers',
    ))
fig7.update_layout(height=450, template='plotly_white',
                   yaxis_title="일별 수익률 (%)",
                   margin=dict(t=30, b=30, l=60, r=30))
print("  [7/7] 수익률 분포")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 통합 HTML 조립
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

charts = [
    ('dashboard',   '1. 성과 요약 대시보드',                     fig1),
    ('cumulative',  '2. 누적 수익률 + BTC 가격',                fig2),
    ('drawdown',    '3. Drawdown 비교',                         fig3),
    ('heatmap',     '4. 월별 수익률 히트맵',                     fig4),
    ('timeline',    '5. 전략 타임라인 (HMM x 레버리지 x BTC)',   fig5),
    ('hitrate',     '6. slope_change Hit Rate 분석',            fig6),
    ('violin',      '7. 일별 수익률 분포',                       fig7),
]

# 성과 테이블
table_rows = ""
for s in strat_names:
    m = metrics[s]
    hl = ' class="highlight"' if s == 'Slope 단독' else ''
    table_rows += f'<tr{hl}><td>{s}</td><td>{m["total"]:.0f}%</td><td>{m["ann"]:.0f}%</td><td>{m["sharpe"]:.2f}</td><td>{m["mdd"]:.0f}%</td><td>{m["vol"]:.0f}%</td></tr>\n'

# 차트 div 생성 (첫 번째만 plotly.js CDN 포함)
chart_sections = ""
for i, (cid, title, fig) in enumerate(charts):
    div = pio.to_html(fig, full_html=False,
                      include_plotlyjs=('cdn' if i == 0 else False))
    chart_sections += f'''
    <div class="chart-card" id="{cid}">
      <h2>{title}</h2>
      <div class="chart-wrap">{div}</div>
    </div>
'''

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IS Backtest Report</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f0f2f5; color: #333;
}}
.header {{
  background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
  color: #fff; padding: 45px 30px 35px; text-align: center;
}}
.header h1 {{ font-size: 30px; font-weight: 700; letter-spacing: -0.5px; }}
.header .sub {{ font-size: 14px; opacity: 0.75; margin-top: 8px; }}

/* Sticky nav */
.nav {{
  background: #fff; border-bottom: 1px solid #ddd;
  position: sticky; top: 0; z-index: 999;
  display: flex; overflow-x: auto; padding: 0 16px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}}
.nav a {{
  flex-shrink: 0; padding: 13px 16px; font-size: 13px; font-weight: 500;
  color: #666; text-decoration: none; border-bottom: 3px solid transparent;
  transition: all 0.15s;
}}
.nav a:hover, .nav a.active {{ color: #FF6B35; border-bottom-color: #FF6B35; }}

.container {{ max-width: 1280px; margin: 0 auto; padding: 24px 16px; }}

/* Summary table card */
.summary-card {{
  background: #fff; border-radius: 14px; padding: 28px 32px;
  margin-bottom: 24px; box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}}
.summary-card h2 {{ font-size: 17px; margin-bottom: 16px; color: #1a1a2e; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th {{
  background: #f7f7f7; padding: 11px 16px; text-align: center;
  font-weight: 600; border-bottom: 2px solid #ddd;
}}
td {{ padding: 11px 16px; text-align: center; border-bottom: 1px solid #eee; }}
tr.highlight {{ background: #fff7ed; font-weight: 600; }}

/* Chart cards */
.chart-card {{
  background: #fff; border-radius: 14px; padding: 22px 18px 18px;
  margin-bottom: 24px; box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}}
.chart-card h2 {{
  font-size: 15px; color: #1a1a2e; margin-bottom: 12px;
  padding-left: 12px; border-left: 4px solid #FF6B35;
}}
.chart-wrap {{ width: 100%; overflow-x: auto; }}

/* Findings */
.findings {{
  background: #fff; border-radius: 14px; padding: 28px 32px;
  margin-bottom: 24px; box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}}
.findings h2 {{ font-size: 17px; margin-bottom: 16px; color: #1a1a2e; }}
.pill {{
  padding: 12px 18px; margin-bottom: 10px; border-radius: 10px;
  font-size: 14px; line-height: 1.65;
}}
.pill.g {{ background: #e8f5e9; border-left: 4px solid #43a047; }}
.pill.w {{ background: #fff8e1; border-left: 4px solid #f9a825; }}
.pill.r {{ background: #fce4ec; border-left: 4px solid #e53935; }}

/* Strategy section */
.strategy-card {{
  background: #fff; border-radius: 14px; padding: 28px 32px;
  margin-bottom: 24px; box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}}
.strategy-card h2 {{
  font-size: 17px; margin-bottom: 20px; color: #1a1a2e;
}}
.strategy-card h3 {{
  font-size: 15px; margin: 20px 0 10px; color: #302b63;
  padding-left: 10px; border-left: 3px solid #7B2D8E;
}}
.strat-box {{
  background: #f8f9fa; border-radius: 10px; padding: 16px 20px;
  margin-bottom: 14px; font-size: 13.5px; line-height: 1.7;
}}
.strat-box .strat-name {{
  font-weight: 700; font-size: 14.5px; margin-bottom: 6px;
}}
.strat-box code {{
  background: #e8eaf6; padding: 2px 6px; border-radius: 4px;
  font-size: 12.5px; font-family: 'SF Mono', Menlo, monospace;
}}
.strat-box .tag {{
  display: inline-block; padding: 2px 10px; border-radius: 12px;
  font-size: 11px; font-weight: 600; margin-left: 8px;
}}
.strat-box .tag.best {{ background: #c8e6c9; color: #2e7d32; }}
.strat-box .tag.fail {{ background: #ffcdd2; color: #c62828; }}
.strat-box .tag.base {{ background: #e0e0e0; color: #555; }}
.flow-diagram {{
  background: #1a1a2e; color: #e0e0e0; border-radius: 10px;
  padding: 20px 24px; margin: 16px 0; font-family: 'SF Mono', Menlo, monospace;
  font-size: 13px; line-height: 1.8; white-space: pre; overflow-x: auto;
}}
.flow-diagram .hl-orange {{ color: #FF6B35; font-weight: 700; }}
.flow-diagram .hl-blue {{ color: #4FC3F7; font-weight: 700; }}
.flow-diagram .hl-green {{ color: #81C784; font-weight: 700; }}
.flow-diagram .hl-pink {{ color: #EF476F; font-weight: 700; }}
.role-table {{
  width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13.5px;
}}
.role-table th {{
  background: #ede7f6; padding: 10px 14px; text-align: left;
  font-weight: 600; border-bottom: 2px solid #b39ddb;
}}
.role-table td {{
  padding: 10px 14px; border-bottom: 1px solid #eee;
}}
.role-table tr:nth-child(even) {{ background: #fafafa; }}

.footer {{
  text-align: center; padding: 32px; font-size: 12px; color: #aaa;
}}
</style>
</head>
<body>

<div class="header">
  <h1>IS Backtest Report</h1>
  <div class="sub">HMM + VIX Slope Change Integration&ensp;|&ensp;IS: {IS_START} ~ {IS_END}&ensp;|&ensp;{len(bt)} trading days</div>
</div>

<div class="nav" id="topnav">
  <a href="#summary">Summary</a>
  <a href="#dashboard">Dashboard</a>
  <a href="#cumulative">Cumulative</a>
  <a href="#drawdown">Drawdown</a>
  <a href="#heatmap">Monthly</a>
  <a href="#timeline">Timeline</a>
  <a href="#hitrate">Hit Rate</a>
  <a href="#violin">Distribution</a>
  <a href="#strategy">Strategy</a>
  <a href="#findings">Findings</a>
</div>

<div class="container">

  <div class="summary-card" id="summary">
    <h2>전략 성과 비교 (IS 구간)</h2>
    <table>
      <thead>
        <tr><th>전략</th><th>총수익률</th><th>연환산</th><th>Sharpe</th><th>MDD</th><th>변동성</th></tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>

{chart_sections}

  <!-- Strategy Explanation -->
  <div class="strategy-card" id="strategy">
    <h2>slope_change x HMM 통합 전략 설계</h2>

    <h3>기본 전제: 두 모듈의 역할 분리</h3>
    <table class="role-table">
      <thead><tr><th>모듈</th><th>입력</th><th>답하는 질문</th><th>해상도</th></tr></thead>
      <tbody>
        <tr><td><strong>slope_change</strong></td><td>VIX 9D - VIX 30D 일변화</td><td>오늘 Long인지 Short인지 (방향)</td><td>일봉</td></tr>
        <tr><td><strong>HMM</strong></td><td>VIX 4종 PCA + delta_slope</td><td>지금 시장이 안정/위험한지 (상태)</td><td>일봉</td></tr>
        <tr><td><strong>LSTM-AE</strong> (Phase B)</td><td>VXX-BTC 1분봉 관계</td><td>장중에 이상 징후가 있는지 (교체)</td><td>분봉</td></tr>
      </tbody>
    </table>

    <h3>테스트한 4가지 전략</h3>

    <div class="strat-box">
      <div class="strat-name">A. slope_change 단독 <span class="tag best">Best Return</span></div>
      <strong>방향:</strong> <code>slope_change &gt; 0</code> &rarr; Short, <code>slope_change &lt; 0</code> &rarr; Long<br>
      <strong>레버리지:</strong> |slope_change| 크기 4단계 &mdash; Q1: 0.5x, Q2: 1.0x, Q3: 1.5x, Q4: 2.0x<br>
      <strong>HMM:</strong> 미사용<br>
      <strong>결과:</strong> 총수익 +467%, Sharpe 2.147, MDD -68%
    </div>

    <div class="strat-box">
      <div class="strat-name">B. HMM 단독 (Champion theta) <span class="tag base">Baseline</span></div>
      <strong>방향:</strong> HMM Soft Leverage가 결정 (Normal&rarr;Long, Fear&rarr;Short)<br>
      <strong>레버리지:</strong> <code>Lev = P(N)&times;1.5 + P(A)&times;0.5 + P(F)&times;(-0.5)</code><br>
      <strong>slope:</strong> 미사용<br>
      <strong>결과:</strong> 총수익 +94%, Sharpe 0.772, MDD -37%
    </div>

    <div class="strat-box">
      <div class="strat-name">C. 통합 Scale (slope 방향 x HMM 스케일링) <span class="tag fail">Failed</span></div>
      <strong>방향:</strong> slope_change가 결정<br>
      <strong>레버리지:</strong> slope 레버리지 &times; HMM 스케일 팩터<br>
      &emsp;&emsp;<code>scale = P(N)&times;1.0 + P(A)&times;0.5 + P(F)&times;(-0.3)</code><br>
      <strong>실패 원인:</strong> Fear 구간에서 scale이 음수 &rarr; slope 방향이 반전 &rarr; 양쪽 다 틀림<br>
      <strong>결과:</strong> 총수익 -8%, Sharpe -0.076, MDD -69%
    </div>

    <div class="strat-box" style="border: 2px solid #43a047;">
      <div class="strat-name">D. 통합 Cap (slope 방향 + HMM 레버리지 상한) <span class="tag best">Adopted</span></div>
      <strong>방향:</strong> slope_change가 결정 (A와 동일, 절대 변경 안 함)<br>
      <strong>레버리지:</strong> slope 레버리지를 HMM 상태별 상한(Cap)으로 제한<br>
      &emsp;&emsp;Normal &rarr; cap 2.0x (slope 그대로)<br>
      &emsp;&emsp;Alert &rarr; cap 1.0x (slope 1.5x/2.0x여도 1.0x로 축소)<br>
      &emsp;&emsp;Fear &rarr; cap 0.5x (최소 레버리지로 제한)<br>
      <strong>핵심:</strong> slope의 방향 판단은 건드리지 않고, HMM은 "위험할 때 크기만 줄이는" 역할<br>
      <strong>결과:</strong> 총수익 +104%, Sharpe 0.811, MDD -66%
    </div>

    <h3>최종 채택 구조</h3>
    <div class="flow-diagram"><span class="hl-orange">slope_change</span>  ──&rarr;  방향 + 기본 레버리지 결정
       │
<span class="hl-blue">HMM 상태</span>      ──&rarr;  레버리지 상한(Cap) 적용
       │
       ▼
  <span class="hl-green">오늘의 포지션 진입</span>
       │
<span class="hl-pink">LSTM-AE</span>      ──&rarr;  장중 이상 탐지 시 포지션 교체  (Phase B)
       │
   15:59 ET 청산</div>

    <h3>왜 이 구조인가?</h3>
    <div class="strat-box">
      <strong>1. slope_change의 방향 판단이 이미 유효 (Hit Rate 55.1%, Q4에서 61.9%)</strong><br>
      &emsp;&rarr; HMM이 방향에 개입하면 noise 추가. 건드리지 않는 게 최선.<br><br>
      <strong>2. HMM은 방향 예측보다 상태 분류에 강점</strong><br>
      &emsp;&rarr; "위험할 때 노출 줄이기"가 HMM의 자연스러운 역할.<br><br>
      <strong>3. 각 모듈이 서로 다른 시간 해상도에서 독립적으로 작동</strong><br>
      &emsp;&rarr; slope(일봉 방향) + HMM(일봉 리스크) + LSTM-AE(분봉 교체). 겹치지 않음.
    </div>
  </div>

  <div class="findings" id="findings">
    <h2>핵심 발견</h2>
    <div class="pill g">
      <strong>slope_change 방향 판단력 확인</strong><br>
      전체 Hit Rate 55.1%. |slope_change| 상위 25%(Q4)에서 61.9% — 신호가 강할수록 정확도 증가.
    </div>
    <div class="pill w">
      <strong>HMM 통합 주의</strong><br>
      방향 결정에 직접 개입(스케일링) 시 성과 하락. Cap 방식(레버리지 상한 제한)만 유의미하나 MDD 개선 미미 (-68% &rarr; -66%).
    </div>
    <div class="pill r">
      <strong>MDD 리스크</strong><br>
      Slope 단독 MDD -68%는 실전 불가. LSTM-AE 장중 교체 또는 레버리지 축소 필수.
    </div>
    <div class="pill g">
      <strong>역할 분리 확인</strong><br>
      slope_change = 방향(일봉), HMM = 리스크 캡(일봉), LSTM-AE = 장중 교체(분봉). 서로 다른 시간 해상도에서 보완.
    </div>
  </div>

</div>

<div class="footer">integrated_backtest_charts.py &ensp;|&ensp; VIX-to-BTC Project</div>

<script>
// Nav active highlight on scroll
const sections = document.querySelectorAll('.summary-card, .chart-card, .findings');
const navLinks = document.querySelectorAll('.nav a');
window.addEventListener('scroll', () => {{
  let cur = '';
  sections.forEach(s => {{ if (window.scrollY >= s.offsetTop - 80) cur = s.id; }});
  navLinks.forEach(a => {{
    a.classList.remove('active');
    if (a.getAttribute('href') === '#' + cur) a.classList.add('active');
  }});
}});
</script>

</body>
</html>"""

output_path = OUTPUT_DIR / "is_backtest_report.html"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n{'='*60}")
print(f"  통합 리포트: {output_path}")
print(f"{'='*60}")

import webbrowser
webbrowser.open(str(output_path))
