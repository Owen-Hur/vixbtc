# results/

## `is_backtest_report.html`

`analysis/other_signals/integrated_backtest_charts.py` 가 생성한 IS 구간 통합 백테스트 리포트
(Plotly, 2026-05-25 생성).

> ### ⚠️ 중간 단계 산출물 — lookahead bias 수정 이전
>
> 이 리포트가 보고하는 slope_change Hit Rate 55.1% (Q4 61.9%) 및 관련 성과 지표는
> **lookahead bias가 제거되지 않은 상태의 결과** 입니다.
> T-day 시간축으로 재검증한 결과는 50.5%(무작위 수준)이며, 유효한 결론은
> [`../analysis/slope_change/slope_change_report.md`](../analysis/slope_change/slope_change_report.md) 를 보십시오.
>
> 연구 과정 기록 보존 목적으로만 포함했습니다.

### 미포함 산출물

같은 스크립트가 생성한 Plotly 대시보드 6종
(`1_cumulative_returns.html`, `2_drawdown.html`, `4_timeline.html`, `5_hit_rate_analysis.html`,
`6_dashboard.html`, `7_return_distribution.html`)은 각 ~4.9MB(합계 ~29MB)이며,
**모두 위와 동일한 폐기된 결과를 시각화** 하므로 저장소에 포함하지 않았습니다.
필요 시 원본 데이터를 재수집한 뒤 생성 스크립트를 실행하면 재현됩니다.

### 유효한 결과 차트

lookahead-free 재검증 결과 차트는 아래에 있습니다.

- `analysis/slope_change/charts/` — 분 단위 상관·적중률, 크기별 적중률, 수익률 분포
- `analysis/vix_duration/charts/` — 1,035시점 metrics, 2σ 이벤트 평균 경로
- `analysis/vix_overnight_granger/charts/` — 야간 지속성, Granger p값 곡선
