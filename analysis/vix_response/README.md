# vix_response/ — VIX 발표 후 BTC 반응 탐색 (중간 단계)

> **이 폴더는 초기 탐색 단계 코드입니다.** 여기서 얻은 결과는 최종 결론을 직접 뒷받침하지 않으며,
> 프로젝트의 최종 결론은 `../slope_change/`, `../vix_duration/`, `../vix_overnight_granger/` 에 있습니다
> (전체 지도는 [`../README.md`](../README.md) 참조).
>
> 남겨 둔 이유는 "VIX 반응이 있는 것처럼 보였던 초기 관찰 → 정렬 오류·동시성 혼입 발견 → 기각"에
> 이르는 과정을 재현 가능한 형태로 보존하기 위해서입니다. **재실행 검증 대상이 아닙니다.**

## 무엇을 물었나

- VIX 종가(16:00 ET) 확정 후 BTC 는 언제·얼마나 반응하는가?
- 그 반응은 **확인한 뒤에도 남아 있어 거래 가능한가**, 아니면 이미 지나간 동시적 움직임인가?
- 실제 거래 가능한 대용물(VIXY)로 바꾸면 결과가 달라지는가?

관찰 결과: 2020~2022 구간에서는 6시간 부근에 반응 피크가 보였으나 2023년 이후 소멸했고,
"VIX 변동을 확인한 뒤" 남는 잔여 반응은 거래 비용을 덮지 못했습니다.
이 결론은 이후 `vix_overnight_granger/` 의 Granger 인과 검정으로 정식 종결되었습니다.

## 폴더 구성

| 폴더 | 내용 |
|------|------|
| `code/` | 데이터 구축·분석 스크립트 13종 (각 파일 상단 docstring 에 입출력 명시) |
| `charts/` | 각 분석 스크립트가 생성한 차트 9종 |
| `reports/` | 중간 단계 보고서 2종 (영향 지속시간 / 트레이딩 전략) |

### `code/` 파일

| 파일 | 역할 |
|------|------|
| `vix_1h_download.py` | yfinance → `data/vix_1h.parquet` (VIX 1시간봉 수집) |
| `rebuild_vix_btc_response.py` | VIX 일별 종가 기준 1분~24h BTC 반응 데이터셋 구축 |
| `vix_1h_btc_response.py` | VIX 1시간봉 기준 1분~4h BTC 반응 데이터셋 구축 |
| `vix_full_analysis.py` | 전체 표본 시차 상관·permutation·dose-response |
| `vix_analysis_2024.py` | 위 분석의 2024년 이후 한정 재검증 |
| `vix_impact_duration.py` | 상관 decay · 1σ event study · VIX 레벨별 상관 |
| `vix_1h_analysis.py` | 1시간봉 기준 전체/장중/2σ 급변 상관 비교 |
| `vix_1h_tradeable.py` | 변동 중 이미 발생한 몫(`btc_during`)과 확인 후 잔여분(`btc_after`) 분리 |
| `vix_strategy_full.py` | 진입조건·보유기간 격자 성과 + Buy & Hold 대조 |
| `vix_strategy_corrected.py` | yfinance 1h bar 타임스탬프 정렬 오류 교정 후 전략 재계산 |
| `vixy_btc_analysis.py` | 거래 가능 대용물 VIXY 5분 2σ 급변 후 BTC 60분 반응 |
| `vixy_effective_duration.py` | VIXY 급변 전후 −5~+120분 BTC 평균 경로와 유효 지속시간 |
| `report_data.py` | 보고서 인용 수치 일괄 출력 (파일 산출 없음) |

## 실행 방법

스크립트는 **저장소 루트를 작업 디렉터리로 가정**하고 `data/...` 상대경로를 참조합니다.

```bash
python3 analysis/vix_response/code/vix_1h_download.py
python3 analysis/vix_response/code/vix_1h_btc_response.py
python3 analysis/vix_response/code/vix_1h_analysis.py
```

차트는 실행한 작업 디렉터리에 저장되므로, `charts/` 를 갱신하려면 생성된 png 를 옮겨 주십시오.
필요한 원본 데이터 수집 방법은 [`../../data/README.md`](../../data/README.md) 를 참조하십시오.
