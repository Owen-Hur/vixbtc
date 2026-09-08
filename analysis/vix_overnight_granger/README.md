# VIX → BTC 분석 최종 산출물 (P0 종결)

> 핵심 질문: **"VIX 종가(16:00 ET) 발표 후, 그 영향이 24시간 거래되는 BTC에 얼마나 오래 남아 예측에 쓸 수 있는가?"**
> 최종 결론: **VIX→BTC는 동시적(contemporaneous) 관계만 강하고, 시차 예측력은 없다 → 방향성 진입 전략 구조적 불가.**

---

## 📖 읽는 순서

### 1단계 — `reports/VIX_overnight_persistence_FULL_REPORT.md` ⭐ 먼저 읽기
야간(VIX 동결 구간) BTC의 분 단위 반응 분석. 아이디어의 출발점과 검증.
- 같이 볼 차트: `charts/vix_overnight_minute_corr.png` (분단위 상관·적중률 곡선), `charts/vix_overnight_summary.png`
- 분단위 원자료: `data/vix_overnight_minute_corr.csv`

### 2단계 — `reports/VIX_granger_report.md` ⭐ 다음 읽기
1단계의 "예측 불가" 결론을 정식 통계검정(Granger 인과)으로 종결. ΔVIX·기간구조 slope·시간별까지 양방향 검정.
- 같이 볼 차트: `charts/vix_granger.png` (일별/slope/시간별 p값 곡선)

### (참고) `reports/vix_overnight_persistence_report.md`
1단계의 초기 간이 버전. FULL_REPORT가 상위호환이므로 생략 가능.

---

## 📂 폴더 구성
- `reports/` — 보고서 3종 (markdown)
- `charts/` — 차트 4종 (png)
- `code/` — 재현 코드 3종 (python)
- `data/` — 분단위 상관 원자료 (csv)

## 🔁 재현 방법 (btc_project 루트에서 실행)
```
python3 code/vix_overnight_full_report.py   # 1단계: 야간 분단위 분석
python3 code/vix_granger.py                 # 2단계: Granger 인과
```
(데이터 경로 `data/vix_daily.parquet`, `data/btc_1m_24h/`, `data/vix_1h.parquet`, `data/vix_slope_daily.parquet` 필요)
