# 팀 라벨링 결과 분석

팀원 5명이 `팀원1`~`팀원5` 시트의 동일한 30문장을 모두 라벨링한 뒤 실행합니다.

## 입력 규칙

- `1` = 보완 필요(누락)
- `0` = 충분함 또는 해당 요청에서는 추가 필요 없음
- 라벨 순서: Task, Audience, Context, Format, Tone, Length, Constraint, Example

## 실행

프로젝트 루트에서:

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\run_label_analysis.ps1 -InputFile "C:\path\Promptune_8요소_팀라벨링_30문장.xlsx"
```

팀원들이 아직 작성 중인 파일도 중간 확인할 수 있습니다.

```powershell
.\scripts\run_label_analysis.ps1 `
  -InputFile "C:\path\Promptune_8요소_팀라벨링_30문장.xlsx" `
  -AllowPartial
```

## 결과

기본 출력 폴더: `artifacts/labeling-analysis/`

- `01_completeness.csv` — 팀원별 240개 라벨 입력 완료율
- `02_element_agreement.csv` — 8요소별 완전 일치율, 평균 다수 의견 일치율, Fleiss' kappa
- `03_sentence_agreement.csv` — 문장별 몇 개 요소에서 의견이 갈렸는지
- `04_disagreements.csv` — 실제 불일치 항목과 팀원별 0/1 값, 메모
- `05_ground_truth_candidates.csv` — 5명 다수결 기반 Ground Truth **후보**
- `summary.json` — 전체 결과의 기계 판독용 JSON
- `SUMMARY.md` — 사람이 빠르게 볼 수 있는 요약

## 회의 때 보는 순서

1. `01_completeness.csv`에서 전원 100%인지 확인
2. `02_element_agreement.csv`에서 가장 일치도가 낮은 요소 확인
3. `04_disagreements.csv`에서 해당 문장/요소만 토론
4. `05_ground_truth_candidates.csv`의 다수결 값은 참고만 하고 최종 Ground Truth를 팀 합의로 확정

## 지표 해석

- **완전 일치율**: 한 문장의 해당 요소에 대해 5명이 전부 같은 값을 준 비율
- **평균 다수 의견 일치율**: 각 문장에서 다수 의견에 몇 명이 동의했는지 평균한 값
- **Fleiss' kappa**: 단순 우연 일치를 보정한 다중 평가자 일치도 지표. 수치가 낮다면 해당 요소의 라벨 기준을 다시 검토해야 합니다.

이 분석의 목적은 팀원 평가에 점수를 매기는 것이 아니라 **라벨 정의가 흔들리는 지점을 찾는 것**입니다.
