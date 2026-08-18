# Promptune V6.1 Seed42 실행 가이드

## 이번 버전
- Parent: V6 Train 2000
- 신규 controlled augmentation: 52
- V6.1 Train: 2052
- Valid: 기존 V6 Valid 400 그대로 사용
- 첫 실험 Seed: **42만**
- Model/Loss/Epoch/Batch/LR/Max Length: V6와 동일
- Final Benchmark 210: **접근 금지**

## 파일 배치

프로젝트 루트:
`C:\Users\hi\promptune-ai-8label-starter`

아래처럼 배치하세요.

```text
data/
└─ v6_1/
   └─ training_ready/
      ├─ Promptune_v6_1_train_2052_training_v1.csv
      ├─ Promptune_v6_1_train_2052_training_v1.sha256
      └─ Promptune_v6_1_train_2052_manifest_v1.json

scripts/
├─ team/
│  └─ run_v6_1_experiment.ps1
└─ v6_1/
   └─ 01_validate/
      └─ check_v6_1_dataset_load.py
```

검토 workbook은 학습 입력이 아니므로 원하는 문서/결과 폴더에 보관하면 됩니다.

## 1. 해시 확인

```powershell
Get-FileHash .\data\v6_1\training_ready\Promptune_v6_1_train_2052_training_v1.csv -Algorithm SHA256
```

예상 SHA256:

```text
1ACFA68284D27B8519182ECEA37A83D81CBBC3063FB04FC9CA8DF97DB5D7B77C
```

## 2. Dataset load check

```powershell
python .\scripts\v6_1\01_validate\check_v6_1_dataset_load.py
```

정상 핵심 출력:

```text
train rows  : 2052
valid rows  : 400
V6.1 KcELECTRA DATASET LOAD CHECK: PASS
```

## 3. Git 정리

V6.1 관련 파일을 저장소에 추가하고 커밋하세요.

```powershell
git status
git add data/v6_1 scripts/v6_1 scripts/team/run_v6_1_experiment.ps1
git commit -m "Add V6.1 controlled augmentation experiment"
git status
```

실험 전 `git status`는 clean 권장입니다.

## 4. Dry Run

```powershell
.\scripts\team\run_v6_1_experiment.ps1 -Seed 42 -PcId PC01 -DryRun
```

## 5. Seed42 1회 실행

```powershell
.\scripts\team\run_v6_1_experiment.ps1 -Seed 42 -PcId PC01
```

## 6. 지금 비교할 것

V6 Seed42 baseline과 V6.1 Seed42를 같은 Valid 400에서 비교합니다.

1. V1 Macro F1
2. V1 Micro F1
3. Exact Match
4. Audience / Format / Tone / Length / Example F1
5. 특히 complex-boundary의 FP가 줄었는지
6. Task/Context/Constraint 등 다른 라벨이 악화되지 않았는지

Valid 400의 오류 패턴을 V6.1 설계에 사용했으므로,
여기서의 개선은 **development-set improvement**로 표현합니다.

Final generalization 성능은 모델/threshold를 모두 freeze한 뒤
untouched Benchmark210에서 1회만 평가합니다.
