# Promptune AI Service — V6 8-Label KcELECTRA

Promptune의 **8요소 프롬프트 누락 진단 모델**과 업무용 AI 서비스 코드를 관리하는 저장소입니다.

현재 ML 기준은 **V6 Train 2,000 / Valid 400**이며, KcELECTRA 기반 8-label multi-label classifier의 Seed별 재현성 실험을 진행합니다.

---

## 1. 현재 V6 공식 Baseline

| 항목 | 기준 |
|---|---|
| Base Model | `beomi/KcELECTRA-base` |
| Train | 2,000 |
| Valid | 400 |
| Labels | 8 |
| Epochs | 3 |
| Batch Size | 4 |
| Learning Rate | `2e-5` |
| Max Length | 256 |
| Seeds | 42 / 52 / 62 |
| Primary Threshold Tuner | V1 — label별 F1 최대화 |
| Threshold Search | `0.01 ~ 0.99`, step `0.01` |
| Secondary Tuner | V2 — Precision floor + F2 |
| Final Benchmark | 210문장, 최종 모델 확정 후 1회 |

### 데이터 SHA256

```text
Train:
135525A382616D4CBAB81E632175AB61FFA86BA74AD4A17622FFF9324452DEE7

Valid:
ACECD25DB4FF6B73D47F7AC447331213254C7D73F5F47E1ED3E60CDEB7E14A84
```

---

## 2. 공식 8개 진단 요소

1. **Task** — 무엇을 요청하는지
2. **Audience** — 대상
3. **Context** — 배경·상황정보
4. **Format** — 출력 형식
5. **Tone** — 어조·말투
6. **Length** — 글자 수·분량
7. **Constraint** — 제약 조건
8. **Example** — 예시·참고 형식

라벨 규칙:

```text
1 = 해당 요소가 부족하여 보완할 필요가 있음
0 = 이미 충분하거나 해당 요청 수행에 추가 정보가 필요하지 않음
```

---

## 3. 공식 실험 버전 관리

### 기존 Tag 주의

기존에 배포한:

```text
v6-official-baseline
```

Tag는 Commit `542e64c`를 가리킵니다.

이 버전에서 통합 PowerShell runner가 Python/Transformers의 정상적인 `stderr` 진행 로그를
`NativeCommandError`로 처리해 중단되는 문제가 확인되었습니다.

따라서 **새 공식 학습에는 기존 Tag를 사용하지 않습니다.**

Runner 수정과 이 README가 GitHub에 반영된 후 새 Tag:

```text
v6-official-baseline-v2
```

를 공식 기준으로 사용합니다.

### 팀원이 공식 버전을 받는 방법

```powershell
git pull
git fetch --tags
git switch --detach v6-official-baseline-v2
```

확인:

```powershell
git rev-parse --short HEAD
git rev-parse --short "v6-official-baseline-v2^{}"
git status
```

두 commit 값이 같고 다음 상태여야 합니다.

```text
nothing to commit, working tree clean
```

> Annotated Tag는 Tag 객체 ID와 실제 Commit ID가 다를 수 있으므로  
> 실제 Commit 확인에는 `"v6-official-baseline-v2^{}"`를 사용합니다.

---

## 4. Repository 구조

```text
promptune-ai-8label-starter/
│
├─ app/                         # AI 서비스/API
│
├─ ml/                          # 학습·평가·Threshold tuning
│  ├─ dataset.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ metrics.py
│  ├─ tune_thresholds.py
│  └─ tune_thresholds_v2.py
│
├─ scripts/
│  ├─ team/
│  │  ├─ check_env.ps1
│  │  ├─ run_dev.ps1
│  │  └─ run_v6_experiment.ps1
│  │
│  ├─ v6/
│  │  ├─ 01_build/
│  │  ├─ 02_review/
│  │  ├─ 03_freeze/
│  │  └─ 04_validate/
│  │
│  └─ benchmark/
│
├─ data/
│  ├─ v5/frozen-v5-v1/
│  ├─ v6/
│  │  ├─ labeling/frozen/
│  │  ├─ labeling/valid_human_review/frozen/
│  │  └─ training_ready/
│  └─ final-holdout/final-benchmark-v1/
│
├─ results/                     # Git에 남기는 소형 평가 결과
├─ artifacts/                   # 모델/checkpoint 로컬 저장, Git 제외
├─ tests/
├─ docs/
│
├─ requirements.txt
├─ requirements-ml.txt
└─ requirements-v6-lock.txt
```

### V6 build/review 스크립트

`scripts/v6/01_build`, `02_review`, `03_freeze`는 데이터 구축 provenance 보존용입니다.

팀원이 공식 V6 학습을 위해 이 과정을 다시 실행할 필요는 없습니다.

실제 학습 입력은:

```text
data/v6/training_ready/Promptune_v6_train_2000_training_v1.csv
data/v6/training_ready/Promptune_v6_valid_400_training_v1.csv
```

입니다.

---

## 5. Windows ML 환경

기준 PC에서 검증된 환경:

```text
Python       : 3.12.10
PyTorch      : 2.12.1+cu130
CUDA         : 13.0
CUDA Enabled : True
GPU          : NVIDIA GeForce RTX 3050 6GB
NumPy        : 2.5.1
Pandas       : 3.0.5
Scikit-learn : 1.9.0
Transformers : 5.14.1
Accelerate   : 1.14.0
```

가상환경:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

PyTorch:

```powershell
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
```

ML dependencies:

```powershell
python -m pip install -r .\requirements-ml.txt
```

환경 확인:

```powershell
.\scripts\team\check_env.ps1
```

V6 Dataset 확인:

```powershell
python .\scripts\v6\04_validate\check_v6_kcelectra_dataset_load.py
```

정상 종료:

```text
V6 KcELECTRA DATASET LOAD CHECK: PASS
```

---

## 6. 공식 통합 Experiment Runner

공식 학습은 개별 `ml.train`, `ml.evaluate` 명령을 팀원이 직접 조합하지 않고:

```text
scripts/team/run_v6_experiment.ps1
```

를 사용합니다.

이 runner는 한 번의 실행으로 다음을 자동 처리합니다.

```text
Git / PC / 환경 확인
→ CUDA 확인
→ Train / Valid SHA256
→ Dataset Load Check
→ KcELECTRA 3 Epoch 학습
→ Valid 기본 0.5 평가
→ probabilities / labels 저장
→ V1 Threshold tuning
→ V1 Valid 평가
→ V2 Threshold tuning
→ V2 Valid 평가
→ run-metadata.json
→ 제출용 ZIP 생성
```

### Dry Run

학습 전에 반드시 실행:

```powershell
.\scripts\team\run_v6_experiment.ps1 `
  -Seed 42 `
  -PcId PC02 `
  -DryRun
```

정상:

```text
DRY RUN PASS
```

### PowerShell 실행 차단이 발생하는 경우

Git clone 파일은 일반적으로 별도 조치가 필요하지 않습니다.

파일을 브라우저 등에서 직접 다운로드한 경우에만:

```powershell
Unblock-File .\scripts\team\run_v6_experiment.ps1
```

후 다시 실행합니다.

---

## 7. 팀 실험 운영

### 메인 실험 PC

메인 PC에서는 동일 환경에서 Seed 3개를 모두 실행합니다.

```powershell
.\scripts\team\run_v6_experiment.ps1 -Seed 42 -PcId PC01
.\scripts\team\run_v6_experiment.ps1 -Seed 52 -PcId PC01
.\scripts\team\run_v6_experiment.ps1 -Seed 62 -PcId PC01
```

이 세 결과가 **Seed 비교의 Primary Experiment**입니다.

### 팀원 PC

팀원은 배정받은 Seed **하나만** 실행합니다.

예:

```powershell
.\scripts\team\run_v6_experiment.ps1 -Seed 52 -PcId PC03
```

팀원 실험은 다른 하드웨어에서도 같은 Seed가 유사하게 재현되는지 확인하는
**Replication Experiment** 용도입니다.

---

## 8. 결과 폴더와 제출 파일

실행 결과 예:

```text
artifacts/
└─ v6-e3-seed52-pc03-YYYYMMDD-HHMMSS/
```

전체 모델 artifact는 로컬에 저장됩니다.

팀 공유용 ZIP도 자동 생성됩니다.

```text
v6-e3-seed52-pc03-YYYYMMDD-HHMMSS-submission.zip
```

ZIP에는 모델 weight/checkpoint를 제외하고 다음과 같은 비교 자료가 포함됩니다.

```text
run-metadata.json
valid-default05.json
valid-tuned-v1.json
thresholds-v1.json
valid-tuned-v2.json
thresholds-v2.json
thresholds-v2_report.json
valid-probabilities.npy
valid-labels.npy
pip-freeze.txt
git-status.txt
nvidia-smi.txt
run.log
```

다음 대용량 파일은 제출 ZIP에 포함하지 않습니다.

```text
model.safetensors
optimizer.pt
scheduler.pt
checkpoint-*
```

---

## 9. Threshold 정책

### Primary — V1

`ml/tune_thresholds.py`

- label별 F1 최대화
- Search range: `0.01 ~ 0.99`
- Step: `0.01`

현재 공식 Seed 비교의 주 Threshold 방식입니다.

### Secondary — V2

`ml/tune_thresholds_v2.py`

- 기본 Precision floor: `0.55`
- F2 / Recall 중심
- V1과 비교 분석용

Seed42 사전 검증에서는 V1이 V2보다 Macro F1 / Micro F1 / Exact Match 측면에서 더 안정적이었지만,
**최종 방식은 Seed 42 / 52 / 62 결과를 모두 확인한 후 결정합니다.**

---

## 10. Final Benchmark 210 사용 규칙

파일:

```text
data/final-holdout/final-benchmark-v1/Promptune_final_benchmark_210_v1.csv
```

### 금지

```text
Seed42 → Benchmark
Seed52 → Benchmark
Seed62 → Benchmark
```

처럼 Benchmark로 Seed를 선택하지 않습니다.

### 공식 순서

```text
Train 2000
   ↓
Seed 42 / 52 / 62
   ↓
Valid 400 비교
   ↓
Replication 안정성 확인
   ↓
최종 Seed 결정
   ↓
최종 Threshold 결정
   ↓
모델 동결
   ↓
Frozen Benchmark 210 최종 1회
```

Benchmark 210은 validation이나 threshold tuning에 사용하지 않습니다.

---

## 11. API / Mock 실행

서비스 API만 확인할 때는 ML 모델 없이 mock provider를 사용할 수 있습니다.

`.env`:

```env
NORMALIZER_PROVIDER=mock
DIAGNOSIS_PROVIDER=mock
```

실행:

```powershell
.\scripts\team\run_dev.ps1
```

확인:

```text
GET  http://localhost:9000/health
GET  http://localhost:9000/docs
POST http://localhost:9000/ai/analyze
```

Mock 결과는 API/UI 개발 확인용이며 모델 성능으로 보고하지 않습니다.

---

## 12. HyperCLOVA X

현재 Promptune에서 HyperCLOVA X는 KcELECTRA 8-label classifier 학습과 별개입니다.

로컬 Normalizer 후보 모델:

```text
naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B
```

현재 `app/services/normalizer/clova.py`는 provider adapter 경계이며,
로컬 HyperCLOVA inference가 완전히 연결된 상태로 간주하지 않습니다.

따라서 해당 adapter가 구현·검증되기 전에는:

```env
NORMALIZER_PROVIDER=mock
```

을 유지합니다.

---

## 13. Git / 보안 규칙

GitHub에 올리지 않는 항목:

```text
.env
.venv/
artifacts/*
*.safetensors
*.pt
*.pth
*PRIVATE*
*DO_NOT_SHARE*
```

확인:

```powershell
git status
git ls-files | Select-String "^\.env$"
```

`.env`는 출력되지 않아야 합니다.

실제 Hugging Face Token / API Key / 인증정보는 GitHub, README, 팀 채팅에 기록하지 않습니다.

---

## 14. 테스트

현재 테스트:

```powershell
python -m pytest -q
```

공식 코드 변경 후에는 테스트와 Dry Run을 다시 통과한 뒤 새로운 공식 Tag를 생성합니다.

---

## 15. 관련 문서

```text
docs/INITIAL_PC_MODEL_TEST_GUIDE.md
docs/Promptune_V6_Team_Precheck_Notice.md
docs/ARCHITECTURE.md
docs/LABELING_ANALYSIS.md
```

---

## V6 공식 실험 체크리스트

- [ ] 최신 공식 Tag fetch
- [ ] 공식 Tag checkout
- [ ] Working tree clean
- [ ] Python / Torch 버전 확인
- [ ] CUDA `True`
- [ ] GPU 정상 인식
- [ ] Train SHA256 일치
- [ ] Valid SHA256 일치
- [ ] Dataset Load Check PASS
- [ ] Dry Run PASS
- [ ] 배정 Seed 확인
- [ ] 공식 runner로 학습
- [ ] submission ZIP 보관
- [ ] Benchmark 210 사전 사용 금지
