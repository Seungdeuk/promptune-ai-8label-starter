# Promptune V6 초기 PC 모델 테스트 가이드

> 기준 시점: 2026-08-18  
> 대상 저장소: `promptune-ai-8label-starter`  
> 목적: 팀원이 새 PC에서 동일한 환경으로 Promptune V6 KcELECTRA 모델 테스트를 시작하기 위한 초기 세팅·검증 절차

---

## 1. 이 가이드의 범위

이 문서는 **V6 모델 학습을 시작하기 전 PC별 환경을 동일하게 맞추고, 데이터셋과 GPU가 정상적으로 동작하는지 검증하는 절차**를 설명한다.

현재 V6 공식 데이터 구성은 다음과 같다.

- Train: **2,000문장**
- Valid: **400문장**
- 라벨 수: **8개**
- 입력 최대 길이: **256**
- 모델: `beomi/KcELECTRA-base`
- 공식 Seed 후보: **42 / 52 / 62**
- 최종 Benchmark: **210문장**
- Benchmark는 최종 Seed 및 Threshold 확정 후 **지정된 1대의 PC에서 1회만 평가**

8개 라벨:

1. Task
2. Audience
3. Context
4. Format
5. Tone
6. Length
7. Constraint
8. Example

---

## 2. 현재 저장소 구조

현재 GitHub 업로드 기준 구조는 다음과 같다.

```text
promptune-ai-8label-starter/
│
├─ app/                         # Promptune AI 서비스/API 코드
├─ ml/                          # 모델 학습·평가 핵심 코드
│  ├─ dataset.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ metrics.py
│  ├─ tune_thresholds.py
│  └─ tune_thresholds_v2.py
│
├─ scripts/
│  ├─ team/                     # 팀원이 주로 실행하는 스크립트
│  │  ├─ check_env.ps1
│  │  └─ run_dev.ps1
│  ├─ v6/
│  │  ├─ 01_build/              # V6 데이터 생성 이력
│  │  ├─ 02_review/             # Human review / merge 이력
│  │  ├─ 03_freeze/             # Frozen / training-ready 생성
│  │  └─ 04_validate/           # V6 데이터셋 검증
│  └─ benchmark/                # 최종 Benchmark 생성·감사 이력
│
├─ data/
│  ├─ v5/
│  │  └─ frozen-v5-v1/          # V5 baseline 재현용 Frozen 데이터
│  ├─ v6/
│  │  ├─ labeling/
│  │  │  ├─ frozen/             # V6 Train 2000 Frozen
│  │  │  └─ valid_human_review/
│  │  │     └─ frozen/          # V6 Valid 400 Frozen
│  │  └─ training_ready/        # 실제 모델 학습 입력
│  └─ final-holdout/
│     └─ final-benchmark-v1/    # Frozen Benchmark 210
│
├─ results/                     # Git에 보관하는 소형 평가 결과
├─ artifacts/                   # 모델/checkpoint 로컬 저장소, Git 제외
├─ tests/
├─ docs/
│
├─ .env.example
├─ .gitignore
├─ Dockerfile
├─ README.md
├─ requirements.txt
├─ requirements-ml.txt
└─ requirements-v6-lock.txt
```

### 중요

`scripts/v6/01_build`, `02_review`, `03_freeze`는 **V6 데이터 구축 과정의 provenance를 보존하기 위한 코드**다.

팀원이 모델 학습을 시작할 때 이 스크립트들을 처음부터 다시 실행할 필요는 없다.

실제 학습 시작에 필요한 핵심 데이터는 다음 두 파일이다.

```text
data/v6/training_ready/Promptune_v6_train_2000_training_v1.csv
data/v6/training_ready/Promptune_v6_valid_400_training_v1.csv
```

---

## 3. 기준 환경

현재 기준 PC에서 검증된 환경:

```text
Python       : 3.12.10
PyTorch      : 2.12.1+cu130
CUDA Runtime : 13.0
CUDA Enabled : True
GPU          : NVIDIA GeForce RTX 3050
GPU Memory   : 6 GB

NumPy        : 2.5.1
Pandas       : 3.0.5
Scikit-learn : 1.9.0
Transformers : 5.14.1
Accelerate   : 1.14.0
```

정확한 Python 패키지 snapshot은 `requirements-v6-lock.txt`에 기록한다.

---

## 4. 팀 PC별 권장 Seed 배정

첫 번째 공식 V6 Multi-PC 실험 기준:

| PC | 역할 | Seed |
|---|---|---:|
| PC-01 | 기본 실험 | 42 |
| PC-02 | 기본 실험 | 52 |
| PC-03 | 기본 실험 | 62 |
| PC-04 | 재현성 검증 | 42 |
| PC-05 | 재현성 검증 | 62 |

### 원칙

- 모든 PC는 동일 Git commit을 사용한다.
- 모든 PC는 동일 Train/Valid 파일을 사용한다.
- Seed 외 학습 조건은 동일하게 유지한다.
- PC-04와 PC-05는 재현성 확인을 위한 replicate run이다.
- Benchmark 210은 각 PC에서 반복 실행하지 않는다.

---

## 5. GitHub에서 프로젝트 받기

GitHub 저장소 생성 및 최초 push가 완료된 이후 팀원은 다음 방식으로 clone한다.

```powershell
git clone <REPOSITORY_URL>
cd promptune-ai-8label-starter
```

현재 commit 확인:

```powershell
git rev-parse HEAD
```

팀 실험 시작 전 모든 PC의 commit hash가 동일해야 한다.

Git 작업 트리 확인:

```powershell
git status --short
```

공식 실험 시작 시 불필요한 변경 사항이 없는 상태를 권장한다.

---

## 6. Python 확인

```powershell
python --version
where.exe python
```

권장 기준:

```text
Python 3.12.10
```

Python 3.12 계열이라도 패치 버전이 다를 경우 기록한다.

---

## 7. 가상환경 생성

프로젝트 루트에서 실행한다.

```powershell
python -m venv .venv
```

활성화:

```powershell
.\.venv\Scripts\Activate.ps1
```

정상적으로 활성화되면 PowerShell 앞부분에 다음처럼 표시된다.

```text
(.venv) PS C:\...\promptune-ai-8label-starter>
```

Python 경로 재확인:

```powershell
where.exe python
```

첫 번째 경로가 다음 형태여야 한다.

```text
...\promptune-ai-8label-starter\.venv\Scripts\python.exe
```

---

## 8. pip 업데이트

```powershell
python -m pip install --upgrade pip
```

---

## 9. CUDA용 PyTorch 설치

V6 기준 환경은 다음을 사용한다.

```text
torch 2.12.1 + cu130
```

설치:

```powershell
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
```

`torchvision`, `torchaudio`는 현재 V6 classifier 학습에 필수 항목이 아니다.

---

## 10. 나머지 ML 의존성 설치

```powershell
python -m pip install -r .\requirements-ml.txt
```

---

## 11. 환경 확인

팀 공용 환경 점검 스크립트:

```powershell
.\scripts\team\check_env.ps1
```

기준 PC의 정상 출력 예:

```text
Python 3.12.10
Torch: 2.12.1+cu130
CUDA: True
GPU: NVIDIA GeForce RTX 3050
Labels: ('task', 'audience', 'context', 'format', 'tone', 'length', 'constraint', 'example')
num_labels: 8
```

### CUDA가 False인 경우

공식 GPU 모델 학습을 시작하지 않는다.

먼저 다음을 확인한다.

```powershell
nvidia-smi
```

그리고:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

---

## 12. 전체 패키지 버전 확인

```powershell
python -c "import torch, numpy, pandas, sklearn, transformers, accelerate; print('Torch:', torch.__version__); print('NumPy:', numpy.__version__); print('Pandas:', pandas.__version__); print('Sklearn:', sklearn.__version__); print('Transformers:', transformers.__version__); print('Accelerate:', accelerate.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

팀 PC별 결과는 실험 시작 전에 기록한다.

---

## 13. V6 Dataset Load 검증

프로젝트 루트에서 실행한다.

```powershell
python .\scripts\v6\04_validate\check_v6_kcelectra_dataset_load.py
```

### 정상 기준

```text
=== V6 KcELECTRA DATASET LOAD CHECK ===

MODEL      : beomi/KcELECTRA-base
NUM_LABELS : 8

=== CUDA ===
cuda avail  : True

=== DATASET LOAD ===
train rows  : 2000
valid rows  : 400

=== SAMPLE CHECK ===
input shape : (256,)
label shape : (8,)

V6 KcELECTRA DATASET LOAD CHECK: PASS
```

`PASS`가 나오지 않으면 공식 학습을 시작하지 않는다.

---

## 14. Frozen 데이터 무결성 확인

### V6 Train

```powershell
Get-FileHash `
  .\data\v6\labeling\frozen\Promptune_v6_train_2000_frozen_v1.csv `
  -Algorithm SHA256
```

### V6 Valid

```powershell
Get-FileHash `
  .\data\v6\labeling\valid_human_review\frozen\Promptune_v6_valid_400_frozen_v1.csv `
  -Algorithm SHA256
```

저장소의 `.sha256` 또는 manifest 값과 비교한다.

실제 학습에는 아래 training-ready 파일을 사용한다.

```text
data\v6\training_ready\Promptune_v6_train_2000_training_v1.csv
data\v6\training_ready\Promptune_v6_valid_400_training_v1.csv
```

팀 PC 5대에서 데이터 파일이 동일해야 한다.

---

## 15. 학습 CLI 확인

실제 학습 전에 현재 CLI 옵션을 확인한다.

```powershell
python -m ml.train --help
```

### 주의

V6 공식 학습 전 다음 하이퍼파라미터는 팀에서 최종 확정 후 고정한다.

- Epoch
- Batch size
- Learning rate
- Threshold tuner 선택
  - `ml.tune_thresholds`
  - `ml.tune_thresholds_v2`

과거 V4/V5 설정을 자동으로 재사용하지 않는다.

---

## 16. Seed별 학습 원칙

공식 학습 명령은 `python -m ml.train --help`의 현재 옵션과 팀에서 확정한 하이퍼파라미터를 기준으로 작성한다.

```text
PC-01 → Seed 42
PC-02 → Seed 52
PC-03 → Seed 62
PC-04 → Seed 42 replicate
PC-05 → Seed 62 replicate
```

학습 조건은 Seed를 제외하고 모두 동일해야 한다.

---

## 17. Threshold Tuning 원칙

Threshold는 **Valid 400에서만 조정**한다.

```text
Train 2000
   ↓
Seed별 학습
   ↓
Valid 400 예측
   ↓
Threshold Tuning
   ↓
Seed별 성능 비교
   ↓
최종 Seed + Threshold 확정
```

Frozen Benchmark 210을 threshold 결정 과정에 사용하면 안 된다.

---

## 18. Frozen Benchmark 210 사용 규칙

파일:

```text
data/final-holdout/final-benchmark-v1/Promptune_final_benchmark_210_v1.csv
```

Benchmark는 다음 순서로 사용한다.

```text
최종 Seed 결정
        +
최종 Threshold 결정
        ↓
Benchmark 210 단 1회 평가
```

하지 말아야 할 것:

```text
Seed42 → Benchmark
Seed52 → Benchmark
Seed62 → Benchmark
```

Benchmark를 validation처럼 반복 사용하지 않는다.

---

## 19. V5 Baseline 비교

V5 비교용 Frozen 데이터:

```text
data/v5/frozen-v5-v1/
```

V5 기존 Benchmark 결과:

```text
results/final-benchmark-v1/v5-seed52-v2/
```

로컬 V5 모델 artifact는 GitHub에 올리지 않는다.

---

## 20. HyperCLOVA X 관련

현재 V6 KcELECTRA classifier 학습 자체에는 HyperCLOVA X가 필요하지 않다.

Promptune 서비스의 Normalizer용 모델로 사용하는 경우:

```text
naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B
```

를 로컬 inference 방식으로 사용한다.

실제 secret 값은 `.env`에만 저장하고 GitHub에는 올리지 않는다.

`.env.example`에는 실제 key를 기록하지 않는다.

---

## 21. Git 보안 규칙

GitHub에 올라가면 안 되는 항목:

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
git ls-files | Select-String "^\.env$"
```

Private 데이터 확인:

```powershell
git ls-files |
    Where-Object { $_ -match "PRIVATE|DO_NOT_SHARE" }
```

두 명령 모두 정상적으로는 출력이 없어야 한다.

---

## 22. 대용량 Git 파일 확인

```powershell
git ls-files |
    ForEach-Object {
        if (Test-Path $_) {
            $f = Get-Item $_
            if ($f.Length -gt 50MB) {
                [PSCustomObject]@{
                    File   = $_
                    SizeMB = [math]::Round($f.Length / 1MB, 2)
                }
            }
        }
    }
```

정상적으로는 출력되는 파일이 없어야 한다.

---

## 23. PC별 환경 기록 권장

각 PC에서 최소 다음 정보를 기록한다.

```text
PC ID:
담당 Seed:
Git Commit:
Python:
Torch:
CUDA Runtime:
CUDA Available:
GPU:
GPU Memory:
Transformers:
Accelerate:
Train SHA256:
Valid SHA256:
학습 시작 시간:
학습 종료 시간:
Artifact 경로:
```

---

## 24. 초기 PC 검증 완료 체크리스트

- [ ] Git clone 완료
- [ ] 팀 공통 Git commit 확인
- [ ] Python 3.12 환경 확인
- [ ] `.venv` 생성 및 활성화
- [ ] Torch 2.12.1+cu130 설치
- [ ] CUDA `True`
- [ ] GPU 정상 인식
- [ ] `requirements-ml.txt` 설치
- [ ] `check_env.ps1` 정상
- [ ] V6 Train 2000 확인
- [ ] V6 Valid 400 확인
- [ ] Input shape `(256,)`
- [ ] Label shape `(8,)`
- [ ] Dataset Load Check `PASS`
- [ ] Frozen 데이터 SHA256 확인
- [ ] `python -m ml.train --help` 정상
- [ ] 담당 Seed 확인
- [ ] Benchmark 210 사전 사용 금지 확인

---

## 25. 권장 초기 실행 순서 요약

```powershell
# 1. 저장소 이동
cd promptune-ai-8label-starter

# 2. 가상환경
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. pip
python -m pip install --upgrade pip

# 4. PyTorch CUDA 13.0
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130

# 5. ML dependencies
python -m pip install -r .\requirements-ml.txt

# 6. Environment check
.\scripts\team\check_env.ps1

# 7. V6 dataset check
python .\scripts\v6\04_validate\check_v6_kcelectra_dataset_load.py

# 8. Training CLI check
python -m ml.train --help
```

여기까지 모두 통과하면 해당 PC는 **V6 공식 모델 학습 준비 완료** 상태다.

---

## 26. 현재 공식 실험 흐름

```text
V6 Frozen Train 2000
        +
V6 Frozen Valid 400
        ↓
5개 PC 환경 동일성 확인
        ↓
KcELECTRA Seed 42 / 52 / 62 학습
        ↓
Seed 42 / 62 replicate 확인
        ↓
Valid 400 Threshold Tuning
        ↓
Seed / Threshold 최종 확정
        ↓
Frozen Benchmark 210
        ↓
V5 Baseline 대비 V6 최종 성능 비교
```

---

## 27. 팀 공통 원칙

1. 데이터가 다르면 실험 결과를 직접 비교하지 않는다.
2. Git commit이 다르면 동일 실험으로 간주하지 않는다.
3. Seed 이외의 학습 조건은 동일하게 유지한다.
4. Benchmark 210은 최종 모델 선정 전에 사용하지 않는다.
5. 모델 checkpoint와 `.env`는 GitHub에 업로드하지 않는다.
6. 문제 발생 시 환경 정보와 전체 오류 로그를 함께 공유한다.
7. 학습 실행 전 `check_env.ps1`과 Dataset Load Check를 반드시 통과한다.

---

**문서 버전:** V6 Initial PC Test Guide v1  
**기준일:** 2026-08-18
