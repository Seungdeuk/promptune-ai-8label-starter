# 📢 Promptune V6 모델 테스트 사전 준비 안내

> **용도:** 팀원 배포용  
> **대상 저장소:** `promptune-ai-8label-starter`  
> **현재 단계:** 학습 전 PC 환경·데이터 준비 확인

---

## 🎨 표시 기준

- 🔴 **필수** — 반드시 확인하거나 제출해야 하는 항목
- 🟡 **권장** — 문제가 생겼을 때 비교·원인 분석에 유용한 항목
- 🟢 **완료 기준** — 이 상태가 되면 해당 단계 준비 완료
- 🔵 **참고** — 이해를 돕기 위한 설명

> GitHub Markdown은 임의의 글자색을 안정적으로 지원하지 않으므로,  
> GitHub에서도 동일하게 보이도록 **색상 이모지 + 굵은 글씨**로 표시했습니다.

---

# 1. 현재 프로젝트 상태

Promptune V6 모델 테스트용 GitHub 저장소 정리가 완료되었습니다.

**Repository**

```text
promptune-ai-8label-starter
```

🔴 **저장소는 Private이므로 팀장의 GitHub 초대를 먼저 수락해야 합니다.**

현재 사전 검증 기준은 다음과 같습니다.

```text
Python       : 3.12.10
PyTorch      : 2.12.1+cu130
CUDA         : True
Model        : beomi/KcELECTRA-base
Train        : 2,000
Valid        : 400
Max Length   : 256
Labels       : 8
```

8개 진단 라벨:

```text
Task / Audience / Context / Format
Tone / Length / Constraint / Example
```

---

# 2. 🔴 지금은 모델 학습을 시작하지 않습니다

현재 단계에서는 **각 PC가 동일한 코드와 데이터로 실행 가능한지 확인하는 것까지만 진행**합니다.

아직 팀 공통으로 확정하지 않은 항목:

```text
Epoch
Batch Size
Learning Rate
Threshold Tuner
PC별 공식 Seed 실행 명령
```

🔴 **개별적으로 임의의 학습 조건으로 먼저 모델을 학습하지 말아주세요.**

공식 학습 조건은 전체 PC 환경 확인 후 한 번에 공유합니다.

---

# 3. GitHub 저장소 받기

GitHub 초대를 수락한 뒤 저장소를 clone합니다.

```powershell
git clone <공유받은 Repository URL>
cd promptune-ai-8label-starter
```

현재 코드 버전 확인:

```powershell
git rev-parse --short HEAD
```

🔴 **팀에서 공지한 기준 Commit과 동일한지 확인해주세요.**

작업 트리 확인:

```powershell
git status --short
```

🟢 처음 clone한 상태라면 별도 변경 사항이 없는 것이 정상입니다.

---

# 4. Python 가상환경 준비

Python 버전 확인:

```powershell
python --version
where.exe python
```

기준:

```text
Python 3.12.10
```

가상환경 생성:

```powershell
python -m venv .venv
```

가상환경 활성화:

```powershell
.\.venv\Scripts\Activate.ps1
```

활성화 후 PowerShell 앞에 다음과 같이 표시되어야 합니다.

```text
(.venv) PS ...
```

---

# 5. PyTorch 및 ML 패키지 설치

pip 업데이트:

```powershell
python -m pip install --upgrade pip
```

CUDA 13.0용 PyTorch:

```powershell
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
```

나머지 ML 의존성:

```powershell
python -m pip install -r .\requirements-ml.txt
```

---

# 6. 🔴 PC 환경 확인

실행:

```powershell
.\scripts\team\check_env.ps1
```

기준 PC 정상 출력 예:

```text
Python 3.12.10
Torch: 2.12.1+cu130
CUDA: True
GPU: NVIDIA GeForce RTX 3050
Labels: ('task', 'audience', 'context', 'format', 'tone', 'length', 'constraint', 'example')
num_labels: 8
```

🔴 반드시 확인할 항목:

- Python 버전
- Torch 버전
- CUDA `True`
- GPU 모델명
- Labels 8개
- `num_labels: 8`

CUDA가 `False`라면 학습을 시작하지 말고 오류 내용을 공유해주세요.

---

# 7. 🔴 V6 Dataset Load Check

프로젝트 루트에서 실행:

```powershell
python .\scripts\v6\04_validate\check_v6_kcelectra_dataset_load.py
```

정상 기준:

```text
MODEL      : beomi/KcELECTRA-base
NUM_LABELS : 8

train rows  : 2000
valid rows  : 400

input shape : (256,)
label shape : (8,)

V6 KcELECTRA DATASET LOAD CHECK: PASS
```

🟢 마지막에 아래 문구가 나오면 PC 기본 준비 완료입니다.

```text
V6 KcELECTRA DATASET LOAD CHECK: PASS
```

---

# 8. 🔴 팀장에게 보내야 하는 자료

세팅 완료 후 아래 형식으로 보내주세요.

```text
[Promptune V6 PC 준비 결과]

이름:
PC 번호:

Git Commit:
Python:
Torch:
CUDA: True / False
GPU:

Train rows: 2000 / 실패
Valid rows: 400 / 실패
Dataset Load Check: PASS / FAIL

특이사항:
```

## 필수 제출

🔴 **Git Commit**

```powershell
git rev-parse --short HEAD
```

🔴 **Python 버전**

```powershell
python --version
```

🔴 **Torch / CUDA / GPU**

```powershell
.\scripts\team\check_env.ps1
```

🔴 **Dataset Load 결과**

```text
Train rows: 2000
Valid rows: 400
Dataset Load Check: PASS
```

---

# 9. 🟡 문제가 있을 때만 추가로 보내주세요

정상 PC는 아래 자료를 굳이 제출하지 않아도 됩니다.

## GPU/Driver 정보

```powershell
nvidia-smi
```

## 전체 패키지 목록

```powershell
python -m pip freeze > pc_environment.txt
```

## 전체 오류 로그

오류가 난 경우에는 **마지막 한 줄만 보내지 말고**:

1. 실행한 명령어
2. 처음부터 끝까지 전체 오류 메시지

를 같이 보내주세요.

---

# 10. ❌ 보내지 말아야 하는 것

다음 파일이나 값은 팀장에게 전달할 필요가 없습니다.

```text
.env
Hugging Face Token
API Key
.venv 폴더
개인 인증정보
```

🔴 `.env`와 실제 Token/API Key는 채팅이나 GitHub에 공유하지 마세요.

---

# 11. 🟡 SHA256 확인

사전 PC 준비 단계에서는 필수가 아닙니다.

다만 **공식 V6 학습 시작 직전**에는 모든 PC의 데이터가 완전히 동일한지 확인하기 위해 Train/Valid SHA256 검사를 진행할 수 있습니다.

현재 Frozen 데이터:

```text
data/v6/labeling/frozen/Promptune_v6_train_2000_frozen_v1.csv
data/v6/labeling/valid_human_review/frozen/Promptune_v6_valid_400_frozen_v1.csv
```

실제 모델 학습 입력:

```text
data/v6/training_ready/Promptune_v6_train_2000_training_v1.csv
data/v6/training_ready/Promptune_v6_valid_400_training_v1.csv
```

---

# 12. 🔴 Benchmark 210 주의

다음 파일은 최종 평가용 Frozen Benchmark입니다.

```text
data/final-holdout/final-benchmark-v1/Promptune_final_benchmark_210_v1.csv
```

🔴 **현재 단계에서 Benchmark 210으로 Seed를 비교하거나 Threshold를 조정하지 않습니다.**

공식 흐름:

```text
Train 2000
    ↓
Seed별 학습
    ↓
Valid 400
    ↓
Threshold Tuning
    ↓
최종 Seed / Threshold 확정
    ↓
Frozen Benchmark 210 최종 1회 평가
```

---

# 13. 🟢 PC 준비 완료 조건

아래가 모두 충족되면 완료입니다.

- [ ] GitHub Private Repository 초대 수락
- [ ] Repository clone 완료
- [ ] 팀 공통 Git Commit 확인
- [ ] `.venv` 생성 및 활성화
- [ ] Python 환경 확인
- [ ] Torch 설치 완료
- [ ] CUDA `True`
- [ ] GPU 정상 인식
- [ ] Train `2000`
- [ ] Valid `400`
- [ ] Input shape `(256,)`
- [ ] Label shape `(8,)`
- [ ] Dataset Load Check `PASS`
- [ ] 결과를 팀장에게 공유
- [ ] 아직 모델 학습은 시작하지 않음

---

# 14. 이후 진행

전체 PC 준비 확인이 완료되면 다음 내용을 별도로 공지합니다.

```text
1. 공식 Git Commit
2. Epoch
3. Batch Size
4. Learning Rate
5. Threshold Tuner
6. PC별 Seed 배정
7. 공식 학습 명령어
8. 학습 후 제출해야 할 결과 파일
```

그때부터 공식 V6 Multi-PC 모델 학습을 시작합니다.

---

**Promptune V6 — PC Ready Check**
