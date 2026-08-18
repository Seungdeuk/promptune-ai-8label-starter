[Promptune V6 공식 실험 기준]



Git Commit: 7219e2f



학습 시작 전 반드시:

git pull

git rev-parse --short HEAD



결과가 7219e2f인지 확인해주세요.

# Promptune AI Service — 8-label clean starter

This project intentionally excludes the previous 5-label experiment code, checkpoints, and datasets.

## Official 8 elements

1. Task — 무엇을 요청하는지
2. Audience — 대상
3. Context — 배경·상황정보
4. Format — 출력 형식
5. Tone — 어조·말투
6. Length — 글자 수·분량
7. Constraint — 제약 조건
8. Example — 예시·참고 형식

Label convention:

- `1` = 보완 필요(누락)
- `0` = 충분함 또는 해당 요청에서 추가 필요 없음

## Why this structure

The service can start before the team finishes Ground Truth labeling. Both the normalizer and diagnosis model default to `mock` providers. Later, replace only the provider implementation rather than rewriting the API.

## 1. Create environment on Windows PowerShell

```powershell
cd promptune-ai-8label-starter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

For the first API-only run, install only the lightweight service dependencies:

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
```

You do **not** need PyTorch, Transformers, a model checkpoint, or HyperCLOVA X for the first run.

## 2. First run — no external AI required

Keep these values in `.env`:

```env
NORMALIZER_PROVIDER=mock
DIAGNOSIS_PROVIDER=mock
```

Run:

```powershell
.\scripts\run_dev.ps1
```

Check:

- `GET http://localhost:9000/health`
- Swagger: `http://localhost:9000/docs`
- `POST http://localhost:9000/ai/analyze`

Example request:

```json
{
  "requestId": "req-001",
  "text": "신제품 홍보 문구 3개 만들어줘"
}
```

The response always contains the eight diagnosis keys in the official order. In mock mode the values are only for API/UI development, never for model evaluation.

## 3. Dataset after team calibration

Do not train yet from the 30-sentence team calibration file. First compare the five annotators, resolve disagreements, and create Ground Truth.

CSV contract:

```text
id
text
missing_task
missing_audience
missing_context
missing_format
missing_tone
missing_length
missing_constraint
missing_example
```

Template: `data/dataset_template.csv`

Then prepare separate train/validation/test CSV files.

## 4. Train KcELECTRA 8-label classifier

At this stage, install the PyTorch build that matches your actual NVIDIA/CUDA environment **first**, then install the ML dependencies:

```powershell
pip install -r requirements-ml.txt
```

The starter intentionally does not hard-code a KcELECTRA model ID. Pass the exact base model or local path you decide to use:

```powershell
python -m ml.train `
  --model <KCELECTRA_BASE_MODEL_OR_PATH> `
  --train data/train.csv `
  --valid data/valid.csv `
  --output artifacts/kcelectra-8label-v1 `
  --seed 42
```

After training, configure:

```env
DIAGNOSIS_PROVIDER=kcelectra
KC_MODEL_PATH=artifacts/kcelectra-8label-v1
```

## 5. Thresholds

The service defaults to `0.5` for all labels. After validation predictions are available, optimize label-specific thresholds rather than assuming one threshold is best for every element.

`tune_thresholds.py` expects two NumPy arrays with shape `[N, 8]`: probabilities and Ground Truth labels.

```powershell
python -m ml.tune_thresholds `
  --probabilities artifacts/valid_probabilities.npy `
  --labels artifacts/valid_labels.npy `
  --output artifacts/thresholds.json
```

Then:

```env
THRESHOLD_FILE=artifacts/thresholds.json
```

## 6. Evaluate

```powershell
python -m ml.evaluate `
  --model artifacts/kcelectra-8label-v1 `
  --test data/test.csv `
  --thresholds artifacts/thresholds.json
```

The report contains label-level Precision/Recall/F1, Macro F1, Micro F1, Exact Match, average inference latency, and device.

## 7. HyperCLOVA X

`app/services/normalizer/clova.py` is deliberately an adapter boundary, not a guessed API implementation. Keep `NORMALIZER_PROVIDER=mock` until the exact current HyperCLOVA X API endpoint, authentication, request body, and response schema are confirmed. Once confirmed, only that file should need provider-specific code.

The pipeline always runs `validate_preservation()` after normalization. If protected numbers/dates/times/money/URLs/emails change, the service falls back to the original text before diagnosis.

## Recommended next order

1. Run this starter in mock mode.
2. Confirm `/ai/analyze` JSON with the Spring Boot owner.
3. Finish 30-sentence team annotation.
4. Resolve disagreements and freeze the 8-element labeling guide.
5. Build larger train/validation/test data.
6. Train/evaluate KcELECTRA 8-label model.
7. Connect the real model to this API.
8. Confirm the current HyperCLOVA X API contract and implement only the normalizer adapter.
9. Then add recommendation generation and Ghost Text.


## Synthetic smoke test before team Ground Truth

Before the real 5-person annotation is finalized, you can verify the entire 8-label training pipeline with deterministic synthetic data. This is **not Ground Truth** and its F1 must not be reported as model quality.

Generate and validate 240 rows (144 train / 48 valid / 48 test):

```powershell
python .\scripts\prepare_synthetic_smoke.py
python .\scripts\check_synthetic_data.py
python -m unittest tests.test_synthetic_dataset_contract -v
```

Then run a 1-epoch smoke training after installing the GPU-compatible PyTorch build and `requirements-ml.txt`:

```powershell
.\scripts\run_synthetic_smoke.ps1
```

The default base model is `beomi/KcELECTRA-base`. You can override it:

```powershell
.\scripts\run_synthetic_smoke.ps1 -Model "<MODEL_OR_LOCAL_PATH>" -Epochs 1 -BatchSize 4
```

Outputs:

- `artifacts/kcelectra-8label-synthetic-smoke/`
- `artifacts/synthetic-smoke/evaluation.json`

Synthetic generation keeps all variants from a scenario in the same split, preventing obvious scenario leakage between train/validation/test.
