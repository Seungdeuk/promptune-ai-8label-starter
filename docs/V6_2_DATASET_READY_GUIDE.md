# Promptune V6.2 Dataset Ready

## Frozen dataset
- Train: `Promptune_v6_2_train_2060_training_v1.csv`
- Rows: 2060 = V6 2000 + V6.2 60
- SHA256: `2B76462D40B3B6B7C82F196FE511AF08C947B11364BCB3470E98E8C0CBB95849`
- Valid: existing V6 Valid400 unchanged
- Final Benchmark210: untouched

## Final contrastive augmentation
- Tone: 10 pairs / 20 rows
- Length: 10 pairs / 20 rows
- Example: 10 pairs / 20 rows
- Total: 30 pairs / 60 rows

## Positive-count delta
{"task": 0, "audience": 0, "context": 0, "format": 0, "tone": 10, "length": 10, "constraint": 0, "example": 10}

## Suggested repo placement
```text
data/v6_2/review/Promptune_V6_2_Final_Selection_Audit_v1.xlsx
data/v6_2/training_ready/Promptune_v6_2_train_2060_training_v1.csv
data/v6_2/training_ready/Promptune_v6_2_train_2060_training_v1.sha256
data/v6_2/training_ready/Promptune_v6_2_train_2060_manifest_v1.json
scripts/v6_2/01_validate/check_v6_2_dataset_load.py
```

## First validation
```powershell
python .\scripts\v6_2\01_validate\check_v6_2_dataset_load.py
```

Do not run Seed52/62 or Final Benchmark210 yet. After dataset-load PASS, freeze Git and prepare a V6.2 Seed42 runner from the current hardened V6 official runner.
