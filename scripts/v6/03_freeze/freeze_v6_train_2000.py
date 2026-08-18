from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd
from datetime import datetime

ROOT = Path(r"data\v6\labeling")

SOURCE = (
    ROOT / "final_candidate"
    / "Promptune_v6_train_2000_final_candidate_v2.csv"
)

FREEZE_DIR = ROOT / "frozen"
FREEZE_DIR.mkdir(parents=True, exist_ok=True)

FROZEN = FREEZE_DIR / "Promptune_v6_train_2000_frozen_v1.csv"
SHA_OUT = FREEZE_DIR / "Promptune_v6_train_2000_frozen_v1.sha256"
MANIFEST_OUT = FREEZE_DIR / "Promptune_v6_train_2000_frozen_v1_manifest.json"

LABEL_COLS = [
    "human_missing_task",
    "human_missing_audience",
    "human_missing_context",
    "human_missing_format",
    "human_missing_tone",
    "human_missing_length",
    "human_missing_constraint",
    "human_missing_example",
]

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

print("=== V6 TRAIN FREEZE ===")

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

df = pd.read_csv(SOURCE, encoding="utf-8-sig")

# Structural validation
if len(df) != 2000:
    raise RuntimeError(f"Expected 2000 rows, got {len(df)}")

if df["id"].nunique() != 2000:
    raise RuntimeError("IDs are not unique")

for col in LABEL_COLS:
    if col not in df.columns:
        raise RuntimeError(f"Missing label column: {col}")

    v = pd.to_numeric(df[col], errors="coerce")

    if v.isna().any():
        raise RuntimeError(f"{col}: blanks/non-numeric values found")

    if not v.isin([0, 1]).all():
        raise RuntimeError(f"{col}: non-binary values found")

# Copy exact final candidate bytes so Freeze is immutable snapshot.
shutil.copy2(SOURCE, FROZEN)

digest = sha256_file(FROZEN)

SHA_OUT.write_text(
    f"{digest}  {FROZEN.name}\n",
    encoding="utf-8",
)

distribution = {}

for col in LABEL_COLS:
    ones = int(pd.to_numeric(df[col]).sum())
    distribution[col] = {
        "missing_1": ones,
        "sufficient_0": 2000 - ones,
        "positive_rate": round(ones / 2000, 4),
    }

manifest = {
    "dataset_name": "Promptune V6 Train 2000",
    "freeze_version": "v1",
    "source_file": str(SOURCE),
    "frozen_file": str(FROZEN),
    "rows": 2000,
    "unique_ids": 2000,
    "sha256": digest,
    "label_semantics": {
        "1": "supplementation needed for actual request execution",
        "0": "sufficient or unnecessary for this request",
    },
    "review_provenance": {
        "context": (
            "Human-reviewed for all 2000 rows using Human Calibration 160, "
            "independent Human Audit 125, and Full Human Review 1715."
        ),
        "task_audience_format": (
            "Primary labels with Human Calibration 160 applied; remaining "
            "positive rows were separately Human-reviewed before freeze."
        ),
        "tone_length_constraint_example": (
            "Primary labels retained after Human Calibration validation; "
            "not exhaustively human-reviewed across all 2000 rows."
        ),
    },
    "known_data_quality_notes": {
        "weak3_task": "69/70 designed contrast pairs survived in final GT.",
        "weak3_format": (
            "0/70 designed contrast pairs survived in final GT. "
            "Treat as generation-design failure, not a reason to force labels."
        ),
        "weak3_constraint": (
            "0/70 designed contrast pairs survived in final GT. "
            "Treat as generation-design failure, not a reason to force labels."
        ),
        "class_imbalance": (
            "Audience, Format, Tone, Length, Constraint, and Example positives "
            "remain sparse. Do not alter GT to balance classes."
        ),
    },
    "distribution": distribution,
    "freeze_note": (
        "Do not edit this frozen CSV in place. Any later augmentation or "
        "correction must produce a new dataset/version."
    ),
    "created_local_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

MANIFEST_OUT.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("rows       :", len(df))
print("unique IDs :", df["id"].nunique())

print()
print("=== FROZEN DISTRIBUTION ===")
for col, d in distribution.items():
    print(
        f"{col:28s}: "
        f"{d['missing_1']:4d}/2000 "
        f"({d['positive_rate']:.2%})"
    )

print()
print("FROZEN  :", FROZEN)
print("SHA256  :", SHA_OUT)
print("MANIFEST:", MANIFEST_OUT)
print()
print("SHA256:", digest)
print()
print("V6 TRAIN FREEZE: PASS")
