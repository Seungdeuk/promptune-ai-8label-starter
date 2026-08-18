from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd
from datetime import datetime

ROOT = Path(r"data\v6\labeling\valid_human_review")

SOURCE = (
    ROOT / "merged"
    / "Promptune_v6_valid_400_human_final_candidate_v1.csv"
)

FREEZE_DIR = ROOT / "frozen"
FREEZE_DIR.mkdir(parents=True, exist_ok=True)

FROZEN = FREEZE_DIR / "Promptune_v6_valid_400_frozen_v1.csv"
SHA_OUT = FREEZE_DIR / "Promptune_v6_valid_400_frozen_v1.sha256"
MANIFEST_OUT = FREEZE_DIR / "Promptune_v6_valid_400_frozen_v1_manifest.json"

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

EXPECTED_ROWS = 400


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


print("=== V6 VALID 400 FREEZE ===")

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

df = pd.read_csv(SOURCE, encoding="utf-8-sig")

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS} rows, got {len(df)}"
    )

if df["id"].nunique() != EXPECTED_ROWS:
    raise RuntimeError("Valid IDs are not unique")

distribution = {}

for col in LABEL_COLS:
    if col not in df.columns:
        raise RuntimeError(f"Missing label column: {col}")

    values = pd.to_numeric(df[col], errors="coerce")

    if values.isna().any():
        raise RuntimeError(f"{col}: blank/non-numeric labels found")

    if not values.isin([0, 1]).all():
        raise RuntimeError(f"{col}: labels other than 0/1 found")

    ones = int(values.sum())

    distribution[col] = {
        "missing_1": ones,
        "sufficient_0": EXPECTED_ROWS - ones,
        "positive_rate": round(ones / EXPECTED_ROWS, 4),
    }

# Immutable snapshot
shutil.copy2(SOURCE, FROZEN)

digest = sha256_file(FROZEN)

SHA_OUT.write_text(
    f"{digest}  {FROZEN.name}\n",
    encoding="utf-8",
)

manifest = {
    "dataset_name": "Promptune V6 Valid 400",
    "freeze_version": "v1",
    "source_file": str(SOURCE),
    "frozen_file": str(FROZEN),
    "rows": EXPECTED_ROWS,
    "unique_ids": EXPECTED_ROWS,
    "sha256": digest,
    "label_semantics": {
        "1": "supplementation needed for actual request execution",
        "0": "sufficient or unnecessary for this request",
    },
    "review_provenance": (
        "All 400 validation prompts were blind human-reviewed "
        "for all 8 labels before freeze."
    ),
    "known_data_quality_notes": {
        "weak3_task": "14/14 designed Task contrast pairs survived final GT.",
        "weak3_format": (
            "0/13 designed Format contrast pairs survived final GT; "
            "treat as generation-design limitation, not forced GT correction."
        ),
        "weak3_constraint": (
            "0/13 designed Constraint contrast pairs survived final GT; "
            "treat as generation-design limitation, not forced GT correction."
        ),
        "sparse_positive_labels": {
            "audience": 7,
            "format": 5,
            "tone": 7,
            "length": 3,
            "constraint": 8,
            "example": 10,
        },
        "threshold_warning": (
            "Per-label threshold tuning on very sparse positive labels "
            "may be statistically unstable. Do not alter GT to balance."
        ),
    },
    "distribution": distribution,
    "freeze_note": (
        "Do not edit this frozen CSV in place. "
        "Any correction must create a new version."
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
print("=== FROZEN VALID DISTRIBUTION ===")

for col, d in distribution.items():
    print(
        f"{col:28s}: "
        f"{d['missing_1']:3d}/400 "
        f"({d['positive_rate']:.2%})"
    )

print()
print("FROZEN  :", FROZEN)
print("SHA256  :", SHA_OUT)
print("MANIFEST:", MANIFEST_OUT)
print()
print("SHA256:", digest)
print()
print("V6 VALID 400 FREEZE: PASS")
