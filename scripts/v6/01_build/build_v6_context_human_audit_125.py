from pathlib import Path
import pandas as pd

ROOT = Path(r"data\v6\labeling")
BASE = ROOT / "targeted_review" / "Promptune_v6_train_2000_labeled_calibrated_v1.csv"
CALIB_DIR = ROOT / "calibration"

OUT_DIR = ROOT / "context_human_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLIND_OUT = OUT_DIR / "Promptune_v6_train_context_human_audit_blind_125.csv"
PRIVATE_OUT = OUT_DIR / "Promptune_v6_train_context_human_audit_reference_PRIVATE_125.csv"
AUDIT_OUT = OUT_DIR / "Promptune_v6_train_context_human_audit_sampling_audit_125.csv"

SEED = 20260814

DATA_TYPES = [
    "general_business",
    "community_messenger",
    "weak3_focus",
    "hard_negative",
    "complex_boundary",
]

PER_TYPE = 25
TARGET_ROWS = PER_TYPE * len(DATA_TYPES)


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def find_completed_calibration():
    candidates = [
        CALIB_DIR / "Promptune_v6_train_calibration_human_160.csv",
        CALIB_DIR / "Promptune_v6_train_calibration_blind_160.csv",
    ]

    candidates.extend(sorted(CALIB_DIR.glob("*calibration*160*.csv")))

    seen = set()

    for path in candidates:
        if not path.exists():
            continue

        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)

        try:
            df = read_csv(path)
        except Exception:
            continue

        if len(df) != 160:
            continue

        if "id" not in df.columns or "calibration_status" not in df.columns:
            continue

        reviewed = (
            df["calibration_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("reviewed")
            .sum()
        )

        if reviewed == 160 and df["id"].nunique() == 160:
            return path, df

    raise FileNotFoundError(
        "완료된 Human Calibration 160 파일을 찾지 못했습니다.\n"
        "data\\v6\\labeling\\calibration 폴더를 확인하세요."
    )


print("=== V6 CONTEXT BLIND HUMAN AUDIT 125 BUILD ===")

if not BASE.exists():
    raise FileNotFoundError(
        f"기준 파일을 찾지 못했습니다:\n{BASE}"
    )

base = read_csv(BASE)

if len(base) != 2000:
    raise RuntimeError(f"BASE rows must be 2000, got {len(base)}")

if base["id"].nunique() != 2000:
    raise RuntimeError("BASE IDs must be unique")

required_cols = {
    "id",
    "prompt_text",
    "data_type",
    "human_missing_context",
}

missing = required_cols - set(base.columns)
if missing:
    raise RuntimeError(f"BASE missing columns: {sorted(missing)}")

ctx = pd.to_numeric(
    base["human_missing_context"],
    errors="coerce"
)

if ctx.isna().any() or not ctx.isin([0, 1]).all():
    raise RuntimeError("BASE human_missing_context must be complete binary 0/1")

base["human_missing_context"] = ctx.astype(int)

# ------------------------------------------------------------
# 1. Exclude the 160 prompts already reviewed by the user
# ------------------------------------------------------------
calib_path, calib = find_completed_calibration()
calib_ids = set(calib["id"])

pool = base[~base["id"].isin(calib_ids)].copy()

print("BASE rows               :", len(base))
print("Prior Human Calibration :", len(calib_ids))
print("Eligible new pool       :", len(pool))

if len(pool) != 1840:
    print(
        "WARNING: eligible pool is not exactly 1840. "
        "This is allowed if the completed calibration set changed."
    )

# ------------------------------------------------------------
# 2. Validate data type counts
# ------------------------------------------------------------
print()
print("=== ELIGIBLE POOL BY DATA TYPE ===")

for dtype in DATA_TYPES:
    n = int((pool["data_type"] == dtype).sum())
    print(f"{dtype:22s}: {n:4d}")

    if n < PER_TYPE:
        raise RuntimeError(
            f"{dtype}: only {n} eligible rows; need at least {PER_TYPE}"
        )

unexpected_types = sorted(
    set(pool["data_type"].dropna().astype(str)) - set(DATA_TYPES)
)

if unexpected_types:
    raise RuntimeError(
        f"Unexpected data_type values: {unexpected_types}"
    )

# ------------------------------------------------------------
# 3. Sample exactly 25 per data_type
# ------------------------------------------------------------
samples = []

for i, dtype in enumerate(DATA_TYPES):
    group = pool[pool["data_type"] == dtype].copy()

    sampled = group.sample(
        n=PER_TYPE,
        random_state=SEED + i,
        replace=False,
    )

    samples.append(sampled)

sample = pd.concat(samples, ignore_index=True)

# Shuffle final order so the reviewer cannot infer data type blocks.
sample = sample.sample(
    frac=1,
    random_state=SEED + 100
).reset_index(drop=True)

if len(sample) != TARGET_ROWS:
    raise RuntimeError(
        f"Expected {TARGET_ROWS} rows, got {len(sample)}"
    )

if sample["id"].nunique() != TARGET_ROWS:
    raise RuntimeError("Sample contains duplicate IDs")

if set(sample["id"]) & calib_ids:
    raise RuntimeError("Sample overlaps prior Human Calibration 160")

# ------------------------------------------------------------
# 4. Create BLIND file
#    Do NOT expose existing labels or design metadata.
# ------------------------------------------------------------
blind = sample[["id", "prompt_text"]].copy()

blind["review_context"] = ""
blind["review_note"] = ""
blind["review_status"] = "unreviewed"

blind.to_csv(
    BLIND_OUT,
    index=False,
    encoding="utf-8-sig",
)

# ------------------------------------------------------------
# 5. Create PRIVATE reference for post-review comparison
# ------------------------------------------------------------
private_cols = [
    "id",
    "prompt_text",
    "data_type",
    "human_missing_context",
]

for optional in [
    "input_style",
    "generation_strategy",
    "business_domain",
    "scenario_family",
    "focus_element",
    "construction_target",
    "pair_id",
    "pair_role",
]:
    if optional in sample.columns:
        private_cols.append(optional)

private = sample[private_cols].copy()

private.to_csv(
    PRIVATE_OUT,
    index=False,
    encoding="utf-8-sig",
)

# ------------------------------------------------------------
# 6. Sampling audit
# ------------------------------------------------------------
audit = (
    sample.groupby("data_type")
    .agg(
        sampled_rows=("id", "count"),
        existing_context_1=("human_missing_context", "sum"),
    )
    .reset_index()
)

audit["existing_context_0"] = (
    audit["sampled_rows"] - audit["existing_context_1"]
)

audit["existing_context_positive_rate"] = (
    audit["existing_context_1"] / audit["sampled_rows"]
).round(4)

audit.to_csv(
    AUDIT_OUT,
    index=False,
    encoding="utf-8-sig",
)

print()
print("=== SAMPLE CHECK ===")
print("rows                 :", len(sample))
print("unique IDs           :", sample["id"].nunique())
print("overlap with Human160:", len(set(sample["id"]) & calib_ids))

print()
print("=== SAMPLE COUNTS BY DATA TYPE ===")
print(
    sample["data_type"]
    .value_counts()
    .reindex(DATA_TYPES)
    .to_string()
)

print()
print("BLIND  :", BLIND_OUT)
print("PRIVATE:", PRIVATE_OUT)
print("AUDIT  :", AUDIT_OUT)
print()
print("IMPORTANT: Human reviewer should open ONLY the BLIND file.")
print()
print("V6 CONTEXT BLIND HUMAN AUDIT 125 BUILD: PASS")
