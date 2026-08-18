from pathlib import Path
import pandas as pd

ROOT = Path(r"data\v6\labeling\targeted_review")

BASE = ROOT / "Promptune_v6_train_2000_labeled_calibrated_v1.csv"
REVIEW = ROOT / "Promptune_v6_train_targeted_review_queue_v3_gpt_reviewed.csv"

OUT = ROOT / "Promptune_v6_train_2000_labeled_after_targeted_v2.csv"
AUDIT = ROOT / "Promptune_v6_train_targeted_apply_audit_v2.csv"
CONTEXT_QUEUE = ROOT / "Promptune_v6_train_context_remaining_queue_v1.csv"
CONTEXT_PRIVATE = ROOT / "Promptune_v6_train_context_remaining_reference_PRIVATE_v1.csv"

LABEL_MAP = {
    "review_task": "human_missing_task",
    "review_audience": "human_missing_audience",
    "review_format": "human_missing_format",
    "review_context": "human_missing_context",
}

CHECK_MAP = {
    "review_task": "check_task",
    "review_audience": "check_audience",
    "review_format": "check_format",
    "review_context": "check_context",
}

def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")

print("=== APPLY TARGETED REVIEW TO TRAIN 2000 ===")

if not BASE.exists():
    raise FileNotFoundError(BASE)

if not REVIEW.exists():
    raise FileNotFoundError(REVIEW)

base = read_csv(BASE)
review = read_csv(REVIEW)

if len(base) != 2000 or base["id"].nunique() != 2000:
    raise RuntimeError("BASE must contain 2000 unique IDs")

if len(review) != 349 or review["id"].nunique() != 349:
    raise RuntimeError("REVIEW must contain 349 unique IDs")

# Validate IDs
unknown = set(review["id"]) - set(base["id"])
if unknown:
    raise RuntimeError(f"Unknown review IDs: {len(unknown)}")

# Validate reviewed status
reviewed = (
    review["review_status"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("reviewed")
)

if int(reviewed.sum()) != len(review):
    raise RuntimeError(
        f"Not all review rows are reviewed: {int(reviewed.sum())}/{len(review)}"
    )

# Apply only checked fields
out = base.copy().set_index("id")
review_idx = review.set_index("id")

audit_rows = []

for review_col, gt_col in LABEL_MAP.items():
    check_col = CHECK_MAP[review_col]

    checked_ids = review.loc[
        pd.to_numeric(review[check_col], errors="coerce").fillna(0).astype(int) == 1,
        "id"
    ].tolist()

    changed = 0
    zero_to_one = 0
    one_to_zero = 0

    for row_id in checked_ids:
        value = pd.to_numeric(
            pd.Series([review_idx.loc[row_id, review_col]]),
            errors="coerce"
        ).iloc[0]

        if pd.isna(value) or int(value) not in (0, 1):
            raise RuntimeError(
                f"{row_id}: invalid {review_col}={review_idx.loc[row_id, review_col]}"
            )

        old = int(out.loc[row_id, gt_col])
        new = int(value)

        if old != new:
            changed += 1
            if old == 0 and new == 1:
                zero_to_one += 1
            elif old == 1 and new == 0:
                one_to_zero += 1

        out.loc[row_id, gt_col] = new

    audit_rows.append({
        "label": gt_col,
        "checked_rows": len(checked_ids),
        "changed_rows": changed,
        "old_0_to_new_1": zero_to_one,
        "old_1_to_new_0": one_to_zero,
    })

out = out.reset_index()
out.to_csv(OUT, index=False, encoding="utf-8-sig")

audit = pd.DataFrame(audit_rows)
audit.to_csv(AUDIT, index=False, encoding="utf-8-sig")

print()
print("=== TARGETED REVIEW APPLIED ===")
print(audit.to_string(index=False))

print()
print("=== LABEL DISTRIBUTION AFTER TARGETED REVIEW ===")

label_cols = [
    "human_missing_task",
    "human_missing_audience",
    "human_missing_context",
    "human_missing_format",
    "human_missing_tone",
    "human_missing_length",
    "human_missing_constraint",
    "human_missing_example",
]

for col in label_cols:
    values = pd.to_numeric(out[col], errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise RuntimeError(f"{col}: invalid binary labels")

    ones = int(values.sum())
    print(f"{col:28s}: {ones:4d}/2000 ({ones/2000:.2%})")

# ---------------------------------------------------------
# Build remaining Context review queue
# ---------------------------------------------------------
# Already independently/second-pass reviewed Context IDs:
# 1) Human Calibration 160 is already reflected in BASE.
#    We identify those via calibration file if present.
# 2) Context IDs from the 349-row targeted review where check_context=1.
# Everything else with current Context=0 still needs review.
# ---------------------------------------------------------

calib_dir = Path(r"data\v6\labeling\calibration")
calib_candidates = [
    calib_dir / "Promptune_v6_train_calibration_human_160.csv",
    calib_dir / "Promptune_v6_train_calibration_blind_160.csv",
]

calib_ids = set()

for p in calib_candidates:
    if not p.exists():
        continue
    try:
        c = read_csv(p)
    except Exception:
        continue

    if len(c) == 160 and "calibration_status" in c.columns:
        reviewed_count = (
            c["calibration_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("reviewed")
            .sum()
        )
        if reviewed_count == 160:
            calib_ids = set(c["id"])
            break

context_targeted_ids = set(
    review.loc[
        pd.to_numeric(review["check_context"], errors="coerce").fillna(0).astype(int) == 1,
        "id"
    ]
)

already_context_reviewed = calib_ids | context_targeted_ids

remaining = out[
    (pd.to_numeric(out["human_missing_context"], errors="coerce").astype(int) == 0)
    & (~out["id"].isin(already_context_reviewed))
].copy()

# Blind queue: only context judgment needed.
blind = remaining[["id", "prompt_text"]].copy()
blind["review_context"] = ""
blind["review_note"] = ""
blind["review_status"] = "unreviewed"

blind.to_csv(
    CONTEXT_QUEUE,
    index=False,
    encoding="utf-8-sig"
)

# Private reference preserves current labels/metadata for later audit.
private_cols = ["id", "prompt_text", "human_missing_context"]

for extra in [
    "data_type",
    "input_style",
    "generation_strategy",
    "business_domain",
    "scenario_family",
    "focus_element",
    "construction_target",
    "pair_id",
    "pair_role",
]:
    if extra in remaining.columns:
        private_cols.append(extra)

remaining[private_cols].to_csv(
    CONTEXT_PRIVATE,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=== REMAINING CONTEXT REVIEW QUEUE ===")
print("Human Calibration Context-reviewed IDs :", len(calib_ids))
print("Targeted Context-reviewed IDs          :", len(context_targeted_ids))
print("Remaining Context=0 rows to review     :", len(blind))

if "data_type" in remaining.columns:
    print()
    print("=== REMAINING CONTEXT=0 BY DATA TYPE ===")
    print(remaining["data_type"].value_counts().to_string())

print()
print("OUTPUT :", OUT)
print("AUDIT  :", AUDIT)
print("QUEUE  :", CONTEXT_QUEUE)
print("PRIVATE:", CONTEXT_PRIVATE)
print()
print("V6 TRAIN TARGETED APPLY + CONTEXT QUEUE BUILD: PASS")
