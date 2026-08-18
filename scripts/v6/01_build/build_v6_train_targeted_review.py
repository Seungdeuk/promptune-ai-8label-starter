from pathlib import Path
from collections import defaultdict
import math
import random
import pandas as pd

ROOT = Path(r"data\v6")
MERGED = ROOT / "labeling" / "merged" / "Promptune_v6_train_2000_labeled_primary_v1.csv"
CALIB_DIR = ROOT / "labeling" / "calibration"

OUT_DIR = ROOT / "labeling" / "targeted_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATED_OUT = OUT_DIR / "Promptune_v6_train_2000_labeled_calibrated_v1.csv"
CALIBRATION_APPLY_AUDIT = OUT_DIR / "Promptune_v6_train_calibration_apply_audit_v1.csv"
QUEUE_OUT = OUT_DIR / "Promptune_v6_train_targeted_review_queue_v3.csv"
PRIVATE_OUT = OUT_DIR / "Promptune_v6_train_targeted_review_reference_PRIVATE_v3.csv"

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

CALIB_COLS = [
    "calib_missing_task",
    "calib_missing_audience",
    "calib_missing_context",
    "calib_missing_format",
    "calib_missing_tone",
    "calib_missing_length",
    "calib_missing_constraint",
    "calib_missing_example",
]

MAP_CALIB_TO_GT = dict(zip(CALIB_COLS, LABEL_COLS))

SEED = 20260814
CONTEXT_NEGATIVE_SAMPLE = 200


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_binary(series, name):
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise RuntimeError(f"{name}: blank/non-numeric values found")
    if not values.isin([0, 1]).all():
        raise RuntimeError(f"{name}: values other than 0/1 found")
    return values.astype(int)


def find_completed_calibration():
    candidates = [
        CALIB_DIR / "Promptune_v6_train_calibration_human_160.csv",
        CALIB_DIR / "Promptune_v6_train_calibration_blind_160.csv",
    ]

    # Also allow any file matching calibration*160.csv.
    candidates.extend(sorted(CALIB_DIR.glob("*calibration*160.csv")))

    seen = set()

    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)

        try:
            df = read_csv(path)
        except Exception:
            continue

        if len(df) != 160:
            continue

        if not all(c in df.columns for c in ["id", "prompt_text", *CALIB_COLS, "calibration_status"]):
            continue

        ok = True
        for col in CALIB_COLS:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.isna().any() or not vals.isin([0, 1]).all():
                ok = False
                break

        reviewed = (
            df["calibration_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("reviewed")
            .sum()
        )

        if ok and reviewed == 160:
            return path, df

    raise FileNotFoundError(
        "완료된 Human Calibration 160 파일을 찾지 못했습니다.\n"
        "다음 중 하나로 저장되어 있어야 합니다:\n"
        "  data\\v6\\labeling\\calibration\\Promptune_v6_train_calibration_human_160.csv\n"
        "  또는 완료된 Promptune_v6_train_calibration_blind_160.csv"
    )


def proportional_stratified_sample(df, n, group_col, seed):
    if len(df) < n:
        raise RuntimeError(f"Context negative pool only has {len(df)} rows; need {n}")

    if group_col not in df.columns:
        return df.sample(n=n, random_state=seed).copy()

    groups = []
    total = len(df)

    # First-pass proportional allocation.
    alloc = {}
    fractions = []

    for key, group in df.groupby(group_col, dropna=False):
        exact = n * len(group) / total
        base = min(len(group), math.floor(exact))
        alloc[key] = base
        fractions.append((exact - base, key))

    # Allocate leftovers by largest remainder.
    used = sum(alloc.values())
    left = n - used

    for _, key in sorted(fractions, reverse=True):
        if left <= 0:
            break
        group_size = len(df[df[group_col].fillna("__NA__") == (key if pd.notna(key) else "__NA__")])
        if alloc[key] < group_size:
            alloc[key] += 1
            left -= 1

    # If any leftovers remain due to tiny groups, fill from available capacity.
    if left > 0:
        for key, group in df.groupby(group_col, dropna=False):
            capacity = len(group) - alloc[key]
            take = min(capacity, left)
            alloc[key] += take
            left -= take
            if left == 0:
                break

    rng = random.Random(seed)

    for key, group in df.groupby(group_col, dropna=False):
        take = alloc.get(key, 0)
        if take <= 0:
            continue

        # Different deterministic seed per group.
        group_seed = rng.randint(0, 2**31 - 1)
        groups.append(group.sample(n=take, random_state=group_seed))

    out = pd.concat(groups, ignore_index=True)

    if len(out) != n:
        raise RuntimeError(f"Stratified sample expected {n}, got {len(out)}")

    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


print("=== V6 TRAIN CALIBRATION APPLY + TARGETED REVIEW BUILD ===")

if not MERGED.exists():
    raise FileNotFoundError(f"Merged Train not found: {MERGED}")

train = read_csv(MERGED)

if len(train) != 2000 or train["id"].nunique() != 2000:
    raise RuntimeError("Merged Train must contain 2000 unique IDs")

for col in LABEL_COLS:
    train[col] = normalize_binary(train[col], col)

calib_path, calib = find_completed_calibration()

print("Human calibration:", calib_path)
print("Calibration rows  :", len(calib))

if calib["id"].nunique() != 160:
    raise RuntimeError("Calibration IDs are not unique")

train_idx = train.set_index("id")
calib_ids = set(calib["id"])

missing_ids = calib_ids - set(train["id"])
if missing_ids:
    raise RuntimeError(f"Calibration contains unknown Train IDs: {len(missing_ids)}")

# Prompt integrity.
prompt_changes = 0
for _, row in calib.iterrows():
    old = str(train_idx.loc[row["id"], "prompt_text"]).strip()
    new = str(row["prompt_text"]).strip()
    if old != new:
        prompt_changes += 1

if prompt_changes:
    raise RuntimeError(f"Calibration prompt_text changed on {prompt_changes} rows")

# Normalize human calibration labels.
for col in CALIB_COLS:
    calib[col] = normalize_binary(calib[col], col)

# ------------------------------------------------------------
# 1. Apply the 160 human labels over Claude labels
# ------------------------------------------------------------
calibrated = train.copy()
calibrated_idx = calibrated.set_index("id")

change_rows = []

calib_idx = calib.set_index("id")

for calib_col, gt_col in MAP_CALIB_TO_GT.items():
    changed = 0
    c01 = 0  # Claude 0 -> Human 1
    c10 = 0  # Claude 1 -> Human 0

    for row_id in calib_idx.index:
        old = int(calibrated_idx.loc[row_id, gt_col])
        new = int(calib_idx.loc[row_id, calib_col])

        if old != new:
            changed += 1
            if old == 0 and new == 1:
                c01 += 1
            elif old == 1 and new == 0:
                c10 += 1

        calibrated_idx.loc[row_id, gt_col] = new

    change_rows.append({
        "label": gt_col,
        "changed_rows": changed,
        "claude_0_to_human_1": c01,
        "claude_1_to_human_0": c10,
    })

calibrated = calibrated_idx.reset_index()
calibrated.to_csv(CALIBRATED_OUT, index=False, encoding="utf-8-sig")

change_df = pd.DataFrame(change_rows)
change_df.to_csv(CALIBRATION_APPLY_AUDIT, index=False, encoding="utf-8-sig")

print()
print("=== HUMAN CALIBRATION APPLIED ===")
print(change_df.to_string(index=False))

print()
print("=== LABEL DISTRIBUTION AFTER HUMAN 160 ===")
for col in LABEL_COLS:
    ones = int(calibrated[col].sum())
    print(f"{col:28s}: {ones:4d}/2000 ({ones/2000:.2%})")

# ------------------------------------------------------------
# 2. Build targeted review pools OUTSIDE the already-reviewed 160
# ------------------------------------------------------------
outside = train[~train["id"].isin(calib_ids)].copy()

# Positive audits: original Claude positives only, excluding already human-reviewed rows.
task_ids = set(outside.loc[outside["human_missing_task"] == 1, "id"])
audience_ids = set(outside.loc[outside["human_missing_audience"] == 1, "id"])
format_ids = set(outside.loc[outside["human_missing_format"] == 1, "id"])

# Context false-negative audit: random/stratified sample from Claude Context=0,
# excluding the 160 calibration rows.
context_negative_pool = outside[outside["human_missing_context"] == 0].copy()

context_sample = proportional_stratified_sample(
    context_negative_pool,
    CONTEXT_NEGATIVE_SAMPLE,
    "data_type",
    SEED,
)

context_ids = set(context_sample["id"])

targets = defaultdict(set)

for row_id in task_ids:
    targets[row_id].add("task")
for row_id in audience_ids:
    targets[row_id].add("audience")
for row_id in format_ids:
    targets[row_id].add("format")
for row_id in context_ids:
    targets[row_id].add("context")

# ------------------------------------------------------------
# 3. De-duplicated blind queue
# ------------------------------------------------------------
source = train.set_index("id")

queue_rows = []

for row_id, elems in targets.items():
    row = source.loc[row_id]

    queue_rows.append({
        "id": row_id,
        "prompt_text": row["prompt_text"],
        "check_task": 1 if "task" in elems else 0,
        "check_audience": 1 if "audience" in elems else 0,
        "check_format": 1 if "format" in elems else 0,
        "check_context": 1 if "context" in elems else 0,
        "review_task": "",
        "review_audience": "",
        "review_format": "",
        "review_context": "",
        "review_note": "",
        "review_status": "unreviewed",
    })

queue = pd.DataFrame(queue_rows)

queue = queue.sample(frac=1, random_state=SEED).reset_index(drop=True)

queue.to_csv(QUEUE_OUT, index=False, encoding="utf-8-sig")

# Private reference: original Claude values + why selected.
private_rows = []

for _, q in queue.iterrows():
    row_id = q["id"]
    row = source.loc[row_id]

    private_rows.append({
        "id": row_id,
        "prompt_text": row["prompt_text"],
        "claude_task": int(row["human_missing_task"]),
        "claude_audience": int(row["human_missing_audience"]),
        "claude_format": int(row["human_missing_format"]),
        "claude_context": int(row["human_missing_context"]),
        "check_task": int(q["check_task"]),
        "check_audience": int(q["check_audience"]),
        "check_format": int(q["check_format"]),
        "check_context": int(q["check_context"]),
    })

pd.DataFrame(private_rows).to_csv(
    PRIVATE_OUT,
    index=False,
    encoding="utf-8-sig",
)

# ------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------
print()
print("=== TARGETED REVIEW COUNTS ===")
print("Task positives remaining     :", len(task_ids))
print("Audience positives remaining :", len(audience_ids))
print("Format positives remaining   :", len(format_ids))
print("Context=0 audit sample       :", len(context_ids))
print("De-duplicated queue rows     :", len(queue))

print()
print("=== QUEUE CHECK COUNTS ===")
print("check_task     :", int(queue["check_task"].sum()))
print("check_audience :", int(queue["check_audience"].sum()))
print("check_format   :", int(queue["check_format"].sum()))
print("check_context  :", int(queue["check_context"].sum()))

if "data_type" in context_sample.columns:
    print()
    print("=== CONTEXT NEGATIVE SAMPLE BY DATA TYPE ===")
    print(context_sample["data_type"].value_counts().to_string())

print()
print("CALIBRATED:", CALIBRATED_OUT)
print("QUEUE     :", QUEUE_OUT)
print("PRIVATE   :", PRIVATE_OUT)
print()
print("V6 TRAIN TARGETED REVIEW BUILD: PASS")
