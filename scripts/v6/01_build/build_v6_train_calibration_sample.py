from pathlib import Path
import random
import pandas as pd

ROOT = Path(r"data\v6")
MERGED = ROOT / "labeling" / "merged" / "Promptune_v6_train_2000_labeled_primary_v1.csv"
OUT_DIR = ROOT / "labeling" / "calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLIND_OUT = OUT_DIR / "Promptune_v6_train_calibration_blind_160.csv"
PRIVATE_OUT = OUT_DIR / "Promptune_v6_train_calibration_reference_PRIVATE.csv"

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

SEED = 20260814
TARGET_ROWS = 160
POS_TARGET_PER_LABEL = 15
ALL_ZERO_TARGET = 40

def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")

print("=== V6 TRAIN HUMAN CALIBRATION SAMPLE BUILD ===")

if not MERGED.exists():
    raise FileNotFoundError(MERGED)

df = read_csv(MERGED)

if len(df) != 2000:
    raise RuntimeError(f"Expected 2000 rows, got {len(df)}")

for col in LABEL_COLS:
    df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    if not df[col].isin([0, 1]).all():
        raise RuntimeError(f"{col} contains non-binary values")

rng = random.Random(SEED)
selected = set()

# 1) Guarantee at least 15 Claude-positive examples per label.
for col in LABEL_COLS:
    positive_ids = df.loc[df[col] == 1, "id"].tolist()
    rng.shuffle(positive_ids)

    already = sum(
        1 for row_id in selected
        if int(df.loc[df["id"] == row_id, col].iloc[0]) == 1
    )

    need = max(0, POS_TARGET_PER_LABEL - already)

    candidates = [row_id for row_id in positive_ids if row_id not in selected]

    for row_id in candidates[:need]:
        selected.add(row_id)

# 2) Add all-zero cases for false-positive calibration.
df["_label_sum"] = df[LABEL_COLS].sum(axis=1)

zero_ids = df.loc[df["_label_sum"] == 0, "id"].tolist()
rng.shuffle(zero_ids)

for row_id in zero_ids:
    if len([x for x in selected if int(df.loc[df["id"] == x, "_label_sum"].iloc[0]) == 0]) >= ALL_ZERO_TARGET:
        break
    if row_id not in selected:
        selected.add(row_id)

# 3) Fill remaining rows with mixed/boundary cases.
remaining = df.loc[~df["id"].isin(selected)].copy()

# Prefer rows with 1~3 positive labels, then random.
mixed = remaining[
    remaining["_label_sum"].between(1, 3)
]["id"].tolist()
rng.shuffle(mixed)

for row_id in mixed:
    if len(selected) >= TARGET_ROWS:
        break
    selected.add(row_id)

# 4) If still short, fill randomly.
if len(selected) < TARGET_ROWS:
    rest = df.loc[~df["id"].isin(selected), "id"].tolist()
    rng.shuffle(rest)

    for row_id in rest:
        if len(selected) >= TARGET_ROWS:
            break
        selected.add(row_id)

if len(selected) != TARGET_ROWS:
    raise RuntimeError(f"Expected {TARGET_ROWS} selected rows, got {len(selected)}")

sample = df[df["id"].isin(selected)].copy()

# Shuffle final row order so the human reviewer does not see sampling groups.
sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Private reference keeps Claude labels.
private_cols = ["id", "prompt_text", *LABEL_COLS]
for extra in ["data_type", "input_style", "scenario_family"]:
    if extra in sample.columns:
        private_cols.append(extra)

sample[private_cols].to_csv(
    PRIVATE_OUT,
    index=False,
    encoding="utf-8-sig",
)

# Blind file hides Claude labels.
blind = sample[["id", "prompt_text"]].copy()

for col in CALIB_COLS:
    blind[col] = ""

blind["calibration_note"] = ""
blind["calibration_status"] = "unreviewed"

blind.to_csv(
    BLIND_OUT,
    index=False,
    encoding="utf-8-sig",
)

print("rows:", len(blind))
print()
print("=== CLAUDE LABEL COVERAGE INSIDE SAMPLE ===")

for col in LABEL_COLS:
    ones = int(sample[col].sum())
    print(f"{col:28s}: positive={ones:3d} negative={len(sample)-ones:3d}")

print()
print("all-zero rows:", int((sample["_label_sum"] == 0).sum()))
print()
print("BLIND  :", BLIND_OUT)
print("PRIVATE:", PRIVATE_OUT)
print()
print("V6 TRAIN CALIBRATION SAMPLE BUILD: PASS")
