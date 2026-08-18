from pathlib import Path
import pandas as pd
import sys

ROOT = Path(r"data\v6")
MASTER = ROOT / "Promptune_v6_train_2000_unlabeled_master_v1.csv"
PACK_DIR = ROOT / "labeling" / "blind" / "train_primary"

OUT_DIR = ROOT / "labeling" / "merged"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_OUT = OUT_DIR / "Promptune_v6_train_2000_labeled_primary_v1.csv"
STRUCTURE_AUDIT_OUT = OUT_DIR / "Promptune_v6_train_primary_structure_audit_v1.csv"
LABEL_DISTRIBUTION_OUT = OUT_DIR / "Promptune_v6_train_primary_label_distribution_v1.csv"
REVIEW_QUEUE_OUT = OUT_DIR / "Promptune_v6_train_primary_review_queue_v1.csv"

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

EXPECTED_FILES = [f"v6_train_primary_pack_{i:02d}_labeled.csv" for i in range(1, 6)]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def s(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


print("=== V6 TRAIN PRIMARY LABEL VALIDATION ===")

if not MASTER.exists():
    raise FileNotFoundError(f"Train master not found: {MASTER}")

missing_files = [name for name in EXPECTED_FILES if not (PACK_DIR / name).exists()]
if missing_files:
    print("Missing files:")
    for name in missing_files:
        print("  ", name)
    raise FileNotFoundError("Train primary pack files are incomplete.")

files = [PACK_DIR / name for name in EXPECTED_FILES]

print("Pack files:", len(files))
for path in files:
    print(" ", path)

master = read_csv(MASTER)
if len(master) != 2000:
    raise RuntimeError(f"Master expected 2000 rows, got {len(master)}")
if master["id"].nunique() != 2000:
    raise RuntimeError("Master IDs are not unique.")

master_by_id = master.set_index("id")

frames = []
structure_rows = []

for pack_no, path in enumerate(files, start=1):
    df = read_csv(path)
    df["_pack"] = f"pack_{pack_no:02d}"

    rows = len(df)
    unique_ids = df["id"].nunique() if "id" in df.columns else 0

    missing_columns = [
        col for col in ["id", "prompt_text", *LABEL_COLS, "review_status"]
        if col not in df.columns
    ]

    blank_labels = 0
    invalid_labels = 0
    reviewed_count = 0
    prompt_changes = 0
    unknown_ids = 0

    if not missing_columns:
        for col in LABEL_COLS:
            vals = df[col].fillna("").astype(str).str.strip()
            blank_labels += int(vals.eq("").sum())
            invalid_labels += int((~vals.isin(["0", "1", "0.0", "1.0"])).sum())

        reviewed_count = int(
            df["review_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("reviewed")
            .sum()
        )

        for _, row in df.iterrows():
            row_id = row["id"]
            if row_id not in master_by_id.index:
                unknown_ids += 1
                continue

            expected_prompt = s(master_by_id.loc[row_id, "prompt_text"])
            actual_prompt = s(row["prompt_text"])
            if expected_prompt != actual_prompt:
                prompt_changes += 1

    pack_pass = (
        rows == 400
        and unique_ids == 400
        and len(missing_columns) == 0
        and blank_labels == 0
        and invalid_labels == 0
        and reviewed_count == 400
        and prompt_changes == 0
        and unknown_ids == 0
    )

    structure_rows.append({
        "pack": f"pack_{pack_no:02d}",
        "rows": rows,
        "unique_ids": unique_ids,
        "missing_columns": ",".join(missing_columns),
        "blank_label_cells": blank_labels,
        "invalid_label_cells": invalid_labels,
        "reviewed_rows": reviewed_count,
        "prompt_text_changes": prompt_changes,
        "unknown_ids": unknown_ids,
        "result": "PASS" if pack_pass else "FAIL",
    })

    frames.append(df)

structure_df = pd.DataFrame(structure_rows)

print()
print("=== PACK STRUCTURE CHECK ===")
print(structure_df.to_string(index=False))

combined = pd.concat(frames, ignore_index=True)

combined_rows = len(combined)
combined_unique_ids = combined["id"].nunique()
duplicate_ids = combined[combined["id"].duplicated(keep=False)]["id"].unique().tolist()

master_ids = set(master["id"])
combined_ids = set(combined["id"])

missing_ids = sorted(master_ids - combined_ids)
extra_ids = sorted(combined_ids - master_ids)

print()
print("=== COMBINED COVERAGE ===")
print("rows          :", combined_rows)
print("unique IDs    :", combined_unique_ids)
print("duplicate IDs :", len(duplicate_ids))
print("missing IDs   :", len(missing_ids))
print("extra IDs     :", len(extra_ids))

structure_ok = (
    (structure_df["result"] == "PASS").all()
    and combined_rows == 2000
    and combined_unique_ids == 2000
    and len(duplicate_ids) == 0
    and len(missing_ids) == 0
    and len(extra_ids) == 0
)

structure_df.to_csv(STRUCTURE_AUDIT_OUT, index=False, encoding="utf-8-sig")

if not structure_ok:
    print()
    print("V6 TRAIN PRIMARY STRUCTURE: FAIL")
    print("Audit:", STRUCTURE_AUDIT_OUT)
    sys.exit(1)

for col in LABEL_COLS:
    combined[col] = (
        combined[col]
        .astype(str)
        .str.strip()
        .replace({"0.0": "0", "1.0": "1"})
        .astype(int)
    )

distribution_rows = []

for pack in sorted(combined["_pack"].unique()):
    dfp = combined[combined["_pack"] == pack]
    for col in LABEL_COLS:
        ones = int(dfp[col].sum())
        zeros = int(len(dfp) - ones)
        rate = ones / len(dfp)

        distribution_rows.append({
            "scope": pack,
            "label": col,
            "rows": len(dfp),
            "missing_1": ones,
            "sufficient_0": zeros,
            "positive_rate": round(rate, 4),
        })

for col in LABEL_COLS:
    ones = int(combined[col].sum())
    zeros = int(len(combined) - ones)
    rate = ones / len(combined)

    distribution_rows.append({
        "scope": "ALL_2000",
        "label": col,
        "rows": len(combined),
        "missing_1": ones,
        "sufficient_0": zeros,
        "positive_rate": round(rate, 4),
    })

distribution_df = pd.DataFrame(distribution_rows)

print()
print("=== LABEL DISTRIBUTION : ALL 2000 ===")
print(
    distribution_df[
        distribution_df["scope"] == "ALL_2000"
    ][
        ["label", "missing_1", "sufficient_0", "positive_rate"]
    ].to_string(index=False)
)

review_reasons = {}

for _, row in combined.iterrows():
    row_id = row["id"]
    label_sum = int(sum(row[col] for col in LABEL_COLS))

    if label_sum == 0:
        review_reasons.setdefault(row_id, []).append("all_zero_labels")

    if label_sum >= 5:
        review_reasons.setdefault(row_id, []).append(f"many_missing_labels_{label_sum}")

pack_rates = (
    distribution_df[
        distribution_df["scope"].str.startswith("pack_")
    ]
    .pivot(index="scope", columns="label", values="positive_rate")
)

for col in LABEL_COLS:
    rates = pack_rates[col]
    median_rate = float(rates.median())

    for pack, rate in rates.items():
        if abs(float(rate) - median_rate) >= 0.20:
            ids = combined.loc[combined["_pack"] == pack, "id"].tolist()
            for row_id in ids:
                review_reasons.setdefault(row_id, []).append(
                    f"pack_rate_outlier:{pack}:{col}"
                )

review_rows = []
combined_idx = combined.set_index("id")

for row_id, reasons in review_reasons.items():
    row = combined_idx.loc[row_id]

    review_rows.append({
        "id": row_id,
        "pack": row["_pack"],
        "prompt_text": row["prompt_text"],
        **{col: int(row[col]) for col in LABEL_COLS},
        "review_reason": " | ".join(sorted(set(reasons))),
        "second_review_status": "",
        "second_review_note": "",
    })

review_queue_df = pd.DataFrame(review_rows)

if len(review_queue_df):
    review_queue_df = review_queue_df.sort_values(["pack", "id"])

label_source = combined.set_index("id")
merged = master.copy()

for col in LABEL_COLS:
    merged[col] = merged["id"].map(label_source[col]).astype(int)

if "review_note" in label_source.columns:
    merged["review_note"] = merged["id"].map(label_source["review_note"]).fillna("")
else:
    merged["review_note"] = ""

merged["review_status"] = "reviewed"

merged.to_csv(MERGED_OUT, index=False, encoding="utf-8-sig")
distribution_df.to_csv(LABEL_DISTRIBUTION_OUT, index=False, encoding="utf-8-sig")
review_queue_df.to_csv(REVIEW_QUEUE_OUT, index=False, encoding="utf-8-sig")

print()
print("=== FINAL CHECK ===")
print("all 5 packs structural PASS :", (structure_df["result"] == "PASS").all())
print("merged rows                 :", len(merged))
print("merged unique IDs           :", merged["id"].nunique())
print("label blanks                :", int(merged[LABEL_COLS].isna().sum().sum()))
print("review queue rows           :", len(review_queue_df))

print()
print("V6 TRAIN PRIMARY MERGE: PASS")
print("Merged :", MERGED_OUT)
print("Struct :", STRUCTURE_AUDIT_OUT)
print("Dist   :", LABEL_DISTRIBUTION_OUT)
print("Review :", REVIEW_QUEUE_OUT)
