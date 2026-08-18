from pathlib import Path
import pandas as pd

ROOT = Path(r"data\v6")
MASTER = ROOT / "Promptune_v6_valid_400_unlabeled_master_v1.csv"

LABEL_ROOT = ROOT / "labeling" / "valid_human_review"
LABELED_DIR = LABEL_ROOT / "labeled"

PACKS = [
    LABELED_DIR / f"Promptune_v6_valid_human_pack_{i:02d}_labeled.csv"
    for i in range(1, 6)
]

OUT_DIR = LABEL_ROOT / "merged"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_OUT = OUT_DIR / "Promptune_v6_valid_400_human_final_candidate_v1.csv"
PACK_AUDIT_OUT = OUT_DIR / "Promptune_v6_valid_400_pack_label_audit_v1.csv"
DIST_OUT = OUT_DIR / "Promptune_v6_valid_400_distribution_v1.csv"
TYPE_OUT = OUT_DIR / "Promptune_v6_valid_400_by_data_type_v1.csv"
WEAK3_OUT = OUT_DIR / "Promptune_v6_valid_400_weak3_pair_audit_v1.csv"
PROVENANCE_OUT = OUT_DIR / "Promptune_v6_valid_400_review_provenance_v1.csv"

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

DATA_TYPES = [
    "general_business",
    "community_messenger",
    "weak3_focus",
    "hard_negative",
    "complex_boundary",
]


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def validate_binary(series, name):
    values = pd.to_numeric(series, errors="coerce")

    if values.isna().any():
        raise RuntimeError(f"{name}: blank/non-numeric labels found")

    if not values.isin([0, 1]).all():
        raise RuntimeError(f"{name}: labels other than 0/1 found")

    return values.astype(int)


print("=== V6 VALID 400 HUMAN MERGE + AUDIT ===")

# ------------------------------------------------------------
# 1. Validate source files
# ------------------------------------------------------------
if not MASTER.exists():
    raise FileNotFoundError(MASTER)

for p in PACKS:
    if not p.exists():
        raise FileNotFoundError(p)

master = read_csv(MASTER)

if len(master) != 400:
    raise RuntimeError(f"Master rows expected 400, got {len(master)}")

if master["id"].nunique() != 400:
    raise RuntimeError("Master IDs are not unique")

master_idx = master.set_index("id")

pack_frames = []
pack_audit_rows = []
provenance_rows = []

# ------------------------------------------------------------
# 2. Validate each labeled pack
# ------------------------------------------------------------
for pack_no, path in enumerate(PACKS, start=1):

    df = read_csv(path)

    if len(df) != 80:
        raise RuntimeError(
            f"{path.name}: expected 80 rows, got {len(df)}"
        )

    if df["id"].nunique() != 80:
        raise RuntimeError(f"{path.name}: duplicate IDs")

    required = {
        "id",
        "prompt_text",
        *LABEL_COLS,
        "review_status",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{path.name}: missing columns {sorted(missing)}"
        )

    reviewed = (
        df["review_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("reviewed")
    )

    if not reviewed.all():
        bad = int((~reviewed).sum())
        raise RuntimeError(
            f"{path.name}: {bad} rows are not reviewed"
        )

    for col in LABEL_COLS:
        df[col] = validate_binary(
            df[col],
            f"{path.name} {col}",
        )

    unknown = set(df["id"]) - set(master["id"])

    if unknown:
        raise RuntimeError(
            f"{path.name}: unknown Valid IDs={len(unknown)}"
        )

    # prompt_text integrity
    mismatch = 0

    for _, row in df.iterrows():
        old = str(master_idx.loc[row["id"], "prompt_text"]).strip()
        new = str(row["prompt_text"]).strip()

        if old != new:
            mismatch += 1

    if mismatch:
        raise RuntimeError(
            f"{path.name}: prompt_text mismatch={mismatch}"
        )

    df["_pack"] = f"pack_{pack_no:02d}"
    pack_frames.append(df)

    audit = {
        "pack": f"pack_{pack_no:02d}",
        "rows": len(df),
    }

    for col in LABEL_COLS:
        ones = int(df[col].sum())
        audit[f"{col}_1"] = ones
        audit[f"{col}_rate"] = round(ones / len(df), 4)

    pack_audit_rows.append(audit)

    for row_id in df["id"]:
        provenance_rows.append({
            "id": row_id,
            "review_pack": f"pack_{pack_no:02d}",
            "review_file": path.name,
            "review_type": "blind_human_8label",
        })

reviews = pd.concat(pack_frames, ignore_index=True)

# ------------------------------------------------------------
# 3. Global coverage
# ------------------------------------------------------------
print()
print("=== LABELED PACK COVERAGE ===")

for pack_no, df in enumerate(pack_frames, start=1):
    print(
        f"pack_{pack_no:02d}: rows={len(df)}, "
        f"unique={df['id'].nunique()}"
    )

print("combined rows      :", len(reviews))
print("combined unique IDs:", reviews["id"].nunique())

if len(reviews) != 400:
    raise RuntimeError(
        f"Combined rows expected 400, got {len(reviews)}"
    )

if reviews["id"].nunique() != 400:
    dupes = reviews.loc[
        reviews["id"].duplicated(keep=False),
        ["id", "_pack"],
    ]
    print()
    print(dupes.to_string(index=False))
    raise RuntimeError("Duplicate IDs across labeled packs")

missing_ids = set(master["id"]) - set(reviews["id"])
extra_ids = set(reviews["id"]) - set(master["id"])

print("missing IDs        :", len(missing_ids))
print("extra IDs          :", len(extra_ids))

if missing_ids or extra_ids:
    raise RuntimeError(
        f"Coverage mismatch: missing={len(missing_ids)}, "
        f"extra={len(extra_ids)}"
    )

# ------------------------------------------------------------
# 4. Merge Human labels into master metadata
# ------------------------------------------------------------
review_labels = reviews[
    ["id", *LABEL_COLS]
].copy()

merged = master.drop(
    columns=[c for c in LABEL_COLS if c in master.columns]
).merge(
    review_labels,
    on="id",
    how="left",
    validate="one_to_one",
)

# Restore master order.
order = {row_id: i for i, row_id in enumerate(master["id"])}
merged["_order"] = merged["id"].map(order)
merged = (
    merged.sort_values("_order")
    .drop(columns=["_order"])
    .reset_index(drop=True)
)

for col in LABEL_COLS:
    merged[col] = validate_binary(
        merged[col],
        f"MERGED {col}",
    )

if len(merged) != 400 or merged["id"].nunique() != 400:
    raise RuntimeError("Merged structure invalid")

merged.to_csv(
    MERGED_OUT,
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(pack_audit_rows).to_csv(
    PACK_AUDIT_OUT,
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(provenance_rows).to_csv(
    PROVENANCE_OUT,
    index=False,
    encoding="utf-8-sig",
)

# ------------------------------------------------------------
# 5. Pack reviewer-distribution audit
# ------------------------------------------------------------
pack_audit = pd.DataFrame(pack_audit_rows)

print()
print("=== PACK LABEL POSITIVE RATES ===")

for col in LABEL_COLS:
    print()
    print(f"[{col}]")

    rates = []

    for _, row in pack_audit.iterrows():
        rate = float(row[f"{col}_rate"])
        rates.append(rate)

        print(
            f"{row['pack']}: "
            f"{int(row[f'{col}_1']):2d}/80 "
            f"({rate:.2%})"
        )

    spread = max(rates) - min(rates)
    print(f"spread: {spread:.2%}")

# ------------------------------------------------------------
# 6. Overall distribution
# ------------------------------------------------------------
dist_rows = []

print()
print("=== VALID 400 LABEL DISTRIBUTION ===")

for col in LABEL_COLS:
    ones = int(merged[col].sum())
    zeros = len(merged) - ones
    rate = ones / len(merged)

    dist_rows.append({
        "label": col,
        "missing_1": ones,
        "sufficient_0": zeros,
        "positive_rate": round(rate, 4),
    })

    print(
        f"{col:28s}: "
        f"{ones:3d}/400 ({rate:.2%})"
    )

pd.DataFrame(dist_rows).to_csv(
    DIST_OUT,
    index=False,
    encoding="utf-8-sig",
)

# ------------------------------------------------------------
# 7. Row cardinality
# ------------------------------------------------------------
merged["_label_sum"] = merged[LABEL_COLS].sum(axis=1)

print()
print("=== VALID ROW LABEL CARDINALITY ===")
print("all-zero rows :", int((merged["_label_sum"] == 0).sum()))
print("1+ missing    :", int((merged["_label_sum"] >= 1).sum()))
print("2+ missing    :", int((merged["_label_sum"] >= 2).sum()))
print("3+ missing    :", int((merged["_label_sum"] >= 3).sum()))
print("mean labels/row:", round(float(merged["_label_sum"].mean()), 4))

merged = merged.drop(columns=["_label_sum"])

# Re-save without helper.
merged.to_csv(
    MERGED_OUT,
    index=False,
    encoding="utf-8-sig",
)

# ------------------------------------------------------------
# 8. Distribution by data_type
# ------------------------------------------------------------
type_rows = []

if "data_type" in merged.columns:
    print()
    print("=== VALID LABELS BY DATA TYPE ===")

    for dtype in DATA_TYPES:
        g = merged[merged["data_type"] == dtype]

        if len(g) == 0:
            continue

        row = {
            "data_type": dtype,
            "rows": len(g),
        }

        for col in LABEL_COLS:
            row[col] = int(g[col].sum())
            row[f"{col}_rate"] = round(float(g[col].mean()), 4)

        type_rows.append(row)

        print(
            f"{dtype:22s} "
            f"rows={len(g):3d} "
            f"task={int(g['human_missing_task'].sum()):2d} "
            f"context={int(g['human_missing_context'].sum()):3d} "
            f"format={int(g['human_missing_format'].sum()):2d} "
            f"constraint={int(g['human_missing_constraint'].sum()):2d}"
        )

pd.DataFrame(type_rows).to_csv(
    TYPE_OUT,
    index=False,
    encoding="utf-8-sig",
)

# ------------------------------------------------------------
# 9. Weak3 post-hoc audit
# Design metadata is not GT; audit only.
# ------------------------------------------------------------
weak_rows = []

needed = {"data_type", "pair_id", "focus_element"}

if needed.issubset(merged.columns):

    weak = merged[
        (merged["data_type"] == "weak3_focus")
        & merged["pair_id"].notna()
        & merged["focus_element"].notna()
    ].copy()

    focus_map = {
        "task": "human_missing_task",
        "format": "human_missing_format",
        "constraint": "human_missing_constraint",
    }

    for pair_id, g in weak.groupby("pair_id"):

        if len(g) != 2:
            weak_rows.append({
                "pair_id": pair_id,
                "focus_element": "",
                "rows": len(g),
                "pattern": "INVALID_PAIR_SIZE",
                "survived_contrast": False,
            })
            continue

        focus_values = (
            g["focus_element"]
            .astype(str)
            .str.lower()
            .unique()
        )

        if len(focus_values) != 1 or focus_values[0] not in focus_map:
            weak_rows.append({
                "pair_id": pair_id,
                "focus_element": "|".join(focus_values),
                "rows": 2,
                "pattern": "UNKNOWN_FOCUS",
                "survived_contrast": False,
            })
            continue

        focus = focus_values[0]
        label_col = focus_map[focus]
        vals = sorted(g[label_col].astype(int).tolist())

        if vals == [0, 1]:
            pattern = "0/1"
            survived = True
        elif vals == [0, 0]:
            pattern = "0/0"
            survived = False
        elif vals == [1, 1]:
            pattern = "1/1"
            survived = False
        else:
            pattern = str(vals)
            survived = False

        weak_rows.append({
            "pair_id": pair_id,
            "focus_element": focus,
            "rows": 2,
            "pattern": pattern,
            "survived_contrast": survived,
        })

weak_df = pd.DataFrame(weak_rows)

weak_df.to_csv(
    WEAK3_OUT,
    index=False,
    encoding="utf-8-sig",
)

if len(weak_df):
    print()
    print("=== VALID WEAK3 POST-HOC CONTRAST AUDIT ===")

    summary = (
        weak_df.groupby(["focus_element", "pattern"])
        .size()
        .unstack(fill_value=0)
    )

    print(summary.to_string())

    survived = int(weak_df["survived_contrast"].sum())
    total = len(weak_df)

    print()
    print(
        f"survived contrast pairs: "
        f"{survived}/{total} ({survived/total:.2%})"
    )

# ------------------------------------------------------------
# 10. Final structural check
# ------------------------------------------------------------
invalid = 0

for col in LABEL_COLS:
    v = pd.to_numeric(merged[col], errors="coerce")
    invalid += int(v.isna().sum())
    invalid += int((~v.isin([0, 1])).sum())

print()
print("=== VALID FINAL STRUCTURAL CHECK ===")
print("rows       :", len(merged))
print("unique IDs :", merged["id"].nunique())
print("label invalid/blanks:", invalid)

if (
    len(merged) != 400
    or merged["id"].nunique() != 400
    or invalid != 0
):
    raise RuntimeError("VALID FINAL STRUCTURAL CHECK FAILED")

print()
print("MERGED    :", MERGED_OUT)
print("PACK AUDIT:", PACK_AUDIT_OUT)
print("DIST      :", DIST_OUT)
print("BY TYPE   :", TYPE_OUT)
print("WEAK3     :", WEAK3_OUT)
print("PROVENANCE:", PROVENANCE_OUT)
print()
print("V6 VALID 400 HUMAN MERGE + AUDIT: PASS")
