from pathlib import Path
import pandas as pd

ROOT = Path(r"data\v6\labeling")

BASE = (
    ROOT / "final_candidate"
    / "Promptune_v6_train_2000_context_human_final_candidate_v1.csv"
)

LABELED_DIR = ROOT / "final_3label_human_review" / "labeled"

PACKS = [
    LABELED_DIR / f"Promptune_v6_final3_human_pack_{i:02d}_labeled.csv"
    for i in range(1, 4)
]

OUT_DIR = ROOT / "final_candidate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_OUT = OUT_DIR / "Promptune_v6_train_2000_final_candidate_v2.csv"
APPLY_AUDIT_OUT = OUT_DIR / "Promptune_v6_train_final3_apply_audit_v2.csv"
DIST_OUT = OUT_DIR / "Promptune_v6_train_final_distribution_v2.csv"
TYPE_OUT = OUT_DIR / "Promptune_v6_train_final_by_data_type_v2.csv"
WEAK3_OUT = OUT_DIR / "Promptune_v6_train_final_weak3_pair_audit_v2.csv"
PROVENANCE_OUT = OUT_DIR / "Promptune_v6_train_final3_review_provenance_v2.csv"

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

REVIEW_MAP = {
    "task": {
        "check": "check_task",
        "review": "review_task",
        "target": "human_missing_task",
    },
    "audience": {
        "check": "check_audience",
        "review": "review_audience",
        "target": "human_missing_audience",
    },
    "format": {
        "check": "check_format",
        "review": "review_format",
        "target": "human_missing_format",
    },
}


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def binary(series, name):
    v = pd.to_numeric(series, errors="coerce")
    if v.isna().any():
        raise RuntimeError(f"{name}: blank/non-numeric values found")
    if not v.isin([0, 1]).all():
        raise RuntimeError(f"{name}: non-binary values found")
    return v.astype(int)


print("=== V6 FINAL 3-LABEL HUMAN REVIEW APPLY ===")

# ------------------------------------------------------------
# 1. Files / base validation
# ------------------------------------------------------------
if not BASE.exists():
    raise FileNotFoundError(BASE)

for p in PACKS:
    if not p.exists():
        raise FileNotFoundError(p)

base = read_csv(BASE)

if len(base) != 2000 or base["id"].nunique() != 2000:
    raise RuntimeError("BASE must contain 2000 unique IDs")

for col in LABEL_COLS:
    if col not in base.columns:
        raise RuntimeError(f"BASE missing {col}")
    base[col] = binary(base[col], f"BASE {col}")

base_idx = base.set_index("id")

# ------------------------------------------------------------
# 2. Load and validate 3 labeled packs
# ------------------------------------------------------------
pack_frames = []

for pack_no, path in enumerate(PACKS, start=1):
    df = read_csv(path)

    required = {
        "id",
        "prompt_text",
        "check_task",
        "check_audience",
        "check_format",
        "review_task",
        "review_audience",
        "review_format",
        "review_status",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{path.name}: missing columns {sorted(missing)}"
        )

    if df["id"].nunique() != len(df):
        raise RuntimeError(f"{path.name}: duplicate IDs")

    status = (
        df["review_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not status.eq("reviewed").all():
        bad = int((~status.eq("reviewed")).sum())
        raise RuntimeError(
            f"{path.name}: {bad} rows are not reviewed"
        )

    # Validate check flags.
    for c in ["check_task", "check_audience", "check_format"]:
        df[c] = binary(df[c], f"{path.name} {c}")

    # Validate only review cells that were actually requested.
    for elem, cfg in REVIEW_MAP.items():
        mask = df[cfg["check"]] == 1

        vals = pd.to_numeric(
            df.loc[mask, cfg["review"]],
            errors="coerce",
        )

        if vals.isna().any() or not vals.isin([0, 1]).all():
            raise RuntimeError(
                f"{path.name}: invalid {cfg['review']} "
                f"on checked {elem} rows"
            )

        df.loc[mask, cfg["review"]] = vals.astype(int)

        # Unchecked cells may be blank; if nonblank, ensure they are not
        # silently applied later.
    df["_pack"] = f"pack_{pack_no:02d}"
    pack_frames.append(df)

reviews = pd.concat(pack_frames, ignore_index=True)

if reviews["id"].nunique() != len(reviews):
    dupes = reviews.loc[
        reviews["id"].duplicated(keep=False),
        ["id", "_pack"]
    ]
    print(dupes.to_string(index=False))
    raise RuntimeError("Duplicate IDs across labeled packs")

unknown = set(reviews["id"]) - set(base["id"])
if unknown:
    raise RuntimeError(f"Unknown review IDs: {len(unknown)}")

# Prompt integrity
prompt_mismatch = 0

for _, r in reviews.iterrows():
    old_prompt = str(base_idx.loc[r["id"], "prompt_text"]).strip()
    new_prompt = str(r["prompt_text"]).strip()

    if old_prompt != new_prompt:
        prompt_mismatch += 1

if prompt_mismatch:
    raise RuntimeError(f"prompt_text mismatch: {prompt_mismatch}")

print()
print("=== LABELED PACK COVERAGE ===")
for pack_no, df in enumerate(pack_frames, start=1):
    print(
        f"pack_{pack_no:02d}: "
        f"rows={len(df)}, "
        f"task={int(df['check_task'].sum())}, "
        f"audience={int(df['check_audience'].sum())}, "
        f"format={int(df['check_format'].sum())}"
    )

print("unique reviewed rows:", reviews["id"].nunique())
print("prompt mismatches   :", prompt_mismatch)

# ------------------------------------------------------------
# 3. Apply only checked Task/Audience/Format fields
# ------------------------------------------------------------
final = base.copy().set_index("id")
audit_rows = []
provenance_rows = []

for elem, cfg in REVIEW_MAP.items():
    mask = reviews[cfg["check"]] == 1
    subset = reviews.loc[
        mask,
        ["id", "_pack", cfg["review"], "review_note"]
        if "review_note" in reviews.columns
        else ["id", "_pack", cfg["review"]]
    ].copy()

    changed = 0
    zero_to_one = 0
    one_to_zero = 0
    new_ones = 0
    new_zeros = 0

    for _, r in subset.iterrows():
        row_id = r["id"]
        old = int(final.loc[row_id, cfg["target"]])
        new = int(r[cfg["review"]])

        if new == 1:
            new_ones += 1
        else:
            new_zeros += 1

        if old != new:
            changed += 1
            if old == 0 and new == 1:
                zero_to_one += 1
            elif old == 1 and new == 0:
                one_to_zero += 1

        final.loc[row_id, cfg["target"]] = new

        provenance_rows.append({
            "id": row_id,
            "element": elem,
            "old_value": old,
            "review_value": new,
            "pack": r["_pack"],
        })

    audit_rows.append({
        "element": elem,
        "checked_rows": len(subset),
        "review_1": new_ones,
        "review_0": new_zeros,
        "changed_rows": changed,
        "old_0_to_new_1": zero_to_one,
        "old_1_to_new_0": one_to_zero,
    })

final = final.reset_index()

apply_audit = pd.DataFrame(audit_rows)
apply_audit.to_csv(
    APPLY_AUDIT_OUT,
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(provenance_rows).to_csv(
    PROVENANCE_OUT,
    index=False,
    encoding="utf-8-sig",
)

print()
print("=== HUMAN REVIEW APPLIED ===")
print(apply_audit.to_string(index=False))

# ------------------------------------------------------------
# 4. Ensure non-targeted labels did not change
# ------------------------------------------------------------
before = base.set_index("id")
after = final.set_index("id")

for col in [
    "human_missing_context",
    "human_missing_tone",
    "human_missing_length",
    "human_missing_constraint",
    "human_missing_example",
]:
    if not before[col].equals(after[col]):
        raise RuntimeError(f"Unexpected label change: {col}")

# ------------------------------------------------------------
# 5. Global distribution
# ------------------------------------------------------------
dist_rows = []

print()
print("=== FINAL LABEL DISTRIBUTION : TRAIN 2000 ===")

for col in LABEL_COLS:
    final[col] = binary(final[col], col)

    ones = int(final[col].sum())
    zeros = len(final) - ones
    rate = ones / len(final)

    dist_rows.append({
        "label": col,
        "missing_1": ones,
        "sufficient_0": zeros,
        "positive_rate": round(rate, 4),
    })

    print(f"{col:28s}: {ones:4d}/2000 ({rate:.2%})")

pd.DataFrame(dist_rows).to_csv(
    DIST_OUT,
    index=False,
    encoding="utf-8-sig",
)

# ------------------------------------------------------------
# 6. Row label cardinality
# ------------------------------------------------------------
final["_label_sum"] = final[LABEL_COLS].sum(axis=1)

print()
print("=== ROW LABEL CARDINALITY ===")
print("all-zero rows :", int((final["_label_sum"] == 0).sum()))
print("1+ missing    :", int((final["_label_sum"] >= 1).sum()))
print("2+ missing    :", int((final["_label_sum"] >= 2).sum()))
print("3+ missing    :", int((final["_label_sum"] >= 3).sum()))
print("mean labels/row:", round(float(final["_label_sum"].mean()), 4))

# Do not persist helper column in final training candidate.
final = final.drop(columns=["_label_sum"])

# ------------------------------------------------------------
# 7. By data type
# ------------------------------------------------------------
type_rows = []

if "data_type" in final.columns:
    print()
    print("=== FINAL LABELS BY DATA TYPE ===")

    for dtype, g in final.groupby("data_type", dropna=False):
        row = {
            "data_type": dtype,
            "rows": len(g),
        }

        for col in LABEL_COLS:
            row[col] = int(g[col].sum())
            row[f"{col}_rate"] = round(float(g[col].mean()), 4)

        type_rows.append(row)

        print(
            f"{str(dtype):22s} "
            f"rows={len(g):4d} "
            f"context={int(g['human_missing_context'].sum()):4d} "
            f"task={int(g['human_missing_task'].sum()):3d} "
            f"aud={int(g['human_missing_audience'].sum()):3d} "
            f"fmt={int(g['human_missing_format'].sum()):3d}"
        )

pd.DataFrame(type_rows).to_csv(
    TYPE_OUT,
    index=False,
    encoding="utf-8-sig",
)

# ------------------------------------------------------------
# 8. Weak3 post-hoc audit
#    Metadata is design intent only, never used to change GT.
# ------------------------------------------------------------
weak3_rows = []

needed = {"data_type", "pair_id", "focus_element"}

if needed.issubset(final.columns):
    weak = final[
        (final["data_type"] == "weak3_focus")
        & final["pair_id"].notna()
        & final["focus_element"].notna()
    ].copy()

    focus_map = {
        "task": "human_missing_task",
        "format": "human_missing_format",
        "constraint": "human_missing_constraint",
    }

    for pair_id, g in weak.groupby("pair_id"):
        if len(g) != 2:
            weak3_rows.append({
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
            weak3_rows.append({
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

        weak3_rows.append({
            "pair_id": pair_id,
            "focus_element": focus,
            "rows": 2,
            "pattern": pattern,
            "survived_contrast": survived,
        })

weak3_df = pd.DataFrame(weak3_rows)
weak3_df.to_csv(
    WEAK3_OUT,
    index=False,
    encoding="utf-8-sig",
)

if len(weak3_df):
    print()
    print("=== WEAK3 POST-HOC CONTRAST AUDIT ===")

    weak_summary = (
        weak3_df.groupby(["focus_element", "pattern"])
        .size()
        .unstack(fill_value=0)
    )

    print(weak_summary.to_string())

    survived = int(weak3_df["survived_contrast"].sum())
    total = len(weak3_df)

    print()
    print(
        f"survived contrast pairs: "
        f"{survived}/{total} ({survived/total:.2%})"
    )

# ------------------------------------------------------------
# 9. Final structural check and save
# ------------------------------------------------------------
invalid = 0

for col in LABEL_COLS:
    v = pd.to_numeric(final[col], errors="coerce")
    invalid += int(v.isna().sum())
    invalid += int((~v.isin([0, 1])).sum())

print()
print("=== FINAL STRUCTURAL CHECK ===")
print("rows       :", len(final))
print("unique IDs :", final["id"].nunique())
print("label invalid/blanks:", invalid)

if len(final) != 2000:
    raise RuntimeError("Final rows != 2000")

if final["id"].nunique() != 2000:
    raise RuntimeError("Final IDs are not unique")

if invalid != 0:
    raise RuntimeError("Final labels contain invalid/blanks")

final.to_csv(
    FINAL_OUT,
    index=False,
    encoding="utf-8-sig",
)

print()
print("FINAL      :", FINAL_OUT)
print("APPLY AUDIT:", APPLY_AUDIT_OUT)
print("DIST       :", DIST_OUT)
print("BY TYPE    :", TYPE_OUT)
print("WEAK3      :", WEAK3_OUT)
print("PROVENANCE :", PROVENANCE_OUT)
print()
print("V6 FINAL 3-LABEL HUMAN REVIEW APPLY: PASS")
