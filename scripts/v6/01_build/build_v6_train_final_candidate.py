from pathlib import Path
import pandas as pd

ROOT = Path(r"data\v6\labeling\targeted_review")

BASE = ROOT / "Promptune_v6_train_2000_labeled_after_targeted_v2.csv"
CONTEXT_REVIEW = ROOT / "Promptune_v6_train_context_remaining_queue_v1_gpt_reviewed.csv"

OUT = ROOT / "Promptune_v6_train_2000_labeled_final_candidate_v1.csv"
APPLY_AUDIT = ROOT / "Promptune_v6_train_final_context_apply_audit_v1.csv"
DIST_AUDIT = ROOT / "Promptune_v6_train_final_distribution_audit_v1.csv"
TYPE_AUDIT = ROOT / "Promptune_v6_train_final_by_data_type_audit_v1.csv"
WEAK3_AUDIT = ROOT / "Promptune_v6_train_final_weak3_pair_audit_v1.csv"

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

def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")

print("=== V6 TRAIN FINAL CANDIDATE BUILD ===")

if not BASE.exists():
    raise FileNotFoundError(f"BASE not found: {BASE}")

if not CONTEXT_REVIEW.exists():
    raise FileNotFoundError(f"CONTEXT REVIEW not found: {CONTEXT_REVIEW}")

base = read_csv(BASE)
review = read_csv(CONTEXT_REVIEW)

# ------------------------------------------------------------
# 1. Structural validation
# ------------------------------------------------------------
if len(base) != 2000:
    raise RuntimeError(f"BASE rows must be 2000, got {len(base)}")

if base["id"].nunique() != 2000:
    raise RuntimeError("BASE IDs must be unique")

if len(review) != 1314:
    raise RuntimeError(f"Context review rows must be 1314, got {len(review)}")

if review["id"].nunique() != 1314:
    raise RuntimeError("Context review IDs must be unique")

unknown = set(review["id"]) - set(base["id"])
if unknown:
    raise RuntimeError(f"Unknown Context review IDs: {len(unknown)}")

for col in LABEL_COLS:
    vals = pd.to_numeric(base[col], errors="coerce")
    if vals.isna().any() or not vals.isin([0, 1]).all():
        raise RuntimeError(f"{col}: BASE has blank/non-binary values")
    base[col] = vals.astype(int)

rv = pd.to_numeric(review["review_context"], errors="coerce")
if rv.isna().any() or not rv.isin([0, 1]).all():
    raise RuntimeError("review_context contains blank/non-binary values")

review["review_context"] = rv.astype(int)

reviewed_count = (
    review["review_status"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("reviewed")
    .sum()
)

if reviewed_count != len(review):
    raise RuntimeError(
        f"Context review_status is not complete: {reviewed_count}/{len(review)}"
    )

# prompt_text integrity
base_idx = base.set_index("id")
prompt_mismatch = 0

for _, row in review.iterrows():
    base_prompt = str(base_idx.loc[row["id"], "prompt_text"]).strip()
    review_prompt = str(row["prompt_text"]).strip()
    if base_prompt != review_prompt:
        prompt_mismatch += 1

if prompt_mismatch:
    raise RuntimeError(f"prompt_text mismatch: {prompt_mismatch}")

# ------------------------------------------------------------
# 2. Apply remaining Context reviews
# ------------------------------------------------------------
out = base.copy().set_index("id")
review_idx = review.set_index("id")

changed = 0
zero_to_one = 0
one_to_zero = 0

for row_id in review_idx.index:
    old = int(out.loc[row_id, "human_missing_context"])
    new = int(review_idx.loc[row_id, "review_context"])

    if old != new:
        changed += 1
        if old == 0 and new == 1:
            zero_to_one += 1
        elif old == 1 and new == 0:
            one_to_zero += 1

    out.loc[row_id, "human_missing_context"] = new

out = out.reset_index()

out.to_csv(OUT, index=False, encoding="utf-8-sig")

apply_audit = pd.DataFrame([{
    "review_rows": len(review),
    "changed_rows": changed,
    "old_0_to_new_1": zero_to_one,
    "old_1_to_new_0": one_to_zero,
    "prompt_mismatch": prompt_mismatch,
}])

apply_audit.to_csv(APPLY_AUDIT, index=False, encoding="utf-8-sig")

print()
print("=== REMAINING CONTEXT APPLIED ===")
print(apply_audit.to_string(index=False))

# ------------------------------------------------------------
# 3. Final global distribution
# ------------------------------------------------------------
dist_rows = []

print()
print("=== FINAL LABEL DISTRIBUTION : TRAIN 2000 ===")

for col in LABEL_COLS:
    ones = int(out[col].sum())
    zeros = len(out) - ones
    rate = ones / len(out)

    dist_rows.append({
        "label": col,
        "missing_1": ones,
        "sufficient_0": zeros,
        "positive_rate": round(rate, 4),
    })

    print(f"{col:28s}: {ones:4d}/2000 ({rate:.2%})")

dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(DIST_AUDIT, index=False, encoding="utf-8-sig")

# Row-level label cardinality
out["_label_sum"] = out[LABEL_COLS].sum(axis=1)

print()
print("=== ROW LABEL CARDINALITY ===")
print("all-zero rows :", int((out["_label_sum"] == 0).sum()))
print("1+ missing    :", int((out["_label_sum"] >= 1).sum()))
print("2+ missing    :", int((out["_label_sum"] >= 2).sum()))
print("3+ missing    :", int((out["_label_sum"] >= 3).sum()))
print("mean labels/row:", round(float(out["_label_sum"].mean()), 4))

# ------------------------------------------------------------
# 4. Distribution by data_type
# ------------------------------------------------------------
if "data_type" in out.columns:
    records = []

    for dtype, g in out.groupby("data_type", dropna=False):
        row = {
            "data_type": dtype,
            "rows": len(g),
        }

        for col in LABEL_COLS:
            row[col] = int(g[col].sum())
            row[f"{col}_rate"] = round(float(g[col].mean()), 4)

        records.append(row)

    type_df = pd.DataFrame(records)
    type_df.to_csv(TYPE_AUDIT, index=False, encoding="utf-8-sig")

    print()
    print("=== CONTEXT BY DATA TYPE ===")
    for _, row in type_df.iterrows():
        print(
            f"{str(row['data_type']):22s} "
            f"{int(row['human_missing_context']):4d}/{int(row['rows']):4d} "
            f"({row['human_missing_context_rate']:.2%})"
        )
else:
    pd.DataFrame().to_csv(TYPE_AUDIT, index=False, encoding="utf-8-sig")
    print()
    print("WARNING: data_type column not found")

# ------------------------------------------------------------
# 5. Weak3 post-hoc pair audit
#    Design metadata is NOT GT. This only checks whether contrast pairs
#    survived in the final labels.
# ------------------------------------------------------------
weak3_rows = []

if all(c in out.columns for c in ["data_type", "pair_id", "focus_element"]):
    weak = out[
        (out["data_type"] == "weak3_focus")
        & out["pair_id"].notna()
        & out["focus_element"].notna()
    ].copy()

    focus_to_label = {
        "task": "human_missing_task",
        "format": "human_missing_format",
        "constraint": "human_missing_constraint",
    }

    for pair_id, g in weak.groupby("pair_id"):
        if len(g) != 2:
            weak3_rows.append({
                "pair_id": pair_id,
                "focus_element": "|".join(sorted(g["focus_element"].astype(str).unique())),
                "rows": len(g),
                "pattern": "INVALID_PAIR_SIZE",
                "survived_contrast": False,
            })
            continue

        focus_values = g["focus_element"].astype(str).str.lower().unique()

        if len(focus_values) != 1 or focus_values[0] not in focus_to_label:
            weak3_rows.append({
                "pair_id": pair_id,
                "focus_element": "|".join(sorted(focus_values)),
                "rows": len(g),
                "pattern": "UNKNOWN_FOCUS",
                "survived_contrast": False,
            })
            continue

        focus = focus_values[0]
        label_col = focus_to_label[focus]

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
weak3_df.to_csv(WEAK3_AUDIT, index=False, encoding="utf-8-sig")

if len(weak3_df):
    print()
    print("=== WEAK3 POST-HOC CONTRAST AUDIT ===")

    summary = (
        weak3_df.groupby(["focus_element", "pattern"])
        .size()
        .unstack(fill_value=0)
    )

    print(summary.to_string())

    survived = int(weak3_df["survived_contrast"].sum())
    total_pairs = len(weak3_df)

    print()
    print(
        f"survived contrast pairs: "
        f"{survived}/{total_pairs} ({survived/total_pairs:.2%})"
    )
else:
    print()
    print("WARNING: Weak3 pair audit could not be produced")

# ------------------------------------------------------------
# 6. Final structural check
# ------------------------------------------------------------
final_invalid = 0

for col in LABEL_COLS:
    vals = pd.to_numeric(out[col], errors="coerce")
    final_invalid += int(vals.isna().sum())
    final_invalid += int((~vals.isin([0, 1])).sum())

print()
print("=== FINAL STRUCTURAL CHECK ===")
print("rows       :", len(out))
print("unique IDs :", out["id"].nunique())
print("label invalid/blanks:", final_invalid)

if len(out) != 2000 or out["id"].nunique() != 2000 or final_invalid != 0:
    raise RuntimeError("FINAL STRUCTURAL CHECK FAILED")

print()
print("FINAL CANDIDATE:", OUT)
print("APPLY AUDIT    :", APPLY_AUDIT)
print("DIST AUDIT     :", DIST_AUDIT)
print("TYPE AUDIT     :", TYPE_AUDIT)
print("WEAK3 AUDIT    :", WEAK3_AUDIT)
print()
print("V6 TRAIN FINAL CANDIDATE BUILD: PASS")
