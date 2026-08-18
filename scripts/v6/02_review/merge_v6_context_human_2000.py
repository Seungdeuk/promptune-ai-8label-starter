from pathlib import Path
import pandas as pd

ROOT = Path(r"data\v6\labeling")

BASE = ROOT / "targeted_review" / "Promptune_v6_train_2000_labeled_calibrated_v1.csv"
CALIB = ROOT / "calibration" / "Promptune_v6_train_calibration_human_160.csv"
AUDIT125 = ROOT / "context_human_audit" / "Promptune_v6_train_context_human_audit_human_125.csv"
LABELED_DIR = ROOT / "context_human_full_review" / "labeled"

OUT_DIR = ROOT / "final_candidate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_OUT = OUT_DIR / "Promptune_v6_train_2000_context_human_final_candidate_v1.csv"
PROVENANCE_OUT = OUT_DIR / "Promptune_v6_train_context_human_provenance_v1.csv"
DIST_OUT = OUT_DIR / "Promptune_v6_train_context_human_distribution_v1.csv"
TYPE_OUT = OUT_DIR / "Promptune_v6_train_context_human_by_data_type_v1.csv"

PACK_FILES = [
    LABELED_DIR / f"Promptune_v6_context_human_pack_{i:02d}_labeled.csv"
    for i in range(1, 6)
]

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

def validate_binary(series, name):
    v = pd.to_numeric(series, errors="coerce")
    if v.isna().any():
        raise RuntimeError(f"{name}: blank/non-numeric value found")
    if not v.isin([0, 1]).all():
        raise RuntimeError(f"{name}: non-binary value found")
    return v.astype(int)

def validate_reviewed(df, status_col, expected, name):
    if status_col not in df.columns:
        raise RuntimeError(f"{name}: missing {status_col}")
    count = (
        df[status_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("reviewed")
        .sum()
    )
    if count != expected:
        raise RuntimeError(f"{name}: reviewed {count}/{expected}")

print("=== V6 CONTEXT HUMAN 2000 MERGE ===")

# ------------------------------------------------------------
# 1. Validate required files
# ------------------------------------------------------------
for p in [BASE, CALIB, AUDIT125, *PACK_FILES]:
    if not p.exists():
        raise FileNotFoundError(p)

base = read_csv(BASE)

if len(base) != 2000 or base["id"].nunique() != 2000:
    raise RuntimeError("BASE must contain 2000 unique IDs")

for col in LABEL_COLS:
    if col not in base.columns:
        raise RuntimeError(f"BASE missing {col}")
    base[col] = validate_binary(base[col], f"BASE {col}")

base_idx = base.set_index("id")

# ------------------------------------------------------------
# 2. Collect Human Context labels from all 3 sources
# ------------------------------------------------------------
records = []

# A) Human Calibration 160
calib = read_csv(CALIB)

if len(calib) != 160 or calib["id"].nunique() != 160:
    raise RuntimeError("Calibration must contain 160 unique IDs")

validate_reviewed(calib, "calibration_status", 160, "Calibration")

calib["calib_missing_context"] = validate_binary(
    calib["calib_missing_context"],
    "Calibration calib_missing_context",
)

for _, r in calib.iterrows():
    records.append({
        "id": r["id"],
        "prompt_text": r["prompt_text"],
        "human_context": int(r["calib_missing_context"]),
        "review_source": "human_calibration_160",
        "review_file": CALIB.name,
    })

# B) Human Audit 125
audit125 = read_csv(AUDIT125)

if len(audit125) != 125 or audit125["id"].nunique() != 125:
    raise RuntimeError("Human Audit125 must contain 125 unique IDs")

validate_reviewed(audit125, "review_status", 125, "Human Audit125")

audit125["review_context"] = validate_binary(
    audit125["review_context"],
    "Human Audit125 review_context",
)

for _, r in audit125.iterrows():
    records.append({
        "id": r["id"],
        "prompt_text": r["prompt_text"],
        "human_context": int(r["review_context"]),
        "review_source": "human_audit_125",
        "review_file": AUDIT125.name,
    })

# C) Human Full Review 5 packs
pack_total = 0

for p in PACK_FILES:
    df = read_csv(p)

    if "id" not in df.columns or "prompt_text" not in df.columns:
        raise RuntimeError(f"{p.name}: missing id/prompt_text")

    if df["id"].nunique() != len(df):
        raise RuntimeError(f"{p.name}: duplicate IDs inside pack")

    validate_reviewed(df, "review_status", len(df), p.name)

    df["review_context"] = validate_binary(
        df["review_context"],
        f"{p.name} review_context",
    )

    pack_total += len(df)

    for _, r in df.iterrows():
        records.append({
            "id": r["id"],
            "prompt_text": r["prompt_text"],
            "human_context": int(r["review_context"]),
            "review_source": "human_full_review_1715",
            "review_file": p.name,
        })

print("Human Calibration rows :", len(calib))
print("Human Audit125 rows    :", len(audit125))
print("Five pack rows         :", pack_total)

# ------------------------------------------------------------
# 3. Coverage / duplicate / overlap checks
# ------------------------------------------------------------
human = pd.DataFrame(records)

print()
print("=== HUMAN CONTEXT COVERAGE ===")
print("total review records :", len(human))
print("unique IDs           :", human["id"].nunique())

dupes = human[human["id"].duplicated(keep=False)]

if len(dupes):
    print()
    print("Duplicate review IDs:")
    print(dupes[["id", "review_source", "review_file"]].to_string(index=False))
    raise RuntimeError(f"Human Context sources overlap on {dupes['id'].nunique()} IDs")

if len(human) != 2000 or human["id"].nunique() != 2000:
    raise RuntimeError(
        f"Human Context coverage must be exactly 2000 unique IDs, "
        f"got rows={len(human)}, unique={human['id'].nunique()}"
    )

missing_ids = set(base["id"]) - set(human["id"])
extra_ids = set(human["id"]) - set(base["id"])

if missing_ids or extra_ids:
    raise RuntimeError(
        f"ID coverage mismatch: missing={len(missing_ids)}, extra={len(extra_ids)}"
    )

# Prompt integrity check
prompt_mismatch = []

for _, r in human.iterrows():
    base_prompt = str(base_idx.loc[r["id"], "prompt_text"]).strip()
    human_prompt = str(r["prompt_text"]).strip()

    if base_prompt != human_prompt:
        prompt_mismatch.append(r["id"])

if prompt_mismatch:
    raise RuntimeError(f"prompt_text mismatch on {len(prompt_mismatch)} rows")

print("missing IDs          :", len(missing_ids))
print("extra IDs            :", len(extra_ids))
print("prompt mismatches    :", len(prompt_mismatch))

# ------------------------------------------------------------
# 4. Replace ONLY Context in canonical calibrated base
# ------------------------------------------------------------
final = base.copy().set_index("id")
human_idx = human.set_index("id")

old_context = final["human_missing_context"].copy()

for row_id in human_idx.index:
    final.loc[row_id, "human_missing_context"] = int(
        human_idx.loc[row_id, "human_context"]
    )

new_context = final["human_missing_context"].astype(int)

changed = int((old_context != new_context).sum())
zero_to_one = int(((old_context == 0) & (new_context == 1)).sum())
one_to_zero = int(((old_context == 1) & (new_context == 0)).sum())

final = final.reset_index()

# Ensure no other label columns were changed by this operation.
base_compare = base.set_index("id")
final_compare = final.set_index("id")

for col in LABEL_COLS:
    if col == "human_missing_context":
        continue
    if not base_compare[col].equals(final_compare[col]):
        raise RuntimeError(f"Unexpected change outside Context: {col}")

final.to_csv(FINAL_OUT, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 5. Provenance
# ------------------------------------------------------------
prov = human[[
    "id",
    "human_context",
    "review_source",
    "review_file",
]].copy()

prov.to_csv(PROVENANCE_OUT, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 6. Final label distribution
# ------------------------------------------------------------
dist_rows = []

print()
print("=== FINAL LABEL DISTRIBUTION : TRAIN 2000 ===")

for col in LABEL_COLS:
    values = validate_binary(final[col], col)
    ones = int(values.sum())
    zeros = len(values) - ones
    rate = ones / len(values)

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
# 7. Context by data_type
# ------------------------------------------------------------
type_rows = []

if "data_type" in final.columns:
    print()
    print("=== HUMAN CONTEXT BY DATA TYPE ===")

    for dtype, g in final.groupby("data_type", dropna=False):
        ones = int(g["human_missing_context"].sum())
        rows = len(g)
        rate = ones / rows

        type_rows.append({
            "data_type": dtype,
            "rows": rows,
            "context_1": ones,
            "context_0": rows - ones,
            "positive_rate": round(rate, 4),
        })

        print(
            f"{str(dtype):22s} "
            f"{ones:4d}/{rows:4d} "
            f"({rate:.2%})"
        )

pd.DataFrame(type_rows).to_csv(
    TYPE_OUT,
    index=False,
    encoding="utf-8-sig",
)

# ------------------------------------------------------------
# 8. Context source distribution
# ------------------------------------------------------------
print()
print("=== REVIEW SOURCE COUNTS ===")
print(human["review_source"].value_counts().to_string())

print()
print("=== CONTEXT CHANGE FROM CALIBRATED BASE ===")
print("changed rows :", changed)
print("0 -> 1       :", zero_to_one)
print("1 -> 0       :", one_to_zero)

print()
print("=== FINAL STRUCTURAL CHECK ===")
print("rows       :", len(final))
print("unique IDs :", final["id"].nunique())

invalid = 0
for col in LABEL_COLS:
    v = pd.to_numeric(final[col], errors="coerce")
    invalid += int(v.isna().sum())
    invalid += int((~v.isin([0, 1])).sum())

print("label invalid/blanks:", invalid)

if len(final) != 2000 or final["id"].nunique() != 2000 or invalid != 0:
    raise RuntimeError("FINAL STRUCTURAL CHECK FAILED")

print()
print("FINAL      :", FINAL_OUT)
print("PROVENANCE :", PROVENANCE_OUT)
print("DIST       :", DIST_OUT)
print("BY TYPE    :", TYPE_OUT)
print()
print("V6 CONTEXT HUMAN 2000 MERGE: PASS")
