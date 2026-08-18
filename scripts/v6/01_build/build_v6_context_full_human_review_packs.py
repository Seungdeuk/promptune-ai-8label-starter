from pathlib import Path
import pandas as pd
import random

ROOT = Path(r"data\v6\labeling")

BASE = ROOT / "targeted_review" / "Promptune_v6_train_2000_labeled_calibrated_v1.csv"
CALIB = ROOT / "calibration" / "Promptune_v6_train_calibration_human_160.csv"
AUDIT125 = ROOT / "context_human_audit" / "Promptune_v6_train_context_human_audit_human_125.csv"

OUT_ROOT = ROOT / "context_human_full_review"
BLIND_DIR = OUT_ROOT / "blind"
BLIND_DIR.mkdir(parents=True, exist_ok=True)

PRIVATE_OUT = OUT_ROOT / "Promptune_v6_context_full_review_private_manifest_DO_NOT_SHARE.csv"
AUDIT_OUT = OUT_ROOT / "Promptune_v6_context_full_review_pack_audit.csv"

N_PACKS = 5
SEED = 20260814

DATA_TYPES = [
    "general_business",
    "community_messenger",
    "weak3_focus",
    "hard_negative",
    "complex_boundary",
]

def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")

print("=== V6 CONTEXT FULL HUMAN REVIEW PACK BUILD ===")

for p in [BASE, CALIB, AUDIT125]:
    if not p.exists():
        raise FileNotFoundError(p)

base = read_csv(BASE)
calib = read_csv(CALIB)
audit125 = read_csv(AUDIT125)

# ------------------------------------------------------------
# 1. Validate base
# ------------------------------------------------------------
if len(base) != 2000 or base["id"].nunique() != 2000:
    raise RuntimeError("BASE must contain 2000 unique IDs")

required_base = {"id", "prompt_text", "data_type", "human_missing_context"}
missing = required_base - set(base.columns)
if missing:
    raise RuntimeError(f"BASE missing columns: {sorted(missing)}")

# ------------------------------------------------------------
# 2. Validate Human 160
# ------------------------------------------------------------
if len(calib) != 160 or calib["id"].nunique() != 160:
    raise RuntimeError("Calibration file must contain 160 unique IDs")

if "calib_missing_context" not in calib.columns:
    raise RuntimeError("Calibration file missing calib_missing_context")

cv = pd.to_numeric(calib["calib_missing_context"], errors="coerce")
if cv.isna().any() or not cv.isin([0, 1]).all():
    raise RuntimeError("calib_missing_context must be complete 0/1")

calib_reviewed = (
    calib["calibration_status"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("reviewed")
    .sum()
)

if calib_reviewed != 160:
    raise RuntimeError(f"Calibration reviewed={calib_reviewed}/160")

# ------------------------------------------------------------
# 3. Validate Human 125
# ------------------------------------------------------------
if len(audit125) != 125 or audit125["id"].nunique() != 125:
    raise RuntimeError("Human Audit file must contain 125 unique IDs")

av = pd.to_numeric(audit125["review_context"], errors="coerce")
if av.isna().any() or not av.isin([0, 1]).all():
    raise RuntimeError("Audit125 review_context must be complete 0/1")

audit_reviewed = (
    audit125["review_status"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("reviewed")
    .sum()
)

if audit_reviewed != 125:
    raise RuntimeError(f"Audit125 reviewed={audit_reviewed}/125")

# ------------------------------------------------------------
# 4. Make sure the two Human sets are independent
# ------------------------------------------------------------
calib_ids = set(calib["id"])
audit_ids = set(audit125["id"])

overlap = calib_ids & audit_ids

if overlap:
    raise RuntimeError(
        f"Human160 and Human125 overlap: {len(overlap)} IDs"
    )

human_done_ids = calib_ids | audit_ids

unknown = human_done_ids - set(base["id"])

if unknown:
    raise RuntimeError(
        f"Human review files contain unknown IDs: {len(unknown)}"
    )

remaining = base[~base["id"].isin(human_done_ids)].copy()

print("Base rows              :", len(base))
print("Human Calibration done :", len(calib_ids))
print("Human Audit125 done    :", len(audit_ids))
print("Human overlap          :", len(overlap))
print("Remaining Context rows :", len(remaining))

# Expected 1715 when the 160 + 125 are independent.
if len(remaining) != 1715:
    print(
        f"WARNING: expected 1715 remaining, got {len(remaining)}. "
        "Packs will still be built from the actual remaining set."
    )

# ------------------------------------------------------------
# 5. Balanced round-robin assignment by data_type
# ------------------------------------------------------------
pack_ids = {i: [] for i in range(1, N_PACKS + 1)}
rng = random.Random(SEED)

for dtype in DATA_TYPES:
    ids = remaining.loc[
        remaining["data_type"] == dtype,
        "id"
    ].tolist()

    rng.shuffle(ids)

    # Rotate starting pack by data type to avoid systematic remainder bias.
    start = DATA_TYPES.index(dtype) % N_PACKS

    for j, row_id in enumerate(ids):
        pack_no = ((start + j) % N_PACKS) + 1
        pack_ids[pack_no].append(row_id)

# Handle any unexpected data types without silently dropping rows.
assigned = set(
    row_id
    for ids in pack_ids.values()
    for row_id in ids
)

unassigned = remaining[
    ~remaining["id"].isin(assigned)
]["id"].tolist()

for j, row_id in enumerate(unassigned):
    pack_no = (j % N_PACKS) + 1
    pack_ids[pack_no].append(row_id)

# ------------------------------------------------------------
# 6. Save blind packs
# ------------------------------------------------------------
base_idx = base.set_index("id")

audit_rows = []
private_rows = []

all_pack_ids = []

for pack_no in range(1, N_PACKS + 1):
    ids = pack_ids[pack_no]

    # Final shuffle within pack.
    rng.shuffle(ids)

    pack = base_idx.loc[ids].reset_index()

    blind = pack[["id", "prompt_text"]].copy()
    blind["review_context"] = ""
    blind["review_note"] = ""
    blind["review_status"] = "unreviewed"

    out_path = BLIND_DIR / f"Promptune_v6_context_human_pack_{pack_no:02d}.csv"
    blind.to_csv(out_path, index=False, encoding="utf-8-sig")

    all_pack_ids.extend(ids)

    counts = pack["data_type"].value_counts()

    row = {
        "pack": f"pack_{pack_no:02d}",
        "rows": len(pack),
    }

    for dtype in DATA_TYPES:
        row[dtype] = int(counts.get(dtype, 0))

    audit_rows.append(row)

    for _, r in pack.iterrows():
        private_rows.append({
            "pack": f"pack_{pack_no:02d}",
            "id": r["id"],
            "data_type": r["data_type"],
            "current_context_reference": int(r["human_missing_context"]),
        })

# ------------------------------------------------------------
# 7. Global integrity checks
# ------------------------------------------------------------
if len(all_pack_ids) != len(remaining):
    raise RuntimeError(
        f"Pack coverage mismatch: {len(all_pack_ids)}/{len(remaining)}"
    )

if len(set(all_pack_ids)) != len(all_pack_ids):
    raise RuntimeError("Duplicate IDs found across packs")

if set(all_pack_ids) != set(remaining["id"]):
    raise RuntimeError("Pack IDs do not exactly match remaining rows")

if set(all_pack_ids) & human_done_ids:
    raise RuntimeError("Packs overlap already Human-reviewed rows")

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(AUDIT_OUT, index=False, encoding="utf-8-sig")

pd.DataFrame(private_rows).to_csv(
    PRIVATE_OUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=== PACK AUDIT ===")
print(audit_df.to_string(index=False))

print()
print("Total pack rows       :", len(all_pack_ids))
print("Unique pack IDs       :", len(set(all_pack_ids)))
print("Already-human overlap :", len(set(all_pack_ids) & human_done_ids))

print()
print("=== REMAINING BY DATA TYPE ===")
print(
    remaining["data_type"]
    .value_counts()
    .reindex(DATA_TYPES)
    .fillna(0)
    .astype(int)
    .to_string()
)

print()
print("BLIND DIR:", BLIND_DIR)
print("PRIVATE  :", PRIVATE_OUT)
print("AUDIT    :", AUDIT_OUT)
print()
print("IMPORTANT: Share ONLY the five files inside the blind folder.")
print()
print("V6 CONTEXT FULL HUMAN REVIEW PACK BUILD: PASS")
