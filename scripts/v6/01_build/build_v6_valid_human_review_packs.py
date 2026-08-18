from pathlib import Path
import pandas as pd
import random

ROOT = Path(r"data\v6")
MASTER = ROOT / "Promptune_v6_valid_400_unlabeled_master_v1.csv"

OUT_ROOT = ROOT / "labeling" / "valid_human_review"
BLIND_DIR = OUT_ROOT / "blind"
BLIND_DIR.mkdir(parents=True, exist_ok=True)

PRIVATE_OUT = OUT_ROOT / "Promptune_v6_valid_human_review_private_manifest_DO_NOT_SHARE.csv"
AUDIT_OUT = OUT_ROOT / "Promptune_v6_valid_human_review_pack_audit.csv"

N_PACKS = 5
SEED = 20260814

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

print("=== V6 VALID 400 HUMAN REVIEW PACK BUILD ===")

if not MASTER.exists():
    raise FileNotFoundError(MASTER)

df = read_csv(MASTER)

if len(df) != 400:
    raise RuntimeError(f"Expected 400 rows, got {len(df)}")

if df["id"].nunique() != 400:
    raise RuntimeError("Valid IDs are not unique")

required = {"id", "prompt_text", "data_type"}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns: {sorted(missing)}")

# Confirm Human GT is still blank in the unlabeled master.
for col in LABEL_COLS:
    if col not in df.columns:
        raise RuntimeError(f"Missing label column: {col}")

    filled = df[col].notna() & df[col].astype(str).str.strip().ne("")
    if int(filled.sum()) != 0:
        raise RuntimeError(
            f"{col}: expected blank Human GT, found {int(filled.sum())} filled rows"
        )

# ------------------------------------------------------------
# Balanced round-robin by data_type
# ------------------------------------------------------------
pack_ids = {i: [] for i in range(1, N_PACKS + 1)}
rng = random.Random(SEED)

for dtype_idx, dtype in enumerate(DATA_TYPES):
    ids = df.loc[df["data_type"] == dtype, "id"].tolist()
    rng.shuffle(ids)

    start = dtype_idx % N_PACKS

    for j, row_id in enumerate(ids):
        pack_no = ((start + j) % N_PACKS) + 1
        pack_ids[pack_no].append(row_id)

assigned = {
    row_id
    for ids in pack_ids.values()
    for row_id in ids
}

# Never silently lose unexpected data types.
unassigned = df.loc[~df["id"].isin(assigned), "id"].tolist()

for j, row_id in enumerate(unassigned):
    pack_no = (j % N_PACKS) + 1
    pack_ids[pack_no].append(row_id)

df_idx = df.set_index("id")

audit_rows = []
private_rows = []
all_ids = []

for pack_no in range(1, N_PACKS + 1):
    ids = pack_ids[pack_no]
    rng.shuffle(ids)

    pack = df_idx.loc[ids].reset_index()

    blind = pack[["id", "prompt_text"]].copy()

    for col in LABEL_COLS:
        blind[col] = ""

    blind["review_note"] = ""
    blind["review_status"] = "unreviewed"

    out = BLIND_DIR / f"Promptune_v6_valid_human_pack_{pack_no:02d}.csv"
    blind.to_csv(out, index=False, encoding="utf-8-sig")

    all_ids.extend(ids)

    counts = pack["data_type"].value_counts()

    audit_row = {
        "pack": f"pack_{pack_no:02d}",
        "rows": len(pack),
    }

    for dtype in DATA_TYPES:
        audit_row[dtype] = int(counts.get(dtype, 0))

    audit_rows.append(audit_row)

    for _, r in pack.iterrows():
        private_rows.append({
            "pack": f"pack_{pack_no:02d}",
            "id": r["id"],
            "data_type": r["data_type"],
        })

# ------------------------------------------------------------
# Integrity
# ------------------------------------------------------------
if len(all_ids) != 400:
    raise RuntimeError(f"Pack coverage rows={len(all_ids)}, expected 400")

if len(set(all_ids)) != 400:
    raise RuntimeError("Duplicate IDs across Valid packs")

if set(all_ids) != set(df["id"]):
    raise RuntimeError("Valid pack IDs do not exactly cover master IDs")

audit = pd.DataFrame(audit_rows)
audit.to_csv(AUDIT_OUT, index=False, encoding="utf-8-sig")

pd.DataFrame(private_rows).to_csv(
    PRIVATE_OUT,
    index=False,
    encoding="utf-8-sig",
)

print()
print("=== PACK AUDIT ===")
print(audit.to_string(index=False))

print()
print("Total rows :", len(all_ids))
print("Unique IDs:", len(set(all_ids)))

print()
print("=== VALID MASTER BY DATA TYPE ===")
print(
    df["data_type"]
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
print("IMPORTANT: Share ONLY the five CSV files inside the blind folder.")
print()
print("V6 VALID 400 HUMAN REVIEW PACK BUILD: PASS")
