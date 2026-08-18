from pathlib import Path
import pandas as pd
import random

ROOT = Path(r"data\v6\labeling")

BASE = ROOT / "final_candidate" / "Promptune_v6_train_2000_context_human_final_candidate_v1.csv"
CALIB = ROOT / "calibration" / "Promptune_v6_train_calibration_human_160.csv"

OUT_DIR = ROOT / "final_3label_human_review"
BLIND_DIR = OUT_DIR / "blind"
BLIND_DIR.mkdir(parents=True, exist_ok=True)

MASTER_OUT = BLIND_DIR / "Promptune_v6_train_task_audience_format_review_master.csv"
PRIVATE_OUT = OUT_DIR / "Promptune_v6_train_task_audience_format_reference_PRIVATE.csv"
AUDIT_OUT = OUT_DIR / "Promptune_v6_train_task_audience_format_review_audit.csv"

N_PACKS = 3
SEED = 20260814

TARGETS = {
    "task": "human_missing_task",
    "audience": "human_missing_audience",
    "format": "human_missing_format",
}

def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")

print("=== V6 FINAL 3-LABEL HUMAN REVIEW BUILD ===")

if not BASE.exists():
    raise FileNotFoundError(BASE)

if not CALIB.exists():
    raise FileNotFoundError(CALIB)

base = read_csv(BASE)
calib = read_csv(CALIB)

if len(base) != 2000 or base["id"].nunique() != 2000:
    raise RuntimeError("BASE must contain 2000 unique IDs")

if len(calib) != 160 or calib["id"].nunique() != 160:
    raise RuntimeError("Calibration must contain 160 unique IDs")

calib_ids = set(calib["id"])

# Validate labels
for label_col in TARGETS.values():
    vals = pd.to_numeric(base[label_col], errors="coerce")
    if vals.isna().any() or not vals.isin([0, 1]).all():
        raise RuntimeError(f"{label_col}: invalid binary labels")
    base[label_col] = vals.astype(int)

# Build target map:
# Review only current positive rows outside the 160 Human Calibration rows.
target_map = {}

for elem, label_col in TARGETS.items():
    ids = base.loc[
        (base[label_col] == 1)
        & (~base["id"].isin(calib_ids)),
        "id"
    ].tolist()

    for row_id in ids:
        target_map.setdefault(row_id, set()).add(elem)

base_idx = base.set_index("id")

rows = []

for row_id, elems in target_map.items():
    r = base_idx.loc[row_id]

    rows.append({
        "id": row_id,
        "prompt_text": r["prompt_text"],
        "check_task": 1 if "task" in elems else 0,
        "check_audience": 1 if "audience" in elems else 0,
        "check_format": 1 if "format" in elems else 0,
        "review_task": "",
        "review_audience": "",
        "review_format": "",
        "review_note": "",
        "review_status": "unreviewed",
    })

master = pd.DataFrame(rows)

# deterministic shuffle
master = master.sample(frac=1, random_state=SEED).reset_index(drop=True)

master.to_csv(
    MASTER_OUT,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------------
# Split into 3 balanced packs by number of required checks.
# Greedy assignment: rows with more checks first.
# ------------------------------------------------------------------
work = master.copy()
work["_check_count"] = (
    work["check_task"]
    + work["check_audience"]
    + work["check_format"]
)

work = work.sample(frac=1, random_state=SEED + 1)
work = work.sort_values(
    "_check_count",
    ascending=False,
    kind="stable"
)

pack_rows = {i: [] for i in range(1, N_PACKS + 1)}
pack_load = {i: 0 for i in range(1, N_PACKS + 1)}

rng = random.Random(SEED)

for _, row in work.iterrows():
    min_load = min(pack_load.values())
    candidates = [
        p for p, load in pack_load.items()
        if load == min_load
    ]
    pack_no = rng.choice(candidates)

    pack_rows[pack_no].append(row.drop(labels=["_check_count"]).to_dict())
    pack_load[pack_no] += int(row["_check_count"])

audit_rows = []

for pack_no in range(1, N_PACKS + 1):
    pack = pd.DataFrame(pack_rows[pack_no])

    if len(pack):
        pack = pack.sample(
            frac=1,
            random_state=SEED + 100 + pack_no
        ).reset_index(drop=True)

    path = BLIND_DIR / f"Promptune_v6_final3_human_pack_{pack_no:02d}.csv"
    pack.to_csv(path, index=False, encoding="utf-8-sig")

    audit_rows.append({
        "pack": f"pack_{pack_no:02d}",
        "rows": len(pack),
        "task_checks": int(pack["check_task"].sum()) if len(pack) else 0,
        "audience_checks": int(pack["check_audience"].sum()) if len(pack) else 0,
        "format_checks": int(pack["check_format"].sum()) if len(pack) else 0,
        "total_checks": (
            int(pack["check_task"].sum())
            + int(pack["check_audience"].sum())
            + int(pack["check_format"].sum())
            if len(pack) else 0
        ),
    })

# Private reference
private_rows = []

for _, r in master.iterrows():
    src = base_idx.loc[r["id"]]

    private_rows.append({
        "id": r["id"],
        "human_missing_task_current": int(src["human_missing_task"]),
        "human_missing_audience_current": int(src["human_missing_audience"]),
        "human_missing_format_current": int(src["human_missing_format"]),
        "check_task": int(r["check_task"]),
        "check_audience": int(r["check_audience"]),
        "check_format": int(r["check_format"]),
    })

pd.DataFrame(private_rows).to_csv(
    PRIVATE_OUT,
    index=False,
    encoding="utf-8-sig"
)

audit = pd.DataFrame(audit_rows)
audit.to_csv(
    AUDIT_OUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=== REVIEW TARGET COUNTS ===")
print("Task checks     :", int(master["check_task"].sum()))
print("Audience checks :", int(master["check_audience"].sum()))
print("Format checks   :", int(master["check_format"].sum()))
print("Unique rows     :", len(master))
print(
    "Total checks    :",
    int(
        master["check_task"].sum()
        + master["check_audience"].sum()
        + master["check_format"].sum()
    )
)

print()
print("=== PACK AUDIT ===")
print(audit.to_string(index=False))

print()
print("MASTER :", MASTER_OUT)
print("PRIVATE:", PRIVATE_OUT)
print("AUDIT  :", AUDIT_OUT)
print()
print("IMPORTANT: Share only files inside the blind folder.")
print()
print("V6 FINAL 3-LABEL HUMAN REVIEW BUILD: PASS")
