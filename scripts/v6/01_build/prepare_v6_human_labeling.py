from pathlib import Path
import hashlib
import random
import pandas as pd

ROOT = Path(r"data\v6")

TRAIN_MASTER = ROOT / "Promptune_v6_train_2000_unlabeled_master_v1.csv"
VALID_MASTER = ROOT / "Promptune_v6_valid_400_unlabeled_master_v1.csv"

OUT = ROOT / "labeling" / "blind"
TRAIN_OUT = OUT / "train_primary"
VALID_OUT = OUT / "valid_double"

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

BLIND_COLS = [
    "id",
    "prompt_text",
    *LABEL_COLS,
    "review_note",
    "review_status",
]

SEED = 20260814
TRAIN_PACKS = 5


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_unlabeled(df: pd.DataFrame, name: str):
    for col in LABEL_COLS:
        if col not in df.columns:
            raise RuntimeError(f"{name}: missing label column {col}")

        filled = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        if filled:
            raise RuntimeError(
                f"{name}: {col} already has {filled} filled cells"
            )


def make_blind(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in LABEL_COLS:
        out[col] = ""

    if "review_note" not in out.columns:
        out["review_note"] = ""
    else:
        out["review_note"] = ""

    out["review_status"] = "unreviewed"

    return out[BLIND_COLS].copy()


print("=== V6 HUMAN GT BLIND PACK PREP ===")

for path, name in [
    (TRAIN_MASTER, "Train Master"),
    (VALID_MASTER, "Valid Master"),
]:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

train = read_csv(TRAIN_MASTER)
valid = read_csv(VALID_MASTER)

if len(train) != 2000:
    raise RuntimeError(f"Train expected 2000 rows, got {len(train)}")

if len(valid) != 400:
    raise RuntimeError(f"Valid expected 400 rows, got {len(valid)}")

if train["id"].nunique() != 2000:
    raise RuntimeError("Train IDs are not unique")

if valid["id"].nunique() != 400:
    raise RuntimeError("Valid IDs are not unique")

assert_unlabeled(train, "Train")
assert_unlabeled(valid, "Valid")

OUT.mkdir(parents=True, exist_ok=True)
TRAIN_OUT.mkdir(parents=True, exist_ok=True)
VALID_OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Train: five balanced primary packs of 400 rows.
# Balance by data_type, but do not expose data_type in blind outputs.
# ------------------------------------------------------------------
rng = random.Random(SEED)
pack_ids = [[] for _ in range(TRAIN_PACKS)]

for data_type, group in train.groupby("data_type", sort=True):
    ids = group["id"].tolist()
    rng.shuffle(ids)

    for idx, row_id in enumerate(ids):
        pack_ids[idx % TRAIN_PACKS].append(row_id)

train_index = train.set_index("id")

print()
print("=== TRAIN PRIMARY PACKS ===")

seen_train = set()

for i, ids in enumerate(pack_ids, start=1):
    rng.shuffle(ids)

    pack = train_index.loc[ids].reset_index()
    blind = make_blind(pack)

    path = TRAIN_OUT / f"v6_train_primary_pack_{i:02d}.csv"
    blind.to_csv(path, index=False, encoding="utf-8-sig")

    seen_train.update(ids)

    counts = pack["data_type"].value_counts().to_dict()

    print(
        f"pack_{i:02d}: rows={len(pack)} "
        f"data_type={counts}"
    )

if len(seen_train) != 2000:
    raise RuntimeError(
        f"Train primary coverage expected 2000 unique IDs, got {len(seen_train)}"
    )

# ------------------------------------------------------------------
# Valid: all 400 rows independently labeled by Reviewer A and B.
# Different row order only; same content.
# ------------------------------------------------------------------
valid_ids = valid["id"].tolist()

rng_a = random.Random(SEED + 1)
rng_b = random.Random(SEED + 2)

ids_a = valid_ids.copy()
ids_b = valid_ids.copy()

rng_a.shuffle(ids_a)
rng_b.shuffle(ids_b)

valid_index = valid.set_index("id")

reviewer_a = make_blind(valid_index.loc[ids_a].reset_index())
reviewer_b = make_blind(valid_index.loc[ids_b].reset_index())

path_a = VALID_OUT / "v6_valid_reviewer_A.csv"
path_b = VALID_OUT / "v6_valid_reviewer_B.csv"

reviewer_a.to_csv(path_a, index=False, encoding="utf-8-sig")
reviewer_b.to_csv(path_b, index=False, encoding="utf-8-sig")

print()
print("=== VALID DOUBLE LABELING ===")
print("reviewer_A rows:", len(reviewer_a))
print("reviewer_B rows:", len(reviewer_b))

# ------------------------------------------------------------------
# Private manifest: DO NOT give this file to labelers.
# It retains construction metadata for later stratified audit only.
# ------------------------------------------------------------------
manifest_cols = [
    c for c in [
        "id",
        "split",
        "data_type",
        "input_style",
        "generation_strategy",
        "business_domain",
        "scenario_family",
        "author_profile",
        "recipient_hint",
        "focus_element",
        "secondary_focus",
        "construction_target",
        "pair_id",
        "pair_role",
        "difficulty",
        "expected_length",
        "noise_level",
        "context_dependency",
    ]
    if c in train.columns
]

manifest_train = train[manifest_cols].copy()
manifest_valid = valid[[c for c in manifest_cols if c in valid.columns]].copy()

manifest = pd.concat(
    [manifest_train, manifest_valid],
    ignore_index=True,
)

manifest_path = OUT / "v6_labeling_private_manifest_DO_NOT_SHARE.csv"
manifest.to_csv(
    manifest_path,
    index=False,
    encoding="utf-8-sig",
)

# ------------------------------------------------------------------
# Freeze hashes
# ------------------------------------------------------------------
hash_path = OUT / "v6_unlabeled_master_sha256.txt"

hash_text = (
    f"{TRAIN_MASTER}\n"
    f"SHA256={sha256(TRAIN_MASTER)}\n\n"
    f"{VALID_MASTER}\n"
    f"SHA256={sha256(VALID_MASTER)}\n"
)

hash_path.write_text(hash_text, encoding="utf-8")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print()
print("=== FINAL CHECK ===")
print("train rows              :", len(train))
print("train primary unique IDs:", len(seen_train))
print("valid rows              :", len(valid))
print("valid reviewer A rows   :", len(reviewer_a))
print("valid reviewer B rows   :", len(reviewer_b))
print("blind columns           :", BLIND_COLS)
print("private manifest        :", manifest_path)
print("hash file               :", hash_path)

print()
print("V6 BLIND LABELING PACK PREP: PASS")
