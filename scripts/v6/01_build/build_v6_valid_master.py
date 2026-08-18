from pathlib import Path
import pandas as pd
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(r"data\v6")
PLAN = ROOT / "Promptune_v6_generation_plan_2400.csv"
GENERATED_VALID = ROOT / "generated_batches" / "valid"
TRAIN_MASTER = ROOT / "Promptune_v6_train_2000_unlabeled_master_v1.csv"

VALID_MASTER = ROOT / "Promptune_v6_valid_400_unlabeled_master_v1.csv"
AUDIT_OUTPUT = ROOT / "Promptune_v6_valid_400_audit_v1.csv"
WITHIN_VALID_OUTPUT = ROOT / "Promptune_v6_valid_400_near_duplicates_v1.csv"
TRAIN_VALID_LEAK_OUTPUT = ROOT / "Promptune_v6_train_valid_leakage_v1.csv"

EXPECTED_COUNTS = {
    "general_business": 120,
    "community_messenger": 100,
    "weak3_focus": 80,
    "hard_negative": 70,
    "complex_boundary": 30,
}

IMMUTABLE_COLUMNS = [
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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w가-힣 ]+", "", text)
    return text


def sval(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


print("=== V6 VALID 400 MASTER BUILD ===")

# ------------------------------------------------------------
# 1. Required paths
# ------------------------------------------------------------
for path, name in [
    (PLAN, "V6 Plan"),
    (GENERATED_VALID, "Generated Valid folder"),
    (TRAIN_MASTER, "Train 2000 Master"),
]:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

# ------------------------------------------------------------
# 2. Plan Valid 400
# ------------------------------------------------------------
plan = read_csv(PLAN)

plan_valid = (
    plan[
        plan["split"]
        .astype(str)
        .str.lower()
        .eq("valid")
    ]
    .copy()
    .reset_index(drop=True)
)

print("Plan Valid rows :", len(plan_valid))

if len(plan_valid) != 400:
    raise RuntimeError(f"Plan Valid expected 400, got {len(plan_valid)}")

# ------------------------------------------------------------
# 3. Filled files
# ------------------------------------------------------------
files = sorted(GENERATED_VALID.rglob("valid_*_batch_*_filled_v1.csv"))

print("Filled files    :", len(files))
for path in files:
    print("  ", path)

if len(files) != 6:
    raise RuntimeError(f"Expected 6 Valid filled files, got {len(files)}")

# ------------------------------------------------------------
# 4. Concatenate generated files
# ------------------------------------------------------------
frames = []

for path in files:
    df = read_csv(path)
    df["_source_file"] = str(path)
    frames.append(df)

generated = pd.concat(frames, ignore_index=True)

print()
print("Generated rows  :", len(generated))

# ------------------------------------------------------------
# 5. ID checks
# ------------------------------------------------------------
generated_ids = set(generated["id"])
expected_ids = set(plan_valid["id"])

duplicate_ids = generated[
    generated["id"].duplicated(keep=False)
].copy()

missing_ids = sorted(expected_ids - generated_ids)
extra_ids = sorted(generated_ids - expected_ids)

print()
print("=== ID CHECK ===")
print("unique IDs    :", generated["id"].nunique())
print("duplicate IDs :", len(duplicate_ids))
print("missing IDs   :", len(missing_ids))
print("extra IDs     :", len(extra_ids))

if len(duplicate_ids):
    print(
        duplicate_ids[["id", "_source_file"]]
        .sort_values("id")
        .to_string(index=False)
    )

if missing_ids:
    print("Missing example:", missing_ids[:20])

if extra_ids:
    print("Extra example:", extra_ids[:20])

# ------------------------------------------------------------
# 6. Metadata preservation
# ------------------------------------------------------------
plan_index = plan_valid.set_index("id")
metadata_errors = []

for _, row in generated.iterrows():
    row_id = row["id"]

    if row_id not in plan_index.index:
        continue

    original = plan_index.loc[row_id]

    for col in IMMUTABLE_COLUMNS:
        if sval(row.get(col, "")) != sval(original.get(col, "")):
            metadata_errors.append(
                {
                    "id": row_id,
                    "column": col,
                    "plan_value": sval(original.get(col, "")),
                    "generated_value": sval(row.get(col, "")),
                    "file": row["_source_file"],
                }
            )

print()
print("=== METADATA CHECK ===")
print("changed cells :", len(metadata_errors))

# ------------------------------------------------------------
# 7. Human GT contamination
# ------------------------------------------------------------
label_cols = [
    col for col in generated.columns
    if col.startswith("human_missing_")
]

human_gt_filled = 0

for col in label_cols:
    human_gt_filled += (
        generated[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

print()
print("=== HUMAN GT CHECK ===")
print("filled GT cells:", int(human_gt_filled))

# ------------------------------------------------------------
# 8. Prompt text checks
# ------------------------------------------------------------
generated["prompt_text"] = (
    generated["prompt_text"]
    .fillna("")
    .astype(str)
)

blank_prompt = (
    generated["prompt_text"]
    .str.strip()
    .eq("")
    .sum()
)

generated["_normalized"] = generated["prompt_text"].map(normalize)

exact_unique = generated["prompt_text"].nunique()
normalized_unique = generated["_normalized"].nunique()

print()
print("=== TEXT CHECK ===")
print("blank prompts  :", int(blank_prompt))
print("exact unique   :", int(exact_unique))
print("normalized uniq:", int(normalized_unique))

# ------------------------------------------------------------
# 9. Data type counts
# ------------------------------------------------------------
actual_counts = generated["data_type"].value_counts().to_dict()

print()
print("=== DATA TYPE COUNTS ===")

for key, expected in EXPECTED_COUNTS.items():
    actual = int(actual_counts.get(key, 0))
    print(f"{key:22s} expected={expected:3d} actual={actual:3d}")

# ------------------------------------------------------------
# 10. Train 2000
# ------------------------------------------------------------
train = read_csv(TRAIN_MASTER)
train["prompt_text"] = train["prompt_text"].fillna("").astype(str)

print()
print("Train rows      :", len(train))

if len(train) != 2000:
    raise RuntimeError(f"Expected Train Master 2000 rows, got {len(train)}")

# ------------------------------------------------------------
# 11. Scenario family overlap
# ------------------------------------------------------------
train_scenarios = {
    x for x in train["scenario_family"].fillna("").astype(str)
    if x.strip()
}
valid_scenarios = {
    x for x in generated["scenario_family"].fillna("").astype(str)
    if x.strip()
}
scenario_overlap = sorted(train_scenarios & valid_scenarios)

print()
print("=== SCENARIO FAMILY CHECK ===")
print("Train scenario families:", len(train_scenarios))
print("Valid scenario families:", len(valid_scenarios))
print("Overlap                :", len(scenario_overlap))

if scenario_overlap:
    print(scenario_overlap[:30])

# ------------------------------------------------------------
# 12. Exact Train/Valid overlap
# ------------------------------------------------------------
train_norm_set = set(train["prompt_text"].map(normalize))
exact_train_valid_overlap = int(
    generated["_normalized"].isin(train_norm_set).sum()
)

print()
print("=== TRAIN / VALID EXACT LEAKAGE ===")
print("exact overlap:", exact_train_valid_overlap)

# ------------------------------------------------------------
# 13. Valid internal near-duplicate audit
# Same pair_id in weak3 contrast pairs is intentionally exempted
# from suspicious-near-duplicate counts.
# ------------------------------------------------------------
valid_texts = generated["prompt_text"].tolist()

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=1,
)

valid_X = vectorizer.fit_transform(valid_texts)

valid_nn = NearestNeighbors(
    n_neighbors=min(11, len(generated)),
    metric="cosine",
)
valid_nn.fit(valid_X)

valid_distances, valid_indices = valid_nn.kneighbors(valid_X)

near_pairs = {}

for i in range(len(generated)):
    for distance, j in zip(valid_distances[i][1:], valid_indices[i][1:]):
        j = int(j)

        if i == j:
            continue

        a = min(i, j)
        b = max(i, j)
        sim = 1.0 - float(distance)

        if (a, b) not in near_pairs or sim > near_pairs[(a, b)]:
            near_pairs[(a, b)] = sim

near_rows = []
same_pair_exempted = 0

for (i, j), similarity in near_pairs.items():
    if similarity < 0.80:
        continue

    pair_i = sval(generated.iloc[i].get("pair_id", ""))
    pair_j = sval(generated.iloc[j].get("pair_id", ""))

    same_contrast_pair = (
        pair_i != ""
        and pair_i == pair_j
    )

    if same_contrast_pair:
        same_pair_exempted += 1
        continue

    near_rows.append(
        {
            "id_1": generated.iloc[i]["id"],
            "id_2": generated.iloc[j]["id"],
            "similarity": similarity,
            "data_type_1": generated.iloc[i]["data_type"],
            "data_type_2": generated.iloc[j]["data_type"],
            "pair_id_1": pair_i,
            "pair_id_2": pair_j,
            "text_1": generated.iloc[i]["prompt_text"],
            "text_2": generated.iloc[j]["prompt_text"],
        }
    )

near_df = pd.DataFrame(near_rows)

if len(near_df):
    near_df = near_df.sort_values("similarity", ascending=False)

near_df.to_csv(
    WITHIN_VALID_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)

print()
print("=== VALID NEAR DUPLICATES ===")
print("same contrast-pair exemptions:", same_pair_exempted)
print("suspicious similarity >= .80:", len(near_df))

if len(near_df):
    print(
        near_df[["id_1", "id_2", "similarity"]]
        .head(20)
        .to_string(index=False)
    )

# ------------------------------------------------------------
# 14. Train / Valid near leakage
# ------------------------------------------------------------
train_texts = train["prompt_text"].tolist()
combined_texts = train_texts + valid_texts

leak_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=1,
)

combined_X = leak_vectorizer.fit_transform(combined_texts)

train_X = combined_X[: len(train_texts)]
valid_X_for_leak = combined_X[len(train_texts):]

train_nn = NearestNeighbors(
    n_neighbors=1,
    metric="cosine",
)
train_nn.fit(train_X)

distances, indices = train_nn.kneighbors(valid_X_for_leak)

leak_rows = []
max_similarity = 0.0
max_pair = None

for valid_idx in range(len(valid_texts)):
    train_idx = int(indices[valid_idx, 0])
    similarity = 1.0 - float(distances[valid_idx, 0])

    if similarity > max_similarity:
        max_similarity = similarity
        max_pair = (valid_idx, train_idx)

    if similarity >= 0.80:
        leak_rows.append(
            {
                "valid_id": generated.iloc[valid_idx]["id"],
                "train_id": train.iloc[train_idx]["id"],
                "similarity": similarity,
                "valid_data_type": generated.iloc[valid_idx]["data_type"],
                "train_data_type": train.iloc[train_idx]["data_type"],
                "valid_text": valid_texts[valid_idx],
                "train_text": train_texts[train_idx],
            }
        )

leak_df = pd.DataFrame(leak_rows)

if len(leak_df):
    leak_df = leak_df.sort_values("similarity", ascending=False)

leak_df.to_csv(
    TRAIN_VALID_LEAK_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)

print()
print("=== TRAIN / VALID NEAR LEAKAGE ===")
print("max similarity   :", round(max_similarity, 4))
print("similarity >= .80:", len(leak_df))

if max_pair is not None:
    valid_idx, train_idx = max_pair
    print("Most similar pair:")
    print("  Valid:", generated.iloc[valid_idx]["id"])
    print("  Train:", train.iloc[train_idx]["id"])

if len(leak_df):
    print()
    print(
        leak_df[["valid_id", "train_id", "similarity"]]
        .head(20)
        .to_string(index=False)
    )

# ------------------------------------------------------------
# 15. Master build
# ------------------------------------------------------------
master = None

id_structure_ok = (
    len(generated) == 400
    and generated["id"].nunique() == 400
    and len(duplicate_ids) == 0
    and len(missing_ids) == 0
    and len(extra_ids) == 0
)

if id_structure_ok:
    prompt_map = (
        generated
        .set_index("id")["prompt_text"]
        .to_dict()
    )

    master = plan_valid.copy()
    master["prompt_text"] = master["id"].map(prompt_map)

    master.to_csv(
        VALID_MASTER,
        index=False,
        encoding="utf-8-sig",
    )

# ------------------------------------------------------------
# 16. Final checks
# Near-duplicate lists are audits, not automatic hard failures.
# ------------------------------------------------------------
checks = {
    "rows_400": len(generated) == 400,
    "unique_ids_400": generated["id"].nunique() == 400,
    "duplicate_ids_0": len(duplicate_ids) == 0,
    "missing_ids_0": len(missing_ids) == 0,
    "extra_ids_0": len(extra_ids) == 0,
    "metadata_changes_0": len(metadata_errors) == 0,
    "blank_prompt_0": int(blank_prompt) == 0,
    "normalized_unique_400": int(normalized_unique) == 400,
    "human_gt_filled_0": int(human_gt_filled) == 0,
    "scenario_family_overlap_0": len(scenario_overlap) == 0,
    "train_valid_exact_overlap_0": exact_train_valid_overlap == 0,
}

for key, expected in EXPECTED_COUNTS.items():
    checks[f"count_{key}"] = int(actual_counts.get(key, 0)) == expected

audit_rows = [
    {
        "check": name,
        "result": "PASS" if passed else "FAIL",
    }
    for name, passed in checks.items()
]

pd.DataFrame(audit_rows).to_csv(
    AUDIT_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)

print()
print("=== FINAL CHECK ===")

for name, passed in checks.items():
    print(
        f"{name:38s}:",
        "PASS" if passed else "FAIL",
    )

overall = all(checks.values())

print()

if overall:
    print("V6 VALID 400 MASTER BUILD: PASS")
    print("Master:", VALID_MASTER)
    print("Audit :", AUDIT_OUTPUT)
    print("Within:", WITHIN_VALID_OUTPUT)
    print("Leak  :", TRAIN_VALID_LEAK_OUTPUT)
else:
    print("V6 VALID 400 MASTER BUILD: FAIL")
    sys.exit(1)
