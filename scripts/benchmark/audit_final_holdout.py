from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


HOLDOUT_PATH = Path(
    r"data\final-holdout\Promptune_final_holdout_100.csv"
)

MASTER_PATH = Path(
    r"data\ground-truth-300\Promptune_ground_truth_300_master.csv"
)

OUTPUT_DIR = Path(
    r"artifacts\final-holdout-audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


LABELS = [
    "missing_task",
    "missing_audience",
    "missing_context",
    "missing_format",
    "missing_tone",
    "missing_length",
    "missing_constraint",
    "missing_example",
]


holdout = pd.read_csv(
    HOLDOUT_PATH,
    encoding="utf-8-sig",
    keep_default_na=False,
)

master = pd.read_csv(
    MASTER_PATH,
    encoding="utf-8-sig",
    keep_default_na=False,
)


# =========================================================
# 1. BASIC STRUCTURE
# =========================================================
print("=== 1. BASIC STRUCTURE ===")

print("rows:", len(holdout))
print("unique ids:", holdout["id"].nunique())
print("text filled:", int((holdout["text"].str.strip() != "").sum()))
print(
    "label cells filled:",
    int((holdout[LABELS] != "").sum().sum())
)
print(
    "natural:",
    int((holdout["cohort"] == "natural").sum())
)
print(
    "challenge:",
    int((holdout["cohort"] == "challenge").sum())
)
print(
    "families:",
    holdout["scenario_family"].nunique()
)

assert len(holdout) == 100
assert holdout["id"].nunique() == 100
assert (holdout["text"].str.strip() != "").all()
assert not (holdout[LABELS] != "").any().any()
assert (holdout["cohort"] == "natural").sum() == 70
assert (holdout["cohort"] == "challenge").sum() == 30
assert holdout["scenario_family"].nunique() == 20
assert holdout.groupby("scenario_family").size().eq(5).all()

expected_ids = {
    f"H{i:03d}"
    for i in range(1, 101)
}

assert set(holdout["id"]) == expected_ids

print("BASIC STRUCTURE: PASS")


# =========================================================
# 2. EXACT DUPLICATES
# =========================================================
print("\n=== 2. EXACT DUPLICATES ===")

external_exact = holdout[
    ["id", "text"]
].merge(
    master[["id", "text"]],
    on="text",
    how="inner",
    suffixes=("_holdout", "_master"),
)

internal_exact = holdout[
    holdout["text"].duplicated(keep=False)
][
    ["id", "text"]
].sort_values("text")

print(
    "holdout <-> master exact duplicates:",
    len(external_exact)
)

print(
    "holdout internal exact duplicate rows:",
    len(internal_exact)
)

if len(external_exact):
    print("\n[EXTERNAL EXACT]")
    print(external_exact.to_string(index=False))

if len(internal_exact):
    print("\n[INTERNAL EXACT]")
    print(internal_exact.to_string(index=False))

external_exact.to_csv(
    OUTPUT_DIR / "01_exact_duplicates_master.csv",
    index=False,
    encoding="utf-8-sig",
)

internal_exact.to_csv(
    OUTPUT_DIR / "02_exact_duplicates_internal.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 3. HOLDOUT VS EXISTING 300 SIMILARITY
# =========================================================
print("\n=== 3. HOLDOUT <-> MASTER SIMILARITY ===")

all_texts = pd.concat(
    [
        master["text"].astype(str),
        holdout["text"].astype(str),
    ],
    ignore_index=True,
)

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=1,
)

vectorizer.fit(all_texts)

master_vec = vectorizer.transform(
    master["text"].astype(str)
)

holdout_vec = vectorizer.transform(
    holdout["text"].astype(str)
)

sim_master = cosine_similarity(
    holdout_vec,
    master_vec,
)

best_master_idx = sim_master.argmax(axis=1)
best_master_score = sim_master.max(axis=1)

master_similarity_rows = []

for i in range(len(holdout)):
    j = int(best_master_idx[i])

    master_similarity_rows.append({
        "holdout_id": holdout.iloc[i]["id"],
        "scenario_family": holdout.iloc[i]["scenario_family"],
        "holdout_text": holdout.iloc[i]["text"],
        "master_id": master.iloc[j]["id"],
        "master_text": master.iloc[j]["text"],
        "similarity": float(best_master_score[i]),
    })

master_similarity = pd.DataFrame(
    master_similarity_rows
).sort_values(
    "similarity",
    ascending=False,
)

for threshold in [0.90, 0.80, 0.70, 0.60]:
    row_count = int(
        (master_similarity["similarity"] >= threshold).sum()
    )

    pair_count = int(
        (sim_master >= threshold).sum()
    )

    print(
        f">= {threshold:.2f}: "
        f"holdout rows={row_count}, "
        f"all pairs={pair_count}"
    )

print("\nTOP 20")
for _, row in master_similarity.head(20).iterrows():
    print("-" * 80)
    print(
        f'{row["holdout_id"]} <-> '
        f'{row["master_id"]} '
        f'sim={row["similarity"]:.4f}'
    )
    print("H:", row["holdout_text"])
    print("M:", row["master_text"])

master_similarity.to_csv(
    OUTPUT_DIR / "03_holdout_vs_master_similarity.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 4. INTERNAL HOLDOUT SIMILARITY
# =========================================================
print("\n=== 4. HOLDOUT INTERNAL SIMILARITY ===")

sim_internal = cosine_similarity(
    holdout_vec,
    holdout_vec,
)

# 자기 자신 제외
np.fill_diagonal(sim_internal, -1.0)

internal_pairs = []

for i in range(len(holdout)):
    for j in range(i + 1, len(holdout)):
        internal_pairs.append({
            "id_a": holdout.iloc[i]["id"],
            "family_a": holdout.iloc[i]["scenario_family"],
            "text_a": holdout.iloc[i]["text"],
            "id_b": holdout.iloc[j]["id"],
            "family_b": holdout.iloc[j]["scenario_family"],
            "text_b": holdout.iloc[j]["text"],
            "similarity": float(sim_internal[i, j]),
        })

internal_similarity = pd.DataFrame(
    internal_pairs
).sort_values(
    "similarity",
    ascending=False,
)

for threshold in [0.90, 0.80, 0.70, 0.60]:
    count = int(
        (internal_similarity["similarity"] >= threshold).sum()
    )

    print(
        f">= {threshold:.2f}: {count} pairs"
    )

print("\nTOP 20")
for _, row in internal_similarity.head(20).iterrows():
    print("-" * 80)
    print(
        f'{row["id_a"]} <-> '
        f'{row["id_b"]} '
        f'sim={row["similarity"]:.4f}'
    )
    print(
        f'family: {row["family_a"]} '
        f'<-> {row["family_b"]}'
    )
    print("A:", row["text_a"])
    print("B:", row["text_b"])

internal_similarity.head(100).to_csv(
    OUTPUT_DIR / "04_internal_similarity_top100.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 5. INPUT STYLE DISTRIBUTION
# =========================================================
print("\n=== 5. INPUT STYLE DISTRIBUTION ===")

style_counts = (
    holdout["input_style"]
    .value_counts()
    .rename_axis("input_style")
    .reset_index(name="count")
)

print(style_counts.to_string(index=False))

style_counts.to_csv(
    OUTPUT_DIR / "05_input_style_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 6. CHALLENGE TAG DISTRIBUTION
# =========================================================
print("\n=== 6. CHALLENGE TAG DISTRIBUTION ===")

challenge = holdout[
    holdout["cohort"] == "challenge"
].copy()

tags = []

for _, row in challenge.iterrows():
    for tag in row["challenge_tag"].split("|"):
        tag = tag.strip()

        if tag:
            tags.append({
                "id": row["id"],
                "tag": tag,
            })

tag_df = pd.DataFrame(tags)

tag_counts = (
    tag_df["tag"]
    .value_counts()
    .rename_axis("tag")
    .reset_index(name="count")
)

print(tag_counts.to_string(index=False))

tag_counts.to_csv(
    OUTPUT_DIR / "06_challenge_tag_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 7. FAMILY DISTRIBUTION
# =========================================================
print("\n=== 7. FAMILY DISTRIBUTION ===")

family_distribution = (
    holdout.groupby(
        ["scenario_family", "cohort"]
    )
    .size()
    .unstack(fill_value=0)
)

print(family_distribution.to_string())

family_distribution.to_csv(
    OUTPUT_DIR / "07_family_distribution.csv",
    encoding="utf-8-sig",
)


# =========================================================
# 8. TYPO CHECK FOR INTENDED TYPO ROWS
# =========================================================
print("\n=== 8. TYPO ROW REVIEW ===")

typo_rows = holdout[
    holdout["input_style"] == "typo"
][
    ["id", "challenge_tag", "text"]
]

print(typo_rows.to_string(index=False))

typo_rows.to_csv(
    OUTPUT_DIR / "08_typo_rows.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# FINAL
# =========================================================
print("\n=== AUDIT COMPLETE ===")
print("Output directory:", OUTPUT_DIR)

print(
    "Highest master similarity:",
    f'{master_similarity.iloc[0]["similarity"]:.4f}',
    master_similarity.iloc[0]["holdout_id"],
    "<->",
    master_similarity.iloc[0]["master_id"],
)

print(
    "Highest internal similarity:",
    f'{internal_similarity.iloc[0]["similarity"]:.4f}',
    internal_similarity.iloc[0]["id_a"],
    "<->",
    internal_similarity.iloc[0]["id_b"],
)
