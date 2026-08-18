import pandas as pd
from pathlib import Path
import hashlib
import shutil

root = Path(r"data\final-holdout")

main_path = root / (
    "Promptune_human_review_all_H001_H100.csv"
)

rc1_dir = root / "rare-label-challenge"

rc1_review_path = rc1_dir / (
    "Promptune_rare_label_challenge_70_human_review.csv"
)

rc1_mapping_path = rc1_dir / (
    "Promptune_rare_label_challenge_70_review_mapping.csv"
)

rc2_dir = root / "rare-label-challenge-2"

rc2_review_path = rc2_dir / (
    "Promptune_rare_label_challenge_2_40_human_review.csv"
)

rc2_mapping_path = rc2_dir / (
    "Promptune_rare_label_challenge_2_40_mapping.csv"
)

freeze_dir = root / "final-benchmark-v1"

freeze_dir.mkdir(
    parents=True,
    exist_ok=True
)

labels = [
    "task",
    "audience",
    "context",
    "format",
    "tone",
    "length",
    "constraint",
    "example",
]

human_cols = [
    f"human_missing_{x}"
    for x in labels
]

# =========================================================
# 1. Read
# =========================================================

def read(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        keep_default_na=False
    )

main = read(main_path)

rc1_review = read(rc1_review_path)
rc1_map = read(rc1_mapping_path)

rc2_review = read(rc2_review_path)
rc2_map = read(rc2_mapping_path)

# =========================================================
# 2. Main 100
# =========================================================

main_final = main[
    [
        "id",
        "text",
        *human_cols,
        "human_review_note",
    ]
].copy()

main_final["benchmark_split"] = "main"

# =========================================================
# 3. RC1 70 ID 복원
# =========================================================

rc1 = rc1_review.merge(
    rc1_map,
    on="review_id",
    how="left",
    validate="one_to_one"
)

if rc1["id"].eq("").any():
    raise ValueError(
        "RC1 mapping failed."
    )

rc1_final = rc1[
    [
        "id",
        "text",
        *human_cols,
        "human_review_note",
    ]
].copy()

rc1_final["benchmark_split"] = "rare_challenge_1"

# =========================================================
# 4. RC2 40 ID 복원
# =========================================================

rc2 = rc2_review.merge(
    rc2_map,
    on="review_id",
    how="left",
    validate="one_to_one"
)

if rc2["id"].eq("").any():
    raise ValueError(
        "RC2 mapping failed."
    )

rc2_final = rc2[
    [
        "id",
        "text",
        *human_cols,
        "human_review_note",
    ]
].copy()

rc2_final["benchmark_split"] = "rare_challenge_2"

# =========================================================
# 5. Combine
# =========================================================

final = pd.concat(
    [
        main_final,
        rc1_final,
        rc2_final,
    ],
    ignore_index=True
)

if len(final) != 210:
    raise ValueError(
        f"Expected 210 rows, got {len(final)}"
    )

if final["id"].nunique() != 210:
    duplicates = final[
        final["id"].duplicated(
            keep=False
        )
    ]

    print(duplicates.to_string())

    raise ValueError(
        "Benchmark IDs are not unique."
    )

# =========================================================
# 6. Label validation
# =========================================================

for col in human_cols:

    values = (
        final[col]
        .astype(str)
        .str.strip()
    )

    if not set(values).issubset(
        {"0", "1"}
    ):
        raise ValueError(
            f"Invalid label: {col}"
        )

    if (values == "").any():
        raise ValueError(
            f"Empty label: {col}"
        )

    final[col] = (
        values.astype(int)
    )

# =========================================================
# 7. Human prefix 제거
#
# 모델 평가 스크립트가 사용하기 쉽게
# missing_task ... missing_example 으로 변경
# =========================================================

rename_map = {
    f"human_missing_{x}":
        f"missing_{x}"
    for x in labels
}

final = final.rename(
    columns=rename_map
)

final_cols = [
    "id",
    "benchmark_split",
    "text",
    *[
        f"missing_{x}"
        for x in labels
    ],
    "human_review_note",
]

final = final[
    final_cols
]

# =========================================================
# 8. 최종 분포 출력
# =========================================================

print(
    "=== FINAL BENCHMARK ==="
)

print(
    "Rows:",
    len(final)
)

print(
    "\n=== SPLITS ==="
)

print(
    final[
        "benchmark_split"
    ]
    .value_counts()
    .to_string()
)

print(
    "\n=== POSITIVE SUPPORT ==="
)

for label in labels:

    col = f"missing_{label}"

    positive = int(
        final[col].sum()
    )

    negative = int(
        len(final) - positive
    )

    print(
        f"{col:20s} "
        f"positive={positive:3d} "
        f"negative={negative:3d}"
    )

# =========================================================
# 9. Save
# =========================================================

output_path = (
    freeze_dir /
    "Promptune_final_benchmark_210_v1.csv"
)

final.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 10. SHA256
# =========================================================

sha256 = hashlib.sha256(
    output_path.read_bytes()
).hexdigest()

hash_path = (
    freeze_dir /
    "SHA256SUMS.txt"
)

hash_path.write_text(
    f"{sha256}  "
    f"{output_path.name}\n",
    encoding="utf-8"
)

print()
print(
    "=== FINAL BENCHMARK FROZEN ==="
)

print(
    "File:",
    output_path
)

print(
    "SHA256:",
    sha256
)

print(
    "Hash file:",
    hash_path
)
