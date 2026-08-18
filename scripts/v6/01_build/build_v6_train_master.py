from pathlib import Path
import pandas as pd
import numpy as np
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(r"data\v6")

PLAN = ROOT / "Promptune_v6_generation_plan_2400.csv"

GENERATED = (
    ROOT
    / "generated_batches"
    / "train"
)

OUTPUT = (
    ROOT
    / "Promptune_v6_train_2000_unlabeled_master_v1.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "Promptune_v6_train_2000_audit_v1.csv"
)

NEAR_OUTPUT = (
    ROOT
    / "Promptune_v6_train_2000_near_duplicates_v1.csv"
)

BENCH_OUTPUT = (
    ROOT
    / "Promptune_v6_train_vs_benchmark_leakage_v1.csv"
)

BENCHMARK = Path(
    r"data\final-holdout\final-benchmark-v1"
    r"\Promptune_final_benchmark_210_v1.csv"
)


EXPECTED_COUNTS = {
    "general_business": 580,
    "community_messenger": 500,
    "weak3_focus": 420,
    "hard_negative": 330,
    "complex_boundary": 170,
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


def normalize(text):
    text = str(text).lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w가-힣 ]+",
        "",
        text
    )

    return text


def read_csv(path):
    return pd.read_csv(
        path,
        encoding="utf-8-sig"
    )


print(
    "=== V6 TRAIN 2000 MASTER BUILD ==="
)


# ============================================================
# 1. 필수 경로 확인
# ============================================================

if not PLAN.exists():
    raise FileNotFoundError(
        f"Plan not found: {PLAN}"
    )

if not GENERATED.exists():
    raise FileNotFoundError(
        f"Generated folder not found: {GENERATED}"
    )


# ============================================================
# 2. 설계표 Train 2000 읽기
# ============================================================

plan = read_csv(
    PLAN
)

plan_train = (
    plan[
        plan["split"] == "train"
    ]
    .copy()
    .reset_index(drop=True)
)

if len(plan_train) != 2000:
    raise RuntimeError(
        f"Plan Train expected 2000, got {len(plan_train)}"
    )

print(
    f"Plan Train rows : {len(plan_train)}"
)


# ============================================================
# 3. 생성 완료 파일 검색
# ============================================================

files = sorted(
    GENERATED.rglob(
        "train_*_batch_*_filled_v1.csv"
    )
)

print(
    f"Filled files    : {len(files)}"
)

for path in files:
    print(
        "  ",
        path
    )


if len(files) != 22:
    print()
    print(
        "[WARNING] Expected 22 filled files."
    )
    print(
        "현재 발견:",
        len(files)
    )


# ============================================================
# 4. 22개 파일 읽기
# ============================================================

frames = []

for path in files:

    df = read_csv(
        path
    )

    df["_source_file"] = str(
        path
    )

    frames.append(
        df
    )


if not frames:
    raise RuntimeError(
        "No filled files found."
    )


generated = pd.concat(
    frames,
    ignore_index=True
)

print()
print(
    "Generated rows :",
    len(generated)
)


# ============================================================
# 5. ID 검사
# ============================================================

duplicate_ids = (
    generated[
        generated["id"].duplicated(
            keep=False
        )
    ]
)

generated_ids = set(
    generated["id"]
)

expected_ids = set(
    plan_train["id"]
)

missing_ids = sorted(
    expected_ids
    - generated_ids
)

extra_ids = sorted(
    generated_ids
    - expected_ids
)


print()
print(
    "=== ID CHECK ==="
)

print(
    "unique IDs     :",
    generated["id"].nunique()
)

print(
    "duplicate IDs  :",
    len(duplicate_ids)
)

print(
    "missing IDs    :",
    len(missing_ids)
)

print(
    "extra IDs      :",
    len(extra_ids)
)


if len(duplicate_ids) > 0:

    print()
    print(
        "=== DUPLICATE ID FILES ==="
    )

    print(
        duplicate_ids[
            [
                "id",
                "_source_file",
            ]
        ]
        .sort_values("id")
        .to_string(
            index=False
        )
    )


if missing_ids:

    print(
        "Missing example:",
        missing_ids[:20]
    )


if extra_ids:

    print(
        "Extra example:",
        extra_ids[:20]
    )


# ============================================================
# 6. 원본 설계 조건이 변경됐는지 검사
# ============================================================

plan_index = (
    plan_train
    .set_index("id")
)

metadata_errors = []


for _, row in generated.iterrows():

    row_id = row["id"]

    if row_id not in plan_index.index:
        continue

    original = plan_index.loc[
        row_id
    ]

    for col in IMMUTABLE_COLUMNS:

        source_value = (
            ""
            if pd.isna(
                row.get(col, "")
            )
            else str(
                row.get(col, "")
            ).strip()
        )

        original_value = (
            ""
            if pd.isna(
                original.get(col, "")
            )
            else str(
                original.get(col, "")
            ).strip()
        )

        if (
            source_value
            != original_value
        ):

            metadata_errors.append({
                "id": row_id,
                "column": col,
                "plan_value": original_value,
                "generated_value": source_value,
                "file": row[
                    "_source_file"
                ],
            })


print()
print(
    "=== METADATA CHECK ==="
)

print(
    "changed cells  :",
    len(metadata_errors)
)


# ============================================================
# 7. Human GT 오염 검사
# ============================================================

label_cols = [
    c
    for c in generated.columns
    if c.startswith(
        "human_missing_"
    )
]


human_label_filled = 0

for col in label_cols:

    human_label_filled += (
        generated[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )


print()
print(
    "=== HUMAN GT CHECK ==="
)

print(
    "filled GT cells:",
    int(
        human_label_filled
    )
)


# ============================================================
# 8. prompt_text 검사
# ============================================================

generated[
    "prompt_text"
] = (
    generated[
        "prompt_text"
    ]
    .fillna("")
    .astype(str)
)


blank_prompt = (
    generated[
        "prompt_text"
    ]
    .str.strip()
    .eq("")
    .sum()
)


exact_unique = (
    generated[
        "prompt_text"
    ]
    .nunique()
)


generated[
    "_normalized"
] = (
    generated[
        "prompt_text"
    ]
    .map(
        normalize
    )
)


normalized_unique = (
    generated[
        "_normalized"
    ]
    .nunique()
)


normalized_duplicates = (
    generated[
        generated[
            "_normalized"
        ]
        .duplicated(
            keep=False
        )
    ]
)


print()
print(
    "=== TEXT CHECK ==="
)

print(
    "blank prompts  :",
    blank_prompt
)

print(
    "exact unique   :",
    exact_unique
)

print(
    "normalized uniq:",
    normalized_unique
)

print(
    "normalized dups:",
    len(
        normalized_duplicates
    )
)


# ============================================================
# 9. 데이터 유형 수량 확인
# ============================================================

actual_counts = (
    generated[
        "data_type"
    ]
    .value_counts()
    .to_dict()
)


print()
print(
    "=== DATA TYPE COUNTS ==="
)

for key, expected in EXPECTED_COUNTS.items():

    actual = int(
        actual_counts.get(
            key,
            0
        )
    )

    print(
        f"{key:22s}"
        f" expected={expected:4d}"
        f" actual={actual:4d}"
    )


# ============================================================
# 10. Near Duplicate 후보 검사
#
# 전체 2,000 × 2,000을 무식하게 비교하지 않고
# 문자 n-gram TF-IDF + 최근접 이웃을 사용
# ============================================================

print()
print(
    "=== NEAR DUPLICATE AUDIT ==="
)


texts = (
    generated[
        "prompt_text"
    ]
    .tolist()
)


vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=1,
)

X = vectorizer.fit_transform(
    texts
)


neighbors = NearestNeighbors(
    n_neighbors=min(
        11,
        len(generated)
    ),
    metric="cosine",
)

neighbors.fit(
    X
)

distances, indices = (
    neighbors.kneighbors(
        X
    )
)


near_pairs = {}

for i in range(
    len(generated)
):

    for distance, j in zip(
        distances[i][1:],
        indices[i][1:]
    ):

        if i == j:
            continue

        a = min(
            i,
            int(j)
        )

        b = max(
            i,
            int(j)
        )

        similarity = (
            1.0
            - float(
                distance
            )
        )

        key = (
            a,
            b
        )

        if (
            key not in near_pairs
            or similarity
            > near_pairs[key]
        ):

            near_pairs[
                key
            ] = similarity


near_rows = []

for (
    i,
    j
), similarity in near_pairs.items():

    if similarity >= 0.80:

        near_rows.append({
            "id_1":
                generated.iloc[i][
                    "id"
                ],
            "id_2":
                generated.iloc[j][
                    "id"
                ],
            "similarity":
                similarity,
            "data_type_1":
                generated.iloc[i][
                    "data_type"
                ],
            "data_type_2":
                generated.iloc[j][
                    "data_type"
                ],
            "text_1":
                generated.iloc[i][
                    "prompt_text"
                ],
            "text_2":
                generated.iloc[j][
                    "prompt_text"
                ],
        })


near_df = (
    pd.DataFrame(
        near_rows
    )
)


if len(
    near_df
) > 0:

    near_df = (
        near_df
        .sort_values(
            "similarity",
            ascending=False
        )
    )


near_df.to_csv(
    NEAR_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


print(
    "TF-IDF >= 0.80:",
    len(
        near_df
    )
)


if len(
    near_df
) > 0:

    print()
    print(
        near_df[
            [
                "id_1",
                "id_2",
                "similarity",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


# ============================================================
# 11. Benchmark 210 Leakage 검사
# ============================================================

benchmark_exact_duplicates = 0
benchmark_near_count = 0
benchmark_max_similarity = 0.0
benchmark_near_df = pd.DataFrame()


if BENCHMARK.exists():

    benchmark = read_csv(
        BENCHMARK
    )

    benchmark_text_col = None

    for candidate in [
        "text",
        "prompt_text",
        "prompt",
    ]:

        if candidate in benchmark.columns:

            benchmark_text_col = (
                candidate
            )

            break


    if benchmark_text_col:

        bench_texts = (
            benchmark[
                benchmark_text_col
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        bench_norm = {
            normalize(
                text
            )
            for text in bench_texts
        }

        benchmark_exact_duplicates = sum(
            n in bench_norm
            for n in generated[
                "_normalized"
            ]
        )


        combined = (
            texts
            + bench_texts
        )

        leak_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
        )

        combined_X = (
            leak_vectorizer
            .fit_transform(
                combined
            )
        )

        train_X = (
            combined_X[
                :len(texts)
            ]
        )

        benchmark_X = (
            combined_X[
                len(texts):
            ]
        )

        leak_nn = NearestNeighbors(
            n_neighbors=1,
            metric="cosine",
        )

        leak_nn.fit(
            benchmark_X
        )

        leak_distances, leak_indices = (
            leak_nn.kneighbors(
                train_X
            )
        )

        leak_rows = []

        for i in range(
            len(texts)
        ):

            j = int(
                leak_indices[
                    i,
                    0
                ]
            )

            similarity = (
                1.0
                - float(
                    leak_distances[
                        i,
                        0
                    ]
                )
            )

            benchmark_max_similarity = max(
                benchmark_max_similarity,
                similarity
            )

            if similarity >= 0.80:

                leak_rows.append({
                    "train_id":
                        generated.iloc[i][
                            "id"
                        ],
                    "benchmark_row":
                        j,
                    "similarity":
                        similarity,
                    "train_text":
                        texts[i],
                    "benchmark_text":
                        bench_texts[j],
                })


        benchmark_near_df = (
            pd.DataFrame(
                leak_rows
            )
        )

        if len(
            benchmark_near_df
        ):

            benchmark_near_df = (
                benchmark_near_df
                .sort_values(
                    "similarity",
                    ascending=False
                )
            )

        benchmark_near_count = len(
            benchmark_near_df
        )

        benchmark_near_df.to_csv(
            BENCH_OUTPUT,
            index=False,
            encoding="utf-8-sig"
        )


print()
print(
    "=== BENCHMARK LEAKAGE ==="
)

print(
    "exact duplicates :",
    benchmark_exact_duplicates
)

print(
    "max similarity   :",
    round(
        benchmark_max_similarity,
        4
    )
)

print(
    "similarity >= .80:",
    benchmark_near_count
)


# ============================================================
# 12. Master를 설계표 순서대로 생성
# ============================================================

if (
    len(duplicate_ids) == 0
    and len(missing_ids) == 0
    and len(extra_ids) == 0
):

    prompt_map = (
        generated
        .set_index(
            "id"
        )[
            "prompt_text"
        ]
        .to_dict()
    )

    master = (
        plan_train
        .copy()
    )

    master[
        "prompt_text"
    ] = (
        master[
            "id"
        ]
        .map(
            prompt_map
        )
    )

    master.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

else:

    master = None


# ============================================================
# 13. PASS / FAIL 결정
# ============================================================

checks = {
    "rows_2000":
        len(
            generated
        ) == 2000,

    "unique_ids_2000":
        generated[
            "id"
        ].nunique() == 2000,

    "missing_ids_0":
        len(
            missing_ids
        ) == 0,

    "extra_ids_0":
        len(
            extra_ids
        ) == 0,

    "metadata_changes_0":
        len(
            metadata_errors
        ) == 0,

    "blank_prompt_0":
        int(
            blank_prompt
        ) == 0,

    "normalized_unique_2000":
        int(
            normalized_unique
        ) == 2000,

    "human_gt_filled_0":
        int(
            human_label_filled
        ) == 0,

    "benchmark_exact_duplicate_0":
        int(
            benchmark_exact_duplicates
        ) == 0,
}


for key, expected in EXPECTED_COUNTS.items():

    checks[
        f"count_{key}"
    ] = (
        int(
            actual_counts.get(
                key,
                0
            )
        )
        == expected
    )


audit_rows = []

for name, passed in checks.items():

    audit_rows.append({
        "check": name,
        "result":
            "PASS"
            if passed
            else "FAIL",
    })


pd.DataFrame(
    audit_rows
).to_csv(
    AUDIT_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "=== FINAL CHECK ==="
)

for name, passed in checks.items():

    print(
        f"{name:38s}:",
        "PASS"
        if passed
        else "FAIL"
    )


overall = all(
    checks.values()
)


print()

if overall:

    print(
        "V6 TRAIN 2000 MASTER BUILD: PASS"
    )

    print(
        "Master:",
        OUTPUT
    )

else:

    print(
        "V6 TRAIN 2000 MASTER BUILD: FAIL"
    )

    sys.exit(
        1
    )
