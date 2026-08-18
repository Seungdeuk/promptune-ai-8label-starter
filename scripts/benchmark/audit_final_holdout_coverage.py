import pandas as pd
from pathlib import Path

master_path = Path(
    r"data\final-holdout\Promptune_final_holdout_100.csv"
)

human_path = Path(
    r"data\final-holdout\Promptune_human_review_all_H001_H100.csv"
)

output_path = Path(
    r"data\final-holdout\Promptune_final_holdout_coverage_audit.csv"
)

human_labels = [
    "human_missing_task",
    "human_missing_audience",
    "human_missing_context",
    "human_missing_format",
    "human_missing_tone",
    "human_missing_length",
    "human_missing_constraint",
    "human_missing_example",
]

# ---------------------------------------------------------
# 1. 파일 읽기
# ---------------------------------------------------------
master = pd.read_csv(
    master_path,
    encoding="utf-8-sig",
    keep_default_na=False
)

human = pd.read_csv(
    human_path,
    encoding="utf-8-sig",
    keep_default_na=False
)

print("Master rows:", len(master))
print("Human rows :", len(human))

# ---------------------------------------------------------
# 2. 기본 무결성 검사
# ---------------------------------------------------------
if len(master) != 100:
    raise ValueError(
        f"Master must have 100 rows, got {len(master)}"
    )

if len(human) != 100:
    raise ValueError(
        f"Human file must have 100 rows, got {len(human)}"
    )

if master["id"].nunique() != 100:
    raise ValueError("Master IDs are not unique.")

if human["id"].nunique() != 100:
    raise ValueError("Human IDs are not unique.")

if set(master["id"]) != set(human["id"]):
    raise ValueError("Master/Human IDs do not match.")

# Human label 검증
for col in human_labels:
    values = set(human[col].astype(str))

    if not values.issubset({"0", "1"}):
        raise ValueError(
            f"{col} contains invalid values: {values}"
        )

# ---------------------------------------------------------
# 3. Master와 Human Review 병합
# ---------------------------------------------------------
human_keep = [
    "id",
    *human_labels,
    "human_review_note",
    "human_review_status",
]

merged = master.merge(
    human[human_keep],
    on="id",
    how="left",
    validate="one_to_one"
)

# text 일치 여부 검사
human_text = human[["id", "text"]].rename(
    columns={"text": "human_text"}
)

check = master[["id", "text"]].merge(
    human_text,
    on="id",
    how="left"
)

mismatch = check[
    check["text"] != check["human_text"]
]

if len(mismatch) > 0:
    print("\nTEXT MISMATCH:")
    print(
        mismatch[
            ["id", "text", "human_text"]
        ].to_string(index=False)
    )
    raise ValueError("Text mismatch detected.")

# ---------------------------------------------------------
# 4. 통합 Audit 파일 저장
# ---------------------------------------------------------
merged.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:", output_path)

# ---------------------------------------------------------
# 5. Human label 전체 분포
# ---------------------------------------------------------
print("\n=== HUMAN LABEL DISTRIBUTION ===")

for col in human_labels:
    s = pd.to_numeric(merged[col])

    pos = int(s.sum())
    neg = int(len(s) - pos)

    print(
        f"{col:28s} "
        f"positive={pos:3d} "
        f"negative={neg:3d} "
        f"rate={pos/len(s):6.1%}"
    )

# ---------------------------------------------------------
# 6. Cohort 분포
# ---------------------------------------------------------
if "cohort" in merged.columns:
    print("\n=== COHORT DISTRIBUTION ===")
    print(
        merged["cohort"]
        .value_counts(dropna=False)
        .to_string()
    )

# ---------------------------------------------------------
# 7. Input style 분포
# ---------------------------------------------------------
if "input_style" in merged.columns:
    print("\n=== INPUT STYLE DISTRIBUTION ===")
    print(
        merged["input_style"]
        .value_counts(dropna=False)
        .to_string()
    )

# ---------------------------------------------------------
# 8. Challenge tag 분포
# ---------------------------------------------------------
if "challenge_tag" in merged.columns:
    print("\n=== CHALLENGE TAG DISTRIBUTION ===")
    print(
        merged["challenge_tag"]
        .value_counts(dropna=False)
        .to_string()
    )

# ---------------------------------------------------------
# 9. Challenge tag별 실제 Human positive
# ---------------------------------------------------------
if "challenge_tag" in merged.columns:
    print(
        "\n=== HUMAN POSITIVES BY CHALLENGE TAG ==="
    )

    table = (
        merged
        .groupby("challenge_tag")[human_labels]
        .apply(
            lambda x:
            x.astype(int).sum()
        )
    )

    print(table.to_string())

# ---------------------------------------------------------
# 10. Scenario family별 실제 Human positive
# ---------------------------------------------------------
if "scenario_family" in merged.columns:
    print(
        "\n=== HUMAN POSITIVES BY SCENARIO FAMILY ==="
    )

    table = (
        merged
        .groupby("scenario_family")[human_labels]
        .apply(
            lambda x:
            x.astype(int).sum()
        )
    )

    print(table.to_string())

# ---------------------------------------------------------
# 11. Positive가 부족한 라벨 자동 표시
# ---------------------------------------------------------
print("\n=== COVERAGE WARNING ===")

for col in human_labels:
    pos = int(
        pd.to_numeric(merged[col]).sum()
    )

    if pos == 0:
        level = "NO POSITIVE"
    elif pos < 5:
        level = "CRITICAL"
    elif pos < 10:
        level = "LOW"
    elif pos < 15:
        level = "LIMITED"
    else:
        level = "OK"

    print(
        f"{col:28s} "
        f"positive={pos:3d} "
        f"coverage={level}"
    )

print("\nCoverage audit completed.")
