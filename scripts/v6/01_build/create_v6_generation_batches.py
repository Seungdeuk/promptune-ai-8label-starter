from pathlib import Path
import math
import pandas as pd

SOURCE = Path(
    r"data\v6\Promptune_v6_generation_plan_2400.csv"
)

OUTPUT_ROOT = Path(
    r"data\v6\generation_batches"
)

BATCH_SIZE = 100

df = pd.read_csv(
    SOURCE,
    encoding="utf-8-sig"
)

required = [
    "id",
    "split",
    "data_type",
    "input_style",
    "generation_strategy",
    "business_domain",
    "scenario_family",
    "focus_element",
    "construction_target",
    "prompt_text",
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

print("=== V6 GENERATION BATCH CREATION ===")
print("Source rows:", len(df))
print("Batch size :", BATCH_SIZE)

manifest = []

for split in ["train", "valid"]:

    split_df = df[
        df["split"] == split
    ]

    for data_type in split_df[
        "data_type"
    ].unique():

        group = split_df[
            split_df["data_type"] == data_type
        ].copy()

        group = group.reset_index(drop=True)

        output_dir = (
            OUTPUT_ROOT
            / split
            / data_type
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        batch_count = math.ceil(
            len(group) / BATCH_SIZE
        )

        for batch_no in range(
            batch_count
        ):

            start = (
                batch_no
                * BATCH_SIZE
            )

            end = min(
                start + BATCH_SIZE,
                len(group)
            )

            batch = group.iloc[
                start:end
            ].copy()

            filename = (
                f"{split}_"
                f"{data_type}_"
                f"batch_{batch_no + 1:02d}.csv"
            )

            output_path = (
                output_dir
                / filename
            )

            batch.to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig"
            )

            manifest.append({
                "split": split,
                "data_type": data_type,
                "batch_no": batch_no + 1,
                "rows": len(batch),
                "first_id": batch.iloc[0]["id"],
                "last_id": batch.iloc[-1]["id"],
                "file": str(output_path),
            })

            print(
                f"[OK] {filename:<50}"
                f" rows={len(batch):3d}"
                f" ids={batch.iloc[0]['id']}"
                f"~{batch.iloc[-1]['id']}"
            )

manifest_df = pd.DataFrame(
    manifest
)

manifest_path = (
    OUTPUT_ROOT
    / "generation_batch_manifest.csv"
)

manifest_df.to_csv(
    manifest_path,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=== SUMMARY ===")

print(
    manifest_df.groupby(
        ["split", "data_type"]
    )["rows"].agg(
        ["sum", "count"]
    )
)

print()
print(
    "Total rows:",
    manifest_df["rows"].sum()
)

print(
    "Total batches:",
    len(manifest_df)
)

print()
print(
    "Manifest:",
    manifest_path
)

print()
print(
    "V6 GENERATION BATCH CREATION: PASS"
)
