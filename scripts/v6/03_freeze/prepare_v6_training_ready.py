from pathlib import Path
import pandas as pd

ROOT = Path(r"data\v6")

TRAIN_SRC = (
    ROOT / "labeling" / "frozen"
    / "Promptune_v6_train_2000_frozen_v1.csv"
)
VALID_SRC = (
    ROOT / "labeling" / "valid_human_review" / "frozen"
    / "Promptune_v6_valid_400_frozen_v1.csv"
)

OUT_DIR = ROOT / "training_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = OUT_DIR / "Promptune_v6_train_2000_training_v1.csv"
VALID_OUT = OUT_DIR / "Promptune_v6_valid_400_training_v1.csv"

LABELS = [
    "task",
    "audience",
    "context",
    "format",
    "tone",
    "length",
    "constraint",
    "example",
]

SOURCE_LABELS = [f"human_missing_{x}" for x in LABELS]
TARGET_LABELS = [f"missing_{x}" for x in LABELS]


def load_and_convert(src: Path, expected_rows: int, name: str) -> pd.DataFrame:
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src, encoding="utf-8-sig")

    if len(df) != expected_rows:
        raise RuntimeError(
            f"{name}: expected {expected_rows} rows, got {len(df)}"
        )

    if "id" not in df.columns or "prompt_text" not in df.columns:
        raise RuntimeError(f"{name}: id/prompt_text missing")

    if df["id"].nunique() != expected_rows:
        raise RuntimeError(f"{name}: IDs are not unique")

    blank = (
        df["prompt_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )
    if blank:
        raise RuntimeError(f"{name}: blank prompt_text={int(blank)}")

    out = pd.DataFrame({
        "id": df["id"],
        "text": df["prompt_text"].astype(str),
    })

    for src_col, dst_col in zip(SOURCE_LABELS, TARGET_LABELS):
        if src_col not in df.columns:
            raise RuntimeError(f"{name}: missing {src_col}")

        values = pd.to_numeric(df[src_col], errors="coerce")

        if values.isna().any() or not values.isin([0, 1]).all():
            raise RuntimeError(f"{name}: invalid labels in {src_col}")

        out[dst_col] = values.astype(int)

    return out


print("=== V6 TRAINING-READY DATA BUILD ===")

train = load_and_convert(TRAIN_SRC, 2000, "TRAIN")
valid = load_and_convert(VALID_SRC, 400, "VALID")

# Leakage checks after schema conversion.
id_overlap = set(train["id"]) & set(valid["id"])
text_overlap = (
    set(train["text"].astype(str).str.strip())
    & set(valid["text"].astype(str).str.strip())
)

if id_overlap:
    raise RuntimeError(f"Train/Valid ID overlap={len(id_overlap)}")

if text_overlap:
    raise RuntimeError(f"Train/Valid exact text overlap={len(text_overlap)}")

train.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
valid.to_csv(VALID_OUT, index=False, encoding="utf-8-sig")

print()
print("=== OUTPUT SCHEMA ===")
print(list(train.columns))

print()
print("=== TRAIN ===")
print("rows       :", len(train))
print("unique IDs :", train["id"].nunique())

print()
print("=== VALID ===")
print("rows       :", len(valid))
print("unique IDs :", valid["id"].nunique())

print()
print("=== LABEL DISTRIBUTION CHECK ===")
for label in LABELS:
    col = f"missing_{label}"
    print(
        f"{col:22s} "
        f"train={int(train[col].sum()):4d}/2000 "
        f"valid={int(valid[col].sum()):3d}/400"
    )

print()
print("Train/Valid ID overlap  :", len(id_overlap))
print("Train/Valid text overlap:", len(text_overlap))

print()
print("TRAIN:", TRAIN_OUT)
print("VALID:", VALID_OUT)
print()
print("V6 TRAINING-READY DATA BUILD: PASS")
