from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from app.core.labels import LABELS, LABEL_COLUMNS, NUM_LABELS
from ml.dataset import PromptuneDataset

MODEL = "beomi/KcELECTRA-base"

TRAIN = Path(r"data\v6_1\training_ready\Promptune_v6_1_train_2052_training_v1.csv")
VALID = Path(r"data\v6\training_ready\Promptune_v6_valid_400_training_v1.csv")

MAX_LENGTH = 256

print("=== V6.1 KcELECTRA DATASET LOAD CHECK ===")
print()
print("MODEL      :", MODEL)
print("NUM_LABELS :", NUM_LABELS)
print("LABELS     :", LABELS)
print("COLUMNS    :", LABEL_COLUMNS)

print()
print("=== CUDA ===")
print("torch       :", torch.__version__)
print("cuda avail  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu         :", torch.cuda.get_device_name(0))
    print(
        "gpu memory  :",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB",
    )

print()
print("=== TOKENIZER LOAD ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print("tokenizer   :", tokenizer.__class__.__name__)

print()
print("=== DATASET LOAD ===")

train_ds = PromptuneDataset(str(TRAIN), tokenizer, MAX_LENGTH)
valid_ds = PromptuneDataset(str(VALID), tokenizer, MAX_LENGTH)

print("train rows  :", len(train_ds))
print("valid rows  :", len(valid_ds))

sample = train_ds[0]

print()
print("=== SAMPLE CHECK ===")
print("keys        :", sorted(sample.keys()))
print("input shape :", tuple(sample["input_ids"].shape))
print("label shape :", tuple(sample["labels"].shape))
print("labels      :", sample["labels"].tolist())

if len(train_ds) != 2052:
    raise RuntimeError(f"Train dataset expected 2052, got {len(train_ds)}")

if len(valid_ds) != 400:
    raise RuntimeError(f"Valid dataset expected 400, got {len(valid_ds)}")

if tuple(sample["input_ids"].shape) != (MAX_LENGTH,):
    raise RuntimeError(
        f"Unexpected input_ids shape: {tuple(sample['input_ids'].shape)}"
    )

if tuple(sample["labels"].shape) != (NUM_LABELS,):
    raise RuntimeError(
        f"Unexpected labels shape: {tuple(sample['labels'].shape)}"
    )

if not torch.all((sample["labels"] == 0) | (sample["labels"] == 1)):
    raise RuntimeError("Sample labels contain non-binary values")

print()
print("V6.1 KcELECTRA DATASET LOAD CHECK: PASS")
