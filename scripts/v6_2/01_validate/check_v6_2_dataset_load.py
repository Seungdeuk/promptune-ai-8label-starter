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
TRAIN = ROOT / "data" / "v6_2" / "training_ready" / "Promptune_v6_2_train_2060_training_v1.csv"
VALID = ROOT / "data" / "v6" / "training_ready" / "Promptune_v6_valid_400_training_v1.csv"
MAX_LENGTH = 256

print("=== V6.2 KcELECTRA DATASET LOAD CHECK ===")
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
    print("gpu memory  :", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), "GB")
print()
print("=== TOKENIZER LOAD ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print("tokenizer   :", type(tokenizer).__name__)
print()
print("=== DATASET LOAD ===")
train_ds = PromptuneDataset(str(TRAIN), tokenizer, max_length=MAX_LENGTH)
valid_ds = PromptuneDataset(str(VALID), tokenizer, max_length=MAX_LENGTH)
print("train rows  :", len(train_ds))
print("valid rows  :", len(valid_ds))
assert len(train_ds) == 2060, len(train_ds)
assert len(valid_ds) == 400, len(valid_ds)
print()
print("=== SAMPLE CHECK ===")
sample = train_ds[-1]
print("keys        :", sorted(sample.keys()))
print("input shape :", tuple(sample["input_ids"].shape))
print("label shape :", tuple(sample["labels"].shape))
print("labels      :", sample["labels"].tolist())
assert tuple(sample["input_ids"].shape) == (MAX_LENGTH,)
assert tuple(sample["labels"].shape) == (NUM_LABELS,)
print()
print("V6.2 KcELECTRA DATASET LOAD CHECK: PASS")
