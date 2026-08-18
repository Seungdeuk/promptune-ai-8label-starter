from __future__ import annotations

import random
import argparse
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from app.core.labels import NUM_LABELS
from ml.dataset import PromptuneDataset
from ml.metrics import compute_metrics_from_logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="KcELECTRA base model name or local path")
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--output", default="artifacts/kcelectra-8label-v1")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # 모델의 classification head 생성 전에 seed 고정
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    train_ds = PromptuneDataset(args.train, tokenizer, args.max_length)
    valid_ds = PromptuneDataset(args.valid, tokenizer, args.max_length)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return compute_metrics_from_logits(np.asarray(logits), np.asarray(labels))

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
