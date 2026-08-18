from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.labels import LABEL_COLUMNS, LABELS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--thresholds", default="")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--output", default="artifacts/evaluation.json")
    p.add_argument("--save-probabilities", default="")
    p.add_argument("--save-labels", default="")
    return p.parse_args()


def load_thresholds(path: str) -> np.ndarray:
    if not path:
        return np.full(len(LABELS), 0.5, dtype=np.float32)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.array([float(data.get(label, 0.5)) for label in LABELS], dtype=np.float32)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.test)
    missing = {"text", *LABEL_COLUMNS}.difference(df.columns)
    if missing:
        raise ValueError(f"missing dataset columns: {sorted(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    thresholds = load_thresholds(args.thresholds)

    probs_all = []
    elapsed = []
    for text in df["text"].astype(str):
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            probs = torch.sigmoid(model(**encoded).logits[0]).cpu().numpy()
        elapsed.append((time.perf_counter() - started) * 1000)
        probs_all.append(probs)

    probs = np.vstack(probs_all)
    truth = df[list(LABEL_COLUMNS)].to_numpy(dtype=int)
    pred = (probs >= thresholds).astype(int)

    if args.save_probabilities:
        prob_path = Path(args.save_probabilities)
        prob_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(prob_path, probs)

    if args.save_labels:
        label_path = Path(args.save_labels)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(label_path, truth)

    precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, average=None, zero_division=0)
    result = {
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(truth, pred, average="micro", zero_division=0)),
        "exact_match": float(np.mean(np.all(truth == pred, axis=1))),
        "average_inference_ms": float(np.mean(elapsed)),
        "device": str(device),
        "labels": {},
    }
    for i, label in enumerate(LABELS):
        result["labels"][label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "threshold": float(thresholds[i]),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
