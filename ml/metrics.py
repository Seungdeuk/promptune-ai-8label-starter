from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from app.core.labels import LABELS


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics_from_logits(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    probabilities = sigmoid(logits)
    predictions = (probabilities >= threshold).astype(int)
    truth = labels.astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, average=None, zero_division=0
    )

    result: dict[str, float] = {
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(truth, predictions, average="micro", zero_division=0)),
        "exact_match": float(np.mean(np.all(truth == predictions, axis=1))),
    }
    for i, label in enumerate(LABELS):
        result[f"{label}_precision"] = float(precision[i])
        result[f"{label}_recall"] = float(recall[i])
        result[f"{label}_f1"] = float(f1[i])
    return result
