from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from app.core.labels import LABELS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--probabilities", required=True, help=".npy shape [N, 8]")
    p.add_argument("--labels", required=True, help=".npy shape [N, 8]")
    p.add_argument("--output", default="artifacts/thresholds.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    probs = np.load(args.probabilities)
    y = np.load(args.labels).astype(int)
    if probs.shape != y.shape or probs.shape[1] != len(LABELS):
        raise ValueError(f"expected matching [N, {len(LABELS)}] arrays")

    thresholds = {}
    grid = np.arange(0.20, 0.81, 0.01)
    for i, label in enumerate(LABELS):
        scores = [f1_score(y[:, i], probs[:, i] >= t, zero_division=0) for t in grid]
        thresholds[label] = round(float(grid[int(np.argmax(scores))]), 2)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(thresholds, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
