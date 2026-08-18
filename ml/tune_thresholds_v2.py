from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
)

from app.core.labels import LABELS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--probabilities", required=True, help=".npy shape [N, 8]")
    p.add_argument("--labels", required=True, help=".npy shape [N, 8]")
    p.add_argument("--output", default="artifacts/thresholds_v2.json")
    p.add_argument(
        "--min-precision",
        type=float,
        default=0.55,
        help="Precision floor. Default: 0.55",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    probs = np.load(args.probabilities)
    y = np.load(args.labels).astype(int)

    if probs.shape != y.shape or probs.shape[1] != len(LABELS):
        raise ValueError(f"expected matching [N, {len(LABELS)}] arrays")

    if not 0.0 <= args.min_precision <= 1.0:
        raise ValueError("--min-precision must be between 0 and 1")

    grid = np.arange(0.20, 0.81, 0.01)

    thresholds: dict[str, float] = {}
    report: dict[str, dict] = {}

    for i, label in enumerate(LABELS):
        candidates = []

        for t in grid:
            pred = (probs[:, i] >= t).astype(int)

            precision = precision_score(
                y[:, i],
                pred,
                zero_division=0,
            )
            recall = recall_score(
                y[:, i],
                pred,
                zero_division=0,
            )
            f2 = fbeta_score(
                y[:, i],
                pred,
                beta=2,
                zero_division=0,
            )

            candidates.append(
                {
                    "threshold": float(t),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f2": float(f2),
                }
            )

        eligible = [
            c for c in candidates
            if c["precision"] >= args.min_precision
        ]

        used_fallback = False

        if not eligible:
            eligible = candidates
            used_fallback = True

        # 우선순위:
        # 1. F2 최대
        # 2. Recall 최대
        # 3. Precision 최대
        # 4. 0.5에 가까운 threshold
        best = max(
            eligible,
            key=lambda c: (
                c["f2"],
                c["recall"],
                c["precision"],
                -abs(c["threshold"] - 0.5),
            ),
        )

        thresholds[label] = round(best["threshold"], 2)

        report[label] = {
            "threshold": round(best["threshold"], 2),
            "precision": best["precision"],
            "recall": best["recall"],
            "f2": best["f2"],
            "min_precision": args.min_precision,
            "fallback_used": used_fallback,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = out.with_name(out.stem + "_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== thresholds ===")
    print(json.dumps(thresholds, ensure_ascii=False, indent=2))

    print("\n=== validation report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
