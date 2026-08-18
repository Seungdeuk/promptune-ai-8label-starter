#!/usr/bin/env python
r"""
Promptune V6 3-seed error analysis.

Purpose
-------
Join the frozen Valid CSV with each official V6 submission package and export
row-level FP/FN cases for human review.

This script intentionally DOES NOT:
- retrain a model,
- touch the frozen Benchmark 210,
- add validation examples back into training data,
- choose a final production threshold automatically.

Expected official label order
-----------------------------
task, audience, context, format, tone, length, constraint, example

Example
-------
python .\scripts\v6\05_analysis\analyze_v6_errors.py `
  --valid .\data\v6\training_ready\Promptune_v6_valid_400_training_v1.csv `
  --submission .\artifacts\v6-e3-seed42-pc01-20260818-130235\v6-e3-seed42-pc01-20260818-130235-submission.zip `
  --submission .\artifacts\v6-e3-seed52-pc01-20260818-131136\v6-e3-seed52-pc01-20260818-131136-submission.zip `
  --submission .\artifacts\v6-e3-seed62-pc01-20260818-132016\v6-e3-seed62-pc01-20260818-132016-submission.zip `
  --output .\results\v6\error_analysis
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

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

PRIMARY_REVIEW_LABELS = ["audience", "format", "tone", "length", "example"]

EXPECTED_VALID_SHA256 = (
    "ACECD25DB4FF6B73D47F7AC447331213254C7D73F5F47E1ED3E60CDEB7E14A84"
)

TEXT_CANDIDATES = [
    "prompt_text",
    "text",
    "prompt",
    "input_text",
    "content",
]

ID_CANDIDATES = [
    "id",
    "prompt_id",
    "sample_id",
    "row_id",
    "example_id",
]

LABEL_PREFIX_CANDIDATES = [
    "human_missing_",
    "missing_",
    "label_",
    "",
]


@dataclass
class Submission:
    path: Path
    seed: int
    run_id: str
    git_commit: str
    working_tree_clean: bool
    valid_sha256: str
    thresholds: dict[str, float]
    probabilities: np.ndarray
    labels: np.ndarray
    report: dict[str, Any]


class PackageReader:
    def __init__(self, path: Path):
        self.path = path
        self._zip: zipfile.ZipFile | None = None

        if path.is_file() and path.suffix.lower() == ".zip":
            self.kind = "zip"
            self._zip = zipfile.ZipFile(path, "r")
        elif path.is_dir():
            self.kind = "dir"
        else:
            raise FileNotFoundError(
                f"Submission must be a ZIP or directory: {path}"
            )

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def read_bytes(self, name: str) -> bytes:
        if self.kind == "zip":
            assert self._zip is not None
            names = self._zip.namelist()

            if name in names:
                return self._zip.read(name)

            # Allow a package whose files are nested under one folder.
            matches = [n for n in names if Path(n).name == name]
            if len(matches) == 1:
                return self._zip.read(matches[0])

            raise FileNotFoundError(
                f"{name} not found in {self.path}"
            )

        direct = self.path / name
        if direct.exists():
            return direct.read_bytes()

        matches = list(self.path.rglob(name))
        if len(matches) == 1:
            return matches[0].read_bytes()

        raise FileNotFoundError(
            f"{name} not found under {self.path}"
        )

    def read_json(self, name: str) -> dict[str, Any]:
        return json.loads(self.read_bytes(name).decode("utf-8-sig"))

    def read_npy(self, name: str) -> np.ndarray:
        return np.load(io.BytesIO(self.read_bytes(name)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def pick_column(
    columns: Iterable[str],
    candidates: list[str],
    required: bool = True,
) -> str | None:
    colset = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in colset:
            return colset[candidate.lower()]

    if required:
        raise ValueError(
            "Could not detect required column. "
            f"Tried: {candidates}. Available: {list(columns)}"
        )

    return None


def detect_label_columns(df: pd.DataFrame) -> dict[str, str] | None:
    lower_map = {str(c).lower(): str(c) for c in df.columns}

    for prefix in LABEL_PREFIX_CANDIDATES:
        found: dict[str, str] = {}
        ok = True

        for label in LABELS:
            candidate = f"{prefix}{label}".lower()
            if candidate not in lower_map:
                ok = False
                break
            found[label] = lower_map[candidate]

        if ok:
            return found

    # Training-ready files occasionally preserve a "human_" naming layer.
    # Try a few complete patterns without mixing prefixes label-by-label.
    explicit_patterns = [
        {label: f"human_{label}" for label in LABELS},
        {label: f"y_{label}" for label in LABELS},
    ]

    for pattern in explicit_patterns:
        if all(v.lower() in lower_map for v in pattern.values()):
            return {
                label: lower_map[col.lower()]
                for label, col in pattern.items()
            }

    return None


def load_submission(path: Path) -> Submission:
    reader = PackageReader(path)

    try:
        metadata = reader.read_json("run-metadata.json")
        thresholds_raw = reader.read_json("thresholds-v1.json")
        report = reader.read_json("valid-tuned-v1.json")
        probs = reader.read_npy("valid-probabilities.npy")
        labels = reader.read_npy("valid-labels.npy")
    finally:
        reader.close()

    thresholds = {label: float(thresholds_raw[label]) for label in LABELS}

    return Submission(
        path=path,
        seed=int(metadata["seed"]),
        run_id=str(metadata["run_id"]),
        git_commit=str(metadata["git"]["commit_short"]),
        working_tree_clean=bool(metadata["git"]["working_tree_clean"]),
        valid_sha256=str(metadata["data"]["valid_sha256"]).upper(),
        thresholds=thresholds,
        probabilities=np.asarray(probs),
        labels=np.asarray(labels).astype(int),
        report=report,
    )


def validate_submission(
    submission: Submission,
    row_count: int,
) -> None:
    expected_shape = (row_count, len(LABELS))

    if submission.probabilities.shape != expected_shape:
        raise ValueError(
            f"Seed {submission.seed}: probabilities shape "
            f"{submission.probabilities.shape} != {expected_shape}"
        )

    if submission.labels.shape != expected_shape:
        raise ValueError(
            f"Seed {submission.seed}: labels shape "
            f"{submission.labels.shape} != {expected_shape}"
        )

    if not np.isfinite(submission.probabilities).all():
        raise ValueError(
            f"Seed {submission.seed}: probabilities contain NaN/Inf."
        )

    if not np.isin(submission.labels, [0, 1]).all():
        raise ValueError(
            f"Seed {submission.seed}: labels are not binary."
        )

    if not submission.working_tree_clean:
        raise ValueError(
            f"Seed {submission.seed}: official package reports dirty Git tree."
        )


def reconstruct_and_validate_metrics(submission: Submission) -> np.ndarray:
    threshold_vec = np.asarray(
        [submission.thresholds[label] for label in LABELS],
        dtype=float,
    )
    preds = (submission.probabilities >= threshold_vec).astype(int)

    for j, label in enumerate(LABELS):
        y = submission.labels[:, j]
        p = preds[:, j]

        precision = precision_score(y, p, zero_division=0)
        recall = recall_score(y, p, zero_division=0)
        f1 = f1_score(y, p, zero_division=0)

        expected = submission.report["labels"][label]

        for metric_name, actual, target in [
            ("precision", precision, float(expected["precision"])),
            ("recall", recall, float(expected["recall"])),
            ("f1", f1, float(expected["f1"])),
        ]:
            if not math.isclose(actual, target, rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(
                    f"Seed {submission.seed} / {label}: reconstructed "
                    f"{metric_name}={actual} != report={target}. "
                    "Check prediction comparison operator / row order."
                )

    macro = f1_score(
        submission.labels,
        preds,
        average="macro",
        zero_division=0,
    )
    micro = f1_score(
        submission.labels,
        preds,
        average="micro",
        zero_division=0,
    )
    exact = float(np.all(submission.labels == preds, axis=1).mean())

    expected_macro = float(submission.report["macro_f1"])
    expected_micro = float(submission.report["micro_f1"])
    expected_exact = float(submission.report["exact_match"])

    for name, actual, target in [
        ("macro_f1", macro, expected_macro),
        ("micro_f1", micro, expected_micro),
        ("exact_match", exact, expected_exact),
    ]:
        if not math.isclose(actual, target, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(
                f"Seed {submission.seed}: reconstructed {name}={actual} "
                f"!= report={target}"
            )

    return preds


def build_long_rows(
    df: pd.DataFrame,
    text_col: str,
    id_col: str | None,
    submissions: list[Submission],
    predictions: dict[int, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for submission in submissions:
        preds = predictions[submission.seed]

        for i in range(len(df)):
            row_id: Any = (
                df.iloc[i][id_col] if id_col is not None else i
            )
            text = str(df.iloc[i][text_col])

            for j, label in enumerate(LABELS):
                true = int(submission.labels[i, j])
                pred = int(preds[i, j])
                prob = float(submission.probabilities[i, j])
                threshold = float(submission.thresholds[label])

                if true == pred:
                    continue

                error_type = "FP" if true == 0 and pred == 1 else "FN"

                rows.append(
                    {
                        "seed": submission.seed,
                        "run_id": submission.run_id,
                        "git_commit": submission.git_commit,
                        "row_index": i,
                        "id": row_id,
                        "text": text,
                        "label": label,
                        "true": true,
                        "pred": pred,
                        "probability": prob,
                        "threshold": threshold,
                        "signed_margin": prob - threshold,
                        "abs_margin": abs(prob - threshold),
                        "error_type": error_type,
                        # Human-review columns are intentionally blank.
                        "review_status": "",
                        "error_category": "",
                        "gt_review": "",
                        "abstract_error_pattern": "",
                        "hard_negative_candidate": "",
                        "reviewer": "",
                        "notes": "",
                    }
                )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        ["label", "error_type", "seed", "abs_margin"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def build_summary(
    submissions: list[Submission],
    predictions: dict[int, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for submission in submissions:
        preds = predictions[submission.seed]

        for j, label in enumerate(LABELS):
            y = submission.labels[:, j]
            p = preds[:, j]

            tp = int(((y == 1) & (p == 1)).sum())
            fp = int(((y == 0) & (p == 1)).sum())
            fn = int(((y == 1) & (p == 0)).sum())
            tn = int(((y == 0) & (p == 0)).sum())

            rows.append(
                {
                    "seed": submission.seed,
                    "label": label,
                    "support_positive": int(y.sum()),
                    "threshold": submission.thresholds[label],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision_score(y, p, zero_division=0),
                    "recall": recall_score(y, p, zero_division=0),
                    "f1": f1_score(y, p, zero_division=0),
                }
            )

    return pd.DataFrame(rows)


def build_cross_seed_consensus(
    all_errors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify the same row/label being wrong in multiple seeds.

    This does not decide the cause. It only helps prioritize human review.
    """
    if all_errors.empty:
        return all_errors.copy()

    grouped = (
        all_errors.groupby(
            ["row_index", "id", "text", "label", "true", "error_type"],
            dropna=False,
        )
        .agg(
            error_seed_count=("seed", "nunique"),
            seeds=("seed", lambda s: ",".join(map(str, sorted(set(s))))),
            mean_probability=("probability", "mean"),
            min_probability=("probability", "min"),
            max_probability=("probability", "max"),
            mean_abs_margin=("abs_margin", "mean"),
        )
        .reset_index()
    )

    return grouped.sort_values(
        ["error_seed_count", "label", "mean_abs_margin"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def build_task_seed_comparison(
    df: pd.DataFrame,
    text_col: str,
    id_col: str | None,
    submissions: list[Submission],
    predictions: dict[int, np.ndarray],
) -> pd.DataFrame:
    by_seed = {s.seed: s for s in submissions}

    if 42 not in by_seed or 62 not in by_seed:
        return pd.DataFrame()

    j = LABELS.index("task")
    s42 = by_seed[42]
    s62 = by_seed[62]

    rows = []

    for i in range(len(df)):
        true = int(s42.labels[i, j])
        p42 = int(predictions[42][i, j])
        p62 = int(predictions[62][i, j])

        # Prioritize rows where the two seed predictions differ, plus all
        # Task errors made by either seed.
        if p42 == p62 and p42 == true:
            continue

        row_id: Any = df.iloc[i][id_col] if id_col is not None else i

        rows.append(
            {
                "row_index": i,
                "id": row_id,
                "text": str(df.iloc[i][text_col]),
                "true_task": true,
                "seed42_prob": float(s42.probabilities[i, j]),
                "seed42_threshold": s42.thresholds["task"],
                "seed42_pred": p42,
                "seed42_error": (
                    "" if p42 == true else ("FP" if true == 0 else "FN")
                ),
                "seed62_prob": float(s62.probabilities[i, j]),
                "seed62_threshold": s62.thresholds["task"],
                "seed62_pred": p62,
                "seed62_error": (
                    "" if p62 == true else ("FP" if true == 0 else "FN")
                ),
                "prediction_disagrees": int(p42 != p62),
                "review_status": "",
                "notes": "",
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["prediction_disagrees", "true_task"],
        ascending=[False, False],
    )


def write_review_protocol(path: Path) -> None:
    text = """# Promptune V6 Error Review Protocol

## 목적

이 단계는 Valid 400의 오류를 이용해 **모델의 실패 패턴을 이해하는 개발 분석**이다.
Final Benchmark 210은 이 단계에서 열거나 평가하지 않는다.

## 사람이 입력할 컬럼

- `review_status`: `reviewed`
- `error_category`:
  - `genuine_model_error`
  - `label_ambiguity`
  - `suspected_annotation_error`
  - `lexical_shortcut`
  - `missing_hard_negative`
  - `context_reasoning_failure`
  - `other`
- `gt_review`: `keep` / `change_suspected` / `uncertain`
- `abstract_error_pattern`: 개별 문장을 재현하지 말고 오류 유형을 추상적으로 기술
- `hard_negative_candidate`: `yes` / `no` / `uncertain`
- `reviewer`: 검토자
- `notes`: 추가 메모

## 우선 검토 대상

1. Audience
2. Tone
3. Length
4. Format
5. Example
6. Task — 특히 Seed42와 Seed62의 예측이 갈리는 사례

## Hard-negative leakage 방지

1. Valid 문장 자체를 Train에 추가하지 않는다.
2. Valid 문장의 단순 패러프레이즈/근접 재작성도 Train에 추가하지 않는다.
3. 오류 분석자는 가능한 한 `abstract_error_pattern`만 전달한다.
4. 신규 학습 예시는 가능하면 원본 Valid 문장을 보지 않은 다른 팀원이 작성한다.
5. AI로 신규 예시를 생성할 경우에도 원문이 아닌 추상화된 오류 카테고리만 입력한다.
6. 신규 예시는 사람의 라벨 검토를 거친다.
7. V6.1 개선에 Valid 오류 패턴을 사용했다면 Valid 400은 development set으로 취급한다.
8. 최종 일반화 성능은 untouched Final Benchmark 210에서만 보고한다.

## 해석 주의

- FP/FN 한 건을 보고 즉시 loss 또는 threshold를 변경하지 않는다.
- 반복되는 오류 패턴이 있는지 먼저 본다.
- `suspected_annotation_error`는 모델 오류와 분리해서 집계한다.
- rare label은 support가 작으므로 단일 사례의 영향이 크다.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Promptune V6 FP/FN cases for human review."
    )
    parser.add_argument(
        "--valid",
        required=True,
        type=Path,
        help="Frozen V6 Valid 400 training-ready CSV.",
    )
    parser.add_argument(
        "--submission",
        required=True,
        action="append",
        type=Path,
        help="Official submission ZIP or result directory. Repeat per Seed.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory.",
    )
    parser.add_argument(
        "--allow-sha-mismatch",
        action="store_true",
        help=(
            "Allow a Valid CSV whose SHA256 differs from the official V6 hash. "
            "Not recommended for official analysis."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    valid_path: Path = args.valid.resolve()
    output_dir: Path = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "by_label").mkdir(parents=True, exist_ok=True)

    if not valid_path.exists():
        raise FileNotFoundError(valid_path)

    actual_sha = sha256_file(valid_path)

    if actual_sha != EXPECTED_VALID_SHA256 and not args.allow_sha_mismatch:
        raise ValueError(
            "Valid SHA256 mismatch.\n"
            f"Expected: {EXPECTED_VALID_SHA256}\n"
            f"Actual  : {actual_sha}\n"
            "Stop official analysis or explicitly use --allow-sha-mismatch "
            "for non-official debugging."
        )

    df = pd.read_csv(valid_path, encoding="utf-8-sig")

    text_col = pick_column(df.columns, TEXT_CANDIDATES, required=True)
    id_col = pick_column(df.columns, ID_CANDIDATES, required=False)
    label_columns = detect_label_columns(df)

    submissions = [
        load_submission(path.resolve())
        for path in args.submission
    ]
    submissions = sorted(submissions, key=lambda s: s.seed)

    if len({s.seed for s in submissions}) != len(submissions):
        raise ValueError("Duplicate Seed submission detected.")

    if len(submissions) < 2:
        raise ValueError("Provide at least two submissions.")

    for submission in submissions:
        validate_submission(submission, len(df))

    # Every seed must have the exact same Ground Truth matrix.
    reference_labels = submissions[0].labels
    for submission in submissions[1:]:
        if not np.array_equal(reference_labels, submission.labels):
            raise ValueError(
                f"Seed {submission.seed}: valid-labels.npy differs from "
                f"Seed {submissions[0].seed}."
            )

    # If the CSV carries labels, validate row order and label contract against
    # the evaluator-exported Ground Truth. This is the key text<->NPY alignment
    # safety check.
    if label_columns is not None:
        csv_labels = np.column_stack(
            [
                pd.to_numeric(df[label_columns[label]], errors="raise")
                .astype(int)
                .to_numpy()
                for label in LABELS
            ]
        )

        if not np.array_equal(csv_labels, reference_labels):
            mismatch_count = int((csv_labels != reference_labels).sum())
            raise ValueError(
                "Valid CSV label columns do not align with valid-labels.npy. "
                f"Binary-cell mismatches: {mismatch_count}. "
                "Do NOT export text-level errors until row order is verified."
            )
    else:
        raise ValueError(
            "Could not detect all 8 label columns in Valid CSV. "
            "The script refuses to join text to NPY without verifying row "
            "alignment. Add the frozen label columns or extend "
            "detect_label_columns() for the actual schema."
        )

    predictions: dict[int, np.ndarray] = {}

    for submission in submissions:
        if submission.valid_sha256 != actual_sha:
            raise ValueError(
                f"Seed {submission.seed}: package Valid SHA256 "
                f"{submission.valid_sha256} != current CSV {actual_sha}"
            )

        predictions[submission.seed] = reconstruct_and_validate_metrics(
            submission
        )

    all_errors = build_long_rows(
        df=df,
        text_col=text_col,
        id_col=id_col,
        submissions=submissions,
        predictions=predictions,
    )

    summary = build_summary(submissions, predictions)
    consensus = build_cross_seed_consensus(all_errors)
    task_compare = build_task_seed_comparison(
        df=df,
        text_col=text_col,
        id_col=id_col,
        submissions=submissions,
        predictions=predictions,
    )

    # Full error export
    all_errors.to_csv(
        output_dir / "all_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        output_dir / "error_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    consensus.to_csv(
        output_dir / "cross_seed_consensus_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    task_compare.to_csv(
        output_dir / "task_seed42_vs_seed62.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Review-priority labels only.
    priority = all_errors[
        all_errors["label"].isin(PRIMARY_REVIEW_LABELS)
    ].copy()

    priority.to_csv(
        output_dir / "priority_rare_label_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # One file per label is easier for distributed human review.
    for label in LABELS:
        label_df = all_errors[all_errors["label"] == label].copy()
        label_df.to_csv(
            output_dir / "by_label" / f"{label}_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

    write_review_protocol(output_dir / "ERROR_REVIEW_PROTOCOL.md")

    run_info = {
        "valid_csv": str(valid_path),
        "valid_sha256": actual_sha,
        "rows": len(df),
        "text_column": text_col,
        "id_column": id_col,
        "label_columns": label_columns,
        "labels": LABELS,
        "submissions": [
            {
                "seed": s.seed,
                "run_id": s.run_id,
                "git_commit": s.git_commit,
                "submission": str(s.path),
            }
            for s in submissions
        ],
        "error_rows": int(len(all_errors)),
        "priority_error_rows": int(len(priority)),
    }

    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("PROMPTUNE V6 ERROR EXTRACTION COMPLETE")
    print("=" * 72)
    print(f"Valid rows            : {len(df)}")
    print(f"Valid SHA256          : {actual_sha}")
    print(f"Text column           : {text_col}")
    print(f"ID column             : {id_col}")
    print(f"Seeds                 : {[s.seed for s in submissions]}")
    print(f"Total FP/FN rows      : {len(all_errors)}")
    print(f"Priority rare errors  : {len(priority)}")
    print()
    print("Review first:")
    print(f"  {output_dir / 'priority_rare_label_errors.csv'}")
    print(f"  {output_dir / 'task_seed42_vs_seed62.csv'}")
    print()
    print("Protocol:")
    print(f"  {output_dir / 'ERROR_REVIEW_PROTOCOL.md'}")
    print()
    print("IMPORTANT:")
    print("  Do not add Valid rows or close paraphrases to Train.")
    print("  Do not evaluate Final Benchmark 210 in this step.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
