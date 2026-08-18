from __future__ import annotations

from typing import Final

# IMPORTANT: this order is the single source of truth for training, inference, and API output.
LABELS: Final[tuple[str, ...]] = (
    "task",
    "audience",
    "context",
    "format",
    "tone",
    "length",
    "constraint",
    "example",
)

LABEL_COLUMNS: Final[tuple[str, ...]] = tuple(f"missing_{label}" for label in LABELS)

LABEL_DESCRIPTIONS: Final[dict[str, str]] = {
    "task": "무엇을 요청하는지",
    "audience": "결과물을 보는 대상",
    "context": "배경·상황정보",
    "format": "출력 형식",
    "tone": "어조·말투",
    "length": "글자 수·분량",
    "constraint": "제약 조건",
    "example": "예시·참고 형식",
}

NUM_LABELS: Final[int] = len(LABELS)
DEFAULT_THRESHOLD: Final[float] = 0.5
