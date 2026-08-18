from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# Conservative protected token set. It is intentionally stricter than a linguistic comparison.
URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
DATE_RE = re.compile(
    r"(?:\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}월\s*\d{1,2}일|\d{1,2}[./-]\d{1,2})"
)
TIME_RE = re.compile(r"(?:오전|오후)?\s*\d{1,2}(?::\d{2})?\s*시?(?:\s*\d{1,2}\s*분)?")
MONEY_RE = re.compile(r"(?:₩|\$)?\s*\d[\d,]*(?:\.\d+)?\s*(?:원|만원|억원|달러|USD|KRW)?", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?![A-Za-z])")


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reason: str


def _extract(regex: re.Pattern[str], text: str) -> Counter[str]:
    return Counter(match.group(0).strip() for match in regex.finditer(text))


def validate_preservation(original: str, candidate: str) -> ValidationResult:
    if not candidate.strip():
        return ValidationResult(False, "normalized text is empty")

    checks = (
        ("URL", URL_RE),
        ("email", EMAIL_RE),
        ("date", DATE_RE),
        ("time", TIME_RE),
        ("money", MONEY_RE),
        ("number", NUMBER_RE),
    )

    for name, regex in checks:
        before = _extract(regex, original)
        after = _extract(regex, candidate)
        if before != after:
            return ValidationResult(False, f"protected {name} tokens changed")

    return ValidationResult(True, "protected tokens preserved")
