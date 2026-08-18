from __future__ import annotations

import re
import time

from app.services.normalizer.base import Normalizer, NormalizerOutput


class MockNormalizer(Normalizer):
    """Deterministic starter normalizer. It does not pretend to correct Korean typos."""

    async def normalize(self, text: str) -> NormalizerOutput:
        started = time.perf_counter()
        normalized = re.sub(r"\s+", " ", text).strip()
        return NormalizerOutput(
            text=normalized,
            provider="mock",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
