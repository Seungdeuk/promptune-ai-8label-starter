from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizerOutput:
    text: str
    provider: str
    latency_ms: float


class Normalizer(ABC):
    @abstractmethod
    async def normalize(self, text: str) -> NormalizerOutput:
        raise NotImplementedError
