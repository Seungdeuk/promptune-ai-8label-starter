from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosisOutput:
    probabilities: dict[str, float]
    thresholds: dict[str, float]
    provider: str
    model_version: str


class DiagnosisService(ABC):
    @abstractmethod
    def predict(self, text: str) -> DiagnosisOutput:
        raise NotImplementedError
