from __future__ import annotations

import time
import uuid

from app.core.config import Settings
from app.core.labels import LABELS
from app.schemas.analyze import (
    AnalyzeResponse,
    Diagnosis,
    LabelDiagnosis,
    NormalizationInfo,
)
from app.services.diagnosis.base import DiagnosisService
from app.services.normalizer.base import Normalizer
from app.services.validator import validate_preservation


class AnalysisPipeline:
    def __init__(self, settings: Settings, normalizer: Normalizer, diagnosis: DiagnosisService) -> None:
        self.settings = settings
        self.normalizer = normalizer
        self.diagnosis_service = diagnosis

    async def analyze(self, text: str, request_id: str | None = None) -> AnalyzeResponse:
        started = time.perf_counter()
        rid = request_id or str(uuid.uuid4())

        normalized = await self.normalizer.normalize(text)
        validation = validate_preservation(text, normalized.text)
        normalized_text = normalized.text if validation.accepted else text

        diagnosis_output = self.diagnosis_service.predict(normalized_text)
        items = {
            label: LabelDiagnosis(
                missing=diagnosis_output.probabilities[label] >= diagnosis_output.thresholds[label],
                probability=diagnosis_output.probabilities[label],
                threshold=diagnosis_output.thresholds[label],
            )
            for label in LABELS
        }

        return AnalyzeResponse(
            requestId=rid,
            originalText=text,
            normalizedText=normalized_text,
            normalization=NormalizationInfo(
                accepted=validation.accepted,
                reason=validation.reason,
                provider=normalized.provider,
                latencyMs=normalized.latency_ms,
            ),
            diagnosis=Diagnosis(**items),
            diagnosisProvider=diagnosis_output.provider,
            modelVersion=diagnosis_output.model_version,
            latencyMs=(time.perf_counter() - started) * 1000,
        )
