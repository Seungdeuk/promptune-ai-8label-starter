from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    request_id: str | None = Field(default=None, alias="requestId")
    text: str = Field(min_length=1, max_length=10000)

    model_config = {"populate_by_name": True}


class LabelDiagnosis(BaseModel):
    missing: bool
    probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)


class Diagnosis(BaseModel):
    task: LabelDiagnosis
    audience: LabelDiagnosis
    context: LabelDiagnosis
    format: LabelDiagnosis
    tone: LabelDiagnosis
    length: LabelDiagnosis
    constraint: LabelDiagnosis
    example: LabelDiagnosis


class NormalizationInfo(BaseModel):
    accepted: bool
    reason: str
    provider: str
    latency_ms: float = Field(alias="latencyMs")

    model_config = {"populate_by_name": True}


class AnalyzeResponse(BaseModel):
    request_id: str = Field(alias="requestId")
    original_text: str = Field(alias="originalText")
    normalized_text: str = Field(alias="normalizedText")
    normalization: NormalizationInfo
    diagnosis: Diagnosis
    diagnosis_provider: str = Field(alias="diagnosisProvider")
    model_version: str = Field(alias="modelVersion")
    latency_ms: float = Field(alias="latencyMs")

    model_config = {"populate_by_name": True}
