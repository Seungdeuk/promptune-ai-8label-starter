from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.core.config import get_settings
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.diagnosis.mock import MockDiagnosisService
from app.services.normalizer.mock import MockNormalizer
from app.services.pipeline import AnalysisPipeline


def build_pipeline() -> AnalysisPipeline:
    settings = get_settings()

    if settings.normalizer_provider == "mock":
        normalizer = MockNormalizer()
    else:
        from app.services.normalizer.clova import ClovaNormalizer

        normalizer = ClovaNormalizer(settings)

    if settings.diagnosis_provider == "mock":
        diagnosis = MockDiagnosisService()
    else:
        # Lazy import: mock/API development mode can start before PyTorch/Transformers are installed.
        from app.services.diagnosis.kcelectra import KcElectraDiagnosisService

        diagnosis = KcElectraDiagnosisService(settings)
    return AnalysisPipeline(settings, normalizer, diagnosis)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = build_pipeline()
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.post("/ai/analyze", response_model=AnalyzeResponse, response_model_by_alias=True)
async def analyze(payload: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    try:
        return await request.app.state.pipeline.analyze(payload.text, payload.request_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
