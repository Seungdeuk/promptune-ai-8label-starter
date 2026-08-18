from __future__ import annotations

import re

from app.core.labels import DEFAULT_THRESHOLD, LABELS
from app.services.diagnosis.base import DiagnosisOutput, DiagnosisService


class MockDiagnosisService(DiagnosisService):
    """Heuristic UI/API development aid only. Never use these values for model evaluation."""

    def predict(self, text: str) -> DiagnosisOutput:
        rules = {
            "task": bool(re.search(r"(작성|정리|요약|분석|비교|추천|만들|고쳐|설명|알려|추출)", text)),
            "audience": bool(re.search(r"(고객|팀장|임원|경영진|팀원|신입|사용자|대학생|직장인|개발자|마케터)", text)),
            "context": bool(re.search(r"(상황|배경|회의|장애|프로젝트|신제품|서비스|리뷰|실적|업데이트|이번 주|지난주)", text)),
            "format": bool(re.search(r"(표|목록|불릿|JSON|메일|이메일|공지|문구|보고서|Markdown|제목)", text, re.IGNORECASE)),
            "tone": bool(re.search(r"(정중|친근|차분|공감|격식|전문적|간결한 말투|말투|어조)", text)),
            "length": bool(re.search(r"(\d+\s*자|\d+\s*문단|\d+\s*개|이내|분량)", text)),
            "constraint": bool(re.search(r"(제외|금지|하지 마|바꾸지 마|변경하지 마|반드시|포함하지|추측하지|사용하지 마)", text)),
            "example": bool(re.search(r"(예:|예시|참고해서|같은 느낌|비슷한 형식)", text)),
        }

        probabilities = {label: (0.15 if rules[label] else 0.75) for label in LABELS}
        thresholds = {label: DEFAULT_THRESHOLD for label in LABELS}
        return DiagnosisOutput(
            probabilities=probabilities,
            thresholds=thresholds,
            provider="mock",
            model_version="mock-heuristic-v1",
        )
