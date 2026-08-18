from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import Settings
from app.core.labels import DEFAULT_THRESHOLD, LABELS, NUM_LABELS
from app.services.diagnosis.base import DiagnosisOutput, DiagnosisService


class KcElectraDiagnosisService(DiagnosisService):
    def __init__(self, settings: Settings) -> None:
        if not settings.kc_model_path:
            raise ValueError("KC_MODEL_PATH is required when DIAGNOSIS_PROVIDER=kcelectra")

        self.model_path = settings.kc_model_path
        self.max_length = settings.max_sequence_length
        self.device = self._resolve_device(settings.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        if self.model.config.num_labels != NUM_LABELS:
            raise ValueError(
                f"model num_labels={self.model.config.num_labels}, expected {NUM_LABELS}"
            )

        self.model.to(self.device)
        self.model.eval()
        self.thresholds = self._load_thresholds(settings.threshold_file)

    @staticmethod
    def _resolve_device(configured: str) -> torch.device:
        if configured != "auto":
            return torch.device(configured)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _load_thresholds(path: str) -> dict[str, float]:
        defaults = {label: DEFAULT_THRESHOLD for label in LABELS}
        if not path:
            return defaults
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {label: float(data.get(label, DEFAULT_THRESHOLD)) for label in LABELS}

    def predict(self, text: str) -> DiagnosisOutput:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = self.model(**encoded).logits[0]
            probs = torch.sigmoid(logits).detach().cpu().tolist()

        probabilities = {label: float(prob) for label, prob in zip(LABELS, probs)}
        return DiagnosisOutput(
            probabilities=probabilities,
            thresholds=self.thresholds,
            provider="kcelectra",
            model_version=Path(self.model_path).name,
        )
