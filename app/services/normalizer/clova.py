from __future__ import annotations

from app.core.config import Settings
from app.services.normalizer.base import Normalizer, NormalizerOutput


class ClovaNormalizer(Normalizer):
    """HyperCLOVA X adapter boundary.

    API request/response contracts are intentionally not hard-coded in this starter.
    Confirm the exact current Naver Cloud endpoint and response schema, then implement here only.
    Keeping the provider contract isolated prevents external API changes from affecting the pipeline.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def normalize(self, text: str) -> NormalizerOutput:
        raise RuntimeError(
            "CLOVA adapter is not configured yet. Keep NORMALIZER_PROVIDER=mock until "
            "the exact current HyperCLOVA X API contract is confirmed and implemented in "
            "app/services/normalizer/clova.py."
        )
