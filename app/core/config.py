from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Promptune AI Service"
    app_env: str = "local"
    host: str = "0.0.0.0"
    port: int = 9000

    normalizer_provider: Literal["mock", "clova"] = "mock"
    diagnosis_provider: Literal["mock", "kcelectra"] = "mock"

    kc_model_path: str = ""
    threshold_file: str = ""
    max_sequence_length: int = 256
    device: str = "auto"

    clova_api_url: str = ""
    clova_api_key: str = ""
    clova_model: str = ""
    clova_timeout_seconds: float = 8.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
