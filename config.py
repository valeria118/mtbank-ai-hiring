"""Централизованная конфигурация. Все значения читаются из .env."""
import logging
import os

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_HF_TOKEN_PREFIX = "hf_"

def _looks_like_hf_token(value: str) -> bool:
    return value.isascii() and value.startswith(_HF_TOKEN_PREFIX) and len(value) > len(_HF_TOKEN_PREFIX)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = "changeme"
    llm_model: str = "openai/gpt-oss-120b"

    # ASR
    whisper_model: str = "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = "ru"
    whisper_batch_size: int = 8

    diarization_backend: str = "auto"
    hf_token: str = ""

    @field_validator("hf_token")
    @classmethod
    def _drop_bogus_hf_token(cls, value: str) -> str:
        """Значение, не похожее на токен, гасим в пустую строку."""
        value = value.strip()
        if value and not _looks_like_hf_token(value):
            logger.warning(
                "config.hf_token_ignored: значение не похоже на токен HuggingFace "
                "(ожидается ASCII-строка вида %sXXXX) — работаем как без токена",
                _HF_TOKEN_PREFIX,
            )
            return ""
        return value

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    max_audio_duration_sec: int = 600
    api_root_path: str = ""

    metrics_exporter_enabled: bool = False
    metrics_port: int = 9100

    # Pipelines
    pipelines_api_key: str = "0p3n-w3bu!"
    openwebui_base_url: str = "http://openwebui:8080"
    openwebui_api_key: str = ""
    openwebui_uploads_dir: str = "/openwebui-data/uploads"

    # Хранилище результатов
    db_path: str = "data/calls.db"

    # Real-time (бонус)
    realtime_sample_rate: int = 16000
    realtime_window_sec: float = 2.5

    # Логирование
    log_level: str = "INFO"
    log_json: bool = True
    log_truncate_chars: int = 0

settings = Settings()

if not settings.hf_token and os.environ.get("HF_TOKEN", "").strip():
    os.environ.pop("HF_TOKEN", None)
