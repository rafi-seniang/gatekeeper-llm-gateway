"""Typed settings. Every environment value the service reads is declared here, nowhere else."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../../secrets/secrets.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    app_name: str = "gatekeeper"
    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"

    # Comma-separated keys accepted in the X-API-Key header. Empty disables auth (local only).
    api_keys: SecretStr = SecretStr("")
    # Browser origins allowed to call the API (the lab frontend in development and preview)
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:4173"]

    # The upstream every allowed request is forwarded to: any OpenAI-compatible endpoint
    upstream_base_url: str = "http://localhost:8090/v1"
    upstream_api_key: SecretStr = Field(
        default=SecretStr("local"),
        validation_alias=AliasChoices("UPSTREAM_API_KEY", "DEEPSEEK_API_KEY"),
    )
    upstream_timeout_seconds: float = Field(default=120, gt=0)
    # The two models a request is routed between
    small_model: str = "Qwen/Qwen3.5-0.8B"
    strong_model: str = "Qwen/Qwen3.5-2B"

    # Decision endpoint: Jev /v1/systemone, or OpenAI /chat/completions with letter log-probabilities
    decision_protocol: Literal["systemone", "openai_logprobs"] = "systemone"
    decision_base_url: str = "http://localhost:8090"
    decision_api_key: SecretStr = Field(
        default=SecretStr("local"),
        validation_alias=AliasChoices("DECISION_API_KEY", "UPSTREAM_API_KEY", "DEEPSEEK_API_KEY"),
    )
    decision_model: str = "Qwen/Qwen3.5-2B"
    decision_timeout_seconds: float = Field(default=10, gt=0)
    # Expected difficulty (0 to 2) from which a request goes to the strong model
    strong_from_difficulty: float = Field(default=1.0, ge=0, le=2)

    # Which guard decides whether a request is a prompt injection
    injection_guard: Literal["classifier", "decision"] = "classifier"
    # A Hugging Face text classifier with an INJECTION label (Apache 2.0)
    injection_classifier_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    # Requests at or above this injection probability are blocked
    block_from_injection_probability: float = Field(default=0.5, gt=0, le=1)
    # cuda or cpu, for the classifier
    device: str = "cuda"

    # Exact-match cache of answers, keyed on the request as sent
    cache_max_entries: int = Field(default=1000, ge=0)
    cache_ttl_seconds: float = Field(default=3600, ge=0)
    # Recent requests kept for /v1/events and /v1/stats
    event_log_size: int = Field(default=500, gt=0)

    @property
    def api_key_list(self) -> list[str]:
        return [key.strip() for key in self.api_keys.get_secret_value().split(",") if key.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
