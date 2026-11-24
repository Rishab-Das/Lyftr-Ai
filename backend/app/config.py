"""Runtime configuration, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Key pool. Key ids are key_1, key_2, ... in the order given here.
    llm_api_keys: list[str] = field(default_factory=list, repr=False)
    llm_model: str = "gemini/gemini-3.1-flash-lite"
    default_system_prompt: str = "You are a helpful, concise assistant."

    database_url: str = "sqlite:////data/app.db"
    allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    enable_failure_simulation: bool = False

    cache_ttl_seconds: int = 3600
    llm_timeout_seconds: float = 30.0
    key_cooldown_seconds: float = 30.0
    max_history_messages: int = 40

    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_api_keys=_split_csv(os.getenv("LLM_API_KEYS", "")),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model).strip() or cls.llm_model,
            default_system_prompt=os.getenv("DEFAULT_SYSTEM_PROMPT", cls.default_system_prompt).strip()
            or cls.default_system_prompt,
            database_url=os.getenv("DATABASE_URL", cls.database_url).strip() or cls.database_url,
            allowed_origins=_split_csv(os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")),
            enable_failure_simulation=_env_bool("ENABLE_FAILURE_SIMULATION", False),
            cache_ttl_seconds=_env_int("CACHE_TTL_SECONDS", 3600),
            llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 30.0),
            key_cooldown_seconds=_env_float("KEY_COOLDOWN_SECONDS", 30.0),
            max_history_messages=_env_int("MAX_HISTORY_MESSAGES", 40),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        )

    @property
    def sqlite_path(self) -> str:
        """Turn sqlite:////data/app.db or sqlite:///./app.db into a filesystem path."""
        url = self.database_url
        prefix = "sqlite:///"
        if not url.startswith(prefix):
            raise ValueError("DATABASE_URL must look like sqlite:////data/app.db (SQLite only)")
        return url[len(prefix):]
