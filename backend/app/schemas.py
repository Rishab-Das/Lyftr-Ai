"""Request and response models (field names follow the assignment contract exactly)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_MESSAGE_CHARS = 4000
MAX_SYSTEM_PROMPT_CHARS = 8000


class ChatRequest(BaseModel):
    session_id: str = Field(
        ..., min_length=1, max_length=128, description="Client-chosen conversation id.", examples=["s-1a2b3c4d"]
    )
    message: str = Field(
        ..., description=f"User message, 1 to {MAX_MESSAGE_CHARS} characters after trimming.", examples=["Explain what an API gateway does."]
    )
    system_prompt: str | None = Field(
        default=None,
        max_length=MAX_SYSTEM_PROMPT_CHARS,
        description="Optional persona for the session. It is stored and reused on later turns.",
        examples=["You are a concise senior engineer."],
    )

    @field_validator("session_id")
    @classmethod
    def _session_id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session_id must not be blank")
        return value

    @field_validator("message")
    @classmethod
    def _message_length_after_trim(cls, value: str) -> str:
        trimmed = value.strip()
        if not 1 <= len(trimmed) <= MAX_MESSAGE_CHARS:
            raise ValueError(f"message must be 1 to {MAX_MESSAGE_CHARS} characters after trimming")
        return trimmed

    @field_validator("system_prompt")
    @classmethod
    def _blank_system_prompt_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ChatMetadata(BaseModel):
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"] = Field(
        description="LLM_CACHE when the answer came from the response cache, otherwise LIVE_INFERENCE."
    )
    model: str = Field(description="Model that produced the answer.")
    latency_ms: int = Field(description="Total server-side time for this request.")
    input_tokens: int = Field(description="Prompt tokens (original usage is replayed on cache hits).")
    output_tokens: int = Field(description="Completion tokens.")
    key_id: str | None = Field(description="Pool key that served the request, null on a cache hit.")
    fallback_attempts: int = Field(description="Keys that failed (429, quota or 5xx) before one succeeded.")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    metadata: dict[str, Any] | None = None


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
