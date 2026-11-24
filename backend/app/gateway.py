"""LiteLLM gateway: key pool, automatic failover and response caching."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import litellm
from litellm.caching.caching import Cache

from .config import Settings
from .simulate import FailurePlan, build_simulated_error

log = logging.getLogger("app.gateway")

litellm.suppress_debug_info = True
litellm.telemetry = False
litellm.turn_off_message_logging = True  # never write user messages to LiteLLM's own logs


class AllProvidersUnavailable(Exception):
    """Every key in the pool failed (rate limit, quota, 5xx, timeout, ...)."""


class EmptyReplyError(Exception):
    """The provider answered 200 but with no text. Treated like a failed key."""


@dataclass(frozen=True)
class LLMResult:
    reply: str
    model: str
    input_tokens: int
    output_tokens: int
    key_id: str | None  # None when served from cache
    fallback_attempts: int  # keys that failed before this one answered, 0 on cache hit
    cache_hit: bool


class LLMGateway:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.llm_model
        self._keys = list(settings.llm_api_keys)
        self._timeout = settings.llm_timeout_seconds
        self._cooldown_seconds = settings.key_cooldown_seconds
        self._cooldown_until: dict[str, float] = {}
        self.cache_enabled = False
        self._cache_ttl = settings.cache_ttl_seconds

    # ---- setup -----------------------------------------------------------------

    def enable_cache(self) -> None:
        """Turn on LiteLLM's built-in in-memory cache (shared by every call below)."""
        litellm.cache = Cache(type="local", ttl=self._cache_ttl)
        self.cache_enabled = True

    async def warm_up(self) -> None:
        """Run one completion and one cache hit through LiteLLM's mock mode (no network).

        LiteLLM imports parts of its completion and cache-hit code lazily, which would make the
        first real request, and especially the first cache hit, noticeably slower.
        """
        model = "gpt-4o-mini"  # never contacted: mock_response short-circuits the provider call
        messages = [{"role": "user", "content": "__startup warm-up 7f3a91__"}]
        kwargs: dict[str, Any] = {
            "model": model, "messages": messages, "api_key": "warm-up",
            "mock_response": "ok", "caching": True, "num_retries": 0,
        }
        try:
            response = await litellm.acompletion(**kwargs)
            await litellm.cache.async_add_cache(response, model=model, messages=messages)
            await litellm.acompletion(**kwargs)
        except Exception as exc:  # warming is best effort
            log.debug("warm-up skipped: %s", type(exc).__name__)

    @property
    def keys_configured(self) -> int:
        return len(self._keys)

    def _pool(self) -> list[tuple[str, str]]:
        return [(f"key_{i}", key) for i, key in enumerate(self._keys, start=1)]

    def _ordered_pool(self, strict_order: bool) -> list[tuple[str, str]]:
        """Priority order key_1, key_2, ...

        Keys that just failed for real are tried last for a short cooldown so we do not
        keep hitting a rate-limited key first. Simulated requests keep strict order so the
        demo is deterministic.
        """
        pool = self._pool()
        if strict_order:
            return pool
        now = time.monotonic()
        healthy = [p for p in pool if self._cooldown_until.get(p[0], 0.0) <= now]
        cooling = [p for p in pool if p not in healthy]
        return healthy + cooling

    # ---- main entry point ------------------------------------------------------

    async def complete(self, messages: list[dict[str, str]], plan: FailurePlan | None = None) -> LLMResult:
        simulated = plan is not None
        # A simulation must exercise the failover path, so it skips the cache lookup.
        bypass_cache = simulated
        failed = 0

        for key_id, api_key in self._ordered_pool(strict_order=simulated):
            status = plan.status_for(key_id) if plan else None
            try:
                if status is not None:
                    raise build_simulated_error(status, self.model)
                response = await self._call(api_key, messages, bypass_cache)
                result = self._to_result(response, key_id, failed)
                if not result.cache_hit:
                    await self._write_through(response, messages)
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # any provider failure moves on to the next key
                failed += 1
                if isinstance(exc, EmptyReplyError):
                    bypass_cache = True  # do not re-read a bad cached entry on the next key
                elif status is None:
                    self._cooldown_until[key_id] = time.monotonic() + self._cooldown_seconds
                log.warning(
                    "%s failed (%s, status=%s): %s",
                    key_id,
                    type(exc).__name__,
                    getattr(exc, "status_code", None),
                    self._redact(str(exc))[:300],
                )

        raise AllProvidersUnavailable()

    # ---- internals -------------------------------------------------------------

    async def _call(self, api_key: str, messages: list[dict[str, str]], bypass_cache: bool) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "api_key": api_key,
            "caching": True,  # LiteLLM checks its cache first, then calls the provider
            "num_retries": 0,  # retrying is our job: next key, not the same one
            "timeout": self._timeout,
        }
        if bypass_cache:
            kwargs["cache"] = {"no-cache": True}
        return await litellm.acompletion(**kwargs)

    def _to_result(self, response: Any, key_id: str, failed: int) -> LLMResult:
        text = response.choices[0].message.content
        if not isinstance(text, str) or not text.strip():
            raise EmptyReplyError("provider returned an empty reply")

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        hidden = getattr(response, "_hidden_params", None) or {}
        cache_hit = bool(hidden.get("cache_hit"))
        return LLMResult(
            reply=text,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            key_id=None if cache_hit else key_id,
            fallback_attempts=0 if cache_hit else failed,
            cache_hit=cache_hit,
        )

    async def _write_through(self, response: Any, messages: list[dict[str, str]]) -> None:
        """Store the live answer in the cache before we reply.

        LiteLLM also writes it from a background task. Awaiting the write here as well means
        the entry exists before this response is sent, so an identical request arriving right
        after is guaranteed to be a hit. Same cache key, so the second write is harmless.
        """
        if litellm.cache is None:
            return
        try:
            await litellm.cache.async_add_cache(response, model=self.model, messages=messages)
        except Exception as exc:  # a cache problem must never fail the chat request
            log.warning("cache write-through failed: %s", type(exc).__name__)

    def _redact(self, text: str) -> str:
        for key in self._keys:
            if len(key) >= 6:
                text = text.replace(key, "[redacted]")
        return text
