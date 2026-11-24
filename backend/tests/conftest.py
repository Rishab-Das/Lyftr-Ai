"""Test harness: a fake LiteLLM provider, so the real LiteLLM code path (cache, exceptions,
usage objects) runs without any network access or real API keys."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import litellm
import pytest
from litellm import CustomLLM, ModelResponse

from app.config import Settings
from app.main import create_app

MODEL = "fake/test-model"


def _words(messages: list[dict]) -> int:
    return sum(len(str(m["content"]).split()) for m in messages)


class FakeProvider(CustomLLM):
    """Behaves like a provider. Behaviour is switched per API key from the tests."""

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.calls: list[str] = []  # api keys that reached the provider, in order
        self.errors: dict[str, Exception] = {}  # api_key -> exception to raise
        self.empty_keys: set[str] = set()
        self.delay = 0.15  # makes "cache hit is faster than a live call" measurable
        self.last_messages: list[dict] = []

    async def acompletion(self, model, messages, api_key=None, **kwargs):  # type: ignore[override]
        self.calls.append(api_key)
        self.last_messages = messages
        if api_key in self.errors:
            raise self.errors[api_key]
        await asyncio.sleep(self.delay)
        n = len(self.calls)
        last_user = [m for m in messages if m["role"] == "user"][-1]["content"]
        text = "" if api_key in self.empty_keys else f"[{n}] you said: {last_user} (msgs={len(messages)})"
        response = ModelResponse(
            model=model,
            choices=[{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        )
        prompt = _words(messages)
        completion = len(text.split())
        response.usage = litellm.Usage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        )
        return response


FAKE = FakeProvider()
litellm.custom_provider_map = [{"provider": "fake", "custom_handler": FAKE}]


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        llm_api_keys=["k1", "k2"],
        llm_model=MODEL,
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        allowed_origins=["http://localhost:3000"],
        enable_failure_simulation=True,
        key_cooldown_seconds=30.0,
    )
    values.update(overrides)
    return Settings(**values)


@asynccontextmanager
async def running_client(settings: Settings):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture(autouse=True)
def _reset_fake():
    FAKE.reset()
    yield


@pytest.fixture
def fake() -> FakeProvider:
    return FAKE


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
async def client(settings):
    async with running_client(settings) as c:
        yield c
