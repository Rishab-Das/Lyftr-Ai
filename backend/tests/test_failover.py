import litellm
import pytest

from app.simulate import parse_failure_header

from .conftest import make_settings, running_client


async def chat(client, session, message, simulate=None):
    headers = {"X-Simulate-Failure": simulate} if simulate else {}
    return await client.post("/chat", json={"session_id": session, "message": message}, headers=headers)


@pytest.mark.parametrize("status", [429, 500, 502, 503])
async def test_simulated_key_1_failure_falls_over_to_key_2(client, fake, status):
    r = await chat(client, f"s-{status}", f"hello {status}", simulate=f"key_1={status}")
    assert r.status_code == 200
    meta = r.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert r.json()["reply"].startswith("[")  # a real answer from the provider, not an error text
    assert fake.calls == ["k2"]  # key_1 was never actually called


async def test_all_keys_failing_returns_clean_503_and_stores_nothing(client):
    r = await chat(client, "s-503", "hello", simulate="*=429")
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}
    assert (await client.get("/sessions/s-503/messages")).status_code == 404  # nothing was added


async def test_503_does_not_touch_existing_history(client):
    await chat(client, "s-keep", "first message")
    bad = await chat(client, "s-keep", "this one fails", simulate="*=500")
    assert bad.status_code == 503
    msgs = (await client.get("/sessions/s-keep/messages")).json()["messages"]
    assert [m["content"] for m in msgs if m["role"] == "user"] == ["first message"]
    assert len(msgs) == 2


async def test_listed_keys_both_failing_is_503(client):
    r = await chat(client, "s-both", "hello", simulate="key_1=429,key_2=500")
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}


async def test_key_2_failure_alone_is_served_by_key_1(client):
    r = await chat(client, "s-k2", "hello", simulate="key_2=503")
    assert r.json()["metadata"]["key_id"] == "key_1"
    assert r.json()["metadata"]["fallback_attempts"] == 0


async def test_simulation_applies_to_one_request_only(client):
    first = await chat(client, "s-one", "one", simulate="key_1=429")
    second = await chat(client, "s-one", "two")
    assert first.json()["metadata"]["key_id"] == "key_2"
    assert second.json()["metadata"]["key_id"] == "key_1"


async def test_simulation_skips_cache_so_failover_is_demonstrable(client):
    first = await chat(client, "s-1", "same prompt")
    assert first.json()["metadata"]["key_id"] == "key_1"
    again = await chat(client, "s-2", "same prompt", simulate="key_1=429")
    assert again.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert again.json()["metadata"]["key_id"] == "key_2"
    blocked = await chat(client, "s-3", "same prompt", simulate="*=429")
    assert blocked.status_code == 503


async def test_503_does_not_poison_the_cache(client):
    await chat(client, "s-1", "retry me", simulate="*=429")
    r = await chat(client, "s-1", "retry me")
    assert r.status_code == 200
    assert r.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"


async def test_header_is_ignored_when_simulation_is_disabled(tmp_path):
    async with running_client(make_settings(tmp_path, enable_failure_simulation=False)) as c:
        r = await chat(c, "s-off", "hello", simulate="*=429")
    assert r.status_code == 200
    assert r.json()["metadata"]["key_id"] == "key_1"


async def test_malformed_header_is_ignored(client):
    for bad in ["garbage", "key_1=418", "key_1=", "=429", "key_1=abc"]:
        r = await chat(client, f"s-{bad}", f"hello {bad}", simulate=bad)
        assert r.status_code == 200, bad
        assert r.json()["metadata"]["key_id"] == "key_1"


# ---- real provider failures (not simulated) ---------------------------------------

def _rate_limit():
    return litellm.RateLimitError(message="quota exceeded", llm_provider="fake", model="m")


@pytest.mark.parametrize(
    "error",
    [
        _rate_limit(),
        litellm.InternalServerError(message="boom", llm_provider="fake", model="m"),
        litellm.ServiceUnavailableError(message="down", llm_provider="fake", model="m"),
        litellm.Timeout(message="slow", model="m", llm_provider="fake"),
        litellm.APIConnectionError(message="no route", llm_provider="fake", model="m"),
        litellm.AuthenticationError(message="bad key", llm_provider="fake", model="m"),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_real_provider_errors_fail_over(client, fake, error):
    fake.errors["k1"] = error
    r = await chat(client, "s-real", "hello")
    assert r.status_code == 200
    assert r.json()["metadata"]["key_id"] == "key_2"
    assert r.json()["metadata"]["fallback_attempts"] == 1


async def test_every_real_key_failing_is_503_without_leaking_details(client, fake):
    fake.errors["k1"] = _rate_limit()
    fake.errors["k2"] = litellm.InternalServerError(message="secret upstream text k1", llm_provider="fake", model="m")
    r = await chat(client, "s-all", "hello")
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}
    assert "secret" not in r.text


async def test_rate_limited_key_is_tried_last_during_cooldown(client, fake):
    fake.errors["k1"] = _rate_limit()
    first = await chat(client, "s-cd", "one")
    assert first.json()["metadata"]["key_id"] == "key_2"
    assert first.json()["metadata"]["fallback_attempts"] == 1
    fake.calls.clear()
    second = await chat(client, "s-cd", "two")
    assert fake.calls == ["k2"]  # key_1 is skipped while cooling down
    assert second.json()["metadata"]["fallback_attempts"] == 0

    # if the healthy key now fails too, the cooling key is still tried before giving up
    fake.errors.pop("k1")
    fake.errors["k2"] = _rate_limit()
    third = await chat(client, "s-cd", "three")
    assert third.status_code == 200
    assert third.json()["metadata"]["key_id"] == "key_1"


async def test_single_key_pool_still_works_and_fails_cleanly(tmp_path, fake):
    async with running_client(make_settings(tmp_path, llm_api_keys=["only"])) as c:
        ok = await chat(c, "s-1", "hello")
        assert ok.status_code == 200
        down = await chat(c, "s-2", "hello again", simulate="*=500")
        assert down.status_code == 503


# ---- header parser ---------------------------------------------------------------

def test_parse_failure_header():
    plan = parse_failure_header("key_1=429, key_2=500")
    assert plan.status_for("key_1") == 429
    assert plan.status_for("key_2") == 500
    assert plan.status_for("key_3") is None

    star = parse_failure_header("*=503")
    assert star.status_for("key_1") == 503 and star.status_for("key_9") == 503

    mixed = parse_failure_header("*=429,key_2=500")
    assert mixed.status_for("key_1") == 429 and mixed.status_for("key_2") == 500

    assert parse_failure_header(None) is None
    assert parse_failure_header("") is None
    assert parse_failure_header("nonsense") is None
