from .conftest import make_settings, running_client


async def chat(client, session, message, headers=None, **extra):
    return await client.post(
        "/chat", json={"session_id": session, "message": message, **extra}, headers=headers or {}
    )


async def test_second_identical_first_message_in_new_session_is_a_cache_hit(client, fake):
    first = await chat(client, "s-1", "What is the capital of France?")
    second = await chat(client, "s-2", "What is the capital of France?")  # immediately, no sleep

    m1, m2 = first.json()["metadata"], second.json()["metadata"]
    assert m1["resolution_source"] == "LIVE_INFERENCE"
    assert m2["resolution_source"] == "LLM_CACHE"
    assert second.json()["reply"] == first.json()["reply"]  # identical text
    assert len(fake.calls) == 1  # provider was called once

    assert m2["key_id"] is None
    assert m2["fallback_attempts"] == 0
    assert m2["model"] == m1["model"]
    assert m2["input_tokens"] == m1["input_tokens"]  # usage stored with the cached response
    assert m2["output_tokens"] == m1["output_tokens"]
    assert m2["latency_ms"] < m1["latency_ms"]  # a cache hit is faster than a live call


async def test_different_message_never_comes_from_cache(client, fake):
    await chat(client, "s-1", "What is the capital of France?")
    other = await chat(client, "s-2", "What is the capital of Spain?")
    assert other.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert len(fake.calls) == 2


async def test_cache_key_includes_history(client, fake):
    # same new message, different earlier context -> different message list -> no cache hit
    await chat(client, "s-1", "my name is Ada")
    await chat(client, "s-1", "what is my name?")
    await chat(client, "s-2", "my name is Grace")
    r = await chat(client, "s-2", "what is my name?")
    assert r.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"


async def test_cache_key_includes_system_prompt(client):
    await chat(client, "s-1", "hello there", system_prompt="Answer like a pirate.")
    r = await chat(client, "s-2", "hello there")  # default system prompt
    assert r.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    r = await chat(client, "s-3", "hello there", system_prompt="Answer like a pirate.")
    assert r.json()["metadata"]["resolution_source"] == "LLM_CACHE"


async def test_cache_hit_in_same_session_follow_up_is_stored_in_history(client):
    await chat(client, "s-1", "ping")
    await chat(client, "s-2", "ping")
    msgs = (await client.get("/sessions/s-2/messages")).json()["messages"]
    assert msgs[1]["metadata"]["resolution_source"] == "LLM_CACHE"
    assert msgs[1]["metadata"]["key_id"] is None


async def test_cache_survives_key_failover(client, fake):
    # answered by key_2 after key_1 fails; the answer is then cached for everyone
    first = await chat(client, "s-1", "cache after failover", headers={"X-Simulate-Failure": "key_1=429"})
    assert first.json()["metadata"]["key_id"] == "key_2"
    second = await chat(client, "s-2", "cache after failover")
    assert second.json()["metadata"]["resolution_source"] == "LLM_CACHE"
    assert second.json()["reply"] == first.json()["reply"]


async def test_empty_provider_reply_is_never_served(client, fake):
    fake.empty_keys = {"k1"}
    r = await chat(client, "s-1", "will be empty on key_1")
    assert r.status_code == 200
    assert r.json()["reply"].strip()
    assert r.json()["metadata"]["key_id"] == "key_2"
    assert r.json()["metadata"]["fallback_attempts"] == 1
    again = await chat(client, "s-2", "will be empty on key_1")
    assert again.json()["reply"] == r.json()["reply"]


async def test_cache_entries_expire(tmp_path, fake):
    import asyncio

    async with running_client(make_settings(tmp_path, cache_ttl_seconds=1)) as c:
        await chat(c, "s-1", "short lived")
        await asyncio.sleep(1.3)
        r = await chat(c, "s-2", "short lived")
        assert r.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
