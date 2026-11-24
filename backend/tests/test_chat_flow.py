import asyncio
import json
import sqlite3

from .conftest import make_settings, running_client


async def chat(client, session, message, **extra):
    return await client.post("/chat", json={"session_id": session, "message": message, **extra})


async def test_multi_turn_sends_history_to_the_model(client, fake):
    await chat(client, "s-1", "my name is Ada")
    r = await chat(client, "s-1", "what is my name?")
    assert r.status_code == 200
    # system + user1 + assistant1 + user2
    assert [m["role"] for m in fake.last_messages] == ["system", "user", "assistant", "user"]
    assert fake.last_messages[1]["content"] == "my name is Ada"


async def test_sessions_stay_separate(client, fake):
    await chat(client, "s-a", "alpha one")
    await chat(client, "s-b", "beta one")
    await chat(client, "s-a", "alpha two")
    contents = [m["content"] for m in fake.last_messages if m["role"] == "user"]
    assert contents == ["alpha one", "alpha two"]

    a = (await client.get("/sessions/s-a/messages")).json()["messages"]
    b = (await client.get("/sessions/s-b/messages")).json()["messages"]
    assert [m["content"] for m in a if m["role"] == "user"] == ["alpha one", "alpha two"]
    assert [m["content"] for m in b if m["role"] == "user"] == ["beta one"]


async def test_history_endpoint_shape_and_order(client):
    await chat(client, "s-h", "first")
    await chat(client, "s-h", "second")
    r = await client.get("/sessions/s-h/messages")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "s-h"
    msgs = body["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "first"
    assert "metadata" not in msgs[0] or msgs[0]["metadata"] is None
    meta = msgs[1]["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert isinstance(meta["input_tokens"], int) and isinstance(meta["output_tokens"], int)
    assert all(m["role"] != "system" for m in msgs)


async def test_unknown_session_is_404(client):
    r = await client.get("/sessions/does-not-exist/messages")
    assert r.status_code == 404
    assert r.json() == {"detail": "session not found"}


async def test_tokens_are_per_request_not_running_totals(client):
    # fake provider counts words: system prompt (6 words) + messages
    r1 = await chat(client, "s-t", "one two three")
    r2 = await chat(client, "s-t", "four five six")
    m1, m2 = r1.json()["metadata"], r2.json()["metadata"]
    sys_words = 6  # "You are a helpful, concise assistant."
    assert m1["input_tokens"] == sys_words + 3
    reply1_words = m1["output_tokens"]
    assert m2["input_tokens"] == sys_words + 3 + reply1_words + 3  # this request's prompt only


async def test_equal_length_first_messages_report_equal_input_tokens(client):
    a = await chat(client, "s-x", "alpha beta gamma")
    b = await chat(client, "s-y", "delta omega sigma")
    assert a.json()["metadata"]["input_tokens"] == b.json()["metadata"]["input_tokens"]


async def test_message_is_trimmed_before_storage(client):
    await chat(client, "s-trim", "   padded   ")
    msgs = (await client.get("/sessions/s-trim/messages")).json()["messages"]
    assert msgs[0]["content"] == "padded"


async def test_system_prompt_overrides_default_and_sticks_to_session(client, fake):
    await chat(client, "s-sp", "hello", system_prompt="You are a pirate.")
    assert fake.last_messages[0] == {"role": "system", "content": "You are a pirate."}
    await chat(client, "s-sp", "again")  # no system_prompt this time: the session keeps its own
    assert fake.last_messages[0]["content"] == "You are a pirate."
    await chat(client, "s-other", "hello")
    assert fake.last_messages[0]["content"] == "You are a helpful, concise assistant."

    history = (await client.get("/sessions/s-sp/messages")).json()["messages"]
    assert all(m["role"] != "system" for m in history)


async def test_history_survives_restart(tmp_path, fake):
    settings = make_settings(tmp_path)
    async with running_client(settings) as c:
        await chat(c, "s-persist", "remember this")
    async with running_client(settings) as c:  # new app instance, same SQLite file
        msgs = (await c.get("/sessions/s-persist/messages")).json()["messages"]
        assert [m["content"] for m in msgs][0] == "remember this"
        await chat(c, "s-persist", "and this")
        assert [m["role"] for m in fake.last_messages] == ["system", "user", "assistant", "user"]


async def test_schema_matches_assignment(settings, client):
    await chat(client, "s-db", "hi")
    path = settings.sqlite_path
    con = sqlite3.connect(path)
    cols = [r[1] for r in con.execute("PRAGMA table_info(messages)")]
    assert cols == ["id", "session_id", "role", "content", "metadata", "created_at"]
    role, content, metadata, created = con.execute(
        "SELECT role, content, metadata, created_at FROM messages WHERE role='assistant'"
    ).fetchone()
    assert json.loads(metadata)["resolution_source"] == "LIVE_INFERENCE"
    assert "T" in created  # ISO-8601
    con.close()


async def test_concurrent_requests_in_one_session_do_not_interleave(client):
    results = await asyncio.gather(*[chat(client, "s-race", f"message {i}") for i in range(6)])
    assert all(r.status_code == 200 for r in results)
    msgs = (await client.get("/sessions/s-race/messages")).json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 6
    # each assistant reply answers the user message right before it
    for user, assistant in zip(msgs[0::2], msgs[1::2]):
        assert user["content"] in assistant["content"]
