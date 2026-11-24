import json

from app.config import Settings

from .conftest import make_settings, running_client


async def test_health_ok(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "keys_configured": 2, "cache_enabled": True}


async def test_health_reports_error_without_keys(tmp_path):
    async with running_client(make_settings(tmp_path, llm_api_keys=[])) as c:
        r = await c.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "error"
    assert r.json()["keys_configured"] == 0


async def test_valid_request_shape(client):
    r = await client.post("/chat", json={"session_id": "s-1", "message": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"session_id", "reply", "metadata"}
    assert body["session_id"] == "s-1"
    assert body["reply"]
    meta = body["metadata"]
    assert set(meta) >= {
        "resolution_source", "model", "latency_ms", "input_tokens",
        "output_tokens", "key_id", "fallback_attempts",
    }
    assert isinstance(meta["latency_ms"], int)
    assert isinstance(meta["input_tokens"], int)
    assert isinstance(meta["output_tokens"], int)


async def test_missing_message_is_422(client):
    r = await client.post("/chat", json={"session_id": "s-1"})
    assert r.status_code == 422
    assert "detail" in r.json()


async def test_missing_session_id_is_422(client):
    r = await client.post("/chat", json={"message": "hi"})
    assert r.status_code == 422


async def test_empty_and_whitespace_message_is_422(client):
    for message in ["", "   ", "\n\t "]:
        r = await client.post("/chat", json={"session_id": "s-1", "message": message})
        assert r.status_code == 422, message


async def test_message_length_limit(client, fake):
    too_long = await client.post("/chat", json={"session_id": "s-1", "message": "a" * 4001})
    assert too_long.status_code == 422
    assert fake.calls == []  # rejected before any provider call

    ok = await client.post("/chat", json={"session_id": "s-1", "message": "a" * 4000})
    assert ok.status_code == 200

    # the limit applies after trimming whitespace
    padded = await client.post("/chat", json={"session_id": "s-2", "message": "  " + "b" * 4000 + "  "})
    assert padded.status_code == 200


async def test_session_id_limits(client):
    assert (await client.post("/chat", json={"session_id": "", "message": "hi"})).status_code == 422
    assert (await client.post("/chat", json={"session_id": "x" * 129, "message": "hi"})).status_code == 422
    assert (await client.post("/chat", json={"session_id": "x" * 128, "message": "hi"})).status_code == 200


async def test_wrong_types_are_422(client):
    r = await client.post("/chat", json={"session_id": 5, "message": ["x"]})
    assert r.status_code == 422


async def test_malformed_json_is_422_json(client):
    r = await client.post("/chat", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert "detail" in r.json()


async def test_failed_validation_is_not_stored(client):
    await client.post("/chat", json={"session_id": "s-bad", "message": ""})
    r = await client.get("/sessions/s-bad/messages")
    assert r.status_code == 404


async def test_errors_are_json_never_html(client):
    for r in [
        await client.get("/nope"),
        await client.get("/chat"),  # wrong method
        await client.post("/health"),
    ]:
        assert r.headers["content-type"].startswith("application/json")
        assert "detail" in r.json()


async def test_unhandled_exception_returns_clean_json_500(tmp_path, monkeypatch):
    async with running_client(make_settings(tmp_path)) as c:
        from app import db as db_module

        async def boom(*_a, **_k):
            raise RuntimeError("disk exploded at /secret/path")

        monkeypatch.setattr(db_module.Database, "get_recent_messages", boom)
        r = await c.post("/chat", json={"session_id": "s-1", "message": "hi"})
    assert r.status_code == 500
    assert r.json() == {"detail": "internal server error"}
    assert "secret" not in r.text and "Traceback" not in r.text


async def test_cors_allows_configured_origin_and_failure_header(client):
    pre = await client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-simulate-failure",
        },
    )
    assert pre.status_code == 200
    assert pre.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "x-simulate-failure" in pre.headers["access-control-allow-headers"].lower()

    other = await client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_sqlite_path_parsing():
    assert Settings(database_url="sqlite:////data/app.db").sqlite_path == "/data/app.db"
    assert Settings(database_url="sqlite:///./local.db").sqlite_path == "./local.db"
