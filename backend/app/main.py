"""FastAPI application: /health, /chat and /sessions/{session_id}/messages."""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from .config import Settings
from .db import Database
from .gateway import AllProvidersUnavailable, LLMGateway
from .schemas import ChatRequest, ChatResponse, HistoryResponse
from .simulate import parse_failure_header

log = logging.getLogger("app")

DESCRIPTION = """
A small gateway in front of an LLM.

* **Key pool with failover.** Keys are tried in order. A 429, quota error or 5xx moves on to the next key
  and the response reports how many keys failed in `fallback_attempts`.
* **Response caching.** Identical prompts are answered from cache (`LLM_CACHE`) with the original token usage replayed.
* **Token tracking.** Every reply carries `input_tokens`, `output_tokens`, `latency_ms`, `model` and `key_id`.
* **Failure simulation.** When `ENABLE_FAILURE_SIMULATION=true`, send `X-Simulate-Failure: key_1=429,key_2=500`
  to exercise failover without burning quota. Simulated requests bypass the cache.
"""

LANDING_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lyftr Chat API</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;color:#eceef6;
font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
background:radial-gradient(800px 460px at 85% -10%,rgba(124,108,255,.2),transparent 60%),radial-gradient(640px 420px at -5% 105%,rgba(56,189,248,.14),transparent 60%),#0b0c12}
main{width:min(720px,100%)}
.logo{width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,#8b7cff,#38bdf8);display:grid;place-items:center;box-shadow:0 10px 30px -8px rgba(110,96,255,.6)}
h1{font-size:30px;letter-spacing:-.03em;margin:18px 0 6px}p{color:#9096ab;margin:0 0 22px}
.status{display:inline-flex;gap:10px;align-items:center;border:1px solid #262a3a;background:#14161f;border-radius:99px;padding:6px 14px;font-size:13px;margin-bottom:22px}
.dot{width:9px;height:9px;border-radius:50%;background:#636982}.ok .dot{background:#4ade80;box-shadow:0 0 0 4px rgba(74,222,128,.2)}.bad .dot{background:#f04438}
.grid{display:grid;gap:10px}
.ep{display:flex;gap:14px;align-items:center;border:1px solid #262a3a;background:#14161f;border-radius:14px;padding:12px 16px;text-decoration:none;color:inherit;transition:border-color .15s,transform .15s}
.ep:hover{border-color:#7c6cff;transform:translateY(-1px)}
.m{font:600 11px ui-monospace,Menlo,monospace;padding:3px 8px;border-radius:7px;min-width:46px;text-align:center}
.get{background:rgba(45,212,191,.14);color:#5eead4}.post{background:rgba(124,108,255,.2);color:#c4b5fd}
code{font-family:ui-monospace,Menlo,monospace;font-size:13.5px}.d{color:#9096ab;font-size:13px;margin-left:auto;text-align:right}
.links{display:flex;gap:14px;margin-top:22px;font-size:14px}.links a{color:#38bdf8}
@media(max-width:560px){.d{display:none}}
</style></head><body><main>
<div class="logo"><svg width="24" height="24" viewBox="0 0 24 24" fill="#fff"><path d="M5 6h14a1.5 1.5 0 0 1 1.5 1.5v7A1.5 1.5 0 0 1 19 16h-7l-4.5 3.5V16H5a1.5 1.5 0 0 1-1.5-1.5v-7A1.5 1.5 0 0 1 5 6z"/></svg></div>
<h1>Lyftr Chat API</h1>
<p>LLM chat with API-key failover, response caching and per-request token metering.</p>
<div id="s" class="status"><span class="dot"></span><span id="t">Checking health…</span></div>
<div class="grid">
<a class="ep" href="/health"><span class="m get">GET</span><code>/health</code><span class="d">liveness and key count</span></a>
<a class="ep" href="/docs#/chat/chat_chat_post"><span class="m post">POST</span><code>/chat</code><span class="d">send a message, get a metered reply</span></a>
<a class="ep" href="/docs#/sessions"><span class="m get">GET</span><code>/sessions/{id}/messages</code><span class="d">stored conversation history</span></a>
</div>
<div class="links"><a href="/docs">Interactive docs</a><a href="/redoc">ReDoc</a><a href="/openapi.json">OpenAPI JSON</a></div>
</main>
<script>
fetch("/health").then(r=>r.json().then(b=>({ok:r.ok,b}))).then(({ok,b})=>{
 document.getElementById("s").className="status "+(ok?"ok":"bad");
 document.getElementById("t").textContent=ok?`Online · ${b.keys_configured} keys · cache ${b.cache_enabled?"on":"off"}`:"Degraded";
}).catch(()=>{document.getElementById("s").className="status bad";document.getElementById("t").textContent="Unreachable"});
</script></body></html>"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    for noisy in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    db = Database(settings.sqlite_path)
    gateway = LLMGateway(settings)
    # One lock per session so two quick messages in the same chat cannot interleave.
    session_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await db.connect()
        gateway.enable_cache()
        await gateway.warm_up()
        if gateway.keys_configured < 2:
            log.warning("only %d API key(s) configured; failover needs at least 2", gateway.keys_configured)
        log.info(
            "ready: model=%s keys=%d cache=on simulation=%s",
            gateway.model,
            gateway.keys_configured,
            settings.enable_failure_simulation,
        )
        yield
        await db.close()

    app = FastAPI(
        title="Lyftr Chat API",
        summary="LLM chat with API-key failover, response caching and token metering.",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "chat", "description": "Send a message and get a metered reply."},
            {"name": "sessions", "description": "Read back a stored conversation."},
            {"name": "ops", "description": "Liveness and readiness."},
        ],
    )

    # ---- error handling: every error is JSON with a `detail` field -------------------

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = [
            {"loc": list(err.get("loc", [])), "msg": str(err.get("msg", "")), "type": str(err.get("type", ""))}
            for err in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.middleware("http")
    async def catch_unhandled(request: Request, call_next):
        # Sits inside the CORS middleware, so even a 500 carries CORS headers and the browser
        # shows a readable error instead of an opaque network failure.
        try:
            return await call_next(request)
        except Exception:
            log.exception("unhandled error on %s %s", request.method, request.url.path)
            return JSONResponse(status_code=500, content={"detail": "internal server error"})

    allow_all = "*" in settings.allowed_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else settings.allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # ---- routes ----------------------------------------------------------------

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    async def landing() -> HTMLResponse:
        return HTMLResponse(LANDING_HTML)

    @app.get(
        "/health",
        tags=["ops"],
        summary="Health check",
        responses={503: {"description": "Database unreachable or no API keys configured."}},
    )
    async def health() -> JSONResponse:
        db_ok = await db.ping()
        keys = gateway.keys_configured
        ok = db_ok and keys >= 1
        return JSONResponse(
            status_code=200 if ok else 503,
            content={
                "status": "ok" if ok else "error",
                "keys_configured": keys,
                "cache_enabled": gateway.cache_enabled,
            },
        )

    @app.post(
        "/chat",
        response_model=ChatResponse,
        tags=["chat"],
        summary="Send a message",
        responses={
            422: {"description": "Invalid body, for example an empty or over-long message."},
            503: {"description": "Every key in the pool failed: `{\"detail\": \"all providers unavailable\"}`."},
        },
    )
    async def chat(
        body: ChatRequest,
        x_simulate_failure: str | None = Header(default=None),
    ) -> dict[str, Any]:
        started = time.perf_counter()
        plan = parse_failure_header(x_simulate_failure) if settings.enable_failure_simulation else None

        lock = session_locks.get(body.session_id)
        if lock is None:
            lock = asyncio.Lock()
            session_locks[body.session_id] = lock

        async with lock:
            history = await db.get_recent_messages(body.session_id, settings.max_history_messages)
            stored_prompt = await db.get_system_prompt(body.session_id)
            system_prompt = body.system_prompt or stored_prompt or settings.default_system_prompt

            messages = (
                [{"role": "system", "content": system_prompt}]
                + history
                + [{"role": "user", "content": body.message}]
            )

            try:
                result = await gateway.complete(messages, plan)
            except AllProvidersUnavailable:
                raise HTTPException(status_code=503, detail="all providers unavailable")

            metadata = {
                "resolution_source": "LLM_CACHE" if result.cache_hit else "LIVE_INFERENCE",
                "model": result.model,
                "latency_ms": max(1, round((time.perf_counter() - started) * 1000)),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "key_id": result.key_id,
                "fallback_attempts": result.fallback_attempts,
            }
            await db.save_exchange(
                body.session_id,
                body.message,
                result.reply,
                metadata,
                system_prompt=body.system_prompt,
            )

        return {"session_id": body.session_id, "reply": result.reply, "metadata": metadata}

    @app.get(
        "/sessions/{session_id}/messages",
        response_model=HistoryResponse,
        tags=["sessions"],
        summary="Conversation history",
        responses={404: {"description": "Unknown session id."}},
    )
    async def session_messages(session_id: str = Path(..., min_length=1, max_length=128)) -> dict[str, Any]:
        if not await db.session_exists(session_id):
            raise HTTPException(status_code=404, detail="session not found")
        return {"session_id": session_id, "messages": await db.get_messages(session_id)}

    return app


# Used by `uvicorn app.main:app`. Importing this module never touches the network or the disk;
# the database is opened in the lifespan hook.
app = create_app()
