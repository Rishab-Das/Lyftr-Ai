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
from fastapi.responses import JSONResponse

from .config import Settings
from .db import Database
from .gateway import AllProvidersUnavailable, LLMGateway
from .schemas import ChatRequest, ChatResponse, HistoryResponse
from .simulate import parse_failure_header

log = logging.getLogger("app")


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

    app = FastAPI(title="LLM Chatbot with Key Failover and Response Caching", lifespan=lifespan)

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

    @app.get("/health")
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

    @app.post("/chat", response_model=ChatResponse)
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

    @app.get("/sessions/{session_id}/messages", response_model=HistoryResponse)
    async def session_messages(session_id: str = Path(..., min_length=1, max_length=128)) -> dict[str, Any]:
        if not await db.session_exists(session_id):
            raise HTTPException(status_code=404, detail="session not found")
        return {"session_id": session_id, "messages": await db.get_messages(session_id)}

    return app


# Used by `uvicorn app.main:app`. Importing this module never touches the network or the disk;
# the database is opened in the lifespan hook.
app = create_app()
