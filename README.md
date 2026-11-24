# LLM Chatbot with Key Failover and Response Caching

A small full-stack chat app. React frontend, FastAPI backend, SQLite storage, and a LiteLLM gateway that
rotates through a pool of API keys and caches responses. Every reply shows where it came from, which model
answered, how long it took and how many tokens that one request used.

```
Browser (React, nginx or static host)
   │  POST /chat   GET /sessions/{id}/messages   GET /health
   ▼
FastAPI ──► SQLite (/data/app.db)         history per session_id
   │
   └──► LiteLLM gateway
          ├─ cache (in-memory)             key = model + full message list
          └─ key pool: key_1 → key_2 → …   next key on 429, quota, 5xx, timeout
```

## 1. Live URLs

| What | URL |
| --- | --- |
| Frontend | `TODO: https://<your-frontend>` |
| Backend | `TODO: https://<your-backend>` |
| Backend health | `TODO: https://<your-backend>/health` |
| Demo video | `TODO: link` |

### How it was deployed

**Backend (Render, Docker).** New Web Service from this repo, runtime Docker, root directory `backend`
(or use the included `render.yaml` Blueprint). Set these environment variables:

| Variable | Value |
| --- | --- |
| `LLM_API_KEYS` | `key_one,key_two` (at least two) |
| `ALLOWED_ORIGINS` | the frontend URL, for example `https://my-chat.vercel.app` |
| `ENABLE_FAILURE_SIMULATION` | `true` (required so the failover hook can be demonstrated) |
| `DATABASE_URL` | `sqlite:////data/app.db` |

Render injects `PORT`; the container listens on it. The free plan has no persistent disk, so history survives
container restarts on a local `make up` (named volume) but not a Render redeploy. Attach a disk mounted at
`/data` if you need it to persist there too.

**Frontend (Vercel, Netlify or Cloudflare Pages).** Root directory `frontend`, build command `npm run build`,
output directory `dist`, environment variable `VITE_API_URL=https://<your-backend>`.

## 2. Run locally

### With Docker (recommended)

```bash
cp .env.example .env        # then put two real keys in LLM_API_KEYS
make up                     # frontend http://localhost:3000, backend http://localhost:8000
make down                   # stop; the SQLite volume is kept (make clean deletes it)
```

`make up` creates `.env` from `.env.example` if it is missing. History is stored in the named volume
`app-data`, mounted at `/data`, so it survives `make down` and `make up`.

### Without Docker

Backend (Python 3.13, the version the Docker image uses):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export LLM_API_KEYS="key_one,key_two"
export DATABASE_URL="sqlite:///./app.db"
export ALLOWED_ORIGINS="http://localhost:3000"
export ENABLE_FAILURE_SIMULATION=true
uvicorn app.main:app --port 8000
```

Frontend (Node 22):

```bash
cd frontend
npm ci
VITE_API_URL=http://localhost:8000 npm run dev      # http://localhost:3000
```

### Tests

```bash
cd backend && python -m pytest -q       # 57 tests, no network or real keys needed
cd frontend && npm test                 # 10 component tests
```

The backend tests register a fake LiteLLM provider, so the real LiteLLM code path (cache, exception types,
usage objects) runs without any network access.

### Configuration

Everything comes from environment variables. `.env.example` lists them all.

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_API_KEYS` | none | Comma-separated key pool. Ids are `key_1`, `key_2`, … in this order |
| `LLM_MODEL` | `gemini/gemini-3.1-flash-lite` | Any LiteLLM model name that matches your keys |
| `DATABASE_URL` | `sqlite:////data/app.db` | SQLite file location |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | CORS origins, or `*` |
| `ENABLE_FAILURE_SIMULATION` | `false` | `true` turns on the `X-Simulate-Failure` header |
| `DEFAULT_SYSTEM_PROMPT` | `You are a helpful, concise assistant.` | Used when a session sets none |
| `CACHE_TTL_SECONDS` | `3600` | How long a cached answer lives |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-key timeout before moving on |
| `KEY_COOLDOWN_SECONDS` | `30` | A key that just failed for real is tried last for this long |
| `MAX_HISTORY_MESSAGES` | `40` | Most recent messages sent to the model as context |
| `API_URL` / `VITE_API_URL` | `http://localhost:8000` | Backend URL for the browser (Docker runtime / build time) |

## 3. Design decisions

### Key fallback

The key pool is an ordered list. For each request the gateway calls `litellm.acompletion` with `key_1`; if that
raises anything (429, quota error, 5xx, timeout, connection error, bad key, empty reply) it moves to `key_2`, and
so on. If every key fails the API returns `503 {"detail": "all providers unavailable"}` and nothing is written to
history. The response reports which key answered (`key_id`) and how many failed first (`fallback_attempts`);
the key itself is never returned, and is redacted from logs.

Why a loop around `litellm.acompletion` instead of `litellm.Router`: the contract needs deterministic order
(`key_1` first), the id of the key that answered, and the exact number of failed attempts. The loop gives all
three directly and is easy to test. LiteLLM still does the provider calls, error mapping and caching.

`num_retries=0` on every call, because retrying the same key is the wrong answer to a 429. A key that failed for
real is moved to the back of the line for `KEY_COOLDOWN_SECONDS`, so the next request does not waste time on it.

**`X-Simulate-Failure`** (only when `ENABLE_FAILURE_SIMULATION=true`) makes listed keys raise the same LiteLLM
exception a real provider status would (`RateLimitError` for 429, `InternalServerError` for 500, and so on) inside
that one request, so it exercises the same failover path as a real outage. Simulated requests skip the cache read
and use strict key order, which keeps the demo deterministic: `key_1=429` always ends on `key_2` with one failed
attempt, even for a message that was cached a minute ago. Malformed header entries are ignored.

### Caching

LiteLLM's built-in in-memory cache is enabled once at startup (`litellm.cache = Cache(type="local", ttl=…)`) and
every call passes `caching=True`. The cache key is the model plus the full message list (system prompt, history,
new message). The API key is not part of it, so an answer produced by `key_2` is a hit for everyone. The same
first message in a new session is a hit, and a different message, system prompt or history is not.

On a hit LiteLLM returns the stored response, so the reply text is identical and the token counts are the usage
stored with the original answer; the API reports `key_id: null` and `fallback_attempts: 0`. A live answer is also
written to the cache with an explicit `await` before the response is sent, so a repeat request right after it is
guaranteed to hit.

In-memory rather than disk: the LiteLLM version used here has no disk cache type, and the assignment allows
either. The consequences are that the cache is cleared on restart and that the container runs a single uvicorn
worker (workers would not share a cache). Replies with empty text are never served: they count as a failed key.

### SQLite session storage

`messages` follows the schema in the assignment exactly. A second table, `sessions`, holds the optional
per-session system prompt. Both are created on startup (`CREATE TABLE IF NOT EXISTS`), and the file location comes from
`DATABASE_URL`, which in Docker points at the `/data` volume.

- A successful turn (user message and assistant reply plus the session row) is written in one transaction after the
  model has answered. Failed requests (422, 503) therefore leave no trace.
- The client sends only the new message. The backend loads the last `MAX_HISTORY_MESSAGES` messages from SQLite,
  starting on a user turn, and sends system prompt, history and new message to the model.
- A per-session lock keeps two quick messages in the same chat from interleaving. Different sessions run in parallel.
- WAL mode and a write lock on the single shared connection keep concurrent writes safe.
- `GET /sessions/{id}/messages` returns stored history oldest first, with assistant metadata and without the system prompt.

### Other choices

- **Tokens** come from the provider's usage object for that request only (system prompt + history + new message in,
  completion out). Nothing is accumulated.
- **Latency** is server-measured from the start of the request handler to the end of the model step. A cache hit
  is typically a few milliseconds against hundreds for a live call.
- **Errors** are always JSON with a `detail` field. Validation errors are 422, an unknown session is 404, all keys
  failing is 503, and anything unexpected is a generic 500 that never exposes a stack trace. The 500 handler sits
  inside the CORS middleware so the browser can still read it.
- **Security:** keys only come from the environment and are never returned or logged, the simulation hook is off
  unless enabled, CORS only allows the configured origins, and both containers run as non-root users.

## 4. Verification

Replace `$BASE` with the backend URL (`http://localhost:8000` locally).

```bash
BASE=http://localhost:8000
curl -s $BASE/health
# {"status":"ok","keys_configured":2,"cache_enabled":true}
```

**Per-message token usage.** Every reply carries its own `input_tokens` and `output_tokens`; in the UI they sit
next to each assistant message. Send two messages in one chat and watch `input_tokens` grow only by what the
history added, not by a running total:

```bash
chat() { curl -s -X POST $BASE/chat -H 'content-type: application/json' "${@:2}" -d "$1"; echo; }
chat '{"session_id":"demo-1","message":"Explain DNS in one sentence."}'
chat '{"session_id":"demo-1","message":"Now explain it to a five year old."}'
```

**Cache hit.** Send the same first message from a new session. The reply text is identical, `resolution_source` is
`LLM_CACHE`, `key_id` is `null` and the latency is far lower:

```bash
chat '{"session_id":"demo-2","message":"Explain DNS in one sentence."}'
```

**Failover.** The header makes `key_1` fail inside that one request:

```bash
chat '{"session_id":"demo-3","message":"Tell me a fun fact about octopuses."}' -H 'X-Simulate-Failure: key_1=429'
# 200, "key_id":"key_2","fallback_attempts":1, and a real answer

chat '{"session_id":"demo-4","message":"Tell me a fun fact about otters."}' -H 'X-Simulate-Failure: *=429'
# 503 {"detail":"all providers unavailable"}   (and nothing is stored)
curl -s -o /dev/null -w '%{http_code}\n' $BASE/sessions/demo-4/messages     # 404
```

In the UI, open **Developer options**, type `key_1=429` and send a message: the reply shows `key key_2` and
`failovers 1`. Use `*=429` to see the readable 503 message in the chat.

**History:**

```bash
curl -s $BASE/sessions/demo-1/messages          # stored turns, oldest first
curl -s $BASE/sessions/nope/messages            # 404 {"detail":"session not found"}
```

**Everything at once.** `scripts/verify_contract.py` checks the whole API contract (health, live answer, cache hit,
tokens, multi-turn history, 404, every 422 case, failover, 503, CORS) against any running backend, local or deployed:

```bash
python3 scripts/verify_contract.py $BASE --origin https://<your-frontend>
```

## 5. Setup Used

- Editor: `TODO: your editor`
- AI tools: Claude (Anthropic) was used to draft the backend, frontend, tests, Docker setup and this README.
- Stack: Python 3.13, FastAPI, LiteLLM 1.110, aiosqlite, React 18 with Vite and TypeScript, nginx, Docker Compose.
- Model provider: `TODO: Google AI Studio (Gemini) keys, or whichever you used`. Note that Gemini rate limits apply per
  Google Cloud project, not per key, so keys from two different projects give real failover headroom.

## 6. Bonus features

- **Optional system prompt** per session (`system-prompt-input`), stored with the session so later messages keep it.
- **Developer options** in the UI to send `X-Simulate-Failure`, so failover can be demoed without curl.
- **Key cooldown:** a key that just failed for real is tried last for a short time, saving a wasted call per request.
- **Per-session locking and a history window** (`MAX_HISTORY_MESSAGES`) so concurrent messages stay ordered and long chats stay within context limits.
- **`scripts/verify_contract.py`:** an executable version of the contract, usable against the deployed URLs.
- **Extra metadata chips** in the UI (`meta-key`, `meta-fallbacks`) showing which key answered and how many failed first.
- **Runtime-configurable frontend image:** the container writes `config.js` from `API_URL` at start-up, so one image works for any backend URL.
- **Readable errors everywhere:** 422, 503, timeouts and network failures all appear as plain messages in the chat.

## Project layout

```
backend/    FastAPI app (app/), tests (tests/), Dockerfile, requirements
frontend/   React + Vite app (src/), nginx.conf, Dockerfile
scripts/    verify_contract.py
docker-compose.yml   Makefile   .env.example   render.yaml   README.md
```

## Known limits

- The cache is in memory: it is empty after a restart and not shared between instances.
- SQLite means one backend instance. That fits this assignment and would be the first thing to change at scale.
- Free hosting tiers sleep when idle, so the first request after a pause can take a minute.
- LiteLLM is heavy: the backend uses roughly 300 MB of RAM, which is tight on a 512 MB free instance.
