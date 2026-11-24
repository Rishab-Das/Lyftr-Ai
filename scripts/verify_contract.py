#!/usr/bin/env python3
"""Check a running backend against the assignment's API contract (standard library only).

    python scripts/verify_contract.py http://localhost:8000
    python scripts/verify_contract.py https://your-backend.example.com --origin https://your-frontend.example.com

The failover checks need ENABLE_FAILURE_SIMULATION=true on the target; pass --no-simulate to skip them.
Every run uses unique prompts, so results do not depend on what was cached before.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid

RESULTS: list[tuple[str, bool, str]] = []


def call(base: str, method: str, path: str, body=None, headers=None, raw: bytes | None = None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = resp.read().decode()
            try:
                parsed = json.loads(text) if text else None
            except json.JSONDecodeError:
                parsed = text
            return resp.status, parsed, dict(resp.headers)
    except urllib.error.HTTPError as err:
        text = err.read().decode()
        try:
            return err.code, json.loads(text), dict(err.headers)
        except json.JSONDecodeError:
            return err.code, text, dict(err.headers)


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    return ok


def chat(base, session, message, headers=None, **extra):
    return call(base, "POST", "/chat", {"session_id": session, "message": message, **extra}, headers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--origin", help="frontend origin to test CORS with")
    ap.add_argument("--no-simulate", action="store_true", help="skip X-Simulate-Failure checks")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    run = uuid.uuid4().hex[:8]

    print("GET /health")
    status, body, _ = call(base, "GET", "/health")
    check("200 and status ok", status == 200 and isinstance(body, dict) and body.get("status") == "ok", str(body))
    if isinstance(body, dict):
        check("keys_configured >= 2", int(body.get("keys_configured", 0)) >= 2, str(body))
        check("cache_enabled is true", body.get("cache_enabled") is True, str(body))

    print("POST /chat: live answer")
    prompt = f"Reply with one short sentence about the number {run}."
    s1, b1, _ = chat(base, f"v-{run}-a", prompt)
    ok = check("200", s1 == 200, str(b1))
    meta1 = (b1 or {}).get("metadata", {}) if ok else {}
    if ok:
        check("session_id echoed, reply non-empty", b1["session_id"] == f"v-{run}-a" and bool(b1["reply"].strip()))
        check("resolution_source LIVE_INFERENCE", meta1.get("resolution_source") == "LIVE_INFERENCE", str(meta1))
        check("model is a non-empty string", bool(meta1.get("model")), str(meta1))
        check("latency_ms is an int > 0", isinstance(meta1.get("latency_ms"), int) and meta1["latency_ms"] > 0, str(meta1))
        check("input/output tokens are ints > 0",
              all(isinstance(meta1.get(k), int) and meta1[k] > 0 for k in ("input_tokens", "output_tokens")), str(meta1))
        check("key_id like key_N, fallback_attempts is int",
              str(meta1.get("key_id", "")).startswith("key_") and isinstance(meta1.get("fallback_attempts"), int), str(meta1))

    print("POST /chat: cache hit (same message, new session)")
    s2, b2, _ = chat(base, f"v-{run}-b", prompt)
    if check("200", s2 == 200, str(b2)) and ok:
        meta2 = b2["metadata"]
        check("resolution_source LLM_CACHE", meta2.get("resolution_source") == "LLM_CACHE", str(meta2))
        check("identical reply text", b2["reply"] == b1["reply"])
        check("key_id null, fallback_attempts 0", meta2.get("key_id") is None and meta2.get("fallback_attempts") == 0, str(meta2))
        check("same tokens as the live call",
              (meta2.get("input_tokens"), meta2.get("output_tokens")) == (meta1["input_tokens"], meta1["output_tokens"]), str(meta2))
        check("cache hit faster than live", meta2.get("latency_ms", 10**9) < meta1["latency_ms"],
              f"live={meta1['latency_ms']}ms cache={meta2.get('latency_ms')}ms")

    print("POST /chat: a different message is never cached")
    s3, b3, _ = chat(base, f"v-{run}-c", f"Reply with one short sentence about the number {run}0.")
    check("LIVE_INFERENCE", s3 == 200 and b3["metadata"]["resolution_source"] == "LIVE_INFERENCE", str(b3))

    print("Multi-turn and history")
    sid = f"v-{run}-mt"
    chat(base, sid, f"Remember this code word: kumquat{run[:4]}. Just say OK.")
    s4, b4, _ = chat(base, sid, "What was the code word I asked you to remember? Answer with just the word.")
    if check("second turn 200", s4 == 200, str(b4)):
        remembered = f"kumquat{run[:4]}" in b4["reply"].lower()
        # Depends on the model, not on the app, so it is reported but never fails the run.
        print(f"  [INFO] model recalled the code word from turn 1: {remembered}")
        check("input_tokens grew with history but is per-request",
              b4["metadata"]["input_tokens"] > 0 and b4["metadata"]["input_tokens"] < 2000, str(b4["metadata"]))
    sh, bh, _ = call(base, "GET", f"/sessions/{sid}/messages")
    check("history 200 with 4 messages oldest first",
          sh == 200 and [m["role"] for m in bh["messages"]] == ["user", "assistant", "user", "assistant"], str(bh)[:200])
    if sh == 200:
        check("assistant history items carry metadata",
              all("resolution_source" in m.get("metadata", {}) for m in bh["messages"] if m["role"] == "assistant"))
        check("no system message in history", all(m["role"] != "system" for m in bh["messages"]))
    sn, bn, _ = call(base, "GET", f"/sessions/does-not-exist-{run}/messages")
    check("unknown session -> 404 {'detail': 'session not found'}", sn == 404 and bn == {"detail": "session not found"}, str(bn))

    print("Token counts per request")
    ta = chat(base, f"v-{run}-t1", f"Say hi to {run[:4]}aa")[1]
    tb = chat(base, f"v-{run}-t2", f"Say hi to {run[:4]}bb")[1]
    if isinstance(ta, dict) and isinstance(tb, dict) and "metadata" in ta and "metadata" in tb:
        a, b = ta["metadata"]["input_tokens"], tb["metadata"]["input_tokens"]
        check("equal-length first messages report about the same input_tokens", abs(a - b) <= 2, f"{a} vs {b}")

    print("Validation (422) and nothing stored on failure")
    bad_sid = f"v-{run}-bad"
    for label, payload in [
        ("empty message", {"session_id": bad_sid, "message": ""}),
        ("whitespace message", {"session_id": bad_sid, "message": "   "}),
        ("message over 4000 chars", {"session_id": bad_sid, "message": "a" * 4001}),
        ("missing message", {"session_id": bad_sid}),
        ("missing session_id", {"message": "hi"}),
    ]:
        st, bd, _ = call(base, "POST", "/chat", payload)
        check(f"422 for {label}", st == 422 and isinstance(bd, dict) and "detail" in bd, f"{st} {bd}")
    st, bd, _ = call(base, "POST", "/chat", raw=b"{not json")
    check("422 for malformed JSON", st == 422 and isinstance(bd, dict) and "detail" in bd, f"{st} {bd}")
    check("rejected requests are not stored", call(base, "GET", f"/sessions/{bad_sid}/messages")[0] == 404)

    if not args.no_simulate:
        print("Failover with X-Simulate-Failure")
        for hdr in ("key_1=429", "key_1=500"):
            st, bd, _ = chat(base, f"v-{run}-{hdr}", f"Failover test {hdr} {run}", {"X-Simulate-Failure": hdr})
            m = (bd or {}).get("metadata", {}) if st == 200 else {}
            check(f"{hdr} -> 200, key_2, 1 failed attempt, real answer",
                  st == 200 and m.get("key_id") == "key_2" and m.get("fallback_attempts") == 1 and bool(bd["reply"].strip()), f"{st} {bd}")
        sid503 = f"v-{run}-503"
        st, bd, _ = chat(base, sid503, f"All keys fail {run}", {"X-Simulate-Failure": "*=429"})
        check("*=429 -> 503 {'detail': 'all providers unavailable'}", st == 503 and bd == {"detail": "all providers unavailable"}, f"{st} {bd}")
        check("failed request not added to history", call(base, "GET", f"/sessions/{sid503}/messages")[0] == 404)
        st, bd, _ = chat(base, f"v-{run}-both", f"Both keys fail {run}", {"X-Simulate-Failure": "key_1=429,key_2=500"})
        check("key_1=429,key_2=500 -> 503", st == 503 and bd == {"detail": "all providers unavailable"}, f"{st} {bd}")
        st, bd, _ = chat(base, f"v-{run}-after", f"Failover test key_1=429 {run}")
        check("simulation affected one request only (next one uses key_1 or cache)",
              st == 200 and (bd["metadata"]["key_id"] in ("key_1", None)), str(bd))

    if args.origin:
        print("CORS")
        st, _, hd = call(base, "OPTIONS", "/chat", headers={
            "Origin": args.origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-simulate-failure"})
        allow = {k.lower(): v for k, v in hd.items()}.get("access-control-allow-origin")
        check(f"preflight allows {args.origin}", st in (200, 204) and allow in (args.origin, "*"), f"{st} {allow}")

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
