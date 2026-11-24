"""X-Simulate-Failure header: make chosen keys fail inside a single request."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import litellm

log = logging.getLogger("app.simulate")

ALLOWED_STATUSES = {429, 500, 502, 503}


@dataclass(frozen=True)
class FailurePlan:
    """Which keys should fail, and with which HTTP status, for one request."""

    by_key: dict[str, int] = field(default_factory=dict)
    all_status: int | None = None  # the `*=<status>` form

    def status_for(self, key_id: str) -> int | None:
        if key_id in self.by_key:
            return self.by_key[key_id]
        return self.all_status

    def __bool__(self) -> bool:
        return bool(self.by_key) or self.all_status is not None


def parse_failure_header(value: str | None) -> FailurePlan | None:
    """Parse `key_1=429,key_2=500` or `*=429`. Bad pairs are skipped, never fatal."""
    if not value:
        return None
    by_key: dict[str, int] = {}
    all_status: int | None = None
    for pair in value.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, sep, raw_status = pair.partition("=")
        name, raw_status = name.strip(), raw_status.strip()
        if not sep or not name or not raw_status.isdigit() or int(raw_status) not in ALLOWED_STATUSES:
            log.warning("ignoring malformed X-Simulate-Failure entry %r", pair)
            continue
        status = int(raw_status)
        if name == "*":
            all_status = status
        else:
            by_key[name] = status
    plan = FailurePlan(by_key=by_key, all_status=all_status)
    return plan if plan else None


def build_simulated_error(status: int, model: str) -> Exception:
    """Build the same LiteLLM exception the provider's HTTP status would produce."""
    message = f"simulated provider failure (HTTP {status})"
    kwargs = {"message": message, "llm_provider": "simulated", "model": model}
    if status == 429:
        return litellm.RateLimitError(**kwargs)
    if status == 502:
        return litellm.BadGatewayError(**kwargs)
    if status == 503:
        return litellm.ServiceUnavailableError(**kwargs)
    return litellm.InternalServerError(**kwargs)
