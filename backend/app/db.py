"""SQLite storage for chat sessions (aiosqlite, one shared connection)."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

import aiosqlite

# The `messages` table is exactly the schema from the assignment. `sessions` is an
# addition that holds the optional per-session system prompt.
SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    system_prompt  TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None
        # One connection is shared, so writes take a lock to keep transactions from interleaving.
        self._write_lock = asyncio.Lock()

    async def connect(self) -> None:
        directory = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(directory, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        await self._conn.executescript(SCHEMA)  # schema is created automatically on startup
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("database is not connected")
        return self._conn

    async def ping(self) -> bool:
        try:
            async with self.conn.execute("SELECT 1") as cur:
                await cur.fetchone()
            return True
        except Exception:
            return False

    # ---- reads -----------------------------------------------------------------

    async def session_exists(self, session_id: str) -> bool:
        async with self.conn.execute(
            "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1", (session_id,)
        ) as cur:
            return await cur.fetchone() is not None

    async def get_system_prompt(self, session_id: str) -> str | None:
        async with self.conn.execute(
            "SELECT system_prompt FROM sessions WHERE session_id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        return row["system_prompt"] if row else None

    async def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Full stored history, oldest first (what GET /sessions/{id}/messages returns)."""
        async with self.conn.execute(
            "SELECT role, content, metadata FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item: dict[str, Any] = {"role": row["role"], "content": row["content"]}
            if row["metadata"]:
                item["metadata"] = json.loads(row["metadata"])
            out.append(item)
        return out

    async def get_recent_messages(self, session_id: str, limit: int) -> list[dict[str, str]]:
        """Last `limit` messages as {"role", "content"}, oldest first, starting on a user turn."""
        async with self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        while history and history[0]["role"] != "user":
            history.pop(0)
        return history

    # ---- writes ----------------------------------------------------------------

    async def save_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        metadata: dict[str, Any],
        system_prompt: str | None = None,
    ) -> None:
        """Persist one successful user/assistant turn atomically.

        Failed requests never reach this method, so they leave no trace in the history.
        """
        now = _now()
        conn = self.conn
        async with self._write_lock:
            await self._save_exchange_locked(
                conn, session_id, user_message, assistant_reply, metadata, system_prompt, now
            )

    @staticmethod
    async def _save_exchange_locked(
        conn: aiosqlite.Connection,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        metadata: dict[str, Any],
        system_prompt: str | None,
        now: str,
    ) -> None:
        try:
            await conn.execute(
                "INSERT INTO sessions (session_id, system_prompt, created_at, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "  system_prompt = COALESCE(excluded.system_prompt, sessions.system_prompt), "
                "  updated_at = excluded.updated_at",
                (session_id, system_prompt, now, now),
            )
            await conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) "
                "VALUES (?, 'user', ?, NULL, ?)",
                (session_id, user_message, now),
            )
            await conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?)",
                (session_id, assistant_reply, json.dumps(metadata), _now()),
            )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
