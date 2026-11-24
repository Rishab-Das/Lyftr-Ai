import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { ChatError, checkHealth, sendChat } from "./api";
import { API_URL, MAX_MESSAGE_CHARS } from "./config";
import Markdown from "./Markdown";
import type { ChatItem, ChatMetadata, HealthInfo, NewChatItem, ServerState } from "./types";

const SUGGESTIONS = [
  { title: "Explain", text: "Explain what an API gateway does in two sentences." },
  { title: "Write", text: "Give me three tips for writing readable SQL." },
  { title: "Compare", text: "What is the difference between a process and a thread?" },
];

const FAILURE_PRESETS = [
  { label: "key_1 rate limited", value: "key_1=429" },
  { label: "key_1 down", value: "key_1=500" },
  { label: "both keys fail", value: "key_1=429,key_2=500" },
  { label: "everything fails", value: "*=429" },
];

type Theme = "dark" | "light";

function newSessionId(): string {
  const bytes = new Uint8Array(4);
  (globalThis.crypto ?? window.crypto).getRandomValues(bytes);
  return "s-" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function initialTheme(): Theme {
  try {
    const saved = window.localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* storage can be unavailable */
  }
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

const nf = new Intl.NumberFormat("en-US");

/* ---------- small pieces ---------- */

function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="logo">
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#8b7cff" />
          <stop offset="1" stopColor="#38bdf8" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="9" fill="url(#lg)" />
      <path d="M9 10.5h14a1.5 1.5 0 0 1 1.5 1.5v7a1.5 1.5 0 0 1-1.5 1.5h-7l-4.5 3.5V20.5H9A1.5 1.5 0 0 1 7.5 19v-7A1.5 1.5 0 0 1 9 10.5z" fill="#fff" fillOpacity=".95" />
      <circle cx="12.5" cy="15.5" r="1.3" fill="#6d5dfc" />
      <circle cx="16" cy="15.5" r="1.3" fill="#6d5dfc" />
      <circle cx="19.5" cy="15.5" r="1.3" fill="#6d5dfc" />
    </svg>
  );
}

function Icon({ name }: { name: "copy" | "check" | "plus" | "sun" | "moon" | "panel" | "send" | "bolt" }) {
  const paths: Record<string, string> = {
    copy: "M8 8V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-3M5 8h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2z",
    check: "M5 12.5l4.5 4.5L19 7.5",
    plus: "M12 5v14M5 12h14",
    sun: "M12 4V2m0 20v-2m8-8h2M2 12h2m13.7-5.7l1.4-1.4M4.9 19.1l1.4-1.4m0-11.4L4.9 4.9m14.2 14.2l-1.4-1.4M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10z",
    moon: "M21 12.8A8.5 8.5 0 1 1 11.2 3a6.5 6.5 0 0 0 9.8 9.8z",
    panel: "M4 5h16v14H4zM15 5v14",
    send: "M5 12l14-7-5 14-2.5-5.5z",
    bolt: "M13 2L4 14h7l-1 8 9-12h-7z",
  };
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={paths[name]} />
    </svg>
  );
}

function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setDone(true);
      window.setTimeout(() => setDone(false), 1400);
    } catch {
      /* clipboard may be blocked; ignore */
    }
  };
  return (
    <button type="button" className="icon-btn" onClick={() => void copy()} aria-label={done ? "Copied" : label} title={done ? "Copied" : label}>
      <Icon name={done ? "check" : "copy"} />
      <span>{done ? "Copied" : label}</span>
    </button>
  );
}

function Chip(props: { label: string; testId: string; value: string | number; tone?: string }) {
  return (
    <span className={`chip ${props.tone ?? ""}`}>
      <span className="chip-k">{props.label}</span>
      <span className="chip-v" data-testid={props.testId}>
        {props.value}
      </span>
    </span>
  );
}

function Metadata({ meta }: { meta: ChatMetadata }) {
  const cached = meta.resolution_source === "LLM_CACHE";
  return (
    <div className="meta">
      <Chip label="source" testId="meta-source" value={meta.resolution_source} tone={cached ? "cache" : "live"} />
      <Chip label="model" testId="meta-model" value={meta.model} />
      <Chip label="latency" testId="meta-latency" value={`${meta.latency_ms} ms`} />
      <Chip label="in tokens" testId="meta-input-tokens" value={meta.input_tokens} />
      <Chip label="out tokens" testId="meta-output-tokens" value={meta.output_tokens} />
      <Chip label="key" testId="meta-key" value={meta.key_id ?? "none (cache)"} />
      <Chip label="failovers" testId="meta-fallbacks" value={meta.fallback_attempts} tone={meta.fallback_attempts > 0 ? "warn" : ""} />
    </div>
  );
}

function StatusPill({ state, health }: { state: ServerState; health: HealthInfo | null }) {
  const text: Record<ServerState, string> = {
    checking: "Connecting",
    waking: "Waking server",
    online: "Backend online",
    offline: "Backend offline",
  };
  const detail =
    state === "online" && health
      ? `${health.keys_configured ?? "?"} keys · cache ${health.cache_enabled ? "on" : "off"}`
      : state === "waking"
        ? "free tier cold start, up to a minute"
        : state === "offline"
          ? "retrying every 15 s"
          : "";
  return (
    <div className={`status ${state}`} role="status">
      <span className="dot" aria-hidden="true" />
      <div>
        <div className="status-t">{text[state]}</div>
        {detail && <div className="status-d">{detail}</div>}
      </div>
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="stat">
      <div className="stat-v">{value}</div>
      <div className="stat-k">{label}</div>
      {hint && <div className="stat-h">{hint}</div>}
    </div>
  );
}

function Insights({ metas }: { metas: ChatMetadata[] }) {
  const total = metas.length;
  const cached = metas.filter((m) => m.resolution_source === "LLM_CACHE").length;
  const live = total - cached;
  const liveMetas = metas.filter((m) => m.resolution_source === "LIVE_INFERENCE");
  const avgLive = liveMetas.length ? Math.round(liveMetas.reduce((a, m) => a + m.latency_ms, 0) / liveMetas.length) : 0;
  const inTok = metas.reduce((a, m) => a + m.input_tokens, 0);
  const outTok = metas.reduce((a, m) => a + m.output_tokens, 0);
  const failovers = metas.reduce((a, m) => a + m.fallback_attempts, 0);
  const maxLat = Math.max(1, ...metas.map((m) => m.latency_ms));

  const keyCounts = new Map<string, number>();
  for (const m of metas) if (m.key_id) keyCounts.set(m.key_id, (keyCounts.get(m.key_id) ?? 0) + 1);

  return (
    <section className="card" aria-label="Session insights">
      <h2>Session insights</h2>
      <div className="stats">
        <Stat label="Replies" value={String(total)} />
        <Stat label="Cache hit rate" value={total ? `${Math.round((cached / total) * 100)}%` : "–"} hint={total ? `${cached} of ${total}` : undefined} />
        <Stat label="Avg live latency" value={avgLive ? `${nf.format(avgLive)} ms` : "–"} />
        <Stat label="Failovers" value={String(failovers)} />
        <Stat label="Tokens in" value={nf.format(inTok)} />
        <Stat label="Tokens out" value={nf.format(outTok)} />
      </div>

      <div className="mix" aria-label="Live versus cached replies">
        <div className="mix-bar">
          {total === 0 ? (
            <span className="mix-empty" />
          ) : (
            <>
              {live > 0 && <span className="mix-live" style={{ flexGrow: live }} />}
              {cached > 0 && <span className="mix-cache" style={{ flexGrow: cached }} />}
            </>
          )}
        </div>
        <div className="mix-legend">
          <span><i className="sw live" /> Live {live}</span>
          <span><i className="sw cache" /> Cache {cached}</span>
        </div>
      </div>

      {total > 0 && (
        <div className="spark" aria-label="Latency per reply">
          <div className="spark-k">Latency per reply</div>
          <div className="spark-bars">
            {metas.slice(-24).map((m, i) => (
              <span
                key={i}
                className={m.resolution_source === "LLM_CACHE" ? "cache" : "live"}
                style={{ height: `${Math.max(6, (m.latency_ms / maxLat) * 100)}%` }}
                title={`${m.latency_ms} ms · ${m.resolution_source}`}
              />
            ))}
          </div>
        </div>
      )}

      {keyCounts.size > 0 && (
        <div className="keys">
          <div className="spark-k">Key usage</div>
          <div className="key-list">
            {[...keyCounts.entries()].map(([k, n]) => (
              <span key={k} className="key-pill">
                {k} <b>×{n}</b>
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

/* ---------- app ---------- */

export default function App({ healthCheck = import.meta.env.MODE !== "test" }: { healthCheck?: boolean }) {
  const [sessionId, setSessionId] = useState(newSessionId); // a new session on every page load
  const [items, setItems] = useState<ChatItem[]>([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulate, setSimulate] = useState("");
  const [pending, setPending] = useState(false);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [panelOpen, setPanelOpen] = useState(false);
  const [server, setServer] = useState<ServerState>("checking");
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const nextId = useRef(1);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      window.localStorage.setItem("theme", theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  // Wake the free-tier backend as soon as the page opens, and keep an eye on it.
  useEffect(() => {
    if (!healthCheck) return;
    let cancelled = false;
    let timer: number | undefined;
    const wakingTimer = window.setTimeout(() => {
      setServer((s) => (s === "checking" ? "waking" : s));
    }, 3500);

    const poll = async () => {
      try {
        const info = await checkHealth();
        if (cancelled) return;
        setHealth(info);
        setServer("online");
        timer = window.setTimeout(poll, 30_000);
      } catch {
        if (cancelled) return;
        setServer("offline");
        timer = window.setTimeout(poll, 15_000);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      window.clearTimeout(wakingTimer);
      window.clearTimeout(timer);
    };
  }, [healthCheck]);

  useEffect(() => {
    if (items.length === 0 && !pending) return; // keep the welcome screen at the top
    endRef.current?.scrollIntoView({ block: "end" });
  }, [items, pending]);

  const push = useCallback((item: NewChatItem) => {
    setItems((prev) => [...prev, { ...item, id: nextId.current++ } as ChatItem]);
  }, []);

  const send = useCallback(
    async (override?: string) => {
      const text = (override ?? input).trim();
      if (!text || pending) return;

      push({ kind: "user", text });
      setInput(""); // the field is empty right after sending
      setPending(true);
      try {
        const data = await sendChat({ sessionId, message: text, systemPrompt, simulateFailure: simulate });
        push({ kind: "assistant", text: data.reply, meta: data.metadata });
        setServer("online");
      } catch (err) {
        const message = err instanceof ChatError ? err.message : "Unexpected error. Please try again.";
        push({ kind: "error", text: message });
      } finally {
        setPending(false);
        inputRef.current?.focus();
      }
    },
    [input, pending, push, sessionId, systemPrompt, simulate],
  );

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void send();
    }
  };

  const newChat = () => {
    setItems([]);
    setInput("");
    setSessionId(newSessionId());
    inputRef.current?.focus();
  };

  const metas = useMemo(
    () => items.flatMap((i) => (i.kind === "assistant" ? [i.meta] : [])),
    [items],
  );
  const lastModel = metas.length ? metas[metas.length - 1].model : null;
  const overLimit = input.trim().length > MAX_MESSAGE_CHARS;
  const apiHost = API_URL.replace(/^https?:\/\//, "");

  return (
    <div className={`shell ${panelOpen ? "panel-open" : ""}`}>
      <aside className="side" aria-label="Details">
        <div className="brand">
          <Logo size={34} />
          <div>
            <div className="brand-n">Lyftr Chat</div>
            <div className="brand-s">Failover · Cache · Token metering</div>
          </div>
        </div>

        <StatusPill state={server} health={health} />

        <button type="button" className="btn-new" onClick={newChat}>
          <Icon name="plus" /> New chat
        </button>

        <section className="card" aria-label="Session">
          <h2>Session</h2>
          <div className="session-row">
            <code data-testid="session-id">{sessionId}</code>
            <CopyButton text={sessionId} label="Copy" />
          </div>
          <div className="kv">
            <span>API</span>
            <code title={API_URL}>{apiHost}</code>
          </div>
          {lastModel && (
            <div className="kv">
              <span>Model</span>
              <code>{lastModel}</code>
            </div>
          )}
        </section>

        <Insights metas={metas} />

        <details className="card dev">
          <summary>Developer options</summary>
          <label htmlFor="simulate">X-Simulate-Failure header</label>
          <input
            id="simulate"
            data-testid="simulate-failure-input"
            value={simulate}
            onChange={(e) => setSimulate(e.target.value)}
            placeholder="key_1=429,key_2=500"
            spellCheck={false}
          />
          <div className="presets" role="group" aria-label="Failure presets">
            {FAILURE_PRESETS.map((p) => (
              <button key={p.value} type="button" className={simulate === p.value ? "preset on" : "preset"} onClick={() => setSimulate(simulate === p.value ? "" : p.value)}>
                {p.label}
              </button>
            ))}
          </div>
          <p className="hint">
            Needs <code>ENABLE_FAILURE_SIMULATION=true</code> on the backend. Simulated requests skip the cache so you can watch the failover happen.
          </p>
        </details>

        <div className="side-foot">Replies are cached per prompt. Ask the same thing twice to see <b>LLM_CACHE</b>.</div>
      </aside>

      <div className="main">
        <header className="topbar">
          <button type="button" className="icon-btn only-mobile" onClick={() => setPanelOpen((v) => !v)} aria-label="Toggle details panel" aria-expanded={panelOpen}>
            <Icon name="panel" />
          </button>
          <div className="title">
            <h1>Chat</h1>
            <span className="sub">Session <code>{sessionId}</code></span>
          </div>
          <div className="top-actions">
            {lastModel && <span className="model-pill"><Icon name="bolt" /> {lastModel}</span>}
            <button type="button" className="icon-btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} title="Toggle theme">
              <Icon name={theme === "dark" ? "sun" : "moon"} />
            </button>
          </div>
        </header>

        <section className="persona" aria-label="System prompt">
          <label htmlFor="system-prompt">System prompt</label>
          <textarea
            id="system-prompt"
            data-testid="system-prompt-input"
            rows={1}
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="Optional persona, e.g. You are a pirate."
          />
        </section>

        <main className="messages" aria-live="polite" data-testid="message-list">
          <div className="messages-inner">
            {items.length === 0 && !pending && (
              <div className="empty">
                <Logo size={56} />
                <h2>How can I help today?</h2>
                <p>Every reply shows where it came from, which model and key answered, how long it took and what it cost in tokens.</p>
                <div className="suggestions">
                  {SUGGESTIONS.map((s) => (
                    <button key={s.text} type="button" className="suggestion" onClick={() => void send(s.text)}>
                      <span className="sg-t">{s.title}</span>
                      <span className="sg-x">{s.text}</span>
                    </button>
                  ))}
                </div>
                {server === "waking" && (
                  <p className="notice">The free-tier server is waking up. The first reply can take up to a minute, after that it is fast.</p>
                )}
              </div>
            )}

            {items.map((item) => {
              if (item.kind === "user") {
                return (
                  <div key={item.id} className="row user" data-testid="user-message">
                    <div className="bubble">{item.text}</div>
                  </div>
                );
              }
              if (item.kind === "error") {
                return (
                  <div key={item.id} className="row" role="alert" data-testid="chat-error">
                    <div className="bubble error">{item.text}</div>
                  </div>
                );
              }
              return (
                <div key={item.id} className="row assistant-row">
                  <div className="avatar" aria-hidden="true"><Logo size={26} /></div>
                  <div className="msg">
                    <article className="bubble assistant" data-testid="assistant-message">
                      <Markdown text={item.text} />
                    </article>
                    <Metadata meta={item.meta} />
                    <div className="msg-actions">
                      <CopyButton text={item.text} label="Copy reply" />
                    </div>
                  </div>
                </div>
              );
            })}

            {pending && (
              <div className="row assistant-row" data-testid="typing-indicator">
                <div className="avatar" aria-hidden="true"><Logo size={26} /></div>
                <div className="bubble assistant typing" aria-label="Assistant is thinking">
                  <span /> <span /> <span />
                  {server === "waking" && <em>waking the server…</em>}
                </div>
              </div>
            )}
            <div ref={endRef} />
          </div>
        </main>

        <footer className="composer">
          <div className="composer-inner">
            {overLimit && (
              <p className="limit" role="status">
                Over the {MAX_MESSAGE_CHARS}-character limit. The server will reject this message.
              </p>
            )}
            <div className="composer-box">
              <textarea
                ref={inputRef}
                data-testid="chat-input"
                aria-label="Message"
                rows={Math.min(6, Math.max(1, input.split("\n").length))}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="Ask anything…"
              />
              <button type="button" className="primary" data-testid="send-button" onClick={() => void send()} disabled={pending || !input.trim()}>
                {pending ? "Sending…" : <>Send <Icon name="send" /></>}
              </button>
            </div>
            <div className="counter">{input.trim().length > 0 ? `${input.trim().length} / ${MAX_MESSAGE_CHARS}` : "Responses are metered and cached"}</div>
          </div>
        </footer>
      </div>
    </div>
  );
}
