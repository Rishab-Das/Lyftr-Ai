import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { ChatError, sendChat } from "./api";
import { MAX_MESSAGE_CHARS } from "./config";
import type { ChatItem, ChatMetadata, NewChatItem } from "./types";

const SUGGESTIONS = [
  "Explain what an API gateway does in two sentences.",
  "Give me three tips for writing readable SQL.",
  "What is the difference between a process and a thread?",
];

function newSessionId(): string {
  const bytes = new Uint8Array(4);
  (globalThis.crypto ?? window.crypto).getRandomValues(bytes);
  return "s-" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
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
      <Chip
        label="failovers"
        testId="meta-fallbacks"
        value={meta.fallback_attempts}
        tone={meta.fallback_attempts > 0 ? "warn" : ""}
      />
    </div>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState(newSessionId); // a new session on every page load
  const [items, setItems] = useState<ChatItem[]>([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulate, setSimulate] = useState("");
  const [pending, setPending] = useState(false);
  const nextId = useRef(1);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
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
  };

  const overLimit = input.trim().length > MAX_MESSAGE_CHARS;

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>LLM Chat</h1>
          <p className="sub">
            Session <code data-testid="session-id">{sessionId}</code> · key failover and response caching
          </p>
        </div>
        <button type="button" className="ghost" onClick={newChat}>
          New chat
        </button>
      </header>

      <section className="settings" aria-label="Session settings">
        <label htmlFor="system-prompt">System prompt for this session (optional)</label>
        <textarea
          id="system-prompt"
          data-testid="system-prompt-input"
          rows={1}
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          placeholder="e.g. You are a pirate."
        />
        <details className="dev">
          <summary>Developer options</summary>
          <label htmlFor="simulate">X-Simulate-Failure header (needs ENABLE_FAILURE_SIMULATION=true on the backend)</label>
          <input
            id="simulate"
            data-testid="simulate-failure-input"
            value={simulate}
            onChange={(e) => setSimulate(e.target.value)}
            placeholder="key_1=429   or   key_1=429,key_2=500   or   *=429"
            spellCheck={false}
          />
        </details>
      </section>

      <main className="messages" aria-live="polite" data-testid="message-list">
        {items.length === 0 && !pending && (
          <div className="empty">
            <p>Ask anything. Each reply shows where it came from, which model answered, how long it took and its token use.</p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" className="suggestion" onClick={() => void send(s)}>
                  {s}
                </button>
              ))}
            </div>
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
            <div key={item.id} className="row">
              <article className="bubble assistant" data-testid="assistant-message">
                <div className="reply">{item.text}</div>
                <Metadata meta={item.meta} />
              </article>
            </div>
          );
        })}

        {pending && (
          <div className="row" data-testid="typing-indicator">
            <div className="bubble assistant typing" aria-label="Assistant is thinking">
              <span /> <span /> <span />
            </div>
          </div>
        )}
        <div ref={endRef} />
      </main>

      <footer className="composer">
        {overLimit && (
          <p className="limit" role="status">
            Over the {MAX_MESSAGE_CHARS}-character limit. The server will reject this message.
          </p>
        )}
        <div className="composer-row">
          <textarea
            ref={inputRef}
            data-testid="chat-input"
            aria-label="Message"
            rows={Math.min(6, Math.max(1, input.split("\n").length))}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type a message. Enter to send, Shift+Enter for a new line."
          />
          <button
            type="button"
            className="primary"
            data-testid="send-button"
            onClick={() => void send()}
            disabled={pending || !input.trim()}
          >
            {pending ? "Sending…" : "Send"}
          </button>
        </div>
      </footer>
    </div>
  );
}
