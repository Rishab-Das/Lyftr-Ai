import { API_URL, REQUEST_TIMEOUT_MS } from "./config";
import type { ChatResponse, HealthInfo } from "./types";

export class ChatError extends Error {}

interface SendOptions {
  sessionId: string;
  message: string;
  systemPrompt?: string;
  simulateFailure?: string;
}

/** Turn a FastAPI/Pydantic 422 body into one readable sentence. */
function describeValidationError(detail: unknown): string {
  if (Array.isArray(detail) && detail.length > 0) {
    const parts = detail.map((item) => {
      const msg = String((item as { msg?: unknown }).msg ?? "invalid value").replace(/^Value error, /, "");
      const loc = (item as { loc?: unknown[] }).loc;
      const field = Array.isArray(loc) ? String(loc[loc.length - 1] ?? "") : "";
      return field && field !== "body" ? `${field}: ${msg}` : msg;
    });
    return `Your message was rejected (${parts.join("; ")}).`;
  }
  return "Your message was rejected. Please check it and try again.";
}

export async function sendChat(options: SendOptions): Promise<ChatResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (options.simulateFailure?.trim()) headers["X-Simulate-Failure"] = options.simulateFailure.trim();

  const body: Record<string, string> = { session_id: options.sessionId, message: options.message };
  if (options.systemPrompt?.trim()) body.system_prompt = options.systemPrompt.trim();

  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    if ((err as Error)?.name === "AbortError") {
      throw new ChatError("The server took too long to answer. If it was asleep it may need a minute, so please try again.");
    }
    throw new ChatError("Could not reach the server. Check your connection and try again.");
  } finally {
    window.clearTimeout(timer);
  }

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    /* non-JSON body: handled below */
  }

  if (response.ok) return payload as ChatResponse;

  const detail = (payload as { detail?: unknown } | null)?.detail;
  if (response.status === 422) throw new ChatError(describeValidationError(detail));
  if (response.status === 503) {
    throw new ChatError("All AI providers are unavailable right now (keys rate-limited or down). Please try again in a moment.");
  }
  const text = typeof detail === "string" ? detail : `status ${response.status}`;
  throw new ChatError(`Something went wrong on the server (${text}). Please try again.`);
}

/** Ping GET /health. Resolves with the body when the server is up, rejects otherwise. */
export async function checkHealth(timeoutMs = 75_000): Promise<HealthInfo> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_URL}/health`, { signal: controller.signal });
    if (!response.ok) throw new Error(`health ${response.status}`);
    return (await response.json()) as HealthInfo;
  } finally {
    window.clearTimeout(timer);
  }
}
