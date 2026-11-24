import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const okBody = (overrides: Record<string, unknown> = {}) => ({
  session_id: "s-test",
  reply: "Paris is the capital of France.",
  metadata: {
    resolution_source: "LIVE_INFERENCE",
    model: "gemini/test-model",
    latency_ms: 812,
    input_tokens: 42,
    output_tokens: 9,
    key_id: "key_1",
    fallback_attempts: 0,
    ...overrides,
  },
});

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn().mockResolvedValue({ ok: status < 400, status, json: async () => body });
  vi.stubGlobal("fetch", fn);
  return fn;
}

beforeEach(() => vi.unstubAllGlobals());
afterEach(() => vi.unstubAllGlobals());

describe("chat contract", () => {
  it("shows reply and all metadata with the required test ids, and clears the input", async () => {
    const fetchMock = mockFetch(200, okBody());
    const user = userEvent.setup();
    render(<App />);

    const input = screen.getByTestId("chat-input") as HTMLTextAreaElement;
    await user.type(input, "What is the capital of France?");
    await user.click(screen.getByTestId("send-button"));

    const msg = await screen.findByTestId("assistant-message");
    expect(msg).toHaveTextContent("Paris is the capital of France.");
    expect(screen.getByTestId("meta-source")).toHaveTextContent(/^LIVE_INFERENCE$/);
    expect(screen.getByTestId("meta-model")).toHaveTextContent(/^gemini\/test-model$/);
    expect(screen.getByTestId("meta-latency")).toHaveTextContent(/^812 ms$/);
    expect(screen.getByTestId("meta-input-tokens")).toHaveTextContent(/^42$/);
    expect(screen.getByTestId("meta-output-tokens")).toHaveTextContent(/^9$/);
    expect(input.value).toBe("");

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/chat$/);
    const body = JSON.parse(init.body);
    expect(body.message).toBe("What is the capital of France?");
    expect(body.session_id).toMatch(/^s-[0-9a-f]{8}$/);
    expect(body.system_prompt).toBeUndefined();
  });

  it("shows LLM_CACHE responses", async () => {
    mockFetch(200, okBody({ resolution_source: "LLM_CACHE", key_id: null }));
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("chat-input"), "hi");
    await user.click(screen.getByTestId("send-button"));
    expect(await screen.findByTestId("meta-source")).toHaveTextContent(/^LLM_CACHE$/);
  });

  it("sends on Enter but not on Shift+Enter", async () => {
    const fetchMock = mockFetch(200, okBody());
    const user = userEvent.setup();
    render(<App />);
    const input = screen.getByTestId("chat-input");
    await user.type(input, "line one{Shift>}{Enter}{/Shift}line two");
    expect(fetchMock).not.toHaveBeenCalled();
    await user.type(input, "{Enter}");
    await screen.findByTestId("assistant-message");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).message).toBe("line one\nline two");
  });

  it("sends the optional system prompt", async () => {
    const fetchMock = mockFetch(200, okBody());
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("system-prompt-input"), "You are a pirate.");
    await user.type(screen.getByTestId("chat-input"), "hello{Enter}");
    await screen.findByTestId("assistant-message");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).system_prompt).toBe("You are a pirate.");
  });

  it("keeps one session id across messages and uses a new one per page load", async () => {
    const fetchMock = mockFetch(200, okBody());
    const user = userEvent.setup();
    const first = render(<App />);
    await user.type(screen.getByTestId("chat-input"), "one{Enter}");
    await screen.findByTestId("assistant-message");
    await user.type(screen.getByTestId("chat-input"), "two{Enter}");
    await screen.findAllByTestId("assistant-message");
    const ids = fetchMock.mock.calls.map((c) => JSON.parse(c[1].body).session_id);
    expect(ids[0]).toBe(ids[1]);
    first.unmount();

    render(<App />);
    await user.type(screen.getByTestId("chat-input"), "three{Enter}");
    await screen.findByTestId("assistant-message");
    const third = JSON.parse(fetchMock.mock.calls[2][1].body).session_id;
    expect(third).not.toBe(ids[0]);
  });

  it("sends X-Simulate-Failure only when the developer field is filled", async () => {
    const fetchMock = mockFetch(200, okBody({ key_id: "key_2", fallback_attempts: 1 }));
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("chat-input"), "plain{Enter}");
    await screen.findByTestId("assistant-message");
    expect(fetchMock.mock.calls[0][1].headers["X-Simulate-Failure"]).toBeUndefined();

    await user.click(screen.getByText("Developer options"));
    await user.type(screen.getByTestId("simulate-failure-input"), "key_1=429");
    await user.type(screen.getByTestId("chat-input"), "simulated{Enter}");
    await screen.findAllByTestId("assistant-message");
    expect(fetchMock.mock.calls[1][1].headers["X-Simulate-Failure"]).toBe("key_1=429");
    expect(screen.getAllByTestId("meta-fallbacks")[1]).toHaveTextContent("1");
  });
});

describe("errors are readable, never a blank screen", () => {
  it("shows a message for 503 and keeps the chat usable", async () => {
    mockFetch(503, { detail: "all providers unavailable" });
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("chat-input"), "hello{Enter}");
    expect(await screen.findByTestId("chat-error")).toHaveTextContent(/unavailable/i);
    expect(screen.queryByTestId("assistant-message")).toBeNull();
    expect(screen.getByTestId("chat-input")).toBeEnabled();
  });

  it("shows a message for 422", async () => {
    mockFetch(422, { detail: [{ loc: ["body", "message"], msg: "Value error, message must be 1 to 4000 characters after trimming", type: "value_error" }] });
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("chat-input"), "x{Enter}");
    expect(await screen.findByTestId("chat-error")).toHaveTextContent(/message must be 1 to 4000/);
  });

  it("shows a message when the server cannot be reached", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("chat-input"), "x{Enter}");
    expect(await screen.findByTestId("chat-error")).toHaveTextContent(/could not reach the server/i);
  });

  it("handles a non-JSON error body", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 502, json: async () => { throw new Error("html"); } }));
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByTestId("chat-input"), "x{Enter}");
    expect(await screen.findByTestId("chat-error")).toHaveTextContent(/something went wrong/i);
  });
});
