declare global {
  interface Window {
    __CONFIG__?: { API_URL?: string };
  }
}

// Runtime value (config.js) wins, then the build-time Vite variable, then local dev default.
const raw = window.__CONFIG__?.API_URL || import.meta.env.VITE_API_URL || "http://localhost:8000";

export const API_URL: string = String(raw).replace(/\/+$/, "");
export const REQUEST_TIMEOUT_MS = 90_000; // free-tier backends can take a while to wake up
export const MAX_MESSAGE_CHARS = 4000;
