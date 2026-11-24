export interface ChatMetadata {
  resolution_source: "LIVE_INFERENCE" | "LLM_CACHE";
  model: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  key_id: string | null;
  fallback_attempts: number;
}

export interface ChatResponse {
  session_id: string;
  reply: string;
  metadata: ChatMetadata;
}

export type ChatItem =
  | { id: number; kind: "user"; text: string }
  | { id: number; kind: "assistant"; text: string; meta: ChatMetadata }
  | { id: number; kind: "error"; text: string };

// Omit<Union, "id"> would collapse the union, so distribute it over each member.
export type NewChatItem = ChatItem extends infer T ? (T extends ChatItem ? Omit<T, "id"> : never) : never;
