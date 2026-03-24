export type ChatRole = "user" | "assistant";

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  ts: number;
};

export type SessionItem = {
  id: string;
  title: string;
  createdAt: number;
  messages: ChatMessage[];
};

export type RealtimeStartResponse = {
  success: boolean;
  data?: {
    session_id: string;
    realtime: Record<string, unknown>;
  };
  error?: string | null;
};
