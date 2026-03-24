import type { RealtimeStartResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_JARVIS_API_BASE || "http://127.0.0.1:8080";
const API_TOKEN = process.env.NEXT_PUBLIC_JARVIS_API_TOKEN || "";
const TOKEN_STORAGE_KEY = "jarvis_api_token";

export type RealtimeWsEvent = {
  type: string;
  id?: string;
  response?: string;
  error?: string;
  [key: string]: unknown;
};

export type ProfileGraphNode = {
  id: string;
  label: string;
  value: string;
  url?: string;
  source_type?: string;
};

export type ProfileGraphEdge = {
  from: string;
  to: string;
  relation: string;
  confidence?: number;
  source?: string;
};

export type ProfileGraph = {
  enabled: boolean;
  profile_id: string;
  display_name?: string;
  nodes: ProfileGraphNode[];
  edges: ProfileGraphEdge[];
  reason?: string;
};

export type VisionIdentity = {
  person_id: string;
  display_name: string;
  sample_count: number;
  created_at: number;
  updated_at: number;
};

export type VisionIdentitySample = {
  sample_id: string;
  image_b64: string;
  created_at: number;
};

export type VisionRecognitionSample = {
  sample_id: string;
  detection_index: number;
  image_b64: string;
};

export type VisionRecognitionMatch = {
  sample_id?: string;
  detection_index?: number;
  unknown: boolean;
  score: number;
  margin?: number;
  person_id?: string;
  display_name?: string;
  candidate_person_id?: string;
  candidate_display_name?: string;
};

export type WorldConcept = {
  concept_id: string;
  topic: string;
  tags: string[];
  notes_count: number;
  latest_note: string;
  detections_count: number;
  web_facts_count: number;
  web_facts: Array<{
    title: string;
    url: string;
    snippet: string;
    score: number;
    source_type: string;
  }>;
  notes: string[];
  detections: Array<Record<string, unknown>>;
  metadata: Record<string, unknown>;
  reference_links_count: number;
  reference_links: Array<{
    link_id: string;
    url: string;
    title: string;
    notes: string;
    source_type: string;
    tags: string[];
    created_at: number;
    updated_at: number;
  }>;
  interaction_logs_count: number;
  interaction_logs: Array<{
    log_id: string;
    link_id: string;
    summary: string;
    pattern_hint: string;
    outcome: string;
    extracted_facts: string[];
    recorded_at: number;
  }>;
  learning_runs_count: number;
  learning_runs: Array<{
    run_id: string;
    mode: string;
    link_id: string;
    link_title: string;
    link_url: string;
    query: string;
    added_facts: number;
    query_result_count: number;
    adapter_inserted_total: number;
    outcome: string;
    recorded_at: number;
  }>;
  updated_at: number;
  created_at: number;
};

export type RealtimeSocialEvent = {
  event_id: string;
  type: string;
  text: string;
  severity: string;
  at_ms: number;
  track_id?: string;
  person_id?: string;
  metadata?: Record<string, unknown>;
};

export type RealtimeSocialTimeline = {
  session_id: string;
  count: number;
  coverage: { tracked: number; matched: number; scanning: number };
  prompt_hint: string;
  last_summary: string;
  items: RealtimeSocialEvent[];
};

function getRuntimeToken(): string {
  if (typeof window === "undefined") return "";
  try {
    return String(window.localStorage.getItem(TOKEN_STORAGE_KEY) || "").trim();
  } catch {
    return "";
  }
}

export function getApiToken(): string {
  return getRuntimeToken() || API_TOKEN;
}

export function setApiToken(token: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(TOKEN_STORAGE_KEY, String(token || "").trim());
  } catch {
    // ignore storage failures
  }
}

function headers(): HeadersInit {
  const h: Record<string, string> = {
    "Content-Type": "application/json"
  };
  const token = getApiToken();
  if (token) {
    h.Authorization = "Bearer " + token;
  }
  return h;
}

export async function startRealtimeSession(userId: string): Promise<string> {
  const res = await fetch(API_BASE + "/api/v1/realtime/sessions/start", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ user_id: userId, max_frames: 16 })
  });
  const body = (await res.json()) as RealtimeStartResponse;
  if (!res.ok || !body?.data?.session_id) {
    throw new Error(body?.error || "Failed to start realtime session");
  }
  return body.data.session_id;
}

export async function sendRealtimeTurn(sessionId: string, text: string): Promise<string> {
  const res = await fetch(API_BASE + "/api/v1/realtime/sessions/" + sessionId + "/turn", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ text, modality: "voice" })
  });
  const body = await res.json();
  if (!res.ok) {
    throw new Error(body?.error || "Failed to send turn");
  }
  return String(body?.data?.response || "").trim();
}

export async function interruptRealtime(sessionId: string, reason = "barge_in"): Promise<void> {
  await fetch(API_BASE + "/api/v1/realtime/sessions/" + sessionId + "/interrupt", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ reason })
  });
}

export async function pushCameraFrame(
  sessionId: string,
  imageB64: string,
  source = "iphone_camera",
  metadata: Record<string, unknown> = { source_device: "webcam" }
): Promise<void> {
  await fetch(API_BASE + "/api/v1/realtime/sessions/" + sessionId + "/media", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({
      source,
      image_b64: imageB64,
      metadata
    })
  });
}

export async function pushRealtimeSummary(
  sessionId: string,
  summary: string,
  source = "camera_detection",
  metadata: Record<string, unknown> = {}
): Promise<void> {
  await fetch(API_BASE + "/api/v1/realtime/sessions/" + sessionId + "/media", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({
      source,
      summary,
      metadata
    })
  });
}

export async function startUrlStream(
  sessionId: string,
  sourceUrl: string,
  sourceType: "rtsp" | "webrtc" | "http" = "http"
): Promise<void> {
  const res = await fetch(API_BASE + "/api/v1/realtime/sessions/" + sessionId + "/streams/start", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ source_type: sourceType, source_url: sourceUrl, interval_ms: 2500 })
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.error || "Failed to start stream ingest");
  }
}

export async function getRealtimeSocialTimeline(
  sessionId: string,
  limit = 40
): Promise<RealtimeSocialTimeline> {
  const res = await fetch(
    API_BASE +
      "/api/v1/realtime/sessions/" +
      encodeURIComponent(sessionId) +
      "/social/timeline?limit=" +
      encodeURIComponent(String(limit)),
    {
      method: "GET",
      headers: headers()
    }
  );
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error || "Failed to load social timeline");
  const data = body?.data || {};
  return {
    session_id: String(data.session_id || sessionId),
    count: Number(data.count || 0),
    coverage: {
      tracked: Number(data?.coverage?.tracked || 0),
      matched: Number(data?.coverage?.matched || 0),
      scanning: Number(data?.coverage?.scanning || 0)
    },
    prompt_hint: String(data.prompt_hint || ""),
    last_summary: String(data.last_summary || ""),
    items: Array.isArray(data.items) ? data.items : []
  };
}

export async function getRealtimeSocialExplain(
  sessionId: string,
  eventId = ""
): Promise<Record<string, unknown>> {
  const qp = eventId ? "?event_id=" + encodeURIComponent(eventId) : "";
  const res = await fetch(
    API_BASE + "/api/v1/realtime/sessions/" + encodeURIComponent(sessionId) + "/social/explain" + qp,
    {
      method: "GET",
      headers: headers()
    }
  );
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error || "Failed to explain social event");
  return (body?.data || {}) as Record<string, unknown>;
}

function wsBase(): string {
  if (API_BASE.startsWith("https://")) return API_BASE.replace("https://", "wss://");
  if (API_BASE.startsWith("http://")) return API_BASE.replace("http://", "ws://");
  return API_BASE;
}

export function openRealtimeSocket(
  sessionId: string,
  handlers: {
    onEvent: (event: RealtimeWsEvent) => void;
    onOpen?: () => void;
    onClose?: (ev: CloseEvent) => void;
    onError?: () => void;
  }
): WebSocket {
  const url = new URL(wsBase() + "/api/v1/realtime/sessions/" + encodeURIComponent(sessionId) + "/ws");
  const token = getApiToken();
  if (token) url.searchParams.set("access_token", token);
  const ws = new WebSocket(url.toString());
  ws.onopen = () => handlers.onOpen?.();
  ws.onerror = () => handlers.onError?.();
  ws.onclose = (ev) => handlers.onClose?.(ev);
  ws.onmessage = (ev) => {
    try {
      const payload = JSON.parse(String(ev.data)) as RealtimeWsEvent;
      handlers.onEvent(payload);
    } catch {
      handlers.onEvent({ type: "error", error: "Invalid WS message payload" });
    }
  };
  return ws;
}

export function sendRealtimeWs(ws: WebSocket | null, payload: Record<string, unknown>): boolean {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  ws.send(JSON.stringify(payload));
  return true;
}

export async function listVisionIdentities(): Promise<VisionIdentity[]> {
  const res = await fetch(API_BASE + "/api/v1/vision/identities", {
    method: "GET",
    headers: headers()
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error || "Failed to list identities");
  return Array.isArray(body?.data?.identities) ? (body.data.identities as VisionIdentity[]) : [];
}

export async function enrollVisionIdentity(
  name: string,
  samples: string[],
  metadata: Record<string, unknown> = { source: "jarvis-web" }
): Promise<VisionIdentity> {
  const cleanName = String(name || "").trim();
  const cleanSamples = (Array.isArray(samples) ? samples : []).map((s) => String(s || "").trim()).filter(Boolean);
  const res = await fetch(API_BASE + "/api/v1/vision/identities/enroll", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({
      name: cleanName,
      samples: cleanSamples,
      metadata
    })
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body?.data?.identity) throw new Error(body?.error || "Failed to enroll identity");
  return body.data.identity as VisionIdentity;
}

export async function deleteVisionIdentity(personId: string): Promise<void> {
  const pid = encodeURIComponent(String(personId || "").trim());
  const res = await fetch(API_BASE + "/api/v1/vision/identities/" + pid, {
    method: "DELETE",
    headers: headers()
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error || "Failed to delete identity");
}

export async function listVisionIdentitySamples(personId: string): Promise<VisionIdentitySample[]> {
  const pid = encodeURIComponent(String(personId || "").trim());
  const res = await fetch(API_BASE + "/api/v1/vision/identities/" + pid + "/samples", {
    method: "GET",
    headers: headers()
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error || "Failed to list identity samples");
  return Array.isArray(body?.data?.samples) ? (body.data.samples as VisionIdentitySample[]) : [];
}

export async function deleteVisionIdentitySample(personId: string, sampleId: string): Promise<void> {
  const pid = encodeURIComponent(String(personId || "").trim());
  const sid = encodeURIComponent(String(sampleId || "").trim());
  const res = await fetch(API_BASE + "/api/v1/vision/identities/" + pid + "/samples/" + sid, {
    method: "DELETE",
    headers: headers()
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error || "Failed to delete sample");
}

export async function recognizeVisionIdentities(samples: VisionRecognitionSample[]): Promise<VisionRecognitionMatch[]> {
  const clean = (Array.isArray(samples) ? samples : [])
    .map((s) => ({
      sample_id: String(s.sample_id || "").trim(),
      detection_index: Number(s.detection_index),
      image_b64: String(s.image_b64 || "").trim()
    }))
    .filter((s) => s.image_b64 && Number.isFinite(s.detection_index));
  const res = await fetch(API_BASE + "/api/v1/vision/identities/recognize", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ samples: clean })
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error || "Failed to recognize identities");
  return Array.isArray(body?.data?.matches) ? (body.data.matches as VisionRecognitionMatch[]) : [];
}

export async function listWorldConcepts(limit = 100): Promise<WorldConcept[]> {
  const res = await fetch(API_BASE + "/api/v1/world/concepts?limit=" + encodeURIComponent(String(limit)), {
    method: "GET",
    headers: headers()
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error || "Failed to list world concepts");
  return Array.isArray(body?.data?.items) ? (body.data.items as WorldConcept[]) : [];
}

export async function teachWorldConcept(args: {
  topic: string;
  notes?: string;
  tags?: string[];
  detections?: Array<Record<string, unknown>>;
  metadata?: Record<string, unknown>;
  enrich_web?: boolean;
  max_items?: number;
}): Promise<WorldConcept> {
  const res = await fetch(API_BASE + "/api/v1/world/teach", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(args || {})
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body?.data?.concept) throw new Error(body?.error || "Failed to teach concept");
  return body.data.concept as WorldConcept;
}

export async function enrichWorldConcept(conceptId: string, maxItems = 5): Promise<WorldConcept> {
  const cid = encodeURIComponent(String(conceptId || "").trim());
  const res = await fetch(API_BASE + "/api/v1/world/concepts/" + cid + "/enrich", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ max_items: maxItems, run_adapters: true })
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body?.data?.concept) throw new Error(body?.error || "Failed to enrich concept");
  return body.data.concept as WorldConcept;
}

export async function enrichWorldConceptWithSource(
  conceptId: string,
  args: {
    max_items?: number;
    run_adapters?: boolean;
    target_source?: "web" | "linkedin" | "github" | "google";
    query?: string;
    url?: string;
    user_id?: string;
  }
): Promise<WorldConcept> {
  const cid = encodeURIComponent(String(conceptId || "").trim());
  const res = await fetch(API_BASE + "/api/v1/world/concepts/" + cid + "/enrich", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({
      max_items: Math.max(1, Math.min(12, Number(args?.max_items || 5))),
      run_adapters: Boolean(args?.run_adapters ?? true),
      target_source: String(args?.target_source || "web"),
      query: String(args?.query || "").trim(),
      url: String(args?.url || "").trim(),
      user_id: String(args?.user_id || "").trim()
    })
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body?.data?.concept) throw new Error(body?.error || "Failed to enrich concept");
  return body.data.concept as WorldConcept;
}

export async function getWorldConcept(conceptId: string): Promise<WorldConcept> {
  const cid = encodeURIComponent(String(conceptId || "").trim());
  const res = await fetch(API_BASE + "/api/v1/world/concepts/" + cid, {
    method: "GET",
    headers: headers()
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body?.data?.concept) throw new Error(body?.error || "Failed to load concept");
  return body.data.concept as WorldConcept;
}

export async function getWorldConceptProfileGraph(conceptId: string): Promise<ProfileGraph> {
  const cid = encodeURIComponent(String(conceptId || "").trim());
  const res = await fetch(API_BASE + "/api/v1/world/concepts/" + cid + "/profile-graph", {
    method: "GET",
    headers: headers()
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body?.data?.graph) throw new Error(body?.error || "Failed to load profile graph");
  return body.data.graph as ProfileGraph;
}

export async function updateWorldConcept(
  conceptId: string,
  patch: {
    topic?: string;
    notes?: string;
    tags?: string[] | string;
    detections?: Array<Record<string, unknown>>;
    metadata?: Record<string, unknown>;
  }
): Promise<WorldConcept> {
  const cid = encodeURIComponent(String(conceptId || "").trim());
  const res = await fetch(API_BASE + "/api/v1/world/concepts/" + cid, {
    method: "PATCH",
    headers: headers(),
    body: JSON.stringify(patch || {})
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body?.data?.concept) throw new Error(body?.error || "Failed to update concept");
  return body.data.concept as WorldConcept;
}

export async function deleteWorldConcept(conceptId: string): Promise<void> {
  const cid = encodeURIComponent(String(conceptId || "").trim());
  const res = await fetch(API_BASE + "/api/v1/world/concepts/" + cid, {
    method: "DELETE",
    headers: headers()
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error || "Failed to delete concept");
}

export async function addWorldConceptLink(
  conceptId: string,
  args: {
    url: string;
    title?: string;
    notes?: string;
    source_type?: string;
    tags?: string[] | string;
    interaction?: {
      summary?: string;
      pattern_hint?: string;
      outcome?: string;
      extracted_facts?: string[];
    };
  }
): Promise<WorldConcept> {
  const cid = encodeURIComponent(String(conceptId || "").trim());
  const res = await fetch(API_BASE + "/api/v1/world/concepts/" + cid + "/links", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(args || {})
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body?.data?.concept) throw new Error(body?.error || "Failed to add concept link");
  return body.data.concept as WorldConcept;
}

export async function deleteWorldConceptLink(conceptId: string, linkId: string): Promise<void> {
  const cid = encodeURIComponent(String(conceptId || "").trim());
  const lid = encodeURIComponent(String(linkId || "").trim());
  const res = await fetch(API_BASE + "/api/v1/world/concepts/" + cid + "/links/" + lid, {
    method: "DELETE",
    headers: headers()
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error || "Failed to delete concept link");
}

export async function logWorldConceptLinkInteraction(
  conceptId: string,
  linkId: string,
  args: {
    summary: string;
    extracted_facts?: string[] | string;
    pattern_hint?: string;
    outcome?: string;
  }
): Promise<WorldConcept> {
  const cid = encodeURIComponent(String(conceptId || "").trim());
  const lid = encodeURIComponent(String(linkId || "").trim());
  const res = await fetch(API_BASE + "/api/v1/world/concepts/" + cid + "/links/" + lid + "/interactions", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(args || {})
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body?.data?.concept) throw new Error(body?.error || "Failed to log interaction");
  return body.data.concept as WorldConcept;
}

export async function runWorldConceptLinkBrowserUse(
  conceptId: string,
  linkId: string,
  args: { max_items?: number; run_adapters?: boolean } = {}
): Promise<WorldConcept> {
  const cid = encodeURIComponent(String(conceptId || "").trim());
  const lid = encodeURIComponent(String(linkId || "").trim());
  const res = await fetch(API_BASE + "/api/v1/world/concepts/" + cid + "/links/" + lid + "/browser-use/run", {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(args || {})
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || !body?.data?.concept) throw new Error(body?.error || "Failed to run browser-use learning");
  return body.data.concept as WorldConcept;
}
