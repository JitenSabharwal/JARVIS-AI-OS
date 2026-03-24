export type SceneDetection = {
  label: string;
  score: number;
  bbox: [number, number, number, number];
  trackId?: string;
  identity?: string;
  identityScore?: number;
  personId?: string;
};

export type SocialSceneEvent = {
  event_id: string;
  type: "person_entered" | "person_identified" | "person_left" | "object_spotted";
  severity: "low" | "medium" | "high";
  text: string;
  at_ms: number;
  track_id?: string;
  person_id?: string;
  metadata: Record<string, unknown>;
};

export type SocialCoverage = {
  tracked: number;
  matched: number;
  scanning: number;
};

type TrackState = {
  firstSeenAt: number;
  lastSeenAt: number;
  bbox: [number, number, number, number];
  personId?: string;
  identity?: string;
  leftAnnounced: boolean;
};

const PERSON_LEAVE_GRACE_MS = 2600;
const TRACK_TTL_MS = 12000;
const EVENT_COOLDOWN_MS = 15000;
const OBJECT_EVENT_COOLDOWN_MS = 45000;
const SOCIAL_PROMPT_COOLDOWN_MS = 28000;

export class SocialSceneDirector {
  private tracks = new Map<string, TrackState>();
  private eventCooldown = new Map<string, number>();
  private objectSeenAt = new Map<string, number>();
  private nextEphemeralTrack = 0;
  private lastPromptAt = 0;

  reset(): void {
    this.tracks.clear();
    this.eventCooldown.clear();
    this.objectSeenAt.clear();
    this.nextEphemeralTrack = 0;
    this.lastPromptAt = 0;
  }

  ingest(
    detections: SceneDetection[],
    nowMs = Date.now()
  ): {
    events: SocialSceneEvent[];
    coverage: SocialCoverage;
    summary: string;
    prompt: string;
  } {
    const events: SocialSceneEvent[] = [];
    const persons = detections.filter((d) => d.label === "person");
    const activeTrackIds = new Set<string>();

    for (let idx = 0; idx < persons.length; idx += 1) {
      const d = persons[idx];
      const trackId = String(d.trackId || `ep-${++this.nextEphemeralTrack}`).trim();
      activeTrackIds.add(trackId);
      const prev = this.tracks.get(trackId);
      const next: TrackState = {
        firstSeenAt: prev?.firstSeenAt || nowMs,
        lastSeenAt: nowMs,
        bbox: d.bbox,
        personId: d.personId,
        identity: d.identity,
        leftAnnounced: false
      };
      this.tracks.set(trackId, next);
      if (!prev && d.score >= 0.48) {
        this.maybePushEvent(
          events,
          {
            event_id: `evt-enter-${trackId}-${nowMs}`,
            type: "person_entered",
            severity: "medium",
            text: "A person entered the scene.",
            at_ms: nowMs,
            track_id: trackId,
            metadata: { score: d.score }
          },
          `enter:${trackId}`,
          EVENT_COOLDOWN_MS,
          nowMs
        );
      }
      if (!prev?.personId && d.personId && d.identity) {
        const idScore = Math.round(Number(d.identityScore || 0) * 100);
        this.maybePushEvent(
          events,
          {
            event_id: `evt-ident-${trackId}-${nowMs}`,
            type: "person_identified",
            severity: "high",
            text: `${d.identity} is now identified in the room (${idScore}%).`,
            at_ms: nowMs,
            track_id: trackId,
            person_id: d.personId,
            metadata: { identity_score: d.identityScore || 0, score: d.score }
          },
          `identified:${d.personId}`,
          EVENT_COOLDOWN_MS,
          nowMs
        );
      }
    }

    for (const [trackId, state] of this.tracks.entries()) {
      if (activeTrackIds.has(trackId)) continue;
      if (!state.leftAnnounced && nowMs - state.lastSeenAt >= PERSON_LEAVE_GRACE_MS) {
        state.leftAnnounced = true;
        this.tracks.set(trackId, state);
        const label = state.identity || "A person";
        this.maybePushEvent(
          events,
          {
            event_id: `evt-left-${trackId}-${nowMs}`,
            type: "person_left",
            severity: "low",
            text: `${label} left the scene.`,
            at_ms: nowMs,
            track_id: trackId,
            person_id: state.personId,
            metadata: {}
          },
          `left:${trackId}`,
          EVENT_COOLDOWN_MS,
          nowMs
        );
      }
      if (nowMs - state.lastSeenAt >= TRACK_TTL_MS) {
        this.tracks.delete(trackId);
      }
    }

    const seenObjectLabels = new Set<string>();
    for (const d of detections) {
      if (d.label === "person") continue;
      if (d.score < 0.58) continue;
      const label = String(d.label || "").trim().toLowerCase();
      if (!label || seenObjectLabels.has(label)) continue;
      seenObjectLabels.add(label);
      const prev = Number(this.objectSeenAt.get(label) || 0);
      if (nowMs - prev < OBJECT_EVENT_COOLDOWN_MS) continue;
      this.objectSeenAt.set(label, nowMs);
      events.push({
        event_id: `evt-obj-${label}-${nowMs}`,
        type: "object_spotted",
        severity: "low",
        text: `New object observed: ${label}.`,
        at_ms: nowMs,
        metadata: { label, score: d.score }
      });
    }

    const matched = persons.filter((p) => !!String(p.personId || "").trim()).length;
    const coverage: SocialCoverage = {
      tracked: persons.length,
      matched,
      scanning: Math.max(0, persons.length - matched)
    };

    const summary = events.length
      ? "Scene events: " + events.map((e) => e.text).join(" ")
      : "";

    const prompt = this.buildPrompt(events, coverage, nowMs);
    return { events, coverage, summary, prompt };
  }

  private buildPrompt(events: SocialSceneEvent[], coverage: SocialCoverage, nowMs: number): string {
    if (events.length === 0) return "";
    if (nowMs - this.lastPromptAt < SOCIAL_PROMPT_COOLDOWN_MS) return "";
    const identified = events.filter((e) => e.type === "person_identified");
    if (identified.length > 0) {
      this.lastPromptAt = nowMs;
      const names = identified
        .map((e) => String(e.text || "").split(" is now identified")[0])
        .filter(Boolean);
      return names.length > 0
        ? `Give a short social room update and greet ${names.join(", ")} naturally.`
        : "Give a short social room update about who was identified.";
    }
    if (coverage.scanning > 0 && coverage.tracked > 0) {
      this.lastPromptAt = nowMs;
      return `Give a short social room update; ${coverage.scanning} visible person(s) are still unidentified.`;
    }
    return "";
  }

  private maybePushEvent(
    out: SocialSceneEvent[],
    event: SocialSceneEvent,
    cooldownKey: string,
    cooldownMs: number,
    nowMs: number
  ): void {
    const prev = Number(this.eventCooldown.get(cooldownKey) || 0);
    if (nowMs - prev < cooldownMs) return;
    this.eventCooldown.set(cooldownKey, nowMs);
    out.push(event);
  }
}
