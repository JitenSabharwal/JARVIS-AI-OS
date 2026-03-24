"""
Backend social scene orchestration for realtime video sessions.

Converts per-frame detections into stable social events and prompt hints with
anti-loop cooldown policy.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _TrackState:
    track_id: str
    bbox: tuple[float, float, float, float]
    first_seen_at: float
    last_seen_at: float
    person_id: str = ""
    identity: str = ""
    left_announced: bool = False


@dataclass
class _SessionState:
    session_id: str
    tracks: dict[str, _TrackState] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    cooldowns: dict[str, float] = field(default_factory=dict)
    object_seen_at: dict[str, float] = field(default_factory=dict)
    greeted_identity_at: dict[str, float] = field(default_factory=dict)
    last_prompt_at: float = 0.0
    next_ephemeral_track: int = 0
    coverage: dict[str, int] = field(default_factory=lambda: {"tracked": 0, "matched": 0, "scanning": 0})
    prompt_hint: str = ""
    last_summary: str = ""


class SocialSceneOrchestrator:
    """
    Stateful session-aware social scene event engine.

    This is intentionally deterministic and lightweight to run on every
    realtime detection update.
    """

    _PERSON_LEAVE_GRACE_SEC = 2.6
    _TRACK_TTL_SEC = 12.0
    _EVENT_COOLDOWN_SEC = 15.0
    _OBJECT_EVENT_COOLDOWN_SEC = 45.0
    _PROMPT_COOLDOWN_SEC = 28.0
    _MAX_EVENTS = 200

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}
        self._identity_event_cooldown_sec = self._read_float_env(
            "JARVIS_SOCIAL_IDENTIFIED_EVENT_COOLDOWN_SEC",
            300.0,
        )
        self._identity_greet_cooldown_sec = self._read_float_env(
            "JARVIS_SOCIAL_IDENTITY_GREET_COOLDOWN_SEC",
            1800.0,
        )

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(str(session_id or "").strip(), None)

    def ingest_detections(
        self,
        *,
        session_id: str,
        detections: list[dict[str, Any]],
        now: float | None = None,
    ) -> dict[str, Any]:
        sid = str(session_id or "").strip()
        if not sid:
            return {"events": [], "coverage": {"tracked": 0, "matched": 0, "scanning": 0}, "prompt_hint": "", "summary": ""}
        ts = float(now if now is not None else time.time())
        state = self._sessions.get(sid)
        if state is None:
            state = _SessionState(session_id=sid)
            self._sessions[sid] = state
        events: list[dict[str, Any]] = []

        rows = [d for d in detections if isinstance(d, dict)]
        persons = [d for d in rows if str(d.get("label", "")).strip().lower() == "person"]
        active_track_ids: set[str] = set()

        for idx, det in enumerate(persons):
            tid = str(det.get("trackId") or det.get("track_id") or "").strip()
            if not tid:
                state.next_ephemeral_track += 1
                tid = f"ep-{state.next_ephemeral_track}"
            active_track_ids.add(tid)
            bbox_raw = det.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
                bbox = (
                    float(bbox_raw[0] or 0.0),
                    float(bbox_raw[1] or 0.0),
                    float(bbox_raw[2] or 0.0),
                    float(bbox_raw[3] or 0.0),
                )
            else:
                bbox = (0.0, 0.0, 0.0, 0.0)
            person_id = str(det.get("personId") or det.get("person_id") or "").strip()
            identity = str(det.get("identity", "")).strip()
            score = float(det.get("score", 0.0) or 0.0)
            id_score = float(det.get("identityScore", 0.0) or 0.0)
            prev = state.tracks.get(tid)
            state.tracks[tid] = _TrackState(
                track_id=tid,
                bbox=bbox,
                first_seen_at=prev.first_seen_at if prev else ts,
                last_seen_at=ts,
                person_id=person_id,
                identity=identity,
                left_announced=False,
            )
            if prev is None and score >= 0.48:
                evt = self._event(
                    event_type="person_entered",
                    text="A person entered the scene.",
                    severity="medium",
                    at=ts,
                    track_id=tid,
                    metadata={"score": score, "index": idx},
                )
                self._append_event_if_allowed(state, events, evt, key=f"enter:{tid}", now=ts, cooldown=self._EVENT_COOLDOWN_SEC)
            if person_id and identity and (prev is None or not prev.person_id):
                evt = self._event(
                    event_type="person_identified",
                    text=f"{identity} is now identified in the room ({round(id_score * 100)}%).",
                    severity="high",
                    at=ts,
                    track_id=tid,
                    person_id=person_id,
                    metadata={"identity_score": id_score, "score": score},
                )
                self._append_event_if_allowed(
                    state,
                    events,
                    evt,
                    key=f"identified:{person_id}",
                    now=ts,
                    cooldown=self._identity_event_cooldown_sec,
                )

        for tid, trk in list(state.tracks.items()):
            if tid in active_track_ids:
                continue
            if not trk.left_announced and (ts - float(trk.last_seen_at)) >= self._PERSON_LEAVE_GRACE_SEC:
                trk.left_announced = True
                state.tracks[tid] = trk
                label = trk.identity or "A person"
                evt = self._event(
                    event_type="person_left",
                    text=f"{label} left the scene.",
                    severity="low",
                    at=ts,
                    track_id=tid,
                    person_id=trk.person_id or "",
                    metadata={},
                )
                self._append_event_if_allowed(state, events, evt, key=f"left:{tid}", now=ts, cooldown=self._EVENT_COOLDOWN_SEC)
            if (ts - float(trk.last_seen_at)) >= self._TRACK_TTL_SEC:
                state.tracks.pop(tid, None)

        seen_labels: set[str] = set()
        for det in rows:
            label = str(det.get("label", "")).strip().lower()
            if not label or label == "person" or label in seen_labels:
                continue
            seen_labels.add(label)
            score = float(det.get("score", 0.0) or 0.0)
            if score < 0.58:
                continue
            prev_at = float(state.object_seen_at.get(label, 0.0) or 0.0)
            if (ts - prev_at) < self._OBJECT_EVENT_COOLDOWN_SEC:
                continue
            state.object_seen_at[label] = ts
            evt = self._event(
                event_type="object_spotted",
                text=f"New object observed: {label}.",
                severity="low",
                at=ts,
                metadata={"label": label, "score": score},
            )
            events.append(evt)

        tracked = len(persons)
        matched = len([p for p in persons if str(p.get("personId") or p.get("person_id") or "").strip()])
        coverage = {"tracked": tracked, "matched": matched, "scanning": max(0, tracked - matched)}
        state.coverage = coverage
        state.prompt_hint = self._build_prompt_hint(events=events, coverage=coverage, now=ts, state=state)
        state.last_summary = ("Scene events: " + " ".join(str(e.get("text", "")).strip() for e in events if str(e.get("text", "")).strip())).strip()
        if events:
            state.events = (events + state.events)[: self._MAX_EVENTS]
        self._sessions[sid] = state
        return {
            "events": events,
            "coverage": dict(coverage),
            "prompt_hint": state.prompt_hint,
            "summary": state.last_summary,
        }

    def get_timeline(self, session_id: str, *, limit: int = 40) -> dict[str, Any]:
        sid = str(session_id or "").strip()
        state = self._sessions.get(sid)
        if state is None:
            return {
                "session_id": sid,
                "count": 0,
                "coverage": {"tracked": 0, "matched": 0, "scanning": 0},
                "prompt_hint": "",
                "last_summary": "",
                "items": [],
            }
        lim = max(1, min(200, int(limit)))
        return {
            "session_id": sid,
            "count": min(lim, len(state.events)),
            "coverage": dict(state.coverage),
            "prompt_hint": state.prompt_hint,
            "last_summary": state.last_summary,
            "items": [dict(x) for x in state.events[:lim]],
        }

    def explain_event(self, session_id: str, *, event_id: str = "") -> dict[str, Any]:
        sid = str(session_id or "").strip()
        state = self._sessions.get(sid)
        if state is None:
            return {"session_id": sid, "found": False, "reason": "session_not_found"}
        target = None
        if event_id:
            target = next((e for e in state.events if str(e.get("event_id", "")) == event_id), None)
        else:
            target = state.events[0] if state.events else None
        if not isinstance(target, dict):
            return {"session_id": sid, "found": False, "reason": "event_not_found"}
        return {
            "session_id": sid,
            "found": True,
            "event": dict(target),
            "explanation": {
                "trigger": str(target.get("type", "")),
                "severity": str(target.get("severity", "")),
                "policy": "cooldown_and_novelty",
                "prompt_hint": state.prompt_hint,
                "coverage": dict(state.coverage),
            },
        }

    @staticmethod
    def _event(
        *,
        event_type: str,
        text: str,
        severity: str,
        at: float,
        track_id: str = "",
        person_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "event_id": f"evt-{uuid.uuid4().hex[:14]}",
            "type": str(event_type or "").strip(),
            "text": str(text or "").strip(),
            "severity": str(severity or "low").strip(),
            "at_ms": int(max(0.0, float(at)) * 1000),
            "track_id": str(track_id or "").strip(),
            "person_id": str(person_id or "").strip(),
            "metadata": dict(metadata or {}),
        }

    @staticmethod
    def _append_event_if_allowed(
        state: _SessionState,
        out: list[dict[str, Any]],
        event: dict[str, Any],
        *,
        key: str,
        now: float,
        cooldown: float,
    ) -> None:
        prev = float(state.cooldowns.get(key, 0.0) or 0.0)
        if (now - prev) < float(cooldown):
            return
        state.cooldowns[key] = now
        out.append(event)

    def _build_prompt_hint(self, *, events: list[dict[str, Any]], coverage: dict[str, int], now: float, state: _SessionState) -> str:
        if not events:
            return ""
        if (now - float(state.last_prompt_at or 0.0)) < self._PROMPT_COOLDOWN_SEC:
            return ""
        identified = [e for e in events if str(e.get("type", "")) == "person_identified"]
        if identified:
            names: list[str] = []
            for ev in identified:
                txt = str(ev.get("text", "")).strip()
                if " is now identified" in txt:
                    name = txt.split(" is now identified", 1)[0].strip()
                    key = str(ev.get("person_id", "")).strip() or name.lower()
                    last_greeted = float(state.greeted_identity_at.get(key, 0.0) or 0.0)
                    if (now - last_greeted) >= self._identity_greet_cooldown_sec:
                        names.append(name)
                        state.greeted_identity_at[key] = now
            if not names:
                return ""
            state.last_prompt_at = now
            if names:
                return f"Give a short social room update and greet {', '.join(names)} naturally."
            return "Give a short social room update about who was identified."
        scanning = int(coverage.get("scanning", 0) or 0)
        tracked = int(coverage.get("tracked", 0) or 0)
        if tracked > 0 and scanning > 0:
            state.last_prompt_at = now
            return f"Give a short social room update; {scanning} visible person(s) are still unidentified."
        return ""

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        raw = str(os.getenv(name, "")).strip()
        if not raw:
            return float(default)
        try:
            val = float(raw)
            return val if val >= 0.0 else float(default)
        except Exception:
            return float(default)
