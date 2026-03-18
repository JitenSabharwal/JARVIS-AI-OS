"""
Proactive event engine for policy-safe, preference-aware suggestions.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProactiveSuggestion:
    suggestion_id: str
    user_id: str
    category: str
    title: str
    message: str
    priority: str = "normal"
    confidence: float = 0.7
    requires_human: bool = False
    status: str = "open"
    snoozed_until: float | None = None
    acknowledged_at: float | None = None
    dismissed_at: float | None = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "user_id": self.user_id,
            "category": self.category,
            "title": self.title,
            "message": self.message,
            "priority": self.priority,
            "confidence": self.confidence,
            "requires_human": self.requires_human,
            "status": self.status,
            "snoozed_until": self.snoozed_until,
            "acknowledged_at": self.acknowledged_at,
            "dismissed_at": self.dismissed_at,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


class ProactiveEventEngine:
    """In-memory proactive recommendation engine."""

    def __init__(self) -> None:
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._suggestions: Dict[str, List[ProactiveSuggestion]] = {}
        self._suggestions_by_id: Dict[str, ProactiveSuggestion] = {}
        self._events: List[Dict[str, Any]] = []
        self._dedupe_last_emitted: Dict[str, float] = {}

    def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        if not user_id.strip():
            raise ValueError("user_id is required")
        profile = self._profiles.setdefault(user_id, self._default_profile())
        for key, value in preferences.items():
            profile[key] = value
        profile["updated_at"] = time.time()
        return {"user_id": user_id, "profile": dict(profile)}

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        profile = self._profiles.setdefault(user_id, self._default_profile())
        return {"user_id": user_id, "profile": dict(profile)}

    def ingest_event(self, *, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        event = {
            "event_id": f"evt-{uuid.uuid4().hex}",
            "event_type": event_type,
            "payload": dict(payload),
            "created_at": time.time(),
        }
        self._events.append(event)
        user_id = str(payload.get("user_id", "default")).strip() or "default"
        generated = self._generate_suggestions_from_event(user_id=user_id, event_type=event_type, payload=payload)
        self._suggestions.setdefault(user_id, []).extend(generated)
        for suggestion in generated:
            self._suggestions_by_id[suggestion.suggestion_id] = suggestion
        return {
            "event_id": event["event_id"],
            "event_type": event_type,
            "generated_count": len(generated),
            "suggestions": [s.to_dict() for s in generated],
        }

    def evaluate_autonomous_action(
        self,
        *,
        user_id: str,
        action_name: str,
        category: str = "general",
        priority: str = "normal",
    ) -> Dict[str, Any]:
        profile = self._profiles.setdefault(user_id, self._default_profile())
        action_low = action_name.strip().lower()
        blocked_tokens = tuple(str(t).lower() for t in profile.get("blocked_action_tokens", []))
        if any(token and token in action_low for token in blocked_tokens):
            return {
                "allowed": False,
                "requires_approval": True,
                "reason": "blocked_action_token",
            }
        if not bool(profile.get("autonomous_actions_enabled", False)):
            return {
                "allowed": False,
                "requires_approval": True,
                "reason": "autonomous_actions_disabled",
            }

        allowed_categories = set(profile.get("allowed_autonomous_categories", []))
        if category not in allowed_categories:
            return {
                "allowed": False,
                "requires_approval": True,
                "reason": "category_not_allowed",
            }

        risk_tolerance = str(profile.get("risk_tolerance", "medium")).lower()
        max_priority = {"low": "normal", "medium": "high", "high": "urgent"}.get(risk_tolerance, "high")
        if self._priority_rank(priority) > self._priority_rank(max_priority):
            return {
                "allowed": False,
                "requires_approval": True,
                "reason": "priority_exceeds_risk_tolerance",
            }
        return {
            "allowed": True,
            "requires_approval": False,
            "reason": "allowed",
        }

    def list_suggestions(
        self,
        *,
        user_id: str,
        max_items: int = 20,
        include_low_priority: bool = False,
    ) -> Dict[str, Any]:
        profile = self._profiles.setdefault(user_id, self._default_profile())
        if not bool(profile.get("proactive_enabled", True)):
            return {"user_id": user_id, "count": 0, "suggestions": []}

        items = list(self._suggestions.get(user_id, []))
        now = time.time()
        items = [
            s
            for s in items
            if s.status != "dismissed" and (s.snoozed_until is None or s.snoozed_until <= now)
        ]
        if not include_low_priority:
            items = [s for s in items if s.priority != "low"]
        items.sort(key=lambda s: (self._priority_rank(s.priority), s.created_at), reverse=True)
        limited = items[: max(1, min(200, int(max_items)))]
        return {"user_id": user_id, "count": len(limited), "suggestions": [s.to_dict() for s in limited]}

    def acknowledge_suggestion(self, *, suggestion_id: str) -> Dict[str, Any]:
        suggestion = self._suggestions_by_id.get(suggestion_id)
        if suggestion is None:
            raise KeyError(f"Unknown suggestion_id: {suggestion_id}")
        suggestion.status = "acknowledged"
        suggestion.acknowledged_at = time.time()
        return {"updated": True, "suggestion": suggestion.to_dict()}

    def dismiss_suggestion(self, *, suggestion_id: str) -> Dict[str, Any]:
        suggestion = self._suggestions_by_id.get(suggestion_id)
        if suggestion is None:
            raise KeyError(f"Unknown suggestion_id: {suggestion_id}")
        suggestion.status = "dismissed"
        suggestion.dismissed_at = time.time()
        return {"updated": True, "suggestion": suggestion.to_dict()}

    def snooze_suggestion(self, *, suggestion_id: str, seconds: int) -> Dict[str, Any]:
        suggestion = self._suggestions_by_id.get(suggestion_id)
        if suggestion is None:
            raise KeyError(f"Unknown suggestion_id: {suggestion_id}")
        duration = max(1, int(seconds))
        suggestion.status = "snoozed"
        suggestion.snoozed_until = time.time() + duration
        return {"updated": True, "suggestion": suggestion.to_dict()}

    def _generate_suggestions_from_event(
        self,
        *,
        user_id: str,
        event_type: str,
        payload: Dict[str, Any],
    ) -> List[ProactiveSuggestion]:
        profile = self._profiles.setdefault(user_id, self._default_profile())
        if not bool(profile.get("proactive_enabled", True)):
            return []

        generated: List[ProactiveSuggestion] = []

        if event_type == "calendar_upcoming":
            title = str(payload.get("title", "Upcoming event")).strip() or "Upcoming event"
            starts_in = str(payload.get("starts_in", "soon"))
            generated.append(
                self._make_suggestion(
                    user_id=user_id,
                    category="calendar",
                    title=f"Prep: {title}",
                    message=f"You have '{title}' starting {starts_in}. Do you want a prep checklist?",
                    priority="normal",
                    confidence=0.78,
                )
            )

        elif event_type == "task_overdue":
            task_name = str(payload.get("task", "Task")).strip() or "Task"
            generated.append(
                self._make_suggestion(
                    user_id=user_id,
                    category="tasks",
                    title=f"Overdue: {task_name}",
                    message=f"'{task_name}' is overdue. I can draft a recovery plan.",
                    priority="high",
                    confidence=0.82,
                )
            )

        elif event_type == "anomaly_detected":
            anomaly = str(payload.get("anomaly", "System anomaly")).strip() or "System anomaly"
            generated.append(
                self._make_suggestion(
                    user_id=user_id,
                    category="anomaly",
                    title="Anomaly detected",
                    message=f"Detected anomaly: {anomaly}. Recommend manual review before any action.",
                    priority="urgent",
                    confidence=0.88,
                    requires_human=True,
                )
            )

        elif event_type == "digest_ready":
            topic = str(payload.get("topic", "watchlist")).strip() or "watchlist"
            generated.append(
                self._make_suggestion(
                    user_id=user_id,
                    category="research",
                    title=f"Digest ready: {topic}",
                    message=f"Your latest digest for '{topic}' is ready. Open summary now?",
                    priority="low",
                    confidence=0.7,
                )
            )
        elif event_type == "reminder_due":
            reminder = str(payload.get("reminder", "Reminder")).strip() or "Reminder"
            generated.append(
                self._make_suggestion(
                    user_id=user_id,
                    category="reminder",
                    title=f"Reminder: {reminder}",
                    message=f"Scheduled reminder: {reminder}.",
                    priority="normal",
                    confidence=0.74,
                )
            )

        for s in generated:
            self._apply_safety_policy(profile=profile, suggestion=s, event_type=event_type, payload=payload)

        # Apply simple dedupe cooldown.
        cooldown = max(0, int(profile.get("cooldown_seconds", 300)))
        accepted: List[ProactiveSuggestion] = []
        now = time.time()
        for s in generated:
            dedupe_key = f"{user_id}:{s.category}:{s.title}"
            last_ts = self._dedupe_last_emitted.get(dedupe_key, 0.0)
            if now - last_ts < cooldown:
                continue
            self._dedupe_last_emitted[dedupe_key] = now
            accepted.append(s)
        return accepted

    @staticmethod
    def _default_profile() -> Dict[str, Any]:
        return {
            "proactive_enabled": True,
            "risk_tolerance": "medium",
            "cooldown_seconds": 300,
            "autonomous_actions_enabled": False,
            "allowed_autonomous_categories": ["calendar", "research", "reminder"],
            "blocked_action_tokens": ["delete", "transfer", "wire", "execute", "shell", "email_send"],
            "updated_at": time.time(),
        }

    @staticmethod
    def _priority_rank(priority: str) -> int:
        mapping = {"low": 1, "normal": 2, "high": 3, "urgent": 4}
        return mapping.get(priority, 2)

    @staticmethod
    def _make_suggestion(
        *,
        user_id: str,
        category: str,
        title: str,
        message: str,
        priority: str,
        confidence: float,
        requires_human: bool = False,
    ) -> ProactiveSuggestion:
        return ProactiveSuggestion(
            suggestion_id=f"sug-{uuid.uuid4().hex}",
            user_id=user_id,
            category=category,
            title=title,
            message=message,
            priority=priority,
            confidence=max(0.0, min(1.0, float(confidence))),
            requires_human=requires_human,
        )

    def _apply_safety_policy(
        self,
        *,
        profile: Dict[str, Any],
        suggestion: ProactiveSuggestion,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        reasons: List[str] = []
        if event_type == "anomaly_detected":
            suggestion.requires_human = True
            reasons.append("anomaly_detected")

        risk_tolerance = str(profile.get("risk_tolerance", "medium")).lower()
        max_priority = {"low": "normal", "medium": "high", "high": "urgent"}.get(risk_tolerance, "high")
        if self._priority_rank(suggestion.priority) > self._priority_rank(max_priority):
            suggestion.requires_human = True
            reasons.append("priority_exceeds_risk_tolerance")

        if bool(payload.get("autonomous", False)):
            decision = self.evaluate_autonomous_action(
                user_id=suggestion.user_id,
                action_name=str(payload.get("proposed_action", suggestion.title)),
                category=suggestion.category,
                priority=suggestion.priority,
            )
            if decision.get("requires_approval", False):
                suggestion.requires_human = True
                reasons.append(str(decision.get("reason", "approval_required")))
            suggestion.metadata["autonomous_decision"] = decision

        if reasons:
            suggestion.metadata["safety_reasons"] = reasons


__all__ = ["ProactiveEventEngine", "ProactiveSuggestion"]
