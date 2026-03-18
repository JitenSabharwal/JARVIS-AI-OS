"""
Event-driven automation rule engine.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from infrastructure.logger import get_logger

logger = get_logger(__name__)

AutomationAction = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass
class AutomationRule:
    rule_id: str
    name: str
    event_type: str
    action_name: str
    match: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    run_count: int = 0
    last_run_at: float | None = None
    last_status: str = "never"
    last_error: str = ""
    max_retries: int = 0
    retry_backoff_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "event_type": self.event_type,
            "action_name": self.action_name,
            "match": self.match,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "run_count": self.run_count,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "max_retries": self.max_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
        }


class AutomationEngine:
    """Simple event-driven rule engine with async action handlers."""

    def __init__(self) -> None:
        self._rules: Dict[str, AutomationRule] = {}
        self._actions: Dict[str, AutomationAction] = {}
        self._history: List[Dict[str, Any]] = []
        self._dead_letters: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def register_action(self, name: str, action: AutomationAction) -> None:
        with self._lock:
            self._actions[name] = action
        logger.info("Registered automation action '%s'", name)

    def create_rule(
        self,
        *,
        name: str,
        event_type: str,
        action_name: str,
        match: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        max_retries: int = 0,
        retry_backoff_seconds: float = 0.0,
    ) -> AutomationRule:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0")
        with self._lock:
            if action_name not in self._actions:
                raise ValueError(f"Unknown action_name: {action_name}")
            rule = AutomationRule(
                rule_id=str(uuid.uuid4()),
                name=name,
                event_type=event_type,
                action_name=action_name,
                match=match or {},
                enabled=enabled,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            self._rules[rule.rule_id] = rule
        return rule

    def list_rules(self) -> List[Dict[str, Any]]:
        with self._lock:
            rules = list(self._rules.values())
        return [r.to_dict() for r in rules]

    def get_rule(self, rule_id: str) -> Optional[AutomationRule]:
        with self._lock:
            return self._rules.get(rule_id)

    def set_rule_enabled(self, rule_id: str, enabled: bool) -> bool:
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule is None:
                return False
            rule.enabled = enabled
            return True

    async def process_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        with self._lock:
            rules = list(self._rules.values())
            actions = dict(self._actions)

        matched = [r for r in rules if self._matches(r, event_type, payload)]
        executions: List[Dict[str, Any]] = []
        for rule in matched:
            action = actions.get(rule.action_name)
            if action is None:
                rule.last_status = "failed"
                rule.last_error = f"action '{rule.action_name}' missing"
                continue

            started = time.time()
            attempts = 0
            result = None
            status = "failed"
            error = ""
            max_attempts = 1 + max(0, rule.max_retries)

            for attempt in range(1, max_attempts + 1):
                attempts = attempt
                try:
                    result = await asyncio.wait_for(action(payload), timeout=timeout_seconds)
                    status = "completed"
                    error = ""
                    break
                except Exception as exc:  # noqa: BLE001
                    status = "failed"
                    error = str(exc)
                    if attempt < max_attempts and rule.retry_backoff_seconds > 0:
                        await asyncio.sleep(rule.retry_backoff_seconds)

            with self._lock:
                rule.run_count += 1
                rule.last_run_at = time.time()
                rule.last_status = status
                rule.last_error = error

                event_record = {
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "event_type": event_type,
                    "status": status,
                    "error": error,
                    "attempts": attempts,
                    "duration_ms": round((time.time() - started) * 1000, 2),
                    "result": result,
                    "timestamp": time.time(),
                }
                self._history.append(event_record)
                if len(self._history) > 1000:
                    self._history = self._history[-1000:]
                if status == "failed":
                    dead_letter = {
                        "dead_letter_id": str(uuid.uuid4()),
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "event_type": event_type,
                        "payload": payload,
                        "error": error,
                        "attempts": attempts,
                        "timestamp": time.time(),
                        "replay_count": 0,
                        "last_replay_at": None,
                        "last_replay_status": "never",
                        "last_replay_result": None,
                    }
                    self._dead_letters.append(dead_letter)
                    if len(self._dead_letters) > 1000:
                        self._dead_letters = self._dead_letters[-1000:]
            executions.append(event_record)

        return {
            "event_type": event_type,
            "matched_rules": len(matched),
            "executions": executions,
        }

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return self._history[-limit:]

    def get_dead_letters(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return self._dead_letters[-limit:]

    def dead_letter_count(self) -> int:
        with self._lock:
            return len(self._dead_letters)

    async def replay_dead_letter(
        self,
        dead_letter_id: str,
        *,
        timeout_seconds: float = 10.0,
        remove_on_success: bool = False,
        payload_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Replay a dead-lettered event and return execution metadata."""
        with self._lock:
            idx = next(
                (i for i, dl in enumerate(self._dead_letters) if dl.get("dead_letter_id") == dead_letter_id),
                -1,
            )
            if idx < 0:
                raise KeyError(f"Dead letter not found: {dead_letter_id}")
            original = dict(self._dead_letters[idx])
            event_type = str(original.get("event_type", ""))
            payload = payload_override if payload_override is not None else dict(original.get("payload", {}))

        if not event_type:
            raise ValueError("Dead letter missing event_type")
        if not isinstance(payload, dict):
            raise ValueError("Dead letter payload must be an object")

        replay_result = await self.process_event(
            event_type,
            payload,
            timeout_seconds=max(0.1, float(timeout_seconds)),
        )
        executions = replay_result.get("executions", []) or []
        succeeded = bool(executions) and all(e.get("status") == "completed" for e in executions)

        with self._lock:
            idx = next(
                (i for i, dl in enumerate(self._dead_letters) if dl.get("dead_letter_id") == dead_letter_id),
                -1,
            )
            if idx >= 0:
                dead_letter = self._dead_letters[idx]
                dead_letter["replay_count"] = int(dead_letter.get("replay_count", 0)) + 1
                dead_letter["last_replay_at"] = time.time()
                dead_letter["last_replay_status"] = "completed" if succeeded else "failed"
                dead_letter["last_replay_result"] = replay_result
                if succeeded and remove_on_success:
                    self._dead_letters.pop(idx)

        return {
            "dead_letter_id": dead_letter_id,
            "event_type": event_type,
            "replayed": True,
            "succeeded": succeeded,
            "removed": bool(succeeded and remove_on_success),
            "result": replay_result,
        }

    def resolve_dead_letter(self, dead_letter_id: str, *, reason: str = "manual_resolve") -> Dict[str, Any]:
        """Resolve (remove) a dead letter entry and log in automation history."""
        with self._lock:
            idx = next(
                (i for i, dl in enumerate(self._dead_letters) if dl.get("dead_letter_id") == dead_letter_id),
                -1,
            )
            if idx < 0:
                raise KeyError(f"Dead letter not found: {dead_letter_id}")
            dead_letter = self._dead_letters.pop(idx)
            resolution = {
                "dead_letter_id": dead_letter_id,
                "resolved": True,
                "reason": reason,
                "resolved_at": time.time(),
                "event_type": dead_letter.get("event_type", ""),
                "rule_id": dead_letter.get("rule_id", ""),
            }
            self._history.append(
                {
                    "rule_id": dead_letter.get("rule_id", ""),
                    "rule_name": dead_letter.get("rule_name", ""),
                    "event_type": dead_letter.get("event_type", ""),
                    "status": "dead_letter_resolved",
                    "error": dead_letter.get("error", ""),
                    "attempts": dead_letter.get("attempts", 0),
                    "duration_ms": 0.0,
                    "result": resolution,
                    "timestamp": resolution["resolved_at"],
                }
            )
            if len(self._history) > 1000:
                self._history = self._history[-1000:]
            return resolution

    @staticmethod
    def _matches(rule: AutomationRule, event_type: str, payload: Dict[str, Any]) -> bool:
        if not rule.enabled or rule.event_type != event_type:
            return False
        for key, expected in rule.match.items():
            if payload.get(key) != expected:
                return False
        return True
