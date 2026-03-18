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
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "event_type": event_type,
                        "payload": payload,
                        "error": error,
                        "attempts": attempts,
                        "timestamp": time.time(),
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

    @staticmethod
    def _matches(rule: AutomationRule, event_type: str, payload: Dict[str, Any]) -> bool:
        if not rule.enabled or rule.event_type != event_type:
            return False
        for key, expected in rule.match.items():
            if payload.get(key) != expected:
                return False
        return True
