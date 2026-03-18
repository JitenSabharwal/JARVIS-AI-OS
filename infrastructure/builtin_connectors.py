"""
Built-in connector set for production-profile API mode.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from infrastructure.connectors import BaseConnector, ConnectorPolicy, ConnectorRegistry


class CalendarConnector(BaseConnector):
    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "calendar"

    @property
    def description(self) -> str:
        return "Calendar connector (create/list event)."

    async def invoke(self, operation: str, params: Dict[str, Any]) -> Any:
        if operation == "create_event":
            title = str(params.get("title", "")).strip()
            start = str(params.get("start", "")).strip()
            end = str(params.get("end", "")).strip()
            if not title or not start or not end:
                raise ValueError("create_event requires title/start/end")
            event = {
                "event_id": f"evt-{int(time.time() * 1000)}-{len(self._events) + 1}",
                "title": title,
                "start": start,
                "end": end,
                "created_at": time.time(),
            }
            self._events.append(event)
            return event
        if operation == "list_events":
            limit = int(params.get("limit", 20))
            limit = max(1, min(500, limit))
            return {"events": self._events[-limit:], "count": min(limit, len(self._events))}
        raise ValueError(f"Unsupported operation for calendar connector: {operation}")

    async def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "connector": self.name, "events_cached": len(self._events)}


class MailConnector(BaseConnector):
    def __init__(self) -> None:
        self._sent: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "mail"

    @property
    def description(self) -> str:
        return "Mail connector (send/list sent messages)."

    async def invoke(self, operation: str, params: Dict[str, Any]) -> Any:
        if operation == "send_mail":
            to = str(params.get("to", "")).strip()
            subject = str(params.get("subject", "")).strip()
            body = str(params.get("body", "")).strip()
            if not to or "@" not in to:
                raise ValueError("send_mail requires a valid 'to' email")
            if not subject:
                raise ValueError("send_mail requires 'subject'")
            if not body:
                raise ValueError("send_mail requires 'body'")
            msg = {
                "message_id": f"msg-{int(time.time() * 1000)}-{len(self._sent) + 1}",
                "to": to,
                "subject": subject,
                "body_preview": body[:80],
                "sent_at": time.time(),
            }
            self._sent.append(msg)
            return msg
        if operation == "list_sent":
            limit = int(params.get("limit", 20))
            limit = max(1, min(500, limit))
            return {"messages": self._sent[-limit:], "count": min(limit, len(self._sent))}
        raise ValueError(f"Unsupported operation for mail connector: {operation}")

    async def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "connector": self.name, "messages_cached": len(self._sent)}


class FilesNotificationsConnector(BaseConnector):
    def __init__(self, base_dir: str = "data/connector_files") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._notifications: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "files_notifications"

    @property
    def description(self) -> str:
        return "Files + notifications connector (write/read notes, emit notifications)."

    def _resolve_safe(self, relative_path: str) -> Path:
        rel = relative_path.strip().lstrip("/")
        if not rel:
            raise ValueError("path is required")
        target = (self._base_dir / rel).resolve()
        base = self._base_dir.resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError("path escapes connector base directory")
        return target

    async def invoke(self, operation: str, params: Dict[str, Any]) -> Any:
        if operation == "write_note":
            path = self._resolve_safe(str(params.get("path", "")).strip())
            content = str(params.get("content", ""))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {"path": str(path), "bytes": len(content.encode("utf-8"))}
        if operation == "read_note":
            path = self._resolve_safe(str(params.get("path", "")).strip())
            if not path.exists():
                raise FileNotFoundError(f"File does not exist: {path}")
            text = path.read_text(encoding="utf-8")
            return {"path": str(path), "content": text}
        if operation == "notify":
            channel = str(params.get("channel", "")).strip() or "general"
            message = str(params.get("message", "")).strip()
            if not message:
                raise ValueError("notify requires 'message'")
            item = {
                "notification_id": f"ntf-{int(time.time() * 1000)}-{len(self._notifications) + 1}",
                "channel": channel,
                "message": message,
                "timestamp": time.time(),
            }
            self._notifications.append(item)
            return item
        if operation == "list_notifications":
            limit = int(params.get("limit", 20))
            limit = max(1, min(500, limit))
            return {
                "notifications": self._notifications[-limit:],
                "count": min(limit, len(self._notifications)),
            }
        raise ValueError(f"Unsupported operation for files_notifications connector: {operation}")

    async def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "connector": self.name,
            "base_dir": str(self._base_dir),
            "notifications_cached": len(self._notifications),
        }


def build_default_connector_registry(data_dir: str = "data") -> ConnectorRegistry:
    """
    Build a connector registry with three production-profile connectors:
    calendar, mail, files_notifications.
    """
    registry = ConnectorRegistry()
    registry.register(
        CalendarConnector(),
        policy=ConnectorPolicy(
            required_scopes_by_operation={
                "create_event": {"connector:calendar:write"},
                "list_events": {"connector:calendar:read"},
            },
            failure_threshold=3,
            recovery_timeout_seconds=30.0,
        ),
    )
    registry.register(
        MailConnector(),
        policy=ConnectorPolicy(
            required_scopes_by_operation={
                "send_mail": {"connector:mail:send"},
                "list_sent": {"connector:mail:read"},
            },
            failure_threshold=3,
            recovery_timeout_seconds=30.0,
        ),
    )
    registry.register(
        FilesNotificationsConnector(base_dir=str(Path(data_dir) / "connector_files")),
        policy=ConnectorPolicy(
            required_scopes_by_operation={
                "write_note": {"connector:files:write"},
                "read_note": {"connector:files:read"},
                "notify": {"connector:notify:send"},
                "list_notifications": {"connector:notify:read"},
            },
            failure_threshold=3,
            recovery_timeout_seconds=30.0,
        ),
    )
    return registry
