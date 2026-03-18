"""
Audit logging utilities for JARVIS AI OS.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AuditEvent:
    event_type: str
    action: str
    request_id: str = ""
    actor: str = "system"
    resource: str = ""
    decision: str = "allow"
    success: bool = True
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """In-memory audit logger with optional JSONL persistence."""

    def __init__(self, persist_path: Optional[str] = None, max_events: int = 5000) -> None:
        self._events: List[AuditEvent] = []
        self._persist_path = Path(persist_path) if persist_path else None
        self._max_events = max_events
        self._lock = threading.RLock()

    def record(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]

            if self._persist_path is not None:
                self._persist_path.parent.mkdir(parents=True, exist_ok=True)
                with self._persist_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event.to_dict(), ensure_ascii=True) + "\n")

    def recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self._events[-limit:]]

