"""
Retention mechanics for workflow checkpoints and runtime artifacts.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict


class RetentionManager:
    def __init__(
        self,
        *,
        checkpoint_retention_days: int = 7,
    ) -> None:
        self.checkpoint_retention_days = max(1, int(checkpoint_retention_days or 1))

    @classmethod
    def from_env(cls) -> "RetentionManager":
        return cls(
            checkpoint_retention_days=int(
                os.getenv("JARVIS_WORKFLOW_CHECKPOINT_RETENTION_DAYS", "7") or 7
            )
        )

    @staticmethod
    def _parse_iso(ts: str) -> float:
        try:
            return datetime.fromisoformat(str(ts)).timestamp()
        except Exception:
            return 0.0

    def prune_workflow_checkpoints(self, checkpoints: Dict[str, Dict[str, Any]]) -> int:
        if not isinstance(checkpoints, dict):
            return 0
        cutoff = time.time() - (self.checkpoint_retention_days * 86400)
        removed = 0
        for workflow_id, payload in list(checkpoints.items()):
            updated_at = ""
            if isinstance(payload, dict):
                updated_at = str(payload.get("updated_at", "")).strip()
            ts = self._parse_iso(updated_at)
            if ts > 0 and ts < cutoff:
                checkpoints.pop(workflow_id, None)
                removed += 1
        return removed

