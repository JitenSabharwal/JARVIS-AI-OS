"""
Append-only artifact persistence for task/workflow execution traces.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict


class ArtifactStore:
    def __init__(
        self,
        *,
        enabled: bool = True,
        base_path: str = "data/artifacts",
        max_age_days: int = 14,
    ) -> None:
        self.enabled = bool(enabled)
        self.base_path = str(base_path or "data/artifacts").strip()
        self.max_age_days = max(1, int(max_age_days or 1))

    @classmethod
    def from_env(cls) -> "ArtifactStore":
        enabled = str(os.getenv("JARVIS_ARTIFACT_PERSIST_ENABLED", "true")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            enabled=enabled,
            base_path=str(os.getenv("JARVIS_ARTIFACT_PERSIST_PATH", "data/artifacts")),
            max_age_days=int(os.getenv("JARVIS_ARTIFACT_RETENTION_DAYS", "14") or 14),
        )

    def append(self, *, artifact_type: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        now = time.time()
        event = {
            "artifact_type": str(artifact_type or "generic"),
            "ts": now,
            "payload": payload if isinstance(payload, dict) else {"value": payload},
        }
        path = Path(self.base_path)
        path.mkdir(parents=True, exist_ok=True)
        file = path / f"{time.strftime('%Y-%m-%d')}.jsonl"
        with file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=True) + "\n")

    def cleanup(self) -> Dict[str, int]:
        if not self.enabled:
            return {"removed_files": 0}
        root = Path(self.base_path)
        if not root.exists():
            return {"removed_files": 0}
        cutoff = time.time() - (self.max_age_days * 86400)
        removed = 0
        for item in root.glob("*.jsonl"):
            try:
                if item.stat().st_mtime < cutoff:
                    item.unlink(missing_ok=True)
                    removed += 1
            except Exception:
                continue
        return {"removed_files": removed}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "base_path": self.base_path,
            "max_age_days": self.max_age_days,
        }

