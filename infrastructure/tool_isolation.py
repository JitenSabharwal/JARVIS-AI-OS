"""
Tool isolation policy for workspace-scoped operations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


@dataclass(slots=True)
class ToolIsolationDecision:
    allowed: bool
    workspace_path: str = ""
    reason: str = ""


class ToolIsolationPolicy:
    def __init__(
        self,
        *,
        enabled: bool = True,
        allowed_roots: Iterable[str] = (),
    ) -> None:
        self.enabled = bool(enabled)
        self.allowed_roots = tuple(
            str(Path(root).expanduser().resolve())
            for root in allowed_roots
            if str(root).strip()
        )

    @classmethod
    def from_env(cls) -> "ToolIsolationPolicy":
        enabled = str(os.getenv("JARVIS_TOOL_ISOLATION_ENABLED", "true")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        raw = str(os.getenv("JARVIS_TOOL_ALLOWED_ROOTS", "")).strip()
        roots = [r.strip() for r in raw.split(",") if r.strip()]
        return cls(enabled=enabled, allowed_roots=roots)

    def validate_workspace(self, workspace_path: str) -> ToolIsolationDecision:
        workspace = str(workspace_path or "").strip()
        if not workspace:
            return ToolIsolationDecision(allowed=False, reason="workspace_missing")
        try:
            resolved = str(Path(workspace).expanduser().resolve())
        except Exception:
            return ToolIsolationDecision(allowed=False, reason="workspace_invalid_path")
        if not self.enabled:
            return ToolIsolationDecision(allowed=True, workspace_path=resolved)
        if not Path(resolved).exists():
            return ToolIsolationDecision(allowed=False, workspace_path=resolved, reason="workspace_not_found")
        if not Path(resolved).is_dir():
            return ToolIsolationDecision(allowed=False, workspace_path=resolved, reason="workspace_not_directory")
        if self.allowed_roots:
            for root in self.allowed_roots:
                if resolved == root or resolved.startswith(f"{root}/"):
                    return ToolIsolationDecision(allowed=True, workspace_path=resolved)
            return ToolIsolationDecision(
                allowed=False,
                workspace_path=resolved,
                reason="workspace_outside_allowed_roots",
            )
        return ToolIsolationDecision(allowed=True, workspace_path=resolved)

    def enforce_payload(
        self,
        *,
        capability: str,
        payload: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if not isinstance(payload, dict):
            return False, "payload_not_object"
        cap = str(capability or "").strip().lower()
        if cap not in {"update_codebase", "understand_codebase"}:
            return True, ""
        workspace = str(payload.get("workspace_path", "")).strip()
        decision = self.validate_workspace(workspace)
        if not decision.allowed:
            return False, decision.reason
        payload["workspace_path"] = decision.workspace_path
        return True, ""

