"""
Optional LangGraph adapter with deterministic fallback execution.
"""

from __future__ import annotations

from typing import Any, Dict, List


class LangGraphWorkflowAdapter:
    """
    Wrapper that can use LangGraph when installed; otherwise falls back to
    deterministic topological wave planning.
    """

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._langgraph_available = self._detect_langgraph()

    @property
    def available(self) -> bool:
        return self.enabled and self._langgraph_available

    @staticmethod
    def _detect_langgraph() -> bool:
        try:
            import langgraph  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def build_execution_waves(self, steps: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Return execution waves from a dependency graph.

        The adapter currently uses a deterministic fallback algorithm even when
        LangGraph is available because workflow execution in this repository is
        still orchestrator-native.
        """
        remaining = {str(s["name"]): set(str(d) for d in s.get("depends_on", [])) for s in steps}
        waves: List[List[str]] = []
        resolved: set[str] = set()
        while remaining:
            ready = sorted([name for name, deps in remaining.items() if deps.issubset(resolved)])
            if not ready:
                unresolved = ", ".join(sorted(remaining.keys()))
                raise ValueError(f"Cycle or unresolved dependencies in workflow: {unresolved}")
            waves.append(ready)
            for name in ready:
                resolved.add(name)
                remaining.pop(name, None)
        return waves


__all__ = ["LangGraphWorkflowAdapter"]
