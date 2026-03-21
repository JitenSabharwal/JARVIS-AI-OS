"""
Optional LangGraph adapter with deterministic fallback execution.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List


class LangGraphWorkflowAdapter:
    """
    Wrapper that can use LangGraph when installed; otherwise falls back to
    deterministic topological wave planning.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_wave_size: int = 0,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_wave_size = max(0, int(max_wave_size or 0))
        self._langgraph_available = self._detect_langgraph()
        self.last_plan_meta: Dict[str, Any] = {
            "engine": "native",
            "steps_total": 0,
            "waves_total": 0,
            "max_wave_size": self.max_wave_size,
            "langgraph_available": self._langgraph_available,
        }

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
        """
        flow = self.build_multi_agent_flow(steps)
        waves = flow.get("waves", [])
        if not isinstance(waves, list):
            return []
        return [list(w) for w in waves if isinstance(w, list)]

    def build_multi_agent_flow(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a multi-agent flow plan.

        Returns object:
        {
          "waves": [["step_a"], ["step_b","step_c"], ...],
          "step_hints": {
            "step_a": {"lane": "...", "capability": "...", "fan_in": 0},
            ...
          },
          "engine": "langgraph|native|native_fallback"
        }
        """
        normalized = self._normalize_steps(steps)
        step_hints = {
            str(s["name"]): {
                "lane": self._lane_for_capability(str(s.get("capability", ""))),
                "capability": str(s.get("capability", "")),
                "fan_in": len(list(s.get("depends_on", []))),
            }
            for s in normalized
        }
        self.last_plan_meta = {
            "engine": "native",
            "steps_total": len(normalized),
            "waves_total": 0,
            "max_wave_size": self.max_wave_size,
            "langgraph_available": self._langgraph_available,
            "strategy": "lane_aware_dependency_waves",
        }
        engine = "native"
        if self.available:
            try:
                waves = self._build_execution_waves_with_langgraph(normalized)
                engine = "langgraph"
                self.last_plan_meta.update({"engine": engine, "waves_total": len(waves)})
                return {"waves": waves, "step_hints": step_hints, "engine": engine}
            except Exception as exc:  # noqa: BLE001
                engine = "native_fallback"
                self.last_plan_meta.update(
                    {
                        "engine": engine,
                        "fallback_reason": str(exc),
                    }
                )
        waves = self._build_execution_waves_native(normalized)
        self.last_plan_meta.update({"waves_total": len(waves)})
        return {"waves": waves, "step_hints": step_hints, "engine": engine}

    def init_execution_state(
        self,
        steps: List[Dict[str, Any]],
        *,
        max_retries: int = 1,
    ) -> Dict[str, Any]:
        normalized = self._normalize_steps(steps)
        step_state = {
            str(s["name"]): {
                "status": "pending",
                "depends_on": list(s.get("depends_on", [])),
                "attempts": 0,
                "max_retries": max(0, int(max_retries)),
            }
            for s in normalized
        }
        return {
            "steps": step_state,
            "last_events": [],
            "ready": self._compute_ready_steps(step_state),
            "terminal": False,
            "failed_steps": [],
        }

    def transition_execution_state(
        self,
        state: Dict[str, Any],
        events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        cur = copy.deepcopy(state if isinstance(state, dict) else {})
        steps = cur.get("steps", {})
        if not isinstance(steps, dict):
            steps = {}
        for evt in events:
            if not isinstance(evt, dict):
                continue
            name = str(evt.get("step", "")).strip()
            status = str(evt.get("status", "")).strip().lower()
            if not name or name not in steps:
                continue
            st = steps[name] if isinstance(steps[name], dict) else {}
            attempts = int(st.get("attempts", 0) or 0) + 1
            st["attempts"] = attempts
            if status == "completed":
                st["status"] = "completed"
            elif status == "failed":
                max_r = int(st.get("max_retries", 0) or 0)
                if attempts <= max_r:
                    st["status"] = "pending"
                else:
                    st["status"] = "failed"
            else:
                st["status"] = status or "pending"
            steps[name] = st
        cur["steps"] = steps
        cur["last_events"] = [dict(e) for e in events if isinstance(e, dict)]
        cur["ready"] = self._compute_ready_steps(steps)
        failed = [name for name, s in steps.items() if isinstance(s, dict) and str(s.get("status", "")) == "failed"]
        cur["failed_steps"] = failed
        all_done = all(
            isinstance(s, dict) and str(s.get("status", "")) == "completed"
            for s in steps.values()
        ) if steps else False
        cur["terminal"] = bool(all_done or failed)
        return cur

    @staticmethod
    def _compute_ready_steps(steps: Dict[str, Any]) -> List[str]:
        completed = {
            name
            for name, st in steps.items()
            if isinstance(st, dict) and str(st.get("status", "")) == "completed"
        }
        ready: List[str] = []
        for name, st in steps.items():
            if not isinstance(st, dict):
                continue
            status = str(st.get("status", "")).strip().lower()
            if status != "pending":
                continue
            deps = st.get("depends_on", [])
            if not isinstance(deps, list):
                deps = []
            if all(str(d) in completed for d in deps):
                ready.append(str(name))
        return sorted(ready)

    def _build_execution_waves_native(self, steps: List[Dict[str, Any]]) -> List[List[str]]:
        remaining = {str(s["name"]): set(str(d) for d in s.get("depends_on", [])) for s in steps}
        capability_by_name = {str(s["name"]): str(s.get("capability", "")) for s in steps}
        waves: List[List[str]] = []
        resolved: set[str] = set()
        while remaining:
            ready = sorted([name for name, deps in remaining.items() if deps.issubset(resolved)])
            if not ready:
                unresolved = ", ".join(sorted(remaining.keys()))
                raise ValueError(f"Cycle or unresolved dependencies in workflow: {unresolved}")
            ready = sorted(
                ready,
                key=lambda n: (
                    self._lane_for_capability(capability_by_name.get(n, "")),
                    n,
                ),
            )
            ready = self._apply_wave_cap(ready)
            waves.append(ready)
            for name in ready:
                resolved.add(name)
                remaining.pop(name, None)
        return waves

    def _build_execution_waves_with_langgraph(self, steps: List[Dict[str, Any]]) -> List[List[str]]:
        # Import locally so runtime without langgraph continues to work.
        from langgraph.graph import END, StateGraph
        lane_by_name = {str(s["name"]): self._lane_for_capability(str(s.get("capability", ""))) for s in steps}

        def planner(state: Dict[str, Any]) -> Dict[str, Any]:
            unresolved = {
                str(k): set(str(d) for d in (v or []))
                for k, v in dict(state.get("unresolved", {})).items()
            }
            resolved = [str(x) for x in (state.get("resolved", []) or [])]
            resolved_set = set(resolved)
            waves = [list(w) for w in (state.get("waves", []) or [])]
            ready = sorted([name for name, deps in unresolved.items() if deps.issubset(resolved_set)])
            if not ready:
                return {
                    "unresolved": {k: sorted(v) for k, v in unresolved.items()},
                    "resolved": resolved,
                    "waves": waves,
                    "complete": True,
                    "error": f"Cycle or unresolved dependencies in workflow: {', '.join(sorted(unresolved.keys()))}",
                }
            ready = sorted(ready, key=lambda n: (lane_by_name.get(n, "z_default"), n))
            ready = self._apply_wave_cap(ready)
            for name in ready:
                unresolved.pop(name, None)
                resolved_set.add(name)
            resolved = sorted(resolved_set)
            waves.append(ready)
            return {
                "unresolved": {k: sorted(v) for k, v in unresolved.items()},
                "resolved": resolved,
                "waves": waves,
                "complete": len(unresolved) == 0,
                "error": "",
            }

        def route(state: Dict[str, Any]) -> str:
            if str(state.get("error", "")).strip():
                return "end"
            if bool(state.get("complete", False)):
                return "end"
            return "continue"

        graph = StateGraph(dict)
        graph.add_node("planner", planner)
        graph.set_entry_point("planner")
        graph.add_conditional_edges("planner", route, {"continue": "planner", "end": END})
        app = graph.compile()

        initial_state: Dict[str, Any] = {
            "unresolved": {str(s["name"]): [str(d) for d in s.get("depends_on", [])] for s in steps},
            "resolved": [],
            "waves": [],
            "complete": False,
            "error": "",
        }
        result = app.invoke(initial_state)
        error = str(result.get("error", "")).strip() if isinstance(result, dict) else ""
        if error:
            raise ValueError(error)
        waves = result.get("waves", []) if isinstance(result, dict) else []
        if not isinstance(waves, list):
            raise ValueError("Invalid LangGraph planner result: waves is not a list")
        output: List[List[str]] = []
        for wave in waves:
            if isinstance(wave, list):
                output.append([str(x) for x in wave])
        return output

    @staticmethod
    def _normalize_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        names: set[str] = set()
        for raw in steps:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            if not name or name in names:
                continue
            depends = raw.get("depends_on", [])
            if not isinstance(depends, list):
                depends = []
            deps = [str(d).strip() for d in depends if str(d).strip()]
            normalized.append(
                {
                    "name": name,
                    "depends_on": deps,
                    "capability": str(raw.get("capability", "")).strip(),
                }
            )
            names.add(name)
        for item in normalized:
            item["depends_on"] = [d for d in item.get("depends_on", []) if d in names and d != item["name"]]
        if not normalized:
            raise ValueError("No valid workflow steps provided")
        return normalized

    def _apply_wave_cap(self, ready: List[str]) -> List[str]:
        if self.max_wave_size <= 0:
            return ready
        if len(ready) <= self.max_wave_size:
            return ready
        return ready[: self.max_wave_size]

    @staticmethod
    def _lane_for_capability(capability: str) -> str:
        cap = str(capability or "").strip().lower()
        if any(token in cap for token in ("manage", "coord", "plan", "approve")):
            return "manager_lane"
        if any(token in cap for token in ("verify", "audit", "policy", "guard")):
            return "verifier_lane"
        if any(token in cap for token in ("code", "build", "test", "deploy", "workflow")):
            return "developer_lane"
        if any(token in cap for token in ("analy", "research", "rag", "retriev")):
            return "analyst_lane"
        return "default_lane"


__all__ = ["LangGraphWorkflowAdapter"]
