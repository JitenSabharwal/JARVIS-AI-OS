"""
Typed response contracts for stable assistant output composition.

These contracts are intentionally lightweight (dataclasses) so they can be
shared across interfaces/orchestration/agents without adding new runtime
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvidenceItem:
    claim: str
    sources: list[str] = field(default_factory=list)

    @staticmethod
    def from_any(value: Any) -> "EvidenceItem | None":
        if not isinstance(value, dict):
            return None
        claim = str(value.get("claim", "")).strip()
        raw_sources = value.get("sources", [])
        sources = [str(s).strip() for s in raw_sources] if isinstance(raw_sources, list) else []
        sources = [s for s in sources if s]
        if not claim and not sources:
            return None
        return EvidenceItem(claim=claim, sources=sources)


@dataclass(slots=True)
class RequestEnvelope:
    request_id: str
    user_id: str
    session_id: str = ""
    model: str = "jarvis-default"
    modality: str = "text"
    workspace_path: str = ""
    route: str = "chat"
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_any(payload: dict[str, Any] | None) -> "RequestEnvelope":
        p = payload if isinstance(payload, dict) else {}
        return RequestEnvelope(
            request_id=str(p.get("request_id", "")).strip(),
            user_id=str(p.get("user_id", "continue_user")).strip() or "continue_user",
            session_id=str(p.get("session_id", "")).strip(),
            model=str(p.get("model", "jarvis-default")).strip() or "jarvis-default",
            modality=str(p.get("modality", "text")).strip() or "text",
            workspace_path=str(p.get("workspace_path", "")).strip(),
            route=str(p.get("route", "chat")).strip() or "chat",
            metadata=p.get("metadata", {}) if isinstance(p.get("metadata", {}), dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "model": self.model,
            "modality": self.modality,
            "workspace_path": self.workspace_path,
            "route": self.route,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CodeAssistRequest:
    workspace_path: str
    instruction: str
    dry_run: bool = False
    run_checks: bool = True
    max_files: int = 10

    @staticmethod
    def from_any(payload: dict[str, Any] | None) -> tuple["CodeAssistRequest | None", str]:
        p = payload if isinstance(payload, dict) else {}
        workspace = str(p.get("workspace_path", "")).strip()
        instruction = str(p.get("instruction", "")).strip()
        if not workspace:
            return None, "workspace_path is required"
        if not instruction:
            return None, "instruction is required"
        max_files = int(p.get("max_files", 10) or 10)
        max_files = max(1, min(400, max_files))
        return (
            CodeAssistRequest(
                workspace_path=workspace,
                instruction=instruction,
                dry_run=bool(p.get("dry_run", False)),
                run_checks=bool(p.get("run_checks", True)),
                max_files=max_files,
            ),
            "",
        )


@dataclass(slots=True)
class RepoUnderstandRequest:
    workspace_path: str
    question: str
    max_files: int = 30
    include_tree: bool = True
    depth: str = "medium"

    @staticmethod
    def from_any(payload: dict[str, Any] | None) -> tuple["RepoUnderstandRequest | None", str]:
        p = payload if isinstance(payload, dict) else {}
        workspace = str(p.get("workspace_path", "")).strip()
        question = str(p.get("question", "")).strip() or "Explain this repository."
        if not workspace:
            return None, "workspace_path is required"
        depth = str(p.get("depth", "medium") or "medium").strip().lower()
        if depth not in {"low", "medium", "high"}:
            depth = "medium"
        max_files = int(p.get("max_files", 30) or 30)
        max_files = max(8, min(400, max_files))
        return (
            RepoUnderstandRequest(
                workspace_path=workspace,
                question=question,
                max_files=max_files,
                include_tree=bool(p.get("include_tree", True)),
                depth=depth,
            ),
            "",
        )


@dataclass(slots=True)
class CodeWorkflowRequest:
    workspace_path: str
    goal: str
    dry_run: bool = False
    run_checks: bool = True
    max_workers: int = 3

    @staticmethod
    def from_any(payload: dict[str, Any] | None) -> tuple["CodeWorkflowRequest | None", str]:
        p = payload if isinstance(payload, dict) else {}
        workspace = str(p.get("workspace_path", "")).strip()
        goal = str(p.get("goal", "")).strip()
        if not workspace:
            return None, "workspace_path is required"
        if not goal:
            return None, "goal is required"
        max_workers = int(p.get("max_workers", 3) or 3)
        max_workers = max(1, min(8, max_workers))
        return (
            CodeWorkflowRequest(
                workspace_path=workspace,
                goal=goal,
                dry_run=bool(p.get("dry_run", False)),
                run_checks=bool(p.get("run_checks", True)),
                max_workers=max_workers,
            ),
            "",
        )


@dataclass(slots=True)
class VerifiedResponse:
    summary: str = ""
    architecture: str = ""
    entry_points: list[str] = field(default_factory=list)
    key_modules: list[str] = field(default_factory=list)
    important_flows: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    confidence: float = 0.0
    coverage_pct: float = 0.0
    open_questions: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    analysis_plan: list[str] = field(default_factory=list)

    @staticmethod
    def from_repo_result(result: dict[str, Any]) -> "VerifiedResponse":
        if not isinstance(result, dict):
            return VerifiedResponse()

        def _list(name: str) -> list[str]:
            raw = result.get(name, [])
            if not isinstance(raw, list):
                return []
            out = [str(x).strip() for x in raw if str(x).strip()]
            return out

        evidence_items: list[EvidenceItem] = []
        raw_evidence = result.get("evidence", [])
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                parsed = EvidenceItem.from_any(item)
                if parsed:
                    evidence_items.append(parsed)

        try:
            confidence = float(result.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        try:
            coverage_pct = float(result.get("coverage_pct", 0.0) or 0.0)
        except Exception:
            coverage_pct = 0.0

        signals = result.get("signals", {})
        if not isinstance(signals, dict):
            signals = {}

        return VerifiedResponse(
            summary=str(result.get("summary", "")).strip(),
            architecture=str(result.get("architecture", "")).strip(),
            entry_points=_list("entry_points"),
            key_modules=_list("key_modules"),
            important_flows=_list("important_flows"),
            risks=_list("risks"),
            next_steps=_list("next_steps"),
            evidence=evidence_items,
            confidence=confidence,
            coverage_pct=coverage_pct,
            open_questions=_list("open_questions"),
            signals=signals,
            analysis_plan=_list("analysis_plan"),
        )

    def to_user_text(self) -> str:
        parts: list[str] = []
        if self.summary:
            parts.append(self.summary)
        if self.architecture:
            parts.append(f"Architecture: {self.architecture}")
        if self.entry_points:
            parts.append(f"Entry points: {', '.join(self.entry_points[:6])}.")
        if self.key_modules:
            parts.append(f"Key modules: {', '.join(self.key_modules[:8])}.")
        if self.important_flows:
            parts.append(f"Important flows: {' | '.join(self.important_flows[:4])}.")

        top_dirs = self.signals.get("top_directories", []) if isinstance(self.signals.get("top_directories", []), list) else []
        module_dist = self.signals.get("module_distribution", []) if isinstance(self.signals.get("module_distribution", []), list) else []
        dep_files = self.signals.get("dependency_files", []) if isinstance(self.signals.get("dependency_files", []), list) else []
        route_files = self.signals.get("route_related_files", []) if isinstance(self.signals.get("route_related_files", []), list) else []
        graph_modules = self.signals.get("graph_top_modules", []) if isinstance(self.signals.get("graph_top_modules", []), list) else []
        path_traces = self.signals.get("path_traces", []) if isinstance(self.signals.get("path_traces", []), list) else []
        if top_dirs:
            parts.append(f"Top directories in sample: {', '.join(str(x) for x in top_dirs[:8])}.")
        if module_dist:
            first = [m for m in module_dist[:5] if isinstance(m, dict)]
            if first:
                parts.append(
                    "Module distribution: "
                    + ", ".join(f"{str(m.get('module', ''))}({int(m.get('count', 0) or 0)})" for m in first)
                    + "."
                )
        if dep_files:
            parts.append(f"Dependency files: {', '.join(str(x) for x in dep_files[:5])}.")
        if route_files:
            parts.append(f"Route-related files: {', '.join(str(x) for x in route_files[:6])}.")
        if graph_modules:
            top = [m for m in graph_modules[:4] if isinstance(m, dict)]
            if top:
                parts.append(
                    "Import-graph hubs: "
                    + ", ".join(f"{str(m.get('module',''))}(score={float(m.get('score',0.0)):.1f})" for m in top)
                    + "."
                )
        if path_traces:
            parts.append(f"Path traces: {' | '.join(str(x) for x in path_traces[:4])}.")
        if self.analysis_plan:
            parts.append(f"Depth plan: {' | '.join(self.analysis_plan[:6])}.")
        if self.evidence:
            first = self.evidence[0]
            if first.claim and first.sources:
                parts.append(f"Evidence: {first.claim} Sources: {', '.join(first.sources[:6])}.")
        if self.confidence > 0:
            parts.append(f"Confidence: {self.confidence:.2f}. Coverage: {self.coverage_pct:.1f}% of repository files.")
        if self.risks:
            parts.append(f"Risks: {' | '.join(self.risks[:3])}.")
        if self.open_questions:
            parts.append(f"Open questions: {' | '.join(self.open_questions[:3])}.")
        if self.next_steps:
            parts.append(f"Next steps: {' | '.join(self.next_steps[:3])}.")

        return "\n".join(parts).strip()
