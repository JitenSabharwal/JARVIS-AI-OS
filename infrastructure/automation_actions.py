"""
Default automation action registrations.
"""

from __future__ import annotations

from typing import Any, Dict, Set

from infrastructure.automation import AutomationEngine
from infrastructure.connectors import ConnectorRegistry
from infrastructure.research_intelligence import ResearchIntelligenceEngine


def register_default_automation_actions(
    engine: AutomationEngine,
    connectors: ConnectorRegistry,
) -> None:
    """
    Register default actions used by production-profile automation rules.
    """

    async def connector_invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
        connector = str(payload.get("connector", "")).strip()
        operation = str(payload.get("operation", "")).strip()
        params = payload.get("params", {})
        scopes_raw = payload.get("actor_scopes", [])
        if not connector:
            raise ValueError("connector_invoke requires 'connector'")
        if not operation:
            raise ValueError("connector_invoke requires 'operation'")
        if not isinstance(params, dict):
            raise ValueError("connector_invoke requires object 'params'")
        if not isinstance(scopes_raw, list) or any(not isinstance(s, str) for s in scopes_raw):
            raise ValueError("connector_invoke requires list[str] 'actor_scopes'")
        scopes: Set[str] = set(scopes_raw)
        result = await connectors.invoke(
            connector,
            operation,
            params,
            actor_scopes=scopes,
        )
        return {
            "connector": connector,
            "operation": operation,
            "result": result,
        }

    async def emit_audit_note(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ok": True,
            "note": str(payload.get("note", "")),
            "category": str(payload.get("category", "automation")),
        }

    engine.register_action("connector_invoke", connector_invoke)
    engine.register_action("emit_audit_note", emit_audit_note)


def register_research_automation_actions(
    engine: AutomationEngine,
    research_engine: ResearchIntelligenceEngine,
) -> None:
    """Register research-related automation actions."""

    async def run_due_research_digests(payload: Dict[str, Any]) -> Dict[str, Any]:
        max_per_topic = payload.get("max_per_topic", 3)
        try:
            max_per_topic = max(1, min(20, int(max_per_topic)))
        except (TypeError, ValueError):
            raise ValueError("max_per_topic must be an integer")
        return research_engine.run_due_digests(max_per_topic=max_per_topic)

    async def ingest_research_from_adapters(payload: Dict[str, Any]) -> Dict[str, Any]:
        topic = str(payload.get("topic", "")).strip()
        if not topic:
            raise ValueError("topic is required")
        max_items_per_adapter = payload.get("max_items_per_adapter", 10)
        try:
            max_items_per_adapter = max(1, min(100, int(max_items_per_adapter)))
        except (TypeError, ValueError):
            raise ValueError("max_items_per_adapter must be an integer")
        return research_engine.run_adapters(
            topic=topic,
            max_items_per_adapter=max_items_per_adapter,
        )

    engine.register_action("run_due_research_digests", run_due_research_digests)
    engine.register_action("ingest_research_from_adapters", ingest_research_from_adapters)
