from __future__ import annotations

import pytest

from infrastructure.automation import AutomationEngine
from infrastructure.automation_actions import register_research_automation_actions
from infrastructure.connectors import (
    BaseConnector,
    ConnectorPolicy,
    ConnectorRegistry,
)
from infrastructure.research_adapters import StaticResearchAdapter
from infrastructure.research_intelligence import ResearchIntelligenceEngine


class _TestConnector(BaseConnector):
    @property
    def name(self) -> str:
        return "test_connector"

    @property
    def description(self) -> str:
        return "connector for tests"

    async def invoke(self, operation: str, params: dict) -> dict:
        if operation == "fail":
            raise RuntimeError("forced connector failure")
        return {"operation": operation, "params": params}


@pytest.mark.asyncio
async def test_connector_registry_invoke() -> None:
    registry = ConnectorRegistry()
    registry.register(_TestConnector())
    result = await registry.invoke("test_connector", "sync", {"value": 1})
    assert result["operation"] == "sync"
    assert result["params"]["value"] == 1


@pytest.mark.asyncio
async def test_connector_registry_scope_and_circuit_policy() -> None:
    registry = ConnectorRegistry()
    registry.register(
        _TestConnector(),
        policy=ConnectorPolicy(
            required_scopes_by_operation={"admin_sync": {"connector:admin"}},
            failure_threshold=2,
            recovery_timeout_seconds=30.0,
        ),
    )

    with pytest.raises(PermissionError):
        await registry.invoke("test_connector", "admin_sync", {"value": 1}, actor_scopes=set())

    result = await registry.invoke(
        "test_connector",
        "admin_sync",
        {"value": 2},
        actor_scopes={"connector:admin"},
    )
    assert result["params"]["value"] == 2

    with pytest.raises(RuntimeError):
        await registry.invoke("test_connector", "fail", {})
    with pytest.raises(RuntimeError):
        await registry.invoke("test_connector", "fail", {})
    with pytest.raises(RuntimeError):
        await registry.invoke("test_connector", "sync", {})

    health = await registry.health("test_connector")
    assert health["circuit_open"] is True
    assert "forced connector failure" in health["last_error"]

    health_all = await registry.health_all()
    assert "test_connector" in health_all
    assert health_all["test_connector"]["circuit_open"] is True


@pytest.mark.asyncio
async def test_automation_engine_process_event() -> None:
    engine = AutomationEngine()

    async def action(payload: dict) -> dict:
        return {"received": payload}

    engine.register_action("capture", action)
    engine.create_rule(
        name="on-high-error",
        event_type="task_failed",
        action_name="capture",
        match={"severity": "high"},
    )

    result = await engine.process_event("task_failed", {"severity": "high", "task_id": "t1"})
    assert result["matched_rules"] == 1
    assert result["executions"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_automation_engine_retries_and_dead_letters() -> None:
    engine = AutomationEngine()
    attempts = {"count": 0}

    async def flaky(payload: dict) -> dict:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")
        return {"ok": True, "payload": payload}

    async def always_fail(_payload: dict) -> dict:
        raise RuntimeError("permanent")

    engine.register_action("flaky", flaky)
    engine.register_action("always_fail", always_fail)
    engine.create_rule(
        name="retry-success",
        event_type="event_a",
        action_name="flaky",
        max_retries=2,
    )
    engine.create_rule(
        name="retry-exhausted",
        event_type="event_b",
        action_name="always_fail",
        max_retries=1,
    )

    success = await engine.process_event("event_a", {"k": "v"})
    assert success["matched_rules"] == 1
    assert success["executions"][0]["status"] == "completed"
    assert success["executions"][0]["attempts"] == 3

    failed = await engine.process_event("event_b", {"k": "v2"})
    assert failed["matched_rules"] == 1
    assert failed["executions"][0]["status"] == "failed"
    assert failed["executions"][0]["attempts"] == 2

    dead_letters = engine.get_dead_letters(limit=10)
    assert len(dead_letters) >= 1
    assert dead_letters[-1]["rule_name"] == "retry-exhausted"
    dead_letter_id = str(dead_letters[-1]["dead_letter_id"])

    replay = await engine.replay_dead_letter(dead_letter_id, remove_on_success=False)
    assert replay["replayed"] is True
    assert replay["succeeded"] is False

    remaining = engine.get_dead_letters(limit=10)
    assert any(str(d.get("dead_letter_id")) == dead_letter_id for d in remaining)

    resolved = engine.resolve_dead_letter(dead_letter_id, reason="unit_test_cleanup")
    assert resolved["resolved"] is True
    after_resolve = engine.get_dead_letters(limit=10)
    assert all(str(d.get("dead_letter_id")) != dead_letter_id for d in after_resolve)


@pytest.mark.asyncio
async def test_research_automation_actions_execute() -> None:
    engine = AutomationEngine()
    research = ResearchIntelligenceEngine()
    research.register_adapter(
        StaticResearchAdapter(
            name="static",
            items=[
                {
                    "title": "AI trend",
                    "url": "https://example.com/ai",
                    "content": "ai increase",
                    "topic": "ai",
                    "source_type": "news",
                }
            ],
        )
    )
    watch = research.create_watchlist(name="AI", topics=["ai"], cadence="hourly")
    register_research_automation_actions(engine, research)

    engine.create_rule(
        name="ingest-adapter",
        event_type="research_ingest_tick",
        action_name="ingest_research_from_adapters",
    )
    ingest = await engine.process_event(
        "research_ingest_tick",
        {"topic": "ai", "max_items_per_adapter": 5},
    )
    assert ingest["matched_rules"] == 1
    assert ingest["executions"][0]["status"] == "completed"

    engine.create_rule(
        name="run-digest",
        event_type="research_digest_tick",
        action_name="run_due_research_digests",
    )
    digest = await engine.process_event(
        "research_digest_tick",
        {"max_per_topic": 3},
    )
    assert digest["matched_rules"] == 1
    assert digest["executions"][0]["status"] == "completed"
    due = research.run_due_digests()
    assert due["generated_count"] == 0
    assert watch["watchlist_id"] in due["skipped_watchlist_ids"]
