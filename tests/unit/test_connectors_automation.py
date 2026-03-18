from __future__ import annotations

import pytest

from infrastructure.automation import AutomationEngine
from infrastructure.connectors import (
    BaseConnector,
    ConnectorPolicy,
    ConnectorRegistry,
)


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
