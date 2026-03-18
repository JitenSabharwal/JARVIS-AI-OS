from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest

from infrastructure.automation import AutomationEngine
from infrastructure.connectors import BaseConnector, ConnectorPolicy, ConnectorRegistry
from infrastructure.approval import ApprovalManager
from interfaces.api_interface import APIInterface
from skills.system_skills import RunCommandSkill

try:
    from aiohttp.test_utils import TestClient, TestServer
except Exception:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]
    TestServer = None  # type: ignore[assignment]


pytestmark = pytest.mark.asyncio


class _DummyConversationManager:
    def get_or_create_session(self, user_id: str) -> str:
        return f"session-{user_id}"

    async def process_input(self, session_id: str, query: str) -> str:
        return f"{session_id}:{query}"


class _DummySkillResult:
    def __init__(self, success: bool, data: Any = None, error: str | None = None) -> None:
        self.success = success
        self.data = data
        self.error = error


class _DummySkillsRegistry:
    def get_all_skills_info(self) -> list[dict[str, Any]]:
        return [{"name": "ping_skill", "description": "test", "parameters": {}}]

    async def execute_skill(self, skill_name: str, params: dict[str, Any]) -> _DummySkillResult:
        if skill_name == "fail":
            return _DummySkillResult(False, error="forced failure")
        return _DummySkillResult(True, data={"skill_name": skill_name, "params": params})


class _PolicySkillsRegistry:
    def __init__(self) -> None:
        self._run_command = RunCommandSkill()

    def get_all_skills_info(self) -> list[dict[str, Any]]:
        return [self._run_command.get_schema()]

    async def execute_skill(self, skill_name: str, params: dict[str, Any]) -> _DummySkillResult:
        if skill_name != "run_command":
            return _DummySkillResult(False, error=f"unknown skill: {skill_name}")
        result = await self._run_command.safe_execute(params)
        return _DummySkillResult(success=result.success, data=result.data, error=result.error)


class _DummyTaskStatusEnum(Enum):
    COMPLETED = "completed"


@dataclass
class _DummyTask:
    status: _DummyTaskStatusEnum
    result: Any = None
    error: str | None = None


class _DummyOrchestrator:
    def __init__(self) -> None:
        self._task_counter = 0
        self._tasks: dict[str, _DummyTask] = {}

    async def submit_task(
        self,
        description: str,
        required_capabilities: list[str],
        *,
        priority: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> str:
        self._task_counter += 1
        task_id = f"orch-task-{self._task_counter}"
        self._tasks[task_id] = _DummyTask(
            status=_DummyTaskStatusEnum.COMPLETED,
            result={
                "description": description,
                "required_capabilities": required_capabilities,
                "priority": priority,
                "payload": payload or {},
            },
        )
        return task_id

    def get_task_status(self, task_id: str) -> _DummyTask | None:
        return self._tasks.get(task_id)

    def get_system_status(self) -> dict[str, Any]:
        return {"agents": [{"name": "dummy-agent", "state": "IDLE"}]}


class _DummyConnector(BaseConnector):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "dummy connector"

    async def invoke(self, operation: str, params: dict[str, Any]) -> Any:
        if operation == "explode":
            raise RuntimeError("connector exploded")
        return {"operation": operation, "params": params, "ok": True}


@pytest.mark.skipif(TestClient is None or TestServer is None, reason="aiohttp test utilities unavailable")
async def test_api_smoke_flow() -> None:
    api = APIInterface()
    api.set_conversation_manager(_DummyConversationManager())
    api.set_skills_registry(_DummySkillsRegistry())
    api.set_orchestrator(_DummyOrchestrator())
    connectors = ConnectorRegistry()
    connectors.register(
        _DummyConnector(),
        policy=ConnectorPolicy(
            required_scopes_by_operation={"admin_ping": {"connector:admin"}},
            failure_threshold=1,
            recovery_timeout_seconds=30.0,
        ),
    )
    api.set_connector_registry(connectors)

    automation = AutomationEngine()

    async def _notify_action(payload: dict[str, Any]) -> dict[str, Any]:
        return {"notified": True, "payload": payload}

    async def _always_fail(_payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("forced failure")

    automation.register_action("notify", _notify_action)
    automation.register_action("always_fail", _always_fail)
    api.set_automation_engine(automation)

    app = api._build_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        headers = {"X-Request-ID": "req-123"}

        query_resp = await client.post(
            "/api/v1/query",
            json={"query": "hello", "user_id": "u1"},
            headers=headers,
        )
        assert query_resp.status == 200
        query_data = await query_resp.json()
        assert query_data["success"] is True
        assert query_data["request_id"] == "req-123"
        assert query_resp.headers.get("X-Request-ID") == "req-123"

        task_resp = await client.post(
            "/api/v1/tasks",
            json={
                "description": "run test",
                "required_capabilities": ["echo"],
                "payload": {"x": 1},
            },
        )
        assert task_resp.status == 202
        task_data = await task_resp.json()
        assert task_data["success"] is True
        task_id = task_data["data"]["task_id"]
        assert task_data["data"]["orchestrator_task_id"].startswith("orch-task-")

        task_status_resp = await client.get(f"/api/v1/tasks/{task_id}")
        assert task_status_resp.status == 200
        task_status_data = await task_status_resp.json()
        assert task_status_data["data"]["status"] == "completed"

        skills_resp = await client.get("/api/v1/skills")
        assert skills_resp.status == 200
        skills_data = await skills_resp.json()
        assert skills_data["success"] is True
        assert skills_data["data"]["skills"]

        status_resp = await client.get("/api/v1/status")
        assert status_resp.status == 200
        status_data = await status_resp.json()
        assert status_data["success"] is True
        assert "orchestrator" in status_data["data"]

        metrics_resp = await client.get("/api/v1/metrics")
        assert metrics_resp.status == 200
        metrics_data = await metrics_resp.json()
        assert metrics_data["success"] is True
        assert "metrics" in metrics_data["data"]
        assert "slo" in metrics_data["data"]
        assert "counters" in metrics_data["data"]["metrics"]
        assert "latency" in metrics_data["data"]["metrics"]

        audit_resp = await client.get("/api/v1/audit?limit=10")
        assert audit_resp.status == 200
        audit_data = await audit_resp.json()
        assert audit_data["success"] is True
        assert audit_data["data"]["count"] >= 1

        approval_req_resp = await client.post(
            "/api/v1/approvals/request",
            json={
                "action": "run_command:ps",
                "reason": "diagnostics",
                "resource": "host/system",
            },
        )
        assert approval_req_resp.status == 201
        approval_req_data = await approval_req_resp.json()
        assert approval_req_data["success"] is True
        approval_id = approval_req_data["data"]["approval_id"]

        approval_approve_resp = await client.post(
            f"/api/v1/approvals/{approval_id}/approve",
            json={"approver": "sec-admin", "note": "approved for test"},
        )
        assert approval_approve_resp.status == 200
        approval_approve_data = await approval_approve_resp.json()
        assert approval_approve_data["data"]["status"] == "approved"

        approval_get_resp = await client.get(f"/api/v1/approvals/{approval_id}")
        assert approval_get_resp.status == 200
        approval_get_data = await approval_get_resp.json()
        assert approval_get_data["success"] is True
        assert approval_get_data["data"]["approval_id"] == approval_id

        connectors_resp = await client.get("/api/v1/connectors")
        assert connectors_resp.status == 200
        connectors_data = await connectors_resp.json()
        assert connectors_data["success"] is True
        assert connectors_data["data"]["count"] >= 1

        connector_invoke_resp = await client.post(
            "/api/v1/connectors/dummy/invoke",
            json={"operation": "ping", "params": {"x": 1}},
        )
        assert connector_invoke_resp.status == 200
        connector_invoke_data = await connector_invoke_resp.json()
        assert connector_invoke_data["data"]["result"]["ok"] is True

        connector_forbidden_resp = await client.post(
            "/api/v1/connectors/dummy/invoke",
            json={"operation": "admin_ping", "params": {"x": 1}},
        )
        assert connector_forbidden_resp.status == 403

        connector_allowed_resp = await client.post(
            "/api/v1/connectors/dummy/invoke",
            json={"operation": "admin_ping", "params": {"x": 2}},
            headers={"X-Scopes": "connector:admin"},
        )
        assert connector_allowed_resp.status == 200

        connector_open_resp = await client.post(
            "/api/v1/connectors/dummy/invoke",
            json={"operation": "explode", "params": {}},
        )
        assert connector_open_resp.status == 500
        connector_circuit_resp = await client.post(
            "/api/v1/connectors/dummy/invoke",
            json={"operation": "ping", "params": {"x": 3}},
        )
        assert connector_circuit_resp.status == 503

        rule_create_resp = await client.post(
            "/api/v1/automation/rules",
            json={
                "name": "notify-errors",
                "event_type": "task_failed",
                "action_name": "notify",
                "match": {"severity": "high"},
                "max_retries": 1,
            },
        )
        assert rule_create_resp.status == 201
        rule_create_data = await rule_create_resp.json()
        assert rule_create_data["success"] is True

        fail_rule_resp = await client.post(
            "/api/v1/automation/rules",
            json={
                "name": "fail-errors",
                "event_type": "task_failed",
                "action_name": "always_fail",
                "match": {"severity": "critical"},
                "max_retries": 1,
            },
        )
        assert fail_rule_resp.status == 201

        rules_list_resp = await client.get("/api/v1/automation/rules")
        assert rules_list_resp.status == 200
        rules_list_data = await rules_list_resp.json()
        assert rules_list_data["data"]["count"] >= 1

        event_resp = await client.post(
            "/api/v1/automation/events",
            json={
                "event_type": "task_failed",
                "payload": {"severity": "high", "task_id": "t1"},
            },
        )
        assert event_resp.status == 200
        event_data = await event_resp.json()
        assert event_data["data"]["matched_rules"] >= 1

        history_resp = await client.get("/api/v1/automation/history?limit=10")
        assert history_resp.status == 200
        history_data = await history_resp.json()
        assert history_data["data"]["count"] >= 1

        fail_event_resp = await client.post(
            "/api/v1/automation/events",
            json={
                "event_type": "task_failed",
                "payload": {"severity": "critical", "task_id": "t2"},
            },
        )
        assert fail_event_resp.status == 200

        dead_letters_resp = await client.get("/api/v1/automation/dead-letters?limit=10")
        assert dead_letters_resp.status == 200
        dead_letters_data = await dead_letters_resp.json()
        assert dead_letters_data["data"]["count"] >= 1

        metrics_end_resp = await client.get("/api/v1/metrics")
        assert metrics_end_resp.status == 200
        metrics_end_data = await metrics_end_resp.json()
        counters = metrics_end_data["data"]["metrics"]["counters"]
        assert any(k.startswith("task_submit_total:") for k in counters)
        assert any(k.startswith("skill_execute_total:") for k in counters)
        assert any(k.startswith("connector_invoke_total:") for k in counters)
        assert any(k.startswith("automation_event_total:") for k in counters)
    finally:
        await client.close()


@pytest.mark.skipif(TestClient is None or TestServer is None, reason="aiohttp test utilities unavailable")
async def test_api_policy_gated_high_risk_smoke() -> None:
    ApprovalManager.reset_instance()
    api = APIInterface()
    api.set_skills_registry(_PolicySkillsRegistry())
    app = api._build_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        denied_resp = await client.post(
            "/api/v1/skills/run_command/execute",
            json={"params": {"command": "ps", "dry_run": True}},
        )
        assert denied_resp.status == 400
        denied_data = await denied_resp.json()
        assert "invalid or not approved" in (denied_data.get("error") or "").lower()

        approval_req_resp = await client.post(
            "/api/v1/approvals/request",
            json={
                "action": "run_command:ps",
                "reason": "policy smoke",
                "resource": "host/process",
            },
        )
        assert approval_req_resp.status == 201
        approval_req_data = await approval_req_resp.json()
        approval_id = approval_req_data["data"]["approval_id"]
        approval_token = approval_req_data["data"]["approval_token"]

        approve_resp = await client.post(
            f"/api/v1/approvals/{approval_id}/approve",
            json={"approver": "test-admin", "note": "policy smoke allow"},
        )
        assert approve_resp.status == 200

        allowed_resp = await client.post(
            "/api/v1/skills/run_command/execute",
            json={
                "params": {
                    "command": "ps",
                    "approval_token": approval_token,
                    "justification": "policy smoke",
                    "dry_run": True,
                }
            },
        )
        assert allowed_resp.status == 200
        allowed_data = await allowed_resp.json()
        assert allowed_data["success"] is True
        assert "planned_command" in allowed_data["data"]
    finally:
        await client.close()
