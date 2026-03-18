from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest

from interfaces.api_interface import APIInterface

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


@pytest.mark.skipif(TestClient is None or TestServer is None, reason="aiohttp test utilities unavailable")
async def test_api_smoke_flow() -> None:
    api = APIInterface()
    api.set_conversation_manager(_DummyConversationManager())
    api.set_skills_registry(_DummySkillsRegistry())
    api.set_orchestrator(_DummyOrchestrator())

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
    finally:
        await client.close()
