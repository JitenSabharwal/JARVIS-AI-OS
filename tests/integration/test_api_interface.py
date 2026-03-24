from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest

from infrastructure.automation import AutomationEngine
from infrastructure.connectors import BaseConnector, ConnectorPolicy, ConnectorRegistry
from infrastructure.approval import ApprovalManager
from infrastructure.builtin_connectors import build_default_connector_registry
from infrastructure.research_adapters import StaticResearchAdapter
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

    def get_context(self, session_id: str) -> Any:
        return SimpleNamespace(
            metadata={
                "latency_ms": {"intent_extract": 1.0, "response_generate": 2.0, "response_summarize": 0.5},
                "model_route": {"provider_name": "local", "latency_ms": 12.3},
                "summary_stage": {"used": True, "source": "model", "latency_ms": 3.4},
            }
        )


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
        self._plans: dict[str, dict[str, Any]] = {}

    async def submit_task(
        self,
        description: str,
        required_capabilities: list[str],
        *,
        priority: int = 0,
        payload: dict[str, Any] | None = None,
        requires_human: bool = False,
        approval_token: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self._task_counter += 1
        task_id = f"orch-task-{self._task_counter}"
        status = _DummyTaskStatusEnum.COMPLETED
        if requires_human and not approval_token:
            status = _DummyTaskStatusEnum.COMPLETED
        self._tasks[task_id] = _DummyTask(
            status=status,
            result={
                "description": description,
                "required_capabilities": required_capabilities,
                "priority": priority,
                "payload": payload or {},
                "requires_human": requires_human,
                "approval_token_present": bool(approval_token),
                "metadata": metadata or {},
            },
        )
        return task_id

    async def retry_task(self, task_id: str, *, approval_token: str | None = None) -> bool:
        _ = approval_token
        return task_id in self._tasks

    async def replan_task(
        self,
        task_id: str,
        *,
        fallback_capabilities: list[str],
        payload_override: dict[str, Any] | None = None,
        description_suffix: str = "replan",
    ) -> str | None:
        if task_id not in self._tasks:
            return None
        self._task_counter += 1
        new_task_id = f"orch-task-{self._task_counter}"
        self._tasks[new_task_id] = _DummyTask(
            status=_DummyTaskStatusEnum.COMPLETED,
            result={
                "replanned_from": task_id,
                "fallback_capabilities": fallback_capabilities,
                "payload": payload_override or {},
                "description_suffix": description_suffix,
            },
        )
        return new_task_id

    async def submit_task_plan(
        self,
        *,
        description: str,
        steps: list[dict[str, Any]],
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        plan_id = f"plan-{len(self._plans) + 1}"
        task_ids_by_step: dict[str, str] = {}
        for step in steps:
            step_name = str(step.get("name", "step")).strip() or "step"
            orch_task_id = await self.submit_task(
                description=f"{description}/{step_name}",
                required_capabilities=[str(step.get("capability", "echo"))],
                priority=priority,
                payload=step.get("payload", {}),
            )
            task_ids_by_step[step_name] = orch_task_id
        self._plans[plan_id] = {
            "plan_id": plan_id,
            "description": description,
            "status": "completed",
            "steps": {k: "completed" for k in task_ids_by_step.keys()},
            "metadata": metadata or {},
        }
        return {
            "plan_id": plan_id,
            "description": description,
            "task_ids_by_step": task_ids_by_step,
            "step_count": len(task_ids_by_step),
        }

    def get_plan_status(self, plan_id: str) -> dict[str, Any] | None:
        return self._plans.get(plan_id)

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
async def test_api_smoke_flow(tmp_path) -> None:
    api = APIInterface()
    api.set_conversation_manager(_DummyConversationManager())
    api.set_skills_registry(_DummySkillsRegistry())
    api.set_orchestrator(_DummyOrchestrator())
    connectors = build_default_connector_registry(str(tmp_path))
    connectors.register(
        _DummyConnector(),
        policy=ConnectorPolicy(
            required_scopes_by_operation={"admin_ping": {"connector:admin"}},
            failure_threshold=1,
            recovery_timeout_seconds=30.0,
        ),
    )
    api.set_connector_registry(connectors)
    api.research_engine.register_adapter(
        StaticResearchAdapter(
            name="static-news",
            items=[
                {
                    "title": "AI chips update",
                    "url": "https://adapter.example.com/ai-chips",
                    "content": "ai chips increase performance",
                    "topic": "ai chips",
                    "source_type": "news",
                }
            ],
        )
    )

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

        task_retry_resp = await client.post(f"/api/v1/tasks/{task_id}/retry", json={})
        assert task_retry_resp.status == 200
        task_retry_data = await task_retry_resp.json()
        assert task_retry_data["data"]["retried"] is True

        task_replan_resp = await client.post(
            f"/api/v1/tasks/{task_id}/replan",
            json={"fallback_capabilities": ["echo"], "payload_override": {"x": 2}},
        )
        assert task_replan_resp.status == 202
        task_replan_data = await task_replan_resp.json()
        assert task_replan_data["success"] is True
        assert task_replan_data["data"]["replanned_from_task_id"] == task_id

        plan_submit_resp = await client.post(
            "/api/v1/plans",
            json={
                "description": "ship feature",
                "steps": [
                    {"name": "design", "capability": "echo", "payload": {"s": 1}},
                    {"name": "implement", "capability": "echo", "depends_on": ["design"]},
                ],
                "metadata": {"owner": "qa"},
            },
        )
        assert plan_submit_resp.status == 202
        plan_submit_data = await plan_submit_resp.json()
        plan_id = plan_submit_data["data"]["plan_id"]
        assert plan_submit_data["data"]["step_count"] == 2

        plan_get_resp = await client.get(f"/api/v1/plans/{plan_id}")
        assert plan_get_resp.status == 200
        plan_get_data = await plan_get_resp.json()
        assert plan_get_data["data"]["plan_id"] == plan_id

        research_ingest_resp = await client.post(
            "/api/v1/research/ingest",
            json={
                "items": [
                    {
                        "title": "AI chips gain speed",
                        "url": "https://news.example.com/ai",
                        "content": "chip performance increase",
                        "topic": "ai chips",
                        "source_type": "news",
                    }
                ]
            },
        )
        assert research_ingest_resp.status == 200
        research_ingest_data = await research_ingest_resp.json()
        assert research_ingest_data["data"]["inserted"] >= 1

        research_query_resp = await client.post(
            "/api/v1/research/query",
            json={"topic": "ai chips", "max_results": 5, "freshness_days": 365, "min_trust": 0.4},
        )
        assert research_query_resp.status == 200
        research_query_data = await research_query_resp.json()
        assert research_query_data["data"]["result_count"] >= 1
        assert len(research_query_data["data"]["citations"]) >= 1
        assert "citation_health_score" in research_query_data["data"]
        assert "rag_context_count" in research_query_data["data"]

        top_source_id = research_query_data["data"]["results"][0]["source_id"]
        research_tree_resp = await client.get(f"/api/v1/research/tree/{top_source_id}")
        assert research_tree_resp.status == 200
        research_tree_data = await research_tree_resp.json()
        assert research_tree_data["data"]["source_id"] == top_source_id

        graph_health_resp = await client.get("/api/v1/research/graph/health")
        assert graph_health_resp.status == 200
        graph_health_data = await graph_health_resp.json()
        assert "enabled" in graph_health_data["data"]

        adapters_list_resp = await client.get("/api/v1/research/adapters")
        assert adapters_list_resp.status == 200
        adapters_list_data = await adapters_list_resp.json()
        assert adapters_list_data["data"]["count"] >= 1

        adapters_run_resp = await client.post(
            "/api/v1/research/adapters/run",
            json={"topic": "ai chips", "max_items_per_adapter": 5},
        )
        assert adapters_run_resp.status == 200
        adapters_run_data = await adapters_run_resp.json()
        assert adapters_run_data["data"]["adapter_count"] >= 1

        watch_create_resp = await client.post(
            "/api/v1/research/watchlists",
            json={"name": "Tech Watch", "topics": ["ai chips"], "cadence": "daily"},
        )
        assert watch_create_resp.status == 201
        watch_create_data = await watch_create_resp.json()
        watch_id = watch_create_data["data"]["watchlist_id"]

        watch_list_resp = await client.get("/api/v1/research/watchlists")
        assert watch_list_resp.status == 200
        watch_list_data = await watch_list_resp.json()
        assert watch_list_data["data"]["count"] >= 1

        digest_resp = await client.post(
            f"/api/v1/research/watchlists/{watch_id}/digest",
            json={"max_per_topic": 2},
        )
        assert digest_resp.status == 200
        digest_data = await digest_resp.json()
        assert digest_data["data"]["watchlist_id"] == watch_id

        run_due_resp = await client.post(
            "/api/v1/research/digests/run-due",
            json={"max_per_topic": 2},
        )
        assert run_due_resp.status == 200
        run_due_data = await run_due_resp.json()
        assert "generated_count" in run_due_data["data"]

        delivery_templates_resp = await client.get("/api/v1/delivery/templates")
        assert delivery_templates_resp.status == 200
        delivery_templates_data = await delivery_templates_resp.json()
        assert delivery_templates_data["data"]["template_count"] >= 1
        assert delivery_templates_data["data"]["profile_count"] >= 1

        delivery_bootstrap_resp = await client.post(
            "/api/v1/delivery/bootstrap",
            json={
                "template_id": "backend_fastapi",
                "project_name": "demo_delivery",
                "cloud_target": "aws",
            },
        )
        assert delivery_bootstrap_resp.status == 201
        delivery_bootstrap_data = await delivery_bootstrap_resp.json()
        assert delivery_bootstrap_data["data"]["project_name"] == "demo_delivery"

        delivery_pipeline_resp = await client.post(
            "/api/v1/delivery/pipelines/run",
            json={
                "project_name": "demo_delivery",
                "gate_inputs": {
                    "lint": True,
                    "test": True,
                    "sast": True,
                    "dependency_audit": True,
                },
            },
        )
        assert delivery_pipeline_resp.status == 200
        delivery_pipeline_data = await delivery_pipeline_resp.json()
        assert delivery_pipeline_data["data"]["all_passed"] is True

        delivery_release_resp = await client.post(
            "/api/v1/delivery/releases",
            json={
                "project_name": "demo_delivery",
                "profile": "prod",
                "pipeline_result": delivery_pipeline_data["data"],
                "approved": True,
                "post_deploy": {
                    "error_rate_pct": 0.5,
                    "p95_latency_ms": 800.0,
                    "availability_pct": 99.9,
                },
            },
        )
        assert delivery_release_resp.status == 201
        delivery_release_data = await delivery_release_resp.json()
        assert delivery_release_data["data"]["status"] == "deployed"
        release_id = delivery_release_data["data"]["release_id"]

        delivery_release_get_resp = await client.get(f"/api/v1/delivery/releases/{release_id}")
        assert delivery_release_get_resp.status == 200
        delivery_release_get_data = await delivery_release_get_resp.json()
        assert delivery_release_get_data["data"]["release_id"] == release_id

        delivery_post_deploy_resp = await client.post(
            f"/api/v1/delivery/releases/{release_id}/post-deploy",
            json={
                "post_deploy": {
                    "error_rate_pct": 10.0,
                    "p95_latency_ms": 2100.0,
                    "availability_pct": 98.0,
                }
            },
        )
        assert delivery_post_deploy_resp.status == 200
        delivery_post_deploy_data = await delivery_post_deploy_resp.json()
        assert delivery_post_deploy_data["data"]["status"] == "rolled_back"

        delivery_lead_time_resp = await client.get("/api/v1/delivery/metrics/lead-time")
        assert delivery_lead_time_resp.status == 200
        delivery_lead_time_data = await delivery_lead_time_resp.json()
        assert delivery_lead_time_data["data"]["release_count"] >= 1

        delivery_capabilities_resp = await client.get("/api/v1/delivery/capabilities")
        assert delivery_capabilities_resp.status == 200
        delivery_capabilities_data = await delivery_capabilities_resp.json()
        assert "lint" in delivery_capabilities_data["data"]["gate_runners"]
        assert "aws" in delivery_capabilities_data["data"]["deploy_adapters"]
        assert "runtime_config" in delivery_capabilities_data["data"]
        runtime_cfg = delivery_capabilities_data["data"]["runtime_config"]
        assert "aws_deploy_command" in runtime_cfg
        assert "allowed_deploy_targets" in runtime_cfg
        assert "deploy_max_retries" in runtime_cfg
        assert "deploy_retry_backoff_seconds" in runtime_cfg
        ci_templates = delivery_capabilities_data["data"]["ci_gate_templates"]
        assert "backend" in ci_templates
        assert "lint" in ci_templates["backend"]
        deploy_specs = delivery_capabilities_data["data"]["deploy_adapter_specs"]
        assert "aws" in deploy_specs
        assert "retryable_error_types" in deploy_specs["aws"]

        delivery_run_resp = await client.post(
            "/api/v1/delivery/releases/run",
            json={
                "project_name": "demo_delivery",
                "profile": "prod",
                "deploy_target": "aws",
                "approved": True,
                "context": {
                    "gates": {
                        "lint": True,
                        "test": True,
                        "sast": True,
                        "dependency_audit": True,
                    },
                    "deploy": {"success": True},
                },
                "post_deploy": {
                    "error_rate_pct": 0.2,
                    "p95_latency_ms": 780.0,
                    "availability_pct": 99.9,
                },
            },
        )
        assert delivery_run_resp.status == 201
        delivery_run_data = await delivery_run_resp.json()
        assert delivery_run_data["data"]["pipeline"]["all_passed"] is True
        assert delivery_run_data["data"]["release"]["status"] == "deployed"
        assert delivery_run_data["data"]["deploy"]["success"] is True

        delivery_run_cmd_resp = await client.post(
            "/api/v1/delivery/releases/run",
            json={
                "project_name": "demo_delivery_cmd",
                "profile": "prod",
                "deploy_target": "aws",
                "approved": True,
                "context": {
                    "gate_commands": {
                        "lint": ["python3", "-c", "raise SystemExit(0)"],
                        "test": ["python3", "-c", "raise SystemExit(0)"],
                        "sast": ["python3", "-c", "raise SystemExit(0)"],
                        "dependency_audit": ["python3", "-c", "raise SystemExit(0)"],
                    },
                    "deploy_commands": {
                        "aws": ["python3", "-c", "raise SystemExit(0)"],
                    },
                    "command_timeout_seconds": 10,
                },
            },
        )
        assert delivery_run_cmd_resp.status == 201
        delivery_run_cmd_data = await delivery_run_cmd_resp.json()
        assert delivery_run_cmd_data["data"]["pipeline"]["all_passed"] is True
        assert delivery_run_cmd_data["data"]["deploy"]["success"] is True

        email_oauth_resp = await client.post(
            "/api/v1/email/oauth_connect",
            json={
                "account_id": "acc-int-1",
                "provider": "gmail",
                "access_token": "tok-int-1",
                "refresh_token": "ref-int-1",
                "expires_in_sec": 3600,
            },
            headers={"X-Scopes": "connector:email:oauth:write"},
        )
        assert email_oauth_resp.status == 200
        email_oauth_data = await email_oauth_resp.json()
        assert email_oauth_data["data"]["result"]["connected"] is True

        email_ingest_resp = await client.post(
            "/api/v1/email/ingest_inbox",
            json={
                "account_id": "acc-int-1",
                "messages": [
                    {
                        "message_id": "m-int-1",
                        "from": "alice@example.com",
                        "subject": "status",
                        "body": "need update",
                    }
                ],
            },
            headers={"X-Scopes": "connector:email:write"},
        )
        assert email_ingest_resp.status == 200
        email_classify_resp = await client.post(
            "/api/v1/email/classify",
            json={"account_id": "acc-int-1", "message_id": "m-int-1", "label": "important"},
            headers={"X-Scopes": "connector:email:triage"},
        )
        assert email_classify_resp.status == 200
        email_classify_data = await email_classify_resp.json()
        action_id = email_classify_data["data"]["result"]["action_id"]
        email_undo_resp = await client.post(
            "/api/v1/email/undo",
            json={"account_id": "acc-int-1", "action_id": action_id},
            headers={"X-Scopes": "connector:email:undo"},
        )
        assert email_undo_resp.status == 200

        file_base = tmp_path / "file_intel"
        file_base.mkdir(parents=True, exist_ok=True)
        (file_base / "project.txt").write_text(
            "project brief with milestones and blockers",
            encoding="utf-8",
        )
        file_index_resp = await client.post(
            "/api/v1/files/intel/index_file",
            json={"path": "project.txt", "acl_tags": ["team-a"]},
            headers={"X-Scopes": "connector:file_intel:index"},
        )
        assert file_index_resp.status == 200
        file_index_data = await file_index_resp.json()
        file_doc_id = file_index_data["data"]["result"]["record"]["doc_id"]
        file_summary_resp = await client.post(
            "/api/v1/files/intel/summarize_indexed",
            json={"doc_id": file_doc_id, "actor_acl_tags": ["team-a"]},
            headers={"X-Scopes": "connector:file_intel:read"},
        )
        assert file_summary_resp.status == 200
        file_summary_data = await file_summary_resp.json()
        assert file_summary_data["data"]["result"]["confidence"] >= 0.6

        image_base = tmp_path / "image_intel" / "source"
        image_base.mkdir(parents=True, exist_ok=True)
        (image_base / "trip_1.jpg").write_bytes(b"a")
        image_preview_resp = await client.post(
            "/api/v1/images/intel/preview_organize",
            json={"path": "source", "target_root": "organized", "recursive": False},
            headers={"X-Scopes": "connector:image_intel:plan"},
        )
        assert image_preview_resp.status == 200
        image_preview_data = await image_preview_resp.json()
        image_plan_id = image_preview_data["data"]["result"]["plan_id"]
        image_apply_resp = await client.post(
            "/api/v1/images/intel/apply_plan",
            json={"plan_id": image_plan_id},
            headers={"X-Scopes": "connector:image_intel:write"},
        )
        assert image_apply_resp.status == 200
        image_undo_resp = await client.post(
            "/api/v1/images/intel/undo_plan",
            json={"plan_id": image_plan_id},
            headers={"X-Scopes": "connector:image_intel:write"},
        )
        assert image_undo_resp.status == 200

        research_rule_resp = await client.post(
            "/api/v1/automation/rules",
            json={
                "name": "run-research-digests",
                "event_type": "research_tick",
                "action_name": "run_due_research_digests",
                "max_retries": 0,
            },
        )
        assert research_rule_resp.status == 201
        research_event_resp = await client.post(
            "/api/v1/automation/events",
            json={"event_type": "research_tick", "payload": {"max_per_topic": 2}},
        )
        assert research_event_resp.status == 200
        research_event_data = await research_event_resp.json()
        assert research_event_data["data"]["matched_rules"] >= 1

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

        connectors_health_resp = await client.get("/api/v1/connectors/health")
        assert connectors_health_resp.status == 200
        connectors_health_data = await connectors_health_resp.json()
        assert connectors_health_data["success"] is True
        assert "dummy" in connectors_health_data["data"]["connectors"]

        dummy_health_resp = await client.get("/api/v1/connectors/dummy/health")
        assert dummy_health_resp.status == 200
        dummy_health_data = await dummy_health_resp.json()
        assert dummy_health_data["data"]["connector"] == "dummy"

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
        dead_letter_id = dead_letters_data["data"]["dead_letters"][-1]["dead_letter_id"]

        replay_resp = await client.post(
            "/api/v1/automation/dead-letters/replay",
            json={"dead_letter_id": dead_letter_id, "timeout_seconds": 5.0},
        )
        assert replay_resp.status == 200
        replay_data = await replay_resp.json()
        assert replay_data["success"] is True
        assert replay_data["data"]["replayed"] is True

        resolve_resp = await client.post(
            f"/api/v1/automation/dead-letters/{dead_letter_id}/resolve",
            json={"reason": "integration_test"},
        )
        assert resolve_resp.status == 200
        resolve_data = await resolve_resp.json()
        assert resolve_data["success"] is True
        assert resolve_data["data"]["resolved"] is True

        proactive_prefs_resp = await client.post(
            "/api/v1/proactive/preferences",
            json={"user_id": "u-pro", "preferences": {"cooldown_seconds": 1, "risk_tolerance": "low"}},
        )
        assert proactive_prefs_resp.status == 200
        proactive_prefs_data = await proactive_prefs_resp.json()
        assert proactive_prefs_data["data"]["user_id"] == "u-pro"

        proactive_event_resp = await client.post(
            "/api/v1/proactive/events",
            json={
                "event_type": "anomaly_detected",
                "payload": {"user_id": "u-pro", "anomaly": "latency p95 spike"},
            },
        )
        assert proactive_event_resp.status == 201
        proactive_event_data = await proactive_event_resp.json()
        assert proactive_event_data["data"]["generated_count"] >= 1

        proactive_suggestions_resp = await client.get(
            "/api/v1/proactive/suggestions?user_id=u-pro&max_items=5",
        )
        assert proactive_suggestions_resp.status == 200
        proactive_suggestions_data = await proactive_suggestions_resp.json()
        assert proactive_suggestions_data["data"]["count"] >= 1
        assert proactive_suggestions_data["data"]["suggestions"][0]["requires_human"] is True
        assert "anomaly_detected" in proactive_suggestions_data["data"]["suggestions"][0]["metadata"].get(
            "safety_reasons",
            [],
        )

        proactive_profile_resp = await client.get("/api/v1/proactive/profile/u-pro")
        assert proactive_profile_resp.status == 200
        proactive_profile_data = await proactive_profile_resp.json()
        assert proactive_profile_data["data"]["profile"]["risk_tolerance"] == "low"

        metrics_end_resp = await client.get("/api/v1/metrics")
        assert metrics_end_resp.status == 200
        metrics_end_data = await metrics_end_resp.json()
        counters = metrics_end_data["data"]["metrics"]["counters"]
        gauges = metrics_end_data["data"]["metrics"].get("gauges", {})
        assert any(k.startswith("task_submit_total:") for k in counters)
        assert any(k.startswith("skill_execute_total:") for k in counters)
        assert any(k.startswith("connector_invoke_total:") for k in counters)
        assert any(k.startswith("automation_event_total:") for k in counters)
        assert any(k.startswith("proactive_event_total:") for k in counters)
        assert "connectors_unhealthy_count:default" in gauges
        assert "automation_dead_letters_backlog:default" in gauges
    finally:
        await client.close()


@pytest.mark.skipif(TestClient is None or TestServer is None, reason="aiohttp test utilities unavailable")
async def test_openai_compat_chat_endpoints() -> None:
    api = APIInterface()
    api.set_conversation_manager(_DummyConversationManager())

    app = api._build_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        models_resp = await client.get("/v1/models")
        assert models_resp.status == 200
        models_data = await models_resp.json()
        assert models_data.get("object") == "list"
        assert isinstance(models_data.get("data"), list)
        assert len(models_data["data"]) >= 1

        completion_resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "jarvis-default",
                "messages": [{"role": "user", "content": "Hello from Continue"}],
                "stream": False,
                "user": "continue-user",
            },
        )
        assert completion_resp.status == 200
        payload = await completion_resp.json()
        assert payload.get("object") == "chat.completion"
        assert payload.get("choices")
        text = payload["choices"][0]["message"]["content"]
        assert "Hello from Continue" in text

        completion_debug_resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "jarvis-default",
                "messages": [{"role": "user", "content": "Hello from Continue"}],
                "stream": False,
                "user": "continue-user",
                "jarvis_debug": True,
            },
        )
        assert completion_debug_resp.status == 200
        payload_debug = await completion_debug_resp.json()
        dbg = payload_debug.get("jarvis_debug", {})
        assert dbg.get("session_id") == "session-continue-user"
        assert "stage_latency_ms" in dbg
        assert "conversation_latency_ms" in dbg
        assert "model_route" in dbg
        assert "summary_stage" in dbg
    finally:
        await client.close()


@pytest.mark.skipif(TestClient is None or TestServer is None, reason="aiohttp test utilities unavailable")
async def test_cors_headers_present_on_unauthorized_responses() -> None:
    api = APIInterface(auth_token="secret")
    app = api._build_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.post(
            "/api/v1/realtime/sessions/start",
            headers={"Origin": "http://localhost:3001"},
            json={"user_id": "u1", "max_frames": 16},
        )
        assert resp.status == 401
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3001"
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


@pytest.mark.skipif(TestClient is None or TestServer is None, reason="aiohttp test utilities unavailable")
async def test_vision_identity_enroll_recognize_and_delete() -> None:
    api = APIInterface()
    app = api._build_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        sample = "data:image/jpeg;base64," + ("QUJD" * 80)
        enroll_resp = await client.post(
            "/api/v1/vision/identities/enroll",
            json={"name": "Jiten", "samples": [sample, sample, sample]},
        )
        assert enroll_resp.status == 201
        enroll_data = await enroll_resp.json()
        identity = enroll_data["data"]["identity"]
        person_id = identity["person_id"]
        assert identity["display_name"] == "Jiten"

        list_resp = await client.get("/api/v1/vision/identities")
        assert list_resp.status == 200
        list_data = await list_resp.json()
        assert list_data["data"]["count"] >= 1

        rec_resp = await client.post(
            "/api/v1/vision/identities/recognize",
            json={"samples": [{"sample_id": "det-0", "detection_index": 0, "image_b64": sample}]},
        )
        assert rec_resp.status == 200
        rec_data = await rec_resp.json()
        assert rec_data["data"]["count"] == 1
        assert isinstance(rec_data["data"]["matches"], list)

        del_resp = await client.delete(f"/api/v1/vision/identities/{person_id}")
        assert del_resp.status == 200
        del_data = await del_resp.json()
        assert del_data["data"]["deleted"] is True
    finally:
        await client.close()


@pytest.mark.skipif(TestClient is None or TestServer is None, reason="aiohttp test utilities unavailable")
async def test_world_teach_and_enrich_endpoints() -> None:
    api = APIInterface()
    app = api._build_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        teach_resp = await client.post(
            "/api/v1/world/teach",
            json={
                "topic": "electric vehicles",
                "notes": "EVs convert stored battery power into wheel torque.",
                "tags": ["mobility", "energy"],
                "enrich_web": False,
            },
        )
        assert teach_resp.status == 201
        teach_data = await teach_resp.json()
        concept = teach_data["data"]["concept"]
        concept_id = concept["concept_id"]
        assert concept["topic"].lower() == "electric vehicles"

        list_resp = await client.get("/api/v1/world/concepts")
        assert list_resp.status == 200
        list_data = await list_resp.json()
        assert list_data["data"]["count"] >= 1

        enrich_resp = await client.post(
            f"/api/v1/world/concepts/{concept_id}/enrich",
            json={"max_items": 2, "run_adapters": False},
        )
        assert enrich_resp.status == 200
        enrich_data = await enrich_resp.json()
        assert "concept" in enrich_data["data"]
    finally:
        await client.close()
