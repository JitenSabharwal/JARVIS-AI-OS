from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict

import pytest

from core.agent_framework import AgentState
from core.orchestrator import MasterOrchestrator, PlanStep, TaskStatus
from infrastructure.approval import ApprovalManager


@dataclass
class _Metrics:
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_response_time: float = 0.01
    uptime: float = 10.0
    error_rate: float = 0.0


class _StubAgent:
    def __init__(
        self,
        *,
        agent_id: str,
        name: str,
        handlers: Dict[str, Callable[[Dict[str, Any]], Any]],
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.state = AgentState.IDLE
        self.metrics = _Metrics()
        self._handlers = handlers

    def has_capability(self, capability: str) -> bool:
        return capability in self._handlers

    def get_capabilities(self):
        return [SimpleNamespace(name=k) for k in self._handlers.keys()]

    async def execute_task(
        self,
        capability: str,
        payload: Dict[str, Any],
        *,
        timeout: float | None = None,
        task_id: str | None = None,
    ) -> Any:
        _ = (timeout, task_id)
        return self._handlers[capability](payload)


@pytest.mark.asyncio
async def test_orchestrator_confidence_escalation_requires_approval() -> None:
    ApprovalManager.reset_instance()
    orch = MasterOrchestrator(worker_poll_interval=0.05)
    await orch.start()
    try:
        agent = _StubAgent(
            agent_id="a1",
            name="executor",
            handlers={"build": lambda payload: {"ok": True, "payload": payload}},
        )
        await orch.register_agent(agent)  # type: ignore[arg-type]
        task_id = await orch.submit_task(
            description="build something",
            required_capabilities=["build"],
            payload={"x": 1},
            min_confidence=0.8,
            confidence_score=0.3,
        )
        task = orch.get_task_status(task_id)
        assert task is not None
        assert task.status == TaskStatus.WAITING_APPROVAL

        approval_manager = ApprovalManager.get_instance()
        req = approval_manager.create_request(
            action="orchestrator:execute:build",
            requested_by="tester",
            reason="execute build task",
            resource="task/build",
        )
        approval_manager.approve(req.approval_id, approver="lead")
        approved = await orch.approve_task(task_id, approval_token=req.approval_token)
        assert approved is True

        finished = await orch.wait_for_task(task_id, timeout=5.0)
        assert finished.status == TaskStatus.COMPLETED
        assert isinstance(finished.result, dict)
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_orchestrator_verifier_rejects_output() -> None:
    orch = MasterOrchestrator(worker_poll_interval=0.05)
    await orch.start()
    try:
        executor = _StubAgent(
            agent_id="exec-1",
            name="executor",
            handlers={"build": lambda payload: {"artifact": payload.get("name", "x")}},
        )
        verifier = _StubAgent(
            agent_id="ver-1",
            name="verifier",
            handlers={"verify_build": lambda payload: {"approved": False, "reason": "failing checks"}},
        )
        await orch.register_agent(executor)  # type: ignore[arg-type]
        await orch.register_agent(verifier)  # type: ignore[arg-type]

        task_id = await orch.submit_task(
            description="ship build",
            required_capabilities=["build"],
            payload={"name": "svc"},
            verifier_capability="verify_build",
        )
        finished = await orch.wait_for_task(task_id, timeout=5.0)
        assert finished.status == TaskStatus.FAILED
        assert "Verifier rejected" in (finished.error or "")
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_orchestrator_submit_task_plan_creates_dependency_graph() -> None:
    orch = MasterOrchestrator(worker_poll_interval=0.05)
    await orch.start()
    try:
        agent = _StubAgent(
            agent_id="a1",
            name="planner-exec",
            handlers={"build": lambda payload: {"done": payload.get("step")}},
        )
        await orch.register_agent(agent)  # type: ignore[arg-type]

        plan = await orch.submit_task_plan(
            description="deliver feature",
            steps=[
                PlanStep(name="design", capability="build", payload={"step": "design"}),
                PlanStep(name="implement", capability="build", payload={"step": "implement"}, depends_on=["design"]),
            ],
        )
        assert plan["step_count"] == 2
        t_design = orch.get_task_status(plan["task_ids_by_step"]["design"])
        t_impl = orch.get_task_status(plan["task_ids_by_step"]["implement"])
        assert t_design is not None and t_impl is not None
        assert t_design.plan_id == t_impl.plan_id
        assert t_impl.dependencies == [t_design.id]
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_orchestrator_retry_failed_task_recovers() -> None:
    attempts = {"count": 0}

    def flaky(_payload: Dict[str, Any]) -> Dict[str, Any]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("first failure")
        return {"ok": True}

    orch = MasterOrchestrator(worker_poll_interval=0.05)
    await orch.start()
    try:
        agent = _StubAgent(
            agent_id="a1",
            name="executor",
            handlers={"build": flaky},
        )
        await orch.register_agent(agent)  # type: ignore[arg-type]
        task_id = await orch.submit_task(description="flaky", required_capabilities=["build"])
        first = await orch.wait_for_task(task_id, timeout=5.0)
        assert first.status == TaskStatus.FAILED

        retried = await orch.retry_task(task_id)
        assert retried is True
        second = await orch.wait_for_task(task_id, timeout=5.0)
        assert second.status == TaskStatus.COMPLETED
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_orchestrator_replan_task_replaces_plan_step_task_id() -> None:
    orch = MasterOrchestrator(worker_poll_interval=0.05)
    await orch.start()
    try:
        executor = _StubAgent(
            agent_id="a1",
            name="executor",
            handlers={
                "primary": lambda _payload: (_ for _ in ()).throw(RuntimeError("primary failed")),
                "fallback": lambda payload: {"ok": True, "payload": payload},
            },
        )
        await orch.register_agent(executor)  # type: ignore[arg-type]

        plan = await orch.submit_task_plan(
            description="recoverable plan",
            steps=[PlanStep(name="step1", capability="primary", payload={"v": 1})],
        )
        old_task_id = plan["task_ids_by_step"]["step1"]
        old_task = await orch.wait_for_task(old_task_id, timeout=5.0)
        assert old_task.status == TaskStatus.FAILED

        new_task_id = await orch.replan_task(
            old_task_id,
            fallback_capabilities=["fallback"],
            payload_override={"v": 2},
        )
        assert new_task_id is not None
        new_task = await orch.wait_for_task(new_task_id, timeout=5.0)
        assert new_task.status == TaskStatus.COMPLETED

        status = orch.get_plan_status(plan["plan_id"])
        assert status is not None
        assert status["steps"]["step1"] == TaskStatus.COMPLETED.value
    finally:
        await orch.stop()


def test_orchestrator_plan_persistence_roundtrip(tmp_path) -> None:
    persist_path = tmp_path / "plans.json"
    orch = MasterOrchestrator(plan_persist_path=str(persist_path))
    record = {
        "plan_id": "plan-test",
        "description": "persisted",
        "task_ids_by_step": {"s1": "task-1"},
        "created_at": "2026-03-18T00:00:00+00:00",
        "updated_at": "2026-03-18T00:00:00+00:00",
        "metadata": {"x": 1},
    }
    persist_path.write_text(f"[{record}]", encoding="utf-8")
    # malformed JSON-like (single quotes) should be ignored safely
    orch2 = MasterOrchestrator(plan_persist_path=str(persist_path))
    assert orch2.get_plan_status("plan-test") is None

    persist_path.write_text(
        '[{"plan_id":"plan-test","description":"persisted","task_ids_by_step":{"s1":"task-1"}}]',
        encoding="utf-8",
    )
    orch3 = MasterOrchestrator(plan_persist_path=str(persist_path))
    status = orch3.get_plan_status("plan-test")
    assert status is not None
    assert status["plan_id"] == "plan-test"


@pytest.mark.asyncio
async def test_orchestrator_auto_replan_policy_creates_replacement_task() -> None:
    orch = MasterOrchestrator(worker_poll_interval=0.05)
    await orch.start()
    try:
        agent = _StubAgent(
            agent_id="a1",
            name="exec",
            handlers={
                "primary": lambda _payload: (_ for _ in ()).throw(RuntimeError("primary failed")),
                "fallback": lambda payload: {"ok": True, "payload": payload},
            },
        )
        await orch.register_agent(agent)  # type: ignore[arg-type]
        task_id = await orch.submit_task(
            description="auto replan task",
            required_capabilities=["primary"],
            metadata={
                "auto_replan_capabilities": ["fallback"],
                "auto_replan_max_attempts": 1,
            },
        )
        failed = await orch.wait_for_task(task_id, timeout=5.0)
        assert failed.status == TaskStatus.FAILED
        replacement_id = str(failed.metadata.get("auto_replan_created_task_id", ""))
        assert replacement_id
        replacement = await orch.wait_for_task(replacement_id, timeout=5.0)
        assert replacement.status == TaskStatus.COMPLETED
        assert replacement.parent_task_id == failed.id
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_orchestrator_requires_valid_approval_token_not_just_presence() -> None:
    ApprovalManager.reset_instance()
    orch = MasterOrchestrator(worker_poll_interval=0.05)
    await orch.start()
    try:
        agent = _StubAgent(
            agent_id="a1",
            name="exec",
            handlers={"build": lambda payload: {"ok": True, "payload": payload}},
        )
        await orch.register_agent(agent)  # type: ignore[arg-type]

        task_id = await orch.submit_task(
            description="guarded build",
            required_capabilities=["build"],
            requires_human=True,
            approval_token="bogus-token",
        )
        task = orch.get_task_status(task_id)
        assert task is not None
        assert task.status == TaskStatus.WAITING_APPROVAL

        bad_approved = await orch.approve_task(task_id, approval_token="still-bogus")
        assert bad_approved is False
        task_after_bad = orch.get_task_status(task_id)
        assert task_after_bad is not None
        assert task_after_bad.status == TaskStatus.WAITING_APPROVAL

        approval_manager = ApprovalManager.get_instance()
        req = approval_manager.create_request(
            action="orchestrator:execute:build",
            requested_by="tester",
            reason="guarded build execution",
            resource="task/build",
        )
        approval_manager.approve(req.approval_id, approver="security")

        ok = await orch.approve_task(task_id, approval_token=req.approval_token)
        assert ok is True
        done = await orch.wait_for_task(task_id, timeout=5.0)
        assert done.status == TaskStatus.COMPLETED
    finally:
        await orch.stop()
