"""
Master task orchestrator with multi-agent coordination for JARVIS AI OS.

Responsibilities:
- Accepting tasks from callers and routing them to capable agents
- Resolving inter-task dependencies (topological sort)
- Parallel execution of independent tasks
- Workflow definition execution
- System-wide status aggregation
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from core.agent_framework import AgentState, BaseAgent
from core.strategy_engine import StrategyEngine
from infrastructure.approval import ApprovalManager
from infrastructure.artifact_store import ArtifactStore
from infrastructure.langgraph_adapter import LangGraphWorkflowAdapter
from infrastructure.retention_manager import RetentionManager
from infrastructure.resource_pool_manager import ResourceLease, ResourcePoolManager
from infrastructure.tool_isolation import ToolIsolationPolicy
from memory.episodic_memory import EpisodicMemory
from infrastructure.logger import get_logger
from utils.exceptions import (
    AgentCapabilityError,
    AgentNotFoundError,
    OrchestratorError,
    TaskDependencyError,
    TaskTimeoutError,
    WorkflowError,
)
from utils.helpers import generate_id, timestamp_now

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskStatus(Enum):
    """All possible states of a submitted task."""

    PENDING = "pending"
    WAITING_DEPS = "waiting_deps"
    WAITING_APPROVAL = "waiting_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A unit of work submitted to the orchestrator.

    Args:
        description: Human-readable task description.
        required_capabilities: List of capability names the executing agent must
            support.  The first entry is treated as the primary capability.
        priority: Higher numbers are scheduled first (default 0).
        dependencies: IDs of tasks that must complete before this task runs.
        payload: Arbitrary parameters forwarded to the agent capability.
        id: Auto-generated unique task ID.
        status: Current lifecycle status.
        result: Task output once completed.
        error: Error message if the task failed.
        created_at: ISO-8601 creation timestamp.
        started_at: ISO-8601 timestamp when execution began.
        completed_at: ISO-8601 timestamp when execution ended.
        assigned_agent_id: ID of the agent assigned to run this task.
        timeout: Optional execution deadline in seconds.
    """

    description: str
    required_capabilities: List[str]
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: generate_id("task"))
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: str = field(default_factory=timestamp_now)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    assigned_agent_id: Optional[str] = None
    timeout: Optional[float] = None
    # Phase 7: planning + verification + human escalation metadata
    plan_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    milestone: Optional[str] = None
    verifier_capability: Optional[str] = None
    min_confidence: Optional[float] = None
    confidence_score: Optional[float] = None
    requires_human: bool = False
    approval_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """``True`` when the task has reached a final state."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Wall-clock duration if the task has both start and end timestamps."""
        if self.started_at and self.completed_at:
            from datetime import datetime, timezone

            fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
            try:
                start = datetime.fromisoformat(self.started_at)
                end = datetime.fromisoformat(self.completed_at)
                return (end - start).total_seconds()
            except ValueError:
                return None
        return None


@dataclass
class WorkflowStep:
    """A single step inside a :class:`WorkflowDefinition`.

    Args:
        name: Step identifier (unique within the workflow).
        capability: Capability to invoke.
        payload: Static parameters merged with runtime context.
        depends_on: Names of prior steps this step depends on.
        timeout: Optional step-level timeout in seconds.
    """

    name: str
    capability: str
    payload: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    timeout: Optional[float] = None


@dataclass
class WorkflowDefinition:
    """A named, ordered graph of :class:`WorkflowStep` objects.

    Args:
        name: Human-readable workflow name.
        steps: Ordered list of steps (dependency order is resolved internally).
        metadata: Free-form dict for tagging, versioning, etc.
        id: Auto-generated workflow definition ID.
    """

    name: str
    steps: List[WorkflowStep]
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: generate_id("wf"))


@dataclass
class PlanStep:
    """Plan graph step definition used by submit_task_plan."""

    name: str
    capability: str
    payload: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    verifier_capability: Optional[str] = None
    min_confidence: Optional[float] = None


@dataclass
class TaskPlanRecord:
    """Persistable in-memory record for a submitted task plan."""

    plan_id: str
    description: str
    task_ids_by_step: Dict[str, str]
    created_at: str = field(default_factory=timestamp_now)
    updated_at: str = field(default_factory=timestamp_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "description": self.description,
            "task_ids_by_step": dict(self.task_ids_by_step),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskPlanRecord":
        return cls(
            plan_id=str(data.get("plan_id", "")),
            description=str(data.get("description", "")),
            task_ids_by_step=dict(data.get("task_ids_by_step", {})),
            created_at=str(data.get("created_at", timestamp_now())),
            updated_at=str(data.get("updated_at", timestamp_now())),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Master orchestrator
# ---------------------------------------------------------------------------


class MasterOrchestrator:
    """Central coordinator that routes tasks to agents and executes workflows.

    The orchestrator maintains an internal registry of agents, a task store,
    and a background worker loop that continuously dispatches pending tasks.

    Args:
        max_concurrent_tasks: Upper limit on simultaneously running tasks.
        default_task_timeout: Fallback timeout (seconds) for tasks without one.
        worker_poll_interval: Seconds between worker loop iterations.
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 20,
        default_task_timeout: float = 60.0,
        worker_poll_interval: float = 0.5,
        episodic_memory: Optional[EpisodicMemory] = None,
        plan_persist_path: Optional[str] = None,
        auto_persist_plans: bool = True,
    ) -> None:
        self._agents: Dict[str, BaseAgent] = {}
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Set[str] = set()
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._running: bool = False
        self._max_concurrent = max_concurrent_tasks
        self._default_timeout = default_task_timeout
        self._poll_interval = worker_poll_interval
        self._episodic_memory: EpisodicMemory = episodic_memory or EpisodicMemory()
        self._start_time: Optional[float] = None
        self._task_callbacks: Dict[str, List[Callable[[Task], None]]] = defaultdict(
            list
        )
        self._plans: Dict[str, TaskPlanRecord] = {}
        self._workflow_checkpoints: Dict[str, Dict[str, Any]] = {}
        self._plan_persist_path = plan_persist_path
        self._auto_persist_plans = auto_persist_plans
        self._workflow_checkpoint_persist_path = str(
            os.getenv("JARVIS_AGENT_WORKFLOW_CHECKPOINT_PATH", "data/workflow_checkpoints.json")
        ).strip()
        self._workflow_checkpoint_backend = str(
            os.getenv("JARVIS_AGENT_WORKFLOW_CHECKPOINT_BACKEND", "file")
        ).strip().lower()
        self._workflow_lane_caps = self._load_workflow_lane_caps_from_env()
        self._workflow_lane_priority = self._load_workflow_lane_priority_from_env()
        self._strategy_engine = StrategyEngine.from_env()
        self._resource_pools = ResourcePoolManager.from_env()
        self._workflow_step_max_retries = max(
            0, int(os.getenv("JARVIS_AGENT_WORKFLOW_STEP_MAX_RETRIES", "1") or 1)
        )
        self._workflow_step_result_contract_strict = str(
            os.getenv("JARVIS_AGENT_WORKFLOW_STEP_RESULT_CONTRACT_STRICT", "true")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._task_payload_contract_strict = str(
            os.getenv("JARVIS_AGENT_TASK_PAYLOAD_CONTRACT_STRICT", "true")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._max_pending_tasks = max(
            1, int(os.getenv("JARVIS_ORCH_MAX_PENDING_TASKS", "2000") or 2000)
        )
        self._tool_isolation = ToolIsolationPolicy.from_env()
        self._artifact_store = ArtifactStore.from_env()
        self._retention = RetentionManager.from_env()
        self._retention_last_run = 0.0
        self._retention_interval_s = max(
            10.0, float(os.getenv("JARVIS_RETENTION_RUN_INTERVAL_SECONDS", "300") or 300)
        )
        self._workflow_lane_rr_offset = 0
        self._workflow_stats: Dict[str, int] = {
            "planner_langgraph": 0,
            "planner_native": 0,
            "planner_native_fallback": 0,
            "lane_backpressure_waits": 0,
            "lane_admission_denied": 0,
            "workflow_recoveries": 0,
            "strategy_baseline": 0,
            "strategy_adaptive": 0,
            "resource_acquire_failed": 0,
            "payload_contract_rejects": 0,
            "ingress_rejected": 0,
            "tool_isolation_rejects": 0,
            "checkpoint_retention_removed": 0,
        }
        self._lock = asyncio.Lock()
        self._approval_manager: ApprovalManager = ApprovalManager.get_instance()
        self._langgraph_adapter: LangGraphWorkflowAdapter | None = None
        self._logger = get_logger(__name__)
        if self._plan_persist_path:
            self._load_plans()
        self._load_workflow_checkpoints()
        removed = self._retention.prune_workflow_checkpoints(self._workflow_checkpoints)
        if removed > 0:
            self._workflow_stats["checkpoint_retention_removed"] = int(
                self._workflow_stats.get("checkpoint_retention_removed", 0)
            ) + int(removed)
            self._persist_workflow_checkpoints()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the orchestrator and its background worker loop."""
        if self._running:
            self._logger.warning("Orchestrator already running; start() ignored")
            return

        self._logger.info("Starting MasterOrchestrator")
        self._running = True
        self._start_time = time.monotonic()
        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="orchestrator-worker"
        )
        self._logger.info("MasterOrchestrator started")

    async def stop(self) -> None:
        """Gracefully stop the orchestrator, cancelling the worker loop."""
        if not self._running:
            return

        self._logger.info("Stopping MasterOrchestrator")
        self._running = False

        # Cancel pending tasks that have not yet started
        async with self._lock:
            for task in self._tasks.values():
                if task.status in (TaskStatus.PENDING, TaskStatus.WAITING_DEPS):
                    task.status = TaskStatus.CANCELLED
                    self._logger.debug("Cancelled pending task %s on shutdown", task.id)

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        self._logger.info("MasterOrchestrator stopped")

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    async def register_agent(self, agent: BaseAgent) -> None:
        """Register *agent* so the orchestrator can assign tasks to it.

        Args:
            agent: A :class:`~core.agent_framework.BaseAgent` instance.
        """
        async with self._lock:
            self._agents[agent.agent_id] = agent
        self._logger.info(
            "Registered agent '%s' (%s) with capabilities: %s",
            agent.name,
            agent.agent_id,
            [c.name for c in agent.get_capabilities()],
        )

    async def unregister_agent(self, agent_id: str) -> None:
        """Remove *agent_id* from the registry.

        Args:
            agent_id: ID of the agent to remove.

        Raises:
            AgentNotFoundError: If no agent with *agent_id* is registered.
        """
        async with self._lock:
            if agent_id not in self._agents:
                raise AgentNotFoundError(agent_id)
            del self._agents[agent_id]
        self._logger.info("Unregistered agent %s", agent_id)

    def set_approval_manager(self, approval_manager: ApprovalManager) -> None:
        self._approval_manager = approval_manager

    def set_langgraph_adapter(self, langgraph_adapter: LangGraphWorkflowAdapter | None) -> None:
        self._langgraph_adapter = langgraph_adapter

    # ------------------------------------------------------------------
    # Task submission
    # ------------------------------------------------------------------

    async def submit_task(
        self,
        description: str,
        required_capabilities: List[str],
        *,
        payload: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        dependencies: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        on_complete: Optional[Callable[[Task], None]] = None,
        plan_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        milestone: Optional[str] = None,
        verifier_capability: Optional[str] = None,
        min_confidence: Optional[float] = None,
        confidence_score: Optional[float] = None,
        requires_human: bool = False,
        approval_token: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a new task and return its ID.

        Args:
            description: Human-readable task description.
            required_capabilities: List of capability names required.
            payload: Parameters forwarded to the agent capability.
            priority: Higher values are scheduled first.
            dependencies: Task IDs that must complete before this task.
            timeout: Execution timeout override (uses default when ``None``).
            on_complete: Optional callback invoked when the task terminates.

        Returns:
            The newly created task ID.

        Raises:
            OrchestratorError: If the orchestrator is not running.
        """
        if not self._running:
            raise OrchestratorError("Orchestrator is not running")
        if not required_capabilities:
            raise OrchestratorError(
                "Task must include at least one required capability"
            )
        pending_count = sum(
            1
            for t in self._tasks.values()
            if t.status in {TaskStatus.PENDING, TaskStatus.WAITING_DEPS, TaskStatus.WAITING_APPROVAL}
        )
        if pending_count >= self._max_pending_tasks:
            self._workflow_stats["ingress_rejected"] = int(
                self._workflow_stats.get("ingress_rejected", 0)
            ) + 1
            raise OrchestratorError(
                f"Task rejected: orchestrator pending queue limit reached ({self._max_pending_tasks})"
            )

        task = Task(
            description=description,
            required_capabilities=required_capabilities,
            payload=payload or {},
            priority=priority,
            dependencies=dependencies or [],
            timeout=timeout,
            plan_id=plan_id,
            parent_task_id=parent_task_id,
            milestone=milestone,
            verifier_capability=verifier_capability,
            min_confidence=min_confidence,
            confidence_score=confidence_score,
            requires_human=requires_human,
            approval_token=approval_token,
            metadata=metadata or {},
        )
        payload_err = self._validate_task_payload_contract(task.payload)
        if payload_err:
            self._workflow_stats["payload_contract_rejects"] = int(
                self._workflow_stats.get("payload_contract_rejects", 0)
            ) + 1
            if self._task_payload_contract_strict:
                raise OrchestratorError(f"Task payload contract rejected: {payload_err}")
            task.metadata["payload_contract_warning"] = payload_err
        isolation_ok, isolation_err = self._tool_isolation.enforce_payload(
            capability=str(required_capabilities[0]),
            payload=task.payload,
        )
        if not isolation_ok:
            self._workflow_stats["tool_isolation_rejects"] = int(
                self._workflow_stats.get("tool_isolation_rejects", 0)
            ) + 1
            raise OrchestratorError(f"Task payload isolation rejected: {isolation_err}")

        if (
            task.min_confidence is not None
            and task.confidence_score is not None
            and float(task.confidence_score) < float(task.min_confidence)
        ):
            task.requires_human = True
            task.metadata.setdefault(
                "escalation_reason",
                f"confidence_below_threshold:{task.confidence_score}<{task.min_confidence}",
            )

        if task.requires_human and not self._is_task_approval_valid(task):
            task.status = TaskStatus.WAITING_APPROVAL
            task.error = "Task requires valid approved token before execution"
        elif task.dependencies:
            task.status = TaskStatus.WAITING_DEPS

        async with self._lock:
            self._tasks[task.id] = task
            if on_complete:
                self._task_callbacks[task.id].append(on_complete)

        self._logger.info(
            "Task submitted: id=%s description='%s' caps=%s priority=%d",
            task.id,
            description,
            required_capabilities,
            priority,
        )
        return task.id

    async def approve_task(self, task_id: str, approval_token: str) -> bool:
        """Approve a task that is waiting on human confirmation."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status != TaskStatus.WAITING_APPROVAL:
                return False
            task.approval_token = approval_token
            if not self._is_task_approval_valid(task):
                task.error = "Invalid or not approved token for task action"
                return False
            task.status = TaskStatus.WAITING_DEPS if task.dependencies else TaskStatus.PENDING
            task.error = None
            task.metadata["approved_at"] = timestamp_now()
            return True

    async def submit_task_plan(
        self,
        *,
        description: str,
        steps: List[Any],
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit a dependency-aware task plan and return plan/task mapping.
        """
        if not steps:
            raise OrchestratorError("Plan must include at least one step")
        normalized_steps: List[PlanStep] = []
        for step in steps:
            if isinstance(step, PlanStep):
                normalized_steps.append(step)
                continue
            if isinstance(step, dict):
                normalized_steps.append(
                    PlanStep(
                        name=str(step.get("name", "")).strip(),
                        capability=str(step.get("capability", "")).strip(),
                        payload=dict(step.get("payload", {}) or {}),
                        depends_on=list(step.get("depends_on", []) or []),
                        verifier_capability=(
                            str(step.get("verifier_capability", "")).strip() or None
                        ),
                        min_confidence=(
                            float(step["min_confidence"]) if step.get("min_confidence") is not None else None
                        ),
                    )
                )
                continue
            raise OrchestratorError("Each plan step must be PlanStep or object")
        steps = normalized_steps
        for step in steps:
            if not step.name or not step.capability:
                raise OrchestratorError("Each plan step requires non-empty 'name' and 'capability'")
        plan_id = generate_id("plan")
        step_index = {s.name: s for s in steps}
        if len(step_index) != len(steps):
            raise OrchestratorError("Plan step names must be unique")
        task_ids_by_step: Dict[str, str] = {}

        for step in steps:
            dep_task_ids: List[str] = []
            for dep_name in step.depends_on:
                if dep_name not in step_index:
                    raise OrchestratorError(f"Unknown plan dependency step: {dep_name}")
                dep_tid = task_ids_by_step.get(dep_name)
                if dep_tid is None:
                    raise OrchestratorError(
                        f"Plan dependency '{dep_name}' must appear before '{step.name}'"
                    )
                dep_task_ids.append(dep_tid)

            tid = await self.submit_task(
                description=f"{description}/{step.name}",
                required_capabilities=[step.capability],
                payload=step.payload,
                priority=priority,
                dependencies=dep_task_ids,
                plan_id=plan_id,
                parent_task_id=None,
                milestone=step.name,
                verifier_capability=step.verifier_capability,
                min_confidence=step.min_confidence,
                confidence_score=None,
                metadata={
                    "plan_description": description,
                    "plan_step": step.name,
                    **(metadata or {}),
                },
            )
            task_ids_by_step[step.name] = tid

        record = TaskPlanRecord(
            plan_id=plan_id,
            description=description,
            task_ids_by_step=task_ids_by_step,
            metadata=metadata or {},
        )
        async with self._lock:
            self._plans[plan_id] = record
        self._persist_plans()

        return {
            "plan_id": plan_id,
            "description": description,
            "task_ids_by_step": task_ids_by_step,
            "step_count": len(task_ids_by_step),
        }

    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        record = self._plans.get(plan_id)
        if record is None:
            return None
        step_statuses: Dict[str, str] = {}
        statuses: List[TaskStatus] = []
        for step_name, task_id in record.task_ids_by_step.items():
            task = self._tasks.get(task_id)
            if task is None:
                step_statuses[step_name] = "missing"
                continue
            step_statuses[step_name] = task.status.value
            statuses.append(task.status)

        if statuses and all(s == TaskStatus.COMPLETED for s in statuses):
            overall = "completed"
        elif any(s == TaskStatus.FAILED for s in statuses):
            overall = "failed"
        elif any(s == TaskStatus.WAITING_APPROVAL for s in statuses):
            overall = "waiting_approval"
        elif any(s == TaskStatus.RUNNING for s in statuses):
            overall = "running"
        else:
            overall = "pending"
        return {
            "plan_id": plan_id,
            "description": record.description,
            "status": overall,
            "steps": step_statuses,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "metadata": record.metadata,
        }

    async def retry_task(self, task_id: str, *, approval_token: Optional[str] = None) -> bool:
        """Reset a failed/cancelled task so worker loop can re-dispatch it."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status not in {TaskStatus.FAILED, TaskStatus.CANCELLED}:
                return False
            task.error = None
            task.result = None
            task.started_at = None
            task.completed_at = None
            if approval_token:
                task.approval_token = approval_token
                if task.requires_human and not self._is_task_approval_valid(task):
                    task.error = "Invalid or not approved token for task action"
                    return False
            if task.requires_human and not self._is_task_approval_valid(task):
                task.status = TaskStatus.WAITING_APPROVAL
                task.error = "Task requires valid approved token before execution"
            elif task.dependencies:
                task.status = TaskStatus.WAITING_DEPS
            else:
                task.status = TaskStatus.PENDING
                task.error = None
            if task.plan_id and task.plan_id in self._plans:
                self._plans[task.plan_id].updated_at = timestamp_now()
        self._persist_plans()
        return True

    async def replan_task(
        self,
        task_id: str,
        *,
        fallback_capabilities: List[str],
        payload_override: Optional[Dict[str, Any]] = None,
        description_suffix: str = "replan",
        approval_token: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a replacement task for a failed task, preserving plan context.
        """
        source = self._tasks.get(task_id)
        if source is None:
            return None
        new_task_id = await self.submit_task(
            description=f"{source.description} ({description_suffix})",
            required_capabilities=fallback_capabilities or source.required_capabilities,
            payload=payload_override if payload_override is not None else dict(source.payload),
            priority=source.priority,
            dependencies=list(source.dependencies),
            timeout=source.timeout,
            plan_id=source.plan_id,
            parent_task_id=source.id,
            milestone=source.milestone,
            verifier_capability=source.verifier_capability,
            min_confidence=source.min_confidence,
            confidence_score=source.confidence_score,
            requires_human=source.requires_human,
            approval_token=approval_token,
            metadata={**source.metadata, "replanned_from_task_id": source.id},
        )
        if source.plan_id and source.plan_id in self._plans:
            plan = self._plans[source.plan_id]
            for step_name, tid in list(plan.task_ids_by_step.items()):
                if tid == source.id:
                    plan.task_ids_by_step[step_name] = new_task_id
                    plan.updated_at = timestamp_now()
                    break
            self._persist_plans()
        return new_task_id

    async def cancel_task(self, task_id: str) -> bool:
        """Request cancellation of *task_id*.

        Running tasks cannot be interrupted mid-execution; they will be
        marked cancelled after they complete or on the next dispatch cycle.

        Args:
            task_id: ID of the task to cancel.

        Returns:
            ``True`` if the task was found and marked for cancellation.
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.is_terminal:
                return False
            if task_id not in self._running_tasks:
                task.status = TaskStatus.CANCELLED
                task.completed_at = timestamp_now()
                self._logger.info("Task %s cancelled", task_id)
                return True

        self._logger.warning(
            "Task %s is currently running; cannot cancel mid-execution", task_id
        )
        return False

    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Return the :class:`Task` object for *task_id*, or ``None``.

        Args:
            task_id: Task identifier to look up.
        """
        return self._tasks.get(task_id)

    async def wait_for_task(
        self, task_id: str, *, poll_interval: float = 0.2, timeout: float = 300.0
    ) -> Task:
        """Block until *task_id* reaches a terminal state.

        Args:
            task_id: Task to wait for.
            poll_interval: Seconds between status checks.
            timeout: Maximum wait time in seconds.

        Returns:
            The completed :class:`Task`.

        Raises:
            TaskTimeoutError: If the task does not complete within *timeout*.
            OrchestratorError: If *task_id* is not found.
        """
        if task_id not in self._tasks:
            raise OrchestratorError(f"Unknown task_id: {task_id}")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            task = self._tasks[task_id]
            if task.is_terminal:
                return task
            await asyncio.sleep(poll_interval)

        raise TaskTimeoutError(task_id=task_id, timeout_seconds=timeout)

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    async def orchestrate_workflow(
        self,
        workflow: WorkflowDefinition,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a :class:`WorkflowDefinition` and return step results.

        Steps are executed in dependency order; independent steps run in
        parallel.  Results from each step are available to subsequent steps
        via the shared *context* dict under the step's name key.

        Args:
            workflow: The workflow to execute.
            context: Optional seed data merged into the execution context.

        Returns:
            A dict mapping step name → step result.

        Raises:
            WorkflowError: On validation or execution failure.
        """
        self._logger.info(
            "Executing workflow '%s' (%s) with %d steps",
            workflow.name,
            workflow.id,
            len(workflow.steps),
        )

        if not workflow.steps:
            raise WorkflowError(
                workflow_id=workflow.id, reason="Workflow contains no steps"
            )

        execution_order = self._resolve_workflow_dependencies(workflow)
        exec_context: Dict[str, Any] = dict(context or {})
        step_results: Dict[str, Any] = {}
        context_checkpoint = exec_context.get("__workflow_checkpoint", {})
        if not isinstance(context_checkpoint, dict):
            context_checkpoint = {}
        stored_checkpoint = self._workflow_checkpoints.get(workflow.id, {})
        if not isinstance(stored_checkpoint, dict):
            stored_checkpoint = {}
        start_wave = int(context_checkpoint.get("next_wave_index", stored_checkpoint.get("next_wave_index", 0)) or 0)
        checkpoint_results = context_checkpoint.get("step_results", stored_checkpoint.get("step_results", {}))
        if isinstance(checkpoint_results, dict):
            for k, v in checkpoint_results.items():
                step_results[str(k)] = v
                exec_context[str(k)] = v
        if start_wave > 0:
            self._workflow_stats["workflow_recoveries"] = int(self._workflow_stats.get("workflow_recoveries", 0)) + 1

        execution_state = self._init_workflow_execution_state(
            workflow=workflow,
            checkpoint=context_checkpoint if context_checkpoint else stored_checkpoint,
        )

        # Group steps into execution waves (all deps satisfied → can run together)
        waves = self._build_execution_waves(workflow, execution_order)
        flow_hints = {}
        if isinstance(workflow.metadata, dict):
            fp = workflow.metadata.get("flow_plan", {})
            if isinstance(fp, dict):
                sh = fp.get("step_hints", {})
                if isinstance(sh, dict):
                    flow_hints = sh
                engine = str(fp.get("engine", "")).strip().lower()
                if engine == "langgraph":
                    self._workflow_stats["planner_langgraph"] = int(self._workflow_stats.get("planner_langgraph", 0)) + 1
                elif engine == "native_fallback":
                    self._workflow_stats["planner_native_fallback"] = int(
                        self._workflow_stats.get("planner_native_fallback", 0)
                    ) + 1
                elif engine:
                    self._workflow_stats["planner_native"] = int(self._workflow_stats.get("planner_native", 0)) + 1

        for wave_idx, wave in enumerate(waves):
            if wave_idx < start_wave:
                continue
            self._logger.debug(
                "Workflow '%s' executing wave %d: %s",
                workflow.id,
                wave_idx,
                [s.name for s in wave],
            )
            wave_batches = self._split_wave_into_lane_batches(wave, flow_hints=flow_hints)
            for batch_idx, batch in enumerate(wave_batches):
                wave_task_ids = []
                step_map = {s.name: s for s in workflow.steps}
                leases_by_task: Dict[str, ResourceLease] = {}
                for step in batch:
                    await self._wait_for_lane_capacity(step=step, flow_hints=flow_hints)
                    pool = self._resource_pool_for_step(step, flow_hints)
                    try:
                        lease = await self._resource_pools.acquire(pool, timeout_s=30.0)
                    except Exception:
                        self._workflow_stats["resource_acquire_failed"] = int(
                            self._workflow_stats.get("resource_acquire_failed", 0)
                        ) + 1
                        raise WorkflowError(
                            workflow_id=workflow.id,
                            reason=f"Resource pool admission failed for step '{step.name}' on pool '{pool}'",
                        )
                    merged_payload = {**exec_context, **step.payload}
                    step_hint = flow_hints.get(step.name, {}) if isinstance(flow_hints, dict) else {}
                    task_metadata = {
                        "workflow_id": workflow.id,
                        "workflow_name": workflow.name,
                        "workflow_step": step.name,
                        "flow_wave_index": wave_idx,
                        "flow_batch_index": batch_idx,
                    }
                    if isinstance(step_hint, dict):
                        if str(step_hint.get("lane", "")).strip():
                            task_metadata["flow_lane"] = str(step_hint.get("lane"))
                        if str(step_hint.get("capability", "")).strip():
                            task_metadata["flow_capability"] = str(step_hint.get("capability"))
                        try:
                            task_metadata["flow_fan_in"] = int(step_hint.get("fan_in", 0) or 0)
                        except Exception:
                            pass
                    task_id = await self.submit_task(
                        description=f"{workflow.name}/{step.name}",
                        required_capabilities=[step.capability],
                        payload=merged_payload,
                        timeout=step.timeout,
                        metadata=task_metadata,
                    )
                    wave_task_ids.append((step.name, task_id))
                    leases_by_task[task_id] = lease

                # Wait for all tasks in this batch
                events: List[Dict[str, Any]] = []
                for step_name, task_id in wave_task_ids:
                    per_task_started = time.monotonic()
                    completed_task = await self._wait_workflow_task_with_retry(
                        workflow=workflow,
                        step_name=step_name,
                        task_id=task_id,
                        flow_hints=flow_hints,
                        exec_context=exec_context,
                        step_map=step_map,
                    )
                    self._resource_pools.release(leases_by_task.pop(task_id, None))
                    wait_ms = (time.monotonic() - per_task_started) * 1000.0
                    target_ms = None
                    step_obj = step_map.get(step_name)
                    if step_obj is not None and step_obj.timeout is not None:
                        target_ms = float(step_obj.timeout) * 1000.0
                    sla_violated = bool(target_ms is not None and wait_ms > target_ms)
                    self._strategy_engine.feedback(wait_ms=wait_ms, sla_violated=sla_violated)
                    if completed_task.status == TaskStatus.FAILED:
                        events.append({"step": step_name, "status": "failed"})
                        execution_state = self._transition_workflow_execution_state(
                            execution_state=execution_state,
                            events=events,
                        )
                        self._store_workflow_checkpoint(
                            workflow_id=workflow.id,
                            workflow_name=workflow.name,
                            next_wave_index=wave_idx,
                            step_results=step_results,
                            execution_state=execution_state,
                            failed_step=step_name,
                            status="failed",
                        )
                        self._artifact_store.append(
                            artifact_type="workflow",
                            payload={
                                "workflow_id": workflow.id,
                                "workflow_name": workflow.name,
                                "status": "failed",
                                "failed_step": step_name,
                                "reason": str(completed_task.error or ""),
                            },
                        )
                        raise WorkflowError(
                            workflow_id=workflow.id,
                            reason=f"Step '{step_name}' failed: {completed_task.error}",
                        )
                    step_contract_err = self._validate_workflow_step_result(
                        step_name=step_name,
                        result=completed_task.result,
                    )
                    if step_contract_err:
                        events.append({"step": step_name, "status": "failed"})
                        execution_state = self._transition_workflow_execution_state(
                            execution_state=execution_state,
                            events=events,
                        )
                        self._store_workflow_checkpoint(
                            workflow_id=workflow.id,
                            workflow_name=workflow.name,
                            next_wave_index=wave_idx,
                            step_results=step_results,
                            execution_state=execution_state,
                            failed_step=step_name,
                            status="failed",
                        )
                        self._artifact_store.append(
                            artifact_type="workflow",
                            payload={
                                "workflow_id": workflow.id,
                                "workflow_name": workflow.name,
                                "status": "failed",
                                "failed_step": step_name,
                                "reason": f"contract:{step_contract_err}",
                            },
                        )
                        raise WorkflowError(
                            workflow_id=workflow.id,
                            reason=f"Step '{step_name}' failed contract verification: {step_contract_err}",
                        )
                    events.append({"step": step_name, "status": "completed"})
                    step_results[step_name] = completed_task.result
                    exec_context[step_name] = completed_task.result
                for _, tid in wave_task_ids:
                    self._resource_pools.release(leases_by_task.pop(tid, None))
                execution_state = self._transition_workflow_execution_state(
                    execution_state=execution_state,
                    events=events,
                )
            self._store_workflow_checkpoint(
                workflow_id=workflow.id,
                workflow_name=workflow.name,
                next_wave_index=wave_idx + 1,
                step_results=step_results,
                execution_state=execution_state,
                status="running",
            )

        self._logger.info(
            "Workflow '%s' completed successfully", workflow.name
        )
        self._store_workflow_checkpoint(
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            next_wave_index=len(waves),
            step_results=step_results,
            execution_state=execution_state,
            status="completed",
        )
        self._artifact_store.append(
            artifact_type="workflow",
            payload={
                "workflow_id": workflow.id,
                "workflow_name": workflow.name,
                "status": "completed",
                "step_count": len(workflow.steps),
                "result_keys": sorted(step_results.keys()),
            },
        )
        return step_results

    # ------------------------------------------------------------------
    # Agent selection
    # ------------------------------------------------------------------

    def _find_suitable_agents(self, capability: str) -> List[BaseAgent]:
        """Return all registered agents that advertise *capability*.

        Args:
            capability: Capability name to match.

        Returns:
            List of matching :class:`~core.agent_framework.BaseAgent` instances.
        """
        return [
            agent
            for agent in self._agents.values()
            if agent.has_capability(capability)
            and agent.state != AgentState.OFFLINE
        ]

    def _select_best_agent(
        self, candidates: List[BaseAgent], *, capability: str = ""
    ) -> Optional[BaseAgent]:
        """Select the most available agent from *candidates* using load balancing.

        Selection heuristic (in order of preference):
        1. IDLE agents with the lowest ``tasks_completed + tasks_failed`` ratio
        2. BUSY agents with the lowest response time as a tiebreaker

        Args:
            candidates: Pool of agents to choose from.

        Returns:
            The best-fit agent, or ``None`` if *candidates* is empty.
        """
        if not candidates:
            return None

        idle = [a for a in candidates if a.state == AgentState.IDLE]
        pool = idle if idle else candidates

        def _score(agent: BaseAgent) -> tuple[float, float, int]:
            learned_success = (
                self._episodic_memory.get_agent_capability_success_rate(
                    agent_id=agent.agent_id,
                    capability=capability,
                )
                if capability
                else 0.5
            )
            # Lower is better: penalize high error/latency and low learned success.
            composite = (
                (agent.metrics.error_rate * 0.5)
                + (agent.metrics.avg_response_time * 0.3)
                + ((1.0 - learned_success) * 0.2)
            )
            return (composite, agent.metrics.avg_response_time, agent.metrics.tasks_completed)

        return min(pool, key=_score)

    # ------------------------------------------------------------------
    # Dependency resolution helpers
    # ------------------------------------------------------------------

    def _resolve_dependencies(self, task_ids: List[str]) -> List[str]:
        """Return *task_ids* sorted so every task appears after its dependencies.

        Uses Kahn's topological sort algorithm.

        Args:
            task_ids: Subset of task IDs to sort.

        Returns:
            A topologically sorted list of task IDs.

        Raises:
            TaskDependencyError: If a cycle is detected in the dependency graph.
        """
        id_set = set(task_ids)
        in_degree: Dict[str, int] = {tid: 0 for tid in task_ids}
        graph: Dict[str, List[str]] = defaultdict(list)

        for tid in task_ids:
            task = self._tasks.get(tid)
            if task is None:
                continue
            for dep in task.dependencies:
                if dep in id_set:
                    graph[dep].append(tid)
                    in_degree[tid] += 1

        queue: deque[str] = deque(
            tid for tid in task_ids if in_degree[tid] == 0
        )
        result: List[str] = []

        while queue:
            tid = queue.popleft()
            result.append(tid)
            for neighbor in graph[tid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(task_ids):
            raise TaskDependencyError(
                task_id="",
                dependency_id="",
            )
        return result

    def _resolve_workflow_dependencies(
        self, workflow: WorkflowDefinition
    ) -> List[str]:
        """Topologically sort workflow step names by their ``depends_on`` edges.

        Args:
            workflow: The workflow to analyse.

        Returns:
            Ordered list of step names.

        Raises:
            WorkflowError: If a cycle is detected.
        """
        name_set = {s.name for s in workflow.steps}
        step_map = {s.name: s for s in workflow.steps}
        in_degree: Dict[str, int] = {s.name: 0 for s in workflow.steps}
        graph: Dict[str, List[str]] = defaultdict(list)

        for step in workflow.steps:
            for dep in step.depends_on:
                if dep not in name_set:
                    raise WorkflowError(
                        workflow_id=workflow.id,
                        reason=f"Step '{step.name}' depends on unknown step '{dep}'",
                    )
                graph[dep].append(step.name)
                in_degree[step.name] += 1

        queue: deque[str] = deque(
            name for name, deg in in_degree.items() if deg == 0
        )
        result: List[str] = []

        while queue:
            name = queue.popleft()
            result.append(name)
            for neighbor in graph[name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(workflow.steps):
            raise WorkflowError(
                workflow_id=workflow.id,
                reason="Circular dependency detected in workflow steps",
            )
        return result

    def _build_execution_waves(
        self,
        workflow: WorkflowDefinition,
        order: List[str],
    ) -> List[List[WorkflowStep]]:
        """Group steps into parallel execution waves.

        Steps with no unsatisfied dependencies at a given point in time form
        a wave and can be executed concurrently.

        Args:
            workflow: Workflow definition.
            order: Topologically sorted step names.

        Returns:
            List of waves, each wave being a list of :class:`WorkflowStep`.
        """
        step_map = {s.name: s for s in workflow.steps}
        if self._langgraph_adapter and self._langgraph_adapter.available:
            try:
                step_defs = [
                    {
                        "name": step.name,
                        "depends_on": list(step.depends_on),
                        "capability": step.capability,
                    }
                    for step in workflow.steps
                ]
                flow_plan = self._langgraph_adapter.build_multi_agent_flow(step_defs)
                names_waves = flow_plan.get("waves", []) if isinstance(flow_plan, dict) else []
                step_hints = flow_plan.get("step_hints", {}) if isinstance(flow_plan, dict) else {}
                meta = getattr(self._langgraph_adapter, "last_plan_meta", {})
                if isinstance(meta, dict):
                    self._logger.info(
                        "Execution waves planned via %s (steps=%s waves=%s max_wave_size=%s)",
                        str(meta.get("engine", "unknown")),
                        int(meta.get("steps_total", len(step_defs)) or 0),
                        int(meta.get("waves_total", len(names_waves)) or 0),
                        int(meta.get("max_wave_size", 0) or 0),
                    )
                if isinstance(workflow.metadata, dict):
                    workflow.metadata["flow_plan"] = {
                        "engine": str(flow_plan.get("engine", "langgraph")) if isinstance(flow_plan, dict) else "langgraph",
                        "step_hints": step_hints if isinstance(step_hints, dict) else {},
                    }
                return [[step_map[name] for name in wave if name in step_map] for wave in names_waves]
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("LangGraph wave planning fallback to native planner: %s", exc)
        completed: Set[str] = set()
        remaining = list(order)
        waves: List[List[WorkflowStep]] = []

        while remaining:
            wave = [
                step_map[name]
                for name in remaining
                if all(dep in completed for dep in step_map[name].depends_on)
            ]
            if not wave:
                # Should never happen after topological sort, guard anyway
                raise WorkflowError(
                    workflow_id=workflow.id,
                    reason="Unable to build execution wave; possible dependency cycle",
                )
            waves.append(wave)
            completed.update(s.name for s in wave)
            remaining = [n for n in remaining if n not in completed]

        return waves

    # ------------------------------------------------------------------
    # Parallel task execution
    # ------------------------------------------------------------------

    async def _execute_parallel_tasks(self, task_ids: List[str]) -> List[Task]:
        """Run *task_ids* concurrently and return their completed :class:`Task` objects.

        Args:
            task_ids: IDs of tasks to execute in parallel.

        Returns:
            List of completed :class:`Task` objects in original order.
        """
        coroutines = [
            self._execute_task(self._tasks[tid])
            for tid in task_ids
            if tid in self._tasks
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        tasks = []
        for tid, result in zip(task_ids, results):
            task = self._tasks.get(tid)
            if isinstance(result, Exception):
                if task:
                    task.status = TaskStatus.FAILED
                    task.error = str(result)
                    task.completed_at = timestamp_now()
            if task:
                tasks.append(task)
        return tasks

    # ------------------------------------------------------------------
    # Core task execution
    # ------------------------------------------------------------------

    async def _execute_task(self, task: Task) -> Any:
        """Assign *task* to a suitable agent and await the result.

        Args:
            task: The :class:`Task` to execute.

        Returns:
            Task result from the agent.

        Raises:
            OrchestratorError: If no suitable agent is available.
            TaskTimeoutError: If execution exceeds the task's timeout.
        """
        required_caps = task.required_capabilities
        if not required_caps:
            task.status = TaskStatus.FAILED
            task.error = "Task has no required_capabilities; cannot assign to an agent"
            task.completed_at = timestamp_now()
            raise OrchestratorError(task.error)
        if task.requires_human and not self._is_task_approval_valid(task):
            task.status = TaskStatus.WAITING_APPROVAL
            task.error = "Task requires valid approved token before execution"
            return None

        # Find agents that satisfy ALL required capabilities
        candidates = [
            agent
            for cap in required_caps
            for agent in self._find_suitable_agents(cap)
        ]
        # Keep only agents that appear for every required capability
        cap_sets = [
            set(a.agent_id for a in self._find_suitable_agents(cap))
            for cap in required_caps
        ]
        if cap_sets:
            common_ids = cap_sets[0].intersection(*cap_sets[1:])
        else:
            common_ids = set()
        candidates = [a for a in self._agents.values() if a.agent_id in common_ids]
        primary_capability = required_caps[0]
        agent = self._select_best_agent(candidates, capability=primary_capability)

        if agent is None:
            task.status = TaskStatus.FAILED
            task.error = f"No available agent for capabilities {required_caps!r}"
            task.completed_at = timestamp_now()
            raise OrchestratorError(task.error)

        task.status = TaskStatus.RUNNING
        task.started_at = timestamp_now()
        task.assigned_agent_id = agent.agent_id

        async with self._lock:
            self._running_tasks.add(task.id)

        effective_timeout = task.timeout or self._default_timeout
        start_time = time.monotonic()

        try:
            result = await agent.execute_task(
                primary_capability,
                task.payload,
                timeout=effective_timeout,
                task_id=task.id,
            )
            verifier_result = None
            if task.verifier_capability:
                verifier_result = await self._run_verifier(task, result, effective_timeout)
                if not self._is_verification_passed(verifier_result):
                    task.status = TaskStatus.FAILED
                    task.error = "Verifier rejected task output"
                    task.result = {
                        "execution_result": result,
                        "verification": verifier_result,
                    }
                    await self._maybe_trigger_auto_replan(task)
                    raise OrchestratorError(task.error)
            task.result = result
            task.status = TaskStatus.COMPLETED
            self._episodic_memory.record_episode(
                task_description=task.description,
                actions_taken=required_caps,
                outcome="Task completed",
                success=True,
                duration=max(0.0, time.monotonic() - start_time),
                learned_facts=[],
                metadata={
                    "task_id": task.id,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "capability": primary_capability,
                    "verifier_capability": task.verifier_capability or "",
                    "verification_passed": bool(
                        True if verifier_result is None else self._is_verification_passed(verifier_result)
                    ),
                    "category": "orchestrator_task",
                },
            )
            self._logger.info(
                "Task %s completed by agent '%s'", task.id, agent.name
            )
            return result

        except (TaskTimeoutError, AgentCapabilityError) as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            await self._maybe_trigger_auto_replan(task)
            self._episodic_memory.record_episode(
                task_description=task.description,
                actions_taken=required_caps,
                outcome=f"Task failed: {exc}",
                success=False,
                duration=max(0.0, time.monotonic() - start_time),
                learned_facts=[f"Failure on capability {primary_capability}: {type(exc).__name__}"],
                metadata={
                    "task_id": task.id,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "capability": primary_capability,
                    "category": "orchestrator_task",
                },
                error=str(exc),
            )
            raise

        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            await self._maybe_trigger_auto_replan(task)
            self._episodic_memory.record_episode(
                task_description=task.description,
                actions_taken=required_caps,
                outcome=f"Task failed: {exc}",
                success=False,
                duration=max(0.0, time.monotonic() - start_time),
                learned_facts=[f"Unhandled failure on capability {primary_capability}"],
                metadata={
                    "task_id": task.id,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "capability": primary_capability,
                    "category": "orchestrator_task",
                },
                error=str(exc),
            )
            self._logger.error("Task %s failed: %s", task.id, exc)
            raise

        finally:
            task.completed_at = timestamp_now()
            async with self._lock:
                self._running_tasks.discard(task.id)
            self._artifact_store.append(
                artifact_type="task",
                payload={
                    "task_id": task.id,
                    "description": task.description,
                    "status": task.status.value,
                    "error": task.error or "",
                    "assigned_agent_id": task.assigned_agent_id or "",
                    "required_capabilities": list(task.required_capabilities),
                    "workflow_id": str(task.metadata.get("workflow_id", "")),
                    "workflow_step": str(task.metadata.get("workflow_step", "")),
                },
            )
            self._fire_callbacks(task)

    # ------------------------------------------------------------------
    # Background worker loop
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Continuously dispatch pending tasks to available agents."""
        self._logger.debug("Orchestrator worker loop started")
        while self._running:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._dispatch_pending_tasks()
                await self._run_retention_if_due()
            except Exception as exc:
                self._logger.error("Worker loop error: %s", exc)
        self._logger.debug("Orchestrator worker loop stopped")

    async def _run_retention_if_due(self) -> None:
        now = time.monotonic()
        if (now - self._retention_last_run) < self._retention_interval_s:
            return
        self._retention_last_run = now
        removed = self._retention.prune_workflow_checkpoints(self._workflow_checkpoints)
        if removed > 0:
            self._workflow_stats["checkpoint_retention_removed"] = int(
                self._workflow_stats.get("checkpoint_retention_removed", 0)
            ) + int(removed)
            self._persist_workflow_checkpoints()
        self._artifact_store.cleanup()

    async def _dispatch_pending_tasks(self) -> None:
        """Promote waiting tasks and dispatch as many pending tasks as possible."""
        async with self._lock:
            # Promote WAITING_DEPS tasks whose deps are all complete
            for task in list(self._tasks.values()):
                if task.status == TaskStatus.WAITING_DEPS:
                    if self._dependencies_met(task):
                        task.status = TaskStatus.PENDING

            # Collect dispatchable tasks, sorted by priority descending
            dispatchable = sorted(
                [
                    t
                    for t in self._tasks.values()
                    if t.status == TaskStatus.PENDING
                    and t.id not in self._running_tasks
                ],
                key=lambda t: t.priority,
                reverse=True,
            )

            slots = self._max_concurrent - len(self._running_tasks)
            to_dispatch = dispatchable[:slots]

        for task in to_dispatch:
            bg_task = asyncio.create_task(
                self._execute_task(task),
                name=f"task-{task.id}",
            )
            bg_task.add_done_callback(self._consume_background_exception)

    def _consume_background_exception(self, fut: "asyncio.Task[Any]") -> None:
        """Consume/log background task exceptions to avoid unhandled warnings."""
        try:
            _ = fut.result()
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("Background task ended with exception: %s", exc)

    async def _run_verifier(self, task: Task, execution_result: Any, timeout: float) -> Any:
        verifier_capability = task.verifier_capability or ""
        candidates = self._find_suitable_agents(verifier_capability)
        verifier = self._select_best_agent(candidates, capability=verifier_capability)
        if verifier is None:
            raise OrchestratorError(
                f"No available verifier agent for capability {verifier_capability!r}"
            )
        return await verifier.execute_task(
            verifier_capability,
            {
                "task_id": task.id,
                "task_description": task.description,
                "task_payload": task.payload,
                "task_result": execution_result,
                "task_metadata": task.metadata,
            },
            timeout=timeout,
            task_id=f"{task.id}:verify",
        )

    async def _maybe_trigger_auto_replan(self, task: Task) -> Optional[str]:
        """
        Optional failover policy:
        metadata.auto_replan_capabilities: list[str]
        metadata.auto_replan_max_attempts: int (default 0 -> disabled)
        """
        policy_caps = task.metadata.get("auto_replan_capabilities", [])
        if not isinstance(policy_caps, list) or not policy_caps:
            return None
        max_attempts = int(task.metadata.get("auto_replan_max_attempts", 0))
        if max_attempts <= 0:
            return None
        current = int(task.metadata.get("auto_replan_count", 0))
        if current >= max_attempts:
            return None
        if task.parent_task_id:
            # Avoid uncontrolled replan chains.
            return None
        task.metadata["auto_replan_count"] = current + 1
        new_task_id = await self.replan_task(
            task.id,
            fallback_capabilities=[str(c) for c in policy_caps if str(c).strip()],
            description_suffix=f"auto-replan-{current + 1}",
        )
        if new_task_id:
            task.metadata["auto_replan_created_task_id"] = new_task_id
        return new_task_id

    @staticmethod
    def _is_verification_passed(verification_result: Any) -> bool:
        if isinstance(verification_result, bool):
            return verification_result
        if isinstance(verification_result, dict):
            if "approved" in verification_result:
                return bool(verification_result.get("approved"))
            if "success" in verification_result:
                return bool(verification_result.get("success"))
            if "status" in verification_result:
                return str(verification_result.get("status", "")).lower() in {"approved", "pass", "passed", "ok"}
        return bool(verification_result)

    def _task_approval_action(self, task: Task) -> str:
        configured = task.metadata.get("approval_action")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        primary_capability = str(task.required_capabilities[0]).strip() if task.required_capabilities else "unknown"
        return f"orchestrator:execute:{primary_capability}"

    def _is_task_approval_valid(self, task: Task) -> bool:
        if not task.requires_human:
            return True
        token = str(task.approval_token or "").strip()
        if not token:
            return False
        action = self._task_approval_action(task)
        valid = self._approval_manager.validate_token(token, expected_action=action)
        if not valid:
            task.metadata["approval_validation_error"] = f"invalid_token_for_action:{action}"
        else:
            task.metadata.pop("approval_validation_error", None)
        return valid

    def _dependencies_met(self, task: Task) -> bool:
        """Return ``True`` if all of *task*'s dependencies have completed."""
        for dep_id in task.dependencies:
            dep = self._tasks.get(dep_id)
            if dep is None or dep.status != TaskStatus.COMPLETED:
                return False
        return True

    # ------------------------------------------------------------------
    # Callback management
    # ------------------------------------------------------------------

    def _fire_callbacks(self, task: Task) -> None:
        """Invoke registered completion callbacks for *task*."""
        for callback in self._task_callbacks.get(task.id, []):
            try:
                callback(task)
            except Exception as exc:
                self._logger.error(
                    "Callback error for task %s: %s", task.id, exc
                )

    # ------------------------------------------------------------------
    # Plan persistence helpers
    # ------------------------------------------------------------------

    def _load_plans(self) -> None:
        path = self._plan_persist_path
        if not path:
            return
        file = Path(path)
        if not file.exists():
            return
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else []
            loaded: Dict[str, TaskPlanRecord] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                record = TaskPlanRecord.from_dict(row)
                if record.plan_id:
                    loaded[record.plan_id] = record
            self._plans = loaded
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed loading persisted plans from %s: %s", path, exc)

    def _persist_plans(self) -> None:
        if not self._auto_persist_plans or not self._plan_persist_path:
            return
        try:
            file = Path(self._plan_persist_path)
            file.parent.mkdir(parents=True, exist_ok=True)
            rows = [record.to_dict() for record in self._plans.values()]
            file.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed persisting plans to %s: %s", self._plan_persist_path, exc)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_system_status(self) -> Dict[str, Any]:
        """Return an aggregate snapshot of the entire system state.

        Returns:
            A dict containing agent and task statistics suitable for
            health-check endpoints or dashboards.
        """
        agent_summaries = [
            {
                "id": a.agent_id,
                "name": a.name,
                "state": a.state.name,
                "capabilities": [c.name for c in a.get_capabilities()],
                "metrics": {
                    "tasks_completed": a.metrics.tasks_completed,
                    "tasks_failed": a.metrics.tasks_failed,
                    "avg_response_time": round(a.metrics.avg_response_time, 3),
                    "uptime": round(a.metrics.uptime, 1),
                    "error_rate": round(a.metrics.error_rate, 4),
                },
            }
            for a in self._agents.values()
        ]

        task_counts: Dict[str, int] = defaultdict(int)
        for task in self._tasks.values():
            task_counts[task.status.value] += 1
        plan_counts: Dict[str, int] = defaultdict(int)
        for plan_id in self._plans.keys():
            status = (self.get_plan_status(plan_id) or {}).get("status", "unknown")
            plan_counts[str(status)] += 1

        uptime = (
            time.monotonic() - self._start_time if self._start_time else 0.0
        )

        return {
            "orchestrator": {
                "running": self._running,
                "uptime_seconds": round(uptime, 1),
                "max_concurrent_tasks": self._max_concurrent,
                "running_tasks": len(self._running_tasks),
                "registered_agents": len(self._agents),
                "workflow_checkpoints": len(self._workflow_checkpoints),
                "workflow_lane_caps": dict(self._workflow_lane_caps),
                "workflow_lane_priority": dict(self._workflow_lane_priority),
                "workflow_step_max_retries": int(self._workflow_step_max_retries),
                "workflow_step_result_contract_strict": bool(self._workflow_step_result_contract_strict),
                "task_payload_contract_strict": bool(self._task_payload_contract_strict),
                "max_pending_tasks": int(self._max_pending_tasks),
                "workflow_checkpoint_persist_path": self._workflow_checkpoint_persist_path,
                "workflow_checkpoint_backend": self._workflow_checkpoint_backend,
                "workflow_stats": dict(self._workflow_stats),
                "resource_pools": self._resource_pools.snapshot(),
                "tool_isolation": {
                    "enabled": bool(self._tool_isolation.enabled),
                    "allowed_roots_count": len(self._tool_isolation.allowed_roots),
                },
                "artifact_store": self._artifact_store.snapshot(),
            },
            "task_counts": dict(task_counts),
            "plan_counts": dict(plan_counts),
            "agents": agent_summaries,
        }

    @staticmethod
    def _load_workflow_lane_caps_from_env() -> Dict[str, int]:
        raw = str(os.getenv("JARVIS_AGENT_WORKFLOW_LANE_CAPS", "")).strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return {}
        except Exception:
            return {}
        caps: Dict[str, int] = {}
        for k, v in payload.items():
            key = str(k).strip()
            if not key:
                continue
            try:
                n = int(v)
            except Exception:
                continue
            if n > 0:
                caps[key] = n
        return caps

    @staticmethod
    def _load_workflow_lane_priority_from_env() -> Dict[str, int]:
        raw = str(os.getenv("JARVIS_AGENT_WORKFLOW_LANE_PRIORITY", "")).strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return {}
        except Exception:
            return {}
        pri: Dict[str, int] = {}
        for k, v in payload.items():
            lane = str(k).strip()
            if not lane:
                continue
            try:
                pri[lane] = int(v)
            except Exception:
                continue
        return pri

    def _split_wave_into_lane_batches(
        self,
        wave: List[WorkflowStep],
        *,
        flow_hints: Dict[str, Any],
    ) -> List[List[WorkflowStep]]:
        if not self._workflow_lane_caps:
            return [wave]
        lane_pressure: Dict[str, int] = defaultdict(int)
        for t in self._tasks.values():
            if t.status not in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.WAITING_DEPS}:
                continue
            meta = t.metadata if isinstance(t.metadata, dict) else {}
            lane = str(meta.get("flow_lane", "default_lane")).strip() or "default_lane"
            lane_pressure[lane] = int(lane_pressure.get(lane, 0)) + 1
        strategy = self._strategy_engine.select(
            lane_caps=dict(self._workflow_lane_caps),
            lane_pressure=dict(lane_pressure),
        )
        if strategy.strategy_id == "adaptive":
            self._workflow_stats["strategy_adaptive"] = int(self._workflow_stats.get("strategy_adaptive", 0)) + 1
        else:
            self._workflow_stats["strategy_baseline"] = int(self._workflow_stats.get("strategy_baseline", 0)) + 1
        caps = dict(strategy.lane_caps) if isinstance(strategy.lane_caps, dict) else dict(self._workflow_lane_caps)
        priorities = dict(self._workflow_lane_priority)
        if isinstance(strategy.lane_priority, dict):
            priorities.update({str(k): int(v) for k, v in strategy.lane_priority.items()})
        lane_groups: Dict[str, List[WorkflowStep]] = defaultdict(list)
        for step in wave:
            hint = flow_hints.get(step.name, {}) if isinstance(flow_hints, dict) else {}
            lane = str(hint.get("lane", "")).strip() if isinstance(hint, dict) else ""
            lane = lane or "default_lane"
            lane_groups[lane].append(step)
        max_batches = 0
        lane_batches: Dict[str, List[List[WorkflowStep]]] = {}
        for lane, steps in lane_groups.items():
            cap = int(caps.get(lane, 0) or 0)
            if cap <= 0:
                lane_batches[lane] = [steps]
            else:
                lane_batches[lane] = [steps[idx : idx + cap] for idx in range(0, len(steps), cap)]
            max_batches = max(max_batches, len(lane_batches[lane]))
        merged: List[List[WorkflowStep]] = []
        ordered_lanes = sorted(
            lane_batches.keys(),
            key=lambda lane: (
                int(priorities.get(lane, 100) or 100),
                lane,
            ),
        )
        if ordered_lanes:
            shift = self._workflow_lane_rr_offset % len(ordered_lanes)
            ordered_lanes = ordered_lanes[shift:] + ordered_lanes[:shift]
            self._workflow_lane_rr_offset = (self._workflow_lane_rr_offset + 1) % max(1, len(ordered_lanes))
        for batch_idx in range(max_batches):
            batch: List[WorkflowStep] = []
            for lane in ordered_lanes:
                chunks = lane_batches[lane]
                if batch_idx < len(chunks):
                    batch.extend(chunks[batch_idx])
            if batch:
                merged.append(batch)
        return merged or [wave]

    async def _wait_for_lane_capacity(
        self,
        *,
        step: WorkflowStep,
        flow_hints: Dict[str, Any],
    ) -> None:
        if not self._workflow_lane_caps:
            return
        hint = flow_hints.get(step.name, {}) if isinstance(flow_hints, dict) else {}
        lane = str(hint.get("lane", "")).strip() if isinstance(hint, dict) else ""
        lane = lane or "default_lane"
        cap = int(self._workflow_lane_caps.get(lane, 0) or 0)
        if cap <= 0:
            return
        deadline = time.monotonic() + 30.0
        while True:
            pending_for_lane = 0
            for t in self._tasks.values():
                if t.status not in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.WAITING_DEPS}:
                    continue
                meta = t.metadata if isinstance(t.metadata, dict) else {}
                if str(meta.get("flow_lane", "")).strip() == lane:
                    pending_for_lane += 1
            if pending_for_lane < cap:
                return
            self._workflow_stats["lane_backpressure_waits"] = int(
                self._workflow_stats.get("lane_backpressure_waits", 0)
            ) + 1
            if time.monotonic() >= deadline:
                self._workflow_stats["lane_admission_denied"] = int(
                    self._workflow_stats.get("lane_admission_denied", 0)
                ) + 1
                raise WorkflowError(
                    workflow_id=str(step.name),
                    reason=f"Lane backpressure timeout for lane '{lane}' (cap={cap})",
                )
            await asyncio.sleep(0.2)

    async def _wait_workflow_task_with_retry(
        self,
        *,
        workflow: WorkflowDefinition,
        step_name: str,
        task_id: str,
        flow_hints: Dict[str, Any],
        exec_context: Dict[str, Any],
        step_map: Dict[str, WorkflowStep],
    ) -> Task:
        current_task_id = str(task_id)
        step = step_map.get(step_name)
        attempts = 0
        max_retries = int(self._workflow_step_max_retries)
        while True:
            completed_task = await self.wait_for_task(
                current_task_id, timeout=self._default_timeout * 5
            )
            if completed_task.status != TaskStatus.FAILED:
                return completed_task
            attempts += 1
            if attempts > max_retries or step is None:
                return completed_task
            await self._wait_for_lane_capacity(step=step, flow_hints=flow_hints)
            retry_payload = {**exec_context, **step.payload}
            step_hint = flow_hints.get(step.name, {}) if isinstance(flow_hints, dict) else {}
            retry_meta = {
                "workflow_id": workflow.id,
                "workflow_name": workflow.name,
                "workflow_step": step.name,
                "flow_lane": str(step_hint.get("lane", "")) if isinstance(step_hint, dict) else "",
                "retry_of_task_id": current_task_id,
                "workflow_retry_attempt": attempts,
            }
            current_task_id = await self.submit_task(
                description=f"{workflow.name}/{step.name}#retry{attempts}",
                required_capabilities=[step.capability],
                payload=retry_payload,
                timeout=step.timeout,
                metadata=retry_meta,
            )

    @staticmethod
    def _resource_pool_for_step(step: WorkflowStep, flow_hints: Dict[str, Any]) -> str:
        hint = flow_hints.get(step.name, {}) if isinstance(flow_hints, dict) else {}
        lane = str(hint.get("lane", "")).strip().lower() if isinstance(hint, dict) else ""
        capability = str(step.capability or "").strip().lower()
        if "verifier" in lane or "verify" in capability:
            return "cpu"
        if any(k in capability for k in ("analysis", "rag", "research", "code", "build", "deploy")):
            return "gpu"
        return "cpu"

    def _store_workflow_checkpoint(
        self,
        *,
        workflow_id: str,
        workflow_name: str,
        next_wave_index: int,
        step_results: Dict[str, Any],
        execution_state: Optional[Dict[str, Any]] = None,
        status: str,
        failed_step: str = "",
    ) -> None:
        payload: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "status": status,
            "next_wave_index": int(next_wave_index),
            "step_results": dict(step_results),
            "updated_at": timestamp_now(),
        }
        if isinstance(execution_state, dict):
            payload["execution_state"] = execution_state
        if failed_step:
            payload["failed_step"] = failed_step
        self._workflow_checkpoints[workflow_id] = payload
        self._persist_workflow_checkpoints()

    def get_workflow_checkpoint(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        cp = self._workflow_checkpoints.get(str(workflow_id))
        if isinstance(cp, dict):
            return dict(cp)
        return None

    def _init_workflow_execution_state(
        self,
        *,
        workflow: WorkflowDefinition,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        cp_state = checkpoint.get("execution_state", {}) if isinstance(checkpoint, dict) else {}
        if isinstance(cp_state, dict) and cp_state.get("steps"):
            return dict(cp_state)
        step_defs = [
            {"name": s.name, "depends_on": list(s.depends_on), "capability": s.capability}
            for s in workflow.steps
        ]
        if self._langgraph_adapter:
            try:
                return self._langgraph_adapter.init_execution_state(
                    step_defs, max_retries=self._workflow_step_max_retries
                )
            except Exception:
                pass
        steps = {
            str(s.name): {
                "status": "pending",
                "depends_on": list(s.depends_on),
                "attempts": 0,
                "max_retries": int(self._workflow_step_max_retries),
            }
            for s in workflow.steps
        }
        return {"steps": steps, "ready": [], "terminal": False, "failed_steps": []}

    def _transition_workflow_execution_state(
        self,
        *,
        execution_state: Dict[str, Any],
        events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self._langgraph_adapter:
            try:
                return self._langgraph_adapter.transition_execution_state(execution_state, events)
            except Exception:
                pass
        cur = dict(execution_state if isinstance(execution_state, dict) else {})
        steps = cur.get("steps", {})
        if not isinstance(steps, dict):
            steps = {}
        for evt in events:
            if not isinstance(evt, dict):
                continue
            nm = str(evt.get("step", "")).strip()
            st = str(evt.get("status", "")).strip()
            if nm in steps and isinstance(steps[nm], dict):
                steps[nm]["status"] = st
        cur["steps"] = steps
        return cur

    def _validate_workflow_step_result(self, *, step_name: str, result: Any) -> str:
        if not self._workflow_step_result_contract_strict:
            return ""
        if result is None:
            return "result_is_none"
        if isinstance(result, str):
            low = result.strip().lower()
            forbidden = ("thinking process", "analyze the request", "<think>", "</think>")
            if any(f in low for f in forbidden):
                return "contains_meta_noise"
            if len(re.findall(r"\S+", result)) < 2:
                return "result_too_short"
        return ""

    @staticmethod
    def _validate_task_payload_contract(payload: Any) -> str:
        if payload is None:
            return ""
        if not isinstance(payload, dict):
            return "payload_not_object"
        forbidden = ("thinking process", "analyze the request", "<think>", "</think>")
        for k, v in payload.items():
            key = str(k).strip()
            if not key:
                return "payload_empty_key"
            if isinstance(v, str):
                low = v.strip().lower()
                if any(tok in low for tok in forbidden):
                    return f"payload_meta_noise:{key}"
        return ""

    def _load_workflow_checkpoints(self) -> None:
        if self._workflow_checkpoint_backend == "sqlite":
            self._load_workflow_checkpoints_sqlite()
            return
        file = Path(self._workflow_checkpoint_persist_path)
        if not file.exists():
            return
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                return
            loaded: Dict[str, Dict[str, Any]] = {}
            for item in payload:
                if not isinstance(item, dict):
                    continue
                workflow_id = str(item.get("workflow_id", "")).strip()
                if workflow_id:
                    loaded[workflow_id] = item
            self._workflow_checkpoints = loaded
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed loading workflow checkpoints from %s: %s", file, exc)

    def _persist_workflow_checkpoints(self) -> None:
        if self._workflow_checkpoint_backend == "sqlite":
            self._persist_workflow_checkpoints_sqlite()
            return
        if not self._workflow_checkpoint_persist_path:
            return
        file = Path(self._workflow_checkpoint_persist_path)
        try:
            file.parent.mkdir(parents=True, exist_ok=True)
            rows = list(self._workflow_checkpoints.values())
            file.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed persisting workflow checkpoints to %s: %s", file, exc)

    def _load_workflow_checkpoints_sqlite(self) -> None:
        db = Path(self._workflow_checkpoint_persist_path)
        if not db.exists():
            return
        try:
            conn = sqlite3.connect(str(db))
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS workflow_checkpoints (workflow_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            cur.execute("SELECT workflow_id, payload FROM workflow_checkpoints")
            rows = cur.fetchall()
            loaded: Dict[str, Dict[str, Any]] = {}
            for workflow_id, payload in rows:
                try:
                    obj = json.loads(str(payload))
                    if isinstance(obj, dict):
                        loaded[str(workflow_id)] = obj
                except Exception:
                    continue
            self._workflow_checkpoints = loaded
            conn.close()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed loading workflow checkpoints sqlite %s: %s", db, exc)

    def _persist_workflow_checkpoints_sqlite(self) -> None:
        db = Path(self._workflow_checkpoint_persist_path)
        try:
            db.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db))
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS workflow_checkpoints (workflow_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            for workflow_id, payload in self._workflow_checkpoints.items():
                cur.execute(
                    "INSERT INTO workflow_checkpoints(workflow_id,payload) VALUES(?,?) "
                    "ON CONFLICT(workflow_id) DO UPDATE SET payload=excluded.payload",
                    (str(workflow_id), json.dumps(payload)),
                )
            conn.commit()
            conn.close()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed persisting workflow checkpoints sqlite %s: %s", db, exc)


__all__ = [
    "TaskStatus",
    "Task",
    "PlanStep",
    "TaskPlanRecord",
    "WorkflowStep",
    "WorkflowDefinition",
    "MasterOrchestrator",
]
