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
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from core.agent_framework import AgentState, BaseAgent
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
        self._plan_persist_path = plan_persist_path
        self._auto_persist_plans = auto_persist_plans
        self._lock = asyncio.Lock()
        self._logger = get_logger(__name__)
        if self._plan_persist_path:
            self._load_plans()

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

        if task.requires_human and not task.approval_token:
            task.status = TaskStatus.WAITING_APPROVAL
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
            task.status = TaskStatus.WAITING_DEPS if task.dependencies else TaskStatus.PENDING
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
            if task.requires_human and not task.approval_token:
                task.status = TaskStatus.WAITING_APPROVAL
            elif task.dependencies:
                task.status = TaskStatus.WAITING_DEPS
            else:
                task.status = TaskStatus.PENDING
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

        # Group steps into execution waves (all deps satisfied → can run together)
        waves = self._build_execution_waves(workflow, execution_order)

        for wave_idx, wave in enumerate(waves):
            self._logger.debug(
                "Workflow '%s' executing wave %d: %s",
                workflow.id,
                wave_idx,
                [s.name for s in wave],
            )
            wave_task_ids = []
            for step in wave:
                merged_payload = {**exec_context, **step.payload}
                task_id = await self.submit_task(
                    description=f"{workflow.name}/{step.name}",
                    required_capabilities=[step.capability],
                    payload=merged_payload,
                    timeout=step.timeout,
                )
                wave_task_ids.append((step.name, task_id))

            # Wait for all tasks in this wave
            for step_name, task_id in wave_task_ids:
                completed_task = await self.wait_for_task(
                    task_id, timeout=self._default_timeout * 5
                )
                if completed_task.status == TaskStatus.FAILED:
                    raise WorkflowError(
                        workflow_id=workflow.id,
                        reason=f"Step '{step_name}' failed: {completed_task.error}",
                    )
                step_results[step_name] = completed_task.result
                exec_context[step_name] = completed_task.result

        self._logger.info(
            "Workflow '%s' completed successfully", workflow.name
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
        if task.requires_human and not task.approval_token:
            task.status = TaskStatus.WAITING_APPROVAL
            task.error = "Task requires human approval before execution"
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
            except Exception as exc:
                self._logger.error("Worker loop error: %s", exc)
        self._logger.debug("Orchestrator worker loop stopped")

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
            },
            "task_counts": dict(task_counts),
            "plan_counts": dict(plan_counts),
            "agents": agent_summaries,
        }


__all__ = [
    "TaskStatus",
    "Task",
    "PlanStep",
    "TaskPlanRecord",
    "WorkflowStep",
    "WorkflowDefinition",
    "MasterOrchestrator",
]
