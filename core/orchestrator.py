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
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from core.agent_framework import AgentState, BaseAgent
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
    ) -> None:
        self._agents: Dict[str, BaseAgent] = {}
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Set[str] = set()
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._running: bool = False
        self._max_concurrent = max_concurrent_tasks
        self._default_timeout = default_task_timeout
        self._poll_interval = worker_poll_interval
        self._start_time: Optional[float] = None
        self._task_callbacks: Dict[str, List[Callable[[Task], None]]] = defaultdict(
            list
        )
        self._lock = asyncio.Lock()
        self._logger = get_logger(__name__)

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
        )

        if task.dependencies:
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

    def _select_best_agent(self, candidates: List[BaseAgent]) -> Optional[BaseAgent]:
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

        return min(
            pool,
            key=lambda a: (
                a.metrics.error_rate,
                a.metrics.avg_response_time,
                a.metrics.tasks_completed,
            ),
        )

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
        agent = self._select_best_agent(candidates)

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

        try:
            result = await agent.execute_task(
                required_caps[0],
                task.payload,
                timeout=effective_timeout,
                task_id=task.id,
            )
            task.result = result
            task.status = TaskStatus.COMPLETED
            self._logger.info(
                "Task %s completed by agent '%s'", task.id, agent.name
            )
            return result

        except (TaskTimeoutError, AgentCapabilityError) as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            raise

        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
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
            asyncio.create_task(
                self._execute_task(task),
                name=f"task-{task.id}",
            )

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
            "agents": agent_summaries,
        }


__all__ = [
    "TaskStatus",
    "Task",
    "WorkflowStep",
    "WorkflowDefinition",
    "MasterOrchestrator",
]
