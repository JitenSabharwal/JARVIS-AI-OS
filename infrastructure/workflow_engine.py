"""
Workflow orchestration engine for JARVIS AI OS.

Supports multi-step workflows with conditional branching, context passing,
pause/resume/cancel, and full execution history.
"""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

from infrastructure.logger import get_logger

logger = get_logger("workflow_engine")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"


class WorkflowStatus(Enum):
    DEFINED = "defined"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WorkflowStep:
    """
    A single step within a workflow definition.

    *skill_or_agent* is a string key looked up in the executor registry at
    runtime. ``on_success`` / ``on_failure`` are step IDs that form the
    conditional DAG edges.
    """
    id: str
    name: str
    skill_or_agent: str                        # registry key
    params: dict[str, Any] = field(default_factory=dict)
    on_success: str | None = None              # next step id on success
    on_failure: str | None = None              # next step id on failure
    timeout: float | None = None               # seconds; None = no limit
    retry_count: int = 0                       # number of automatic retries
    input_mapping: dict[str, str] = field(default_factory=dict)
    # Maps param name -> "context.key" so previous outputs can be forwarded


@dataclass
class WorkflowDefinition:
    """
    Immutable blueprint for a workflow.

    ``steps`` is a dict of step_id -> WorkflowStep.
    ``entry_step`` names the first step to execute.
    """
    id: str
    name: str
    description: str = ""
    steps: dict[str, WorkflowStep] = field(default_factory=dict)
    entry_step: str | None = None              # first step; defaults to first added
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class StepRecord:
    """Execution record for a single step run."""
    step_id: str
    step_name: str
    status: StepStatus = StepStatus.PENDING
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: str | None = None
    attempt: int = 1

    @property
    def duration(self) -> float | None:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None


@dataclass
class WorkflowInstance:
    """
    A running (or finished) execution of a WorkflowDefinition.

    ``context`` is the shared data dictionary passed between steps.
    """
    id: str
    definition_id: str
    definition_name: str
    status: WorkflowStatus = WorkflowStatus.RUNNING
    context: dict[str, Any] = field(default_factory=dict)
    step_records: dict[str, StepRecord] = field(default_factory=dict)
    current_step_id: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    # Pause mechanism: set by pause(), cleared by resume()
    _pause_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    # Cancel flag
    _cancelled: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        # The event starts in "set" state so the workflow runs freely.
        # pause() clears it; resume() sets it again.
        self._pause_event.set()

    @property
    def duration(self) -> float | None:
        if self.finished_at:
            return self.finished_at - self.started_at
        return None


# ---------------------------------------------------------------------------
# Executor registry type
# ---------------------------------------------------------------------------

StepExecutor = Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[Any]]
# (skill_or_agent_key, params, context) -> result


# ---------------------------------------------------------------------------
# WorkflowEngine
# ---------------------------------------------------------------------------

class WorkflowEngine:
    """
    Orchestrates multi-step workflows.

    Register workflows with ``define_workflow()``, then execute them with
    ``execute_workflow()``.  Register step executors (skill/agent dispatchers)
    via ``register_executor()``.
    """

    _MAX_HISTORY = 200      # max completed instances kept in history

    def __init__(self) -> None:
        self._definitions: dict[str, WorkflowDefinition] = {}
        self._instances: dict[str, WorkflowInstance] = {}       # active
        self._history: list[WorkflowInstance] = []              # completed
        self._executors: dict[str, StepExecutor] = {}           # key -> fn
        self._default_executor: StepExecutor | None = None
        self._lock = asyncio.Lock()
        logger.debug("WorkflowEngine initialised")

    # ------------------------------------------------------------------
    # Executor registration
    # ------------------------------------------------------------------

    def register_executor(self, key: str, executor: StepExecutor) -> None:
        """
        Map *key* (the ``skill_or_agent`` value in a step) to an async callable.

        The callable receives ``(key, params, context)`` and should return
        the step result.
        """
        self._executors[key] = executor
        logger.debug("Executor registered for key '%s'", key)

    def set_default_executor(self, executor: StepExecutor) -> None:
        """Fallback executor used when no exact key match is found."""
        self._default_executor = executor

    # ------------------------------------------------------------------
    # Workflow definition
    # ------------------------------------------------------------------

    def define_workflow(self, definition: WorkflowDefinition) -> None:
        """Register (or overwrite) a workflow definition."""
        if not definition.steps:
            raise ValueError(f"Workflow '{definition.name}' has no steps")
        if definition.entry_step and definition.entry_step not in definition.steps:
            raise ValueError(
                f"entry_step '{definition.entry_step}' not in steps for '{definition.name}'"
            )
        # Default entry_step to first step added
        if not definition.entry_step:
            definition.entry_step = next(iter(definition.steps))
        self._definitions[definition.id] = definition
        logger.info("Workflow defined: '%s' (%s)", definition.name, definition.id)

    def define_workflow_dict(
        self,
        name: str,
        steps: list[dict[str, Any]],
        *,
        description: str = "",
        entry_step: str | None = None,
        metadata: dict[str, Any] | None = None,
        workflow_id: str | None = None,
    ) -> WorkflowDefinition:
        """
        Convenience method — build and register a WorkflowDefinition from a
        plain list-of-dicts representation.

        Each dict must have at least ``id``, ``name``, and ``skill_or_agent``.
        """
        step_map: dict[str, WorkflowStep] = {}
        for s in steps:
            ws = WorkflowStep(
                id=s["id"],
                name=s.get("name", s["id"]),
                skill_or_agent=s["skill_or_agent"],
                params=s.get("params", {}),
                on_success=s.get("on_success"),
                on_failure=s.get("on_failure"),
                timeout=s.get("timeout"),
                retry_count=s.get("retry_count", 0),
                input_mapping=s.get("input_mapping", {}),
            )
            step_map[ws.id] = ws

        defn = WorkflowDefinition(
            id=workflow_id or str(uuid.uuid4()),
            name=name,
            description=description,
            steps=step_map,
            entry_step=entry_step,
            metadata=metadata or {},
        )
        self.define_workflow(defn)
        return defn

    def get_workflow_definition(self, definition_id: str) -> WorkflowDefinition | None:
        """Return the named workflow definition, or None."""
        return self._definitions.get(definition_id)

    def list_definitions(self) -> list[dict[str, Any]]:
        """Return summary info for all registered definitions."""
        return [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "step_count": len(d.steps),
                "entry_step": d.entry_step,
            }
            for d in self._definitions.values()
        ]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_workflow(
        self,
        definition_id: str,
        initial_context: dict[str, Any] | None = None,
        *,
        instance_id: str | None = None,
        timeout: float | None = None,
    ) -> WorkflowInstance:
        """
        Run the workflow identified by *definition_id*.

        ``initial_context`` seeds the shared data dict available to all steps.
        Returns the completed (or failed) WorkflowInstance.

        Raises ``KeyError`` if the definition does not exist.
        Raises ``asyncio.TimeoutError`` if the overall *timeout* elapses.
        """
        defn = self._definitions.get(definition_id)
        if defn is None:
            raise KeyError(f"No workflow definition: {definition_id}")

        instance = WorkflowInstance(
            id=instance_id or str(uuid.uuid4()),
            definition_id=defn.id,
            definition_name=defn.name,
            context=copy.deepcopy(initial_context or {}),
        )

        async with self._lock:
            self._instances[instance.id] = instance

        logger.info(
            "Workflow '%s' started (instance=%s)", defn.name, instance.id
        )

        try:
            coro = self._run_instance(instance, defn)
            if timeout:
                await asyncio.wait_for(coro, timeout=timeout)
            else:
                await coro
        except asyncio.TimeoutError:
            instance.status = WorkflowStatus.FAILED
            instance.error = "Workflow timed out"
            instance.finished_at = time.time()
            logger.error("Workflow instance %s timed out", instance.id)
        except Exception as exc:  # noqa: BLE001
            instance.status = WorkflowStatus.FAILED
            instance.error = str(exc)
            instance.finished_at = time.time()
            logger.error("Workflow instance %s failed: %s", instance.id, exc)
        finally:
            async with self._lock:
                self._instances.pop(instance.id, None)
                self._history.append(instance)
                if len(self._history) > self._MAX_HISTORY:
                    self._history = self._history[-self._MAX_HISTORY :]

        logger.info(
            "Workflow '%s' finished with status=%s (instance=%s)",
            defn.name, instance.status.value, instance.id,
        )
        return instance

    async def _run_instance(
        self, instance: WorkflowInstance, defn: WorkflowDefinition
    ) -> None:
        """Drive the step-by-step execution of *instance*."""
        current_id: str | None = defn.entry_step

        while current_id is not None:
            # Cancel check
            if instance._cancelled:
                instance.status = WorkflowStatus.CANCELLED
                instance.finished_at = time.time()
                return

            # Pause check — blocks until resume() is called
            await instance._pause_event.wait()

            step = defn.steps.get(current_id)
            if step is None:
                instance.status = WorkflowStatus.FAILED
                instance.error = f"Unknown step id: {current_id}"
                instance.finished_at = time.time()
                return

            instance.current_step_id = current_id
            record = StepRecord(step_id=step.id, step_name=step.name)
            instance.step_records[step.id] = record

            success = await self._execute_step(step, record, instance.context)
            current_id = step.on_success if success else step.on_failure

        # If we reach here every step completed (or was routed to None)
        if instance.status == WorkflowStatus.RUNNING:
            instance.status = WorkflowStatus.COMPLETED
            instance.finished_at = time.time()

    async def _execute_step(
        self,
        step: WorkflowStep,
        record: StepRecord,
        context: dict[str, Any],
    ) -> bool:
        """
        Execute a single step, honoring retry_count and timeout.

        Updates *record* in-place. Returns True on success, False on failure.
        """
        params = self._resolve_params(step, context)
        executor = self._executors.get(step.skill_or_agent) or self._default_executor
        if executor is None:
            record.status = StepStatus.FAILED
            record.error = f"No executor registered for '{step.skill_or_agent}'"
            record.finished_at = time.time()
            logger.error("Step '%s': %s", step.name, record.error)
            return False

        max_attempts = step.retry_count + 1
        for attempt in range(1, max_attempts + 1):
            record.attempt = attempt
            record.started_at = time.time()
            record.status = StepStatus.RUNNING

            try:
                coro = executor(step.skill_or_agent, params, context)
                if step.timeout:
                    result = await asyncio.wait_for(coro, timeout=step.timeout)
                else:
                    result = await coro

                record.result = result
                record.status = StepStatus.COMPLETED
                record.finished_at = time.time()

                # Publish step output into the shared context
                context[f"step.{step.id}.result"] = result
                context["last_result"] = result

                logger.debug(
                    "Step '%s' completed (attempt %d/%d)", step.name, attempt, max_attempts
                )
                return True

            except asyncio.TimeoutError:
                record.error = f"Step timed out after {step.timeout}s"
                record.status = StepStatus.TIMED_OUT
                record.finished_at = time.time()
                logger.warning("Step '%s' timed out (attempt %d/%d)", step.name, attempt, max_attempts)

            except Exception as exc:  # noqa: BLE001
                record.error = str(exc)
                record.status = StepStatus.FAILED
                record.finished_at = time.time()
                logger.warning(
                    "Step '%s' failed (attempt %d/%d): %s", step.name, attempt, max_attempts, exc
                )

            if attempt < max_attempts:
                # Brief back-off between retries
                await asyncio.sleep(min(2.0 ** (attempt - 1), 10.0))

        return False

    @staticmethod
    def _resolve_params(step: WorkflowStep, context: dict[str, Any]) -> dict[str, Any]:
        """
        Merge static ``step.params`` with values extracted from *context*
        via ``step.input_mapping``.

        Mapping format: ``{"param_name": "context.key"}`` where ``context.key``
        is looked up in the flat context dict.
        """
        params = copy.deepcopy(step.params)
        for param_name, context_key in step.input_mapping.items():
            if context_key in context:
                params[param_name] = context[context_key]
        return params

    # ------------------------------------------------------------------
    # Pause / Resume / Cancel
    # ------------------------------------------------------------------

    async def pause(self, instance_id: str) -> bool:
        """Pause a running workflow. Returns False if not found or not running."""
        instance = self._instances.get(instance_id)
        if not instance or instance.status != WorkflowStatus.RUNNING:
            return False
        instance.status = WorkflowStatus.PAUSED
        instance._pause_event.clear()
        logger.info("Workflow instance %s paused", instance_id)
        return True

    async def resume(self, instance_id: str) -> bool:
        """Resume a paused workflow."""
        instance = self._instances.get(instance_id)
        if not instance or instance.status != WorkflowStatus.PAUSED:
            return False
        instance.status = WorkflowStatus.RUNNING
        instance._pause_event.set()
        logger.info("Workflow instance %s resumed", instance_id)
        return True

    async def cancel(self, instance_id: str) -> bool:
        """Cancel a running or paused workflow."""
        instance = self._instances.get(instance_id)
        if not instance:
            return False
        if instance.status not in (WorkflowStatus.RUNNING, WorkflowStatus.PAUSED):
            return False
        instance._cancelled = True
        instance._pause_event.set()   # unblock if paused so the loop can see the cancel flag
        logger.info("Workflow instance %s cancellation requested", instance_id)
        return True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_instance(self, instance_id: str) -> WorkflowInstance | None:
        """Return an active instance, or search history."""
        if instance_id in self._instances:
            return self._instances[instance_id]
        for inst in reversed(self._history):
            if inst.id == instance_id:
                return inst
        return None

    def list_active_instances(self) -> list[dict[str, Any]]:
        """Summary of all currently running/paused workflow instances."""
        return [
            {
                "id": i.id,
                "definition_name": i.definition_name,
                "status": i.status.value,
                "current_step": i.current_step_id,
                "started_at": i.started_at,
            }
            for i in self._instances.values()
        ]

    def get_execution_history(
        self,
        definition_id: str | None = None,
        limit: int = 50,
    ) -> list[WorkflowInstance]:
        """Return recent execution history, optionally filtered by definition."""
        items = list(reversed(self._history))
        if definition_id:
            items = [i for i in items if i.definition_id == definition_id]
        return items[:limit]

    def get_engine_stats(self) -> dict[str, Any]:
        """Return summary statistics for the engine."""
        total = len(self._history)
        completed = sum(1 for i in self._history if i.status == WorkflowStatus.COMPLETED)
        failed = sum(1 for i in self._history if i.status == WorkflowStatus.FAILED)
        return {
            "definitions": len(self._definitions),
            "active_instances": len(self._instances),
            "history_size": total,
            "completed": completed,
            "failed": failed,
            "success_rate": round(completed / total, 3) if total else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"<WorkflowEngine definitions={len(self._definitions)} "
            f"active={len(self._instances)}>"
        )
