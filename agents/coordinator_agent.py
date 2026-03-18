"""
Workflow coordinator agent for JARVIS AI OS.

:class:`CoordinatorAgent` extends :class:`~agents.base_agent.ConcreteAgent`
with the ability to:
- Coordinate multi-step workflows across other agents
- Assign individual tasks via the :class:`~agents.agent_registry.AgentRegistry`
- Monitor workflow and task progress in real time
- Report aggregate agent status
- Perform load-balanced task assignment
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.agent_registry import AgentRegistry
from agents.base_agent import ConcreteAgent
from core.agent_framework import (
    AgentCapability,
    AgentMessage,
    AgentState,
    BaseAgent,
    MessageType,
)
from infrastructure.logger import get_logger
from utils.exceptions import AgentNotFoundError, OrchestratorError, WorkflowError
from utils.helpers import generate_id, timestamp_now


# ---------------------------------------------------------------------------
# Supporting data structures
# ---------------------------------------------------------------------------


@dataclass
class WorkflowState:
    """Runtime state tracked by the coordinator for a running workflow.

    Args:
        workflow_id: Unique identifier for this workflow execution.
        name: Human-readable workflow name.
        steps: Ordered list of step descriptors (dicts with ``name``,
            ``capability``, ``payload``, ``depends_on`` keys).
        context: Shared execution context; accumulates step results.
        step_statuses: Maps step name → status string.
        step_results: Maps step name → step result.
        started_at: Epoch timestamp when execution began.
        completed_at: Epoch timestamp when execution finished.
        error: Error message if the workflow failed.
    """

    workflow_id: str
    name: str
    steps: List[Dict[str, Any]]
    context: Dict[str, Any] = field(default_factory=dict)
    step_statuses: Dict[str, str] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)
    completed_at: Optional[float] = None
    error: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return all(s == "completed" for s in self.step_statuses.values())

    @property
    def has_failed(self) -> bool:
        return any(s == "failed" for s in self.step_statuses.values())

    @property
    def elapsed_seconds(self) -> float:
        end = self.completed_at or time.monotonic()
        return end - self.started_at


# ---------------------------------------------------------------------------
# CoordinatorAgent
# ---------------------------------------------------------------------------


class CoordinatorAgent(ConcreteAgent):
    """Agent responsible for coordinating multi-agent workflows.

    Capabilities:
    - ``coordinate_workflow`` – execute a named workflow across multiple agents
    - ``assign_task`` – delegate a single capability task to a suitable agent
    - ``monitor_progress`` – query progress of a running workflow
    - ``get_agent_status`` – retrieve status of all or a specific agent

    Args:
        name: Agent name (default ``"coordinator"``).
        agent_id: Optional pre-assigned ID.
        registry: Optional :class:`~agents.agent_registry.AgentRegistry`
            instance; defaults to the singleton.
    """

    _COORDINATOR_CAPABILITIES: List[AgentCapability] = [
        AgentCapability(
            name="coordinate_workflow",
            description=(
                "Execute a multi-step workflow by distributing steps across "
                "available agents according to required capabilities."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "workflow_name": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "capability": {"type": "string"},
                                "payload": {"type": "object"},
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["name", "capability"],
                        },
                    },
                    "context": {"type": "object"},
                },
                "required": ["workflow_name", "steps"],
            },
        ),
        AgentCapability(
            name="assign_task",
            description=(
                "Assign a single capability task to the most suitable available agent."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "capability": {"type": "string"},
                    "payload": {"type": "object"},
                    "preferred_agent_id": {"type": "string"},
                },
                "required": ["capability"],
            },
        ),
        AgentCapability(
            name="monitor_progress",
            description="Return progress information for a running or completed workflow.",
            parameters_schema={
                "type": "object",
                "properties": {"workflow_id": {"type": "string"}},
                "required": ["workflow_id"],
            },
        ),
        AgentCapability(
            name="get_agent_status",
            description=(
                "Return status of all registered agents, or a single agent "
                "when 'agent_id' is provided."
            ),
            parameters_schema={
                "type": "object",
                "properties": {"agent_id": {"type": "string"}},
            },
        ),
    ]

    def __init__(
        self,
        name: str = "coordinator",
        agent_id: Optional[str] = None,
        registry: Optional[AgentRegistry] = None,
    ) -> None:
        super().__init__(name=name, agent_id=agent_id)
        self._registry: AgentRegistry = registry or AgentRegistry.get_instance()
        self._workflows: Dict[str, WorkflowState] = {}
        self._logger = get_logger(f"agent.{name}")

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def get_capabilities(self) -> List[AgentCapability]:
        """Return coordinator-specific plus built-in capabilities."""
        return super().get_capabilities() + list(self._COORDINATOR_CAPABILITIES)

    # ------------------------------------------------------------------
    # Capability handlers
    # ------------------------------------------------------------------

    async def handle_coordinate_workflow(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a multi-step workflow, distributing steps across agents.

        Steps are grouped into parallel execution waves based on their
        ``depends_on`` declarations.

        Args:
            parameters: Must contain ``workflow_name`` (str) and ``steps``
                (list).  Optional ``context`` (dict) seeds the execution context.

        Returns:
            Dict with ``workflow_id``, ``results`` (step→result), and ``elapsed_s``.

        Raises:
            WorkflowError: On workflow validation or execution failure.
        """
        workflow_name = parameters["workflow_name"]
        steps: List[Dict[str, Any]] = parameters["steps"]
        seed_context: Dict[str, Any] = parameters.get("context", {})

        if not steps:
            raise WorkflowError(reason="Workflow must have at least one step")

        workflow_id = generate_id("coord-wf")
        wf_state = WorkflowState(
            workflow_id=workflow_id,
            name=workflow_name,
            steps=steps,
            context=dict(seed_context),
        )
        for step in steps:
            wf_state.step_statuses[step["name"]] = "pending"

        self._workflows[workflow_id] = wf_state
        self._logger.info(
            "Coordinator '%s' starting workflow '%s' (%s) with %d steps",
            self.name,
            workflow_name,
            workflow_id,
            len(steps),
        )

        try:
            await self._execute_workflow(wf_state)
        except Exception as exc:
            wf_state.error = str(exc)
            wf_state.completed_at = time.monotonic()
            raise WorkflowError(workflow_id=workflow_id, reason=str(exc)) from exc

        wf_state.completed_at = time.monotonic()
        return {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "results": wf_state.step_results,
            "elapsed_s": round(wf_state.elapsed_seconds, 3),
        }

    async def handle_assign_task(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delegate a task to the best available agent for the requested capability.

        Args:
            parameters: Must contain ``capability`` (str).  Optional
                ``payload`` (dict) and ``preferred_agent_id`` (str).

        Returns:
            Dict with ``agent_id``, ``agent_name``, and ``result``.

        Raises:
            OrchestratorError: If no suitable agent is available.
        """
        capability: str = parameters["capability"]
        payload: Dict[str, Any] = parameters.get("payload", {})
        preferred_id: Optional[str] = parameters.get("preferred_agent_id")

        agent = self._select_agent(capability, preferred_id)
        if agent is None:
            raise OrchestratorError(
                f"No available agent for capability '{capability}'"
            )

        self._logger.info(
            "Coordinator assigning capability='%s' to agent '%s'",
            capability,
            agent.name,
        )
        result = await agent.execute_task(capability, payload)
        return {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "capability": capability,
            "result": result,
        }

    async def handle_monitor_progress(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return progress information for a workflow.

        Args:
            parameters: Must contain ``workflow_id`` (str).

        Returns:
            Dict summarising workflow progress.

        Raises:
            OrchestratorError: If *workflow_id* is not found.
        """
        workflow_id: str = parameters["workflow_id"]
        wf = self._workflows.get(workflow_id)
        if wf is None:
            raise OrchestratorError(f"Unknown workflow_id: '{workflow_id}'")

        total = len(wf.steps)
        done = sum(1 for s in wf.step_statuses.values() if s == "completed")
        failed = sum(1 for s in wf.step_statuses.values() if s == "failed")

        return {
            "workflow_id": workflow_id,
            "workflow_name": wf.name,
            "total_steps": total,
            "completed_steps": done,
            "failed_steps": failed,
            "progress_pct": round(done / total * 100, 1) if total else 0.0,
            "step_statuses": dict(wf.step_statuses),
            "is_complete": wf.is_complete,
            "has_failed": wf.has_failed,
            "elapsed_s": round(wf.elapsed_seconds, 3),
            "error": wf.error,
        }

    async def handle_get_agent_status(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return status of one or all registered agents.

        Args:
            parameters: Optional ``agent_id`` (str).  If omitted, all agents
                are returned.

        Returns:
            Dict with ``agents`` list of status summaries.
        """
        agent_id: Optional[str] = parameters.get("agent_id")

        if agent_id:
            try:
                agent = self._registry.get_agent(agent_id)
                agents = [agent]
            except AgentNotFoundError as exc:
                raise OrchestratorError(str(exc)) from exc
        else:
            agents = self._registry.get_all_agents()

        summaries = [self._summarise_agent(a) for a in agents]
        return {
            "agents": summaries,
            "total": len(summaries),
            "idle": sum(1 for a in agents if a.state == AgentState.IDLE),
            "busy": sum(1 for a in agents if a.state == AgentState.BUSY),
        }

    # ------------------------------------------------------------------
    # Internal workflow execution
    # ------------------------------------------------------------------

    async def _execute_workflow(self, wf: WorkflowState) -> None:
        """Execute workflow steps respecting dependency order.

        Builds execution waves (steps with all deps satisfied run together),
        then executes each wave concurrently.
        """
        step_map = {s["name"]: s for s in wf.steps}
        completed: set[str] = set()
        remaining = list(step_map.keys())

        while remaining:
            # Collect steps whose dependencies are all complete
            wave = [
                step_map[name]
                for name in remaining
                if all(dep in completed for dep in step_map[name].get("depends_on", []))
            ]
            if not wave:
                raise WorkflowError(
                    workflow_id=wf.workflow_id,
                    reason="Dependency cycle detected or unsatisfied dependency",
                )

            # Execute the wave concurrently
            await self._execute_wave(wf, wave)

            # Check for failures before continuing
            if wf.has_failed:
                failed_steps = [
                    name
                    for name, status in wf.step_statuses.items()
                    if status == "failed"
                ]
                raise WorkflowError(
                    workflow_id=wf.workflow_id,
                    reason=f"Steps failed: {failed_steps}",
                )

            completed.update(s["name"] for s in wave)
            remaining = [n for n in remaining if n not in completed]

    async def _execute_wave(
        self, wf: WorkflowState, steps: List[Dict[str, Any]]
    ) -> None:
        """Run all *steps* concurrently and update *wf* state."""

        async def _run_step(step: Dict[str, Any]) -> None:
            step_name: str = step["name"]
            capability: str = step["capability"]
            payload: Dict[str, Any] = {**wf.context, **step.get("payload", {})}
            timeout: Optional[float] = step.get("timeout")

            wf.step_statuses[step_name] = "running"
            self._logger.debug(
                "Coordinator executing step '%s' (capability='%s')",
                step_name,
                capability,
            )

            agent = self._select_agent(capability, None)
            if agent is None:
                wf.step_statuses[step_name] = "failed"
                raise OrchestratorError(
                    f"No agent available for capability '{capability}'"
                )

            try:
                result = await agent.execute_task(
                    capability,
                    payload,
                    timeout=timeout,
                    task_id=generate_id(f"coord-{step_name}"),
                )
                wf.step_results[step_name] = result
                wf.context[step_name] = result
                wf.step_statuses[step_name] = "completed"
                self._logger.debug(
                    "Step '%s' completed by agent '%s'", step_name, agent.name
                )
            except Exception as exc:
                wf.step_statuses[step_name] = "failed"
                self._logger.error("Step '%s' failed: %s", step_name, exc)
                raise

        await asyncio.gather(*[_run_step(s) for s in steps], return_exceptions=False)

    # ------------------------------------------------------------------
    # Load-balanced agent selection
    # ------------------------------------------------------------------

    def _select_agent(
        self,
        capability: str,
        preferred_id: Optional[str],
    ) -> Optional[BaseAgent]:
        """Return the best available agent for *capability*.

        If *preferred_id* is set and that agent is available, it takes
        priority.  Otherwise the idle agent with the lowest error rate is
        chosen; a BUSY agent is selected as a last resort.

        Args:
            capability: Required capability name.
            preferred_id: Optional ID of the preferred agent.

        Returns:
            The selected :class:`~core.agent_framework.BaseAgent`, or ``None``.
        """
        # Try the preferred agent first
        if preferred_id:
            try:
                preferred = self._registry.get_agent(preferred_id)
                if preferred.has_capability(capability) and preferred.is_available:
                    return preferred
            except AgentNotFoundError:
                pass

        candidates = self._registry.discover(capability)
        if not candidates:
            return None

        # Prefer idle over busy
        idle = [a for a in candidates if a.state == AgentState.IDLE]
        pool = idle if idle else [
            a for a in candidates if a.state == AgentState.BUSY
        ]
        if not pool:
            return None

        return min(pool, key=lambda a: (a.metrics.error_rate, a.metrics.avg_response_time))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_agent(agent: BaseAgent) -> Dict[str, Any]:
        """Build a compact status summary for *agent*."""
        m = agent.metrics
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "state": agent.state.name,
            "capabilities": [c.name for c in agent.get_capabilities()],
            "tasks_completed": m.tasks_completed,
            "tasks_failed": m.tasks_failed,
            "avg_response_time_s": round(m.avg_response_time, 4),
            "error_rate": round(m.error_rate, 4),
            "uptime_s": round(m.uptime, 1),
        }


__all__ = ["CoordinatorAgent", "WorkflowState"]
