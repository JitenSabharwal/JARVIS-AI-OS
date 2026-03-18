"""
Agent registry: discovery and lifecycle management for JARVIS AI OS agents.

The :class:`AgentRegistry` is a singleton that acts as the authoritative
source of truth for all registered agents.  It supports:
- Registration / deregistration
- Capability-based discovery
- ID and name-based lookup
- Bulk health checking
- Per-agent workload statistics
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional

from core.agent_framework import AgentCapability, AgentState, BaseAgent
from infrastructure.logger import get_logger
from utils.exceptions import AgentError, AgentNotFoundError
from utils.helpers import timestamp_now

logger = get_logger(__name__)


class AgentRegistry:
    """Singleton registry for :class:`~core.agent_framework.BaseAgent` instances.

    Thread-safe for reads; writes are protected by an :class:`asyncio.Lock`.

    Usage::

        registry = AgentRegistry.get_instance()
        await registry.register(my_agent)
        agents = registry.discover("code_review")

    Agents are stored by their :attr:`~core.agent_framework.BaseAgent.agent_id`.
    A secondary index on name allows lookup by human-readable identifier.
    """

    _instance: Optional["AgentRegistry"] = None
    _class_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}
        self._name_index: Dict[str, str] = {}  # name → agent_id
        self._registration_times: Dict[str, float] = {}  # agent_id → epoch
        self._lock: asyncio.Lock = asyncio.Lock()
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "AgentRegistry":
        """Return the process-wide singleton instance.

        The instance is created lazily on first access.

        Returns:
            The singleton :class:`AgentRegistry`.
        """
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.debug("AgentRegistry singleton created")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Destroy the singleton (primarily useful in tests).

        Warning:
            All registered agents and state are lost.
        """
        with cls._class_lock:
            cls._instance = None
            logger.debug("AgentRegistry singleton reset")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register(self, agent: BaseAgent) -> None:
        """Register *agent* in the registry.

        If an agent with the same ID is already registered it is replaced.

        Args:
            agent: The :class:`~core.agent_framework.BaseAgent` to register.
        """
        async with self._lock:
            if agent.agent_id in self._agents:
                self._logger.warning(
                    "Re-registering agent '%s' (%s); replacing previous entry",
                    agent.name,
                    agent.agent_id,
                )
            self._agents[agent.agent_id] = agent
            self._name_index[agent.name] = agent.agent_id
            self._registration_times[agent.agent_id] = time.monotonic()

        self._logger.info(
            "Registered agent '%s' (%s) — capabilities: %s",
            agent.name,
            agent.agent_id,
            [c.name for c in agent.get_capabilities()],
        )

    async def unregister(self, agent_id: str) -> None:
        """Remove the agent identified by *agent_id*.

        Args:
            agent_id: Unique ID of the agent to remove.

        Raises:
            AgentNotFoundError: If no agent with *agent_id* exists.
        """
        async with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                raise AgentNotFoundError(agent_id)
            self._name_index.pop(agent.name, None)
            self._registration_times.pop(agent_id, None)
            del self._agents[agent_id]

        self._logger.info("Unregistered agent %s", agent_id)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> BaseAgent:
        """Return the agent with *agent_id*.

        Args:
            agent_id: Unique agent identifier.

        Returns:
            The matching :class:`~core.agent_framework.BaseAgent`.

        Raises:
            AgentNotFoundError: If no agent with *agent_id* is registered.
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            raise AgentNotFoundError(agent_id)
        return agent

    def get_agent_by_name(self, name: str) -> BaseAgent:
        """Return the agent registered under *name*.

        Args:
            name: Human-readable agent name.

        Returns:
            The matching :class:`~core.agent_framework.BaseAgent`.

        Raises:
            AgentNotFoundError: If no agent with *name* is registered.
        """
        agent_id = self._name_index.get(name)
        if agent_id is None:
            raise AgentNotFoundError(name)
        return self.get_agent(agent_id)

    def get_all_agents(self) -> List[BaseAgent]:
        """Return all registered agents regardless of state.

        Returns:
            List of :class:`~core.agent_framework.BaseAgent` instances.
        """
        return list(self._agents.values())

    def get_active_agents(self) -> List[BaseAgent]:
        """Return agents that are in IDLE or BUSY state.

        Returns:
            List of active :class:`~core.agent_framework.BaseAgent` instances.
        """
        return [
            a
            for a in self._agents.values()
            if a.state in (AgentState.IDLE, AgentState.BUSY)
        ]

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self, capability: str) -> List[BaseAgent]:
        """Find all agents that advertise *capability*.

        Args:
            capability: Capability name to search for (case-insensitive).

        Returns:
            List of matching :class:`~core.agent_framework.BaseAgent` instances,
            sorted by ascending error rate (most reliable first).
        """
        matches = [
            a
            for a in self._agents.values()
            if a.has_capability(capability)
        ]
        return sorted(matches, key=lambda a: a.metrics.error_rate)

    def discover_available(self, capability: str) -> List[BaseAgent]:
        """Like :meth:`discover` but limits to IDLE agents.

        Args:
            capability: Capability name to search for.

        Returns:
            List of idle agents matching *capability*.
        """
        return [a for a in self.discover(capability) if a.state == AgentState.IDLE]

    def list_all_capabilities(self) -> Dict[str, List[str]]:
        """Return a mapping of capability → list of agent IDs that support it.

        Returns:
            Dict where each key is a capability name and the value is a list
            of agent IDs.
        """
        mapping: Dict[str, List[str]] = {}
        for agent in self._agents.values():
            for cap in agent.get_capabilities():
                mapping.setdefault(cap.name, []).append(agent.agent_id)
        return mapping

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    async def health_check_all(
        self, *, timeout: float = 10.0
    ) -> Dict[str, Dict[str, Any]]:
        """Run health checks concurrently across all registered agents.

        Agents that do not expose a ``health_check`` method receive a default
        status-based assessment instead.

        Args:
            timeout: Per-agent health check timeout in seconds.

        Returns:
            Dict mapping agent_id → health-check result dict.
        """
        results: Dict[str, Dict[str, Any]] = {}

        async def _check(agent: BaseAgent) -> None:
            try:
                checker = getattr(agent, "health_check", None)
                if callable(checker):
                    result = await asyncio.wait_for(checker(), timeout=timeout)
                else:
                    result = {
                        "agent_id": agent.agent_id,
                        "name": agent.name,
                        "healthy": agent.state
                        in (AgentState.IDLE, AgentState.BUSY),
                        "state": agent.state.name,
                        "timestamp": timestamp_now(),
                    }
                results[agent.agent_id] = result
            except asyncio.TimeoutError:
                results[agent.agent_id] = {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "healthy": False,
                    "error": f"Health check timed out after {timeout}s",
                    "timestamp": timestamp_now(),
                }
            except Exception as exc:
                results[agent.agent_id] = {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "healthy": False,
                    "error": str(exc),
                    "timestamp": timestamp_now(),
                }

        agents = list(self._agents.values())
        await asyncio.gather(*[_check(a) for a in agents])
        return results

    # ------------------------------------------------------------------
    # Workload statistics
    # ------------------------------------------------------------------

    def get_load_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return per-agent workload and performance statistics.

        Returns:
            Dict mapping agent_id → stats dict containing:
            - ``state``, ``tasks_completed``, ``tasks_failed``,
              ``avg_response_time``, ``error_rate``, ``uptime``,
              ``registered_since``.
        """
        now = time.monotonic()
        stats: Dict[str, Dict[str, Any]] = {}
        for agent_id, agent in self._agents.items():
            m = agent.metrics
            registered = self._registration_times.get(agent_id, now)
            stats[agent_id] = {
                "name": agent.name,
                "state": agent.state.name,
                "tasks_completed": m.tasks_completed,
                "tasks_failed": m.tasks_failed,
                "avg_response_time_s": round(m.avg_response_time, 4),
                "error_rate": round(m.error_rate, 4),
                "uptime_s": round(m.uptime, 1),
                "registered_since_s": round(now - registered, 1),
                "capabilities": [c.name for c in agent.get_capabilities()],
            }
        return stats

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def __repr__(self) -> str:
        return f"AgentRegistry(agents={len(self._agents)})"


__all__ = ["AgentRegistry"]
