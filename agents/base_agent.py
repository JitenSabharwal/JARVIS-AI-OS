"""
Concrete base agent implementation for JARVIS AI OS.

Provides :class:`ConcreteAgent`, which extends :class:`BaseAgent` with:
- Built-in capability handlers for ``echo``, ``ping``, and ``status``
- Message routing via a handler-method dispatch table
- Agent configuration loading from :class:`~core.config.JARVISConfig`
- A health-check helper
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from core.agent_framework import (
    AgentCapability,
    AgentMessage,
    AgentState,
    BaseAgent,
    MessageType,
)
from core.config import get_config
from infrastructure.logger import get_logger
from infrastructure.model_router import ModelRequest, ModelRouter, PrivacyLevel
from utils.exceptions import AgentCapabilityError, AgentError
from utils.helpers import generate_id, timestamp_now


class ConcreteAgent(BaseAgent):
    """A fully functional base agent with built-in utility capabilities.

    Subclass this to create domain-specific agents.  Override
    :meth:`get_capabilities` to advertise additional capabilities and add
    ``handle_<capability>`` methods to implement them.

    Args:
        name: Human-readable agent name.
        agent_id: Optional pre-assigned ID; auto-generated when ``None``.
        config_overrides: Dict of configuration values that override defaults
            loaded from the global :class:`~core.config.JARVISConfig`.
    """

    # Built-in capability descriptors shared by all concrete agents
    _BUILTIN_CAPABILITIES: List[AgentCapability] = [
        AgentCapability(
            name="echo",
            description="Return the provided message payload unchanged.",
            parameters_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        ),
        AgentCapability(
            name="ping",
            description="Liveness check: returns a pong response with timestamp.",
            parameters_schema={"type": "object", "properties": {}},
        ),
        AgentCapability(
            name="status",
            description="Return the agent's current state and metrics snapshot.",
            parameters_schema={"type": "object", "properties": {}},
        ),
    ]

    def __init__(
        self,
        name: str,
        agent_id: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, agent_id=agent_id)
        self._config_overrides: Dict[str, Any] = config_overrides or {}
        self._agent_config: Dict[str, Any] = {}
        self._model_router: Optional[ModelRouter] = None
        self._logger = get_logger(f"agent.{name}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load agent configuration and prepare resources."""
        self._agent_config = self._load_config()
        self._logger.info(
            "ConcreteAgent '%s' initialized (timeout=%.1fs, retries=%d)",
            self.name,
            self._agent_config.get("default_timeout", 30.0),
            self._agent_config.get("max_retries", 3),
        )

    async def shutdown(self) -> None:
        """Release agent resources on shutdown."""
        self._logger.info("ConcreteAgent '%s' shutting down", self.name)

    def set_model_router(self, model_router: Optional[ModelRouter]) -> None:
        self._model_router = model_router

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def get_capabilities(self) -> List[AgentCapability]:
        """Return built-in capabilities plus any defined by subclasses.

        Subclasses should call ``super().get_capabilities()`` and extend
        the returned list with their own :class:`~core.agent_framework.AgentCapability`
        descriptors.
        """
        return list(self._BUILTIN_CAPABILITIES)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Route an inbound message to the appropriate handler.

        Dispatch strategy:
        1. ``REQUEST`` messages with a ``capability`` key in the payload are
           forwarded to :meth:`~core.agent_framework.BaseAgent.execute_task`.
        2. ``COMMAND`` messages with an ``action`` key are handled directly.
        3. ``HEARTBEAT`` messages are acknowledged silently.
        4. Unrecognised messages produce an ERROR reply.

        Args:
            message: Inbound :class:`~core.agent_framework.AgentMessage`.

        Returns:
            An optional reply message.
        """
        self._logger.debug(
            "Agent '%s' received %s from %s (id=%s)",
            self.name,
            message.message_type.value,
            message.sender_id,
            message.id,
        )

        if message.message_type == MessageType.HEARTBEAT:
            return None  # Heartbeats are silently acknowledged

        if message.message_type == MessageType.REQUEST:
            return await self._handle_request(message)

        if message.message_type == MessageType.COMMAND:
            return await self._handle_command(message)

        # Unrecognised message type
        return message.make_reply(
            {"error": f"Unsupported message type: {message.message_type.value}"},
            message_type=MessageType.ERROR,
        )

    async def _handle_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle a REQUEST message by dispatching to the named capability."""
        capability = message.payload.get("capability")
        if not capability:
            return message.make_reply(
                {"error": "Request payload missing 'capability' field"},
                message_type=MessageType.ERROR,
            )

        parameters = message.payload.get("parameters", {})
        try:
            result = await self.execute_task(
                capability,
                parameters,
                timeout=self._agent_config.get("default_timeout"),
                task_id=message.id,
            )
            return message.make_reply(
                {"result": result, "capability": capability},
                message_type=MessageType.RESPONSE,
            )
        except AgentCapabilityError as exc:
            return message.make_reply(
                {"error": str(exc)},
                message_type=MessageType.ERROR,
            )
        except AgentError as exc:
            return message.make_reply(
                {"error": str(exc)},
                message_type=MessageType.ERROR,
            )

    async def _handle_command(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle a COMMAND message."""
        action = message.payload.get("action", "")
        if action == "stop":
            asyncio.create_task(self.stop())
            return message.make_reply(
                {"status": "stopping"},
                message_type=MessageType.RESPONSE,
            )
        return message.make_reply(
            {"error": f"Unknown command action: '{action}'"},
            message_type=MessageType.ERROR,
        )

    # ------------------------------------------------------------------
    # Built-in capability handlers
    # ------------------------------------------------------------------

    async def handle_echo(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Echo capability: return *message* unchanged.

        Args:
            parameters: Must contain ``"message"`` (str).

        Returns:
            Dict with ``"echo"`` key set to the original message.
        """
        message_text = parameters.get("message", "")
        return {"echo": message_text, "agent_id": self.agent_id}

    async def handle_ping(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Ping capability: liveness probe.

        Returns:
            Dict with ``"pong": true`` and current timestamp.
        """
        return {
            "pong": True,
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "timestamp": timestamp_now(),
        }

    async def handle_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Status capability: return state and metrics snapshot.

        Returns:
            Dict summarising the agent's current condition.
        """
        m = self.metrics
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.name,
            "capabilities": [c.name for c in self.get_capabilities()],
            "metrics": {
                "tasks_completed": m.tasks_completed,
                "tasks_failed": m.tasks_failed,
                "messages_sent": m.messages_sent,
                "messages_received": m.messages_received,
                "avg_response_time_s": round(m.avg_response_time, 4),
                "uptime_s": round(m.uptime, 1),
                "error_rate": round(m.error_rate, 4),
                "last_active_at": m.last_active_at,
            },
        }

    async def _route_text_generation(
        self,
        *,
        prompt: str,
        task_type: str,
        privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM,
    ) -> Optional[str]:
        """Use model router when configured; return None on fallback path."""
        if not self._model_router or not self._model_router.has_provider():
            return None
        try:
            response = await self._model_router.generate(
                ModelRequest(
                    prompt=prompt,
                    task_type=task_type,
                    modality="text",
                    privacy_level=privacy_level,
                    metadata={"agent_id": self.agent_id, "agent_name": self.name},
                )
            )
            return response.text
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Model router generation failed in agent '%s': %s", self.name, exc)
            return None

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check on the agent.

        Verifies that the agent is in a responsive state and that its
        built-in capabilities are functional.

        Returns:
            A dict with ``"healthy": bool`` and diagnostic details.
        """
        health: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "name": self.name,
            "healthy": False,
            "state": self.state.name,
            "checks": {},
            "timestamp": timestamp_now(),
        }

        # Check 1: state must be IDLE or BUSY (not ERROR/OFFLINE)
        state_ok = self.state in (AgentState.IDLE, AgentState.BUSY)
        health["checks"]["state"] = {
            "passed": state_ok,
            "detail": self.state.name,
        }

        # Check 2: built-in ping
        ping_ok = False
        ping_detail = "not attempted"
        if state_ok:
            try:
                result = await asyncio.wait_for(
                    self.execute_task("ping", {}, task_id=generate_id("hc")),
                    timeout=5.0,
                )
                ping_ok = result.get("pong") is True
                ping_detail = "ok" if ping_ok else "unexpected response"
            except Exception as exc:
                ping_detail = str(exc)

        health["checks"]["ping"] = {"passed": ping_ok, "detail": ping_detail}

        health["healthy"] = state_ok and ping_ok
        return health

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        """Load agent configuration from the global config with overrides applied.

        Returns:
            Dict containing resolved configuration values.
        """
        try:
            cfg = get_config()
            base: Dict[str, Any] = {
                "default_timeout": cfg.agent.default_timeout,
                "max_retries": cfg.agent.max_retries,
                "retry_delay": cfg.agent.retry_delay,
                "retry_backoff": cfg.agent.retry_backoff,
                "heartbeat_interval": cfg.agent.heartbeat_interval,
            }
        except Exception:
            # Fallback defaults if config is unavailable
            base = {
                "default_timeout": 30.0,
                "max_retries": 3,
                "retry_delay": 1.0,
                "retry_backoff": 2.0,
                "heartbeat_interval": 5.0,
            }
        base.update(self._config_overrides)
        return base


__all__ = ["ConcreteAgent"]
