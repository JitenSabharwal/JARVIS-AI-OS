"""
Base agent classes and lifecycle management for the JARVIS AI OS system.

This module provides the foundational abstractions for all agents:
- :class:`AgentState` – lifecycle states
- :class:`AgentCapability` – declarative capability descriptor
- :class:`AgentMessage` – typed inter-agent message envelope
- :class:`AgentMetrics` – operational telemetry
- :class:`BaseAgent` – abstract base that all concrete agents must extend
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from infrastructure.logger import get_logger
from utils.exceptions import AgentCapabilityError, AgentError, TaskTimeoutError
from utils.helpers import generate_id, timestamp_now

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AgentState(Enum):
    """All possible lifecycle states for an agent."""

    IDLE = auto()
    BUSY = auto()
    ERROR = auto()
    OFFLINE = auto()
    INITIALIZING = auto()
    SHUTTING_DOWN = auto()


class MessageType(Enum):
    """Semantic classification of inter-agent messages."""

    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class MessagePriority(Enum):
    """Priority levels for message processing order."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AgentCapability:
    """Declarative descriptor for a single capability an agent can perform.

    Args:
        name: Unique capability identifier (e.g. ``"code_review"``).
        description: Human-readable explanation of what the capability does.
        parameters_schema: JSON-Schema-style dict describing accepted parameters.
        version: Optional semantic version string.
    """

    name: str
    description: str
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

    def matches(self, capability_name: str) -> bool:
        """Return ``True`` if *capability_name* matches this capability."""
        return self.name.lower() == capability_name.lower()


@dataclass
class AgentMessage:
    """Typed envelope for all inter-agent communication.

    Args:
        sender_id: Agent ID of the message originator.
        recipient_id: Agent ID of the intended recipient.  Empty string means
            broadcast.
        message_type: Semantic classification (:class:`MessageType`).
        payload: Arbitrary dict payload; must be JSON-serialisable.
        id: Auto-generated unique message ID.
        timestamp: ISO-8601 creation timestamp.
        priority: Processing priority (:class:`MessagePriority`).
        correlation_id: Optional ID linking request/response pairs.
        metadata: Optional free-form extra metadata.
    """

    sender_id: str
    recipient_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    id: str = field(default_factory=lambda: generate_id("msg"))
    timestamp: str = field(default_factory=timestamp_now)
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def make_reply(
        self,
        payload: Dict[str, Any],
        *,
        message_type: MessageType = MessageType.RESPONSE,
        priority: Optional[MessagePriority] = None,
    ) -> "AgentMessage":
        """Construct a reply message, inheriting correlation context."""
        return AgentMessage(
            sender_id=self.recipient_id,
            recipient_id=self.sender_id,
            message_type=message_type,
            payload=payload,
            correlation_id=self.id,
            priority=priority or self.priority,
        )


@dataclass
class AgentMetrics:
    """Operational telemetry snapshot for an agent.

    All counters and timings are accumulated since the agent started.
    """

    tasks_completed: int = 0
    tasks_failed: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    avg_response_time: float = 0.0  # seconds
    uptime: float = 0.0  # seconds since start
    last_active_at: Optional[str] = None
    error_rate: float = 0.0  # fraction 0.0-1.0

    def record_task(self, *, success: bool, duration: float) -> None:
        """Update counters and rolling average after a task finishes."""
        total = self.tasks_completed + self.tasks_failed
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        new_total = total + 1
        # Rolling average of response time
        self.avg_response_time = (
            (self.avg_response_time * total + duration) / new_total
        )
        self.last_active_at = timestamp_now()
        if new_total > 0:
            self.error_rate = self.tasks_failed / new_total


# ---------------------------------------------------------------------------
# Abstract base agent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base class that all JARVIS agents must extend.

    Subclasses implement :meth:`initialize`, :meth:`shutdown`,
    :meth:`handle_message`, and :meth:`get_capabilities`.  The base class
    provides the messaging queue, lifecycle management, and task execution
    scaffolding.

    Args:
        name: Human-readable agent name.
        agent_id: Optional pre-assigned ID; auto-generated when omitted.
        queue_size: Maximum pending messages in the inbox before callers block.
    """

    def __init__(
        self,
        name: str,
        agent_id: Optional[str] = None,
        queue_size: int = 100,
    ) -> None:
        self._agent_id: str = agent_id or generate_id("agent")
        self._name: str = name
        self._state: AgentState = AgentState.OFFLINE
        self._metrics: AgentMetrics = AgentMetrics()
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue(
            maxsize=queue_size
        )
        self._start_time: Optional[float] = None
        self._message_processor_task: Optional[asyncio.Task[None]] = None
        self._running: bool = False
        self._logger = get_logger(f"agent.{name}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def agent_id(self) -> str:
        """Unique agent identifier."""
        return self._agent_id

    @property
    def name(self) -> str:
        """Human-readable agent name."""
        return self._name

    @property
    def state(self) -> AgentState:
        """Current lifecycle state of the agent."""
        return self._state

    @property
    def metrics(self) -> AgentMetrics:
        """Live telemetry snapshot (uptime is updated on access)."""
        if self._start_time is not None:
            self._metrics.uptime = time.monotonic() - self._start_time
        return self._metrics

    @property
    def is_available(self) -> bool:
        """``True`` when the agent can accept new tasks."""
        return self._state == AgentState.IDLE

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def initialize(self) -> None:
        """Set up resources required by the agent (DB connections, models, etc.).

        Called once during :meth:`start`.  Implementations must be idempotent.
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources and perform graceful shutdown.

        Called once during :meth:`stop`.
        """

    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process a single inbound message and optionally return a reply.

        Args:
            message: The incoming :class:`AgentMessage`.

        Returns:
            An optional reply :class:`AgentMessage`, or ``None``.
        """

    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Return the list of capabilities this agent can perform."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the agent: initialize resources and begin processing messages.

        Transitions:  OFFLINE → INITIALIZING → IDLE
        """
        if self._state not in (AgentState.OFFLINE, AgentState.ERROR):
            self._logger.warning(
                "start() called on agent %s in state %s; ignored",
                self._agent_id,
                self._state.name,
            )
            return

        self._logger.info("Starting agent '%s' (%s)", self._name, self._agent_id)
        self.update_state(AgentState.INITIALIZING)
        try:
            await self.initialize()
        except Exception as exc:
            self.update_state(AgentState.ERROR)
            raise AgentError(
                f"Agent '{self._name}' failed to initialize: {exc}"
            ) from exc

        self._start_time = time.monotonic()
        self._running = True
        self._message_processor_task = asyncio.create_task(
            self._process_messages(), name=f"msg-processor-{self._agent_id}"
        )
        self.update_state(AgentState.IDLE)
        self._logger.info("Agent '%s' started successfully", self._name)

    async def stop(self) -> None:
        """Gracefully stop the agent: drain the queue and release resources.

        Transitions:  * → SHUTTING_DOWN → OFFLINE
        """
        if self._state == AgentState.OFFLINE:
            return

        self._logger.info("Stopping agent '%s' (%s)", self._name, self._agent_id)
        self.update_state(AgentState.SHUTTING_DOWN)
        self._running = False

        if self._message_processor_task and not self._message_processor_task.done():
            self._message_processor_task.cancel()
            try:
                await self._message_processor_task
            except asyncio.CancelledError:
                pass

        try:
            await self.shutdown()
        except Exception as exc:
            self._logger.error(
                "Error during shutdown of agent '%s': %s", self._name, exc
            )

        self.update_state(AgentState.OFFLINE)
        self._logger.info("Agent '%s' stopped", self._name)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def update_state(self, new_state: AgentState) -> None:
        """Transition the agent to *new_state* and emit a log entry.

        Args:
            new_state: The target :class:`AgentState`.
        """
        old_state = self._state
        self._state = new_state
        self._logger.debug(
            "Agent '%s' state: %s → %s",
            self._name,
            old_state.name,
            new_state.name,
        )

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def send_message(
        self,
        recipient_id: str,
        payload: Dict[str, Any],
        *,
        message_type: MessageType = MessageType.REQUEST,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
    ) -> AgentMessage:
        """Create and return a new outbound :class:`AgentMessage`.

        The caller is responsible for delivering the message to the recipient
        (typically via the registry or orchestrator bus).

        Args:
            recipient_id: Destination agent ID.
            payload: Message body dict.
            message_type: Semantic message type.
            priority: Processing priority hint.
            correlation_id: Optional link to a prior message.

        Returns:
            The constructed :class:`AgentMessage`.
        """
        message = AgentMessage(
            sender_id=self._agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
        )
        self._metrics.messages_sent += 1
        self._logger.debug(
            "Agent '%s' sending %s to %s (msg=%s)",
            self._name,
            message_type.value,
            recipient_id,
            message.id,
        )
        return message

    async def receive_message(self, message: AgentMessage) -> None:
        """Enqueue *message* for processing.

        Args:
            message: Inbound :class:`AgentMessage` to enqueue.

        Raises:
            AgentError: If the agent is offline or shutting down.
        """
        if self._state in (AgentState.OFFLINE, AgentState.SHUTTING_DOWN):
            raise AgentError(
                f"Agent '{self._name}' is not accepting messages (state={self._state.name})"
            )
        self._metrics.messages_received += 1
        await self._message_queue.put(message)

    # ------------------------------------------------------------------
    # Task execution helpers
    # ------------------------------------------------------------------

    async def execute_task(
        self,
        capability: str,
        parameters: Dict[str, Any],
        *,
        timeout: Optional[float] = None,
        task_id: Optional[str] = None,
    ) -> Any:
        """Execute *capability* with *parameters*, honouring an optional timeout.

        Looks up the capability, marks the agent as BUSY, runs the handler,
        records metrics, and restores IDLE state on completion.

        Args:
            capability: Name of the capability to invoke.
            parameters: Runtime arguments for the capability.
            timeout: Maximum execution time in seconds.
            task_id: Optional caller-assigned task ID for logging.

        Returns:
            The result of the capability handler.

        Raises:
            AgentCapabilityError: If the capability is not registered.
            TaskTimeoutError: If the task exceeds *timeout*.
            AgentError: On other execution failures.
        """
        task_id = task_id or generate_id("task")
        self._validate_capability(capability)

        self._logger.info(
            "Agent '%s' executing capability='%s' task_id=%s",
            self._name,
            capability,
            task_id,
        )
        prev_state = self._state
        self.update_state(AgentState.BUSY)
        start = time.monotonic()
        success = False

        try:
            coro = self._dispatch_capability(capability, parameters)
            if timeout is not None:
                try:
                    result = await asyncio.wait_for(coro, timeout=timeout)
                except asyncio.TimeoutError as exc:
                    raise TaskTimeoutError(
                        task_id=task_id, timeout_seconds=timeout
                    ) from exc
            else:
                result = await coro

            success = True
            return result

        except TaskTimeoutError:
            raise
        except Exception as exc:
            self._logger.error(
                "Agent '%s' capability='%s' task_id=%s failed: %s",
                self._name,
                capability,
                task_id,
                exc,
            )
            raise AgentError(
                f"Task '{task_id}' on agent '{self._name}' failed: {exc}"
            ) from exc

        finally:
            duration = time.monotonic() - start
            self._metrics.record_task(success=success, duration=duration)
            if self._state == AgentState.BUSY:
                self.update_state(prev_state if prev_state == AgentState.IDLE else AgentState.IDLE)

    async def _dispatch_capability(
        self, capability: str, parameters: Dict[str, Any]
    ) -> Any:
        """Route a capability call to its registered handler method.

        Subclasses may override to customise dispatch.  Default implementation
        calls ``handle_<capability>(parameters)`` if the method exists.

        Args:
            capability: Capability name.
            parameters: Runtime arguments.

        Returns:
            Handler return value.

        Raises:
            AgentCapabilityError: If no handler method is found.
        """
        handler_name = f"handle_{capability}"
        handler = getattr(self, handler_name, None)
        if handler is None or not callable(handler):
            raise AgentCapabilityError(
                agent_id=self._agent_id, capability=capability
            )
        return await handler(parameters)

    def _validate_capability(self, capability: str) -> None:
        """Raise :class:`AgentCapabilityError` if *capability* is not registered."""
        supported = {cap.name for cap in self.get_capabilities()}
        if capability not in supported:
            raise AgentCapabilityError(
                agent_id=self._agent_id, capability=capability
            )

    def has_capability(self, capability: str) -> bool:
        """Return ``True`` if the agent advertises *capability*."""
        return any(cap.matches(capability) for cap in self.get_capabilities())

    # ------------------------------------------------------------------
    # Internal message processing loop
    # ------------------------------------------------------------------

    async def _process_messages(self) -> None:
        """Background coroutine: consume the inbox and dispatch each message."""
        self._logger.debug("Agent '%s' message processor started", self._name)
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                reply = await self.handle_message(message)
                if reply is not None:
                    self._logger.debug(
                        "Agent '%s' produced reply msg=%s", self._name, reply.id
                    )
            except Exception as exc:
                self._logger.error(
                    "Agent '%s' error handling message %s: %s",
                    self._name,
                    message.id,
                    exc,
                )
            finally:
                self._message_queue.task_done()

        self._logger.debug("Agent '%s' message processor stopped", self._name)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self._agent_id!r}, "
            f"name={self._name!r}, "
            f"state={self._state.name})"
        )


__all__ = [
    "AgentState",
    "MessageType",
    "MessagePriority",
    "AgentCapability",
    "AgentMessage",
    "AgentMetrics",
    "BaseAgent",
]
