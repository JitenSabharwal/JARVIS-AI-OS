"""
Custom exception hierarchy for the JARVIS AI OS system.

All exceptions inherit from JARVISError, allowing callers to catch the base
class when they do not need to distinguish between sub-types.
"""

from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class JARVISError(Exception):
    """Root exception for all JARVIS AI OS errors."""

    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        details: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.code: Optional[str] = code
        self.details: Optional[Any] = details

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"details={self.details!r})"
        )


# ---------------------------------------------------------------------------
# Agent errors
# ---------------------------------------------------------------------------


class AgentError(JARVISError):
    """Base class for agent-related errors."""


class AgentNotFoundError(AgentError):
    """Raised when a requested agent cannot be located in the registry."""

    def __init__(self, agent_id: str, **kwargs: Any) -> None:
        super().__init__(f"Agent not found: '{agent_id}'", **kwargs)
        self.agent_id = agent_id


class AgentCapabilityError(AgentError):
    """Raised when an agent is asked to perform a task outside its capabilities."""

    def __init__(
        self,
        agent_id: str,
        capability: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            f"Agent '{agent_id}' does not support capability '{capability}'",
            **kwargs,
        )
        self.agent_id = agent_id
        self.capability = capability


# ---------------------------------------------------------------------------
# Skill errors
# ---------------------------------------------------------------------------


class SkillError(JARVISError):
    """Base class for skill-related errors."""


class SkillNotFoundError(SkillError):
    """Raised when a requested skill cannot be found."""

    def __init__(self, skill_name: str, **kwargs: Any) -> None:
        super().__init__(f"Skill not found: '{skill_name}'", **kwargs)
        self.skill_name = skill_name


class SkillExecutionError(SkillError):
    """Raised when a skill fails during execution."""

    def __init__(
        self,
        skill_name: str,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        msg = f"Skill '{skill_name}' execution failed"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg, **kwargs)
        self.skill_name = skill_name
        self.reason = reason


# ---------------------------------------------------------------------------
# Memory errors
# ---------------------------------------------------------------------------


class MemoryError(JARVISError):
    """Base class for memory-subsystem errors."""


class MemoryStorageError(MemoryError):
    """Raised when data cannot be persisted to or retrieved from memory storage."""

    def __init__(
        self,
        operation: str = "",
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        parts = ["Memory storage error"]
        if operation:
            parts.append(f"during '{operation}'")
        if reason:
            parts.append(f": {reason}")
        super().__init__(" ".join(parts), **kwargs)
        self.operation = operation
        self.reason = reason


# ---------------------------------------------------------------------------
# Orchestrator / workflow errors
# ---------------------------------------------------------------------------


class OrchestratorError(JARVISError):
    """Base class for orchestrator errors."""


class WorkflowError(OrchestratorError):
    """Raised when a workflow cannot be parsed, planned, or executed."""

    def __init__(
        self,
        workflow_id: str = "",
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        msg = "Workflow error"
        if workflow_id:
            msg = f"Workflow '{workflow_id}' error"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg, **kwargs)
        self.workflow_id = workflow_id
        self.reason = reason


# ---------------------------------------------------------------------------
# Task errors
# ---------------------------------------------------------------------------


class TaskError(JARVISError):
    """Base class for task-related errors."""


class TaskTimeoutError(TaskError):
    """Raised when a task exceeds its allowed execution time."""

    def __init__(
        self,
        task_id: str = "",
        timeout_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        msg = "Task timed out"
        if task_id:
            msg = f"Task '{task_id}' timed out"
        if timeout_seconds is not None:
            msg = f"{msg} after {timeout_seconds}s"
        super().__init__(msg, **kwargs)
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds


class TaskDependencyError(TaskError):
    """Raised when a task's dependencies cannot be resolved or have failed."""

    def __init__(
        self,
        task_id: str = "",
        dependency_id: str = "",
        **kwargs: Any,
    ) -> None:
        msg = "Task dependency error"
        if task_id and dependency_id:
            msg = f"Task '{task_id}' dependency '{dependency_id}' failed or is missing"
        elif task_id:
            msg = f"Task '{task_id}' has unresolved dependencies"
        super().__init__(msg, **kwargs)
        self.task_id = task_id
        self.dependency_id = dependency_id


# ---------------------------------------------------------------------------
# Configuration / validation errors
# ---------------------------------------------------------------------------


class ConfigurationError(JARVISError):
    """Raised when the system or a component is misconfigured."""

    def __init__(self, setting: str = "", reason: str = "", **kwargs: Any) -> None:
        msg = "Configuration error"
        if setting:
            msg = f"Configuration error for '{setting}'"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg, **kwargs)
        self.setting = setting
        self.reason = reason


class ValidationError(JARVISError):
    """Raised when input or data fails validation checks."""

    def __init__(
        self,
        field: str = "",
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        msg = "Validation error"
        if field:
            msg = f"Validation error for field '{field}'"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg, **kwargs)
        self.field = field
        self.reason = reason


# ---------------------------------------------------------------------------
# Message bus / queue errors
# ---------------------------------------------------------------------------


class MessageBusError(JARVISError):
    """Base class for message-bus errors."""


class QueueError(MessageBusError):
    """Raised on queue operations that fail (full, empty, closed, etc.)."""

    def __init__(
        self,
        queue_name: str = "",
        operation: str = "",
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        msg = "Queue error"
        if queue_name:
            msg = f"Queue '{queue_name}' error"
        if operation:
            msg = f"{msg} during '{operation}'"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg, **kwargs)
        self.queue_name = queue_name
        self.operation = operation
        self.reason = reason


# ---------------------------------------------------------------------------
# Interface errors
# ---------------------------------------------------------------------------


class VoiceInterfaceError(JARVISError):
    """Raised when the voice interface encounters an error (STT/TTS/audio)."""

    def __init__(self, component: str = "", reason: str = "", **kwargs: Any) -> None:
        msg = "Voice interface error"
        if component:
            msg = f"Voice interface error in '{component}'"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg, **kwargs)
        self.component = component
        self.reason = reason


class APIError(JARVISError):
    """Raised when an external or internal API call fails."""

    def __init__(
        self,
        endpoint: str = "",
        status_code: Optional[int] = None,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        msg = "API error"
        if endpoint:
            msg = f"API error calling '{endpoint}'"
        if status_code is not None:
            msg = f"{msg} (HTTP {status_code})"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg, **kwargs)
        self.endpoint = endpoint
        self.status_code = status_code
        self.reason = reason


__all__ = [
    "JARVISError",
    "AgentError",
    "AgentNotFoundError",
    "AgentCapabilityError",
    "SkillError",
    "SkillNotFoundError",
    "SkillExecutionError",
    "MemoryError",
    "MemoryStorageError",
    "OrchestratorError",
    "WorkflowError",
    "TaskError",
    "TaskTimeoutError",
    "TaskDependencyError",
    "ConfigurationError",
    "ValidationError",
    "MessageBusError",
    "QueueError",
    "VoiceInterfaceError",
    "APIError",
]
