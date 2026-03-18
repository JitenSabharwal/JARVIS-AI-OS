"""
Abstract base class for all JARVIS AI OS skills.

Skills are modular, self-describing capabilities that agents can discover and
execute.  Every skill must inherit from :class:`BaseSkill` and implement
:meth:`execute`, :meth:`get_schema`, and :meth:`validate_params`.

Usage::

    from skills.base_skill import BaseSkill, SkillResult, SkillParameter

    class MySkill(BaseSkill):
        @property
        def name(self) -> str:
            return "my_skill"
        ...
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from infrastructure.logger import get_logger
from utils.exceptions import SkillExecutionError
from utils.helpers import generate_id, retry_async, timestamp_now

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class SkillStatus(Enum):
    """Lifecycle status of a registered skill."""

    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class SkillParameter:
    """Describes a single parameter accepted by a skill.

    Attributes:
        name: Machine-readable parameter name.
        type: JSON-schema type string (``"string"``, ``"integer"``, etc.).
        description: Human-readable description of the parameter.
        required: Whether the parameter must be supplied by the caller.
        default: Default value used when the parameter is omitted.
    """

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-schema-compatible property dict."""
        d: Dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if not self.required and self.default is not None:
            d["default"] = self.default
        return d


@dataclass
class SkillResult:
    """The outcome of a skill execution.

    Attributes:
        success: ``True`` when the skill completed without error.
        data: Payload returned by the skill on success.
        error: Human-readable error message on failure.
        execution_time: Wall-clock duration in seconds.
        metadata: Arbitrary extra context (e.g. HTTP status, file path).
    """

    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }

    @classmethod
    def failure(
        cls,
        error: str,
        execution_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SkillResult":
        """Convenience constructor for a failed result."""
        return cls(
            success=False,
            error=error,
            execution_time=execution_time,
            metadata=metadata or {},
        )

    @classmethod
    def ok(
        cls,
        data: Any,
        execution_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SkillResult":
        """Convenience constructor for a successful result."""
        return cls(
            success=True,
            data=data,
            execution_time=execution_time,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseSkill(ABC):
    """Abstract base class for all JARVIS skills.

    Subclasses must implement:
    - :meth:`execute` – core logic, receives validated params dict.
    - :meth:`get_schema` – returns the JSON-schema dict for LLM function calling.
    - :meth:`validate_params` – raises :exc:`ValueError` on bad params.

    Public properties that **must** be overridden:
    ``name``, ``description``, ``category``, ``version``.

    Use :meth:`safe_execute` instead of :meth:`execute` when you want built-in
    error handling, timing, and optional retry logic.
    """

    def __init__(self) -> None:
        self._status: SkillStatus = SkillStatus.ACTIVE
        self._execution_count: int = 0
        self._error_count: int = 0
        self._total_execution_time: float = 0.0
        self._skill_logger = get_logger(f"skill.{self.name}")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique machine-readable skill identifier (snake_case)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description used in registry listings."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Category string matching a :class:`~skills.tools_registry.SkillCategory` value."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string, e.g. ``"1.0.0"``."""

    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        """Execute the skill with *params* and return a :class:`SkillResult`.

        Implementations must **not** catch generic exceptions—let them
        propagate so :meth:`safe_execute` can handle them uniformly.

        Args:
            params: Validated parameter dictionary (keys from :meth:`get_schema`).

        Returns:
            A :class:`SkillResult` indicating success or failure.
        """

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON-schema dict describing this skill's parameters.

        The schema is used by the LLM for function calling.  It must follow
        the JSON Schema specification for ``object`` types::

            {
                "name": "my_skill",
                "description": "...",
                "parameters": {
                    "type": "object",
                    "properties": { ... },
                    "required": [...],
                },
            }
        """

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> None:
        """Validate *params* against the skill's schema.

        Args:
            params: Raw parameter dict supplied by the caller.

        Raises:
            ValueError: When a required parameter is missing or a value fails
                type/range checks.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    @property
    def status(self) -> SkillStatus:
        """Current lifecycle status of this skill."""
        return self._status

    @status.setter
    def status(self, value: SkillStatus) -> None:
        self._status = value

    def get_parameters(self) -> List[SkillParameter]:
        """Return a list of :class:`SkillParameter` objects for this skill.

        The default implementation derives parameters from :meth:`get_schema`.
        Override for richer metadata.
        """
        schema = self.get_schema()
        params_schema = schema.get("parameters", {})
        properties = params_schema.get("properties", {})
        required_names = set(params_schema.get("required", []))

        parameters: List[SkillParameter] = []
        for param_name, prop in properties.items():
            parameters.append(
                SkillParameter(
                    name=param_name,
                    type=prop.get("type", "string"),
                    description=prop.get("description", ""),
                    required=param_name in required_names,
                    default=prop.get("default"),
                )
            )
        return parameters

    def get_info(self) -> Dict[str, Any]:
        """Return a rich metadata dict suitable for registry listings."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "status": self._status.value,
            "schema": self.get_schema(),
            "stats": {
                "execution_count": self._execution_count,
                "error_count": self._error_count,
                "total_execution_time": round(self._total_execution_time, 4),
                "avg_execution_time": (
                    round(self._total_execution_time / self._execution_count, 4)
                    if self._execution_count
                    else 0.0
                ),
            },
        }

    async def safe_execute(self, params: Dict[str, Any]) -> SkillResult:
        """Execute the skill with full error handling, timing, and retry support.

        This is the **recommended** entry-point for callers.  It:

        1. Checks the skill is :attr:`SkillStatus.ACTIVE`.
        2. Validates *params* via :meth:`validate_params`.
        3. Times the execution.
        4. Catches all exceptions and returns a failure :class:`SkillResult`
           rather than propagating.
        5. Updates internal counters.

        Args:
            params: Raw parameter dict from the caller.

        Returns:
            A :class:`SkillResult` (never raises).
        """
        if self._status != SkillStatus.ACTIVE:
            return SkillResult.failure(
                error=f"Skill '{self.name}' is {self._status.value}; cannot execute."
            )

        # Validate parameters before timing starts.
        try:
            self.validate_params(params)
        except ValueError as exc:
            return SkillResult.failure(error=f"Parameter validation failed: {exc}")

        start = time.perf_counter()
        try:
            result = await self._execute_with_retry(params)
            elapsed = time.perf_counter() - start
            result.execution_time = elapsed
            self._execution_count += 1
            self._total_execution_time += elapsed
            if not result.success:
                self._error_count += 1
            self._skill_logger.debug(
                "Skill '%s' completed in %.3fs (success=%s)",
                self.name,
                elapsed,
                result.success,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            self._execution_count += 1
            self._error_count += 1
            self._total_execution_time += elapsed
            self._skill_logger.error(
                "Skill '%s' raised unhandled exception: %s", self.name, exc
            )
            return SkillResult.failure(
                error=str(exc),
                execution_time=elapsed,
                metadata={"exception_type": type(exc).__name__},
            )

    async def _execute_with_retry(self, params: Dict[str, Any]) -> SkillResult:
        """Inner execute wrapped with optional retry logic.

        Skills that want retry behaviour should override :attr:`max_retries`
        and :attr:`retry_delay` class attributes.
        """
        max_attempts = getattr(self, "max_retries", 1)
        delay = getattr(self, "retry_delay", 1.0)
        backoff = getattr(self, "retry_backoff", 2.0)
        last_result: Optional[SkillResult] = None

        for attempt in range(1, max_attempts + 1):
            last_result = await self.execute(params)
            if last_result.success:
                return last_result
            if attempt < max_attempts:
                self._skill_logger.debug(
                    "Skill '%s' attempt %d/%d failed; retrying in %.1fs",
                    self.name,
                    attempt,
                    max_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= backoff

        return last_result  # type: ignore[return-value]

    def _build_schema(
        self,
        parameters: List[SkillParameter],
    ) -> Dict[str, Any]:
        """Helper to build a standard JSON-schema function descriptor.

        Args:
            parameters: List of :class:`SkillParameter` objects.

        Returns:
            A dict ready for LLM function-calling APIs.
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for param in parameters:
            properties[param.name] = param.to_dict()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def enable(self) -> None:
        """Set status to ACTIVE."""
        self._status = SkillStatus.ACTIVE
        self._skill_logger.info("Skill '%s' enabled.", self.name)

    def disable(self) -> None:
        """Set status to DISABLED."""
        self._status = SkillStatus.DISABLED
        self._skill_logger.info("Skill '%s' disabled.", self.name)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"version={self.version!r} status={self._status.value!r}>"
        )


__all__ = [
    "SkillStatus",
    "SkillParameter",
    "SkillResult",
    "BaseSkill",
]
