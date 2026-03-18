"""
Connector framework for external/system integrations.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from infrastructure.logger import get_logger

logger = get_logger(__name__)


class BaseConnector(ABC):
    """Abstract connector contract."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def description(self) -> str:
        return ""

    @abstractmethod
    async def invoke(self, operation: str, params: Dict[str, Any]) -> Any:
        pass

    async def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "connector": self.name}


@dataclass
class ConnectorInfo:
    name: str
    description: str
    status: str = "registered"
    failure_count: int = 0
    circuit_open_until: float = 0.0
    circuit_open: bool = False
    last_error: str = ""
    last_failure_at: float | None = None
    last_success_at: float | None = None


@dataclass
class ConnectorPolicy:
    required_scopes_by_operation: Dict[str, Set[str]] = field(default_factory=dict)
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 30.0


@dataclass
class ConnectorState:
    failure_count: int = 0
    circuit_open_until: float = 0.0
    last_error: str = ""
    last_failure_at: float | None = None
    last_success_at: float | None = None


class ConnectorRegistry:
    """Thread-safe registry for connectors."""

    def __init__(self) -> None:
        self._connectors: Dict[str, BaseConnector] = {}
        self._policies: Dict[str, ConnectorPolicy] = {}
        self._states: Dict[str, ConnectorState] = {}
        self._lock = threading.RLock()

    def register(
        self,
        connector: BaseConnector,
        *,
        overwrite: bool = False,
        policy: Optional[ConnectorPolicy] = None,
    ) -> None:
        with self._lock:
            if connector.name in self._connectors and not overwrite:
                raise ValueError(
                    f"Connector '{connector.name}' already registered. Use overwrite=True."
                )
            self._connectors[connector.name] = connector
            self._policies[connector.name] = policy or ConnectorPolicy()
            self._states.setdefault(connector.name, ConnectorState())
        logger.info("Registered connector '%s'", connector.name)

    def unregister(self, name: str) -> bool:
        with self._lock:
            if name not in self._connectors:
                return False
            del self._connectors[name]
            self._policies.pop(name, None)
            self._states.pop(name, None)
        logger.info("Unregistered connector '%s'", name)
        return True

    def get(self, name: str) -> Optional[BaseConnector]:
        with self._lock:
            return self._connectors.get(name)

    def list_info(self) -> List[Dict[str, Any]]:
        with self._lock:
            connectors = list(self._connectors.values())
            states = dict(self._states)
            now = time.time()
        return [
            ConnectorInfo(
                name=c.name,
                description=c.description,
                status="circuit_open" if states.get(c.name, ConnectorState()).circuit_open_until > now else "registered",
                failure_count=states.get(c.name, ConnectorState()).failure_count,
                circuit_open_until=states.get(c.name, ConnectorState()).circuit_open_until,
                circuit_open=states.get(c.name, ConnectorState()).circuit_open_until > now,
                last_error=states.get(c.name, ConnectorState()).last_error,
                last_failure_at=states.get(c.name, ConnectorState()).last_failure_at,
                last_success_at=states.get(c.name, ConnectorState()).last_success_at,
            ).__dict__
            for c in connectors
        ]

    async def invoke(
        self,
        name: str,
        operation: str,
        params: Dict[str, Any],
        *,
        actor_scopes: Optional[Set[str]] = None,
    ) -> Any:
        connector = self.get(name)
        if connector is None:
            raise KeyError(f"Connector not found: {name}")

        actor_scopes = actor_scopes or set()
        with self._lock:
            policy = self._policies.get(name, ConnectorPolicy())
            state = self._states.setdefault(name, ConnectorState())
            now = time.time()
            if state.circuit_open_until > now:
                raise RuntimeError(
                    f"Connector '{name}' circuit is open until {state.circuit_open_until:.3f}"
                )

            required = policy.required_scopes_by_operation.get(operation, set())
            if required and not required.issubset(actor_scopes):
                raise PermissionError(
                    f"Missing required scopes for connector '{name}' operation '{operation}': "
                    f"{sorted(required)}"
                )

        try:
            result = await connector.invoke(operation, params)
        except Exception as exc:
            with self._lock:
                policy = self._policies.get(name, ConnectorPolicy())
                state = self._states.setdefault(name, ConnectorState())
                state.failure_count += 1
                state.last_error = str(exc)
                state.last_failure_at = time.time()
                if state.failure_count >= max(1, policy.failure_threshold):
                    state.circuit_open_until = (
                        time.time() + max(1.0, policy.recovery_timeout_seconds)
                    )
                    logger.warning(
                        "Connector '%s' circuit opened for %.1fs after %d failures",
                        name,
                        policy.recovery_timeout_seconds,
                        state.failure_count,
                    )
            raise

        with self._lock:
            state = self._states.setdefault(name, ConnectorState())
            state.failure_count = 0
            state.last_error = ""
            state.circuit_open_until = 0.0
            state.last_success_at = time.time()
        return result

    async def health(self, name: str) -> Dict[str, Any]:
        connector = self.get(name)
        if connector is None:
            raise KeyError(f"Connector not found: {name}")
        base = await connector.health_check()
        with self._lock:
            state = self._states.get(name, ConnectorState())
        base["failure_count"] = state.failure_count
        base["circuit_open_until"] = state.circuit_open_until
        base["circuit_open"] = state.circuit_open_until > time.time()
        base["last_error"] = state.last_error
        base["last_failure_at"] = state.last_failure_at
        base["last_success_at"] = state.last_success_at
        return base

    async def health_all(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            names = list(self._connectors.keys())
        report: Dict[str, Dict[str, Any]] = {}
        for name in names:
            try:
                report[name] = await self.health(name)
            except Exception as exc:  # noqa: BLE001
                report[name] = {"healthy": False, "connector": name, "error": str(exc)}
        return report
