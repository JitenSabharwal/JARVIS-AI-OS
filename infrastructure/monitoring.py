"""
System health monitoring for JARVIS AI OS.

Collects CPU / memory / disk metrics (via psutil with a pure-Python fallback),
tracks per-component health, fires alert callbacks, and maintains a rolling
history of health checks.
"""

from __future__ import annotations

import asyncio
import platform
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PSUTIL_AVAILABLE = False

from infrastructure.logger import get_logger

logger = get_logger("monitoring")

AlertHandler = Callable[["ComponentHealth"], Awaitable[None]]


# ---------------------------------------------------------------------------
# Enums & data structures
# ---------------------------------------------------------------------------

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health snapshot for a single registered component."""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: float = field(default_factory=time.time)
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    check_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check,
            "message": self.message,
            "metrics": self.metrics,
            "check_duration_ms": round(self.check_duration_ms, 2),
        }


@dataclass
class SystemMetrics:
    """Point-in-time snapshot of host resource usage."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    active_agents: int = 0
    active_tasks: int = 0
    uptime_seconds: float = 0.0
    collected_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "memory_used_mb": round(self.memory_used_mb, 1),
            "memory_total_mb": round(self.memory_total_mb, 1),
            "disk_percent": round(self.disk_percent, 1),
            "disk_used_gb": round(self.disk_used_gb, 2),
            "disk_total_gb": round(self.disk_total_gb, 2),
            "active_agents": self.active_agents,
            "active_tasks": self.active_tasks,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "collected_at": self.collected_at,
        }


@dataclass
class HealthReport:
    """Full system health report."""
    overall_status: HealthStatus
    components: list[ComponentHealth]
    system_metrics: SystemMetrics
    generated_at: float = field(default_factory=time.time)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "summary": self.summary,
            "generated_at": self.generated_at,
            "system_metrics": self.system_metrics.to_dict(),
            "components": [c.to_dict() for c in self.components],
        }


# ---------------------------------------------------------------------------
# CheckFn type: optional async callable that checks a component
# ---------------------------------------------------------------------------

# Signature: () -> ComponentHealth
ComponentCheckFn = Callable[[], Awaitable["ComponentHealth"]]


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class Monitor:
    """
    System health monitor.

    Usage::

        monitor = Monitor()
        monitor.register_component("redis", my_redis_check_fn)
        await monitor.start_monitoring(interval=30)
        report = await monitor.get_health_report()
        await monitor.stop_monitoring()
    """

    _MAX_HISTORY = 500          # health records per component
    _PROCESS_START = time.time()

    def __init__(self, check_interval: float = 60.0) -> None:
        self._check_interval = check_interval
        self._components: dict[str, ComponentCheckFn] = {}
        self._health: dict[str, ComponentHealth] = {}
        self._history: dict[str, deque[ComponentHealth]] = {}
        self.alert_handlers: list[AlertHandler] = []
        self._running = False
        self._monitor_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        logger.debug("Monitor created (interval=%.1fs, psutil=%s)", check_interval, _PSUTIL_AVAILABLE)

    # ------------------------------------------------------------------
    # Component registration
    # ------------------------------------------------------------------

    def register_component(
        self,
        name: str,
        check_fn: ComponentCheckFn,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a health-check function for component *name*.

        *check_fn* is an async callable that returns a ``ComponentHealth``
        instance (status + optional metrics).
        """
        if name in self._components and not overwrite:
            raise ValueError(f"Component already registered: '{name}'. Use overwrite=True.")
        self._components[name] = check_fn
        self._history[name] = deque(maxlen=self._MAX_HISTORY)
        # Seed with UNKNOWN until first check
        self._health[name] = ComponentHealth(name=name, status=HealthStatus.UNKNOWN)
        logger.debug("Component registered: '%s'", name)

    def unregister_component(self, name: str) -> bool:
        """Remove a registered component. Returns False if not found."""
        if name not in self._components:
            return False
        del self._components[name]
        del self._history[name]
        self._health.pop(name, None)
        return True

    # ------------------------------------------------------------------
    # Manual checks
    # ------------------------------------------------------------------

    async def check_component(self, name: str) -> ComponentHealth:
        """Run the health check for a single component and update its record."""
        check_fn = self._components.get(name)
        if check_fn is None:
            raise KeyError(f"Unknown component: '{name}'")

        start = time.time()
        try:
            result = await check_fn()
        except Exception as exc:  # noqa: BLE001
            result = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check raised exception: {exc}",
            )
            logger.warning("Component '%s' check raised: %s", name, exc)

        result.check_duration_ms = (time.time() - start) * 1000
        result.last_check = time.time()

        async with self._lock:
            self._health[name] = result
            self._history[name].append(result)

        # Fire alerts for non-healthy results
        if result.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY):
            await self._fire_alerts(result)

        return result

    async def check_all(self) -> dict[str, ComponentHealth]:
        """Run health checks on all registered components concurrently."""
        tasks = {name: asyncio.create_task(self.check_component(name)) for name in self._components}
        results: dict[str, ComponentHealth] = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as exc:  # noqa: BLE001
                results[name] = ComponentHealth(
                    name=name, status=HealthStatus.UNHEALTHY, message=str(exc)
                )
        return results

    # ------------------------------------------------------------------
    # System metrics
    # ------------------------------------------------------------------

    def collect_system_metrics(
        self,
        *,
        active_agents: int = 0,
        active_tasks: int = 0,
    ) -> SystemMetrics:
        """
        Collect host resource usage.

        Uses psutil when available; falls back to /proc-based reads on Linux,
        and returns zeros on other platforms.
        """
        uptime = time.time() - self._PROCESS_START

        if _PSUTIL_AVAILABLE:
            return self._collect_via_psutil(active_agents, active_tasks, uptime)
        return self._collect_fallback(active_agents, active_tasks, uptime)

    @staticmethod
    def _collect_via_psutil(
        active_agents: int, active_tasks: int, uptime: float
    ) -> SystemMetrics:
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            return SystemMetrics(
                cpu_percent=cpu,
                memory_percent=mem.percent,
                memory_used_mb=mem.used / 1024 / 1024,
                memory_total_mb=mem.total / 1024 / 1024,
                disk_percent=disk.percent,
                disk_used_gb=disk.used / 1024 / 1024 / 1024,
                disk_total_gb=disk.total / 1024 / 1024 / 1024,
                active_agents=active_agents,
                active_tasks=active_tasks,
                uptime_seconds=uptime,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("psutil collection failed: %s", exc)
            return SystemMetrics(active_agents=active_agents, active_tasks=active_tasks, uptime_seconds=uptime)

    @staticmethod
    def _collect_fallback(
        active_agents: int, active_tasks: int, uptime: float
    ) -> SystemMetrics:
        """Minimal /proc-based collection for Linux; zeros otherwise."""
        metrics = SystemMetrics(
            active_agents=active_agents,
            active_tasks=active_tasks,
            uptime_seconds=uptime,
        )
        if platform.system() != "Linux":
            return metrics

        try:
            with open("/proc/meminfo") as fh:
                mem_data: dict[str, int] = {}
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 2:
                        mem_data[parts[0].rstrip(":")] = int(parts[1])
            total_kb = mem_data.get("MemTotal", 0)
            avail_kb = mem_data.get("MemAvailable", mem_data.get("MemFree", 0))
            used_kb = total_kb - avail_kb
            metrics.memory_total_mb = total_kb / 1024
            metrics.memory_used_mb = used_kb / 1024
            metrics.memory_percent = (used_kb / total_kb * 100) if total_kb else 0.0
        except OSError:
            pass

        try:
            import os
            st = os.statvfs("/")
            total_bytes = st.f_blocks * st.f_frsize
            free_bytes = st.f_bavail * st.f_frsize
            used_bytes = total_bytes - free_bytes
            metrics.disk_total_gb = total_bytes / 1024 ** 3
            metrics.disk_used_gb = used_bytes / 1024 ** 3
            metrics.disk_percent = (used_bytes / total_bytes * 100) if total_bytes else 0.0
        except OSError:
            pass

        return metrics

    # ------------------------------------------------------------------
    # Health report
    # ------------------------------------------------------------------

    async def get_health_report(
        self,
        *,
        run_checks: bool = False,
        active_agents: int = 0,
        active_tasks: int = 0,
    ) -> HealthReport:
        """
        Build a full health report.

        If *run_checks* is True, fresh component checks are performed first;
        otherwise the most recent cached results are used.
        """
        if run_checks:
            await self.check_all()

        async with self._lock:
            component_healths = list(self._health.values())

        system_metrics = self.collect_system_metrics(
            active_agents=active_agents, active_tasks=active_tasks
        )
        overall = self._aggregate_status(component_healths, system_metrics)
        summary = self._build_summary(overall, component_healths, system_metrics)

        return HealthReport(
            overall_status=overall,
            components=component_healths,
            system_metrics=system_metrics,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def add_alert_handler(self, handler: AlertHandler) -> None:
        """Register a callback invoked whenever a component becomes non-healthy."""
        self.alert_handlers.append(handler)

    async def _fire_alerts(self, health: ComponentHealth) -> None:
        for handler in self.alert_handlers:
            try:
                await handler(health)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Alert handler raised: %s", exc)

    # ------------------------------------------------------------------
    # Background monitoring loop
    # ------------------------------------------------------------------

    async def start_monitoring(self, interval: float | None = None) -> None:
        """Start the periodic background health-check loop."""
        if self._running:
            return
        if interval is not None:
            self._check_interval = interval
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitoring_loop(), name="health_monitor"
        )
        logger.info("Health monitor started (interval=%.1fs)", self._check_interval)

    async def stop_monitoring(self) -> None:
        """Stop the background loop."""
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")

    async def _monitoring_loop(self) -> None:
        while self._running:
            try:
                await self.check_all()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Monitoring loop error: %s", exc)
            await asyncio.sleep(self._check_interval)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_component_history(
        self, name: str, limit: int = 20
    ) -> list[ComponentHealth]:
        """Return the last *limit* health records for *name*."""
        hist = self._history.get(name)
        if hist is None:
            return []
        items = list(reversed(hist))
        return items[:limit]

    def get_component_health(self, name: str) -> ComponentHealth | None:
        """Return the most recent cached health for *name*."""
        return self._health.get(name)

    def list_components(self) -> list[str]:
        return list(self._components.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_status(
        healths: list[ComponentHealth], metrics: SystemMetrics
    ) -> HealthStatus:
        """Roll up component statuses; also degrade on high resource usage."""
        if any(h.status == HealthStatus.UNHEALTHY for h in healths):
            return HealthStatus.UNHEALTHY
        if any(h.status == HealthStatus.DEGRADED for h in healths):
            return HealthStatus.DEGRADED
        # Resource pressure thresholds
        if metrics.cpu_percent > 90 or metrics.memory_percent > 90 or metrics.disk_percent > 95:
            return HealthStatus.DEGRADED
        if not healths:
            return HealthStatus.UNKNOWN
        return HealthStatus.HEALTHY

    @staticmethod
    def _build_summary(
        status: HealthStatus,
        healths: list[ComponentHealth],
        metrics: SystemMetrics,
    ) -> str:
        unhealthy = [h.name for h in healths if h.status == HealthStatus.UNHEALTHY]
        degraded = [h.name for h in healths if h.status == HealthStatus.DEGRADED]
        parts = [f"Overall: {status.value.upper()}"]
        if unhealthy:
            parts.append(f"Unhealthy: {', '.join(unhealthy)}")
        if degraded:
            parts.append(f"Degraded: {', '.join(degraded)}")
        parts.append(
            f"CPU={metrics.cpu_percent:.1f}% MEM={metrics.memory_percent:.1f}% "
            f"DISK={metrics.disk_percent:.1f}%"
        )
        return " | ".join(parts)

    def __repr__(self) -> str:
        return (
            f"<Monitor components={len(self._components)} "
            f"running={self._running} interval={self._check_interval}s>"
        )
