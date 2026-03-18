"""
In-memory SLO metrics collector for latency/error/success tracking.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LatencySeries:
    samples_ms: List[float] = field(default_factory=list)
    max_samples: int = 5000

    def observe(self, value_ms: float) -> None:
        self.samples_ms.append(max(0.0, float(value_ms)))
        if len(self.samples_ms) > self.max_samples:
            self.samples_ms = self.samples_ms[-self.max_samples :]

    def quantile(self, q: float) -> float:
        if not self.samples_ms:
            return 0.0
        q = min(1.0, max(0.0, float(q)))
        arr = sorted(self.samples_ms)
        idx = int(round((len(arr) - 1) * q))
        return round(arr[idx], 2)

    def avg(self) -> float:
        if not self.samples_ms:
            return 0.0
        return round(sum(self.samples_ms) / len(self.samples_ms), 2)

    def count(self) -> int:
        return len(self.samples_ms)


class SLOMetrics:
    """Thread-safe in-memory metric accumulator."""

    def __init__(self) -> None:
        self._counters: Dict[Tuple[str, str], int] = {}
        self._latency: Dict[Tuple[str, str], LatencySeries] = {}
        self._started_at = time.time()
        self._lock = threading.RLock()

    def inc(self, name: str, *, label: str = "default", value: int = 1) -> None:
        key = (name, label)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + int(value)

    def observe_latency(self, name: str, value_ms: float, *, label: str = "default") -> None:
        key = (name, label)
        with self._lock:
            series = self._latency.get(key)
            if series is None:
                series = LatencySeries()
                self._latency[key] = series
            series.observe(value_ms)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters = {
                f"{name}:{label}": value
                for (name, label), value in sorted(self._counters.items())
            }
            latency = {
                f"{name}:{label}": {
                    "count": series.count(),
                    "avg_ms": series.avg(),
                    "p50_ms": series.quantile(0.50),
                    "p95_ms": series.quantile(0.95),
                    "p99_ms": series.quantile(0.99),
                }
                for (name, label), series in sorted(self._latency.items())
            }
        return {
            "started_at": self._started_at,
            "uptime_seconds": round(time.time() - self._started_at, 2),
            "counters": counters,
            "latency": latency,
        }


_global_metrics: Optional[SLOMetrics] = None
_global_lock = threading.Lock()


def get_slo_metrics() -> SLOMetrics:
    global _global_metrics
    with _global_lock:
        if _global_metrics is None:
            _global_metrics = SLOMetrics()
    return _global_metrics


def reset_slo_metrics() -> None:
    global _global_metrics
    with _global_lock:
        _global_metrics = None


def evaluate_slo_snapshot(
    snapshot: Dict[str, Any],
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    defaults: Dict[str, float] = {
        "api_request_p95_ms": 1500.0,
        "api_error_rate_pct": 5.0,
        "run_command_p95_ms": 30000.0,
        "min_samples": 20.0,
    }
    if thresholds:
        defaults.update({k: float(v) for k, v in thresholds.items()})

    counters = snapshot.get("counters", {}) or {}
    latency = snapshot.get("latency", {}) or {}
    min_samples = int(defaults.get("min_samples", 20.0))

    total_responses = sum(
        int(v) for k, v in counters.items() if str(k).startswith("api_responses_total:")
    )
    total_errors = sum(
        int(v) for k, v in counters.items() if str(k).startswith("api_errors_total:")
    )
    api_error_rate_pct = round((total_errors / total_responses * 100.0), 3) if total_responses else 0.0

    api_p95_values = [
        float(v.get("p95_ms", 0.0))
        for k, v in latency.items()
        if str(k).startswith("api_request_latency_ms:") and int(v.get("count", 0)) >= min_samples
    ]
    run_command_p95_values = [
        float(v.get("p95_ms", 0.0))
        for k, v in latency.items()
        if str(k).startswith("run_command_total_latency_ms:") and int(v.get("count", 0)) >= min_samples
    ]
    api_request_p95_ms = round(max(api_p95_values), 2) if api_p95_values else None
    run_command_p95_ms = round(max(run_command_p95_values), 2) if run_command_p95_values else None

    violations: List[Dict[str, Any]] = []
    if total_responses >= min_samples and api_error_rate_pct > defaults["api_error_rate_pct"]:
        violations.append(
            {
                "metric": "api_error_rate_pct",
                "value": api_error_rate_pct,
                "threshold": defaults["api_error_rate_pct"],
                "severity": "warning",
            }
        )
    if api_request_p95_ms is not None and api_request_p95_ms > defaults["api_request_p95_ms"]:
        violations.append(
            {
                "metric": "api_request_p95_ms",
                "value": api_request_p95_ms,
                "threshold": defaults["api_request_p95_ms"],
                "severity": "warning",
            }
        )
    if run_command_p95_ms is not None and run_command_p95_ms > defaults["run_command_p95_ms"]:
        violations.append(
            {
                "metric": "run_command_p95_ms",
                "value": run_command_p95_ms,
                "threshold": defaults["run_command_p95_ms"],
                "severity": "warning",
            }
        )

    return {
        "healthy": len(violations) == 0,
        "thresholds": defaults,
        "measurements": {
            "api_error_rate_pct": api_error_rate_pct,
            "api_request_p95_ms": api_request_p95_ms,
            "run_command_p95_ms": run_command_p95_ms,
            "sample_counts": {
                "api_responses": total_responses,
                "api_errors": total_errors,
            },
        },
        "violations": violations,
    }
