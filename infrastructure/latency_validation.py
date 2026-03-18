"""
Latency validation helpers for sprint acceptance checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LatencyValidationResult:
    sample_count: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    target_p95_ms: float
    passed: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "sample_count": self.sample_count,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
            "target_p95_ms": self.target_p95_ms,
            "passed": self.passed,
        }


def percentile(values: Sequence[float], q: float) -> float:
    """Return nearest-rank percentile for q in [0, 1]."""
    if not values:
        return 0.0
    clamped = max(0.0, min(1.0, q))
    ordered = sorted(float(v) for v in values)
    rank = max(1, int(math.ceil(clamped * len(ordered))))
    return round(ordered[rank - 1], 2)


def validate_latency_budget(
    latencies_ms: Sequence[float],
    *,
    target_p95_ms: float,
) -> LatencyValidationResult:
    if not latencies_ms:
        return LatencyValidationResult(
            sample_count=0,
            p50_ms=0.0,
            p95_ms=0.0,
            max_ms=0.0,
            target_p95_ms=float(target_p95_ms),
            passed=False,
        )
    p50 = percentile(latencies_ms, 0.50)
    p95 = percentile(latencies_ms, 0.95)
    max_ms = round(max(float(v) for v in latencies_ms), 2)
    target = float(target_p95_ms)
    return LatencyValidationResult(
        sample_count=len(latencies_ms),
        p50_ms=p50,
        p95_ms=p95,
        max_ms=max_ms,
        target_p95_ms=target,
        passed=p95 <= target,
    )
