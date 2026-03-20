"""
Ingress admission control with bounded queueing.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True)
class IngressDecision:
    allowed: bool
    reason: str = ""
    queue_wait_ms: float = 0.0


class IngressController:
    """
    Bounded admission + queue controller for request ingress.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_inflight: int = 32,
        max_queue: int = 128,
        queue_wait_timeout_ms: int = 3000,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_inflight = max(1, int(max_inflight or 1))
        self.max_queue = max(0, int(max_queue or 0))
        self.queue_wait_timeout_ms = max(1, int(queue_wait_timeout_ms or 1))
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)
        self._inflight = 0
        self._queued = 0
        self._stats: Dict[str, int] = {
            "accepted": 0,
            "queued": 0,
            "rejected": 0,
            "timed_out": 0,
            "released": 0,
        }

    @classmethod
    def from_env(cls) -> "IngressController":
        enabled = str(os.getenv("JARVIS_INGRESS_ENABLED", "false")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            enabled=enabled,
            max_inflight=int(os.getenv("JARVIS_INGRESS_MAX_INFLIGHT", "32") or 32),
            max_queue=int(os.getenv("JARVIS_INGRESS_MAX_QUEUE", "128") or 128),
            queue_wait_timeout_ms=int(
                os.getenv("JARVIS_INGRESS_QUEUE_WAIT_TIMEOUT_MS", "3000") or 3000
            ),
        )

    async def acquire(self) -> IngressDecision:
        if not self.enabled:
            return IngressDecision(allowed=True)
        start = time.monotonic()
        async with self._cond:
            if self._inflight < self.max_inflight:
                self._inflight += 1
                self._stats["accepted"] += 1
                return IngressDecision(allowed=True, queue_wait_ms=0.0)

            if self._queued >= self.max_queue:
                self._stats["rejected"] += 1
                return IngressDecision(allowed=False, reason="ingress_queue_full")

            self._queued += 1
            self._stats["queued"] += 1
            timeout_s = self.queue_wait_timeout_ms / 1000.0
            deadline = time.monotonic() + timeout_s
            try:
                while self._inflight >= self.max_inflight:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._stats["timed_out"] += 1
                        return IngressDecision(allowed=False, reason="ingress_queue_timeout")
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                self._inflight += 1
                self._stats["accepted"] += 1
                return IngressDecision(
                    allowed=True,
                    queue_wait_ms=max(0.0, (time.monotonic() - start) * 1000.0),
                )
            finally:
                self._queued = max(0, self._queued - 1)

    async def release(self) -> None:
        if not self.enabled:
            return
        async with self._cond:
            self._inflight = max(0, self._inflight - 1)
            self._stats["released"] += 1
            self._cond.notify(1)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_inflight": self.max_inflight,
            "max_queue": self.max_queue,
            "queue_wait_timeout_ms": self.queue_wait_timeout_ms,
            "inflight": self._inflight,
            "queued": self._queued,
            "stats": dict(self._stats),
        }

