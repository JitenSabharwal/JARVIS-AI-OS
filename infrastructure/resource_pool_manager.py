"""
Async resource pool manager for CPU/GPU admission control.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class ResourceLease:
    pool: str
    acquired_at: float


class ResourcePoolManager:
    def __init__(self, *, cpu_slots: int, gpu_slots: int, gpu_enabled: bool = True) -> None:
        self.cpu_slots = max(1, int(cpu_slots))
        self.gpu_slots = max(0, int(gpu_slots))
        self.gpu_enabled = bool(gpu_enabled and self.gpu_slots > 0)
        self._cpu_sem = asyncio.Semaphore(self.cpu_slots)
        self._gpu_sem = asyncio.Semaphore(self.gpu_slots if self.gpu_enabled else 0)
        self._in_use: Dict[str, int] = {"cpu": 0, "gpu": 0}
        self._rejected: Dict[str, int] = {"cpu": 0, "gpu": 0}

    @classmethod
    def from_env(cls) -> "ResourcePoolManager":
        cpu = int(os.getenv("JARVIS_POOL_CPU_SLOTS", "6") or 6)
        gpu = int(os.getenv("JARVIS_POOL_GPU_SLOTS", "1") or 1)
        gpu_enabled = str(os.getenv("JARVIS_POOL_GPU_ENABLED", "true")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(cpu_slots=cpu, gpu_slots=gpu, gpu_enabled=gpu_enabled)

    async def acquire(self, pool: str, *, timeout_s: float = 30.0) -> ResourceLease:
        p = "gpu" if str(pool).strip().lower() == "gpu" and self.gpu_enabled else "cpu"
        sem = self._gpu_sem if p == "gpu" else self._cpu_sem
        try:
            await asyncio.wait_for(sem.acquire(), timeout=timeout_s)
        except Exception:
            self._rejected[p] = int(self._rejected.get(p, 0)) + 1
            raise
        self._in_use[p] = int(self._in_use.get(p, 0)) + 1
        return ResourceLease(pool=p, acquired_at=time.monotonic())

    def release(self, lease: ResourceLease | None) -> None:
        if lease is None:
            return
        p = "gpu" if str(getattr(lease, "pool", "cpu")) == "gpu" else "cpu"
        sem = self._gpu_sem if p == "gpu" else self._cpu_sem
        sem.release()
        self._in_use[p] = max(0, int(self._in_use.get(p, 0)) - 1)

    def snapshot(self) -> Dict[str, int]:
        return {
            "cpu_slots": int(self.cpu_slots),
            "gpu_slots": int(self.gpu_slots),
            "gpu_enabled": int(1 if self.gpu_enabled else 0),
            "cpu_in_use": int(self._in_use.get("cpu", 0)),
            "gpu_in_use": int(self._in_use.get("gpu", 0)),
            "cpu_rejected": int(self._rejected.get("cpu", 0)),
            "gpu_rejected": int(self._rejected.get("gpu", 0)),
        }

