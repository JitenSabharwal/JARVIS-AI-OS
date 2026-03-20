from __future__ import annotations

import pytest

from infrastructure.resource_pool_manager import ResourcePoolManager


@pytest.mark.asyncio
async def test_resource_pool_manager_acquire_release() -> None:
    mgr = ResourcePoolManager(cpu_slots=1, gpu_slots=1, gpu_enabled=True)
    lease = await mgr.acquire("cpu", timeout_s=1.0)
    snap = mgr.snapshot()
    assert snap["cpu_in_use"] == 1
    mgr.release(lease)
    assert mgr.snapshot()["cpu_in_use"] == 0
