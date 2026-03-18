"""
Priority-based async task queue with dependency tracking, retry logic,
dead-letter queue, and a configurable worker pool.
"""

from __future__ import annotations

import asyncio
import heapq
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

from infrastructure.logger import get_logger

logger = get_logger("task_queue")


# ---------------------------------------------------------------------------
# Enums & data structures
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    """Lifecycle states a queued task can be in."""
    PENDING = "pending"
    WAITING_DEPS = "waiting_deps"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD = "dead"            # exhausted all retries
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Convenience aliases that map to integer sort keys (lower = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 5
    LOW = 10


# Callable type for actual task work
TaskCallable = Callable[..., Awaitable[Any]]


@dataclass
class QueuedTask:
    """A single unit of work held by the TaskQueue."""
    id: str
    task: TaskCallable
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    priority: int = TaskPriority.NORMAL.value
    dependencies: list[str] = field(default_factory=list)   # IDs that must complete first
    added_at: float = field(default_factory=time.time)
    scheduled_at: float | None = None                        # earliest run time (for backoff)
    attempts: int = 0
    max_attempts: int = 3
    timeout: float | None = None                             # per-execution timeout
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # heapq comparison: (priority, added_at) so equal-priority tasks are FIFO
    def __lt__(self, other: "QueuedTask") -> bool:
        return (self.priority, self.added_at) < (other.priority, other.added_at)

    def __le__(self, other: "QueuedTask") -> bool:
        return (self.priority, self.added_at) <= (other.priority, other.added_at)

    @property
    def is_ready(self) -> bool:
        """True when the task is past its scheduled_at time."""
        if self.scheduled_at is None:
            return True
        return time.time() >= self.scheduled_at

    @property
    def is_retryable(self) -> bool:
        return self.attempts < self.max_attempts

    def next_retry_delay(self) -> float:
        """Exponential back-off: 2^(attempt-1) seconds, capped at 60 s."""
        return min(60.0, math.pow(2, self.attempts - 1))


@dataclass
class QueueStats:
    """Snapshot of queue state."""
    total: int
    pending: int
    waiting_deps: int
    running: int
    completed: int
    failed: int
    dead: int
    cancelled: int
    dlq_size: int


# ---------------------------------------------------------------------------
# TaskQueue
# ---------------------------------------------------------------------------

class TaskQueue:
    """
    Async priority task queue with:
    - heapq-based priority ordering
    - Dependency graph (tasks only run after their deps complete)
    - Retry with exponential back-off
    - Dead-letter queue (DLQ) for permanently failed tasks
    - Configurable worker-pool concurrency
    """

    def __init__(self, max_workers: int = 5) -> None:
        self._max_workers = max_workers

        # The heap only contains tasks that are PENDING / WAITING_DEPS
        self._heap: list[QueuedTask] = []
        # All tasks ever enqueued (id -> task)
        self._all: dict[str, QueuedTask] = {}
        # Tasks waiting for dependency resolution (id -> task)
        self._waiting: dict[str, QueuedTask] = {}
        # Completed task IDs (for dep checking)
        self._completed_ids: set[str] = set()

        self._dlq: list[QueuedTask] = []

        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._running = False
        self._lock = asyncio.Lock()

        # Completion callbacks: task_id -> list of futures
        self._completion_waiters: dict[str, list[asyncio.Future[QueuedTask]]] = {}

        logger.debug("TaskQueue created (max_workers=%d)", max_workers)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn the background dispatcher loop."""
        if self._running:
            return
        self._running = True
        self._semaphore = asyncio.Semaphore(self._max_workers)
        self._worker_task = asyncio.create_task(self._dispatcher_loop(), name="task_queue_dispatcher")
        logger.info("TaskQueue started (max_workers=%d)", self._max_workers)

    async def stop(self, wait: bool = True) -> None:
        """Stop accepting new work; optionally wait for running tasks to finish."""
        self._running = False
        if wait:
            if self._running_tasks:
                await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("TaskQueue stopped")

    # ------------------------------------------------------------------
    # Enqueue / Dequeue
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        task: TaskCallable,
        *args: Any,
        task_id: str | None = None,
        priority: int = TaskPriority.NORMAL.value,
        dependencies: list[str] | None = None,
        max_attempts: int = 3,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Add a callable to the queue.

        Returns the task ID. The task will not execute until all *dependencies*
        (by task ID) have completed successfully.
        """
        tid = task_id or str(uuid.uuid4())
        deps = list(dependencies or [])

        qt = QueuedTask(
            id=tid,
            task=task,
            args=args,
            kwargs=kwargs,
            priority=priority,
            dependencies=deps,
            max_attempts=max_attempts,
            timeout=timeout,
            metadata=metadata or {},
        )

        async with self._lock:
            if tid in self._all:
                raise ValueError(f"Task ID already exists: {tid}")
            self._all[tid] = qt

            # Check if all deps are already satisfied
            unsatisfied = [d for d in deps if d not in self._completed_ids]
            if unsatisfied:
                qt.status = TaskStatus.WAITING_DEPS
                self._waiting[tid] = qt
                logger.debug("Task %s waiting for deps: %s", tid, unsatisfied)
            else:
                qt.status = TaskStatus.PENDING
                heapq.heappush(self._heap, qt)
                logger.debug("Task %s enqueued (priority=%d)", tid, priority)

        return tid

    async def dequeue(self) -> QueuedTask | None:
        """
        Pop the highest-priority ready task from the queue.

        A task is *ready* when it is past its ``scheduled_at`` time.
        Returns None if the queue is empty or no task is ready yet.
        """
        async with self._lock:
            # Rebuild heap if some tasks aren't ready yet (peek ahead)
            while self._heap:
                qt = self._heap[0]
                if qt.status == TaskStatus.CANCELLED:
                    heapq.heappop(self._heap)
                    continue
                if qt.is_ready:
                    return heapq.heappop(self._heap)
                break   # the soonest task isn't ready; stop
        return None

    async def peek(self) -> QueuedTask | None:
        """Return the next task without removing it."""
        async with self._lock:
            for qt in self._heap:
                if qt.status != TaskStatus.CANCELLED and qt.is_ready:
                    return qt
        return None

    # ------------------------------------------------------------------
    # Completion / failure signalling
    # ------------------------------------------------------------------

    async def mark_completed(self, task_id: str, result: Any = None) -> None:
        """Record a task as successfully completed and unblock dependents."""
        async with self._lock:
            qt = self._all.get(task_id)
            if qt:
                qt.status = TaskStatus.COMPLETED
                qt.result = result
            self._completed_ids.add(task_id)
            await self._release_waiting_tasks(task_id)

        self._notify_waiters(task_id)
        logger.debug("Task %s completed", task_id)

    async def mark_failed(self, task_id: str, error: str = "") -> None:
        """
        Record a task execution failure.

        If retries remain, requeue with exponential back-off.
        Otherwise move to DLQ.
        """
        async with self._lock:
            qt = self._all.get(task_id)
            if qt is None:
                return
            qt.error = error
            if qt.is_retryable:
                delay = qt.next_retry_delay()
                qt.status = TaskStatus.PENDING
                qt.scheduled_at = time.time() + delay
                heapq.heappush(self._heap, qt)
                logger.warning(
                    "Task %s failed (attempt %d/%d); retrying in %.1fs — %s",
                    task_id, qt.attempts, qt.max_attempts, delay, error,
                )
            else:
                qt.status = TaskStatus.DEAD
                self._dlq.append(qt)
                logger.error("Task %s exhausted retries; moved to DLQ — %s", task_id, error)

        self._notify_waiters(task_id)

    async def cancel(self, task_id: str) -> bool:
        """Cancel a pending/waiting task. Returns False if already running or done."""
        async with self._lock:
            qt = self._all.get(task_id)
            if qt is None:
                return False
            if qt.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.DEAD):
                return False
            qt.status = TaskStatus.CANCELLED
            self._waiting.pop(task_id, None)
        logger.info("Task %s cancelled", task_id)
        return True

    # ------------------------------------------------------------------
    # Wait for a specific task
    # ------------------------------------------------------------------

    async def wait_for(self, task_id: str, timeout: float | None = None) -> QueuedTask:
        """
        Await the completion (or failure) of *task_id*.

        Raises ``asyncio.TimeoutError`` if *timeout* elapses first.
        Raises ``KeyError`` if task_id is unknown.
        """
        async with self._lock:
            qt = self._all.get(task_id)
            if qt is None:
                raise KeyError(f"Unknown task: {task_id}")
            if qt.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.DEAD, TaskStatus.CANCELLED):
                return qt
            loop = asyncio.get_event_loop()
            fut: asyncio.Future[QueuedTask] = loop.create_future()
            self._completion_waiters.setdefault(task_id, []).append(fut)

        return await asyncio.wait_for(fut, timeout=timeout)

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> QueueStats:
        """Return a current snapshot of queue state."""
        counts: dict[TaskStatus, int] = {s: 0 for s in TaskStatus}
        for qt in self._all.values():
            counts[qt.status] += 1
        return QueueStats(
            total=len(self._all),
            pending=counts[TaskStatus.PENDING],
            waiting_deps=counts[TaskStatus.WAITING_DEPS],
            running=counts[TaskStatus.RUNNING],
            completed=counts[TaskStatus.COMPLETED],
            failed=counts[TaskStatus.FAILED],
            dead=counts[TaskStatus.DEAD],
            cancelled=counts[TaskStatus.CANCELLED],
            dlq_size=len(self._dlq),
        )

    def get_task(self, task_id: str) -> QueuedTask | None:
        """Look up a task by ID."""
        return self._all.get(task_id)

    def get_dlq(self) -> list[QueuedTask]:
        """Return all dead-letter tasks."""
        return list(self._dlq)

    def clear_dlq(self) -> int:
        """Discard all dead-letter tasks and return the count removed."""
        count = len(self._dlq)
        self._dlq.clear()
        return count

    def queue_size(self) -> int:
        """Number of items in the active heap (pending + retrying)."""
        return len(self._heap)

    # ------------------------------------------------------------------
    # Internal: background dispatcher
    # ------------------------------------------------------------------

    async def _dispatcher_loop(self) -> None:
        """Continuously dequeue and dispatch tasks to the worker pool."""
        while self._running:
            qt = await self.dequeue()
            if qt is None:
                await asyncio.sleep(0.05)
                continue

            assert self._semaphore is not None
            await self._semaphore.acquire()
            qt.status = TaskStatus.RUNNING
            qt.attempts += 1

            worker = asyncio.create_task(
                self._run_task(qt),
                name=f"task_worker_{qt.id}",
            )
            self._running_tasks[qt.id] = worker
            worker.add_done_callback(lambda _t, tid=qt.id: self._running_tasks.pop(tid, None))

    async def _run_task(self, qt: QueuedTask) -> None:
        """Execute *qt* and record the outcome."""
        assert self._semaphore is not None
        try:
            coro = qt.task(*qt.args, **qt.kwargs)
            if qt.timeout:
                result = await asyncio.wait_for(coro, timeout=qt.timeout)
            else:
                result = await coro
            await self.mark_completed(qt.id, result=result)
        except asyncio.CancelledError:
            qt.status = TaskStatus.CANCELLED
        except Exception as exc:  # noqa: BLE001
            await self.mark_failed(qt.id, error=str(exc))
        finally:
            self._semaphore.release()

    # ------------------------------------------------------------------
    # Internal: dependency resolution
    # ------------------------------------------------------------------

    async def _release_waiting_tasks(self, completed_id: str) -> None:
        """
        Check all waiting tasks to see if *completed_id* was their last
        unsatisfied dependency, and if so move them to the heap.

        Must be called while holding ``self._lock``.
        """
        to_release = []
        for tid, qt in self._waiting.items():
            remaining = [d for d in qt.dependencies if d not in self._completed_ids]
            if not remaining:
                to_release.append(tid)

        for tid in to_release:
            qt = self._waiting.pop(tid)
            qt.status = TaskStatus.PENDING
            heapq.heappush(self._heap, qt)
            logger.debug("Task %s deps satisfied; moved to ready queue", tid)

    # ------------------------------------------------------------------
    # Internal: completion waiters
    # ------------------------------------------------------------------

    def _notify_waiters(self, task_id: str) -> None:
        """Resolve any futures waiting on *task_id*."""
        qt = self._all.get(task_id)
        waiters = self._completion_waiters.pop(task_id, [])
        for fut in waiters:
            if not fut.done() and qt:
                fut.set_result(qt)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"<TaskQueue pending={stats.pending} running={stats.running} "
            f"completed={stats.completed} failed={stats.failed} dlq={stats.dlq_size}>"
        )
