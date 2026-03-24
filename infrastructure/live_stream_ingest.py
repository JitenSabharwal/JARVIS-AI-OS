"""
Live stream ingest service for realtime multimodal sessions.

This service performs lightweight periodic pulls from a source URL and writes
grounded frame summaries into ConversationManager realtime sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from infrastructure.logger import get_logger

logger = get_logger("live_stream_ingest")


@dataclass
class StreamWorker:
    stream_id: str
    session_id: str
    source_type: str
    source_url: str
    interval_ms: int = 1500
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    active: bool = True
    started_at: float = field(default_factory=time.time)
    last_tick_at: float = 0.0
    ticks: int = 0
    errors: int = 0
    last_error: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "session_id": self.session_id,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "interval_ms": int(self.interval_ms),
            "note": self.note,
            "metadata": dict(self.metadata),
            "active": bool(self.active),
            "started_at": float(self.started_at),
            "last_tick_at": float(self.last_tick_at),
            "ticks": int(self.ticks),
            "errors": int(self.errors),
            "last_error": self.last_error,
        }


class LiveStreamIngestService:
    """Background stream workers that keep realtime sessions visually grounded."""

    def __init__(self) -> None:
        self._conversation_manager: Any = None
        self._workers: dict[str, StreamWorker] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        try:
            self._worker_concurrency = max(1, int(os.getenv("JARVIS_STREAM_WORKER_CONCURRENCY", "2") or 2))
        except Exception:
            self._worker_concurrency = 2
        try:
            self._default_interval_ms = max(500, min(10000, int(os.getenv("JARVIS_STREAM_DEFAULT_INTERVAL_MS", "2500") or 2500)))
        except Exception:
            self._default_interval_ms = 2500
        try:
            self._queue_max = max(8, int(os.getenv("JARVIS_STREAM_QUEUE_MAX", "256") or 256))
        except Exception:
            self._queue_max = 256
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self._queue_max)
        self._pending_stream_ids: set[str] = set()
        self._worker_tasks: list[asyncio.Task[Any]] = []
        self._workers_started = False

    def set_conversation_manager(self, conversation_manager: Any) -> None:
        self._conversation_manager = conversation_manager

    async def shutdown(self) -> None:
        for worker in list(self._workers.values()):
            worker.active = False
        for task in list(self._tasks.values()):
            if task and not task.done():
                task.cancel()
        for task in list(self._tasks.values()):
            with contextlib.suppress(Exception):
                await task
        self._tasks.clear()
        self._workers.clear()
        self._pending_stream_ids.clear()
        for task in list(self._worker_tasks):
            if task and not task.done():
                task.cancel()
        for task in list(self._worker_tasks):
            with contextlib.suppress(Exception):
                await task
        self._worker_tasks.clear()
        self._workers_started = False

    async def _ensure_worker_pool(self) -> None:
        if self._workers_started:
            return
        for idx in range(self._worker_concurrency):
            task = asyncio.create_task(self._run_worker_loop(idx + 1), name=f"stream_worker:{idx + 1}")
            self._worker_tasks.append(task)
        self._workers_started = True

    def list_streams(self, *, session_id: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for worker in self._workers.values():
            if session_id and worker.session_id != session_id:
                continue
            out.append(worker.snapshot())
        out.sort(key=lambda x: float(x.get("started_at", 0.0)))
        return out

    async def start_stream(
        self,
        *,
        session_id: str,
        source_type: str,
        source_url: str,
        interval_ms: int = 1500,
        note: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cm = self._conversation_manager
        if cm is None:
            raise RuntimeError("Conversation manager not configured for stream ingest")
        await self._ensure_worker_pool()
        stream_id = f"stream-{uuid.uuid4().hex[:10]}"
        worker = StreamWorker(
            stream_id=stream_id,
            session_id=str(session_id or "").strip(),
            source_type=str(source_type or "http").strip().lower(),
            source_url=str(source_url or "").strip(),
            interval_ms=max(500, min(10000, int(interval_ms or self._default_interval_ms))),
            note=str(note or "").strip(),
            metadata=dict(metadata or {}),
        )
        self._workers[stream_id] = worker
        task = asyncio.create_task(self._run_scheduler(worker), name=f"stream_scheduler:{stream_id}")
        self._tasks[stream_id] = task
        logger.info(
            "Started stream worker stream_id=%s session_id=%s source_type=%s",
            stream_id,
            worker.session_id,
            worker.source_type,
        )
        return worker.snapshot()

    async def stop_stream(self, stream_id: str) -> dict[str, Any] | None:
        sid = str(stream_id or "").strip()
        if not sid:
            return None
        worker = self._workers.get(sid)
        if worker is None:
            return None
        worker.active = False
        task = self._tasks.get(sid)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._tasks.pop(sid, None)
        snap = worker.snapshot()
        self._workers.pop(sid, None)
        self._pending_stream_ids.discard(sid)
        logger.info("Stopped stream worker stream_id=%s", sid)
        return snap

    async def _run_scheduler(self, worker: StreamWorker) -> None:
        while worker.active:
            worker.last_tick_at = time.time()
            worker.ticks += 1
            try:
                if worker.stream_id not in self._pending_stream_ids:
                    self._pending_stream_ids.add(worker.stream_id)
                    try:
                        self._queue.put_nowait(worker.stream_id)
                    except asyncio.QueueFull:
                        self._pending_stream_ids.discard(worker.stream_id)
                        worker.errors += 1
                        worker.last_error = "stream queue is full"
                        logger.warning("Dropped stream tick stream_id=%s reason=queue_full", worker.stream_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                worker.errors += 1
                worker.last_error = str(exc)
                logger.warning("Stream worker error stream_id=%s error=%s", worker.stream_id, exc)
            await asyncio.sleep(max(0.5, float(worker.interval_ms) / 1000.0))

    async def _run_worker_loop(self, worker_num: int) -> None:
        while True:
            stream_id = ""
            got_item = False
            try:
                stream_id = await self._queue.get()
                got_item = True
                worker = self._workers.get(stream_id)
                if worker is None or not worker.active:
                    continue
                cm = self._conversation_manager
                if cm is None:
                    continue
                summary = await cm.summarize_visual_observation(
                    worker.session_id,
                    source=worker.source_type,
                    image_url=worker.source_url,
                    note=worker.note,
                    metadata=worker.metadata,
                )
                cm.ingest_realtime_frame(
                    worker.session_id,
                    source=worker.source_type,
                    summary=summary,
                    metadata={
                        **worker.metadata,
                        "stream_id": worker.stream_id,
                        "source_url": worker.source_url,
                        "source_type": worker.source_type,
                        "stream_worker": worker_num,
                    },
                    ts=time.time(),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                if stream_id:
                    worker = self._workers.get(stream_id)
                    if worker is not None:
                        worker.errors += 1
                        worker.last_error = str(exc)
                logger.warning("Stream worker processing error worker=%s stream_id=%s error=%s", worker_num, stream_id, exc)
            finally:
                if stream_id:
                    self._pending_stream_ids.discard(stream_id)
                if got_item:
                    try:
                        self._queue.task_done()
                    except Exception:
                        pass
