"""
Live stream ingest service for realtime multimodal sessions.

This service performs lightweight periodic pulls from a source URL and writes
grounded frame summaries into ConversationManager realtime sessions.
"""

from __future__ import annotations

import asyncio
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

    def set_conversation_manager(self, conversation_manager: Any) -> None:
        self._conversation_manager = conversation_manager

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
        stream_id = f"stream-{uuid.uuid4().hex[:10]}"
        worker = StreamWorker(
            stream_id=stream_id,
            session_id=str(session_id or "").strip(),
            source_type=str(source_type or "http").strip().lower(),
            source_url=str(source_url or "").strip(),
            interval_ms=max(200, min(10000, int(interval_ms or 1500))),
            note=str(note or "").strip(),
            metadata=dict(metadata or {}),
        )
        self._workers[stream_id] = worker
        task = asyncio.create_task(self._run_worker(worker), name=f"stream_ingest:{stream_id}")
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
        logger.info("Stopped stream worker stream_id=%s", sid)
        return snap

    async def _run_worker(self, worker: StreamWorker) -> None:
        cm = self._conversation_manager
        while worker.active:
            worker.last_tick_at = time.time()
            worker.ticks += 1
            try:
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
                    },
                    ts=time.time(),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                worker.errors += 1
                worker.last_error = str(exc)
                logger.warning("Stream worker error stream_id=%s error=%s", worker.stream_id, exc)
            await asyncio.sleep(max(0.2, float(worker.interval_ms) / 1000.0))

