from __future__ import annotations

import asyncio

import pytest

from infrastructure.live_stream_ingest import LiveStreamIngestService


pytestmark = pytest.mark.asyncio


class _CMStub:
    def __init__(self) -> None:
        self.summaries: list[dict] = []
        self.frames: list[dict] = []

    async def summarize_visual_observation(
        self,
        session_id: str,
        *,
        source: str = "camera",
        image_url: str = "",
        image_b64: str = "",
        note: str = "",
        metadata: dict | None = None,
    ) -> str:
        self.summaries.append(
            {
                "session_id": session_id,
                "source": source,
                "image_url": image_url,
                "image_b64": image_b64,
                "note": note,
                "metadata": metadata or {},
            }
        )
        return "Frame summary"

    def ingest_realtime_frame(
        self,
        session_id: str,
        *,
        source: str,
        summary: str,
        metadata: dict | None = None,
        ts: float | None = None,
    ) -> dict:
        item = {
            "session_id": session_id,
            "source": source,
            "summary": summary,
            "metadata": metadata or {},
            "ts": ts,
        }
        self.frames.append(item)
        return item


async def test_live_stream_ingest_start_and_stop_worker() -> None:
    cm = _CMStub()
    svc = LiveStreamIngestService()
    svc.set_conversation_manager(cm)

    stream = await svc.start_stream(
        session_id="s1",
        source_type="rtsp",
        source_url="rtsp://example.com/live",
        interval_ms=200,
        note="camera stream",
        metadata={"cam": "iphone"},
    )
    assert stream["active"] is True
    assert stream["source_type"] == "rtsp"

    await asyncio.sleep(0.45)
    assert len(cm.frames) >= 1

    stopped = await svc.stop_stream(stream["stream_id"])
    assert stopped is not None
    assert stopped["stream_id"] == stream["stream_id"]

