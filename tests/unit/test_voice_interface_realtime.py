from __future__ import annotations

import asyncio

import pytest

from interfaces.voice_interface import VoiceConfig, VoiceInterface


pytestmark = pytest.mark.asyncio


class _VoiceTestInterface(VoiceInterface):
    def __init__(self, transcripts: list[str | None], config: VoiceConfig | None = None) -> None:
        super().__init__(config=config)
        self._transcripts = transcripts
        self.spoken: list[str] = []
        self.interrupt_called = False

    async def listen_for_speech(self) -> str | None:
        await asyncio.sleep(0)
        if self._transcripts:
            return self._transcripts.pop(0)
        self.stop()
        return None

    async def speak_async(self, text: str) -> None:
        self.spoken.append(text)
        await asyncio.sleep(0)

    def is_speaking(self) -> bool:
        return True

    def stop_speaking(self) -> None:
        self.interrupt_called = True


async def test_voice_loop_barge_in_interrupts_speaking() -> None:
    vi = _VoiceTestInterface(
        transcripts=["jarvis status", None],
        config=VoiceConfig(barge_in_enabled=True, callback_timeout=5.0),
    )
    vi.start()

    async def cb(text: str) -> str:
        return f"ack:{text}"

    await vi.listen_and_respond(cb, require_wake_word=True)
    assert vi.interrupt_called is True
    assert vi.spoken
    assert vi.spoken[0].startswith("ack:")


async def test_voice_loop_timeout_response() -> None:
    vi = _VoiceTestInterface(
        transcripts=["jarvis do slow task", None],
        config=VoiceConfig(callback_timeout=0.05),
    )
    vi.start()

    async def slow_cb(_: str) -> str:
        await asyncio.sleep(0.2)
        return "done"

    await vi.listen_and_respond(slow_cb, require_wake_word=True)
    assert any("too long" in msg.lower() for msg in vi.spoken)


async def test_voice_loop_uses_multimodal_callback_payload() -> None:
    vi = _VoiceTestInterface(
        transcripts=["jarvis summarize this", None],
        config=VoiceConfig(callback_timeout=5.0),
    )
    vi.start()
    seen: dict = {}

    async def mm_cb(payload: dict) -> str:
        seen.update(payload)
        return f"ok:{payload['modality']}:{payload['text']}"

    vi.set_multimodal_callback(mm_cb)
    await vi.listen_and_respond(require_wake_word=True)
    assert seen.get("modality") == "voice"
    assert seen.get("text") == "summarize this"
    assert any(msg.startswith("ok:voice:") for msg in vi.spoken)
