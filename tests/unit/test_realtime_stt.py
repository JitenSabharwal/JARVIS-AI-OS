import base64

import pytest

from infrastructure.realtime_stt import RealtimeSTTService


def _pcm16_b64(samples: list[int]) -> str:
    payload = bytearray()
    for s in samples:
        v = max(-32768, min(32767, int(s)))
        payload.extend(int(v).to_bytes(2, byteorder="little", signed=True))
    return base64.b64encode(bytes(payload)).decode("ascii")


@pytest.mark.asyncio
async def test_realtime_stt_ingest_and_transcribe_resets_buffer() -> None:
    calls: list[tuple[str, str]] = []

    async def _fake_local_transcriber(wav_path: str, language: str) -> str:
        calls.append((wav_path, language))
        return "hello jarvis"

    svc = RealtimeSTTService(local_transcriber=_fake_local_transcriber)
    snap = svc.ingest_pcm16_chunk("s1", pcm16_b64=_pcm16_b64([0, 1000, -1000, 0]), sample_rate=16000)
    assert snap["session_id"] == "s1"
    assert snap["buffered_bytes"] > 0
    text = await svc.transcribe_and_reset_async("s1", language="en-IN")
    assert text == "hello jarvis"
    assert calls and calls[0][1] == "en-IN"
    text2 = await svc.transcribe_and_reset_async("s1", language="en-IN")
    assert text2 == ""


def test_realtime_stt_rejects_invalid_base64() -> None:
    svc = RealtimeSTTService()
    with pytest.raises(ValueError):
        svc.ingest_pcm16_chunk("s2", pcm16_b64="not-base64", sample_rate=16000)
