"""
Realtime speech-to-text buffering and transcription helpers.

Default mode is fully local MLX Whisper transcription (no cloud STT).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

from infrastructure.logger import get_logger

logger = get_logger("realtime_stt")


@dataclass
class STTSessionBuffer:
    sample_rate: int = 16000
    sample_width: int = 2
    pcm_bytes: bytearray | None = None

    def __post_init__(self) -> None:
        if self.pcm_bytes is None:
            self.pcm_bytes = bytearray()


class RealtimeSTTService:
    """In-memory audio chunk store + local final transcript generation."""

    def __init__(
        self,
        *,
        max_buffer_seconds: int = 90,
        default_sample_rate: int = 16000,
        engine: str | None = None,
        python_executable: str | None = None,
        runner_module: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float = 90.0,
        local_transcriber: Callable[[str, str], Awaitable[str]] | None = None,
    ) -> None:
        self.max_buffer_seconds = max(10, int(max_buffer_seconds or 90))
        self.default_sample_rate = max(8000, int(default_sample_rate or 16000))
        self._sessions: dict[str, STTSessionBuffer] = {}
        self._local_transcriber = local_transcriber
        self._timeout_seconds = max(5.0, float(timeout_seconds or 90.0))

        cfg_engine = "local_whisper"
        cfg_python = "python3"
        cfg_runner = "mlx_whisper.transcribe"
        cfg_model = "models--mlx-community--whisper-large-v3-turbo"
        try:
            from core.config import get_config

            cfg = get_config()
            cfg_python = str(cfg.model.mlx_command_python or cfg_python).strip() or cfg_python
            cfg_runner = str(cfg.model.mlx_audio_runner_module or cfg_runner).strip() or cfg_runner
            cfg_model = str(cfg.model.mlx_audio_model or cfg_model).strip() or cfg_model
            voice_engine = str(cfg.voice.stt_engine or "").strip().lower()
            if voice_engine in {"whisper", "mlx", "local", "local_whisper"}:
                cfg_engine = "local_whisper"
            elif voice_engine:
                cfg_engine = voice_engine
        except Exception:
            pass

        self.engine = str(engine or os.getenv("JARVIS_REALTIME_STT_ENGINE", cfg_engine)).strip().lower() or "local_whisper"
        self.python_executable = str(python_executable or os.getenv("JARVIS_REALTIME_STT_PYTHON", cfg_python)).strip() or "python3"
        self.runner_module = str(runner_module or os.getenv("JARVIS_REALTIME_STT_RUNNER", cfg_runner)).strip() or "mlx_whisper.transcribe"
        self.model_name = str(model_name or os.getenv("JARVIS_REALTIME_STT_MODEL", cfg_model)).strip() or cfg_model

    def _session(self, session_id: str) -> STTSessionBuffer:
        key = str(session_id).strip()
        if not key:
            raise ValueError("session_id is required")
        return self._sessions.setdefault(key, STTSessionBuffer(sample_rate=self.default_sample_rate))

    def reset(self, session_id: str) -> None:
        key = str(session_id).strip()
        if not key:
            return
        self._sessions.pop(key, None)

    def ingest_pcm16_chunk(
        self,
        session_id: str,
        *,
        pcm16_b64: str,
        sample_rate: int | None = None,
    ) -> dict[str, Any]:
        sess = self._session(session_id)
        encoded = str(pcm16_b64 or "").strip()
        if not encoded:
            raise ValueError("pcm16_b64 is required")
        try:
            chunk = base64.b64decode(encoded, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid base64 audio chunk: {exc}") from exc
        if not chunk:
            raise ValueError("Decoded audio chunk is empty")
        if len(chunk) % sess.sample_width != 0:
            raise ValueError("PCM16 chunk byte length must be even")
        if sample_rate is not None:
            try:
                sess.sample_rate = max(8000, min(96000, int(sample_rate)))
            except Exception:
                sess.sample_rate = self.default_sample_rate
        sess.pcm_bytes.extend(chunk)
        max_bytes = int(self.max_buffer_seconds * sess.sample_rate * sess.sample_width)
        if len(sess.pcm_bytes) > max_bytes:
            sess.pcm_bytes = sess.pcm_bytes[-max_bytes:]
        buffered_ms = int((len(sess.pcm_bytes) / (sess.sample_rate * sess.sample_width)) * 1000)
        return {
            "session_id": session_id,
            "sample_rate": sess.sample_rate,
            "buffered_bytes": len(sess.pcm_bytes),
            "buffered_ms": buffered_ms,
        }

    async def transcribe_and_reset_async(self, session_id: str, *, language: str = "en-IN") -> str:
        sess = self._session(session_id)
        raw = bytes(sess.pcm_bytes or b"")
        self.reset(session_id)
        if not raw:
            return ""
        if self.engine != "local_whisper":
            raise RuntimeError(f"Unsupported realtime STT engine for local mode: {self.engine}")
        return await self._transcribe_local_whisper(raw, sample_rate=sess.sample_rate, language=language)

    async def _transcribe_local_whisper(self, raw_pcm16: bytes, *, sample_rate: int, language: str) -> str:
        wav_path = await self._write_temp_wav(raw_pcm16, sample_rate=sample_rate)
        try:
            if self._local_transcriber is not None:
                out = await self._local_transcriber(wav_path, language)
                return str(out or "").strip()
            cmd = [
                self.python_executable,
                "-m",
                self.runner_module,
                "--model",
                self._resolve_mlx_model_arg(self.model_name),
                wav_path,
            ]
            # mlx_whisper.transcribe supports --language in current releases;
            # keep best-effort behavior if a local install ignores it.
            if language:
                cmd.extend(["--language", language.split("-")[0].lower()])
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout_seconds)
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.communicate()
                raise RuntimeError("Local STT timed out") from exc
            if proc.returncode != 0:
                err = (stderr.decode("utf-8", "ignore") or stdout.decode("utf-8", "ignore")).strip()
                raise RuntimeError(f"Local STT failed: {err[:400]}")
            text = self._normalize_transcript_output(stdout.decode("utf-8", "ignore"))
            return text
        finally:
            with contextlib.suppress(Exception):
                Path(wav_path).unlink(missing_ok=True)

    async def _write_temp_wav(self, raw_pcm16: bytes, *, sample_rate: int) -> str:
        def _write() -> str:
            fd, path = tempfile.mkstemp(prefix="jarvis_rt_", suffix=".wav")
            os.close(fd)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(int(sample_rate))
                wf.writeframes(raw_pcm16)
            return path

        return await asyncio.to_thread(_write)

    @staticmethod
    def _resolve_mlx_model_arg(model_name: str) -> str:
        text = str(model_name or "").strip()
        if text.startswith("models--"):
            return text.replace("models--", "", 1).replace("--", "/", 1)
        return text

    @staticmethod
    def _normalize_transcript_output(output: str) -> str:
        lines = [ln.strip() for ln in str(output or "").splitlines() if ln.strip()]
        if not lines:
            return ""
        for line in reversed(lines):
            lower = line.lower()
            if "transcribing" in lower or "loaded model" in lower or lower.startswith("peak memory"):
                continue
            if lower.startswith("text:"):
                return line.split(":", 1)[1].strip()
            return line.strip()
        return lines[-1].strip()
