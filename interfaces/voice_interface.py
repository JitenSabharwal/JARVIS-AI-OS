"""
Voice interface for JARVIS AI OS.

Supports microphone-based speech recognition (via speech_recognition) and
text-to-speech output (via pyttsx3).  Both libraries are optional — the class
degrades gracefully when they are not installed.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SR_AVAILABLE = False

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYTTSX3_AVAILABLE = False

from infrastructure.logger import get_logger

logger = get_logger("voice_interface")

# Conversation callback: async fn(text: str) -> str (or None to skip response)
ConversationCallback = Callable[[str], Awaitable[str | None]]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VoiceConfig:
    """Configuration for the voice interface."""
    sample_rate: int = 16_000
    language: str = "en-US"
    tts_rate: int = 175           # words per minute
    tts_volume: float = 0.9       # 0.0 – 1.0
    wake_word: str = "jarvis"
    timeout: float = 5.0          # seconds to wait for speech start
    phrase_time_limit: float = 15.0  # max seconds per utterance
    energy_threshold: int = 300   # ambient noise level
    dynamic_energy: bool = True   # auto-adjust energy threshold
    pause_threshold: float = 0.8  # silence (s) that ends a phrase


# ---------------------------------------------------------------------------
# VoiceInterface
# ---------------------------------------------------------------------------

class VoiceInterface:
    """
    Voice I/O layer.

    When audio libraries are unavailable the interface still works in a
    degraded text-passthrough mode so the rest of the system is unaffected.
    """

    def __init__(self, config: VoiceConfig | None = None) -> None:
        self.config = config or VoiceConfig()
        self._running = False
        self._conversation_callback: ConversationCallback | None = None
        self._tts_engine: Any = None          # pyttsx3 engine (if available)
        self._recognizer: Any = None          # speech_recognition.Recognizer
        self._microphone: Any = None          # speech_recognition.Microphone
        self._tts_queue: queue.Queue[str | None] = queue.Queue()
        self._tts_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        if _SR_AVAILABLE:
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = self.config.energy_threshold
            self._recognizer.dynamic_energy_threshold = self.config.dynamic_energy
            self._recognizer.pause_threshold = self.config.pause_threshold
        if _PYTTSX3_AVAILABLE:
            self._init_tts()

        logger.info(
            "VoiceInterface created (sr=%s tts=%s wake_word='%s')",
            _SR_AVAILABLE, _PYTTSX3_AVAILABLE, self.config.wake_word,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Initialise the TTS background thread and mark as running."""
        if self._running:
            return
        self._running = True
        self._loop = loop or asyncio.get_event_loop()
        if _PYTTSX3_AVAILABLE and self._tts_engine:
            self._tts_thread = threading.Thread(
                target=self._tts_worker, daemon=True, name="tts_worker"
            )
            self._tts_thread.start()
        logger.info("VoiceInterface started")

    def stop(self) -> None:
        """Stop the TTS worker and clean up."""
        self._running = False
        self._tts_queue.put(None)       # sentinel to stop worker
        if self._tts_thread and self._tts_thread.is_alive():
            self._tts_thread.join(timeout=3)
        logger.info("VoiceInterface stopped")

    # ------------------------------------------------------------------
    # Capability check
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return True when at least speech recognition is usable."""
        if not _SR_AVAILABLE:
            return False
        try:
            sr.Microphone()
            return True
        except Exception:  # noqa: BLE001
            return False

    def has_tts(self) -> bool:
        return _PYTTSX3_AVAILABLE and self._tts_engine is not None

    # ------------------------------------------------------------------
    # Speech recognition
    # ------------------------------------------------------------------

    async def listen_for_speech(self) -> str | None:
        """
        Capture one utterance from the microphone and return its transcription.

        Returns None if recognition is unavailable, no speech is detected, or
        recognition fails.  Non-blocking: runs in an executor thread.
        """
        if not _SR_AVAILABLE:
            logger.debug("Speech recognition not available; returning None")
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._blocking_listen)

    def _blocking_listen(self) -> str | None:
        """Blocking speech capture — runs in a thread pool executor."""
        try:
            mic = sr.Microphone(sample_rate=self.config.sample_rate)
            with mic as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.debug("Listening for speech…")
                audio = self._recognizer.listen(
                    source,
                    timeout=self.config.timeout,
                    phrase_time_limit=self.config.phrase_time_limit,
                )
            text = self._recognizer.recognize_google(
                audio, language=self.config.language
            )
            logger.info("Heard: '%s'", text)
            return str(text)
        except sr.WaitTimeoutError:
            logger.debug("Listen timeout — no speech detected")
        except sr.UnknownValueError:
            logger.debug("Could not understand audio")
        except sr.RequestError as exc:
            logger.warning("Speech recognition API error: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("listen_for_speech error: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Wake-word detection
    # ------------------------------------------------------------------

    def detect_wake_word(self, text: str) -> bool:
        """Return True when *text* contains the configured wake word."""
        return self.config.wake_word.lower() in text.lower()

    # ------------------------------------------------------------------
    # Text-to-speech
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """
        Speak *text* aloud.

        Uses pyttsx3 via a background worker when available; falls back to
        printing to stdout so the interface remains functional.
        """
        if not text:
            return
        if _PYTTSX3_AVAILABLE and self._tts_engine and self._running:
            self._tts_queue.put(text)
        else:
            print(f"[JARVIS]: {text}")

    async def speak_async(self, text: str) -> None:
        """Async wrapper that offloads speech to the executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.speak, text)

    # ------------------------------------------------------------------
    # Main interaction loop
    # ------------------------------------------------------------------

    async def listen_and_respond(
        self,
        callback: ConversationCallback | None = None,
        *,
        require_wake_word: bool = True,
    ) -> None:
        """
        Continuous listen-then-respond loop.

        - Listens for speech.
        - Optionally checks for the wake word.
        - Passes the transcript to *callback* (or the stored callback).
        - Speaks the returned response.

        Runs until ``stop()`` is called.
        """
        cb = callback or self._conversation_callback
        if cb is None:
            raise ValueError("No conversation callback set. Pass one to listen_and_respond().")

        logger.info(
            "Voice loop started (wake_word_required=%s, wake_word='%s')",
            require_wake_word, self.config.wake_word,
        )
        while self._running:
            text = await self.listen_for_speech()
            if text is None:
                await asyncio.sleep(0.1)
                continue

            if require_wake_word and not self.detect_wake_word(text):
                logger.debug("Wake word not detected in: '%s'", text)
                continue

            # Strip wake word from input before forwarding
            clean = text.lower().replace(self.config.wake_word.lower(), "").strip()
            if not clean:
                continue

            try:
                response = await cb(clean)
                if response:
                    await self.speak_async(response)
            except Exception as exc:  # noqa: BLE001
                logger.error("Conversation callback error: %s", exc)
                await self.speak_async("I encountered an error processing your request.")

    def set_conversation_callback(self, callback: ConversationCallback) -> None:
        """Store a default callback used by ``listen_and_respond``."""
        self._conversation_callback = callback

    # ------------------------------------------------------------------
    # TTS engine helpers
    # ------------------------------------------------------------------

    def _init_tts(self) -> None:
        """Initialise the pyttsx3 engine in the current thread."""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.config.tts_rate)
            engine.setProperty("volume", self.config.tts_volume)
            self._tts_engine = engine
        except Exception as exc:  # noqa: BLE001
            logger.warning("pyttsx3 init failed: %s", exc)
            self._tts_engine = None

    def _tts_worker(self) -> None:
        """Background thread that serialises TTS calls."""
        while True:
            text = self._tts_queue.get()
            if text is None:            # stop sentinel
                break
            try:
                if self._tts_engine:
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TTS error: %s", exc)
                print(f"[JARVIS]: {text}")

    def set_voice(self, voice_id: str) -> bool:
        """Select a specific TTS voice by ID. Returns False if TTS unavailable."""
        if not self._tts_engine:
            return False
        try:
            self._tts_engine.setProperty("voice", voice_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not set voice: %s", exc)
            return False

    def list_voices(self) -> list[dict[str, str]]:
        """Return available TTS voices as a list of dicts."""
        if not self._tts_engine:
            return []
        try:
            return [
                {"id": v.id, "name": v.name, "languages": str(v.languages)}
                for v in self._tts_engine.getProperty("voices")
            ]
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------
    # Debug / test helpers
    # ------------------------------------------------------------------

    async def say_and_listen(self, prompt: str) -> str | None:
        """Speak a *prompt*, then listen and return the user's response."""
        await self.speak_async(prompt)
        await asyncio.sleep(0.5)       # brief gap so TTS finishes
        return await self.listen_for_speech()

    def __repr__(self) -> str:
        return (
            f"<VoiceInterface running={self._running} "
            f"sr={_SR_AVAILABLE} tts={self.has_tts()} "
            f"wake_word='{self.config.wake_word}'>"
        )
