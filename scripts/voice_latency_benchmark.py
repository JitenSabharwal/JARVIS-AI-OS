#!/usr/bin/env python3
"""
Synthetic voice latency benchmark for Sprint 9 Part 2 acceptance.

This runs a configurable number of voice turns through the multimodal callback
path and reports p50/p95/max latency against a target budget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.latency_validation import validate_latency_budget
from interfaces.conversation_manager import ConversationManager
from interfaces.voice_interface import VoiceConfig, VoiceInterface


class BenchmarkVoiceInterface(VoiceInterface):
    def __init__(self, transcripts: list[str], *, config: VoiceConfig | None = None) -> None:
        super().__init__(config=config)
        self._transcripts = transcripts
        self.spoken: list[str] = []

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
        return False


@dataclass
class BenchmarkConfig:
    workers: int
    turns_per_worker: int
    target_p95_ms: float
    processing_delay_ms: float
    jitter_ms: float


def _parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run synthetic voice latency benchmark.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--turns-per-worker", type=int, default=20)
    parser.add_argument("--target-p95-ms", type=float, default=900.0)
    parser.add_argument("--processing-delay-ms", type=float, default=120.0)
    parser.add_argument("--jitter-ms", type=float, default=80.0)
    args = parser.parse_args()
    return BenchmarkConfig(
        workers=max(1, int(args.workers)),
        turns_per_worker=max(1, int(args.turns_per_worker)),
        target_p95_ms=float(args.target_p95_ms),
        processing_delay_ms=max(0.0, float(args.processing_delay_ms)),
        jitter_ms=max(0.0, float(args.jitter_ms)),
    )


async def _run_worker(worker_id: int, cfg: BenchmarkConfig) -> list[float]:
    latencies_ms: list[float] = []
    cm = ConversationManager(
        llm_handler=lambda prompt: asyncio.sleep(0, result=f"ok:{prompt[:32]}")
    )
    session = cm.start_session(user_id=f"bench-worker-{worker_id}")
    transcripts = [f"jarvis benchmark turn {i}" for i in range(cfg.turns_per_worker)]
    vi = BenchmarkVoiceInterface(
        transcripts=transcripts,
        config=VoiceConfig(callback_timeout=10.0, barge_in_enabled=False),
    )

    async def multimodal_cb(payload: dict[str, Any]) -> str:
        started = time.monotonic()
        noise_ms = random.uniform(0.0, cfg.jitter_ms) if cfg.jitter_ms > 0 else 0.0
        delay_ms = cfg.processing_delay_ms + noise_ms
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        response = await cm.process_input(
            session,
            payload.get("text", ""),
            modality=payload.get("modality", "voice"),
            media=payload.get("media", {}) or {},
            context=payload.get("context", {}) or {},
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        latencies_ms.append(elapsed_ms)
        return response

    vi.set_multimodal_callback(multimodal_cb)
    vi.start()
    await vi.listen_and_respond(require_wake_word=True)
    return latencies_ms


async def _run(cfg: BenchmarkConfig) -> dict[str, Any]:
    worker_results = await asyncio.gather(*[_run_worker(i, cfg) for i in range(cfg.workers)])
    all_latencies = [v for worker in worker_results for v in worker]
    result = validate_latency_budget(all_latencies, target_p95_ms=cfg.target_p95_ms)
    return {
        "config": {
            "workers": cfg.workers,
            "turns_per_worker": cfg.turns_per_worker,
            "target_p95_ms": cfg.target_p95_ms,
            "processing_delay_ms": cfg.processing_delay_ms,
            "jitter_ms": cfg.jitter_ms,
        },
        "latency": result.to_dict(),
    }


def main() -> None:
    cfg = _parse_args()
    report = asyncio.run(_run(cfg))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
