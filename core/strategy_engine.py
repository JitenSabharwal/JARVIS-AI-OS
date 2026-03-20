"""
Adaptive scheduling strategy selection.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class StrategyDecision:
    strategy_id: str
    lane_priority: Dict[str, int] = field(default_factory=dict)
    lane_caps: Dict[str, int] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)


class StrategyEngine:
    def __init__(
        self,
        *,
        adaptive_enabled: bool = False,
        state_path: str = "data/strategy_state.json",
        persist_every: int = 10,
    ) -> None:
        self.adaptive_enabled = bool(adaptive_enabled)
        self.state_path = str(state_path or "data/strategy_state.json").strip()
        self.persist_every = max(1, int(persist_every or 1))
        self._ema_wait_ms: float = 0.0
        self._ema_sla_violations: float = 0.0
        self._samples: int = 0
        self._updates_since_persist = 0
        self._load_state()

    @classmethod
    def from_env(cls) -> "StrategyEngine":
        adaptive = str(os.getenv("JARVIS_ADAPTIVE_STRATEGY_ENABLED", "false")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            adaptive_enabled=adaptive,
            state_path=str(os.getenv("JARVIS_ADAPTIVE_STRATEGY_STATE_PATH", "data/strategy_state.json")),
            persist_every=int(os.getenv("JARVIS_ADAPTIVE_STRATEGY_PERSIST_EVERY", "10") or 10),
        )

    def select(
        self,
        *,
        lane_caps: Dict[str, int],
        lane_pressure: Dict[str, int] | None = None,
    ) -> StrategyDecision:
        pressure = lane_pressure or {}
        if not self.adaptive_enabled:
            return StrategyDecision(
                strategy_id="baseline",
                lane_caps=dict(lane_caps),
                lane_priority={},
                tags={"mode": "baseline"},
            )
        # Adaptive heuristic using pressure + recent execution feedback.
        ordered = sorted(
            set(list(lane_caps.keys()) + list(pressure.keys())),
            key=lambda lane: (-int(pressure.get(lane, 0) or 0), lane),
        )
        priority = {lane: idx * 10 for idx, lane in enumerate(ordered, start=1)}
        adaptive_caps = dict(lane_caps)
        hot_mode = self._ema_wait_ms >= 450.0 or self._ema_sla_violations >= 0.08
        for lane in ordered:
            p = int(pressure.get(lane, 0) or 0)
            if p >= 3 or (hot_mode and p >= 2):
                adaptive_caps[lane] = max(1, int(adaptive_caps.get(lane, 1) or 1) + 1)
        return StrategyDecision(
            strategy_id="adaptive",
            lane_caps=adaptive_caps,
            lane_priority=priority,
            tags={
                "mode": "adaptive",
                "high_pressure_lanes": [l for l in ordered if int(pressure.get(l, 0) or 0) >= 3],
                "ema_wait_ms": round(self._ema_wait_ms, 2),
                "ema_sla_violations": round(self._ema_sla_violations, 4),
                "hot_mode": hot_mode,
            },
        )

    def feedback(self, *, wait_ms: float, sla_violated: bool) -> None:
        self._samples += 1
        alpha = 0.12
        w = max(0.0, float(wait_ms or 0.0))
        v = 1.0 if bool(sla_violated) else 0.0
        if self._samples == 1:
            self._ema_wait_ms = w
            self._ema_sla_violations = v
            self._persist_state()
            return
        self._ema_wait_ms = ((1.0 - alpha) * self._ema_wait_ms) + (alpha * w)
        self._ema_sla_violations = ((1.0 - alpha) * self._ema_sla_violations) + (alpha * v)
        self._updates_since_persist += 1
        if self._updates_since_persist >= self.persist_every:
            self._persist_state()
            self._updates_since_persist = 0

    def _load_state(self) -> None:
        p = Path(self.state_path)
        if not p.exists():
            return
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            self._ema_wait_ms = float(payload.get("ema_wait_ms", 0.0) or 0.0)
            self._ema_sla_violations = float(payload.get("ema_sla_violations", 0.0) or 0.0)
            self._samples = int(payload.get("samples", 0) or 0)
        except Exception:
            return

    def _persist_state(self) -> None:
        p = Path(self.state_path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(
                    {
                        "ema_wait_ms": self._ema_wait_ms,
                        "ema_sla_violations": self._ema_sla_violations,
                        "samples": self._samples,
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
        except Exception:
            return
