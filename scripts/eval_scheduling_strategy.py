#!/usr/bin/env python3
"""
Compare baseline vs adaptive scheduling strategy on synthetic traces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.strategy_engine import StrategyEngine


def _load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run(trace: Dict[str, Any], *, adaptive: bool) -> Dict[str, Any]:
    tasks = trace.get("tasks", [])
    if not isinstance(tasks, list):
        tasks = []
    lane_pressure: Dict[str, int] = {}
    caps = {"developer_lane": 2, "analyst_lane": 2, "manager_lane": 1, "verifier_lane": 1}
    engine = StrategyEngine(adaptive_enabled=adaptive)
    total_latency = 0.0
    violations = 0
    for t in tasks:
        if not isinstance(t, dict):
            continue
        lane = str(t.get("lane", "developer_lane"))
        expected = float(t.get("expected_latency_ms", 700) or 700)
        lane_pressure[lane] = int(lane_pressure.get(lane, 0)) + 1
        decision = engine.select(lane_caps=caps, lane_pressure=lane_pressure)
        lane_cap = int(decision.lane_caps.get(lane, caps.get(lane, 1)) or 1)
        # Approximate queueing latency with pressure-sensitive overhead.
        queue_penalty = max(0, int(lane_pressure.get(lane, 0)) - lane_cap) * 45.0
        adjusted = (expected / max(1, lane_cap)) + queue_penalty
        total_latency += adjusted
        tier = str(t.get("sla_tier", "standard"))
        sla_target = 900 if tier == "gold" else 1400 if tier == "standard" else 1900
        if adjusted > sla_target:
            violations += 1
        lane_pressure[lane] = max(0, lane_pressure[lane])
    count = max(1, len(tasks))
    return {
        "strategy": "adaptive" if adaptive else "baseline",
        "tasks": len(tasks),
        "avg_latency_ms": round(total_latency / count, 2),
        "sla_violations": int(violations),
        "violation_rate": round(float(violations / count), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs adaptive scheduling strategies.")
    parser.add_argument("--trace", default="runtime/evals/load_trace.json")
    parser.add_argument("--out", default="runtime/evals/strategy_eval_report.json")
    args = parser.parse_args()

    trace = _load(args.trace)
    baseline = _run(trace, adaptive=False)
    adaptive = _run(trace, adaptive=True)
    improvement = {
        "avg_latency_gain_ms": round(float(baseline["avg_latency_ms"] - adaptive["avg_latency_ms"]), 2),
        "sla_violation_delta": int(baseline["sla_violations"] - adaptive["sla_violations"]),
    }
    report = {"version": 1, "baseline": baseline, "adaptive": adaptive, "improvement": improvement}
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote_report={out}")


if __name__ == "__main__":
    main()
