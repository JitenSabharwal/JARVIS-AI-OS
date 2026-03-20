#!/usr/bin/env python3
"""
Synthetic workload generator for scheduling strategy evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic workload traces.")
    parser.add_argument("--tasks", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runtime/evals/load_trace.json")
    args = parser.parse_args()

    random.seed(int(args.seed))
    lanes = ["developer_lane", "analyst_lane", "manager_lane", "verifier_lane"]
    tiers = ["bronze", "standard", "gold"]
    weights = [0.42, 0.34, 0.14, 0.10]
    records = []
    for idx in range(int(args.tasks)):
        lane = random.choices(lanes, weights=weights, k=1)[0]
        tier = random.choices(tiers, weights=[0.45, 0.40, 0.15], k=1)[0]
        base_latency = {
            "developer_lane": 980,
            "analyst_lane": 860,
            "manager_lane": 420,
            "verifier_lane": 560,
        }[lane]
        jitter = random.randint(-150, 220)
        records.append(
            {
                "id": f"t-{idx+1}",
                "lane": lane,
                "sla_tier": tier,
                "arrival_ms": idx * random.randint(15, 45),
                "expected_latency_ms": max(80, base_latency + jitter),
            }
        )
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"version": 1, "tasks": records}, indent=2), encoding="utf-8")
    print(f"wrote_trace={out}")


if __name__ == "__main__":
    main()

