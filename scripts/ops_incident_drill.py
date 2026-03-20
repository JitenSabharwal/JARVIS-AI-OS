#!/usr/bin/env python3
"""
Operational drill runner for resilience/rollback readiness.

This script is safe-by-default and runs dry checks only.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def main() -> int:
    checks: list[dict[str, object]] = []

    # Drill 1: checkpoint file write/read cycle (simulated restart safety).
    with tempfile.TemporaryDirectory(prefix="jarvis_ops_drill_") as td:
        cp = Path(td) / "workflow_checkpoints.json"
        payload = [{"workflow_id": "wf-drill", "status": "running", "next_wave_index": 1}]
        cp.write_text(json.dumps(payload), encoding="utf-8")
        roundtrip = json.loads(cp.read_text(encoding="utf-8"))
        checks.append(
            {
                "name": "checkpoint_roundtrip",
                "ok": isinstance(roundtrip, list) and bool(roundtrip),
                "details": {"path": str(cp)},
            }
        )

    # Drill 2: quality gate assets present.
    required = [
        Path("scripts/eval_repo_quality.py"),
        Path("scripts/eval_response_quality.py"),
        Path("config/evals/repo_quality_cases.json"),
        Path("config/evals/response_quality_cases.json"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    checks.append({"name": "quality_gate_assets", "ok": not missing, "details": {"missing": missing}})

    overall = all(bool(c.get("ok")) for c in checks)
    print(json.dumps({"overall_ok": overall, "checks": checks}, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())

