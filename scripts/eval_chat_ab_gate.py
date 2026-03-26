#!/usr/bin/env python3
"""
Gate A/B chat benchmark reports with explicit pass/latency constraints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("report must be a JSON object")
    return payload


def _summary_for(report: dict[str, Any], label: str) -> dict[str, Any]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return {}
    row = summary.get(label, {})
    return row if isinstance(row, dict) else {}


def _fnum(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate chat A/B report metrics.")
    parser.add_argument("--report", required=True, help="Path to ab_model_benchmark JSON report.")
    parser.add_argument("--baseline", required=True, help="Baseline target label from report summary.")
    parser.add_argument("--candidate", required=True, help="Candidate target label from report summary.")
    parser.add_argument("--min-pass-rate-delta", type=float, default=0.0)
    parser.add_argument("--min-avg-score-delta", type=float, default=0.01)
    parser.add_argument("--max-p95-latency-regression-ms", type=float, default=200.0)
    parser.add_argument("--max-avg-latency-regression-ms", type=float, default=80.0)
    args = parser.parse_args()

    report = _load_json(Path(args.report).expanduser().resolve())
    baseline = _summary_for(report, str(args.baseline))
    candidate = _summary_for(report, str(args.candidate))
    if not baseline:
        raise SystemExit(f"missing baseline summary for label={args.baseline}")
    if not candidate:
        raise SystemExit(f"missing candidate summary for label={args.candidate}")

    base_pass = _fnum(((baseline.get("quality", {}) or {}).get("pass_rate")))
    cand_pass = _fnum(((candidate.get("quality", {}) or {}).get("pass_rate")))
    base_score = _fnum(((baseline.get("quality", {}) or {}).get("avg_score")))
    cand_score = _fnum(((candidate.get("quality", {}) or {}).get("avg_score")))
    base_p95 = _fnum(((baseline.get("latency_ms", {}) or {}).get("p95")))
    cand_p95 = _fnum(((candidate.get("latency_ms", {}) or {}).get("p95")))
    base_avg = _fnum(((baseline.get("latency_ms", {}) or {}).get("avg")))
    cand_avg = _fnum(((candidate.get("latency_ms", {}) or {}).get("avg")))

    checks = []
    checks.append(
        {
            "name": "pass_rate_delta",
            "baseline": base_pass,
            "candidate": cand_pass,
            "delta": round(cand_pass - base_pass, 4),
            "threshold": float(args.min_pass_rate_delta),
            "ok": (cand_pass - base_pass) >= float(args.min_pass_rate_delta),
        }
    )
    checks.append(
        {
            "name": "avg_score_delta",
            "baseline": base_score,
            "candidate": cand_score,
            "delta": round(cand_score - base_score, 4),
            "threshold": float(args.min_avg_score_delta),
            "ok": (cand_score - base_score) >= float(args.min_avg_score_delta),
        }
    )
    checks.append(
        {
            "name": "p95_latency_regression_ms",
            "baseline": base_p95,
            "candidate": cand_p95,
            "delta": round(cand_p95 - base_p95, 3),
            "threshold": float(args.max_p95_latency_regression_ms),
            "ok": (cand_p95 - base_p95) <= float(args.max_p95_latency_regression_ms),
        }
    )
    checks.append(
        {
            "name": "avg_latency_regression_ms",
            "baseline": base_avg,
            "candidate": cand_avg,
            "delta": round(cand_avg - base_avg, 3),
            "threshold": float(args.max_avg_latency_regression_ms),
            "ok": (cand_avg - base_avg) <= float(args.max_avg_latency_regression_ms),
        }
    )

    passed = all(bool(c.get("ok")) for c in checks)
    out = {
        "version": 1,
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "passed": passed,
        "checks": checks,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
