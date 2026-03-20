#!/usr/bin/env python3
"""
Evaluate general response quality for chat/code/workflow style outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.response_quality_eval import evaluate_response_cases


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generic response quality.")
    parser.add_argument("--cases", default="config/evals/response_quality_cases.json")
    parser.add_argument("--responses-file", default="config/evals/response_quality_sample_responses.json")
    parser.add_argument("--min-pass-rate", type=float, default=0.95)
    parser.add_argument("--min-case-score", type=float, default=0.70)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cases_payload = _load_json(Path(args.cases))
    if not isinstance(cases_payload, dict):
        raise SystemExit("cases must be a JSON object with 'cases'")
    cases = cases_payload.get("cases", [])
    if not isinstance(cases, list):
        raise SystemExit("cases['cases'] must be a list")
    responses_payload = _load_json(Path(args.responses_file))
    if not isinstance(responses_payload, dict):
        raise SystemExit("responses-file must be a JSON object")
    responses = {str(k): str(v) for k, v in responses_payload.items()}
    report = evaluate_response_cases(cases, responses)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if str(args.out).strip():
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote_report={out}")
    else:
        print(rendered)

    failures: list[str] = []
    pass_rate = float(report.get("pass_rate", 0.0) or 0.0)
    if pass_rate < float(args.min_pass_rate):
        failures.append(f"pass_rate {pass_rate:.4f} < min-pass-rate {float(args.min_pass_rate):.4f}")
    for case in report.get("cases", []):
        if not isinstance(case, dict):
            continue
        score = float(case.get("score", 0.0) or 0.0)
        case_id = str(case.get("id", "<unknown>"))
        if score < float(args.min_case_score):
            failures.append(f"case '{case_id}' score {score:.4f} < min-case-score {float(args.min_case_score):.4f}")
    if failures:
        for f in failures:
            print(f"GATE_FAIL: {f}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()

