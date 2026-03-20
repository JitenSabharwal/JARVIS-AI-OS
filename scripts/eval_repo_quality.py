#!/usr/bin/env python3
"""
Evaluate `/repo` response quality using reproducible benchmark cases.

Usage examples:
1) Live API mode:
   python3 scripts/eval_repo_quality.py \
     --api-base http://127.0.0.1:8080 \
     --api-token "$JARVIS_API_TOKEN" \
     --workspace /app \
     --depth high \
     --model jarvis-default

2) Offline mode (score known responses):
   python3 scripts/eval_repo_quality.py \
     --responses-file runtime/evals/repo_responses.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

# Ensure project root is importable when running as standalone script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.repo_quality_eval import evaluate_repo_cases


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _chat_completion(
    *,
    api_base: str,
    api_token: str,
    model: str,
    message: str,
    timeout_s: float,
) -> str:
    url = f"{api_base.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": message}],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as res:
        data = json.loads(res.read().decode("utf-8"))
    choices = data.get("choices", []) if isinstance(data, dict) else []
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    return str(msg.get("content", "")).strip() if isinstance(msg, dict) else ""


def _build_repo_prompt(case_prompt: str, *, workspace: str, depth: str, max_files: int) -> str:
    return f"/repo --workspace {workspace} --depth {depth} --max-files {max_files} {case_prompt}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate /repo answer quality.")
    parser.add_argument(
        "--cases",
        default="config/evals/repo_quality_cases.json",
        help="JSON file containing benchmark cases.",
    )
    parser.add_argument("--responses-file", default="", help="Optional JSON map {case_id: response_text}.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8080", help="Jarvis API base URL.")
    parser.add_argument("--api-token", default="", help="Bearer token for Jarvis API.")
    parser.add_argument("--model", default="jarvis-default", help="Model id.")
    parser.add_argument("--workspace", default="/app", help="Workspace path for /repo prompt.")
    parser.add_argument("--depth", default="high", choices=["low", "medium", "high"], help="Repo analysis depth.")
    parser.add_argument("--max-files", type=int, default=180, help="Max files for /repo evaluation prompts.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout seconds.")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="If > 0, fail with non-zero exit when report pass_rate is below this threshold.",
    )
    parser.add_argument(
        "--min-case-score",
        type=float,
        default=0.0,
        help="If > 0, fail when any case score is below this threshold.",
    )
    parser.add_argument("--out", default="", help="Optional output report path.")
    args = parser.parse_args()

    cases_path = Path(args.cases).expanduser().resolve()
    if not cases_path.exists():
        raise SystemExit(f"Cases file not found: {cases_path}")
    cases_payload = _load_json(cases_path)
    if not isinstance(cases_payload, dict):
        raise SystemExit("Cases file must be an object with key 'cases'.")
    cases = cases_payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise SystemExit("Cases file must include non-empty 'cases' list.")

    responses_by_id: dict[str, str] = {}
    if str(args.responses_file).strip():
        responses_path = Path(args.responses_file).expanduser().resolve()
        if not responses_path.exists():
            raise SystemExit(f"responses-file not found: {responses_path}")
        payload = _load_json(responses_path)
        if not isinstance(payload, dict):
            raise SystemExit("responses-file must be JSON object {case_id: response_text}.")
        responses_by_id = {str(k): str(v) for k, v in payload.items()}
    else:
        if not str(args.api_token).strip():
            raise SystemExit("--api-token is required in live API mode.")
        for case in cases:
            case_id = str(case.get("id", "")).strip()
            prompt = str(case.get("prompt", "")).strip()
            if not case_id or not prompt:
                continue
            user_message = _build_repo_prompt(
                prompt,
                workspace=str(args.workspace),
                depth=str(args.depth),
                max_files=int(args.max_files),
            )
            response_text = _chat_completion(
                api_base=str(args.api_base),
                api_token=str(args.api_token),
                model=str(args.model),
                message=user_message,
                timeout_s=float(args.timeout),
            )
            responses_by_id[case_id] = response_text

    report = evaluate_repo_cases(cases, responses_by_id)
    report["meta"] = {
        "api_base": str(args.api_base),
        "model": str(args.model),
        "workspace": str(args.workspace),
        "depth": str(args.depth),
        "max_files": int(args.max_files),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)

    if str(args.out).strip():
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote_report={out_path}")
    else:
        print(rendered)

    # Optional quality gates for CI/release checks.
    failed_reasons: list[str] = []
    min_pass_rate = float(args.min_pass_rate or 0.0)
    if min_pass_rate > 0.0:
        pass_rate = float(report.get("pass_rate", 0.0) or 0.0)
        if pass_rate < min_pass_rate:
            failed_reasons.append(
                f"pass_rate {pass_rate:.4f} is below min-pass-rate {min_pass_rate:.4f}"
            )
    min_case_score = float(args.min_case_score or 0.0)
    if min_case_score > 0.0:
        cases = report.get("cases", [])
        if isinstance(cases, list):
            for case in cases:
                if not isinstance(case, dict):
                    continue
                case_id = str(case.get("id", "")).strip() or "<unknown>"
                score = float(case.get("score", 0.0) or 0.0)
                if score < min_case_score:
                    failed_reasons.append(
                        f"case '{case_id}' score {score:.4f} is below min-case-score {min_case_score:.4f}"
                    )

    if failed_reasons:
        for reason in failed_reasons:
            print(f"GATE_FAIL: {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
