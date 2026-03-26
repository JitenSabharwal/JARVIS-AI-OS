#!/usr/bin/env python3
"""
Run A/B benchmark across two model targets with speed and quality scoring.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.response_quality_eval import score_response


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ranked = sorted(float(v) for v in values)
    rank = max(0.0, min(100.0, float(pct))) / 100.0 * (len(ranked) - 1)
    lo = int(rank)
    hi = min(len(ranked) - 1, lo + 1)
    frac = rank - lo
    return float(ranked[lo] + ((ranked[hi] - ranked[lo]) * frac))


@dataclass(slots=True)
class TargetConfig:
    label: str
    kind: str
    url: str
    model: str
    api_key: str
    temperature: float
    max_tokens: int
    system_prompt: str
    timeout_s: float


def _http_json_post(url: str, payload: dict[str, Any], *, api_key: str, timeout_s: float) -> dict[str, Any]:
    raw = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url=url, data=raw, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="ignore")
            return json.loads(body) if body.strip() else {}
    except HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"http_error status={exc.code} body={msg[-800:]}") from exc
    except URLError as exc:
        raise RuntimeError(f"url_error reason={exc.reason}") from exc


def _call_ollama(target: TargetConfig, prompt: str) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": target.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": target.temperature,
            "num_predict": target.max_tokens,
        },
    }
    if target.system_prompt:
        payload["system"] = target.system_prompt
    data = _http_json_post(f"{target.url.rstrip('/')}/api/generate", payload, api_key="", timeout_s=target.timeout_s)
    text = str(data.get("response", "")).strip()
    if not text:
        raise RuntimeError("ollama_empty_response")
    eval_count = int(data.get("eval_count", 0) or 0)
    eval_duration_ns = int(data.get("eval_duration", 0) or 0)
    tok_per_s = 0.0
    if eval_count > 0 and eval_duration_ns > 0:
        tok_per_s = float(eval_count) / (float(eval_duration_ns) / 1_000_000_000.0)
    return text, {"completion_tokens": eval_count, "tokens_per_second": round(tok_per_s, 4)}


def _call_openai_compatible(target: TargetConfig, prompt: str) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": target.model,
        "messages": [],
        "temperature": target.temperature,
        "max_tokens": target.max_tokens,
        "stream": False,
    }
    if target.system_prompt:
        payload["messages"].append({"role": "system", "content": target.system_prompt})
    payload["messages"].append({"role": "user", "content": prompt})
    data = _http_json_post(
        f"{target.url.rstrip('/')}/chat/completions",
        payload,
        api_key=target.api_key,
        timeout_s=target.timeout_s,
    )
    choices = data.get("choices", [])
    msg = {}
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message", {}) or {}
    text = str(msg.get("content", "")).strip()
    if not text:
        raise RuntimeError("openai_compatible_empty_response")
    usage = data.get("usage", {}) if isinstance(data.get("usage"), dict) else {}
    return text, {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _invoke_target(target: TargetConfig, prompt: str) -> tuple[str, dict[str, Any], float]:
    started = time.perf_counter()
    if target.kind == "ollama":
        text, usage = _call_ollama(target, prompt)
    elif target.kind == "openai":
        text, usage = _call_openai_compatible(target, prompt)
    else:
        raise RuntimeError(f"unsupported_target_kind={target.kind}")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return text, usage, round(elapsed_ms, 3)


def _target_summary(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if not r.get("error")]
    latencies = [float(r.get("latency_ms", 0.0) or 0.0) for r in ok_rows]
    scores = [float(r.get("score", 0.0) or 0.0) for r in ok_rows]
    passed = [bool(r.get("passed", False)) for r in ok_rows]
    tps_rows = []
    for row in ok_rows:
        usage = row.get("usage", {}) if isinstance(row.get("usage"), dict) else {}
        if float(usage.get("tokens_per_second", 0.0) or 0.0) > 0.0:
            tps_rows.append(float(usage.get("tokens_per_second", 0.0)))
            continue
        c_tokens = int(usage.get("completion_tokens", 0) or 0)
        latency_ms = float(row.get("latency_ms", 0.0) or 0.0)
        if c_tokens > 0 and latency_ms > 1:
            tps_rows.append(float(c_tokens) / (latency_ms / 1000.0))
    return {
        "label": label,
        "requests_total": len(rows),
        "requests_ok": len(ok_rows),
        "requests_failed": len(rows) - len(ok_rows),
        "quality": {
            "avg_score": round(statistics.fmean(scores), 4) if scores else 0.0,
            "pass_rate": round((sum(1 for p in passed if p) / len(passed)), 4) if passed else 0.0,
        },
        "latency_ms": {
            "avg": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p50": round(_percentile(latencies, 50), 3) if latencies else 0.0,
            "p95": round(_percentile(latencies, 95), 3) if latencies else 0.0,
            "min": round(min(latencies), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "throughput": {
            "completion_tokens_per_second_avg": round(statistics.fmean(tps_rows), 3) if tps_rows else 0.0
        },
    }


def _parse_target(parser: argparse.ArgumentParser, prefix: str, args: argparse.Namespace) -> TargetConfig:
    label = str(getattr(args, f"{prefix}_label")).strip() or prefix.upper()
    kind = str(getattr(args, f"{prefix}_kind")).strip().lower()
    url = str(getattr(args, f"{prefix}_url")).strip()
    model = str(getattr(args, f"{prefix}_model")).strip()
    if not url or not model:
        parser.error(f"--{prefix}-url and --{prefix}-model are required")
    return TargetConfig(
        label=label,
        kind=kind,
        url=url,
        model=model,
        api_key=str(getattr(args, f"{prefix}_api_key") or "").strip(),
        temperature=float(getattr(args, f"{prefix}_temperature")),
        max_tokens=int(getattr(args, f"{prefix}_max_tokens")),
        system_prompt=str(getattr(args, f"{prefix}_system_prompt") or "").strip(),
        timeout_s=float(args.timeout_s),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A/B benchmark for local model endpoints with quality+latency report."
    )
    p.add_argument("--cases", default="config/evals/response_quality_cases.json")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--timeout-s", type=float, default=90.0)
    p.add_argument("--out", default="")

    for prefix in ("a", "b"):
        p.add_argument(f"--{prefix}-label", default=prefix.upper())
        p.add_argument(f"--{prefix}-kind", choices=["ollama", "openai"], required=True)
        p.add_argument(f"--{prefix}-url", required=True)
        p.add_argument(f"--{prefix}-model", required=True)
        p.add_argument(f"--{prefix}-api-key", default="")
        p.add_argument(f"--{prefix}-temperature", type=float, default=0.2)
        p.add_argument(f"--{prefix}-max-tokens", type=int, default=512)
        p.add_argument(f"--{prefix}-system-prompt", default="")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cases_payload = _load_json(Path(args.cases))
    if not isinstance(cases_payload, dict):
        raise SystemExit("cases must be a JSON object with key 'cases'")
    cases = cases_payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise SystemExit("cases['cases'] must be a non-empty list")

    repeats = max(1, int(args.repeats))
    a = _parse_target(parser, "a", args)
    b = _parse_target(parser, "b", args)

    rows_by_target: dict[str, list[dict[str, Any]]] = {a.label: [], b.label: []}
    per_case: list[dict[str, Any]] = []

    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id", "")).strip()
        prompt = str(case.get("prompt", "")).strip()
        expected = case.get("expected_phrases", [])
        if not case_id or not prompt:
            continue
        if not isinstance(expected, list):
            expected = []
        min_words = int(case.get("min_words", 1) or 1)
        max_words = int(case.get("max_words", 220) or 220)
        pass_threshold = float(case.get("pass_threshold", 0.7) or 0.7)

        case_row: dict[str, Any] = {"id": case_id, "prompt": prompt, "runs": []}
        for run_idx in range(repeats):
            run_row: dict[str, Any] = {"repeat": run_idx + 1, "targets": {}}
            for target in (a, b):
                result_row: dict[str, Any] = {}
                try:
                    text, usage, latency_ms = _invoke_target(target, prompt)
                    scored = score_response(
                        text,
                        expected_phrases=[str(x) for x in expected],
                        min_words=min_words,
                        max_words=max_words,
                        pass_threshold=pass_threshold,
                    )
                    result_row = {
                        "latency_ms": latency_ms,
                        "usage": usage,
                        "score": scored.score,
                        "passed": scored.passed,
                        "result": scored.to_dict(),
                        "response": text,
                    }
                except Exception as exc:
                    result_row = {
                        "latency_ms": 0.0,
                        "usage": {},
                        "score": 0.0,
                        "passed": False,
                        "error": str(exc),
                        "response": "",
                    }
                rows_by_target[target.label].append(
                    {
                        "case_id": case_id,
                        "repeat": run_idx + 1,
                        **result_row,
                    }
                )
                run_row["targets"][target.label] = result_row
            case_row["runs"].append(run_row)
        per_case.append(case_row)

    report = {
        "version": 1,
        "meta": {
            "cases_file": str(Path(args.cases).resolve()),
            "repeats": repeats,
            "generated_at_epoch_s": round(time.time(), 3),
        },
        "targets": [
            {
                "label": a.label,
                "kind": a.kind,
                "url": a.url,
                "model": a.model,
                "temperature": a.temperature,
                "max_tokens": a.max_tokens,
            },
            {
                "label": b.label,
                "kind": b.kind,
                "url": b.url,
                "model": b.model,
                "temperature": b.temperature,
                "max_tokens": b.max_tokens,
            },
        ],
        "summary": {
            a.label: _target_summary(a.label, rows_by_target[a.label]),
            b.label: _target_summary(b.label, rows_by_target[b.label]),
        },
        "cases": per_case,
    }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if str(args.out).strip():
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote_report={out}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
