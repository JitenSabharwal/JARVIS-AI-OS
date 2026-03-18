#!/usr/bin/env python3
"""
Phase 12/13 evaluation scaffold for domain expansion.

Produces a structured report template and optionally merges observed metrics
from a JSON file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DOMAINS: Dict[str, Dict[str, Any]] = {
    "productivity": {
        "description": "Calendar/task/email productivity workflows",
        "metrics": {
            "followup_on_time_rate": {"target": 0.85, "higher_is_better": True},
            "meeting_summary_quality": {"target": 0.80, "higher_is_better": True},
            "priority_alignment": {"target": 0.75, "higher_is_better": True},
        },
    },
    "finance": {
        "description": "Budgeting, extraction, accounting insights",
        "metrics": {
            "invoice_extraction_f1": {"target": 0.85, "higher_is_better": True},
            "spend_category_f1": {"target": 0.82, "higher_is_better": True},
            "anomaly_false_positive_rate": {"target": 0.10, "higher_is_better": False},
        },
    },
    "support": {
        "description": "Customer support triage and response quality",
        "metrics": {
            "triage_accuracy": {"target": 0.85, "higher_is_better": True},
            "sla_risk_recall": {"target": 0.80, "higher_is_better": True},
            "response_acceptance_rate": {"target": 0.70, "higher_is_better": True},
        },
    },
    "legal": {
        "description": "Legal/compliance extraction and mappings",
        "metrics": {
            "clause_extraction_f1": {"target": 0.80, "higher_is_better": True},
            "control_mapping_precision": {"target": 0.80, "higher_is_better": True},
            "audit_trace_completeness": {"target": 0.90, "higher_is_better": True},
        },
    },
    "analyst": {
        "description": "Spreadsheet/dashboard QA and narrative insights",
        "metrics": {
            "table_qa_accuracy": {"target": 0.80, "higher_is_better": True},
            "insight_usefulness": {"target": 0.75, "higher_is_better": True},
            "trend_detection_recall": {"target": 0.75, "higher_is_better": True},
        },
    },
    "design": {
        "description": "UI/UX critique and multimodal design support",
        "metrics": {
            "spec_completeness": {"target": 0.80, "higher_is_better": True},
            "ui_heuristic_coverage": {"target": 0.75, "higher_is_better": True},
            "edit_instruction_accuracy": {"target": 0.75, "higher_is_better": True},
        },
    },
    "language_voice": {
        "description": "Cross-lingual text/audio understanding and translation",
        "metrics": {
            "translation_quality_proxy": {"target": 0.70, "higher_is_better": True},
            "asr_wer": {"target": 0.20, "higher_is_better": False},
            "fluency_adequacy_review": {"target": 0.75, "higher_is_better": True},
        },
    },
    "self_improvement": {
        "description": "Agent reflection and failure-reduction loop",
        "metrics": {
            "task_success_rate": {"target": 0.90, "higher_is_better": True},
            "retries_per_task": {"target": 0.20, "higher_is_better": False},
            "cost_per_successful_task": {"target": 1.00, "higher_is_better": False},
        },
    },
}


def _load_observed(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _metric_status(observed: float | None, target: float, higher_is_better: bool) -> str:
    if observed is None:
        return "not-run"
    if higher_is_better:
        return "pass" if observed >= target else "fail"
    return "pass" if observed <= target else "fail"


def build_report(selected_domains: List[str], observed: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "version": 1,
        "run_id": run_id,
        "domains": {},
        "summary": {
            "domains_total": len(selected_domains),
            "metrics_total": 0,
            "pass": 0,
            "fail": 0,
            "not_run": 0,
        },
    }

    for domain in selected_domains:
        definition = DOMAINS[domain]
        observed_domain = observed.get(domain, {}) if isinstance(observed, dict) else {}
        if not isinstance(observed_domain, dict):
            observed_domain = {}

        metric_rows: Dict[str, Any] = {}
        for metric_name, config in definition["metrics"].items():
            report["summary"]["metrics_total"] += 1
            raw_value = observed_domain.get(metric_name)
            observed_value = float(raw_value) if isinstance(raw_value, (int, float)) else None
            status = _metric_status(observed_value, float(config["target"]), bool(config["higher_is_better"]))
            if status == "pass":
                report["summary"]["pass"] += 1
            elif status == "fail":
                report["summary"]["fail"] += 1
            else:
                report["summary"]["not_run"] += 1

            metric_rows[metric_name] = {
                "observed": observed_value,
                "target": config["target"],
                "higher_is_better": config["higher_is_better"],
                "status": status,
            }

        report["domains"][domain] = {
            "description": definition["description"],
            "metrics": metric_rows,
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 12/13 domain expansion metrics.")
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="Domain to evaluate (repeatable). Defaults to all domains.",
    )
    parser.add_argument(
        "--observed-metrics",
        default="",
        help="Optional JSON file of observed values keyed by domain/metric.",
    )
    parser.add_argument("--run-id", default="phase12-scaffold", help="Evaluation run identifier")
    parser.add_argument("--out", default="", help="Optional output file path for JSON report")
    args = parser.parse_args()

    requested = [d.strip() for d in args.domain if d.strip()]
    selected_domains = requested if requested else sorted(DOMAINS.keys())
    invalid = [d for d in selected_domains if d not in DOMAINS]
    if invalid:
        raise SystemExit(f"Unknown domain(s): {', '.join(invalid)}")

    observed = (
        _load_observed(Path(args.observed_metrics).expanduser().resolve())
        if str(args.observed_metrics).strip()
        else {}
    )
    report = build_report(selected_domains, observed, run_id=str(args.run_id))

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if str(args.out).strip():
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote_report={out_path}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
