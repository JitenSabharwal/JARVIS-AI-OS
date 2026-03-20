#!/usr/bin/env python3
"""
Sprint 9 release-readiness checker.

Runs a small set of deterministic checks and prints a JSON report.
Exit code is non-zero when any check fails.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run_compile_check() -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "compileall", "-q", "core", "infrastructure", "interfaces", "agents", "memory", "tests"]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    ok = proc.returncode == 0
    return {
        "name": "compileall",
        "ok": ok,
        "details": proc.stdout.strip() if ok else (proc.stderr.strip() or proc.stdout.strip()),
    }


def _check_required_docs() -> Dict[str, Any]:
    required = [
        PROJECT_ROOT / "DEVELOPMENT_PLAN.md",
        PROJECT_ROOT / "docs" / "SPRINT9_PART3_OPERATIONS.md",
    ]
    missing = [str(p) for p in required if not p.exists()]
    return {
        "name": "required_docs",
        "ok": not missing,
        "details": {"missing": missing},
    }


def _check_api_routes_declared() -> Dict[str, Any]:
    api_file = PROJECT_ROOT / "interfaces" / "api_interface.py"
    text = api_file.read_text(encoding="utf-8")
    required_fragments = [
        "/api/v1/connectors/health",
        "/api/v1/connectors/{connector_name}/health",
        "/api/v1/automation/dead-letters/replay",
        "/api/v1/automation/dead-letters/{dead_letter_id}/resolve",
    ]
    missing = [frag for frag in required_fragments if frag not in text]
    return {
        "name": "api_routes_declared",
        "ok": not missing,
        "details": {"missing": missing},
    }


def _check_quality_gate_assets() -> Dict[str, Any]:
    required = [
        PROJECT_ROOT / "scripts" / "eval_repo_quality.py",
        PROJECT_ROOT / "scripts" / "eval_response_quality.py",
        PROJECT_ROOT / "scripts" / "simulate_load.py",
        PROJECT_ROOT / "scripts" / "eval_scheduling_strategy.py",
        PROJECT_ROOT / "scripts" / "ops_incident_drill.py",
        PROJECT_ROOT / "config" / "evals" / "repo_quality_cases.json",
        PROJECT_ROOT / "config" / "evals" / "response_quality_cases.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    return {
        "name": "quality_gate_assets",
        "ok": not missing,
        "details": {"missing": missing},
    }


def _check_lane_cap_env_contract() -> Dict[str, Any]:
    env_template = PROJECT_ROOT / "config" / "env_template.env"
    if not env_template.exists():
        return {"name": "lane_cap_env_contract", "ok": False, "details": {"missing": [str(env_template)]}}
    text = env_template.read_text(encoding="utf-8")
    required = [
        "JARVIS_RESEARCH_LANGGRAPH_ENABLED",
        "JARVIS_RESEARCH_LANGGRAPH_MAX_WAVE_SIZE",
        "JARVIS_AGENT_WORKFLOW_LANE_CAPS",
        "JARVIS_AGENT_WORKFLOW_LANE_PRIORITY",
        "JARVIS_AGENT_WORKFLOW_STEP_MAX_RETRIES",
        "JARVIS_AGENT_WORKFLOW_STEP_RESULT_CONTRACT_STRICT",
        "JARVIS_AGENT_TASK_PAYLOAD_CONTRACT_STRICT",
        "JARVIS_AGENT_WORKFLOW_CHECKPOINT_BACKEND",
        "JARVIS_AGENT_WORKFLOW_CHECKPOINT_PATH",
        "JARVIS_POLICY_COST_ENABLED",
        "JARVIS_POLICY_COST_LEDGER_PATH",
        "JARVIS_ADAPTIVE_STRATEGY_ENABLED",
        "JARVIS_POOL_CPU_SLOTS",
        "JARVIS_POOL_GPU_SLOTS",
        "JARVIS_POOL_GPU_ENABLED",
    ]
    missing = [k for k in required if k not in text]
    return {
        "name": "lane_cap_env_contract",
        "ok": not missing,
        "details": {"missing": missing},
    }


async def _runtime_sanity() -> Dict[str, Any]:
    from infrastructure.automation import AutomationEngine
    from infrastructure.builtin_connectors import build_default_connector_registry

    checks: List[str] = []
    registry = build_default_connector_registry(str(PROJECT_ROOT / "data"))
    health = await registry.health_all()
    if {"calendar", "mail", "files_notifications"}.issubset(set(health.keys())):
        checks.append("connectors_registered")

    engine = AutomationEngine()

    async def always_fail(_: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("forced")

    engine.register_action("always_fail", always_fail)
    engine.create_rule(name="fail", event_type="evt", action_name="always_fail")
    await engine.process_event("evt", {"x": 1})
    dead = engine.get_dead_letters(limit=1)
    if dead:
        dead_letter_id = str(dead[-1]["dead_letter_id"])
        await engine.replay_dead_letter(dead_letter_id)
        resolved = engine.resolve_dead_letter(dead_letter_id, reason="release_check")
        if resolved.get("resolved"):
            checks.append("dead_letter_replay_resolve")

    ok = {"connectors_registered", "dead_letter_replay_resolve"}.issubset(set(checks))
    return {
        "name": "runtime_sanity",
        "ok": ok,
        "details": {"checks": checks},
    }


def main() -> int:
    results: List[Dict[str, Any]] = []
    results.append(_run_compile_check())
    results.append(_check_required_docs())
    results.append(_check_api_routes_declared())
    results.append(_check_quality_gate_assets())
    results.append(_check_lane_cap_env_contract())
    results.append(asyncio.run(_runtime_sanity()))

    overall_ok = all(bool(r.get("ok")) for r in results)
    report = {"overall_ok": overall_ok, "checks": results}
    print(json.dumps(report, indent=2))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
