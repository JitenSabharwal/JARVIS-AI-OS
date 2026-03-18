#!/usr/bin/env python3
"""
Reference delivery smoke runner for Phase 9 acceptance.

Runs two end-to-end API-driven delivery flows:
1) backend template
2) frontend template

Each flow performs bootstrap + run-release and reports JSON output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _post_json(
    *,
    api_base: str,
    path: str,
    payload: Dict[str, Any],
    timeout_seconds: float,
    auth_token: str | None,
) -> Dict[str, Any]:
    url = api_base.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if auth_token:
        req.add_header("Authorization", f"Bearer {auth_token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            text = resp.read().decode("utf-8")
            return json.loads(text)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except Exception:  # noqa: BLE001
            data = {"success": False, "error": raw}
        return {
            "success": False,
            "http_status": exc.code,
            "data": data,
        }
    except urllib.error.URLError as exc:
        return {
            "success": False,
            "error": f"request_failed:{exc}",
        }


def _run_flow(
    *,
    api_base: str,
    auth_token: str | None,
    timeout_seconds: float,
    name: str,
    template_id: str,
    stack: str,
    deploy_target: str,
    real_gates: bool,
) -> Dict[str, Any]:
    suffix = int(time.time())
    project_name = f"{name}-{suffix}"

    bootstrap_resp = _post_json(
        api_base=api_base,
        path="/api/v1/delivery/bootstrap",
        payload={
            "template_id": template_id,
            "project_name": project_name,
            "cloud_target": "local",
        },
        timeout_seconds=timeout_seconds,
        auth_token=auth_token,
    )

    context: Dict[str, Any] = {
        "auto_gate_commands": True,
        "stack": stack,
    }
    if real_gates:
        context["deploy"] = {"success": True}
    else:
        ok_cmd = ["python3", "-c", "raise SystemExit(0)"]
        context["gate_command_overrides"] = {
            "lint": ok_cmd,
            "test": ok_cmd,
            "sast": ok_cmd,
            "dependency_audit": ok_cmd,
        }
        context["deploy_commands"] = {
            deploy_target: ok_cmd,
        }

    run_resp = _post_json(
        api_base=api_base,
        path="/api/v1/delivery/releases/run",
        payload={
            "project_name": project_name,
            "profile": "dev",
            "deploy_target": deploy_target,
            "approved": True,
            "context": context,
            "metadata": {"smoke": True, "flow": name},
        },
        timeout_seconds=timeout_seconds,
        auth_token=auth_token,
    )

    release_status = "unknown"
    deploy_success = False
    if isinstance(run_resp, dict) and run_resp.get("success"):
        data = run_resp.get("data", {})
        if isinstance(data, dict):
            release = data.get("release", {})
            deploy = data.get("deploy", {})
            if isinstance(release, dict):
                release_status = str(release.get("status", "unknown"))
            if isinstance(deploy, dict):
                deploy_success = bool(deploy.get("success", False))

    ok = bool(bootstrap_resp.get("success")) and bool(run_resp.get("success"))
    ok = ok and release_status == "deployed" and deploy_success
    return {
        "flow": name,
        "project_name": project_name,
        "template_id": template_id,
        "stack": stack,
        "deploy_target": deploy_target,
        "ok": ok,
        "release_status": release_status,
        "deploy_success": deploy_success,
        "bootstrap_response": bootstrap_resp,
        "run_response": run_resp,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reference delivery smoke flows via API.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8080")
    parser.add_argument("--auth-token", default=os.environ.get("JARVIS_API_TOKEN", ""))
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--real-gates",
        action="store_true",
        help="Use real stack gate templates without override commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    flows: List[Dict[str, Any]] = [
        {
            "name": "backend-reference",
            "template_id": "backend_fastapi",
            "stack": "backend",
            "deploy_target": "local",
        },
        {
            "name": "frontend-reference",
            "template_id": "frontend_react",
            "stack": "frontend",
            "deploy_target": "local",
        },
    ]

    results = [
        _run_flow(
            api_base=str(args.api_base),
            auth_token=str(args.auth_token or "") or None,
            timeout_seconds=float(args.timeout_seconds),
            real_gates=bool(args.real_gates),
            **flow,
        )
        for flow in flows
    ]

    overall_ok = all(bool(r.get("ok")) for r in results)
    report = {
        "overall_ok": overall_ok,
        "real_gates": bool(args.real_gates),
        "api_base": str(args.api_base),
        "results": results,
    }
    print(json.dumps(report, indent=2))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
