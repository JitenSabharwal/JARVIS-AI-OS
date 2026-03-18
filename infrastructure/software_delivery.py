"""
Production software delivery engine for template bootstrap, quality gates,
and deployment/rollback controls.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TemplateSpec:
    template_id: str
    stack: str
    description: str
    cloud_targets: List[str]
    files: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "stack": self.stack,
            "description": self.description,
            "cloud_targets": list(self.cloud_targets),
            "file_count": len(self.files),
        }


@dataclass
class DeploymentProfile:
    name: str
    require_approval: bool
    auto_rollback: bool
    max_error_rate_pct: float
    max_p95_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "require_approval": self.require_approval,
            "auto_rollback": self.auto_rollback,
            "max_error_rate_pct": self.max_error_rate_pct,
            "max_p95_latency_ms": self.max_p95_latency_ms,
        }


@dataclass
class ReleaseRecord:
    release_id: str
    project_name: str
    profile: str
    status: str
    created_at: float
    started_at: float
    completed_at: Optional[float] = None
    gates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    post_deploy: Dict[str, Any] = field(default_factory=dict)
    rollback_reason: Optional[str] = None
    incident_note: Optional[str] = None
    lead_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "release_id": self.release_id,
            "project_name": self.project_name,
            "profile": self.profile,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "gates": self.gates,
            "post_deploy": self.post_deploy,
            "rollback_reason": self.rollback_reason,
            "incident_note": self.incident_note,
            "lead_time_seconds": self.lead_time_seconds,
            "metadata": dict(self.metadata),
        }


class SoftwareDeliveryEngine:
    """In-memory SDLC automation for production delivery workflows."""

    DEFAULT_GATES = ["lint", "test", "sast", "dependency_audit"]

    def __init__(self, *, delivery_config: Optional[Dict[str, Any]] = None) -> None:
        self._templates: Dict[str, TemplateSpec] = self._default_templates()
        self._profiles: Dict[str, DeploymentProfile] = self._default_profiles()
        self._releases: Dict[str, ReleaseRecord] = {}
        self._gate_runners: Dict[str, Callable[[str, Dict[str, Any]], Dict[str, Any]]] = {}
        self._deploy_adapters: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self._runtime_config = self._default_runtime_config()
        self._provider_deploy_commands: Dict[str, List[str]] = {}
        if delivery_config:
            self.apply_runtime_config(delivery_config)
        self._register_default_gate_runners()
        self._register_default_deploy_adapters()

    def list_templates(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._templates.values()]

    def list_profiles(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._profiles.values()]

    def list_gate_runners(self) -> List[str]:
        return sorted(self._gate_runners.keys())

    def list_deploy_adapters(self) -> List[str]:
        return sorted(self._deploy_adapters.keys())

    def get_deploy_adapter_specs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "local": {
                "mode": "command_or_simulation",
                "required_context_keys": [],
                "retryable_error_types": ["network", "timeout", "throttle", "transient"],
            },
            "aws": {
                "mode": "command_or_simulation",
                "required_context_keys": ["deploy_commands.aws OR runtime aws_deploy_command OR context.deploy.success"],
                "retryable_error_types": ["network", "timeout", "throttle", "transient"],
            },
            "gcp": {
                "mode": "command_or_simulation",
                "required_context_keys": ["deploy_commands.gcp OR runtime gcp_deploy_command OR context.deploy.success"],
                "retryable_error_types": ["network", "timeout", "throttle", "transient"],
            },
            "vercel": {
                "mode": "command_or_simulation",
                "required_context_keys": ["deploy_commands.vercel OR runtime vercel_deploy_command OR context.deploy.success"],
                "retryable_error_types": ["network", "timeout", "throttle", "transient"],
            },
        }

    def list_ci_gate_templates(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            stack: {gate: list(cmd) for gate, cmd in gates.items()}
            for stack, gates in self._default_ci_gate_templates().items()
        }

    def get_runtime_config(self) -> Dict[str, Any]:
        return dict(self._runtime_config)

    def apply_runtime_config(self, delivery_config: Dict[str, Any]) -> None:
        merged = dict(self._runtime_config)
        merged.update(dict(delivery_config or {}))
        try:
            merged["command_execution_enabled"] = bool(merged.get("command_execution_enabled", True))
            merged["command_timeout_seconds"] = max(1.0, float(merged.get("command_timeout_seconds", 120.0)))
            merged["max_output_chars"] = max(200, int(merged.get("max_output_chars", 2000)))
            merged["deploy_max_retries"] = max(0, int(merged.get("deploy_max_retries", 1)))
            merged["deploy_retry_backoff_seconds"] = max(
                0.0, float(merged.get("deploy_retry_backoff_seconds", 1.0))
            )
            merged["allowed_deploy_targets"] = [
                str(x).strip() for x in merged.get("allowed_deploy_targets", ["local", "aws", "gcp", "vercel"])
                if str(x).strip()
            ]
            merged["default_working_dir"] = str(merged.get("default_working_dir", "") or "")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid delivery runtime config: {exc}") from exc
        self._runtime_config = merged
        self._provider_deploy_commands = {}
        for target in ("local", "aws", "gcp", "vercel"):
            raw = merged.get(f"{target}_deploy_command", "")
            cmd = self._parse_command(raw)
            if cmd:
                self._provider_deploy_commands[target] = cmd

    def register_gate_runner(
        self,
        gate_name: str,
        runner: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        *,
        overwrite: bool = False,
    ) -> None:
        gate = gate_name.strip()
        if not gate:
            raise ValueError("gate_name is required")
        if gate in self._gate_runners and not overwrite:
            raise ValueError(f"gate runner already registered: {gate}")
        self._gate_runners[gate] = runner

    def register_deploy_adapter(
        self,
        target: str,
        adapter: Callable[[Dict[str, Any]], Dict[str, Any]],
        *,
        overwrite: bool = False,
    ) -> None:
        key = target.strip()
        if not key:
            raise ValueError("target is required")
        if key in self._deploy_adapters and not overwrite:
            raise ValueError(f"deploy adapter already registered: {key}")
        self._deploy_adapters[key] = adapter

    def bootstrap_project(
        self,
        *,
        template_id: str,
        project_name: str,
        cloud_target: str = "local",
        include_ci: bool = True,
    ) -> Dict[str, Any]:
        if template_id not in self._templates:
            raise KeyError(f"unknown template_id: {template_id}")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{2,64}", project_name):
            raise ValueError("project_name must match [a-zA-Z0-9_-]{2,64}")

        template = self._templates[template_id]
        if cloud_target not in template.cloud_targets:
            raise ValueError(f"cloud_target unsupported by template: {cloud_target}")

        rendered: List[Dict[str, str]] = []
        for path, content in template.files.items():
            rendered.append(
                {
                    "path": path,
                    "content": content.replace("__PROJECT_NAME__", project_name),
                }
            )
        if include_ci:
            rendered.append(
                {
                    "path": ".github/workflows/ci.yml",
                    "content": self._default_ci_workflow(),
                }
            )

        return {
            "project_name": project_name,
            "template_id": template_id,
            "cloud_target": cloud_target,
            "required_gates": list(self.DEFAULT_GATES),
            "files": rendered,
            "file_count": len(rendered),
        }

    def run_pipeline(
        self,
        *,
        project_name: str,
        gate_inputs: Dict[str, Any],
        required_gates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not project_name.strip():
            raise ValueError("project_name is required")
        if not isinstance(gate_inputs, dict):
            raise ValueError("gate_inputs must be an object")

        gates = list(required_gates or self.DEFAULT_GATES)
        gate_results: Dict[str, Dict[str, Any]] = {}
        failed: List[str] = []

        for gate in gates:
            raw = gate_inputs.get(gate)
            passed, details = self._normalize_gate_input(raw)
            gate_results[gate] = {
                "passed": passed,
                "details": details,
                "required": True,
            }
            if not passed:
                failed.append(gate)

        all_passed = len(failed) == 0
        return {
            "project_name": project_name,
            "required_gates": gates,
            "gate_results": gate_results,
            "all_passed": all_passed,
            "failed_gates": failed,
            "risk_score": round((len(failed) / max(1, len(gates))) * 100.0, 2),
            "evaluated_at": time.time(),
        }

    def run_pipeline_with_runners(
        self,
        *,
        project_name: str,
        required_gates: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        gates = list(required_gates or self.DEFAULT_GATES)
        ctx = dict(context or {})
        gate_inputs: Dict[str, Any] = {}
        self._inject_auto_gate_commands(ctx)
        for gate in gates:
            runner = self._gate_runners.get(gate)
            if runner is None:
                gate_inputs[gate] = {"passed": False, "reason": "no_gate_runner_registered"}
                continue
            try:
                gate_inputs[gate] = runner(project_name, ctx)
            except Exception as exc:  # noqa: BLE001
                gate_inputs[gate] = {"passed": False, "reason": f"runner_exception:{exc}"}
        result = self.run_pipeline(
            project_name=project_name,
            gate_inputs=gate_inputs,
            required_gates=gates,
        )
        result["mode"] = "runner_execution"
        return result

    def create_release(
        self,
        *,
        project_name: str,
        profile: str,
        pipeline_result: Dict[str, Any],
        post_deploy: Optional[Dict[str, Any]] = None,
        approved: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        build_started_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        if profile not in self._profiles:
            raise KeyError(f"unknown profile: {profile}")

        profile_spec = self._profiles[profile]
        all_passed = bool(pipeline_result.get("all_passed", False))
        gates = dict(pipeline_result.get("gate_results", {}) or {})
        now_ts = time.time()
        started_at = float(build_started_at) if build_started_at else now_ts

        record = ReleaseRecord(
            release_id=f"rel-{uuid.uuid4().hex}",
            project_name=project_name.strip(),
            profile=profile,
            status="blocked",
            created_at=now_ts,
            started_at=started_at,
            gates=gates,
            metadata=metadata or {},
        )

        if not all_passed:
            record.status = "blocked"
            record.rollback_reason = "quality_gates_failed"
            record.incident_note = "Release blocked before deploy: mandatory quality gates failed."
            record.completed_at = time.time()
            record.lead_time_seconds = round(record.completed_at - record.started_at, 3)
            self._releases[record.release_id] = record
            return record.to_dict()

        if profile_spec.require_approval and not approved:
            record.status = "waiting_approval"
            self._releases[record.release_id] = record
            return record.to_dict()

        record.status = "deployed"
        if post_deploy:
            self._apply_post_deploy_checks(record, profile_spec, post_deploy)

        record.completed_at = time.time()
        record.lead_time_seconds = round(record.completed_at - record.started_at, 3)
        self._releases[record.release_id] = record
        return record.to_dict()

    def run_release_pipeline(
        self,
        *,
        project_name: str,
        profile: str,
        deploy_target: str,
        approved: bool = False,
        required_gates: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        post_deploy: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        allowed_targets = set(self._runtime_config.get("allowed_deploy_targets", []))
        if allowed_targets and deploy_target not in allowed_targets:
            raise ValueError(f"deploy_target not allowed: {deploy_target}")
        pipeline = self.run_pipeline_with_runners(
            project_name=project_name,
            required_gates=required_gates,
            context=context,
        )
        release = self.create_release(
            project_name=project_name,
            profile=profile,
            pipeline_result=pipeline,
            approved=approved,
            metadata=metadata,
            post_deploy=post_deploy,
        )

        deploy_result: Dict[str, Any] = {
            "target": deploy_target,
            "status": "skipped",
            "reason": "release_not_deployed",
        }
        if release.get("status") == "deployed":
            adapter = self._deploy_adapters.get(deploy_target.strip())
            if adapter is None:
                release["status"] = "degraded"
                release["rollback_reason"] = "missing_deploy_adapter"
                release["incident_note"] = f"No deploy adapter registered for target: {deploy_target}"
                deploy_result = {
                    "target": deploy_target,
                    "status": "failed",
                    "reason": "missing_deploy_adapter",
                }
            else:
                deploy_payload = {
                    "release_id": release["release_id"],
                    "project_name": project_name,
                    "profile": profile,
                    "target": deploy_target,
                    "context": dict(context or {}),
                }
                deploy_result = self._run_deploy_with_retries(
                    adapter=adapter,
                    deploy_payload=deploy_payload,
                    target=deploy_target,
                )
                if not bool(deploy_result.get("success", False)):
                    release["status"] = "rolled_back"
                    release["rollback_reason"] = str(deploy_result.get("error_type", "deploy_failed"))
                    release["incident_note"] = (
                        "Deploy adapter reported failure. "
                        + str(deploy_result.get("reason", "unknown"))
                    )

            rid = str(release.get("release_id", ""))
            if rid in self._releases:
                self._releases[rid].status = str(release.get("status", "deployed"))
                self._releases[rid].rollback_reason = release.get("rollback_reason")
                self._releases[rid].incident_note = release.get("incident_note")

        return {
            "pipeline": pipeline,
            "release": release,
            "deploy": deploy_result,
        }

    def evaluate_post_deploy(
        self,
        release_id: str,
        post_deploy: Dict[str, Any],
    ) -> Dict[str, Any]:
        record = self._releases.get(release_id)
        if record is None:
            raise KeyError(f"release not found: {release_id}")
        profile_spec = self._profiles.get(record.profile)
        if profile_spec is None:
            raise KeyError(f"profile not found: {record.profile}")
        self._apply_post_deploy_checks(record, profile_spec, post_deploy)
        record.completed_at = time.time()
        record.lead_time_seconds = round(record.completed_at - record.started_at, 3)
        return record.to_dict()

    def get_release(self, release_id: str) -> Optional[Dict[str, Any]]:
        record = self._releases.get(release_id)
        return record.to_dict() if record else None

    def get_lead_time_summary(self) -> Dict[str, Any]:
        completed = [r for r in self._releases.values() if r.lead_time_seconds is not None]
        if not completed:
            return {
                "release_count": 0,
                "average_lead_time_seconds": 0.0,
                "median_lead_time_seconds": 0.0,
                "p95_lead_time_seconds": 0.0,
            }
        values = sorted(float(r.lead_time_seconds or 0.0) for r in completed)
        avg = sum(values) / len(values)
        median = values[len(values) // 2]
        p95_index = min(len(values) - 1, int(round((len(values) - 1) * 0.95)))
        return {
            "release_count": len(values),
            "average_lead_time_seconds": round(avg, 3),
            "median_lead_time_seconds": round(median, 3),
            "p95_lead_time_seconds": round(values[p95_index], 3),
        }

    def _apply_post_deploy_checks(
        self,
        record: ReleaseRecord,
        profile_spec: DeploymentProfile,
        post_deploy: Dict[str, Any],
    ) -> None:
        error_rate_pct = float(post_deploy.get("error_rate_pct", 0.0))
        p95_latency_ms = float(post_deploy.get("p95_latency_ms", 0.0))
        availability_pct = float(post_deploy.get("availability_pct", 100.0))

        record.post_deploy = {
            "error_rate_pct": error_rate_pct,
            "p95_latency_ms": p95_latency_ms,
            "availability_pct": availability_pct,
            "checked_at": time.time(),
        }

        violations: List[str] = []
        if error_rate_pct > profile_spec.max_error_rate_pct:
            violations.append("error_rate")
        if p95_latency_ms > profile_spec.max_p95_latency_ms:
            violations.append("latency")
        if availability_pct < 99.0:
            violations.append("availability")

        if violations and profile_spec.auto_rollback:
            record.status = "rolled_back"
            record.rollback_reason = ",".join(violations)
            record.incident_note = (
                "Auto-rollback executed after post-deploy check violations: "
                + ", ".join(violations)
            )
        elif violations:
            record.status = "degraded"
            record.rollback_reason = ",".join(violations)
            record.incident_note = "Post-deploy checks degraded; manual intervention required."

    def _register_default_gate_runners(self) -> None:
        def _default_gate_runner(gate_name: str) -> Callable[[str, Dict[str, Any]], Dict[str, Any]]:
            def _runner(_project_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
                gate_commands = context.get("gate_commands", {})
                if isinstance(gate_commands, dict) and gate_name in gate_commands:
                    if not self._runtime_config.get("command_execution_enabled", True):
                        return {"passed": False, "reason": "command_execution_disabled", "runner": "subprocess"}
                    command = self._parse_command(gate_commands.get(gate_name))
                    if not command:
                        return {"passed": False, "reason": "empty_gate_command", "runner": "subprocess"}
                    completed = self._run_subprocess(
                        command=command,
                        context=context,
                    )
                    return {
                        "passed": completed["exit_code"] == 0,
                        "runner": "subprocess",
                        "command": command,
                        "exit_code": completed["exit_code"],
                        "stdout": completed["stdout"],
                        "stderr": completed["stderr"],
                        "duration_ms": completed["duration_ms"],
                    }
                gates_raw = context.get("gates", {})
                if not isinstance(gates_raw, dict):
                    return {"passed": False, "reason": "context.gates must be object"}
                raw = gates_raw.get(gate_name)
                passed, details = self._normalize_gate_input(raw)
                details["runner"] = "context_default"
                return {"passed": passed, **details}

            return _runner

        for gate in self.DEFAULT_GATES:
            self._gate_runners[gate] = _default_gate_runner(gate)

    def _inject_auto_gate_commands(self, context: Dict[str, Any]) -> None:
        if not bool(context.get("auto_gate_commands", False)):
            return
        existing = context.get("gate_commands", {})
        if existing is not None and not isinstance(existing, dict):
            return
        gate_commands: Dict[str, Any] = dict(existing or {})

        stack = self._resolve_stack_key(context)
        templates = self._default_ci_gate_templates().get(stack, {})
        overrides = context.get("gate_command_overrides", {})
        if not isinstance(overrides, dict):
            overrides = {}

        for gate in self.DEFAULT_GATES:
            if gate in gate_commands:
                continue
            raw_override = overrides.get(gate)
            cmd = self._parse_command(raw_override)
            if not cmd:
                cmd = list(templates.get(gate, []))
            if cmd:
                gate_commands[gate] = cmd
        context["gate_commands"] = gate_commands

    @staticmethod
    def _resolve_stack_key(context: Dict[str, Any]) -> str:
        raw = str(context.get("stack", "")).strip().lower()
        if raw in {"backend", "frontend", "fullstack"}:
            return raw
        template_id = str(context.get("template_id", "")).strip().lower()
        if "backend" in template_id:
            return "backend"
        if "frontend" in template_id:
            return "frontend"
        if "fullstack" in template_id or "next" in template_id:
            return "fullstack"
        return "backend"

    def _register_default_deploy_adapters(self) -> None:
        def _provider_adapter(provider: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
            def _adapter(payload: Dict[str, Any]) -> Dict[str, Any]:
                context = payload.get("context", {})
                target = str(payload.get("target", provider)).strip() or provider
                if not isinstance(context, dict):
                    return {
                        "success": False,
                        "target": target,
                        "provider": provider,
                        "release_id": payload.get("release_id", ""),
                        "error_type": "config",
                        "reason": "context must be object",
                        "retryable": False,
                    }
                return self._execute_provider_deploy(
                    provider=provider,
                    target=target,
                    payload=payload,
                    context=context,
                )

            return _adapter

        for provider in ("local", "aws", "gcp", "vercel"):
            self._deploy_adapters[provider] = _provider_adapter(provider)

    def _execute_provider_deploy(
        self,
        *,
        provider: str,
        target: str,
        payload: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self._runtime_config.get("command_execution_enabled", True):
            return {
                "success": False,
                "target": target,
                "provider": provider,
                "release_id": payload.get("release_id", ""),
                "error_type": "policy",
                "reason": "command_execution_disabled",
                "retryable": False,
            }

        deploy_commands = context.get("deploy_commands", {})
        command: List[str] = []
        source = "none"
        if isinstance(deploy_commands, dict) and target in deploy_commands:
            command = self._parse_command(deploy_commands.get(target))
            source = "request_context"
        if not command:
            command = list(self._provider_deploy_commands.get(target, []))
            if command:
                source = "runtime_config"

        if command:
            completed = self._run_subprocess(command=command, context=context)
            success = int(completed.get("exit_code", 1)) == 0
            error_type = self._classify_deploy_error(
                exit_code=int(completed.get("exit_code", 1)),
                stderr=str(completed.get("stderr", "")),
            )
            return {
                "success": success,
                "target": target,
                "provider": provider,
                "release_id": payload.get("release_id", ""),
                "source": source,
                "command": command,
                "exit_code": completed["exit_code"],
                "stdout": completed["stdout"],
                "stderr": completed["stderr"],
                "duration_ms": completed["duration_ms"],
                "error_type": "" if success else error_type,
                "reason": "" if success else "deploy_command_failed",
                "retryable": (not success and error_type in {"network", "timeout", "throttle", "transient"}),
            }

        deploy = context.get("deploy", {})
        if isinstance(deploy, dict):
            requested = bool(deploy.get("success", True))
            return {
                "success": requested,
                "target": target,
                "provider": provider,
                "release_id": payload.get("release_id", ""),
                "source": "simulation",
                "error_type": "" if requested else "simulation_failure",
                "reason": "" if requested else "context.deploy.success=false",
                "retryable": False,
            }

        return {
            "success": False,
            "target": target,
            "provider": provider,
            "release_id": payload.get("release_id", ""),
            "source": "none",
            "error_type": "config",
            "reason": "missing deploy command or simulation directive",
            "retryable": False,
        }

    def _run_deploy_with_retries(
        self,
        *,
        adapter: Callable[[Dict[str, Any]], Dict[str, Any]],
        deploy_payload: Dict[str, Any],
        target: str,
    ) -> Dict[str, Any]:
        retries = int(self._runtime_config.get("deploy_max_retries", 1))
        backoff = float(self._runtime_config.get("deploy_retry_backoff_seconds", 1.0))
        attempts_allowed = max(1, retries + 1)
        attempts = 0
        last_result: Dict[str, Any] = {
            "success": False,
            "target": target,
            "error_type": "unknown",
            "reason": "adapter_not_run",
            "retryable": False,
        }
        for attempt in range(1, attempts_allowed + 1):
            attempts = attempt
            try:
                result = adapter(deploy_payload)
            except Exception as exc:  # noqa: BLE001
                result = {
                    "success": False,
                    "target": target,
                    "error_type": "adapter_exception",
                    "reason": f"deploy_adapter_exception:{exc}",
                    "retryable": False,
                }
            last_result = dict(result)
            last_result["attempt"] = attempt
            if bool(last_result.get("success", False)):
                break
            if not bool(last_result.get("retryable", False)):
                break
            if attempt < attempts_allowed:
                time.sleep(max(0.0, backoff) * attempt)
        last_result["attempts"] = attempts
        return last_result

    @staticmethod
    def _classify_deploy_error(*, exit_code: int, stderr: str) -> str:
        text = (stderr or "").lower()
        if exit_code == 124 or "timed out" in text or "timeout" in text:
            return "timeout"
        if any(token in text for token in ("unauthorized", "forbidden", "auth", "permission")):
            return "auth"
        if any(token in text for token in ("throttle", "rate limit", "quota")):
            return "throttle"
        if any(token in text for token in ("network", "connection", "dns", "unreachable")):
            return "network"
        if any(token in text for token in ("temporary", "transient", "try again")):
            return "transient"
        if exit_code != 0:
            return "unknown"
        return ""

    @staticmethod
    def _default_ci_gate_templates() -> Dict[str, Dict[str, List[str]]]:
        return {
            "backend": {
                "lint": ["python3", "-m", "ruff", "check", "."],
                "test": ["python3", "-m", "pytest", "-q"],
                "sast": ["python3", "-m", "bandit", "-q", "-r", "."],
                "dependency_audit": ["python3", "-m", "pip_audit", "-f", "json"],
            },
            "frontend": {
                "lint": ["npm", "run", "lint", "--if-present"],
                "test": ["npm", "run", "test", "--", "--watch=false"],
                "sast": ["npm", "audit", "--audit-level=high", "--json"],
                "dependency_audit": ["npm", "audit", "--production", "--json"],
            },
            "fullstack": {
                "lint": ["npm", "run", "lint", "--if-present"],
                "test": ["npm", "run", "test", "--", "--watch=false"],
                "sast": ["npm", "audit", "--audit-level=high", "--json"],
                "dependency_audit": ["npm", "audit", "--production", "--json"],
            },
        }

    @staticmethod
    def _parse_command(raw: Any) -> List[str]:
        if isinstance(raw, list) and all(isinstance(p, str) for p in raw):
            return [str(p) for p in raw if str(p)]
        if isinstance(raw, str):
            return shlex.split(raw)
        return []

    @staticmethod
    def _safe_cwd(raw_cwd: Any) -> Optional[str]:
        if raw_cwd is None:
            return None
        cwd = str(raw_cwd).strip()
        if not cwd:
            return None
        if not re.fullmatch(r"[a-zA-Z0-9_./-]+", cwd):
            return None
        return cwd

    def _run_subprocess(self, *, command: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        timeout_seconds = context.get(
            "command_timeout_seconds",
            self._runtime_config.get("command_timeout_seconds", 120.0),
        )
        truncate_output = context.get(
            "max_output_chars",
            self._runtime_config.get("max_output_chars", 2000),
        )
        env = None
        env_allowlist = context.get("env_allowlist", {})
        if isinstance(env_allowlist, dict):
            env = {k: str(v) for k, v in env_allowlist.items() if isinstance(k, str)}
        raw_cwd = context.get("cwd", None)
        if raw_cwd is None:
            raw_cwd = self._runtime_config.get("default_working_dir", "")
        cwd = self._safe_cwd(raw_cwd)
        started = time.time()
        try:
            timeout = max(1.0, float(timeout_seconds))
        except (TypeError, ValueError):
            timeout = 120.0
        try:
            max_chars = max(200, int(truncate_output))
        except (TypeError, ValueError):
            max_chars = 2000
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=env,
                check=False,
            )
            stdout = (proc.stdout or "")[:max_chars]
            stderr = (proc.stderr or "")[:max_chars]
            return {
                "exit_code": int(proc.returncode),
                "stdout": stdout,
                "stderr": stderr,
                "duration_ms": round((time.time() - started) * 1000.0, 2),
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": 124,
                "stdout": "",
                "stderr": f"command timed out after {timeout}s",
                "duration_ms": round((time.time() - started) * 1000.0, 2),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": f"command execution failed: {exc}",
                "duration_ms": round((time.time() - started) * 1000.0, 2),
            }

    @staticmethod
    def _normalize_gate_input(raw: Any) -> tuple[bool, Dict[str, Any]]:
        if isinstance(raw, bool):
            return raw, {}
        if isinstance(raw, dict):
            passed = bool(raw.get("passed", False))
            details = dict(raw)
            details.pop("passed", None)
            return passed, details
        return False, {"reason": "missing_or_invalid_gate_input"}

    @staticmethod
    def _default_ci_workflow() -> str:
        return (
            "name: ci\n"
            "on: [push, pull_request]\n"
            "jobs:\n"
            "  quality:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - name: Lint\n"
            "        run: echo \"run lint\"\n"
            "      - name: Test\n"
            "        run: echo \"run tests\"\n"
            "      - name: SAST\n"
            "        run: echo \"run sast\"\n"
            "      - name: Dependency Audit\n"
            "        run: echo \"run dep audit\"\n"
        )

    @staticmethod
    def _default_templates() -> Dict[str, TemplateSpec]:
        return {
            "backend_fastapi": TemplateSpec(
                template_id="backend_fastapi",
                stack="backend",
                description="Python FastAPI service template",
                cloud_targets=["local", "aws", "gcp"],
                files={
                    "README.md": "# __PROJECT_NAME__\\nFastAPI backend service.\\n",
                    "app/main.py": (
                        "from fastapi import FastAPI\\n\\n"
                        "app = FastAPI(title='__PROJECT_NAME__')\\n\\n"
                        "@app.get('/health')\\n"
                        "def health() -> dict[str, str]:\\n"
                        "    return {'status': 'ok'}\\n"
                    ),
                },
            ),
            "frontend_react": TemplateSpec(
                template_id="frontend_react",
                stack="frontend",
                description="React SPA template",
                cloud_targets=["local", "aws", "vercel"],
                files={
                    "README.md": "# __PROJECT_NAME__\\nReact frontend app.\\n",
                    "src/main.tsx": (
                        "import React from 'react';\\n"
                        "import { createRoot } from 'react-dom/client';\\n"
                        "createRoot(document.getElementById('root')!).render(" 
                        "<h1>__PROJECT_NAME__</h1>);\\n"
                    ),
                },
            ),
            "fullstack_next": TemplateSpec(
                template_id="fullstack_next",
                stack="fullstack",
                description="Next.js full-stack template",
                cloud_targets=["local", "aws", "vercel"],
                files={
                    "README.md": "# __PROJECT_NAME__\\nNext.js full-stack app.\\n",
                    "app/page.tsx": "export default function Page() { return <h1>__PROJECT_NAME__</h1>; }\\n",
                    "app/api/health/route.ts": (
                        "export async function GET() {\\n"
                        "  return Response.json({ status: 'ok' });\\n"
                        "}\\n"
                    ),
                },
            ),
        }

    @staticmethod
    def _default_profiles() -> Dict[str, DeploymentProfile]:
        return {
            "dev": DeploymentProfile(
                name="dev",
                require_approval=False,
                auto_rollback=False,
                max_error_rate_pct=10.0,
                max_p95_latency_ms=3000.0,
            ),
            "stage": DeploymentProfile(
                name="stage",
                require_approval=True,
                auto_rollback=True,
                max_error_rate_pct=5.0,
                max_p95_latency_ms=1800.0,
            ),
            "prod": DeploymentProfile(
                name="prod",
                require_approval=True,
                auto_rollback=True,
                max_error_rate_pct=2.0,
                max_p95_latency_ms=1000.0,
            ),
        }

    @staticmethod
    def _default_runtime_config() -> Dict[str, Any]:
        return {
            "command_execution_enabled": True,
            "command_timeout_seconds": 120.0,
            "max_output_chars": 2000,
            "allowed_deploy_targets": ["local", "aws", "gcp", "vercel"],
            "deploy_max_retries": 1,
            "deploy_retry_backoff_seconds": 1.0,
            "default_working_dir": "",
            "local_deploy_command": "",
            "aws_deploy_command": "",
            "gcp_deploy_command": "",
            "vercel_deploy_command": "",
        }


__all__ = [
    "SoftwareDeliveryEngine",
    "TemplateSpec",
    "DeploymentProfile",
    "ReleaseRecord",
]
