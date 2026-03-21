"""
System information and command-execution skills for JARVIS AI OS.

Skills provided:
- :class:`SystemInfoSkill`   – CPU, memory, disk, OS details
- :class:`RunCommandSkill`   – Execute shell commands with allowlist + timeout
- :class:`ProcessListSkill`  – List running processes
- :class:`EnvironmentSkill`  – Read/write environment variables (safe subset)
- :class:`NetworkInfoSkill`  – Network interface information
"""

from __future__ import annotations

import asyncio
import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional

from infrastructure.approval import ApprovalManager
from infrastructure.logger import get_logger
from infrastructure.slo_metrics import get_slo_metrics
from skills.base_skill import BaseSkill, SkillParameter, SkillResult

logger = get_logger(__name__)

# Optional: psutil gives richer system data.
try:
    import psutil  # type: ignore

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    logger.debug("psutil not available; system skills use fallback implementations.")

# ---------------------------------------------------------------------------
# Command allowlist – conservative default set.
# Operators like pipes, redirects, and shell meta-chars are blocked.
# ---------------------------------------------------------------------------

_ALLOWED_COMMANDS: frozenset[str] = frozenset(
    [
        # System information (read-only)
        "date",
        "uptime",
        "uname",
        "hostname",
        "whoami",
        "id",
        "groups",
        # Disk / memory / CPU stats (read-only)
        "df",
        "free",
        "ps",
        "pwd",
        # Directory listing only
        "ls",
        "dir",
        # Simple output
        "echo",
        # Checksums (read-only)
        "md5sum",
        "sha256sum",
        # Network info (passive)
        "ifconfig",
        "ip",
        "ss",
        "netstat",
    ]
)

# Commands that require explicit approval token because they can reveal
# network topology or process state details useful for reconnaissance.
_REQUIRES_APPROVAL_COMMANDS: frozenset[str] = frozenset(
    ["ifconfig", "ip", "ss", "netstat", "ps"]
)

# Environment variable allowlist for EnvironmentSkill.
_ALLOWED_ENV_READ: frozenset[str] = frozenset(
    [
        "HOME",
        "USER",
        "USERNAME",
        "SHELL",
        "TERM",
        "LANG",
        "LC_ALL",
        "PATH",
        "PWD",
        "OLDPWD",
        "HOSTNAME",
        "LOGNAME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "EDITOR",
        "VISUAL",
        "PAGER",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "NODE_ENV",
        "PORT",
        "HOST",
        "JAVA_HOME",
        "GOPATH",
        "GOROOT",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "NVM_DIR",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "CI",
        "GITHUB_ACTIONS",
        "GITHUB_WORKSPACE",
        "GITHUB_REPOSITORY",
        "TRAVIS",
        "CIRCLECI",
        "BUILD_ID",
        "BUILD_NUMBER",
    ]
)

# Variables that must never be read or set via the skill.
_BLOCKED_ENV: frozenset[str] = frozenset(
    [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AZURE_CLIENT_SECRET",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DATABASE_URL",
        "SECRET_KEY",
        "PRIVATE_KEY",
        "PASSWORD",
        "TOKEN",
        "API_KEY",
        "AUTH_TOKEN",
    ]
)


# ---------------------------------------------------------------------------
# SystemInfoSkill
# ---------------------------------------------------------------------------


class SystemInfoSkill(BaseSkill):
    """Return CPU, memory, disk, and OS information."""

    @property
    def name(self) -> str:
        return "system_info"

    @property
    def description(self) -> str:
        return (
            "Retrieve system information including CPU, memory, disk usage, "
            "operating system details, and Python runtime information."
        )

    @property
    def category(self) -> str:
        return "system"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter(
                "include",
                "string",
                "Comma-separated list of sections: cpu,memory,disk,os,python,all. Default: all.",
                required=False,
                default="all",
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        include = params.get("include", "all")
        valid = {"cpu", "memory", "disk", "os", "python", "all"}
        for part in include.split(","):
            part = part.strip()
            if part and part not in valid:
                raise ValueError(f"Invalid section '{part}'. Valid: {sorted(valid)}")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        include_str = params.get("include", "all")
        sections = {s.strip() for s in include_str.split(",")} if include_str else {"all"}
        all_sections = "all" in sections

        info: Dict[str, Any] = {}

        if all_sections or "os" in sections:
            info["os"] = {
                "system": platform.system(),
                "node": platform.node(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            }

        if all_sections or "python" in sections:
            import sys
            info["python"] = {
                "version": sys.version,
                "executable": sys.executable,
                "platform": sys.platform,
            }

        if _PSUTIL_AVAILABLE:
            if all_sections or "cpu" in sections:
                info["cpu"] = {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "logical_cores": psutil.cpu_count(logical=True),
                    "frequency_mhz": getattr(psutil.cpu_freq(), "current", None),
                    "usage_percent": psutil.cpu_percent(interval=0.1),
                    "load_avg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                }

            if all_sections or "memory" in sections:
                vm = psutil.virtual_memory()
                swap = psutil.swap_memory()
                info["memory"] = {
                    "total_mb": round(vm.total / 1024 / 1024, 1),
                    "available_mb": round(vm.available / 1024 / 1024, 1),
                    "used_mb": round(vm.used / 1024 / 1024, 1),
                    "percent": vm.percent,
                    "swap_total_mb": round(swap.total / 1024 / 1024, 1),
                    "swap_used_mb": round(swap.used / 1024 / 1024, 1),
                    "swap_percent": swap.percent,
                }

            if all_sections or "disk" in sections:
                disks = []
                for part in psutil.disk_partitions(all=False):
                    try:
                        usage = psutil.disk_usage(part.mountpoint)
                        disks.append({
                            "device": part.device,
                            "mountpoint": part.mountpoint,
                            "fstype": part.fstype,
                            "total_gb": round(usage.total / 1024**3, 2),
                            "used_gb": round(usage.used / 1024**3, 2),
                            "free_gb": round(usage.free / 1024**3, 2),
                            "percent": usage.percent,
                        })
                    except (PermissionError, OSError):
                        continue
                info["disk"] = disks
        else:
            # Fallback without psutil
            if all_sections or "cpu" in sections:
                info["cpu"] = {
                    "logical_cores": os.cpu_count(),
                    "load_avg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                }
            if all_sections or "memory" in sections:
                info["memory"] = {"note": "psutil not available; install for full memory info."}
            if all_sections or "disk" in sections:
                try:
                    stat = os.statvfs("/")
                    info["disk"] = [{
                        "mountpoint": "/",
                        "total_gb": round(stat.f_blocks * stat.f_frsize / 1024**3, 2),
                        "free_gb": round(stat.f_bavail * stat.f_frsize / 1024**3, 2),
                    }]
                except AttributeError:
                    info["disk"] = [{"note": "statvfs not available on this OS."}]

        return SkillResult.ok(data=info, metadata={"psutil_available": _PSUTIL_AVAILABLE})


# ---------------------------------------------------------------------------
# RunCommandSkill
# ---------------------------------------------------------------------------


class RunCommandSkill(BaseSkill):
    """Execute a shell command from the allowlist with a configurable timeout."""

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its stdout, stderr, and exit code. "
            "Only commands from a predefined allowlist are permitted."
        )

    @property
    def category(self) -> str:
        return "system"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter("command", "string", "The command to execute (must be in the allowlist).", required=True),
            SkillParameter(
                "args",
                "array",
                "List of string arguments to pass to the command.",
                required=False,
                default=[],
            ),
            SkillParameter(
                "timeout",
                "integer",
                "Maximum execution time in seconds (1–120).",
                required=False,
                default=30,
            ),
            SkillParameter(
                "working_dir",
                "string",
                "Working directory for the command. Defaults to current directory.",
                required=False,
                default="",
            ),
            SkillParameter(
                "approval_token",
                "string",
                "Required for elevated read commands (network/process introspection).",
                required=False,
                default="",
            ),
            SkillParameter(
                "justification",
                "string",
                "Short reason for running this command.",
                required=False,
                default="",
            ),
            SkillParameter(
                "expected_exit_codes",
                "array",
                "Allowed exit codes used in post-execution verification.",
                required=False,
                default=[0],
            ),
            SkillParameter(
                "dry_run",
                "boolean",
                "When true, run pre-check only and return planned command.",
                required=False,
                default=False,
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        command = params.get("command", "").strip()
        if not command:
            raise ValueError("'command' must be a non-empty string.")
        base_cmd = os.path.basename(command.split()[0])
        if base_cmd not in _ALLOWED_COMMANDS:
            raise ValueError(
                f"Command '{base_cmd}' is not in the allowlist. "
                f"Allowed commands: {sorted(_ALLOWED_COMMANDS)}"
            )
        approval_token = str(params.get("approval_token", "")).strip()
        justification = str(params.get("justification", "")).strip()
        if base_cmd in _REQUIRES_APPROVAL_COMMANDS:
            if approval_token and not justification:
                raise ValueError(
                    f"Command '{base_cmd}' requires a non-empty 'justification' when approval_token is provided."
                )
        timeout = params.get("timeout", 30)
        if not isinstance(timeout, int) or not (1 <= timeout <= 120):
            raise ValueError("'timeout' must be between 1 and 120 seconds.")
        expected_exit_codes = params.get("expected_exit_codes", [0])
        if not isinstance(expected_exit_codes, list) or not expected_exit_codes:
            raise ValueError("'expected_exit_codes' must be a non-empty list of integers.")
        if any(not isinstance(code, int) for code in expected_exit_codes):
            raise ValueError("'expected_exit_codes' must contain only integers.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        started_at = time.time()
        command = params["command"].strip()
        base_cmd = os.path.basename(command.split()[0])
        args: List[str] = [str(a) for a in (params.get("args") or [])]
        timeout = int(params.get("timeout", 30))
        working_dir_str: str = params.get("working_dir", "").strip()
        approval_token: str = str(params.get("approval_token", "")).strip()
        justification: str = str(params.get("justification", "")).strip()
        expected_exit_codes: List[int] = [int(code) for code in params.get("expected_exit_codes", [0])]
        dry_run: bool = bool(params.get("dry_run", False))
        approval_manager = ApprovalManager.get_instance()
        metrics = get_slo_metrics()

        # Pre-check phase: policy gate before execution.
        precheck_started = time.time()
        if base_cmd in _REQUIRES_APPROVAL_COMMANDS:
            if not approval_manager.validate_token(
                approval_token,
                expected_action=f"run_command:{base_cmd}",
            ):
                metrics.inc("run_command_policy_denied_total", label=base_cmd)
                return SkillResult.failure(
                    error=(
                        f"Approval token is invalid or not approved for action "
                        f"'run_command:{base_cmd}'."
                    ),
                    metadata={
                        "command": command,
                        "phase": "pre_check",
                        "approval_required": True,
                        "justification": justification,
                        "pre_check_ms": round((time.time() - precheck_started) * 1000, 2),
                    },
                )

        cwd: Optional[str] = None
        if working_dir_str:
            from pathlib import Path as _Path
            cwd = str((_Path(os.getcwd()) / working_dir_str) if not os.path.isabs(working_dir_str) else working_dir_str)
            if not os.path.isdir(cwd):
                return SkillResult.failure(
                    error=f"Working directory not found: '{working_dir_str}'",
                    metadata={"phase": "pre_check", "command": command},
                )

        cmd_list = [command] + args
        if dry_run:
            return SkillResult.ok(
                data={
                    "planned_command": " ".join(cmd_list),
                    "working_dir": cwd or os.getcwd(),
                    "timeout": timeout,
                },
                metadata={
                    "phase": "pre_check",
                    "dry_run": True,
                    "approval_required": base_cmd in _REQUIRES_APPROVAL_COMMANDS,
                    "expected_exit_codes": expected_exit_codes,
                    "pre_check_ms": round((time.time() - precheck_started) * 1000, 2),
                },
            )

        logger.info("RunCommandSkill: executing %s", cmd_list)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=float(timeout)
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return SkillResult.failure(
                    error=f"Command timed out after {timeout}s.",
                    metadata={"command": command, "args": args, "timeout": timeout},
                )
        except FileNotFoundError:
            return SkillResult.failure(
                error=f"Command not found: '{command}'",
                metadata={"command": command},
            )
        except OSError as exc:
            return SkillResult.failure(
                error=f"Failed to execute command: {exc}",
                metadata={"command": command},
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        exit_code = proc.returncode

        # Verify phase: command result must match expected exit-code policy.
        success = exit_code in expected_exit_codes
        total_ms = round((time.time() - started_at) * 1000, 2)
        metrics.observe_latency("run_command_total_latency_ms", total_ms, label=base_cmd)
        metrics.inc("run_command_total", label=base_cmd)
        if not success:
            metrics.inc("run_command_verify_failed_total", label=base_cmd)
        return SkillResult(
            success=success,
            data={
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "command": " ".join(cmd_list),
            },
            error=stderr if not success and stderr else None,
            metadata={
                "phase": "verify",
                "command": command,
                "args": args,
                "working_dir": cwd,
                "approval_required": base_cmd in _REQUIRES_APPROVAL_COMMANDS,
                "approval_provided": bool(approval_token),
                "justification": justification,
                "expected_exit_codes": expected_exit_codes,
                "pre_check_ms": round((time.time() - precheck_started) * 1000, 2),
                "total_ms": total_ms,
            },
        )


# ---------------------------------------------------------------------------
# ProcessListSkill
# ---------------------------------------------------------------------------


class ProcessListSkill(BaseSkill):
    """Return information about currently running processes."""

    @property
    def name(self) -> str:
        return "list_processes"

    @property
    def description(self) -> str:
        return (
            "List running processes with PID, name, CPU and memory usage. "
            "Requires psutil for full details."
        )

    @property
    def category(self) -> str:
        return "system"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter(
                "filter_name",
                "string",
                "Return only processes whose name contains this substring (case-insensitive).",
                required=False,
                default="",
            ),
            SkillParameter(
                "sort_by",
                "string",
                "Sort field: 'cpu', 'memory', 'pid', 'name'. Default: 'cpu'.",
                required=False,
                default="cpu",
            ),
            SkillParameter(
                "max_results",
                "integer",
                "Maximum number of processes to return (1–200).",
                required=False,
                default=20,
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        sort_by = params.get("sort_by", "cpu")
        if sort_by not in ("cpu", "memory", "pid", "name"):
            raise ValueError("'sort_by' must be one of: cpu, memory, pid, name.")
        max_results = params.get("max_results", 20)
        if not isinstance(max_results, int) or not (1 <= max_results <= 200):
            raise ValueError("'max_results' must be between 1 and 200.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        filter_name = params.get("filter_name", "").lower().strip()
        sort_by = params.get("sort_by", "cpu")
        max_results = int(params.get("max_results", 20))

        if not _PSUTIL_AVAILABLE:
            # Fallback: run `ps` command
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ps", "aux",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
                lines = stdout_bytes.decode("utf-8", errors="replace").splitlines()
                processes = []
                for line in lines[1:max_results + 1]:
                    parts = line.split(None, 10)
                    if len(parts) >= 11:
                        name = parts[10].split("/")[-1][:40]
                        if filter_name and filter_name not in name.lower():
                            continue
                        processes.append({
                            "pid": parts[1],
                            "user": parts[0],
                            "cpu_percent": parts[2],
                            "mem_percent": parts[3],
                            "name": name,
                        })
                return SkillResult.ok(
                    data={"processes": processes, "count": len(processes)},
                    metadata={"source": "ps aux", "psutil": False},
                )
            except Exception as exc:
                return SkillResult.failure(error=f"Failed to list processes: {exc}")

        processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status", "username"]):
            try:
                info = proc.info
                name = info.get("name", "") or ""
                if filter_name and filter_name not in name.lower():
                    continue
                processes.append({
                    "pid": info.get("pid"),
                    "name": name,
                    "cpu_percent": round(info.get("cpu_percent") or 0.0, 2),
                    "memory_percent": round(info.get("memory_percent") or 0.0, 2),
                    "status": info.get("status", ""),
                    "username": info.get("username", ""),
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        sort_key = {
            "cpu": lambda p: p["cpu_percent"],
            "memory": lambda p: p["memory_percent"],
            "pid": lambda p: p["pid"] or 0,
            "name": lambda p: (p["name"] or "").lower(),
        }[sort_by]
        processes.sort(key=sort_key, reverse=sort_by in ("cpu", "memory"))
        processes = processes[:max_results]

        return SkillResult.ok(
            data={"processes": processes, "count": len(processes)},
            metadata={"psutil": True, "sort_by": sort_by},
        )


# ---------------------------------------------------------------------------
# EnvironmentSkill
# ---------------------------------------------------------------------------


class EnvironmentSkill(BaseSkill):
    """Read or write environment variables from a safe allowlist."""

    @property
    def name(self) -> str:
        return "environment"

    @property
    def description(self) -> str:
        return (
            "Read or set environment variables. "
            "Access is limited to a predefined safe list; secrets are blocked."
        )

    @property
    def category(self) -> str:
        return "system"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter(
                "action",
                "string",
                "Action to perform: 'get', 'set', or 'list'.",
                required=True,
            ),
            SkillParameter(
                "name",
                "string",
                "Variable name (required for 'get' and 'set').",
                required=False,
                default="",
            ),
            SkillParameter(
                "value",
                "string",
                "Value to set (required for 'set' action).",
                required=False,
                default="",
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        action = params.get("action", "")
        if action not in ("get", "set", "list"):
            raise ValueError("'action' must be 'get', 'set', or 'list'.")
        if action in ("get", "set"):
            name = params.get("name", "").strip().upper()
            if not name:
                raise ValueError(f"'name' is required for action '{action}'.")
            if any(blocked in name for blocked in _BLOCKED_ENV):
                raise ValueError(f"Access to variable '{name}' is blocked for security reasons.")
        if action == "set" and "value" not in params:
            raise ValueError("'value' is required for action 'set'.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        action = params["action"]
        name = params.get("name", "").strip().upper()

        if action == "list":
            env_data = {
                k: v
                for k, v in os.environ.items()
                if k.upper() in _ALLOWED_ENV_READ
                and not any(blocked in k.upper() for blocked in _BLOCKED_ENV)
            }
            return SkillResult.ok(
                data={"variables": env_data, "count": len(env_data)},
                metadata={"action": "list"},
            )

        if action == "get":
            if name not in _ALLOWED_ENV_READ:
                return SkillResult.failure(
                    error=(
                        f"Variable '{name}' is not in the allowed read list. "
                        "Use action='list' to see available variables."
                    )
                )
            value = os.environ.get(name)
            return SkillResult.ok(
                data={"name": name, "value": value, "exists": value is not None},
                metadata={"action": "get"},
            )

        # action == "set"
        value = str(params.get("value", ""))
        os.environ[name] = value
        logger.info("EnvironmentSkill: set %s", name)
        return SkillResult.ok(
            data={"name": name, "value": value},
            metadata={"action": "set"},
        )


# ---------------------------------------------------------------------------
# NetworkInfoSkill
# ---------------------------------------------------------------------------


class NetworkInfoSkill(BaseSkill):
    """Return network interface configuration and connectivity information."""

    @property
    def name(self) -> str:
        return "network_info"

    @property
    def description(self) -> str:
        return (
            "Retrieve network interface addresses, connection counts, "
            "and basic connectivity information."
        )

    @property
    def category(self) -> str:
        return "system"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter(
                "include_loopback",
                "boolean",
                "Include loopback interfaces (e.g. 127.0.0.1).",
                required=False,
                default=False,
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        pass  # All parameters are optional with safe defaults.

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        include_loopback = bool(params.get("include_loopback", False))

        if _PSUTIL_AVAILABLE:
            interfaces = []
            for iface_name, addrs in psutil.net_if_addrs().items():
                iface_addresses = []
                for addr in addrs:
                    addr_info: Dict[str, Any] = {
                        "family": str(addr.family),
                        "address": addr.address,
                        "netmask": addr.netmask,
                        "broadcast": addr.broadcast,
                    }
                    iface_addresses.append(addr_info)

                is_loopback = iface_name.startswith("lo") or any(
                    a["address"] in ("127.0.0.1", "::1") for a in iface_addresses
                )
                if not include_loopback and is_loopback:
                    continue

                stats = psutil.net_if_stats().get(iface_name)
                interfaces.append({
                    "name": iface_name,
                    "addresses": iface_addresses,
                    "is_up": stats.isup if stats else None,
                    "speed_mbps": stats.speed if stats else None,
                    "mtu": stats.mtu if stats else None,
                })

            net_io = psutil.net_io_counters()
            connections = len(psutil.net_connections(kind="inet"))

            return SkillResult.ok(
                data={
                    "hostname": platform.node(),
                    "interfaces": interfaces,
                    "connections": connections,
                    "bytes_sent_mb": round(net_io.bytes_sent / 1024 / 1024, 2),
                    "bytes_recv_mb": round(net_io.bytes_recv / 1024 / 1024, 2),
                },
                metadata={"psutil": True},
            )

        # Fallback: socket-based info
        import socket

        interfaces: List[Dict[str, Any]] = []
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            interfaces.append({
                "name": "primary",
                "addresses": [{"address": local_ip}],
            })
        except socket.error:
            hostname = platform.node()

        return SkillResult.ok(
            data={
                "hostname": hostname,
                "interfaces": interfaces,
                "note": "Install psutil for detailed network information.",
            },
            metadata={"psutil": False},
        )


__all__ = [
    "SystemInfoSkill",
    "RunCommandSkill",
    "ProcessListSkill",
    "EnvironmentSkill",
    "NetworkInfoSkill",
]
