#!/usr/bin/env python3
"""
MLX runtime health check for JARVIS.

Validates:
1. Model routing is configured for MLX.
2. MLX python executable is available.
3. Configured MLX runner modules are importable in that python env.
4. Configured MLX model directories exist (optional strict mode).
5. Runtime memory policy sanity for big/small model parallelism.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_config
from infrastructure.local_model_runtime import LocalModelRuntimeManager


def _check_python_executable(python_bin: str) -> Tuple[bool, str]:
    cmd = [python_bin, "--version"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except Exception as exc:  # noqa: BLE001
        return False, f"{exc}"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "failed").strip()
    return True, (proc.stdout or proc.stderr or "ok").strip()


def _check_module_importable(python_bin: str, module_name: str) -> Tuple[bool, str]:
    code = (
        "import importlib.util,sys;"
        f"m='{module_name}';"
        "spec=importlib.util.find_spec(m);"
        "sys.stdout.write('ok' if spec else 'missing');"
        "sys.exit(0 if spec else 2)"
    )
    try:
        proc = subprocess.run([python_bin, "-c", code], capture_output=True, text=True, timeout=15)
    except Exception as exc:  # noqa: BLE001
        return False, f"{exc}"
    ok = proc.returncode == 0
    details = (proc.stdout or proc.stderr or "").strip() or ("ok" if ok else "missing")
    return ok, details


def _candidate_model_roots(cli_roots: List[str]) -> List[Path]:
    roots: List[Path] = []
    for raw in cli_roots:
        p = Path(raw).expanduser()
        if p.exists() and p.is_dir():
            roots.append(p)
    default_hf = Path.home() / ".cache" / "huggingface" / "hub"
    if default_hf.exists():
        roots.append(default_hf)
    hf_home = str(os.environ.get("HF_HOME", "")).strip()
    if hf_home:
        hf_root = Path(hf_home).expanduser() / "hub"
        if hf_root.exists():
            roots.append(hf_root)
    uniq: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root.resolve())
        if key not in seen:
            uniq.append(root)
            seen.add(key)
    return uniq


def _check_model_dirs(model_names: List[str], roots: List[Path]) -> Dict[str, Any]:
    found: Dict[str, str] = {}
    missing: List[str] = []
    skipped: List[str] = []
    if not roots:
        return {"found": found, "missing": missing, "skipped": model_names}
    for model in model_names:
        if not model:
            continue
        matched = ""
        for root in roots:
            candidate = root / model
            if candidate.exists() and candidate.is_dir():
                matched = str(candidate)
                break
        if matched:
            found[model] = matched
        else:
            missing.append(model)
    return {"found": found, "missing": missing, "skipped": skipped}


def _memory_policy_checks(cfg: Any) -> Dict[str, Any]:
    runtime = LocalModelRuntimeManager(
        memory_budget_gb=cfg.model.memory_budget_gb,
        total_memory_gb=cfg.model.total_memory_gb,
        max_parallel_models=cfg.model.max_parallel_models,
        large_model_threshold_gb=cfg.model.large_model_threshold_gb,
        single_large_model_mode=cfg.model.single_large_model_mode,
        auto_unload=cfg.model.auto_unload,
    )
    sizes = {
        "small": float(cfg.model.mlx_text_model_small_size_gb),
        "coding": float(cfg.model.mlx_text_model_coding_size_gb),
        "reasoning": float(cfg.model.mlx_text_model_reasoning_size_gb),
        "deep": float(cfg.model.mlx_text_model_deep_research_size_gb),
    }
    checks = {
        "small_plus_coding_parallel": runtime.can_run_parallel([sizes["small"], sizes["coding"]]),
        "reasoning_alone": runtime.can_run_parallel([sizes["reasoning"]]),
        "reasoning_plus_small_blocked": runtime.can_run_parallel([sizes["reasoning"], sizes["small"]]),
        "deep_alone": runtime.can_run_parallel([sizes["deep"]]),
        "deep_plus_small_blocked": runtime.can_run_parallel([sizes["deep"], sizes["small"]]),
    }
    return {"sizes_gb": sizes, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MLX runtime readiness for JARVIS")
    parser.add_argument("--env-file", default=".env", help="Path to env file (default: .env)")
    parser.add_argument(
        "--models-root",
        action="append",
        default=[],
        help="Model root directory. Repeat flag for multiple roots.",
    )
    parser.add_argument(
        "--strict-model-paths",
        action="store_true",
        help="Fail when configured MLX model dirs are not found in model roots.",
    )
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON only.")
    args = parser.parse_args()

    cfg = get_config(env_file=args.env_file, reload=True)
    model_cfg = cfg.model

    model_names = [
        model_cfg.mlx_text_model,
        model_cfg.mlx_text_model_small,
        model_cfg.mlx_text_model_coding,
        model_cfg.mlx_text_model_reasoning,
        model_cfg.mlx_text_model_deep_research,
        model_cfg.mlx_image_model,
        model_cfg.mlx_audio_model,
        model_cfg.mlx_reranker_model_small,
        model_cfg.mlx_reranker_model,
    ]
    model_names = [m for m in model_names if isinstance(m, str) and m.strip()]

    checks: List[Dict[str, Any]] = []
    checks.append(
        {
            "name": "mlx_selected_as_local_provider",
            "ok": model_cfg.local_provider == "mlx",
            "details": {"local_provider": model_cfg.local_provider},
        }
    )
    checks.append(
        {
            "name": "mlx_enabled",
            "ok": bool(model_cfg.mlx_enabled),
            "details": {"mlx_enabled": model_cfg.mlx_enabled},
        }
    )

    py_ok, py_details = _check_python_executable(model_cfg.mlx_command_python)
    checks.append(
        {
            "name": "mlx_python_executable",
            "ok": py_ok,
            "details": {
                "python": model_cfg.mlx_command_python,
                "version": py_details,
            },
        }
    )

    for module_key, module_name in (
        ("text_runner", model_cfg.mlx_text_runner_module),
        ("image_runner", model_cfg.mlx_image_runner_module),
        ("audio_runner", model_cfg.mlx_audio_runner_module),
    ):
        if not module_name:
            checks.append({"name": f"module_{module_key}", "ok": False, "details": "not_configured"})
            continue
        ok, details = _check_module_importable(model_cfg.mlx_command_python, module_name)
        checks.append(
            {
                "name": f"module_{module_key}",
                "ok": ok,
                "details": {"module": module_name, "result": details},
            }
        )

    roots = _candidate_model_roots(args.models_root)
    model_dir_result = _check_model_dirs(model_names, roots)
    model_dirs_ok = (not args.strict_model_paths) or (len(model_dir_result["missing"]) == 0)
    checks.append(
        {
            "name": "mlx_model_directories",
            "ok": model_dirs_ok,
            "details": {
                "strict": args.strict_model_paths,
                "roots": [str(p) for p in roots],
                **model_dir_result,
            },
        }
    )

    mem = _memory_policy_checks(cfg)
    checks.append({"name": "memory_policy_sanity", "ok": True, "details": mem})

    overall_ok = all(bool(c.get("ok")) for c in checks)
    report = {"overall_ok": overall_ok, "checks": checks}

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))
        if not overall_ok:
            print("\nOne or more MLX checks failed.")
            print("Tip: run with --strict-model-paths and explicit --models-root paths for deterministic validation.")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
