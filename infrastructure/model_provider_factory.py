"""
Factory to build model router/providers from global config.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from core.config import JARVISConfig, get_config
from infrastructure.local_model_runtime import LocalModelRuntimeManager
from infrastructure.model_providers import CohereProvider, MLXProvider, OllamaProvider
from infrastructure.model_router import ModelRouter


def _parse_model_billions(name: str) -> float:
    text = str(name or "").strip().lower()
    if not text:
        return 0.0
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*b\b", text)
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except Exception:
        return 0.0


def _allowed_model(name: str, *, max_billions: float) -> bool:
    if not str(name or "").strip():
        return False
    b = _parse_model_billions(name)
    if b <= 0:
        return True
    return b <= max_billions


def build_model_router_from_config(config: Optional[JARVISConfig] = None) -> ModelRouter | None:
    cfg = config or get_config()
    model_cfg = cfg.model
    if not model_cfg.enabled:
        return None
    max_local_billions_raw = str(os.getenv("JARVIS_MAX_LOCAL_MODEL_BILLIONS", "34")).strip() or "34"
    try:
        max_local_billions = max(1.0, float(max_local_billions_raw))
    except Exception:
        max_local_billions = 34.0

    base_model_name = str(model_cfg.base_model_name or "").strip()
    if not base_model_name:
        if model_cfg.local_provider == "mlx":
            base_model_name = str(model_cfg.mlx_text_model_small or model_cfg.mlx_text_model or "").strip()
        elif model_cfg.local_provider == "ollama":
            base_model_name = str(model_cfg.ollama_text_model or "").strip()
    if base_model_name and not _allowed_model(base_model_name, max_billions=max_local_billions):
        if model_cfg.local_provider == "mlx":
            base_model_name = str(model_cfg.mlx_text_model_small or model_cfg.mlx_text_model_coding or "").strip()
        elif model_cfg.local_provider == "ollama":
            base_model_name = str(model_cfg.ollama_text_model or "").strip()

    runtime = LocalModelRuntimeManager(
        memory_budget_gb=model_cfg.memory_budget_gb,
        total_memory_gb=model_cfg.total_memory_gb,
        max_parallel_models=model_cfg.max_parallel_models,
        large_model_threshold_gb=model_cfg.large_model_threshold_gb,
        single_large_model_mode=model_cfg.single_large_model_mode,
        auto_unload=model_cfg.auto_unload,
        keep_base_model_loaded=model_cfg.keep_base_model_loaded,
        base_model_name=base_model_name,
        unload_base_when_required=model_cfg.unload_base_when_required,
    )

    local_provider = None
    if model_cfg.local_provider == "ollama":
        ollama_text_model = (
            str(model_cfg.ollama_text_model or "").strip()
            if _allowed_model(str(model_cfg.ollama_text_model or "").strip(), max_billions=max_local_billions)
            else ""
        )
        ollama_image_model = (
            str(model_cfg.ollama_image_model or "").strip()
            if _allowed_model(str(model_cfg.ollama_image_model or "").strip(), max_billions=max_local_billions)
            else ""
        )
        ollama_audio_model = str(model_cfg.ollama_audio_model or "").strip()
        local_provider = OllamaProvider(
            base_url=model_cfg.ollama_base_url,
            text_model=ollama_text_model,
            image_model=ollama_image_model,
            audio_model=ollama_audio_model,
            timeout_seconds=model_cfg.local_timeout_seconds,
            runtime_manager=runtime,
            model_sizes_gb={
                ollama_text_model: model_cfg.ollama_text_model_size_gb,
                ollama_image_model: model_cfg.ollama_image_model_size_gb,
                ollama_audio_model: model_cfg.ollama_audio_model_size_gb,
            },
        )
    elif model_cfg.local_provider == "mlx":
        mlx_text_small = str(model_cfg.mlx_text_model_small or "").strip()
        mlx_text_coding = str(model_cfg.mlx_text_model_coding or "").strip()
        mlx_text = str(model_cfg.mlx_text_model or "").strip()
        mlx_text_reasoning = str(model_cfg.mlx_text_model_reasoning or "").strip()
        mlx_text_deep = str(model_cfg.mlx_text_model_deep_research or "").strip()
        mlx_image = str(model_cfg.mlx_image_model or "").strip()
        mlx_audio = str(model_cfg.mlx_audio_model or "").strip()

        # Enforce local thermal cap by parameter-scale hint in model names.
        if not _allowed_model(mlx_text, max_billions=max_local_billions):
            mlx_text = mlx_text_small or mlx_text_coding or ""
        if not _allowed_model(mlx_text_reasoning, max_billions=max_local_billions):
            mlx_text_reasoning = ""
        if not _allowed_model(mlx_text_deep, max_billions=max_local_billions):
            mlx_text_deep = ""
        if not _allowed_model(mlx_image, max_billions=max_local_billions):
            mlx_image = ""

        local_provider = MLXProvider(
            enabled=model_cfg.mlx_enabled,
            python_executable=model_cfg.mlx_command_python,
            text_runner_module=model_cfg.mlx_text_runner_module,
            image_runner_module=model_cfg.mlx_image_runner_module,
            audio_runner_module=model_cfg.mlx_audio_runner_module,
            timeout_seconds=model_cfg.mlx_timeout_seconds,
            temperature=model_cfg.mlx_temperature,
            max_tokens=model_cfg.mlx_max_tokens,
            dry_run=model_cfg.mlx_dry_run,
            runtime_manager=runtime,
            model_sizes_gb={
                mlx_text: model_cfg.mlx_text_model_size_gb,
                mlx_text_small: model_cfg.mlx_text_model_small_size_gb,
                mlx_text_coding: model_cfg.mlx_text_model_coding_size_gb,
                mlx_text_reasoning: model_cfg.mlx_text_model_reasoning_size_gb,
                mlx_text_deep: model_cfg.mlx_text_model_deep_research_size_gb,
                mlx_image: model_cfg.mlx_image_model_size_gb,
                mlx_audio: model_cfg.mlx_audio_model_size_gb,
            },
            text_model=mlx_text,
            text_model_small=mlx_text_small,
            text_model_coding=mlx_text_coding,
            text_model_reasoning=mlx_text_reasoning,
            text_model_deep_research=mlx_text_deep,
            image_model=mlx_image,
            audio_model=mlx_audio,
            enable_reasoning_model=model_cfg.mlx_enable_reasoning_model and bool(mlx_text_reasoning),
            enable_deep_research_model=model_cfg.mlx_enable_deep_research_model and bool(mlx_text_deep),
            persistent_enabled=model_cfg.mlx_persistent_enabled,
            persistent_base_url=model_cfg.mlx_persistent_base_url,
            persistent_endpoint=model_cfg.mlx_persistent_endpoint,
            persistent_api_key=model_cfg.mlx_persistent_api_key,
            persistent_fallback_cli=model_cfg.mlx_persistent_fallback_cli,
        )

    api_provider = None
    if model_cfg.api_provider == "cohere" and model_cfg.cohere_api_key:
        api_provider = CohereProvider(
            api_key=model_cfg.cohere_api_key,
            model=model_cfg.cohere_text_model,
            base_url=model_cfg.cohere_base_url,
            timeout_seconds=model_cfg.api_timeout_seconds,
        )

    if not local_provider and not api_provider:
        return None

    return ModelRouter(
        local_provider=local_provider,
        api_provider=api_provider,
        fallback_enabled=model_cfg.fallback_enabled,
        shadow_mode=model_cfg.shadow_mode,
    )
