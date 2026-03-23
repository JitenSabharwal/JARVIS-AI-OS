"""
Factory to build model router/providers from global config.
"""

from __future__ import annotations

from typing import Optional

from core.config import JARVISConfig, get_config
from infrastructure.local_model_runtime import LocalModelRuntimeManager
from infrastructure.model_providers import CohereProvider, MLXProvider, OllamaProvider
from infrastructure.model_router import ModelRouter


def build_model_router_from_config(config: Optional[JARVISConfig] = None) -> ModelRouter | None:
    cfg = config or get_config()
    model_cfg = cfg.model
    if not model_cfg.enabled:
        return None

    base_model_name = str(model_cfg.base_model_name or "").strip()
    if not base_model_name:
        if model_cfg.local_provider == "mlx":
            base_model_name = str(model_cfg.mlx_text_model_small or model_cfg.mlx_text_model or "").strip()
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
        local_provider = OllamaProvider(
            base_url=model_cfg.ollama_base_url,
            text_model=model_cfg.ollama_text_model,
            image_model=model_cfg.ollama_image_model,
            audio_model=model_cfg.ollama_audio_model,
            timeout_seconds=model_cfg.local_timeout_seconds,
            runtime_manager=runtime,
            model_sizes_gb={
                model_cfg.ollama_text_model: model_cfg.ollama_text_model_size_gb,
                model_cfg.ollama_image_model: model_cfg.ollama_image_model_size_gb,
                model_cfg.ollama_audio_model: model_cfg.ollama_audio_model_size_gb,
            },
        )
    elif model_cfg.local_provider == "mlx":
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
                model_cfg.mlx_text_model: model_cfg.mlx_text_model_size_gb,
                model_cfg.mlx_text_model_small: model_cfg.mlx_text_model_small_size_gb,
                model_cfg.mlx_text_model_coding: model_cfg.mlx_text_model_coding_size_gb,
                model_cfg.mlx_text_model_reasoning: model_cfg.mlx_text_model_reasoning_size_gb,
                model_cfg.mlx_text_model_deep_research: model_cfg.mlx_text_model_deep_research_size_gb,
                model_cfg.mlx_image_model: model_cfg.mlx_image_model_size_gb,
                model_cfg.mlx_audio_model: model_cfg.mlx_audio_model_size_gb,
            },
            text_model=model_cfg.mlx_text_model,
            text_model_small=model_cfg.mlx_text_model_small,
            text_model_coding=model_cfg.mlx_text_model_coding,
            text_model_reasoning=model_cfg.mlx_text_model_reasoning,
            text_model_deep_research=model_cfg.mlx_text_model_deep_research,
            image_model=model_cfg.mlx_image_model,
            audio_model=model_cfg.mlx_audio_model,
            enable_reasoning_model=model_cfg.mlx_enable_reasoning_model,
            enable_deep_research_model=model_cfg.mlx_enable_deep_research_model,
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
