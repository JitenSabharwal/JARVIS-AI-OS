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

    runtime = LocalModelRuntimeManager(
        memory_budget_gb=model_cfg.memory_budget_gb,
        total_memory_gb=model_cfg.total_memory_gb,
        max_parallel_models=model_cfg.max_parallel_models,
        auto_unload=model_cfg.auto_unload,
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
        local_provider = MLXProvider(enabled=model_cfg.mlx_enabled)

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
