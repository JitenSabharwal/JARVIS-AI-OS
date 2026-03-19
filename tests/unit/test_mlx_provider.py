from __future__ import annotations

import pytest

from infrastructure.local_model_runtime import LocalModelRuntimeManager
from infrastructure.model_providers import MLXProvider
from infrastructure.model_router import ModelRequest


@pytest.mark.asyncio
async def test_mlx_provider_task_model_selection_dry_run() -> None:
    runtime = LocalModelRuntimeManager(
        memory_budget_gb=35,
        total_memory_gb=48,
        max_parallel_models=3,
        large_model_threshold_gb=18,
        single_large_model_mode=True,
    )
    provider = MLXProvider(
        enabled=True,
        python_executable="python3",
        dry_run=True,
        runtime_manager=runtime,
        model_sizes_gb={
            "small-model": 4,
            "coding-model": 10,
            "reasoning-model": 28,
            "deep-model": 34,
            "general-model": 24,
            "image-model": 3,
            "audio-model": 3,
        },
        text_model="general-model",
        text_model_small="small-model",
        text_model_coding="coding-model",
        text_model_reasoning="reasoning-model",
        text_model_deep_research="deep-model",
        image_model="image-model",
        audio_model="audio-model",
    )

    coding_resp = await provider.generate(
        ModelRequest(prompt="write api code", task_type="coding", modality="text")
    )
    assert "coding-model" in coding_resp

    deep_resp = await provider.generate(
        ModelRequest(prompt="long study", task_type="research_query", modality="text")
    )
    assert "deep-model" in deep_resp

    simple_resp = await provider.generate(
        ModelRequest(prompt="hi", task_type="greeting", modality="text")
    )
    assert "small-model" in simple_resp


def test_mlx_provider_is_unavailable_when_python_binary_missing() -> None:
    provider = MLXProvider(
        enabled=True,
        python_executable="/path/that/does/not/exist/python3",
        text_model="general-model",
    )
    assert provider.is_available() is False
