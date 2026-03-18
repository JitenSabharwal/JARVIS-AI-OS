from __future__ import annotations

from core.config import JARVISConfig
from infrastructure.local_model_runtime import LocalModelRuntimeManager, LocalModelSpec
from infrastructure.model_provider_factory import build_model_router_from_config
from infrastructure.model_router import ModelRequest


def test_build_model_router_from_config_for_ollama_and_cohere() -> None:
    cfg = JARVISConfig()
    cfg.model.enabled = True
    cfg.model.local_provider = "ollama"
    cfg.model.api_provider = "cohere"
    cfg.model.cohere_api_key = "cohere-key"
    cfg.model.ollama_text_model = "qwen2.5:7b"
    cfg.model.memory_budget_gb = 35
    cfg.model.total_memory_gb = 48

    router = build_model_router_from_config(cfg)
    assert router is not None
    request = ModelRequest(
        prompt="hello",
        task_type="information_query",
    )
    decision = router.route(request)
    assert decision.chain
    assert decision.primary in {"cohere", "ollama"}


def test_local_model_runtime_budget_unloads_lru() -> None:
    runtime = LocalModelRuntimeManager(memory_budget_gb=10, total_memory_gb=48, auto_unload=True)
    runtime.mark_loaded(LocalModelSpec(name="m1", size_gb=4))
    runtime.mark_loaded(LocalModelSpec(name="m2", size_gb=4))
    runtime.mark_loaded(LocalModelSpec(name="m3", size_gb=1))

    unloaded = runtime.ensure_capacity(4)
    assert unloaded
    assert runtime.used_memory_gb() <= 10
    status = runtime.status()
    assert status["memory_budget_gb"] == 10
    assert status["used_memory_gb"] <= 10


def test_local_model_runtime_parallel_budget() -> None:
    runtime = LocalModelRuntimeManager(memory_budget_gb=35, total_memory_gb=48, max_parallel_models=3)
    assert runtime.can_run_parallel([8, 12, 10]) is True
    assert runtime.can_run_parallel([20, 10, 8]) is False
