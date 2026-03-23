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


def test_build_model_router_sets_default_base_model_for_mlx_runtime() -> None:
    cfg = JARVISConfig()
    cfg.model.enabled = True
    cfg.model.local_provider = "mlx"
    cfg.model.api_provider = ""
    cfg.model.mlx_enabled = True
    cfg.model.mlx_text_model_small = "small-mlx"
    cfg.model.mlx_text_model = "large-mlx"
    cfg.model.keep_base_model_loaded = True
    cfg.model.base_model_name = ""
    cfg.model.unload_base_when_required = True

    router = build_model_router_from_config(cfg)
    assert router is not None
    local = router._local_provider  # type: ignore[attr-defined]
    assert local is not None
    runtime = local._runtime  # type: ignore[attr-defined]
    assert runtime is not None
    status = runtime.status()
    assert status.get("base_model_name") == "small-mlx"
    assert status.get("keep_base_model_loaded") is True


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


def test_local_model_runtime_keeps_base_model_loaded_when_possible() -> None:
    runtime = LocalModelRuntimeManager(
        memory_budget_gb=10,
        total_memory_gb=48,
        auto_unload=True,
        keep_base_model_loaded=True,
        base_model_name="base",
        unload_base_when_required=True,
    )
    runtime.mark_loaded(LocalModelSpec(name="base", size_gb=4))
    runtime.mark_loaded(LocalModelSpec(name="heavy", size_gb=5))
    runtime.mark_loaded(LocalModelSpec(name="tiny", size_gb=1))

    unloaded = runtime.ensure_capacity(2)
    assert "base" not in unloaded
    status = runtime.status()
    loaded_names = {str(m["name"]) for m in status["loaded_models"]}  # type: ignore[index]
    assert "base" in loaded_names


def test_local_model_runtime_unloads_base_model_only_as_last_resort() -> None:
    runtime = LocalModelRuntimeManager(
        memory_budget_gb=10,
        total_memory_gb=48,
        auto_unload=True,
        keep_base_model_loaded=True,
        base_model_name="base",
        unload_base_when_required=True,
    )
    runtime.mark_loaded(LocalModelSpec(name="base", size_gb=4))
    runtime.mark_loaded(LocalModelSpec(name="small", size_gb=2))

    unloaded = runtime.ensure_capacity(8)
    assert "small" in unloaded
    assert "base" in unloaded


def test_local_model_runtime_does_not_unload_when_target_already_loaded() -> None:
    runtime = LocalModelRuntimeManager(
        memory_budget_gb=35,
        total_memory_gb=48,
        auto_unload=True,
        keep_base_model_loaded=True,
        base_model_name="base-35b",
        unload_base_when_required=True,
    )
    runtime.mark_loaded(LocalModelSpec(name="base-35b", size_gb=24))
    runtime.mark_loaded(LocalModelSpec(name="other", size_gb=8))
    unloaded = runtime.ensure_capacity(24, target_model_name="base-35b")
    assert unloaded == []
    status = runtime.status()
    loaded_names = {str(m["name"]) for m in status["loaded_models"]}  # type: ignore[index]
    assert "base-35b" in loaded_names


def test_local_model_runtime_parallel_budget() -> None:
    runtime = LocalModelRuntimeManager(memory_budget_gb=35, total_memory_gb=48, max_parallel_models=3)
    assert runtime.can_run_parallel([8, 12, 10]) is True
    assert runtime.can_run_parallel([20, 10, 8]) is False


def test_local_model_runtime_large_model_single_flight_policy() -> None:
    runtime = LocalModelRuntimeManager(
        memory_budget_gb=35,
        total_memory_gb=48,
        max_parallel_models=3,
        large_model_threshold_gb=18,
        single_large_model_mode=True,
    )
    assert runtime.can_run_parallel([8, 10]) is True
    assert runtime.can_run_parallel([20]) is True
    assert runtime.can_run_parallel([20, 8]) is False
    assert runtime.can_run_parallel([20, 19]) is False

    runtime.mark_loaded(LocalModelSpec(name="small", size_gb=8))
    runtime.mark_loaded(LocalModelSpec(name="large", size_gb=20))
    runtime.mark_in_use("small")
    try:
        try:
            runtime.mark_in_use("large")
            assert False, "expected large model single-flight policy failure"
        except RuntimeError:
            pass
    finally:
        runtime.mark_released("small")

    runtime.mark_in_use("large")
    try:
        try:
            runtime.mark_in_use("small")
            assert False, "expected small model blocked while large is active"
        except RuntimeError:
            pass
    finally:
        runtime.mark_released("large")
