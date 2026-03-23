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

    forced_large = await provider.generate(
        ModelRequest(
            prompt="hello",
            task_type="greeting",
            modality="text",
            metadata={"model_tier": "large"},
        )
    )
    assert "general-model" in forced_large

    forced_small = await provider.generate(
        ModelRequest(
            prompt="hello",
            task_type="analysis",
            modality="text",
            metadata={"model_tier": "small"},
        )
    )
    assert "small-model" in forced_small

    voice_resp = await provider.generate(
        ModelRequest(prompt="hello from mic", task_type="greeting", modality="voice")
    )
    assert "small-model" in voice_resp


def test_mlx_provider_is_unavailable_when_python_binary_missing() -> None:
    provider = MLXProvider(
        enabled=True,
        python_executable="/path/that/does/not/exist/python3",
        text_model="general-model",
    )
    assert provider.is_available() is False


def test_mlx_provider_normalizes_hf_cache_folder_model_id_for_text_command() -> None:
    provider = MLXProvider(
        enabled=True,
        python_executable="python3",
        text_runner_module="mlx_lm.generate",
        text_model="models--mlx-community--Qwen3.5-4B-MLX-4bit",
    )
    cmd = provider._build_text_command(
        model_name="models--mlx-community--Qwen3.5-4B-MLX-4bit",
        prompt="hello",
    )
    model_value = cmd[cmd.index("--model") + 1]
    assert model_value == "mlx-community/Qwen3.5-4B-MLX-4bit"


def test_mlx_provider_keeps_regular_repo_id_for_text_command() -> None:
    provider = MLXProvider(
        enabled=True,
        python_executable="python3",
        text_runner_module="mlx_lm.generate",
        text_model="mlx-community/Qwen3.5-4B-MLX-4bit",
    )
    cmd = provider._build_text_command(
        model_name="mlx-community/Qwen3.5-4B-MLX-4bit",
        prompt="hello",
    )
    model_value = cmd[cmd.index("--model") + 1]
    assert model_value == "mlx-community/Qwen3.5-4B-MLX-4bit"


def test_mlx_provider_uses_new_mlx_lm_cli_form_for_default_text_runner() -> None:
    provider = MLXProvider(
        enabled=True,
        python_executable="python3",
        text_runner_module="mlx_lm.generate",
        text_model="mlx-community/Qwen3.5-4B-MLX-4bit",
    )
    cmd = provider._build_text_command(
        model_name="mlx-community/Qwen3.5-4B-MLX-4bit",
        prompt="hello",
    )
    assert cmd[:4] == ["python3", "-m", "mlx_lm", "generate"]


def test_mlx_provider_normalize_output_ignores_mlx_telemetry_footer() -> None:
    raw = """
    Today in Berlin it is cloudy with light wind.
    Peak memory: 18.704 GB
    """
    assert (
        MLXProvider._normalize_output(raw) == "Today in Berlin it is cloudy with light wind."
    )


def test_mlx_provider_normalize_output_preserves_multiline_answer() -> None:
    raw = """
    Part one.
    Part two.
    Tokens per second: 31.2
    """
    assert MLXProvider._normalize_output(raw) == "Part one.\nPart two."


def test_mlx_provider_normalize_output_strips_deprecation_and_reasoning_leak() -> None:
    raw = """
    Calling `python -m mlx_lm.generate...` directly is deprecated. Use `mlx_lm.generate...` or `python -m mlx_lm generate ...` instead.
    <think>
    hidden reasoning
    </think>
    # Response as JARVIS
    It is sunny today.
    # Reasoning/Explanation
    Should not appear.
    """
    assert MLXProvider._normalize_output(raw) == "# Response as JARVIS\nIt is sunny today."


def test_mlx_provider_normalize_output_returns_empty_for_separator_only() -> None:
    assert MLXProvider._normalize_output("==========") == ""


def test_mlx_provider_normalize_output_returns_empty_for_separator_and_telemetry() -> None:
    raw = """
    ==========
    Peak memory: 18.704 GB
    """
    assert MLXProvider._normalize_output(raw) == ""


def test_mlx_provider_normalize_output_strips_context_fusion_analysis_wrapper() -> None:
    raw = """
    Let me analyze this conversation step by step:
    1. First turn...
    </think>
    **Analysis of Context Fusion Data:**
    | Element | Turn 1 | Turn 2 |
    **JARVIS Response:**
    Excellent! Now I have the complete information needed.
    India is a large country with diverse climates.
    """
    out = MLXProvider._normalize_output(raw)
    assert "Let me analyze this conversation step by step" not in out
    assert "Analysis of Context Fusion Data" not in out
    assert out.startswith("Excellent! Now I have the complete information needed.")


def test_mlx_provider_sanitize_thinking_process_extracts_final_response_line() -> None:
    raw = """
    Thinking Process:
    1. Analyze request
    Final Decision: "It is around 18 degrees Celsius in Munich."
    """
    assert MLXProvider._normalize_output(raw) == "It is around 18 degrees Celsius in Munich."


def test_mlx_provider_sanitize_thinking_process_only_returns_empty() -> None:
    raw = """
    Thinking Process:
    1. Analyze the request
    2. Check constraints
    * Wait, one more check
    """
    assert MLXProvider._normalize_output(raw) == ""


def test_mlx_provider_strips_leading_analysis_preamble_and_keeps_answer() -> None:
    raw = """
    Let me analyze this conversation carefully:
    1. User says "Hi" - standard greeting
    2. Assistant responds with "Hi there! What can I do for you?" - appropriate response
    3. User says "Can you help me React" - this is the current message
    The user's message appears to be incomplete.
    Since this is a help request, I should ask for clarification.
    Absolutely! I'd be happy to help you with React. What specifically would you like assistance with?
    """
    out = MLXProvider._normalize_output(raw)
    assert out.startswith("Absolutely! I'd be happy to help you with React.")
    assert "Let me analyze this conversation carefully" not in out


@pytest.mark.asyncio
async def test_mlx_provider_uses_small_model_for_weather_query() -> None:
    provider = MLXProvider(
        enabled=True,
        python_executable="python3",
        dry_run=True,
        text_model="general-model",
        text_model_small="small-model",
    )
    out = await provider.generate(
        ModelRequest(prompt="temp in Munich", task_type="weather_query", modality="text")
    )
    assert "small-model" in out


def test_mlx_provider_extract_persistent_text_openai_shape() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello from persistent mlx.",
                }
            }
        ]
    }
    assert MLXProvider._extract_persistent_text(payload) == "Hello from persistent mlx."


@pytest.mark.asyncio
async def test_mlx_provider_persistent_mode_falls_back_to_cli() -> None:
    provider = MLXProvider(
        enabled=True,
        python_executable="python3",
        text_runner_module="mlx_lm.generate",
        text_model="general-model",
        persistent_enabled=True,
        persistent_fallback_cli=True,
    )

    async def _fail_persistent(*, model_name: str, prompt: str, max_tokens: int) -> str:
        raise RuntimeError("persistent unavailable")

    async def _ok_cli(cmd: list[str]) -> str:
        return "cli fallback output"

    provider._run_persistent_text_request = _fail_persistent  # type: ignore[method-assign]
    provider._run_subprocess = _ok_cli  # type: ignore[method-assign]
    out = await provider.generate(
        ModelRequest(prompt="hello", task_type="status_query", modality="text")
    )
    assert out == "cli fallback output"


def test_mlx_provider_caps_tokens_for_summarization_and_weather() -> None:
    provider = MLXProvider(
        enabled=True,
        python_executable="python3",
        text_runner_module="mlx_lm.generate",
        max_tokens=2048,
        text_model="general-model",
    )
    sum_cmd = provider._build_text_command(
        model_name="general-model",
        prompt="summarize this",
        max_tokens=provider._max_tokens_for_request(
            ModelRequest(prompt="x", task_type="summarization", modality="text")
        ),
    )
    weather_cmd = provider._build_text_command(
        model_name="general-model",
        prompt="weather",
        max_tokens=provider._max_tokens_for_request(
            ModelRequest(prompt="x", task_type="weather_query", modality="text")
        ),
    )
    assert sum_cmd[sum_cmd.index("--max-tokens") + 1] == "256"
    assert weather_cmd[weather_cmd.index("--max-tokens") + 1] == "192"
