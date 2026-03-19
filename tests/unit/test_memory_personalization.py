from __future__ import annotations

from datetime import datetime
import time

import pytest

from interfaces.conversation_manager import ConversationContext, ConversationManager
from memory.knowledge_base import KnowledgeBase
from memory.user_profile import UserProfileStore


@pytest.mark.asyncio
async def test_conversation_manager_learns_user_preferences() -> None:
    kb = KnowledgeBase()
    profiles = UserProfileStore()
    manager = ConversationManager(knowledge_base=kb, user_profile_store=profiles)

    session_id = manager.start_session(user_id="alice")
    await manager.process_input(session_id, "My favorite language is python")
    await manager.process_input(session_id, "Call me Alice")

    summary = manager.get_user_profile_summary("alice")
    assert "favorite_language=python" in summary
    assert "display_name=Alice" in summary


def test_knowledge_base_search_semantic_ranking() -> None:
    kb = KnowledgeBase()
    kb.store(
        "python_asyncio_guide",
        "Use asyncio gather and tasks for concurrency in Python.",
        category="programming",
        tags=["python", "asyncio"],
    )
    kb.store(
        "gardening_tips",
        "Water plants in morning and monitor soil moisture.",
        category="hobby",
        tags=["garden", "plants"],
    )

    ranked = kb.search_semantic("python concurrency tasks", max_results=2)
    assert ranked
    assert ranked[0]["key"] == "python_asyncio_guide"
    assert ranked[0]["score"] >= ranked[-1]["score"]


@pytest.mark.asyncio
async def test_conversation_manager_kb_freshness_and_confidence_gates() -> None:
    kb = KnowledgeBase()
    entry_fresh = kb.store(
        "recent_fact",
        "Python supports asyncio for concurrency.",
        category="facts",
        tags=["python", "asyncio"],
    )
    entry_old = kb.store(
        "old_fact",
        "Legacy scheduler details.",
        category="facts",
        tags=["legacy"],
    )
    # Force stale timestamp.
    entry_old.updated_at = "2020-01-01T00:00:00+00:00"

    manager = ConversationManager(
        knowledge_base=kb,
        kb_min_confidence=0.15,
        kb_max_age_days=30,
    )
    sid = manager.start_session("bob")
    response = await manager.process_input(sid, "What do you know about asyncio?")
    assert "Based on what I know:" in response
    assert "recent_fact" in response
    assert "old_fact" not in response


@pytest.mark.asyncio
async def test_conversation_manager_records_context_fusion_for_voice_modality() -> None:
    async def local_handler(_request):
        return "voice-answer"

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text", "voice"},
        )
    )
    manager = ConversationManager(model_router=router)
    sid = manager.start_session("eve")
    out = await manager.process_input(
        sid,
        "hello jarvis",
        modality="voice",
        context={"mic": "default"},
    )
    assert out == "voice-answer"
    ctx = manager.get_context(sid)
    assert ctx is not None
    fusion = ctx.metadata.get("context_fusion")
    assert fusion
    assert fusion.get("modality") == "voice"
    continuity = fusion.get("continuity", {})
    assert continuity.get("last_modality") == "voice"


@pytest.mark.asyncio
async def test_conversation_manager_cross_modal_switch_continuity() -> None:
    async def local_handler(_request):
        return "ok"

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text", "voice", "image"},
        )
    )
    manager = ConversationManager(model_router=router)
    sid = manager.start_session("eve")
    await manager.process_input(sid, "first text turn", modality="text")
    await manager.process_input(
        sid,
        "now check this image",
        modality="image",
        media={"image_url": "https://example.com/a.png"},
    )
    ctx = manager.get_context(sid)
    assert ctx is not None
    continuity = ctx.metadata.get("cross_modal", {})
    assert continuity.get("last_modality") == "image"
    assert continuity.get("switch_count", 0) >= 1
    assert "text" in continuity.get("modality_history", [])
    assert "image" in continuity.get("modality_history", [])


@pytest.mark.asyncio
async def test_conversation_manager_learns_extended_preferences() -> None:
    manager = ConversationManager()
    sid = manager.start_session(user_id="pref-user")
    await manager.process_input(sid, "Use concise tone")
    await manager.process_input(sid, "Notify me daily")
    await manager.process_input(sid, "Risk tolerance is low")
    await manager.process_input(sid, "My routine is standup at 10")

    summary = manager.get_user_profile_summary("pref-user")
    assert "tone=concise" in summary
    assert "cadence=daily" in summary
    assert "risk_tolerance=low" in summary
    assert "routine=standup at 10" in summary


@pytest.mark.asyncio
async def test_summarize_response_for_chat_preserves_short_plain_text() -> None:
    text = "It is currently 18°C in Munich."
    manager = ConversationManager()
    ctx = ConversationContext(session_id="s-short", intent="weather_query")
    out = await manager._summarize_response_for_chat(ctx=ctx, user_input="temp in Munich", response=text)
    assert out == text


@pytest.mark.asyncio
async def test_summarize_response_for_chat_compacts_structured_output() -> None:
    raw = """
    Thinking Process:
    1. Analyze request
    **Weather Overview**
    - It is currently around 18 degrees Celsius in Munich.
    - Light clouds expected.
    | Col | Val |
    |---|---|
    """
    manager = ConversationManager()
    ctx = ConversationContext(session_id="s-structured", intent="weather_query")
    out = await manager._summarize_response_for_chat(
        ctx=ctx,
        user_input="weather in Munich",
        response=raw,
    )
    assert "Thinking Process" not in out
    assert "Weather Overview" in out
    assert "18 degrees Celsius in Munich" in out


@pytest.mark.asyncio
async def test_summarize_response_for_chat_uses_light_model_when_available() -> None:
    captured: dict[str, str] = {}

    async def local_handler(request):
        captured["task_type"] = request.task_type
        captured["prompt"] = request.prompt
        return "Munich is around 18°C right now."

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        )
    )
    manager = ConversationManager(model_router=router)
    ctx = ConversationContext(session_id="s-model", intent="weather_query")
    raw = "Thinking Process:\n1. Analyze\n2. Draft\nA very long verbose answer..." * 8
    out = await manager._summarize_response_for_chat(
        ctx=ctx,
        user_input="temperature in Munich",
        response=raw,
    )
    assert out == "Munich is around 18°C right now."
    assert captured.get("task_type") == "summarization"
    assert "\"final\"" in captured.get("prompt", "")
    assert "JSON:" in captured.get("prompt", "")


@pytest.mark.asyncio
async def test_summarize_response_for_chat_extracts_final_concise_marker_from_model_output() -> None:
    async def local_handler(_request):
        return (
            "Thinking Process:\n"
            "1. Analyze the Request\n"
            "Final concise response:\n"
            "I don't have real-time access to check the current temperature in Munich right now. "
            "You can easily find the latest forecast on a weather app or search engine."
        )

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        )
    )
    manager = ConversationManager(model_router=router)
    ctx = ConversationContext(session_id="s-final-marker", intent="weather_query")
    raw = ("Thinking Process:\nLong draft...\n" * 10).strip()
    out = await manager._summarize_response_for_chat(
        ctx=ctx,
        user_input="temperature in munich",
        response=raw,
    )
    assert out.startswith("I don't have real-time access to check")
    assert "Thinking Process" not in out


@pytest.mark.asyncio
async def test_summarize_response_for_chat_extracts_last_quoted_line_when_marker_missing() -> None:
    async def local_handler(_request):
        return (
            "Thinking Process:\n"
            "1. Analyze request\n"
            '* "It looks like it\'s about 18 degrees Celsius in Munich right now."\n'
            '* "It looks like it\'s about 18 degrees Celsius in Munich right now. Let me know if you need anything else."\n'
        )

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        )
    )
    manager = ConversationManager(model_router=router)
    ctx = ConversationContext(session_id="s-quoted", intent="weather_query")
    raw = ("Thinking Process:\nlong...\n" * 10).strip()
    out = await manager._summarize_response_for_chat(
        ctx=ctx,
        user_input="temperature in munich",
        response=raw,
    )
    assert out == "It looks like it's about 18 degrees Celsius in Munich right now. Let me know if you need anything else."


@pytest.mark.asyncio
async def test_summarize_response_for_chat_discards_partial_thinking_dump_from_model() -> None:
    async def local_handler(_request):
        return (
            "Thinking Process:\n"
            "1. Analyze the Request:\n"
            "* Role: JARVIS\n"
            "* Response Policy: concise\n"
        )

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        )
    )
    manager = ConversationManager(model_router=router)
    ctx = ConversationContext(session_id="s-partial-thinking", intent="weather_query")
    raw = (
        "Munich is currently around 18 degrees Celsius with light clouds. "
        "Let me know if you want the forecast for tomorrow. "
    ) * 4
    out = await manager._summarize_response_for_chat(
        ctx=ctx,
        user_input="temperature in munich",
        response=raw,
    )
    assert out != raw.strip()
    assert "Thinking Process" not in out
    assert "Role: JARVIS" not in out


@pytest.mark.asyncio
async def test_summarize_response_for_chat_trims_trailing_noise_after_valid_answer() -> None:
    async def local_handler(_request):
        return (
            "The current temperature in Munich is 18°C with partly cloudy skies. "
            "Would you like to know the forecast for the next few days?\n"
            "3. Refine the Output:\n"
            "* Check constraints: 1-2 sentences max.\n"
        )

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        )
    )
    manager = ConversationManager(model_router=router)
    ctx = ConversationContext(session_id="s-tail-noise", intent="weather_query")
    raw = ("Thinking Process:\nlong...\n" * 10).strip()
    out = await manager._summarize_response_for_chat(
        ctx=ctx,
        user_input="temperature in munich",
        response=raw,
    )
    assert out == (
        "The current temperature in Munich is 18°C with partly cloudy skies. "
        "Would you like to know the forecast for the next few days?"
    )


@pytest.mark.asyncio
async def test_summarize_response_for_chat_drops_bare_thinking_process_output() -> None:
    async def local_handler(_request):
        return "Thinking Process"

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        )
    )
    manager = ConversationManager(model_router=router)
    ctx = ConversationContext(session_id="s-bare-thinking", intent="weather_query")
    raw = (
        "The current temperature in Munich is around 18°C with partly cloudy skies. "
        "Would you like the forecast too? "
    ) * 5
    out = await manager._summarize_response_for_chat(
        ctx=ctx,
        user_input="temperature in munich",
        response=raw,
    )
    # Model output is invalid, so we fall back to deterministic summary cleanup.
    assert "Thinking Process" not in out
    assert "temperature in Munich" in out


@pytest.mark.asyncio
async def test_summarize_response_for_chat_rejects_user_question_echo_from_summarizer() -> None:
    async def local_handler(_request):
        return "What is the temperature in Munich right now?"

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        )
    )
    manager = ConversationManager(model_router=router)
    ctx = ConversationContext(session_id="s-echo", intent="weather_query")
    raw = (
        "The current temperature in Munich is around 18°C with partly cloudy skies. "
        "Would you like the forecast for tomorrow? "
    ) * 4
    out = await manager._summarize_response_for_chat(
        ctx=ctx,
        user_input="What is the temperature in Munich right now?",
        response=raw,
    )
    assert out != "What is the temperature in Munich right now?"
    assert "temperature in Munich" in out


@pytest.mark.asyncio
async def test_summarize_response_for_chat_meta_only_outputs_fallback_generic() -> None:
    async def local_handler(_request):
        return "Analyze the Request: Role: JARVIS (intelligent AI assistant). Tone: Natural, human-like, plain conversational text."

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        )
    )
    manager = ConversationManager(model_router=router)
    ctx = ConversationContext(session_id="s-meta-only", intent="weather_query")
    raw = (
        "Analyze the Request: Role: JARVIS.\n"
        "Response Policy: concise.\n"
        "Current intent: weather_query.\n"
    ) * 5
    out = await manager._summarize_response_for_chat(
        ctx=ctx,
        user_input="What is the temperature in Munich right now?",
        response=raw,
    )
    assert "Analyze the Request" not in out
    assert "Response Policy" not in out
    assert out


@pytest.mark.asyncio
async def test_summarize_response_for_chat_strips_analyze_request_pattern() -> None:
    raw = """
    Analyze the Request: Role: JARVIS, an intelligent AI assistant.
    Response Policy: Sound natural and human, plain conversational text.
    Current intent: weather_query
    It is around 18°C in Munich right now.
    """
    manager = ConversationManager()
    ctx = ConversationContext(session_id="s-analyze", intent="weather_query")
    out = await manager._summarize_response_for_chat(
        ctx=ctx,
        user_input="temperature in munich",
        response=raw,
    )
    assert out.strip() == "It is around 18°C in Munich right now."


@pytest.mark.asyncio
async def test_summarize_response_for_chat_rejects_meta_model_summary_and_falls_back() -> None:
    async def local_handler(_request):
        return (
            "Analyze the Request: Role: JARVIS.\n"
            "Response Policy: concise.\n"
            "Current intent: weather_query."
        )

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        )
    )
    manager = ConversationManager(model_router=router)
    ctx = ConversationContext(session_id="s-meta", intent="weather_query")
    raw = (
        "Analyze the Request: Role: JARVIS.\n"
        "Response Policy: concise.\n"
        "It is around 18°C in Munich right now.\n"
    ) * 6
    out = await manager._summarize_response_for_chat(
        ctx=ctx,
        user_input="temperature in munich",
        response=raw,
    )
    assert "Analyze the Request" not in out
    assert "Response Policy" not in out
    assert "18°C in Munich" in out


@pytest.mark.asyncio
async def test_conversation_manager_router_fast_path_sets_metadata_and_latency() -> None:
    captured = {}

    async def local_handler(request):
        captured["metadata"] = request.metadata
        captured["max_latency_ms"] = request.max_latency_ms
        return "sunny"

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        )
    )
    manager = ConversationManager(model_router=router)
    sid = manager.start_session("fast-user")
    out = await manager.process_input(sid, "Hi there")
    assert out
    assert captured["metadata"].get("fast_path") is True
    assert captured["max_latency_ms"] == 8000
    ctx = manager.get_context(sid)
    assert ctx is not None
    assert "latency_ms" in ctx.metadata
    assert "summary_stage" in ctx.metadata


@pytest.mark.asyncio
async def test_conversation_manager_weather_direct_path_avoids_router() -> None:
    called = {"router": False}

    async def local_handler(_request):
        called["router"] = True
        return "should-not-be-used"

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        )
    )
    manager = ConversationManager(model_router=router)

    async def fake_weather(_loc: str) -> str:
        return "In Munich, it is 18°C with partly cloudy skies right now."

    manager._quick_weather_lookup = fake_weather  # type: ignore[method-assign]
    sid = manager.start_session("wx-user")
    out = await manager.process_input(sid, "What is the temperature in Munich right now?")
    assert "18°C" in out
    assert called["router"] is False


@pytest.mark.asyncio
async def test_conversation_manager_day_query_returns_day_not_time() -> None:
    manager = ConversationManager()
    sid = manager.start_session("day-user")
    out = await manager.process_input(sid, "What is the day today?")
    assert out.startswith("Today is ")
    assert datetime.now().strftime("%A") in out


def test_get_or_create_session_expires_idle_sessions() -> None:
    manager = ConversationManager()
    sid1 = manager.start_session("u1")
    sid2 = manager.start_session("u2")
    ctx1 = manager.get_context(sid1)
    assert ctx1 is not None
    ctx1.last_activity = time.time() - (manager._SESSION_TIMEOUT + 10)
    got = manager.get_or_create_session("u2")
    assert got == sid2
    assert manager.get_context(sid1) is None


def test_extract_weather_location_strips_time_suffix() -> None:
    manager = ConversationManager()
    assert (
        manager._extract_weather_location("What is the temperature in Munich right now?")
        == "Munich"
    )


def test_open_meteo_code_to_desc_mapping() -> None:
    manager = ConversationManager()
    assert manager._open_meteo_code_to_desc(2) == "partly cloudy skies"
    assert manager._open_meteo_code_to_desc(95) == "a thunderstorm"
    assert manager._open_meteo_code_to_desc(999) == ""


def test_build_llm_prompt_includes_human_style_and_weather_guidance() -> None:
    ctx = ConversationContext(
        session_id="s1",
        intent="weather_query",
        current_topic="weather",
    )
    prompt = ConversationManager._build_llm_prompt(
        ctx,
        "What is the weather today in New Delhi?",
        history="user: hi\nassistant: hello",
    )
    assert "Sound natural and human" in prompt
    assert "Keep replies concise by default (1-3 sentences)." in prompt
    assert "Do not include internal reasoning" in prompt
    assert "Weather guidance: reply in 1-2 sentences" in prompt
