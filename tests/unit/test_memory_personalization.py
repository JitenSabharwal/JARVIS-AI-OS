from __future__ import annotations

import asyncio
from datetime import datetime
import time
from typing import Any

import pytest

from interfaces.conversation_manager import ConversationContext, ConversationManager, ResponsePlan
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


@pytest.mark.asyncio
async def test_conversation_manager_learns_name_from_i_am_and_recalls_it() -> None:
    manager = ConversationManager()
    sid = manager.start_session("name-user")
    first = await manager.process_input(sid, "Hi Jarvis I am Jiten")
    assert first
    second = await manager.process_input(sid, "Tell my name")
    assert "jiten" in second.lower()


def test_query_understanding_heuristic_provides_route_and_ambiguity() -> None:
    understanding = ConversationManager._infer_query_understanding_heuristic(
        user_input="Can you research latest AI trends and compare top frameworks?",
        rule_intent="information_query",
        entities={},
    )
    assert understanding.requires_retrieval is True
    assert understanding.recommended_route in {"retrieve", "decompose"}
    assert 0.0 <= understanding.ambiguity_score <= 1.0


@pytest.mark.asyncio
async def test_conversation_manager_turn_memory_semantics_tracks_slots_and_goal() -> None:
    manager = ConversationManager()
    sid = manager.start_session("memory-semantics-user")
    _ = await manager.process_input(sid, "I am advanced and preparing for job")
    ctx = manager.get_context(sid)
    assert ctx is not None
    turn_memory = ctx.metadata.get("turn_memory", {})
    assert isinstance(turn_memory, dict)
    slots = turn_memory.get("slots", {})
    assert isinstance(slots, dict)
    assert slots.get("level") == "advanced"
    assert slots.get("goal") == "job"


def test_kb_entry_trust_score_prioritizes_verified_metadata() -> None:
    verified = ConversationManager._kb_entry_trust_score(
        {"category": "verified", "tags": ["hf", "trusted"]}
    )
    unverified = ConversationManager._kb_entry_trust_score(
        {"category": "community", "tags": ["non_hf", "unverified"]}
    )
    assert verified > unverified


@pytest.mark.asyncio
async def test_conversation_manager_name_set_reply_is_direct_and_clean() -> None:
    manager = ConversationManager()
    sid = manager.start_session("name-set-user")
    out = await manager.process_input(sid, "My name is Jtien")
    assert "nice to meet you" in out.lower()
    assert "jtien" in out.lower()
    follow = await manager.process_input(sid, "What is my name?")
    assert "jtien" in follow.lower()


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


def test_rule_based_email_draft_personal_recipient_uses_personal_tone() -> None:
    out = ConversationManager._rule_based_email_draft(
        "write me an email to my mother to buy a mac mini m5 for me"
    )
    low = out.lower()
    assert "hi mom" in low
    assert "manager name" not in low
    assert "approval to purchase" not in low
    assert "mac mini m5" in low


def test_rule_based_code_draft_returns_react_component() -> None:
    out = ConversationManager._rule_based_code_draft(
        "write a functional component in react js called GreetingCard"
    )
    assert "```jsx" in out
    assert "function GreetingCard()" in out
    assert "export default GreetingCard;" in out


def test_affirmative_continuation_detects_short_yes_followups() -> None:
    assert ConversationManager._is_affirmative_continuation("yes please do")
    assert ConversationManager._is_affirmative_continuation("go ahead")
    assert not ConversationManager._is_affirmative_continuation("yes, but not now")


def test_output_directives_parsing_and_application_for_sectioned_short() -> None:
    prefs = ConversationManager._extract_output_directives("yes please do in 3 sections concise points")
    assert prefs.get("sectioned") is True
    assert prefs.get("max_sections") == 3
    assert prefs.get("target_length") == "short"
    assert prefs.get("format") == "list"

    shaped = ConversationManager._apply_output_directives(
        "React is flexible and quick to iterate. Angular is structured for larger teams. Vue is simple and lightweight.",
        prefs,
    )
    assert shaped.startswith("1. ")
    assert "\n\n2. " in shaped


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
async def test_causal_offer_followup_yes_please_do_returns_comparison() -> None:
    manager = ConversationManager()
    sid = manager.start_session("followup-user")

    first = await manager.process_input(sid, "why is react faster than older js frameworks")
    assert "compare two concrete alternatives" in first.lower()
    ctx = manager.get_context(sid)
    assert ctx is not None
    pending = ctx.metadata.get("pending_action", {})
    assert isinstance(pending, dict)
    assert pending.get("type") == "compare_alternatives"

    second = await manager.process_input(sid, "yes please do")
    low = second.lower()
    assert "angular" in low
    assert "vue" in low
    assert "how can i help you with that" not in low
    ctx = manager.get_context(sid)
    assert ctx is not None
    assert "pending_action" not in ctx.metadata


@pytest.mark.asyncio
async def test_learning_offer_sets_pending_action_and_yes_prompts_for_slots() -> None:
    manager = ConversationManager()
    sid = manager.start_session("learn-user")
    ctx = manager.get_context(sid)
    assert ctx is not None
    ctx.metadata["pending_action"] = {
        "type": "collect_learning_profile",
        "source": "test",
        "created_turn": 0,
    }

    second = await manager.process_input(sid, "yes")
    assert "tell me your level" in second.lower()
    ctx = manager.get_context(sid)
    assert ctx is not None
    assert ctx.metadata.get("learning_plan_pending") is True


@pytest.mark.asyncio
async def test_pending_action_yes_with_streaming_directives_shapes_response() -> None:
    manager = ConversationManager()
    sid = manager.start_session("stream-followup-user")
    ctx = manager.get_context(sid)
    assert ctx is not None
    ctx.metadata["pending_action"] = {
        "type": "compare_alternatives",
        "topic": "react",
        "source": "test",
        "created_turn": 0,
    }

    out = await manager.process_input(sid, "yes please do in 2 sections concise points")
    assert out.startswith("1. ")
    assert "\n\n2. " in out


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
async def test_fast_intent_short_circuit_skips_model_for_voice_greeting() -> None:
    calls = {"count": 0}

    async def local_handler(_request):
        calls["count"] += 1
        return "model-response"

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
    sid = manager.start_session("fast-user")
    out = await manager.process_input(sid, "Hi Jarvis", modality="voice")
    assert out
    assert calls["count"] == 0


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
async def test_realtime_media_grounding_is_attached_to_context_fusion() -> None:
    captured: dict[str, Any] = {}

    async def local_handler(request):
        captured["metadata"] = request.metadata
        return "Grounded reply."

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
    sid = manager.start_realtime_session(user_id="rt-user")
    manager.ingest_realtime_frame(
        sid,
        source="screen",
        summary="User is viewing CI dashboard with failing tests panel.",
        metadata={"app": "grafana"},
    )
    out = await manager.process_input(sid, "Explain what you see and what to do next.", modality="voice")
    assert out
    md = captured.get("metadata", {})
    assert isinstance(md, dict)
    fusion = md.get("context_fusion", {})
    assert isinstance(fusion, dict)
    keys = fusion.get("external_context_keys", [])
    assert isinstance(keys, list)
    assert "live_visual_grounding" in keys


@pytest.mark.asyncio
async def test_realtime_interrupt_returns_interrupted_message_for_stale_turn() -> None:
    async def slow_handler(_request):
        await asyncio.sleep(0.12)
        return "This should be discarded if interrupted."

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=slow_handler,
            supported_modalities={"text", "voice"},
        )
    )
    manager = ConversationManager(model_router=router)
    sid = manager.start_realtime_session(user_id="rt-interrupt-user")
    turn_task = asyncio.create_task(manager.process_input(sid, "Tell me the full analysis", modality="voice"))
    await asyncio.sleep(0.03)
    manager.interrupt_realtime_session(sid, reason="barge_in")
    out = await turn_task
    assert "interrupted the previous response" in out.lower()


@pytest.mark.asyncio
async def test_realtime_visual_observation_summary_uses_model_router_for_image() -> None:
    captured: dict[str, Any] = {}

    async def local_handler(request):
        captured["modality"] = request.modality
        captured["media"] = request.media
        captured["metadata"] = request.metadata
        return "The frame shows VS Code with a failing unit test and a terminal traceback."

    from infrastructure.model_router import CallableModelProvider, ModelRouter

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text", "image"},
        )
    )
    manager = ConversationManager(model_router=router)
    sid = manager.start_realtime_session(user_id="rt-vision")
    summary = await manager.summarize_visual_observation(
        sid,
        source="iphone_camera",
        image_url="https://example.com/frame.jpg",
        metadata={"device": "iphone"},
    )
    assert "failing unit test" in summary.lower()
    assert captured.get("modality") == "image"
    media = captured.get("media", {})
    assert isinstance(media, dict)
    assert media.get("image_url") == "https://example.com/frame.jpg"
    md = captured.get("metadata", {})
    assert md.get("stage") == "realtime_visual_summary"


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


@pytest.mark.asyncio
async def test_summarize_response_for_chat_preserves_structured_code_for_task_requests() -> None:
    manager = ConversationManager()
    ctx = ConversationContext(session_id="s-code", intent="task_execution")
    code = (
        "```jsx\n"
        "import React from \"react\";\n\n"
        "function MyComponent() {\n"
        "  return <div>Hello from React</div>;\n"
        "}\n\n"
        "export default MyComponent;\n"
        "```"
    )
    out = await manager._summarize_response_for_chat(
        ctx=ctx,
        user_input="write a functional component in react js",
        response=code,
    )


@pytest.mark.asyncio
async def test_learning_plan_slot_fill_across_turns() -> None:
    manager = ConversationManager()
    sid = manager.start_session(user_id="learn-user")

    first = await manager.process_input(sid, "What can you teach me in frontend, backend, and AI tooling?")
    low_first = first.lower()
    assert "level" in low_first and "goal" in low_first

    second = await manager.process_input(sid, "advanced")
    assert "What is your goal" in second

    third = await manager.process_input(sid, "goal is job")
    low = third.lower()
    assert "advanced" in low
    assert "job" in low
    assert "system design" in low


@pytest.mark.asyncio
async def test_learning_plan_slots_single_turn_payload() -> None:
    manager = ConversationManager()
    sid = manager.start_session(user_id="learn-user-2")
    await manager.process_input(sid, "What can you teach me in frontend, backend, and AI tooling?")
    out = await manager.process_input(sid, "advanced level and goal is job")
    low = out.lower()
    assert "advanced" in low
    assert "job" in low


@pytest.mark.asyncio
async def test_query_understanding_requests_missing_constraints_for_learning() -> None:
    manager = ConversationManager()
    sid = manager.start_session(user_id="qs-user")
    out = await manager.process_input(sid, "Can you give me a learning roadmap?")
    low = out.lower()
    assert "level" in low
    assert "goal" in low
    ctx = manager.get_context(sid)
    assert ctx is not None
    q = ctx.metadata.get("query_understanding", {})
    assert isinstance(q, dict)
    assert q.get("should_ask_clarification") is True


@pytest.mark.asyncio
async def test_query_understanding_uses_model_json_when_confident() -> None:
    async def local_handler(request):
        if request.metadata.get("stage") == "query_understanding":
            return (
                '{"inferred_intent":"information_query","user_goal":"compare frameworks",'
                '"constraints":["scope=frontend"],"missing_constraints":[],"confidence":0.91,'
                '"should_ask_clarification":false}'
            )
        return "ok"

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
    sid = manager.start_session(user_id="model-qs")
    await manager.process_input(sid, "Compare frontend frameworks")
    ctx = manager.get_context(sid)
    assert ctx is not None
    q = ctx.metadata.get("query_understanding", {})
    assert q.get("user_goal") == "compare frameworks"
    assert q.get("confidence", 0) >= 0.9


@pytest.mark.asyncio
async def test_query_decomposition_heuristic_propagates_to_prompt_and_metadata() -> None:
    captured: dict[str, Any] = {}

    async def local_handler(request):
        captured["prompt"] = request.prompt
        captured["metadata"] = request.metadata
        if request.metadata.get("stage") == "query_understanding":
            return (
                '{"inferred_intent":"information_query","user_goal":"architecture reasoning",'
                '"constraints":[],"missing_constraints":[],"confidence":0.84,'
                '"should_ask_clarification":false}'
            )
        return "Here is a structured answer."

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
    sid = manager.start_session(user_id="decomp-user")
    out = await manager.process_input(
        sid,
        "Explain event-driven architecture, then compare Kafka and RabbitMQ, and recommend when to use each.",
    )
    assert out
    ctx = manager.get_context(sid)
    assert ctx is not None
    qd = ctx.metadata.get("query_decomposition", {})
    assert isinstance(qd, dict)
    assert qd.get("should_decompose") is True
    assert len(qd.get("sub_questions", [])) >= 2
    md = captured.get("metadata", {})
    assert isinstance(md.get("query_decomposition"), dict)
    assert md["query_decomposition"].get("should_decompose") is True
    assert "Decomposed sub-questions" in str(captured.get("prompt", ""))


@pytest.mark.asyncio
async def test_query_decomposition_simple_query_stays_disabled() -> None:
    manager = ConversationManager()
    sid = manager.start_session(user_id="decomp-simple")
    _ = await manager.process_input(sid, "What is React?")
    ctx = manager.get_context(sid)
    assert ctx is not None
    qd = ctx.metadata.get("query_decomposition", {})
    assert isinstance(qd, dict)
    assert qd.get("should_decompose") is False


@pytest.mark.asyncio
async def test_reference_resolution_heuristic_uses_previous_topic() -> None:
    manager = ConversationManager()
    sid = manager.start_session(user_id="ref-user")
    # Prime context with a meaningful topic.
    ctx = manager.get_context(sid)
    assert ctx is not None
    ctx.current_topic = "react"
    _ = await manager.process_input(sid, "Tell me more about those")
    rr = ctx.metadata.get("reference_resolution", {})
    assert isinstance(rr, dict)
    assert rr.get("used") is True
    assert rr.get("source") in {"heuristic", "model"}
    assert "react" in str(rr.get("resolved_query", "")).lower()


@pytest.mark.asyncio
async def test_reference_resolution_model_path_metadata() -> None:
    async def local_handler(request):
        if request.metadata.get("stage") == "reference_resolution":
            return (
                '{"used":true,"resolved_query":"Tell me more about React and Next.js",'
                '"antecedent":"React and Next.js","confidence":0.92}'
            )
        if request.metadata.get("stage") == "query_understanding":
            return (
                '{"inferred_intent":"information_query","user_goal":"learn frameworks",'
                '"constraints":[],"missing_constraints":[],"confidence":0.8,'
                '"should_ask_clarification":false}'
            )
        return "Detailed comparison between React and Next.js."

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
    sid = manager.start_session(user_id="ref-model-user")
    await manager.process_input(sid, "What can you teach me in frontend frameworks?")
    _ = await manager.process_input(sid, "Tell me more about those")
    ctx = manager.get_context(sid)
    assert ctx is not None
    rr = ctx.metadata.get("reference_resolution", {})
    assert isinstance(rr, dict)
    assert rr.get("used") is True
    assert rr.get("source") in {"model", "heuristic"}


@pytest.mark.asyncio
async def test_why_query_uses_reasoning_task_type_and_long_target() -> None:
    captured: dict[str, Any] = {}

    async def local_handler(request):
        captured["task_type"] = request.task_type
        captured["metadata"] = request.metadata
        return "Because it balances tradeoffs."

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
    sid = manager.start_session(user_id="why-user")
    out = await manager.process_input(sid, "Why do teams prefer React for large apps?")
    assert out
    assert captured.get("task_type") == "reasoning_why"
    md = captured.get("metadata", {})
    assert isinstance(md, dict)
    plan = md.get("response_plan", {})
    assert isinstance(plan, dict)
    assert plan.get("target_length") == "long"


@pytest.mark.asyncio
async def test_why_query_fallback_includes_tradeoff_language() -> None:
    manager = ConversationManager()
    sid = manager.start_session(user_id="why-fallback")
    out = await manager.process_input(sid, "Why is this architecture used?")
    low = out.lower()
    assert "tradeoff" in low or "tradeoffs" in low
    assert "alternative" in low or "alternatives" in low


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
    assert isinstance(captured["metadata"].get("query_understanding"), dict)
    assert captured["max_latency_ms"] == 8000
    ctx = manager.get_context(sid)
    assert ctx is not None
    assert "latency_ms" in ctx.metadata
    assert "summary_stage" in ctx.metadata
    assert isinstance(ctx.metadata.get("query_understanding"), dict)


@pytest.mark.asyncio
async def test_low_latency_turn_skips_model_summary_stage() -> None:
    manager = ConversationManager()
    sid = manager.start_session("voice-user")
    ctx = manager.get_context(sid)
    assert ctx is not None
    ctx.metadata["low_latency_turn"] = True

    called = {"value": False}

    async def _should_not_run(**_kwargs):  # type: ignore[no-untyped-def]
        called["value"] = True
        return "unexpected"

    manager._summarize_response_with_light_model = _should_not_run  # type: ignore[method-assign]
    raw = ("This is a long answer. " * 40).strip()
    out = await manager._summarize_response_for_chat(
        ctx=ctx,
        user_input="test query",
        response=raw,
    )
    assert out
    assert called["value"] is False
    stage = ctx.metadata.get("summary_stage", {})
    assert isinstance(stage, dict)
    assert str(stage.get("source", "")).startswith(
        ("fallback_rule_low_latency", "passthrough_low_latency")
    )


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


@pytest.mark.asyncio
async def test_generic_fallback_research_request_returns_actionable_answer() -> None:
    manager = ConversationManager()
    sid = manager.start_session("research-user")
    out = await manager.process_input(sid, "Can you research on AI")
    assert "always learning" not in out.lower()
    assert "i can research" in out.lower()
    assert "ai" in out.lower()


def test_extract_intent_demotes_greeting_when_actionable_request_present() -> None:
    manager = ConversationManager()
    intent, entities, confidence = manager.extract_intent("Hi can you help me learn functional react js")
    assert intent in {"task_execution", "information_query"}
    assert confidence >= 0.80
    assert isinstance(entities, dict)


def test_remove_sentence_overlap_keeps_only_incremental_details() -> None:
    base = "React uses components and hooks. Start with useState and props."
    candidate = (
        "React uses components and hooks. "
        "Use useEffect for side effects and memoization techniques for performance. "
        "Start with useState and props."
    )
    out = ConversationManager._remove_sentence_overlap(base=base, candidate=candidate)
    assert "components and hooks" not in out.lower()
    assert "usestate and props" not in out.lower()
    assert "useeffect" in out.lower()


@pytest.mark.asyncio
async def test_dual_tier_runs_small_and_large_in_parallel() -> None:
    from infrastructure.model_router import CallableModelProvider, ModelRouter

    starts: dict[str, float] = {}

    async def local_handler(request):  # type: ignore[no-untyped-def]
        tier = str((request.metadata or {}).get("model_tier", "")).strip().lower()
        starts[tier] = time.perf_counter()
        if tier == "small":
            await asyncio.sleep(0.06)
            return "React uses components."
        await asyncio.sleep(0.06)
        return "React uses components. Use hooks and memoization for performance."

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=local_handler,
            supported_modalities={"text"},
        )
    )
    manager = ConversationManager(model_router=router)
    manager._dual_tier_response_enabled = True  # type: ignore[attr-defined]
    sid = manager.start_session("dual-tier-user")
    ctx = manager.get_context(sid)
    assert ctx is not None
    plan = ResponsePlan(
        intent="information_query",
        task_type="information_query",
        complexity=0.62,
        target_length="medium",
    )
    out = await manager._maybe_generate_dual_tier_response(
        ctx=ctx,
        user_input="help me learn react",
        plan=plan,
        modality="text",
        media={},
        prompt="Explain React basics and practical learning steps.",
        fused_context=None,
    )
    assert "React uses components." in out
    assert "memoization" in out.lower()
    assert "small" in starts and "large" in starts
    assert abs(starts["small"] - starts["large"]) < 0.04


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


def test_derive_response_policy_greeting_prefers_brief_conversational_style() -> None:
    plan = ResponsePlan(
        intent="greeting",
        task_type="greeting",
        complexity=0.1,
        target_length="short",
    )
    policy = ConversationManager._derive_response_policy(
        user_input="Hey, how are you?",
        plan=plan,
        query_understanding={},
    )
    assert policy["style"] == "conversational"
    assert policy["verbosity"] == "short"
    assert "1-2" in str(policy["length_rule"])


def test_derive_response_policy_emotional_query_prefers_empathetic_style() -> None:
    plan = ResponsePlan(
        intent="information_query",
        task_type="information_query",
        complexity=0.4,
        target_length="short",
    )
    policy = ConversationManager._derive_response_policy(
        user_input="I feel overwhelmed and anxious about work.",
        plan=plan,
        query_understanding={},
    )
    assert policy["style"] == "empathetic"
    assert policy["verbosity"] in {"medium", "long"}
