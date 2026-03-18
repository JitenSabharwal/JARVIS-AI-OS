from __future__ import annotations

import pytest

from interfaces.conversation_manager import ConversationManager
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
