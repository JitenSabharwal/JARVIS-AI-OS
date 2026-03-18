from __future__ import annotations

from memory.episodic_memory import EpisodicMemory


def test_episodic_memory_agent_capability_success_rate() -> None:
    em = EpisodicMemory()
    em.record_episode(
        task_description="analyze logs",
        actions_taken=["analyze_data"],
        outcome="ok",
        success=True,
        metadata={"agent_id": "a1", "capability": "analyze_data"},
    )
    em.record_episode(
        task_description="analyze metrics",
        actions_taken=["analyze_data"],
        outcome="failed",
        success=False,
        metadata={"agent_id": "a1", "capability": "analyze_data"},
    )
    rate = em.get_agent_capability_success_rate(agent_id="a1", capability="analyze_data")
    assert 0.0 <= rate <= 1.0
    assert rate == 0.5


def test_episodic_memory_recommend_actions_for_task() -> None:
    em = EpisodicMemory()
    em.record_episode(
        task_description="research python asyncio patterns",
        actions_taken=["research_topic", "generate_report"],
        outcome="ok",
        success=True,
    )
    em.record_episode(
        task_description="research python concurrency",
        actions_taken=["research_topic", "analyze_data"],
        outcome="ok",
        success=True,
    )

    actions = em.recommend_actions_for_task("python concurrency research", top_n=2)
    assert actions
    assert "research_topic" in actions
