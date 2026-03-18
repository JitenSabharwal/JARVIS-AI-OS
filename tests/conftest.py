"""
JARVIS AI OS - Pytest Configuration and Fixtures
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.agent_framework import AgentCapability, AgentMessage, AgentState, MessageType
from memory.conversation_memory import ConversationMemory
from memory.episodic_memory import Episode, EpisodicMemory
from memory.knowledge_base import KnowledgeBase
from utils.exceptions import JARVISError


# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_task() -> dict:
    return {
        "task_id": "task-001",
        "description": "Analyse sales data and generate a summary report",
        "required_capabilities": ["analyze_data", "generate_report"],
        "priority": 5,
        "dependencies": [],
        "metadata": {"source": "test"},
    }


@pytest.fixture
def sample_agent_message() -> AgentMessage:
    return AgentMessage(
        id="msg-001",
        sender_id="agent-a",
        recipient_id="agent-b",
        message_type=MessageType.TASK_REQUEST,
        payload={"action": "ping"},
        timestamp=datetime.now(timezone.utc),
        priority=1,
    )


@pytest.fixture
def sample_episode() -> Episode:
    return Episode(
        id="ep-001",
        task_description="Research Python async patterns",
        actions_taken=["web_search", "parse_results", "generate_report"],
        outcome="Produced a 3-page summary on async patterns",
        success=True,
        duration=12.5,
        learned_facts=["asyncio.gather is faster for independent coroutines"],
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_knowledge_entry() -> dict:
    return {
        "key": "python_async_tips",
        "value": "Use asyncio.gather for concurrent coroutines",
        "category": "programming",
        "tags": ["python", "async", "performance"],
    }


# ---------------------------------------------------------------------------
# Memory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conversation_memory() -> ConversationMemory:
    return ConversationMemory(max_size=50)


@pytest.fixture
def knowledge_base(tmp_path) -> KnowledgeBase:
    kb = KnowledgeBase(persist_path=str(tmp_path / "kb.json"))
    return kb


@pytest.fixture
def episodic_memory(tmp_path) -> EpisodicMemory:
    return EpisodicMemory(persist_path=str(tmp_path / "episodes.json"))


# ---------------------------------------------------------------------------
# Mock skill / agent fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_skill():
    """A lightweight mock skill for testing the registry."""
    from skills.base_skill import BaseSkill, SkillResult

    class _TestSkill(BaseSkill):
        def __init__(self):
            super().__init__(
                name="test_skill",
                description="A no-op skill used in tests",
                category="custom",
            )

        async def execute(self, **kwargs) -> SkillResult:  # type: ignore[override]
            return SkillResult(success=True, data={"echo": kwargs})

        def get_schema(self) -> dict:
            return {"type": "object", "properties": {}, "required": []}

        def validate_params(self, **kwargs) -> bool:
            return True

    return _TestSkill()


@pytest.fixture
def mock_config():
    """Return a minimal config suitable for unit tests."""
    from core.config import JARVISConfig
    return JARVISConfig()
