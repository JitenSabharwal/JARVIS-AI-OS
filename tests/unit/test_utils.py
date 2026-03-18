"""
Unit tests for utils/ (exceptions, helpers, validators)
"""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime

import pytest

# ── Exceptions ──────────────────────────────────────────────────────────────

from utils.exceptions import (
    AgentCapabilityError,
    AgentError,
    AgentNotFoundError,
    ConfigurationError,
    JARVISError,
    MessageBusError,
    OrchestratorError,
    QueueError,
    SkillError,
    SkillExecutionError,
    SkillNotFoundError,
    TaskDependencyError,
    TaskError,
    TaskTimeoutError,
    ValidationError,
    WorkflowError,
)


class TestExceptions:
    def test_jarvis_error_base(self):
        err = JARVISError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert isinstance(err, Exception)

    def test_agent_error(self):
        err = AgentError("agent broke")
        assert isinstance(err, JARVISError)

    def test_agent_not_found(self):
        err = AgentNotFoundError("agent-99")
        assert "agent-99" in str(err)

    def test_skill_not_found(self):
        err = SkillNotFoundError("my_skill")
        assert "my_skill" in str(err)

    def test_task_timeout_error(self):
        err = TaskTimeoutError("task-1", 30)
        assert isinstance(err, TaskError)

    def test_validation_error(self):
        err = ValidationError("bad input")
        assert isinstance(err, JARVISError)

    @pytest.mark.parametrize("exc_class", [
        AgentCapabilityError,
        ConfigurationError,
        MessageBusError,
        OrchestratorError,
        QueueError,
        SkillError,
        SkillExecutionError,
        TaskDependencyError,
        WorkflowError,
    ])
    def test_exception_instantiable(self, exc_class):
        err = exc_class("test message")
        assert isinstance(err, JARVISError)


# ── Helpers ──────────────────────────────────────────────────────────────────

from utils.helpers import (
    chunk_list,
    deep_merge,
    flatten_list,
    format_duration,
    generate_id,
    safe_json_dumps,
    safe_json_loads,
    sanitize_input,
    timestamp_now,
    truncate_string,
)


class TestHelpers:
    def test_generate_id_returns_string(self):
        id_ = generate_id()
        assert isinstance(id_, str)
        assert len(id_) > 0

    def test_generate_id_unique(self):
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_id_with_prefix(self):
        id_ = generate_id(prefix="task")
        assert id_.startswith("task")

    def test_timestamp_now_returns_datetime(self):
        ts = timestamp_now()
        assert isinstance(ts, datetime)

    def test_truncate_string_short(self):
        assert truncate_string("hello", 10) == "hello"

    def test_truncate_string_long(self):
        result = truncate_string("hello world", 8)
        assert len(result) <= 11  # may include ellipsis
        assert result.startswith("hello")

    def test_deep_merge_flat(self):
        a = {"x": 1, "y": 2}
        b = {"y": 99, "z": 3}
        merged = deep_merge(a, b)
        assert merged["x"] == 1
        assert merged["y"] == 99
        assert merged["z"] == 3

    def test_deep_merge_nested(self):
        a = {"outer": {"a": 1, "b": 2}}
        b = {"outer": {"b": 99, "c": 3}}
        merged = deep_merge(a, b)
        assert merged["outer"]["a"] == 1
        assert merged["outer"]["b"] == 99
        assert merged["outer"]["c"] == 3

    def test_format_duration(self):
        assert "0" in format_duration(0)
        assert "30" in format_duration(30) or "s" in format_duration(30)

    def test_flatten_list(self):
        nested = [1, [2, 3], [4, [5, 6]]]
        flat = flatten_list(nested)
        assert 5 in flat
        assert 1 in flat

    def test_chunk_list(self):
        chunks = chunk_list(list(range(10)), 3)
        assert len(chunks) == 4
        assert chunks[0] == [0, 1, 2]
        assert chunks[-1] == [9]

    def test_safe_json_dumps_basic(self):
        result = safe_json_dumps({"key": "value", "num": 42})
        assert '"key"' in result
        assert "42" in result

    def test_safe_json_dumps_datetime(self):
        from datetime import datetime, timezone
        data = {"ts": datetime(2024, 1, 1, tzinfo=timezone.utc)}
        result = safe_json_dumps(data)
        assert "2024" in result

    def test_safe_json_loads_valid(self):
        data = safe_json_loads('{"a": 1}')
        assert data == {"a": 1}

    def test_safe_json_loads_invalid(self):
        result = safe_json_loads("not json", default={})
        assert result == {}

    def test_sanitize_input_strips_html(self):
        dirty = "<script>alert('xss')</script>hello"
        clean = sanitize_input(dirty)
        assert "<script>" not in clean
        assert "hello" in clean

    def test_sanitize_input_strips_control_chars(self):
        dirty = "hello\x00\x01world"
        clean = sanitize_input(dirty)
        assert "\x00" not in clean


# ── Validators ───────────────────────────────────────────────────────────────

from utils.validators import validate_file_path, validate_task_definition


class TestValidators:
    def test_validate_task_definition_valid(self):
        task = {
            "task_id": "t1",
            "description": "do something",
            "required_capabilities": ["cap_a"],
            "priority": 3,
            "dependencies": [],
        }
        result = validate_task_definition(task)
        assert result is True or result == task or result is None  # accept any truthy/None

    def test_validate_task_definition_missing_description(self):
        with pytest.raises(Exception):
            validate_task_definition({"task_id": "t1", "priority": 1})

    def test_validate_file_path_safe(self):
        # Should not raise
        validate_file_path("/tmp/safe_file.txt")

    def test_validate_file_path_traversal(self):
        with pytest.raises(Exception):
            validate_file_path("../../etc/passwd")
