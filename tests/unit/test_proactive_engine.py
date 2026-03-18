from __future__ import annotations

from infrastructure.proactive_engine import ProactiveEventEngine


def test_ingest_calendar_event_generates_suggestion() -> None:
    engine = ProactiveEventEngine()
    result = engine.ingest_event(
        event_type="calendar_upcoming",
        payload={"user_id": "u1", "title": "Design Review", "starts_in": "in 20 minutes"},
    )
    assert result["generated_count"] == 1
    assert result["suggestions"][0]["category"] == "calendar"
    listed = engine.list_suggestions(user_id="u1")
    assert listed["count"] == 1


def test_anomaly_event_marks_human_required() -> None:
    engine = ProactiveEventEngine()
    result = engine.ingest_event(
        event_type="anomaly_detected",
        payload={"user_id": "u2", "anomaly": "spike in failed logins"},
    )
    suggestion = result["suggestions"][0]
    assert suggestion["priority"] == "urgent"
    assert suggestion["requires_human"] is True


def test_proactive_disabled_blocks_generation() -> None:
    engine = ProactiveEventEngine()
    engine.set_user_preferences("u3", {"proactive_enabled": False})
    result = engine.ingest_event(
        event_type="task_overdue",
        payload={"user_id": "u3", "task": "prepare report"},
    )
    assert result["generated_count"] == 0
    listed = engine.list_suggestions(user_id="u3")
    assert listed["count"] == 0


def test_dedupe_cooldown_suppresses_repeated_suggestions() -> None:
    engine = ProactiveEventEngine()
    engine.set_user_preferences("u4", {"cooldown_seconds": 3600})
    first = engine.ingest_event(
        event_type="task_overdue",
        payload={"user_id": "u4", "task": "submit timesheet"},
    )
    second = engine.ingest_event(
        event_type="task_overdue",
        payload={"user_id": "u4", "task": "submit timesheet"},
    )
    assert first["generated_count"] == 1
    assert second["generated_count"] == 0


def test_low_priority_hidden_by_default() -> None:
    engine = ProactiveEventEngine()
    engine.ingest_event(event_type="digest_ready", payload={"user_id": "u5", "topic": "ai"})
    hidden = engine.list_suggestions(user_id="u5")
    shown = engine.list_suggestions(user_id="u5", include_low_priority=True)
    assert hidden["count"] == 0
    assert shown["count"] == 1


def test_low_risk_profile_requires_human_for_high_priority() -> None:
    engine = ProactiveEventEngine()
    engine.set_user_preferences("u6", {"risk_tolerance": "low", "cooldown_seconds": 0})
    result = engine.ingest_event(
        event_type="task_overdue",
        payload={"user_id": "u6", "task": "pay invoice"},
    )
    assert result["generated_count"] == 1
    assert result["suggestions"][0]["requires_human"] is True
    assert "priority_exceeds_risk_tolerance" in result["suggestions"][0]["metadata"].get("safety_reasons", [])


def test_autonomous_action_safety_decision_blocks_sensitive_actions() -> None:
    engine = ProactiveEventEngine()
    decision = engine.evaluate_autonomous_action(
        user_id="u7",
        action_name="execute shell command",
        category="tasks",
        priority="high",
    )
    assert decision["allowed"] is False
    assert decision["requires_approval"] is True

    engine.set_user_preferences(
        "u7",
        {
            "autonomous_actions_enabled": True,
            "allowed_autonomous_categories": ["calendar", "tasks"],
            "risk_tolerance": "high",
        },
    )
    allowed = engine.evaluate_autonomous_action(
        user_id="u7",
        action_name="prepare task plan",
        category="tasks",
        priority="normal",
    )
    assert allowed["allowed"] is True
    assert allowed["requires_approval"] is False
