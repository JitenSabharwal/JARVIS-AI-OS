from __future__ import annotations

from core.response_governance import apply_response_governance


def test_apply_response_governance_rejects_meta() -> None:
    res = apply_response_governance("Thinking Process: step by step", route="chat")
    assert res.rejected is True
    assert "couldn't generate" in res.text.lower()


def test_apply_response_governance_accepts_repo_length() -> None:
    text = "Architecture: x. Entry points: a,b. Key modules: c,d. Important flows: f. Evidence: s. Confidence: 0.8. Risks: r. Next steps: n. " * 4
    res = apply_response_governance(text, route="repo")
    assert res.rejected is False
    assert res.word_count >= 40


def test_apply_response_governance_chat_long_reasoning_allows_richer_length() -> None:
    text = " ".join(["detail"] * 320)
    res = apply_response_governance(
        text,
        route="chat",
        hints={
            "user_input": "Explain in detail why this architecture is better and include tradeoffs.",
            "response_plan": {"target_length": "long", "task_type": "reasoning_why"},
            "query_understanding": {"inferred_intent": "why_reasoning"},
        },
    )
    assert res.rejected is False
    assert res.word_count == 320
    assert res.verbosity_tier == "long"
    assert res.max_words >= 700


def test_apply_response_governance_chat_short_weather_tightens_cap() -> None:
    text = " ".join(["weather"] * 140)
    res = apply_response_governance(
        text,
        route="chat",
        hints={
            "user_input": "What is the weather in Munich today?",
            "response_plan": {"target_length": "short", "task_type": "weather_query"},
        },
    )
    assert res.rejected is False
    assert res.word_count <= 80
    assert res.verbosity_tier == "short"


def test_apply_response_governance_prefers_sentence_boundary_when_truncating() -> None:
    text = (
        ("Sentence one keeps coherence under truncation. " * 15)
        + ("Sentence two should be dropped entirely when budget is short. " * 15)
    )
    res = apply_response_governance(
        text,
        route="chat",
        hints={"response_plan": {"target_length": "short", "task_type": "general"}},
    )
    assert res.rejected is False
    assert res.word_count <= 120
    assert res.text.endswith(".")


def test_apply_response_governance_result_includes_verbosity_fields() -> None:
    res = apply_response_governance(
        "This is a compact answer.",
        route="chat",
        hints={"response_plan": {"target_length": "short", "task_type": "general"}},
    )
    payload = res.to_dict()
    assert "verbosity_tier" in payload
    assert "min_words" in payload
    assert "max_words" in payload
    assert payload["verbosity_tier"] == "short"


def test_apply_response_governance_salvages_reasoning_preamble_when_answer_present() -> None:
    text = (
        "Let me analyze this conversation carefully:\n"
        "1. User asked for React help.\n"
        "2. I should respond directly.\n"
        "Absolutely! I'd be happy to help you learn functional React JS. "
        "Tell me your current level and I will give a focused plan."
    )
    res = apply_response_governance(text, route="chat")
    assert res.rejected is False
    assert res.text.startswith("Absolutely! I'd be happy to help you learn functional React JS.")
