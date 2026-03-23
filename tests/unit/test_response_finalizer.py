from __future__ import annotations

from core.response_finalizer import finalize_user_response


def test_finalize_user_response_strips_meta() -> None:
    text = "Thinking Process: Analyze the request."
    out = finalize_user_response(text, fallback="fallback")
    assert out == "fallback"


def test_finalize_user_response_keeps_clean_text() -> None:
    text = "Workflow completed. Completed 3/3 step(s)."
    out = finalize_user_response(text, fallback="fallback")
    assert out == text


def test_finalize_user_response_salvages_answer_from_analysis_preamble() -> None:
    text = (
        "Let me analyze this conversation carefully:\n"
        "1. User asked for help.\n"
        "2. Assistant should ask clarification.\n"
        "Absolutely! I'd be happy to help you with React. What specifically do you need?\n"
    )
    out = finalize_user_response(text, fallback="fallback")
    assert out.startswith("Absolutely! I'd be happy to help you with React.")
