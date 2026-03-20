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
