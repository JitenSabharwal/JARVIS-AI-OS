from __future__ import annotations

from core.response_quality_eval import evaluate_response_cases, score_response


def test_score_response_penalizes_meta_noise() -> None:
    noisy = "Thinking Process: Analyze the request step by step."
    clean = "Workflow completed. Completed 4/4 step(s)."
    noisy_score = score_response(noisy, expected_phrases=["workflow"], min_words=2, max_words=80, pass_threshold=0.5)
    clean_score = score_response(clean, expected_phrases=["workflow"], min_words=2, max_words=80, pass_threshold=0.5)
    assert noisy_score.score < clean_score.score
    assert noisy_score.noise_hits


def test_evaluate_response_cases() -> None:
    cases = [
        {"id": "a", "expected_phrases": ["hello"], "pass_threshold": 0.6},
        {"id": "b", "expected_phrases": ["missing"], "pass_threshold": 0.8},
    ]
    responses = {"a": "hello there", "b": "short"}
    report = evaluate_response_cases(cases, responses)
    assert report["cases_total"] == 2
    assert report["cases_passed"] == 1
