from __future__ import annotations

from core.repo_quality_eval import evaluate_repo_cases, score_repo_response


def test_score_repo_response_good_shape() -> None:
    text = (
        "This is a layered service.\n"
        "Architecture: API routes through orchestrator and specialized agents.\n"
        "Entry points: jarvis_main.py, main.py.\n"
        "Key modules: interfaces, agents, core, infrastructure.\n"
        "Important flows: request -> routing -> execution -> response.\n"
        "Evidence: Entry points identified from startup files. Sources: jarvis_main.py, main.py.\n"
        "Confidence: 0.84. Coverage: 41.0% of repository files.\n"
        "Risks: weak tests on edge paths.\n"
        "Next steps: add regression tests and verify production config.\n"
    )
    result = score_repo_response(
        text,
        expected_keywords=["interfaces", "agents", "confidence"],
        gold_keywords=["orchestrator", "specialized agents"],
        required_evidence_sources=["jarvis_main.py", "interfaces/api_interface.py"],
    )
    assert result.score >= 0.72
    assert result.checks["has_architecture"] is True
    assert result.checks["has_evidence"] is True
    assert result.gold_hit_rate > 0.0
    assert result.evidence_hit_rate > 0.0
    assert result.noise_hits == []


def test_score_repo_response_penalizes_noise() -> None:
    text = "Thinking Process: Let me analyze the request step by step."
    result = score_repo_response(text, expected_keywords=["architecture"])
    assert result.score < 0.35
    assert result.noise_hits


def test_evaluate_repo_cases_counts_passes() -> None:
    cases = [
        {
            "id": "c1",
            "prompt": "p1",
            "expected_keywords": ["architecture"],
            "gold_keywords": ["entry points"],
            "required_evidence_sources": ["main.py"],
            "pass_threshold": 0.5,
        },
        {"id": "c2", "prompt": "p2", "expected_keywords": ["architecture"], "pass_threshold": 0.8},
    ]
    responses = {
        "c1": "Architecture: good. Entry points: a. Key modules: b. Important flows: c. Evidence: x. Confidence: 0.8. Risks: r. Next steps: n.",
        "c2": "short",
    }
    report = evaluate_repo_cases(cases, responses)
    assert report["cases_total"] == 2
    assert report["cases_passed"] == 1


def test_score_repo_response_penalizes_missing_required_evidence_sources() -> None:
    text = (
        "Architecture: API routes through orchestrator.\n"
        "Entry points: jarvis_main.py.\n"
        "Key modules: interfaces, core.\n"
        "Important flows: request -> route -> response.\n"
        "Evidence: startup file observed. Sources: jarvis_main.py.\n"
        "Confidence: 0.8.\n"
        "Risks: medium.\n"
        "Next steps: add tests.\n"
    )
    with_evidence = score_repo_response(
        text,
        expected_keywords=["architecture", "entry points"],
        required_evidence_sources=["jarvis_main.py"],
    )
    without_evidence = score_repo_response(
        text,
        expected_keywords=["architecture", "entry points"],
        required_evidence_sources=["interfaces/api_interface.py"],
    )
    assert with_evidence.evidence_hit_rate > without_evidence.evidence_hit_rate
    assert with_evidence.score > without_evidence.score
