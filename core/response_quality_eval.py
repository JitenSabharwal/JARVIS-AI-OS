"""
General response quality evaluator for chat/code/workflow routes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


META_NOISE_MARKERS = (
    "thinking process",
    "analyze the request",
    "<think>",
    "</think>",
)


@dataclass(slots=True)
class ResponseQualityResult:
    score: float
    word_count: int
    phrase_hit_rate: float
    noise_hits: list[str]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "word_count": self.word_count,
            "phrase_hit_rate": self.phrase_hit_rate,
            "noise_hits": self.noise_hits,
            "passed": self.passed,
        }


def score_response(
    text: str,
    *,
    expected_phrases: list[str] | None = None,
    min_words: int = 1,
    max_words: int = 220,
    pass_threshold: float = 0.7,
) -> ResponseQualityResult:
    raw = str(text or "").strip()
    lowered = raw.lower()
    words = re.findall(r"\S+", raw)
    wc = len(words)
    expected = [str(x).strip().lower() for x in (expected_phrases or []) if str(x).strip()]
    hits = 0
    for phrase in expected:
        if phrase in lowered:
            hits += 1
    phrase_hit_rate = float(hits / len(expected)) if expected else 1.0
    noise_hits = [m for m in META_NOISE_MARKERS if m in lowered]
    length_score = 1.0 if min_words <= wc <= max_words else 0.0
    score = (0.6 * phrase_hit_rate) + (0.4 * length_score)
    if noise_hits:
        score -= min(0.35, 0.10 * len(noise_hits))
    score = max(0.0, min(1.0, score))
    passed = score >= float(pass_threshold)
    return ResponseQualityResult(
        score=round(score, 4),
        word_count=wc,
        phrase_hit_rate=round(phrase_hit_rate, 4),
        noise_hits=noise_hits,
        passed=passed,
    )


def evaluate_response_cases(
    cases: list[dict[str, Any]],
    responses_by_id: dict[str, str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    passed = 0
    for case in cases:
        case_id = str(case.get("id", "")).strip()
        if not case_id:
            continue
        expected = case.get("expected_phrases", [])
        if not isinstance(expected, list):
            expected = []
        min_words = int(case.get("min_words", 1) or 1)
        max_words = int(case.get("max_words", 220) or 220)
        pass_threshold = float(case.get("pass_threshold", 0.7) or 0.7)
        result = score_response(
            str(responses_by_id.get(case_id, "")),
            expected_phrases=expected,
            min_words=min_words,
            max_words=max_words,
            pass_threshold=pass_threshold,
        )
        if result.passed:
            passed += 1
        rows.append(
            {
                "id": case_id,
                "route": str(case.get("route", "")),
                "score": result.score,
                "pass_threshold": pass_threshold,
                "result": result.to_dict(),
            }
        )
    total = len(rows)
    return {
        "version": 1,
        "cases_total": total,
        "cases_passed": passed,
        "pass_rate": round(float(passed / total), 4) if total else 0.0,
        "cases": rows,
    }

