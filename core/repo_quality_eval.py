"""
Repo response quality evaluation helpers.

Used to measure depth/appropriateness regressions for `/repo` style answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


NOISE_PATTERNS = [
    r"\bthinking process\b",
    r"\banalyze the request\b",
    r"\blet me analyze\b",
    r"<think>",
    r"</think>",
]


@dataclass(slots=True)
class RepoQualityResult:
    score: float
    checks: dict[str, bool]
    keyword_hit_rate: float
    gold_hit_rate: float
    evidence_hit_rate: float
    word_count: int
    noise_hits: list[str]
    failed_checks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "checks": self.checks,
            "keyword_hit_rate": self.keyword_hit_rate,
            "gold_hit_rate": self.gold_hit_rate,
            "evidence_hit_rate": self.evidence_hit_rate,
            "word_count": self.word_count,
            "noise_hits": self.noise_hits,
            "failed_checks": self.failed_checks,
        }


def _contains(text_l: str, token: str) -> bool:
    return token.lower() in text_l


def score_repo_response(
    text: str,
    *,
    expected_keywords: list[str] | None = None,
    gold_keywords: list[str] | None = None,
    required_evidence_sources: list[str] | None = None,
) -> RepoQualityResult:
    raw = str(text or "").strip()
    text_l = raw.lower()
    words = re.findall(r"\S+", raw)
    word_count = len(words)
    expected_keywords = expected_keywords or []
    gold_keywords = gold_keywords or []
    required_evidence_sources = required_evidence_sources or []

    checks = {
        "has_architecture": _contains(text_l, "architecture:"),
        "has_entry_points": _contains(text_l, "entry points:"),
        "has_key_modules": _contains(text_l, "key modules:"),
        "has_flows": _contains(text_l, "important flows:"),
        "has_evidence": _contains(text_l, "evidence:"),
        "has_confidence": _contains(text_l, "confidence:"),
        "has_risks": _contains(text_l, "risks:"),
        "has_next_steps": _contains(text_l, "next steps:"),
        "non_trivial_length": word_count >= 90,
    }

    hits = 0
    for kw in expected_keywords:
        kw_s = str(kw).strip().lower()
        if kw_s and kw_s in text_l:
            hits += 1
    keyword_hit_rate = float(hits / len(expected_keywords)) if expected_keywords else 1.0
    gold_hits = 0
    for kw in gold_keywords:
        kw_s = str(kw).strip().lower()
        if kw_s and kw_s in text_l:
            gold_hits += 1
    gold_hit_rate = float(gold_hits / len(gold_keywords)) if gold_keywords else 1.0
    evidence_hits = 0
    for src in required_evidence_sources:
        src_s = str(src).strip().lower()
        if src_s and src_s in text_l:
            evidence_hits += 1
    evidence_hit_rate = (
        float(evidence_hits / len(required_evidence_sources))
        if required_evidence_sources
        else 1.0
    )

    noise_hits: list[str] = []
    for pat in NOISE_PATTERNS:
        if re.search(pat, text_l):
            noise_hits.append(pat)

    # Weighted score normalized to [0,1]
    weights = {
        "has_architecture": 0.14,
        "has_entry_points": 0.12,
        "has_key_modules": 0.12,
        "has_flows": 0.12,
        "has_evidence": 0.14,
        "has_confidence": 0.10,
        "has_risks": 0.09,
        "has_next_steps": 0.09,
        "non_trivial_length": 0.08,
    }
    score = 0.0
    for k, w in weights.items():
        if checks.get(k, False):
            score += w
    semantic_multiplier = (0.45 * keyword_hit_rate) + (0.35 * gold_hit_rate) + (0.20 * evidence_hit_rate)
    score = score * (0.6 + 0.4 * semantic_multiplier)
    if noise_hits:
        score -= min(0.30, 0.08 * len(noise_hits))
    score = max(0.0, min(1.0, score))

    failed_checks = [k for k, ok in checks.items() if not ok]
    return RepoQualityResult(
        score=round(score, 4),
        checks=checks,
        keyword_hit_rate=round(keyword_hit_rate, 4),
        gold_hit_rate=round(gold_hit_rate, 4),
        evidence_hit_rate=round(evidence_hit_rate, 4),
        word_count=word_count,
        noise_hits=noise_hits,
        failed_checks=failed_checks,
    )


def evaluate_repo_cases(
    cases: list[dict[str, Any]],
    responses_by_id: dict[str, str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    passed = 0
    for case in cases:
        case_id = str(case.get("id", "")).strip()
        if not case_id:
            continue
        expected_keywords = case.get("expected_keywords", [])
        if not isinstance(expected_keywords, list):
            expected_keywords = []
        gold_keywords = case.get("gold_keywords", [])
        if not isinstance(gold_keywords, list):
            gold_keywords = []
        required_evidence_sources = case.get("required_evidence_sources", [])
        if not isinstance(required_evidence_sources, list):
            required_evidence_sources = []
        pass_threshold = float(case.get("pass_threshold", 0.72) or 0.72)
        response_text = str(responses_by_id.get(case_id, ""))
        result = score_repo_response(
            response_text,
            expected_keywords=expected_keywords,
            gold_keywords=gold_keywords,
            required_evidence_sources=required_evidence_sources,
        )
        ok = result.score >= pass_threshold
        if ok:
            passed += 1
        rows.append(
            {
                "id": case_id,
                "prompt": str(case.get("prompt", "")).strip(),
                "score": result.score,
                "pass_threshold": pass_threshold,
                "passed": ok,
                "required_evidence_sources": required_evidence_sources,
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
