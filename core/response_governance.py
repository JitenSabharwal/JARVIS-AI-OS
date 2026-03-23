"""
Route-level response governance contracts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.response_finalizer import finalize_user_response
from core.response_fallbacks import get_fallback


@dataclass(slots=True)
class ResponseContract:
    route: str
    min_words: int = 1
    max_words: int = 260
    must_not_contain: tuple[str, ...] = (
        "thinking process",
        "analyze the request",
        "<think>",
        "</think>",
    )
    fallback: str = ""


@dataclass(slots=True)
class GovernanceResult:
    text: str
    route: str
    changed: bool
    rejected: bool
    reason: str = ""
    word_count: int = 0
    verbosity_tier: str = "default"
    min_words: int = 0
    max_words: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "changed": self.changed,
            "rejected": self.rejected,
            "reason": self.reason,
            "word_count": self.word_count,
            "verbosity_tier": self.verbosity_tier,
            "min_words": self.min_words,
            "max_words": self.max_words,
        }


_CONTRACTS: dict[str, ResponseContract] = {
    "chat": ResponseContract(route="chat", min_words=1, max_words=260, fallback=get_fallback("chat")),
    "repo": ResponseContract(route="repo", min_words=40, max_words=700, fallback=get_fallback("repo")),
    "code_assist": ResponseContract(route="code_assist", min_words=4, max_words=120, fallback=get_fallback("code_assist")),
    "code_workflow": ResponseContract(route="code_workflow", min_words=4, max_words=150, fallback=get_fallback("code_workflow")),
}


def apply_response_governance(
    text: str,
    *,
    route: str,
    hints: dict[str, Any] | None = None,
) -> GovernanceResult:
    route_key = str(route or "chat").strip().lower()
    base_contract = _CONTRACTS.get(route_key) or _CONTRACTS["chat"]
    contract, verbosity_tier = _derive_contract(base_contract, hints or {})
    original = str(text or "")
    orig_low = original.lower()
    for marker in contract.must_not_contain:
        if marker in orig_low:
            return GovernanceResult(
                text=contract.fallback,
                route=contract.route,
                changed=True,
                rejected=True,
                reason=f"contains_forbidden_marker:{marker}",
                word_count=0,
                verbosity_tier=verbosity_tier,
                min_words=int(contract.min_words),
                max_words=int(contract.max_words),
            )
    finalized = finalize_user_response(original, fallback=contract.fallback).strip()
    changed = finalized.strip() != original.strip()
    lowered = finalized.lower()
    for marker in contract.must_not_contain:
        if marker in lowered:
            return GovernanceResult(
                text=contract.fallback,
                route=contract.route,
                changed=True,
                rejected=True,
                reason=f"contains_forbidden_marker:{marker}",
                word_count=0,
                verbosity_tier=verbosity_tier,
                min_words=int(contract.min_words),
                max_words=int(contract.max_words),
            )
    wc = len(re.findall(r"\S+", finalized))
    if wc < int(contract.min_words):
        return GovernanceResult(
            text=contract.fallback,
            route=contract.route,
            changed=True,
            rejected=True,
            reason=f"too_short:{wc}<{contract.min_words}",
            word_count=wc,
            verbosity_tier=verbosity_tier,
            min_words=int(contract.min_words),
            max_words=int(contract.max_words),
        )
    if wc > int(contract.max_words):
        finalized = _truncate_to_word_budget_preserving_sentences(finalized, int(contract.max_words))
        changed = True
        wc = len(re.findall(r"\S+", finalized))
    return GovernanceResult(
        text=finalized,
        route=contract.route,
        changed=changed,
        rejected=False,
        reason="",
        word_count=wc,
        verbosity_tier=verbosity_tier,
        min_words=int(contract.min_words),
        max_words=int(contract.max_words),
    )


def _derive_contract(base: ResponseContract, hints: dict[str, Any]) -> tuple[ResponseContract, str]:
    contract = ResponseContract(
        route=base.route,
        min_words=int(base.min_words),
        max_words=int(base.max_words),
        must_not_contain=base.must_not_contain,
        fallback=base.fallback,
    )
    tier = "default"
    response_plan = hints.get("response_plan")
    query_understanding = hints.get("query_understanding")
    user_input = str(hints.get("user_input", "") or "").strip().lower()
    target_length = ""
    task_type = ""
    inferred_intent = ""
    if isinstance(response_plan, dict):
        target_length = str(response_plan.get("target_length", "")).strip().lower()
        task_type = str(response_plan.get("task_type", "")).strip().lower()
    if isinstance(query_understanding, dict):
        inferred_intent = str(query_understanding.get("inferred_intent", "")).strip().lower()

    if contract.route == "chat":
        if target_length == "short":
            contract.max_words = min(int(contract.max_words), 120)
            tier = "short"
        elif target_length == "medium":
            contract.max_words = max(int(contract.max_words), 360)
            tier = "medium"
        elif target_length == "long":
            contract.min_words = max(int(contract.min_words), 12)
            contract.max_words = max(int(contract.max_words), 700)
            tier = "long"

        if task_type in {"time_query", "weather_query"}:
            contract.max_words = min(int(contract.max_words), 80)
            tier = "short"

        why_reasoning = task_type == "reasoning_why" or inferred_intent == "why_reasoning"
        if why_reasoning:
            contract.min_words = max(int(contract.min_words), 16)
            contract.max_words = max(int(contract.max_words), 700)
            tier = "long"

        if re.search(r"\b(in detail|step by step|deep dive|go deeper|tradeoff|compare)\b", user_input):
            contract.min_words = max(int(contract.min_words), 14)
            contract.max_words = max(int(contract.max_words), 700)
            tier = "long"

    if int(contract.max_words) < int(contract.min_words):
        contract.max_words = int(contract.min_words)
    return contract, tier


def _truncate_to_word_budget_preserving_sentences(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return text.strip()
    # Try sentence-aware trimming first so response remains coherent.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    count = 0
    for sentence in sentences:
        sentence_words = re.findall(r"\S+", sentence)
        if not sentence_words:
            continue
        next_count = count + len(sentence_words)
        if next_count > max_words:
            break
        kept.append(sentence.strip())
        count = next_count
    if kept:
        return " ".join(kept).strip()
    return " ".join(words[:max_words]).strip()
