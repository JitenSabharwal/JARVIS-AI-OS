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

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "changed": self.changed,
            "rejected": self.rejected,
            "reason": self.reason,
            "word_count": self.word_count,
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
) -> GovernanceResult:
    route_key = str(route or "chat").strip().lower()
    contract = _CONTRACTS.get(route_key) or _CONTRACTS["chat"]
    original = str(text or "")
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
        )
    if wc > int(contract.max_words):
        words = re.findall(r"\S+", finalized)
        finalized = " ".join(words[: contract.max_words]).strip()
        changed = True
    return GovernanceResult(
        text=finalized,
        route=contract.route,
        changed=changed,
        rejected=False,
        reason="",
        word_count=len(re.findall(r"\S+", finalized)),
    )
