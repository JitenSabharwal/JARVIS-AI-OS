"""
Deterministic route-level fallback responses.
"""

from __future__ import annotations


_FALLBACKS: dict[str, str] = {
    "chat": "I couldn't complete that response cleanly. Please retry your request.",
    "repo": "I couldn't generate a valid repository analysis yet. Please retry with the same workspace.",
    "code_assist": "I couldn't generate a safe code-assist response. Please retry with a clearer instruction.",
    "code_workflow": "I couldn't complete the workflow response. Please retry or reduce the task scope.",
}


def get_fallback(route: str) -> str:
    key = str(route or "chat").strip().lower()
    return _FALLBACKS.get(key, _FALLBACKS["chat"])

