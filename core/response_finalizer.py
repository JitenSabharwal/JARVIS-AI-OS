"""
Central assistant response finalization and noise filtering.
"""

from __future__ import annotations

import re


_LEADING_META_PATTERNS = (
    r"(?is)^\s*thinking process\s*:.*$",
    r"(?is)^\s*analyze the request\s*:.*$",
    r"(?is)^\s*reasoning/explanation\s*:.*$",
    r"(?is)^\s*response policy\s*:.*$",
    r"(?is)^\s*user request\s*:.*$",
    r"(?is)^\s*assistant draft response\s*:.*$",
)

_INLINE_META_MARKERS = (
    "thinking process",
    "analyze the request",
    "<think>",
    "</think>",
)


def finalize_user_response(text: str, *, fallback: str = "I'm sorry, I couldn't generate a valid response.") -> str:
    raw = str(text or "").strip()
    if not raw:
        return fallback
    salvaged = _salvage_from_reasoning_preamble(raw)
    if salvaged:
        raw = salvaged
    lowered = raw.lower()
    for pat in _LEADING_META_PATTERNS:
        if re.match(pat, raw):
            return fallback
    lines = [ln.rstrip() for ln in raw.splitlines()]
    kept: list[str] = []
    for line in lines:
        ll = line.strip().lower()
        if ll and any(marker in ll for marker in _INLINE_META_MARKERS):
            # Stop at meta leakage tails.
            if kept:
                break
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    if not cleaned:
        return fallback
    return cleaned


def _salvage_from_reasoning_preamble(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if not re.match(r"(?is)\A\s*(let me analy[sz]e|thinking process)\b", raw):
        return ""
    lines = [ln.rstrip() for ln in raw.splitlines()]
    if not lines:
        return ""
    start = -1
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if re.match(r"^\d+\.\s+", s):
            continue
        if low.startswith(("let me analyze", "let me analyse", "thinking process", "the user", "since this", "i should")):
            continue
        if low.endswith(":"):
            continue
        if len(s.split()) >= 6 or low.startswith(("absolutely", "sure", "yes", "hello", "hi", "here")):
            start = idx
            break
    if start < 0:
        return ""
    return "\n".join(lines[start:]).strip()
