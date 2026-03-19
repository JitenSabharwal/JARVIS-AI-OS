"""
Ingest quality policy for research source relevance and quarantine decisions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import urlparse


_SOURCE_TRUST = {
    "official": 1.0,
    "news": 0.85,
    "blog": 0.65,
    "social": 0.45,
}


def _canon(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _token_overlap(query: str, text: str) -> float:
    q_tokens = [t for t in _canon(query).split() if t]
    if not q_tokens:
        return 0.0
    body = _canon(text)
    hits = sum(1 for token in q_tokens if token in body)
    return hits / max(1, len(q_tokens))


def evaluate_ingest_quality(
    *,
    title: str,
    url: str,
    content: str,
    topic: str,
    source_type: str,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    meta = dict(metadata or {})
    override = bool(meta.get("quality_override", False))
    content_len = len(str(content or ""))
    host = urlparse(str(url or "")).netloc.lower()

    min_len_ok = content_len >= 120
    length_score = min(1.0, content_len / 1200.0)
    trust_score = _SOURCE_TRUST.get(str(source_type or "blog").lower(), 0.5)
    text = f"{title}\n{content}"
    topic_score = _token_overlap(topic, text) if str(topic or "").strip() else 0.4

    # Penalize obvious low-signal hosts/pages.
    host_penalty = 0.0
    if any(h in host for h in ("login.", "accounts.", "auth.", "signin")):
        host_penalty = 0.25

    relevance_score = max(0.0, min(1.0, (topic_score * 0.45) + (length_score * 0.25) + (trust_score * 0.30) - host_penalty))
    threshold = 0.46

    reasons: List[str] = []
    if not min_len_ok:
        reasons.append("content_too_short")
    if topic_score < 0.18 and str(topic or "").strip():
        reasons.append("low_topic_alignment")
    if trust_score < 0.5:
        reasons.append("low_source_trust")
    if host_penalty > 0:
        reasons.append("low_signal_host_pattern")

    quarantined = (not override) and (not min_len_ok or relevance_score < threshold)
    status = "approved" if not quarantined else "quarantined"
    return {
        "status": status,
        "quarantined": quarantined,
        "override": override,
        "threshold": threshold,
        "relevance_score": round(relevance_score, 4),
        "signals": {
            "topic_score": round(topic_score, 4),
            "length_score": round(length_score, 4),
            "trust_score": round(trust_score, 4),
            "content_length": content_len,
        },
        "reasons": reasons,
    }


__all__ = ["evaluate_ingest_quality"]
