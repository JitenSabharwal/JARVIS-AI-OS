"""
Research intelligence core for source ingestion, ranking, and digesting.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from infrastructure.hierarchical_rag import HierarchicalRAGIndex
from infrastructure.neo4j_graph_store import Neo4jGraphStore
from infrastructure.research_adapters import BaseResearchAdapter

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_iso(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:  # noqa: BLE001
        return time.time()


def _canon_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _canon_url(url: str) -> str:
    # Lightweight canonicalization: strip fragment + common tracking query params.
    raw = url.strip()
    if "#" in raw:
        raw = raw.split("#", 1)[0]
    if "?" not in raw:
        return raw
    base, query = raw.split("?", 1)
    parts = []
    for token in query.split("&"):
        k = token.split("=", 1)[0].lower()
        if k.startswith("utm_") or k in {"fbclid", "gclid"}:
            continue
        parts.append(token)
    if not parts:
        return base
    return f"{base}?{'&'.join(parts)}"


@dataclass
class ResearchSource:
    source_id: str
    title: str
    url: str
    content: str = ""
    topic: str = ""
    source_type: str = "blog"  # official | news | blog | social
    published_at: str = field(default_factory=_now_iso)
    ingested_at: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "topic": self.topic,
            "source_type": self.source_type,
            "published_at": self.published_at,
            "ingested_at": self.ingested_at,
            "metadata": self.metadata,
        }


@dataclass
class Watchlist:
    watchlist_id: str
    name: str
    topics: List[str]
    cadence: str = "daily"  # hourly | daily | weekly
    created_at: str = field(default_factory=_now_iso)
    last_digest_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "watchlist_id": self.watchlist_id,
            "name": self.name,
            "topics": list(self.topics),
            "cadence": self.cadence,
            "created_at": self.created_at,
            "last_digest_at": self.last_digest_at,
            "metadata": dict(self.metadata),
        }


class ResearchIntelligenceEngine:
    """In-memory research source registry with ranking and digest generation."""

    _SOURCE_TRUST = {
        "official": 1.0,
        "news": 0.85,
        "blog": 0.65,
        "social": 0.45,
    }
    _CADENCE_SECONDS = {
        "hourly": 3600,
        "daily": 86400,
        "weekly": 7 * 86400,
    }

    def __init__(self) -> None:
        self._sources: Dict[str, ResearchSource] = {}
        self._dedupe_index: Dict[str, str] = {}
        self._watchlists: Dict[str, Watchlist] = {}
        self._adapters: Dict[str, BaseResearchAdapter] = {}
        self._rag_index: HierarchicalRAGIndex = HierarchicalRAGIndex()
        self._graph_store: Neo4jGraphStore | None = None
        self._hierarchical_rag_enabled: bool = True

    def set_hierarchical_rag_enabled(self, enabled: bool) -> None:
        self._hierarchical_rag_enabled = bool(enabled)

    def set_graph_store(self, graph_store: Neo4jGraphStore | None) -> None:
        self._graph_store = graph_store

    def close(self) -> None:
        if self._graph_store is not None:
            try:
                self._graph_store.close()
            except Exception:  # noqa: BLE001
                pass

    def ingest_sources(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        inserted = 0
        skipped_duplicates = 0
        for item in items:
            title = str(item.get("title", "")).strip()
            url = _canon_url(str(item.get("url", "")).strip())
            if not title or not url:
                continue
            content = str(item.get("content", "") or "")
            topic = str(item.get("topic", "") or "")
            source_type = str(item.get("source_type", "blog") or "blog").lower()
            published_at = str(item.get("published_at", "") or _now_iso())
            dedupe_key = self._dedupe_key(title=title, url=url, content=content)
            if dedupe_key in self._dedupe_index:
                skipped_duplicates += 1
                continue
            source_id = f"src-{uuid.uuid4().hex}"
            src = ResearchSource(
                source_id=source_id,
                title=title,
                url=url,
                content=content,
                topic=topic,
                source_type=source_type,
                published_at=published_at,
                metadata=dict(item.get("metadata", {}) or {}),
            )
            self._sources[source_id] = src
            self._dedupe_index[dedupe_key] = source_id
            self._rag_index.index_document(
                source_id=source_id,
                title=title,
                content=content,
                metadata={"topic": topic, "source_type": source_type},
            )
            if self._graph_store is not None:
                self._graph_store.upsert_source(
                    source_id=source_id,
                    title=title,
                    url=url,
                    topic=topic,
                    source_type=source_type,
                )
                tree = self._rag_index.get_source_tree(source_id)
                for node in tree.get("nodes", []):
                    self._graph_store.upsert_node(
                        node_id=str(node.get("node_id", "")),
                        source_id=source_id,
                        level=int(node.get("level", 0)),
                        title=str(node.get("title", "")),
                        content=str(node.get("content", "")),
                        parent_id=str(node.get("parent_id", "")) or None,
                    )
            inserted += 1
        return {
            "inserted": inserted,
            "skipped_duplicates": skipped_duplicates,
            "total_sources": len(self._sources),
        }

    def query(
        self,
        topic: str,
        *,
        max_results: int = 5,
        freshness_days: int = 30,
        min_trust: float = 0.0,
    ) -> Dict[str, Any]:
        query_text = _canon_text(topic)
        if not query_text:
            return {"topic": topic, "results": [], "citations": [], "contradictions": []}
        now_ts = time.time()
        candidates: List[tuple[float, ResearchSource]] = []
        for src in self._sources.values():
            combined = _canon_text(f"{src.topic} {src.title} {src.content}")
            if query_text not in combined and not any(tok in combined for tok in query_text.split()):
                continue
            age_days = max(0.0, (now_ts - _parse_iso(src.published_at)) / 86400.0)
            if freshness_days > 0 and age_days > float(freshness_days):
                continue
            trust = self._SOURCE_TRUST.get(src.source_type, 0.5)
            if trust < float(min_trust):
                continue
            relevance = self._relevance_score(query_text, combined)
            freshness = max(0.0, 1.0 - (age_days / max(1.0, float(freshness_days))))
            score = round((relevance * 0.6) + (freshness * 0.25) + (trust * 0.15), 4)
            candidates.append((score, src))

        ranked = sorted(candidates, key=lambda t: t[0], reverse=True)[: max(1, int(max_results))]
        results = []
        for score, src in ranked:
            item = src.to_dict()
            item["score"] = score
            item["citation_quality_score"] = round(
                (self._SOURCE_TRUST.get(src.source_type, 0.5) * 0.7)
                + (score * 0.3),
                4,
            )
            item["citation"] = {
                "title": src.title,
                "url": src.url,
                "published_at": src.published_at,
                "source_type": src.source_type,
            }
            results.append(item)
        contradictions = self.detect_contradictions(results)
        citations = [r["citation"] for r in results]
        coverage = self._source_type_coverage(results)
        citation_health = self._citation_health_score(results)
        rag_context = []
        if self._hierarchical_rag_enabled:
            rag_q = self._rag_index.query(query=topic, max_nodes=max_results * 2, expand_neighbors=True)
            rag_context = rag_q.get("nodes", [])
            self._attach_rag_context(results, rag_q.get("contexts", []))
        graph_context: Dict[str, Any] = {"enabled": False, "relationships": []}
        if self._graph_store is not None:
            relationships: List[Dict[str, Any]] = []
            for item in results[:3]:
                source_id = str(item.get("source_id", ""))
                if source_id:
                    relationships.extend(self._graph_store.query_related(source_id=source_id, limit=4))
            graph_context = {
                "enabled": True,
                "relationship_count": len(relationships),
                "relationships": relationships[:24],
            }
        return {
            "topic": topic,
            "result_count": len(results),
            "results": results,
            "citations": citations,
            "contradictions": contradictions,
            "source_type_coverage": coverage,
            "citation_health_score": citation_health,
            "rag_context_count": len(rag_context),
            "rag_context": rag_context,
            "graph_context": graph_context,
        }

    def create_watchlist(
        self,
        *,
        name: str,
        topics: List[str],
        cadence: str = "daily",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        clean_topics = [str(t).strip() for t in topics if str(t).strip()]
        if not name.strip():
            raise ValueError("watchlist name is required")
        if not clean_topics:
            raise ValueError("watchlist topics must be non-empty")
        watchlist = Watchlist(
            watchlist_id=f"wl-{uuid.uuid4().hex}",
            name=name.strip(),
            topics=clean_topics,
            cadence=cadence.strip() or "daily",
            metadata=metadata or {},
        )
        self._watchlists[watchlist.watchlist_id] = watchlist
        return watchlist.to_dict()

    def list_watchlists(self) -> List[Dict[str, Any]]:
        return [w.to_dict() for w in self._watchlists.values()]

    def register_adapter(self, adapter: BaseResearchAdapter, *, overwrite: bool = False) -> None:
        if adapter.name in self._adapters and not overwrite:
            raise ValueError(f"adapter already registered: {adapter.name}")
        self._adapters[adapter.name] = adapter

    def list_adapters(self) -> List[Dict[str, Any]]:
        return [
            {"name": a.name, "description": a.description}
            for a in self._adapters.values()
        ]

    def run_adapters(self, *, topic: str, max_items_per_adapter: int = 10) -> Dict[str, Any]:
        inserted_total = 0
        skipped_total = 0
        adapter_results: List[Dict[str, Any]] = []
        for adapter in self._adapters.values():
            items = adapter.fetch_sources(topic=topic, max_items=max_items_per_adapter)
            ingest = self.ingest_sources(items)
            inserted_total += int(ingest.get("inserted", 0))
            skipped_total += int(ingest.get("skipped_duplicates", 0))
            adapter_results.append(
                {
                    "adapter": adapter.name,
                    "fetched": len(items),
                    "inserted": int(ingest.get("inserted", 0)),
                    "skipped_duplicates": int(ingest.get("skipped_duplicates", 0)),
                }
            )
        return {
            "topic": topic,
            "adapter_count": len(self._adapters),
            "inserted_total": inserted_total,
            "skipped_duplicates_total": skipped_total,
            "adapter_results": adapter_results,
        }

    def get_source_tree(self, source_id: str) -> Dict[str, Any]:
        return self._rag_index.get_source_tree(source_id)

    def graph_health(self) -> Dict[str, Any]:
        if self._graph_store is None:
            return {"enabled": False, "healthy": False, "reason": "graph_store_not_configured"}
        return self._graph_store.health()

    def generate_digest(self, watchlist_id: str, *, max_per_topic: int = 3) -> Dict[str, Any]:
        watch = self._watchlists.get(watchlist_id)
        if watch is None:
            raise KeyError(f"watchlist not found: {watchlist_id}")
        sections: List[Dict[str, Any]] = []
        for topic in watch.topics:
            # Stricter default freshness for digests.
            query_result = self.query(topic, max_results=max_per_topic, freshness_days=14, min_trust=0.45)
            sections.append(
                {
                    "topic": topic,
                    "count": query_result["result_count"],
                    "items": query_result["results"],
                    "contradictions": query_result["contradictions"],
                    "source_type_coverage": query_result.get("source_type_coverage", {}),
                    "citation_health_score": query_result.get("citation_health_score", 0.0),
                }
            )
        now_iso = _now_iso()
        watch.last_digest_at = now_iso
        return {
            "watchlist_id": watch.watchlist_id,
            "name": watch.name,
            "generated_at": now_iso,
            "sections": sections,
        }

    def run_due_digests(self, *, max_per_topic: int = 3, now_ts: Optional[float] = None) -> Dict[str, Any]:
        now = now_ts if now_ts is not None else time.time()
        generated: List[Dict[str, Any]] = []
        skipped: List[str] = []
        for watch in self._watchlists.values():
            if not self._is_watchlist_due(watch, now):
                skipped.append(watch.watchlist_id)
                continue
            digest = self.generate_digest(watch.watchlist_id, max_per_topic=max_per_topic)
            generated.append(digest)
        return {
            "generated_count": len(generated),
            "generated": generated,
            "skipped_count": len(skipped),
            "skipped_watchlist_ids": skipped,
        }

    def detect_contradictions(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Heuristic polarity checks on claim key (explicit) or title.
        claims: Dict[str, Dict[str, List[str]]] = {}
        for item in results:
            claim_key = _canon_text(
                str(item.get("metadata", {}).get("claim_key", "") or item.get("title", ""))
            )
            if not claim_key:
                continue
            text = _canon_text(f"{item.get('title', '')} {item.get('content', '')}")
            polarity = self._polarity(text)
            bucket = claims.setdefault(claim_key, {"positive": [], "negative": [], "neutral": []})
            bucket[polarity].append(str(item.get("url", "")))
        contradictions = []
        for claim, bucket in claims.items():
            if bucket["positive"] and bucket["negative"]:
                contradictions.append(
                    {
                        "claim_key": claim,
                        "positive_sources": bucket["positive"],
                        "negative_sources": bucket["negative"],
                    }
                )
        return contradictions

    @staticmethod
    def _dedupe_key(*, title: str, url: str, content: str) -> str:
        h = hashlib.sha256()
        h.update(_canon_text(title).encode("utf-8"))
        h.update(b"|")
        h.update(_canon_text(url).encode("utf-8"))
        h.update(b"|")
        # only prefix of content for stable low-cost hashing
        h.update(_canon_text(content[:500]).encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _relevance_score(query: str, text: str) -> float:
        toks = [t for t in query.split() if t]
        if not toks:
            return 0.0
        hits = sum(1 for t in toks if t in text)
        return hits / max(1, len(toks))

    @staticmethod
    def _polarity(text: str) -> str:
        neg = {"not", "no", "false", "decline", "decrease", "down", "drop", "fall"}
        pos = {"yes", "true", "increase", "up", "rise", "growth", "gain"}
        neg_hits = sum(1 for t in neg if re.search(rf"\b{re.escape(t)}\b", text))
        pos_hits = sum(1 for t in pos if re.search(rf"\b{re.escape(t)}\b", text))
        if pos_hits > neg_hits:
            return "positive"
        if neg_hits > pos_hits:
            return "negative"
        return "neutral"

    def _is_watchlist_due(self, watch: Watchlist, now_ts: float) -> bool:
        cadence = (watch.cadence or "daily").lower()
        interval = self._CADENCE_SECONDS.get(cadence, self._CADENCE_SECONDS["daily"])
        if not watch.last_digest_at:
            return True
        last = _parse_iso(watch.last_digest_at)
        return (now_ts - last) >= float(interval)

    @staticmethod
    def _source_type_coverage(results: List[Dict[str, Any]]) -> Dict[str, int]:
        coverage: Dict[str, int] = {}
        for item in results:
            st = str(item.get("source_type", "unknown"))
            coverage[st] = coverage.get(st, 0) + 1
        return coverage

    @staticmethod
    def _citation_health_score(results: List[Dict[str, Any]]) -> float:
        if not results:
            return 0.0
        base = 0.0
        for item in results:
            base += float(item.get("citation_quality_score", 0.0))
        # reward source diversity slightly
        diversity = len({str(i.get("source_type", "")) for i in results})
        return round((base / len(results)) + (0.03 * diversity), 4)

    @staticmethod
    def _attach_rag_context(results: List[Dict[str, Any]], contexts: List[Dict[str, Any]]) -> None:
        if not results or not contexts:
            return
        by_source: Dict[str, List[Dict[str, Any]]] = {}
        for ctx in contexts:
            parent = ctx.get("parent") or {}
            source_id = str(parent.get("source_id", "")) or ""
            if not source_id:
                # fallback through focus node lookup
                focus = ctx.get("focus_node_id", "")
                source_id = str(focus).split(":")[0] if ":" in str(focus) else ""
            if source_id:
                by_source.setdefault(source_id, []).append(ctx)
        for item in results:
            sid = str(item.get("source_id", ""))
            item["rag_supporting_context"] = by_source.get(sid, [])[:2]
