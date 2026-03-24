"""
World knowledge memory + enrichment service.

Supports:
- teaching concepts with notes/tags/examples
- enriching concepts from web-backed research engine
- enriching detected labels with stored + web context
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any
import json
from urllib.parse import urlparse

from infrastructure.logger import get_logger

logger = get_logger("world_knowledge")


def _now_ts() -> float:
    return time.time()


class WorldKnowledgeService:
    def __init__(self, *, state_path: str = "data/world_knowledge.json") -> None:
        self._state_path = Path(state_path).expanduser()
        self._concepts: dict[str, dict[str, Any]] = {}
        self._topic_index: dict[str, str] = {}
        self._label_web_cooldown: dict[str, float] = {}
        self._load()

    @classmethod
    def from_env(cls) -> "WorldKnowledgeService":
        path = str(os.getenv("JARVIS_WORLD_KNOWLEDGE_PATH", "data/world_knowledge.json")).strip()
        return cls(state_path=path or "data/world_knowledge.json")

    def list_concepts(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = list(self._concepts.values())
        rows.sort(key=lambda r: float(r.get("updated_at", 0.0)), reverse=True)
        lim = max(1, min(500, int(limit)))
        return [self._public_row(r) for r in rows[:lim]]

    def get_concept(self, *, concept_id: str) -> dict[str, Any]:
        row = self._concepts.get(str(concept_id or "").strip())
        if row is None:
            raise KeyError("concept not found")
        return self._public_row(row)

    def teach_concept(
        self,
        *,
        topic: str,
        notes: str = "",
        tags: list[str] | None = None,
        detections: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = str(topic or "").strip().lower()
        display_topic = str(topic or "").strip()
        if not normalized:
            raise ValueError("topic is required")
        cid = self._topic_index.get(normalized)
        now = _now_ts()
        if not cid:
            cid = f"wk_{uuid.uuid4().hex[:12]}"
            row = {
                "concept_id": cid,
                "topic": display_topic,
                "topic_normalized": normalized,
                "notes": [],
                "tags": [],
                "detections": [],
                "metadata": {},
                "web_facts": [],
                "created_at": now,
                "updated_at": now,
            }
            self._concepts[cid] = row
            self._topic_index[normalized] = cid
        row = self._concepts[cid]
        if str(notes or "").strip():
            row["notes"] = [str(notes).strip()] + list(row.get("notes", []))
            row["notes"] = row["notes"][:50]
        merged_tags = set(str(t).strip().lower() for t in list(row.get("tags", [])) if str(t).strip())
        for t in tags or []:
            clean = str(t).strip().lower()
            if clean:
                merged_tags.add(clean)
        row["tags"] = sorted(merged_tags)
        if isinstance(detections, list):
            drows = [d for d in detections if isinstance(d, dict)]
            row["detections"] = (drows + list(row.get("detections", [])))[:120]
        if isinstance(metadata, dict):
            md = dict(row.get("metadata", {}) or {})
            md.update({k: v for k, v in metadata.items() if k})
            row["metadata"] = md
        row["updated_at"] = now
        self._save()
        return self._public_row(row)

    def update_concept(
        self,
        *,
        concept_id: str,
        topic: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
        detections: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = self._concepts.get(str(concept_id or "").strip())
        if row is None:
            raise KeyError("concept not found")
        now = _now_ts()
        if topic is not None:
            next_topic = str(topic).strip()
            if not next_topic:
                raise ValueError("topic cannot be empty")
            next_norm = next_topic.lower()
            prev_norm = str(row.get("topic_normalized", "")).strip().lower()
            existing = self._topic_index.get(next_norm)
            if existing and existing != str(row.get("concept_id", "")):
                raise ValueError("topic already exists")
            if prev_norm and prev_norm in self._topic_index:
                self._topic_index.pop(prev_norm, None)
            row["topic"] = next_topic
            row["topic_normalized"] = next_norm
            self._topic_index[next_norm] = str(row.get("concept_id", ""))
        if notes is not None and str(notes).strip():
            row["notes"] = [str(notes).strip()] + list(row.get("notes", []))
            row["notes"] = row["notes"][:50]
        if tags is not None:
            merged_tags = set(str(t).strip().lower() for t in list(row.get("tags", [])) if str(t).strip())
            for t in tags:
                clean = str(t).strip().lower()
                if clean:
                    merged_tags.add(clean)
            row["tags"] = sorted(merged_tags)
        if detections is not None:
            clean_detections = [d for d in detections if isinstance(d, dict)]
            row["detections"] = (clean_detections + list(row.get("detections", [])))[:120]
        if metadata is not None:
            md = dict(row.get("metadata", {}) or {})
            md.update({k: v for k, v in metadata.items() if str(k).strip()})
            row["metadata"] = md
        row["updated_at"] = now
        self._save()
        return self._public_row(row)

    def delete_concept(self, *, concept_id: str) -> bool:
        cid = str(concept_id or "").strip()
        row = self._concepts.pop(cid, None)
        if row is None:
            return False
        topic_norm = str(row.get("topic_normalized", "")).strip().lower()
        if topic_norm:
            self._topic_index.pop(topic_norm, None)
        self._save()
        return True

    def add_reference_link(
        self,
        *,
        concept_id: str,
        url: str,
        title: str = "",
        notes: str = "",
        source_type: str = "manual",
        tags: list[str] | None = None,
        interaction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = self._concepts.get(str(concept_id or "").strip())
        if row is None:
            raise KeyError("concept not found")
        clean_url = str(url or "").strip()
        if not clean_url:
            raise ValueError("url is required")
        refs = list(row.get("reference_links", []))
        dedupe = str(clean_url).lower()
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            if str(ref.get("url", "")).strip().lower() == dedupe:
                rid = str(ref.get("link_id", "")).strip() or f"wrl_{uuid.uuid4().hex[:12]}"
                ref["link_id"] = rid
                ref["title"] = str(title or ref.get("title", "")).strip()[:180]
                ref["notes"] = str(notes or ref.get("notes", "")).strip()[:400]
                ref["source_type"] = str(source_type or ref.get("source_type", "manual")).strip() or "manual"
                ref["tags"] = sorted(
                    set(
                        str(t).strip().lower()
                        for t in ((tags or []) + list(ref.get("tags", [])))
                        if str(t).strip()
                    )
                )
                ref["updated_at"] = _now_ts()
                row["reference_links"] = refs[:150]
                if isinstance(interaction, dict):
                    self._append_interaction(row=row, link_id=rid, interaction=interaction)
                row["updated_at"] = _now_ts()
                self._save()
                return self._public_row(row)
        link_id = f"wrl_{uuid.uuid4().hex[:12]}"
        refs.insert(
            0,
            {
                "link_id": link_id,
                "url": clean_url,
                "title": str(title or "").strip()[:180],
                "notes": str(notes or "").strip()[:400],
                "source_type": str(source_type or "manual").strip() or "manual",
                "tags": sorted(set(str(t).strip().lower() for t in (tags or []) if str(t).strip())),
                "created_at": _now_ts(),
                "updated_at": _now_ts(),
            },
        )
        row["reference_links"] = refs[:150]
        if isinstance(interaction, dict):
            self._append_interaction(row=row, link_id=link_id, interaction=interaction)
        row["updated_at"] = _now_ts()
        self._save()
        return self._public_row(row)

    def remove_reference_link(self, *, concept_id: str, link_id: str) -> bool:
        row = self._concepts.get(str(concept_id or "").strip())
        if row is None:
            raise KeyError("concept not found")
        lid = str(link_id or "").strip()
        refs = [r for r in list(row.get("reference_links", [])) if str(r.get("link_id", "")).strip() != lid]
        deleted = len(refs) != len(list(row.get("reference_links", [])))
        if deleted:
            row["reference_links"] = refs
            row["updated_at"] = _now_ts()
            self._save()
        return deleted

    def log_link_interaction(
        self,
        *,
        concept_id: str,
        link_id: str,
        summary: str,
        extracted_facts: list[str] | None = None,
        pattern_hint: str = "",
        outcome: str = "",
    ) -> dict[str, Any]:
        row = self._concepts.get(str(concept_id or "").strip())
        if row is None:
            raise KeyError("concept not found")
        if not str(summary or "").strip():
            raise ValueError("summary is required")
        self._append_interaction(
            row=row,
            link_id=str(link_id or "").strip(),
            interaction={
                "summary": str(summary or "").strip(),
                "pattern_hint": str(pattern_hint or "").strip(),
                "outcome": str(outcome or "").strip(),
                "extracted_facts": [str(x).strip() for x in (extracted_facts or []) if str(x).strip()],
            },
        )
        row["updated_at"] = _now_ts()
        self._save()
        return self._public_row(row)

    def run_link_learning(
        self,
        *,
        concept_id: str,
        link_id: str,
        research_engine: Any,
        max_items: int = 6,
        run_adapters: bool = True,
    ) -> dict[str, Any]:
        row = self._concepts.get(str(concept_id or "").strip())
        if row is None:
            raise KeyError("concept not found")
        lid = str(link_id or "").strip()
        link = None
        for item in list(row.get("reference_links", [])):
            if not isinstance(item, dict):
                continue
            if str(item.get("link_id", "")).strip() == lid:
                link = item
                break
        if link is None:
            raise KeyError("link not found")

        topic = str(row.get("topic", "")).strip()
        url = str(link.get("url", "")).strip()
        title = str(link.get("title", "")).strip()
        notes = str(link.get("notes", "")).strip()
        query_text = " ".join(x for x in [topic, title, url, notes] if x).strip()
        if not query_text:
            query_text = topic or title or url

        adapter_result: dict[str, Any] = {}
        if run_adapters and research_engine is not None and hasattr(research_engine, "run_adapters"):
            try:
                adapter_result = research_engine.run_adapters(topic=query_text, max_items_per_adapter=max(1, int(max_items)))
            except Exception as exc:  # noqa: BLE001
                adapter_result = {"error": str(exc)}

        query_result: dict[str, Any] = {"results": []}
        if research_engine is not None and hasattr(research_engine, "query"):
            try:
                query_result = research_engine.query(
                    query_text,
                    max_results=max(1, int(max_items)),
                    freshness_days=90,
                    min_trust=0.0,
                )
            except Exception as exc:  # noqa: BLE001
                query_result = {"error": str(exc), "results": []}

        facts = self._facts_from_query(query_result)
        existing = list(row.get("web_facts", []))
        dedupe: set[str] = set()
        merged: list[dict[str, Any]] = []
        for item in facts + existing:
            key = f"{item.get('url', '')}|{item.get('title', '')}".strip().lower()
            if not key or key in dedupe:
                continue
            dedupe.add(key)
            merged.append(item)
            if len(merged) >= 40:
                break
        row["web_facts"] = merged
        link["updated_at"] = _now_ts()
        domain = str(urlparse(url).netloc or "").strip().lower()
        extracted = []
        for f in facts[:4]:
            line = str(f.get("snippet", "")).strip() or str(f.get("title", "")).strip()
            if line:
                extracted.append(line[:200])
        self._append_interaction(
            row=row,
            link_id=lid,
            interaction={
                "summary": f"Auto learn run for {title or topic or 'concept'} from {url}".strip(),
                "pattern_hint": f"domain:{domain}" if domain else "",
                "outcome": "useful" if facts else "partial",
                "extracted_facts": extracted,
            },
        )
        learning_runs = list(row.get("learning_runs", []))
        query_count = len(list(query_result.get("results", []) if isinstance(query_result, dict) else []))
        learning_runs.insert(
            0,
            {
                "run_id": f"wlr_{uuid.uuid4().hex[:12]}",
                "mode": "browser_use_auto",
                "link_id": lid,
                "link_title": title,
                "link_url": url,
                "query": query_text,
                "added_facts": len(facts),
                "query_result_count": query_count,
                "adapter_inserted_total": int(adapter_result.get("inserted_total", 0) or 0)
                if isinstance(adapter_result, dict)
                else 0,
                "outcome": "useful" if facts else "partial",
                "recorded_at": _now_ts(),
            },
        )
        row["learning_runs"] = learning_runs[:300]
        row["updated_at"] = _now_ts()
        self._save()
        return {
            "concept": self._public_row(row),
            "link_id": lid,
            "query": query_text,
            "added_facts": len(facts),
            "query_result_count": query_count,
            "adapter_result": adapter_result,
        }

    def enrich_concept_from_web(
        self,
        *,
        concept_id: str,
        research_engine: Any,
        max_items: int = 5,
        run_adapters: bool = True,
    ) -> dict[str, Any]:
        row = self._concepts.get(str(concept_id or "").strip())
        if row is None:
            raise KeyError("concept not found")
        topic = str(row.get("topic", "")).strip()
        adapter_result: dict[str, Any] = {}
        if run_adapters and research_engine is not None and hasattr(research_engine, "run_adapters"):
            try:
                adapter_result = research_engine.run_adapters(topic=topic, max_items_per_adapter=max(1, int(max_items)))
            except Exception as exc:  # noqa: BLE001
                adapter_result = {"error": str(exc)}
        query_result: dict[str, Any] = {"results": []}
        if research_engine is not None and hasattr(research_engine, "query"):
            try:
                query_result = research_engine.query(
                    topic=topic,
                    query=topic,
                    top_k=max(1, int(max_items)),
                )
            except Exception as exc:  # noqa: BLE001
                query_result = {"error": str(exc), "results": []}
        facts = self._facts_from_query(query_result)
        link_facts = self._facts_from_reference_links(row)
        existing = list(row.get("web_facts", []))
        dedupe: set[str] = set()
        merged: list[dict[str, Any]] = []
        for item in link_facts + facts + existing:
            key = f"{item.get('url', '')}|{item.get('title', '')}".strip().lower()
            if not key or key in dedupe:
                continue
            dedupe.add(key)
            merged.append(item)
            if len(merged) >= 40:
                break
        row["web_facts"] = merged
        row["updated_at"] = _now_ts()
        self._save()
        return {
            "concept": self._public_row(row),
            "added_facts": len(facts),
            "adapter_result": adapter_result,
            "query_result_count": len(list(query_result.get("results", []) if isinstance(query_result, dict) else [])),
        }

    def enrich_detection_labels(
        self,
        *,
        labels: list[str],
        research_engine: Any,
        max_items_per_label: int = 2,
        allow_web: bool = True,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        now = _now_ts()
        for raw in labels:
            label = str(raw or "").strip().lower()
            if not label:
                continue
            item: dict[str, Any] = {"label": label, "facts": [], "concept_matches": []}
            concept = self._concept_by_topic(label)
            if concept:
                item["concept_matches"] = [self._public_row(concept)]
                item["facts"] = list(concept.get("web_facts", []))[:max(1, int(max_items_per_label))]
                item["pattern_hints"] = [
                    str(log.get("pattern_hint", "")).strip()
                    for log in list(concept.get("interaction_logs", []))
                    if str(log.get("pattern_hint", "")).strip()
                ][:4]
            should_web = allow_web and (now - float(self._label_web_cooldown.get(label, 0.0)) > 600.0)
            if should_web and research_engine is not None:
                try:
                    if hasattr(research_engine, "run_adapters"):
                        research_engine.run_adapters(topic=label, max_items_per_adapter=max(1, int(max_items_per_label)))
                    if hasattr(research_engine, "query"):
                        q = research_engine.query(topic=label, query=label, top_k=max(1, int(max_items_per_label)))
                        facts = self._facts_from_query(q)[: max(1, int(max_items_per_label))]
                        if facts:
                            item["facts"] = facts
                    self._label_web_cooldown[label] = now
                except Exception:  # noqa: BLE001
                    pass
            out.append(item)
        return out

    def _concept_by_topic(self, topic: str) -> dict[str, Any] | None:
        cid = self._topic_index.get(str(topic or "").strip().lower())
        if not cid:
            return None
        return self._concepts.get(cid)

    @staticmethod
    def _append_interaction(*, row: dict[str, Any], link_id: str, interaction: dict[str, Any]) -> None:
        logs = list(row.get("interaction_logs", []))
        logs.insert(
            0,
            {
                "log_id": f"wil_{uuid.uuid4().hex[:12]}",
                "link_id": str(link_id or "").strip(),
                "summary": str(interaction.get("summary", "")).strip()[:400],
                "pattern_hint": str(interaction.get("pattern_hint", "")).strip()[:240],
                "outcome": str(interaction.get("outcome", "")).strip()[:120],
                "extracted_facts": [
                    str(x).strip()[:200]
                    for x in list(interaction.get("extracted_facts", []) or [])
                    if str(x).strip()
                ][:10],
                "recorded_at": _now_ts(),
            },
        )
        row["interaction_logs"] = logs[:250]

    @staticmethod
    def _facts_from_query(query_result: dict[str, Any]) -> list[dict[str, Any]]:
        rows = list(query_result.get("results", []) if isinstance(query_result, dict) else [])
        out: list[dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            title = str(r.get("title", "")).strip()
            url = str(r.get("url", "")).strip()
            snippet = str(r.get("summary", "") or r.get("content", "") or "").strip()
            score = float(r.get("score", 0.0) or 0.0)
            if not title and not snippet:
                continue
            out.append(
                {
                    "title": title[:180],
                    "url": url,
                    "snippet": snippet[:300],
                    "score": round(score, 4),
                    "source_type": str(r.get("source_type", "web")),
                    "captured_at": _now_ts(),
                }
            )
        return out

    @staticmethod
    def _facts_from_reference_links(row: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for link in list(row.get("reference_links", [])):
            if not isinstance(link, dict):
                continue
            url = str(link.get("url", "")).strip()
            if not url:
                continue
            title = str(link.get("title", "")).strip() or url
            notes = str(link.get("notes", "")).strip()
            out.append(
                {
                    "title": title[:180],
                    "url": url,
                    "snippet": notes[:300],
                    "score": 1.0,
                    "source_type": "reference_link",
                    "captured_at": float(link.get("updated_at", _now_ts()) or _now_ts()),
                }
            )
        return out

    @staticmethod
    def _public_row(row: dict[str, Any]) -> dict[str, Any]:
        refs = [r for r in list(row.get("reference_links", [])) if isinstance(r, dict)]
        logs = [r for r in list(row.get("interaction_logs", [])) if isinstance(r, dict)]
        runs = [r for r in list(row.get("learning_runs", [])) if isinstance(r, dict)]
        return {
            "concept_id": row.get("concept_id"),
            "topic": row.get("topic"),
            "tags": list(row.get("tags", [])),
            "notes_count": len(list(row.get("notes", []))),
            "latest_note": (list(row.get("notes", [])) or [""])[0],
            "notes": list(row.get("notes", []))[:12],
            "detections_count": len(list(row.get("detections", []))),
            "detections": list(row.get("detections", []))[:24],
            "web_facts_count": len(list(row.get("web_facts", []))),
            "web_facts": list(row.get("web_facts", []))[:6],
            "reference_links_count": len(refs),
            "reference_links": refs[:24],
            "interaction_logs_count": len(logs),
            "interaction_logs": logs[:30],
            "learning_runs_count": len(runs),
            "learning_runs": runs[:40],
            "metadata": dict(row.get("metadata", {}) or {}),
            "updated_at": float(row.get("updated_at", 0.0)),
            "created_at": float(row.get("created_at", 0.0)),
        }

    def _load(self) -> None:
        self._concepts = {}
        self._topic_index = {}
        try:
            if not self._state_path.exists():
                return
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            rows = list(payload.get("concepts", []))
            for r in rows:
                if not isinstance(r, dict):
                    continue
                cid = str(r.get("concept_id", "")).strip()
                topic_norm = str(r.get("topic_normalized", "")).strip().lower()
                if not cid or not topic_norm:
                    continue
                if not isinstance(r.get("reference_links"), list):
                    r["reference_links"] = []
                if not isinstance(r.get("interaction_logs"), list):
                    r["interaction_logs"] = []
                if not isinstance(r.get("learning_runs"), list):
                    r["learning_runs"] = []
                self._concepts[cid] = r
                self._topic_index[topic_norm] = cid
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load world knowledge state: %s", exc)

    def _save(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "saved_at": _now_ts(),
                "concepts": list(self._concepts.values()),
            }
            self._state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save world knowledge state: %s", exc)


__all__ = ["WorldKnowledgeService"]
