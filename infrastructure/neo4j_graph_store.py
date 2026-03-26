"""
Optional Neo4j-backed graph store for research/source relationships.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    from neo4j import GraphDatabase  # type: ignore

    _NEO4J_AVAILABLE = True
except Exception:  # noqa: BLE001
    GraphDatabase = None  # type: ignore[assignment]
    _NEO4J_AVAILABLE = False


class Neo4jGraphStore:
    """Best-effort Neo4j adapter; no-op when Neo4j driver is unavailable."""

    def __init__(
        self,
        *,
        uri: str = "",
        username: str = "",
        password: str = "",
        database: str = "neo4j",
        enabled: bool = False,
    ) -> None:
        self.enabled = bool(enabled and _NEO4J_AVAILABLE and uri and username)
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._driver = None
        if self.enabled and GraphDatabase is not None:
            self._driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def health(self) -> Dict[str, Any]:
        if not self.enabled or self._driver is None:
            return {"enabled": False, "healthy": False, "reason": "neo4j_disabled_or_missing_driver"}
        try:
            with self._driver.session(database=self._database) as session:
                row = session.run("RETURN 1 AS ok").single()
            return {"enabled": True, "healthy": bool(row and row.get("ok") == 1)}
        except Exception as exc:  # noqa: BLE001
            return {"enabled": True, "healthy": False, "reason": str(exc)}

    def upsert_source(self, *, source_id: str, title: str, url: str, topic: str, source_type: str) -> None:
        if not self.enabled or self._driver is None:
            return
        query = (
            "MERGE (s:Source {source_id: $source_id}) "
            "SET s.title = $title, s.url = $url, s.topic = $topic, s.source_type = $source_type"
        )
        self._run(query, source_id=source_id, title=title, url=url, topic=topic, source_type=source_type)

    def upsert_node(
        self,
        *,
        node_id: str,
        source_id: str,
        level: int,
        title: str,
        content: str,
        parent_id: Optional[str] = None,
    ) -> None:
        if not self.enabled or self._driver is None:
            return
        query = (
            "MERGE (n:DocNode {node_id: $node_id}) "
            "SET n.source_id = $source_id, n.level = $level, n.title = $title, n.content = $content "
            "WITH n "
            "MERGE (s:Source {source_id: $source_id}) "
            "MERGE (s)-[:HAS_NODE]->(n)"
        )
        self._run(
            query,
            node_id=node_id,
            source_id=source_id,
            level=int(level),
            title=title,
            content=content[:2000],
        )
        if parent_id:
            self.link_nodes(parent_id=parent_id, child_id=node_id, relation="PARENT_OF")

    def link_nodes(self, *, parent_id: str, child_id: str, relation: str = "RELATED_TO") -> None:
        if not self.enabled or self._driver is None:
            return
        rel = "".join(ch for ch in relation.upper() if ch.isalnum() or ch == "_")
        if not rel:
            rel = "RELATED_TO"
        query = (
            f"MERGE (a:DocNode {{node_id: $parent_id}}) "
            f"MERGE (b:DocNode {{node_id: $child_id}}) "
            f"MERGE (a)-[:{rel}]->(b)"
        )
        self._run(query, parent_id=parent_id, child_id=child_id)

    def query_related(self, *, source_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        if not self.enabled or self._driver is None:
            return []
        query = (
            "MATCH (s:Source {source_id: $source_id})-[:HAS_NODE]->(n:DocNode) "
            "OPTIONAL MATCH (n)-[r]->(m:DocNode) "
            "RETURN n.node_id AS node_id, n.level AS level, n.title AS title, "
            "type(r) AS relation, m.node_id AS related_node_id "
            "LIMIT $limit"
        )
        records = self._run(query, source_id=source_id, limit=max(1, int(limit)), return_records=True)
        out: List[Dict[str, Any]] = []
        for rec in records:
            out.append(
                {
                    "node_id": rec.get("node_id"),
                    "level": rec.get("level"),
                    "title": rec.get("title"),
                    "relation": rec.get("relation"),
                    "related_node_id": rec.get("related_node_id"),
                }
            )
        return out

    def upsert_profile(
        self,
        *,
        profile_id: str,
        concept_id: str,
        display_name: str,
        person_id: str = "",
    ) -> None:
        if not self.enabled or self._driver is None:
            return
        query = (
            "MERGE (p:Profile {profile_id: $profile_id}) "
            "SET p.concept_id = $concept_id, p.display_name = $display_name, p.person_id = $person_id"
        )
        self._run(
            query,
            profile_id=str(profile_id or "").strip(),
            concept_id=str(concept_id or "").strip(),
            display_name=str(display_name or "").strip(),
            person_id=str(person_id or "").strip(),
        )

    def upsert_profile_entity(
        self,
        *,
        profile_id: str,
        entity_type: str,
        value: str,
        relation: str,
        confidence: float = 0.8,
        source: str = "profile_enrichment",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or self._driver is None:
            return
        clean_value = str(value or "").strip()
        if not clean_value:
            return
        label = self._safe_label(entity_type) or "Entity"
        rel = self._safe_rel(relation) or "RELATED_TO"
        query = (
            "MERGE (p:Profile {profile_id: $profile_id}) "
            f"MERGE (e:{label} {{value: $value}}) "
            "SET e.entity_type = $entity_type, e.updated_at = timestamp(), "
            "e.metadata_json = $metadata_json, e.metadata_kind = $metadata_kind "
            f"MERGE (p)-[r:{rel}]->(e) "
            "SET r.confidence = $confidence, r.source = $source, r.updated_at = timestamp()"
        )
        md = dict(metadata or {})
        md_kind = str(md.get("kind", "")).strip()
        self._run(
            query,
            profile_id=str(profile_id or "").strip(),
            value=clean_value,
            entity_type=str(entity_type or "").strip().lower(),
            confidence=max(0.0, min(1.0, float(confidence))),
            source=str(source or "").strip() or "profile_enrichment",
            metadata_json=json.dumps(md, ensure_ascii=True, sort_keys=True),
            metadata_kind=md_kind,
        )

    def upsert_profile_source(
        self,
        *,
        profile_id: str,
        url: str,
        title: str = "",
        source_type: str = "reference",
    ) -> None:
        if not self.enabled or self._driver is None:
            return
        clean_url = str(url or "").strip()
        if not clean_url:
            return
        query = (
            "MERGE (p:Profile {profile_id: $profile_id}) "
            "MERGE (s:ProfileSource {url: $url}) "
            "SET s.title = $title, s.source_type = $source_type, s.updated_at = timestamp() "
            "MERGE (p)-[r:EVIDENCED_BY]->(s) "
            "SET r.updated_at = timestamp()"
        )
        self._run(
            query,
            profile_id=str(profile_id or "").strip(),
            url=clean_url,
            title=str(title or "").strip()[:220],
            source_type=str(source_type or "").strip() or "reference",
        )

    def clear_profile_entities(self, *, profile_id: str) -> None:
        """Remove non-source relations for a profile so sync can rebuild cleanly."""
        if not self.enabled or self._driver is None:
            return
        pid = str(profile_id or "").strip()
        if not pid:
            return
        # Keep EVIDENCED_BY links; rebuild semantic entities each sync.
        self._run(
            "MATCH (p:Profile {profile_id: $profile_id})-[r]->(n) "
            "WHERE type(r) <> 'EVIDENCED_BY' "
            "DELETE r",
            profile_id=pid,
        )
        self._run(
            "MATCH (p:Profile {profile_id: $profile_id})-[r:EVIDENCED_BY]->(s:ProfileSource) "
            "WHERE s.url IS NULL OR trim(toLower(s.url)) IN ['', 'none', 'null'] "
            "DELETE r",
            profile_id=pid,
        )
        # Best-effort orphan cleanup for profile entity nodes.
        self._run(
            "MATCH (n) "
            "WHERE n.entity_type IS NOT NULL "
            "AND NOT (()-[]->(n)) "
            "DELETE n"
        )
        self._run(
            "MATCH (s:ProfileSource) "
            "WHERE (s.url IS NULL OR trim(toLower(s.url)) IN ['', 'none', 'null']) "
            "AND NOT (()-[]->(s)) "
            "DELETE s"
        )

    def get_profile_graph(self, *, profile_id: str, limit: int = 120) -> Dict[str, Any]:
        if not self.enabled or self._driver is None:
            return {"enabled": False, "profile_id": profile_id, "nodes": [], "edges": []}
        lim = max(1, min(500, int(limit)))
        query = (
            "MATCH (p:Profile {profile_id: $profile_id}) "
            "OPTIONAL MATCH (p)-[r]->(n) "
            "RETURN p.profile_id AS profile_id, p.display_name AS display_name, labels(n) AS labels, "
            "n.value AS value, n.title AS title, n.url AS url, n.source_type AS source_type, "
            "type(r) AS relation, r.confidence AS confidence, r.source AS source "
            "LIMIT $limit"
        )
        records = self._run(query, profile_id=str(profile_id or "").strip(), limit=lim, return_records=True)
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        display_name = ""
        for rec in records:
            display_name = str(rec.get("display_name", "")).strip() or display_name
            labels = rec.get("labels") or []
            label = ""
            if isinstance(labels, list) and labels:
                label = str(labels[0])
            value = str(rec.get("value", "")).strip()
            title = str(rec.get("title", "")).strip()
            url = str(rec.get("url", "")).strip()
            source_type = str(rec.get("source_type", "")).strip()
            relation = str(rec.get("relation", "")).strip()
            if value or title or url:
                node_key = value or title or url
                if node_key not in nodes:
                    nodes[node_key] = {
                        "id": node_key,
                        "label": label or "Entity",
                        "value": value or title or url,
                        "url": url,
                        "source_type": source_type,
                    }
                edges.append(
                    {
                        "from": str(profile_id or "").strip(),
                        "to": node_key,
                        "relation": relation,
                        "confidence": float(rec.get("confidence", 0.0) or 0.0),
                        "source": str(rec.get("source", "")).strip(),
                    }
                )
        root = {
            "id": str(profile_id or "").strip(),
            "label": "Profile",
            "value": display_name or str(profile_id or "").strip(),
        }
        return {
            "enabled": True,
            "profile_id": str(profile_id or "").strip(),
            "display_name": display_name,
            "nodes": [root] + list(nodes.values()),
            "edges": edges,
        }

    @staticmethod
    def _safe_label(value: str) -> str:
        clean = "".join(ch for ch in str(value or "") if ch.isalnum())
        if not clean:
            return ""
        return clean[0].upper() + clean[1:]

    @staticmethod
    def _safe_rel(value: str) -> str:
        clean = "".join(ch for ch in str(value or "").upper() if ch.isalnum() or ch == "_")
        if not clean:
            return ""
        return clean

    def _run(self, query: str, return_records: bool = False, **params: Any) -> Any:
        if self._driver is None:
            return [] if return_records else None
        with self._driver.session(database=self._database) as session:
            result = session.run(query, **params)
            if return_records:
                return [dict(r) for r in result]
            return None


__all__ = ["Neo4jGraphStore"]
