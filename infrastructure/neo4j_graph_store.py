"""
Optional Neo4j-backed graph store for research/source relationships.
"""

from __future__ import annotations

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

    def _run(self, query: str, return_records: bool = False, **params: Any) -> Any:
        if self._driver is None:
            return [] if return_records else None
        with self._driver.session(database=self._database) as session:
            result = session.run(query, **params)
            if return_records:
                return [dict(r) for r in result]
            return None


__all__ = ["Neo4jGraphStore"]
