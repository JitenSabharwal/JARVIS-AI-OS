"""
Hierarchical tree-based RAG indexing and retrieval.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _canon(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _token_overlap_score(query: str, text: str) -> float:
    q_tokens = [t for t in _canon(query).split() if t]
    if not q_tokens:
        return 0.0
    body = _canon(text)
    hits = sum(1 for token in q_tokens if token in body)
    return hits / max(1, len(q_tokens))


@dataclass
class HierarchyNode:
    node_id: str
    source_id: str
    level: int
    title: str
    content: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "source_id": self.source_id,
            "level": self.level,
            "title": self.title,
            "content": self.content,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }


class HierarchicalRAGIndex:
    """In-memory tree index that preserves document structure and context."""

    def __init__(self) -> None:
        self._nodes: Dict[str, HierarchyNode] = {}
        self._roots_by_source: Dict[str, List[str]] = {}
        self._source_nodes: Dict[str, List[str]] = {}

    def index_document(
        self,
        *,
        source_id: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source_id = str(source_id).strip()
        if not source_id:
            raise ValueError("source_id is required")
        doc_title = str(title).strip() or "document"
        doc_content = str(content or "")
        # Replace prior source nodes if source is re-ingested.
        self._remove_source(source_id)

        source_nodes: List[str] = []
        roots: List[str] = []

        root = HierarchyNode(
            node_id=f"rag-{uuid.uuid4().hex}",
            source_id=source_id,
            level=0,
            title=doc_title,
            content=doc_content[:4000],
            parent_id=None,
            metadata=dict(metadata or {}),
        )
        self._nodes[root.node_id] = root
        source_nodes.append(root.node_id)
        roots.append(root.node_id)

        sections = self._split_sections(doc_content)
        section_nodes: List[str] = []
        for idx, section in enumerate(sections):
            section_node = HierarchyNode(
                node_id=f"rag-{uuid.uuid4().hex}",
                source_id=source_id,
                level=1,
                title=section["title"] or f"Section {idx + 1}",
                content=section["content"],
                parent_id=root.node_id,
                metadata={"section_index": idx, **dict(metadata or {})},
            )
            self._nodes[section_node.node_id] = section_node
            source_nodes.append(section_node.node_id)
            section_nodes.append(section_node.node_id)
            root.children_ids.append(section_node.node_id)
            for cidx, chunk in enumerate(self._split_chunks(section["content"])):
                chunk_node = HierarchyNode(
                    node_id=f"rag-{uuid.uuid4().hex}",
                    source_id=source_id,
                    level=2,
                    title=f"{section_node.title} / chunk {cidx + 1}",
                    content=chunk,
                    parent_id=section_node.node_id,
                    metadata={"chunk_index": cidx, **dict(metadata or {})},
                )
                self._nodes[chunk_node.node_id] = chunk_node
                source_nodes.append(chunk_node.node_id)
                section_node.children_ids.append(chunk_node.node_id)

        self._roots_by_source[source_id] = roots
        self._source_nodes[source_id] = source_nodes
        return {
            "source_id": source_id,
            "root_count": len(roots),
            "section_count": len(section_nodes),
            "node_count": len(source_nodes),
        }

    def query(
        self,
        *,
        query: str,
        max_nodes: int = 8,
        expand_neighbors: bool = True,
    ) -> Dict[str, Any]:
        q = str(query or "").strip()
        if not q:
            return {"count": 0, "nodes": [], "contexts": []}
        scored: List[tuple[float, HierarchyNode]] = []
        for node in self._nodes.values():
            body = f"{node.title} {node.content}"
            score = _token_overlap_score(q, body)
            # Prefer section/chunk matches for retrieval granularity.
            if node.level == 1:
                score += 0.05
            elif node.level == 2:
                score += 0.1
            if score <= 0.0:
                continue
            scored.append((round(score, 4), node))
        ranked = sorted(scored, key=lambda t: t[0], reverse=True)[: max(1, min(64, int(max_nodes)))]
        nodes = []
        contexts = []
        for score, node in ranked:
            nodes.append({**node.to_dict(), "score": score})
            if expand_neighbors:
                contexts.append(self._build_context(node))
        return {"count": len(nodes), "nodes": nodes, "contexts": contexts}

    def get_source_tree(self, source_id: str) -> Dict[str, Any]:
        node_ids = list(self._source_nodes.get(source_id, []))
        nodes = [self._nodes[nid].to_dict() for nid in node_ids if nid in self._nodes]
        return {"source_id": source_id, "count": len(nodes), "nodes": nodes}

    def _build_context(self, node: HierarchyNode) -> Dict[str, Any]:
        parent = self._nodes.get(node.parent_id) if node.parent_id else None
        siblings: List[str] = []
        if parent:
            siblings = [sid for sid in parent.children_ids if sid != node.node_id][:2]
        sibling_nodes = [self._nodes[sid].to_dict() for sid in siblings if sid in self._nodes]
        return {
            "focus_node_id": node.node_id,
            "parent": parent.to_dict() if parent else None,
            "siblings": sibling_nodes,
            "children": [self._nodes[cid].to_dict() for cid in node.children_ids[:3] if cid in self._nodes],
        }

    def _remove_source(self, source_id: str) -> None:
        old = list(self._source_nodes.get(source_id, []))
        for node_id in old:
            self._nodes.pop(node_id, None)
        self._source_nodes.pop(source_id, None)
        self._roots_by_source.pop(source_id, None)

    @staticmethod
    def _split_sections(content: str) -> List[Dict[str, str]]:
        text = str(content or "").strip()
        if not text:
            return [{"title": "Overview", "content": ""}]
        sections: List[Dict[str, str]] = []
        lines = text.splitlines()
        current_title = "Overview"
        current_lines: List[str] = []
        heading_re = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
        for line in lines:
            m = heading_re.match(line)
            if m:
                if current_lines:
                    sections.append({"title": current_title, "content": "\n".join(current_lines).strip()})
                    current_lines = []
                current_title = m.group(2).strip()
                continue
            current_lines.append(line)
        if current_lines:
            sections.append({"title": current_title, "content": "\n".join(current_lines).strip()})
        return sections or [{"title": "Overview", "content": text}]

    @staticmethod
    def _split_chunks(content: str, *, max_chars: int = 600) -> List[str]:
        text = str(content or "").strip()
        if not text:
            return [""]
        parts = re.split(r"\n\s*\n", text)
        chunks: List[str] = []
        buf = ""
        for part in parts:
            piece = part.strip()
            if not piece:
                continue
            if not buf:
                buf = piece
                continue
            if len(buf) + 2 + len(piece) <= max_chars:
                buf = f"{buf}\n\n{piece}"
            else:
                chunks.append(buf)
                buf = piece
        if buf:
            chunks.append(buf)
        return chunks or [text[:max_chars]]


__all__ = ["HierarchyNode", "HierarchicalRAGIndex"]
