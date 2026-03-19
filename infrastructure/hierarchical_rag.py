"""
Hierarchical tree-based RAG indexing and retrieval.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from infrastructure.multimodal_embedding import MultiModalEmbeddingEngine


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
    embedding: List[float] = field(default_factory=list)
    multimodal_embedding: List[float] = field(default_factory=list)
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
            "embedding_dim": len(self.embedding),
            "multimodal_embedding_dim": len(self.multimodal_embedding),
            "created_at": self.created_at,
        }


class HierarchicalRAGIndex:
    """In-memory tree index that preserves document structure and context."""

    _VISUAL_QUERY_HINTS = {
        "image",
        "photo",
        "picture",
        "screenshot",
        "diagram",
        "chart",
        "visual",
        "design",
        "ui",
        "ux",
    }

    def __init__(
        self,
        *,
        embedding_backend: str = "local_deterministic",
        embedding_dim: int = 64,
        multimodal_embedding_backend: str = "mlx_clip",
        multimodal_embedding_dim: Optional[int] = None,
        fusion_text_weight: float = 0.65,
        fusion_multimodal_weight: float = 0.35,
        reranker_enabled: bool = True,
        reranker_top_k: int = 24,
    ) -> None:
        self._nodes: Dict[str, HierarchyNode] = {}
        self._roots_by_source: Dict[str, List[str]] = {}
        self._source_nodes: Dict[str, List[str]] = {}
        self._text_embedder = MultiModalEmbeddingEngine(backend=embedding_backend, dim=embedding_dim)
        self._mm_embedder = MultiModalEmbeddingEngine(
            backend=multimodal_embedding_backend,
            dim=embedding_dim if multimodal_embedding_dim is None else int(multimodal_embedding_dim),
        )
        self._fusion_text_weight = max(0.0, float(fusion_text_weight))
        self._fusion_mm_weight = max(0.0, float(fusion_multimodal_weight))
        self._reranker_enabled = bool(reranker_enabled)
        self._reranker_top_k = max(1, int(reranker_top_k))

    def set_embedding_backend(self, *, backend: str, dim: Optional[int] = None) -> None:
        self._text_embedder = MultiModalEmbeddingEngine(
            backend=backend,
            dim=self._text_embedder.dim if dim is None else int(dim),
        )

    def set_multimodal_embedding_backend(self, *, backend: str, dim: Optional[int] = None) -> None:
        self._mm_embedder = MultiModalEmbeddingEngine(
            backend=backend,
            dim=self._mm_embedder.dim if dim is None else int(dim),
        )

    def get_embedding_config(self) -> Dict[str, Any]:
        return {
            "backend": self._text_embedder.backend,
            "backend_requested": self._text_embedder.backend_requested,
            "dim": self._text_embedder.dim,
            "multimodal_backend": self._mm_embedder.backend,
            "multimodal_backend_requested": self._mm_embedder.backend_requested,
            "multimodal_dim": self._mm_embedder.dim,
            "fusion_text_weight": self._fusion_text_weight,
            "fusion_multimodal_weight": self._fusion_mm_weight,
            "reranker_enabled": self._reranker_enabled,
            "reranker_top_k": self._reranker_top_k,
        }

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
            metadata={"modality": "text", **dict(metadata or {})},
            embedding=self._text_embedder.embed_text(f"{doc_title}\n{doc_content[:4000]}"),
            multimodal_embedding=self._mm_embedder.embed_text(f"{doc_title}\n{doc_content[:4000]}"),
        )
        self._nodes[root.node_id] = root
        source_nodes.append(root.node_id)
        roots.append(root.node_id)

        meta = dict(metadata or {})
        image_bytes = MultiModalEmbeddingEngine.image_bytes_from_metadata(meta)
        if image_bytes:
            image_title = str(meta.get("image_title", "")).strip() or f"{doc_title} image"
            image_caption = str(meta.get("image_caption", "")).strip()
            image_node = HierarchyNode(
                node_id=f"rag-{uuid.uuid4().hex}",
                source_id=source_id,
                level=1,
                title=image_title,
                content=image_caption,
                parent_id=root.node_id,
                metadata={"modality": "image", **meta},
                embedding=self._text_embedder.embed_text(f"{image_title}\n{image_caption}"),
                multimodal_embedding=self._mm_embedder.embed_image(image_bytes),
            )
            self._nodes[image_node.node_id] = image_node
            source_nodes.append(image_node.node_id)
            root.children_ids.append(image_node.node_id)

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
                metadata={"modality": "text", "section_index": idx, **dict(metadata or {})},
                embedding=self._text_embedder.embed_text(f"{section['title']}\n{section['content']}"),
                multimodal_embedding=self._mm_embedder.embed_text(
                    f"{section['title']}\n{section['content']}"
                ),
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
                    metadata={"modality": "text", "chunk_index": cidx, **dict(metadata or {})},
                    embedding=self._text_embedder.embed_text(chunk),
                    multimodal_embedding=self._mm_embedder.embed_text(chunk),
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
        use_reranker: Optional[bool] = None,
        reranker_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        q = str(query or "").strip()
        if not q:
            return {"count": 0, "nodes": [], "contexts": []}
        q_text_embedding = self._text_embedder.embed_text(q)
        q_mm_embedding = self._mm_embedder.embed_text(q)
        visual_hint = self._has_visual_hint(q)
        scored: List[tuple[float, HierarchyNode]] = []
        score_details: Dict[str, Dict[str, float]] = {}
        for node in self._nodes.values():
            body = f"{node.title} {node.content}"
            lexical = _token_overlap_score(q, body)
            text_sem = self._text_embedder.cosine_similarity(q_text_embedding, node.embedding)
            mm_vec = node.multimodal_embedding or node.embedding
            mm_sem = self._mm_embedder.cosine_similarity(q_mm_embedding, mm_vec)
            fused_sem = (text_sem * self._fusion_text_weight) + (mm_sem * self._fusion_mm_weight)
            score = (lexical * 0.55) + (fused_sem * 0.35)
            # Prefer section/chunk matches for retrieval granularity.
            if node.level == 1:
                score += 0.05
            elif node.level == 2:
                score += 0.1
            if visual_hint and node.metadata.get("modality") == "image":
                score += 0.05
            if score <= 0.0:
                continue
            score_details[node.node_id] = {
                "lexical_score": round(lexical, 4),
                "semantic_text_score": round(text_sem, 4),
                "semantic_multimodal_score": round(mm_sem, 4),
                "fused_semantic_score": round(fused_sem, 4),
            }
            scored.append((round(score, 4), node))
        ranked = sorted(scored, key=lambda t: t[0], reverse=True)[: max(1, min(64, int(max_nodes)))]
        rerank_active = self._reranker_enabled if use_reranker is None else bool(use_reranker)
        rerank_k = self._reranker_top_k if reranker_top_k is None else max(1, int(reranker_top_k))
        if rerank_active and ranked:
            ranked = self._rerank(query=q, ranked=ranked, top_k=rerank_k)
        nodes = []
        contexts = []
        for score, node in ranked:
            details = score_details.get(node.node_id, {})
            rerank_score = self._rerank_pair_score(q, f"{node.title}\n{node.content}")
            nodes.append(
                {
                    **node.to_dict(),
                    "score": score,
                    "rerank_score": round(rerank_score, 4),
                    **details,
                }
            )
            if expand_neighbors:
                contexts.append(self._build_context(node))
        return {
            "count": len(nodes),
            "nodes": nodes,
            "contexts": contexts,
            "strategy": {
                "fusion_text_weight": self._fusion_text_weight,
                "fusion_multimodal_weight": self._fusion_mm_weight,
                "reranker_enabled": rerank_active,
                "reranker_top_k": rerank_k,
            },
        }

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

    @staticmethod
    def _has_visual_hint(query: str) -> bool:
        low = _canon(query)
        return any(tok in low for tok in HierarchicalRAGIndex._VISUAL_QUERY_HINTS)

    @staticmethod
    def _rerank_pair_score(query: str, text: str) -> float:
        q = _canon(query)
        body = _canon(text)
        if not q or not body:
            return 0.0
        token_score = _token_overlap_score(q, body)
        phrase_bonus = 0.15 if q in body else 0.0
        # Encourage title/early passage matches.
        early = body[:320]
        early_bonus = 0.1 if any(tok in early for tok in q.split()) else 0.0
        return min(1.0, token_score * 0.75 + phrase_bonus + early_bonus)

    def _rerank(
        self,
        *,
        query: str,
        ranked: List[tuple[float, HierarchyNode]],
        top_k: int,
    ) -> List[tuple[float, HierarchyNode]]:
        k = max(1, min(len(ranked), int(top_k)))
        head = ranked[:k]
        tail = ranked[k:]
        rescored: List[tuple[float, HierarchyNode]] = []
        for base_score, node in head:
            pair = self._rerank_pair_score(query, f"{node.title}\n{node.content}")
            final = (base_score * 0.7) + (pair * 0.3)
            rescored.append((round(final, 4), node))
        rescored.sort(key=lambda x: x[0], reverse=True)
        return rescored + tail

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
