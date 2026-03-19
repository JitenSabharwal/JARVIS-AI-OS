from infrastructure.hierarchical_rag import HierarchicalRAGIndex


def test_hierarchical_rag_indexes_structure_and_queries_context() -> None:
    idx = HierarchicalRAGIndex()
    ingest = idx.index_document(
        source_id="src-1",
        title="RAG Design",
        content=(
            "# Intro\n"
            "This document explains retrieval.\n\n"
            "# Architecture\n"
            "Hierarchy has sections and chunks.\n\n"
            "Chunk details include neighbors and parent context."
        ),
        metadata={"topic": "rag"},
    )
    assert ingest["node_count"] >= 3

    q = idx.query(query="hierarchy chunks", max_nodes=5, expand_neighbors=True)
    assert q["count"] >= 1
    assert q["nodes"][0]["level"] in {1, 2}
    assert q["contexts"]


def test_hierarchical_rag_indexes_image_metadata_node() -> None:
    idx = HierarchicalRAGIndex()
    ingest = idx.index_document(
        source_id="src-img",
        title="Design Board",
        content="Landing page design notes.",
        metadata={
            "topic": "design",
            "image_title": "Landing Mockup",
            "image_caption": "Hero section with CTA and gradients",
            "image_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/n6kAAAAASUVORK5CYII=",
        },
    )
    assert ingest["node_count"] >= 3

    tree = idx.get_source_tree("src-img")
    image_nodes = [n for n in tree["nodes"] if n.get("metadata", {}).get("modality") == "image"]
    assert len(image_nodes) == 1
    assert int(image_nodes[0].get("embedding_dim", 0)) > 0


def test_hierarchical_rag_embedding_backend_switch_and_fallback() -> None:
    idx = HierarchicalRAGIndex(embedding_backend="mlx_clip", embedding_dim=48)
    cfg = idx.get_embedding_config()
    assert cfg["backend_requested"] == "mlx_clip"
    assert cfg["backend"] in {"mlx_clip", "local_deterministic"}
    assert cfg["dim"] == 48
    assert "multimodal_backend" in cfg

    idx.set_embedding_backend(backend="local_deterministic", dim=32)
    idx.set_multimodal_embedding_backend(backend="mlx_clip", dim=40)
    cfg2 = idx.get_embedding_config()
    assert cfg2["backend"] == "local_deterministic"
    assert cfg2["dim"] == 32
    assert cfg2["multimodal_dim"] == 40


def test_hierarchical_rag_query_strategy_includes_fusion_and_reranker() -> None:
    idx = HierarchicalRAGIndex(
        embedding_backend="local_deterministic",
        embedding_dim=32,
        multimodal_embedding_backend="mlx_clip",
        multimodal_embedding_dim=32,
        fusion_text_weight=0.6,
        fusion_multimodal_weight=0.4,
        reranker_enabled=True,
        reranker_top_k=5,
    )
    idx.index_document(
        source_id="src-fusion",
        title="UI Design Spec",
        content="# Overview\nA dashboard layout and visual hierarchy.",
        metadata={"topic": "design"},
    )
    out = idx.query(query="visual dashboard design", max_nodes=3, expand_neighbors=False)
    assert out["count"] >= 1
    assert "strategy" in out
    assert out["strategy"]["reranker_enabled"] is True
    first = out["nodes"][0]
    assert "semantic_text_score" in first
    assert "semantic_multimodal_score" in first
    assert "rerank_score" in first
