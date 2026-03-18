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
