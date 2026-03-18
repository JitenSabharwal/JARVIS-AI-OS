from infrastructure.research_intelligence import ResearchIntelligenceEngine
from infrastructure.research_adapters import StaticResearchAdapter


def test_research_engine_ingest_dedupe_and_query() -> None:
    engine = ResearchIntelligenceEngine()
    result = engine.ingest_sources(
        [
            {
                "title": "Python 3.14 release gains speed",
                "url": "https://example.com/python?utm_source=x",
                "content": "Python increase performance gain",
                "topic": "python",
                "source_type": "news",
                "published_at": "2026-03-17T00:00:00+00:00",
            },
            {
                "title": "Python 3.14 release gains speed",
                "url": "https://example.com/python",
                "content": "Python increase performance gain",
                "topic": "python",
                "source_type": "news",
                "published_at": "2026-03-17T00:00:00+00:00",
            },
        ]
    )
    assert result["inserted"] == 1
    assert result["skipped_duplicates"] == 1

    q = engine.query("python", max_results=5, freshness_days=365)
    assert q["result_count"] >= 1
    assert q["citations"][0]["url"].startswith("https://example.com/python")
    assert "rag_context_count" in q
    assert "graph_context" in q


def test_research_engine_contradictions_and_digest() -> None:
    engine = ResearchIntelligenceEngine()
    engine.ingest_sources(
        [
            {
                "title": "Model X shows growth",
                "url": "https://source-a.com/a",
                "content": "The benchmark results show increase and growth",
                "topic": "model x",
                "source_type": "blog",
                "metadata": {"claim_key": "model-x-performance"},
            },
            {
                "title": "Model X has decline",
                "url": "https://source-b.com/b",
                "content": "The benchmark results show decrease and drop",
                "topic": "model x",
                "source_type": "news",
                "metadata": {"claim_key": "model-x-performance"},
            },
        ]
    )
    query = engine.query("model x", max_results=10, freshness_days=365)
    assert len(query["contradictions"]) >= 1

    watch = engine.create_watchlist(name="AI Watch", topics=["model x"], cadence="daily")
    digest = engine.generate_digest(watch["watchlist_id"], max_per_topic=2)
    assert digest["watchlist_id"] == watch["watchlist_id"]
    assert len(digest["sections"]) == 1
    listed = engine.list_watchlists()
    assert listed[0]["last_digest_at"] is not None


def test_research_engine_run_due_digests_and_freshness_controls() -> None:
    engine = ResearchIntelligenceEngine()
    engine.ingest_sources(
        [
            {
                "title": "Old social post",
                "url": "https://social.example.com/p1",
                "content": "ai rise",
                "topic": "ai",
                "source_type": "social",
                "published_at": "2020-01-01T00:00:00+00:00",
            },
            {
                "title": "Fresh official update",
                "url": "https://official.example.com/p2",
                "content": "ai rise and increase",
                "topic": "ai",
                "source_type": "official",
                "published_at": "2026-03-17T00:00:00+00:00",
            },
        ]
    )
    q = engine.query("ai", max_results=5, freshness_days=30, min_trust=0.6)
    assert q["result_count"] >= 1
    assert q["citation_health_score"] > 0
    assert q["source_type_coverage"].get("official", 0) >= 1

    watch = engine.create_watchlist(name="Due Watch", topics=["ai"], cadence="hourly")
    due_1 = engine.run_due_digests(max_per_topic=2)
    assert due_1["generated_count"] == 1
    due_2 = engine.run_due_digests(max_per_topic=2)
    assert due_2["generated_count"] == 0
    assert watch["watchlist_id"] in due_2["skipped_watchlist_ids"]


def test_research_engine_adapters_run_and_ingest() -> None:
    engine = ResearchIntelligenceEngine()
    engine.register_adapter(
        StaticResearchAdapter(
            name="static-test",
            items=[
                {
                    "title": "AI trend item",
                    "url": "https://adapter.example.com/1",
                    "content": "ai trend rise",
                    "topic": "ai",
                    "source_type": "blog",
                }
            ],
        )
    )
    adapters = engine.list_adapters()
    assert adapters and adapters[0]["name"] == "static-test"
    run = engine.run_adapters(topic="ai", max_items_per_adapter=5)
    assert run["adapter_count"] == 1
    assert run["inserted_total"] >= 1


def test_research_engine_supports_multimodal_rag_metadata() -> None:
    engine = ResearchIntelligenceEngine()
    ingest = engine.ingest_sources(
        [
            {
                "title": "UI Moodboard",
                "url": "https://example.com/ui-moodboard",
                "content": "Design exploration for dashboard.",
                "topic": "design",
                "source_type": "official",
                "metadata": {
                    "image_title": "Dashboard Concept",
                    "image_caption": "Modern UI with card layout",
                    "image_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/n6kAAAAASUVORK5CYII=",
                },
            }
        ]
    )
    assert ingest["inserted"] == 1
    source_id = next(iter(engine._sources.keys()))
    tree = engine.get_source_tree(source_id)
    assert any(n.get("metadata", {}).get("modality") == "image" for n in tree.get("nodes", []))


def test_research_engine_embedding_backend_configurable() -> None:
    engine = ResearchIntelligenceEngine(embedding_backend="local_deterministic", embedding_dim=40)
    cfg = engine.get_embedding_config()
    assert cfg["backend"] == "local_deterministic"
    assert cfg["dim"] == 40
    engine.set_embedding_backend(backend="mlx_clip", dim=56)
    cfg2 = engine.get_embedding_config()
    assert cfg2["backend_requested"] == "mlx_clip"
    assert cfg2["dim"] == 56
