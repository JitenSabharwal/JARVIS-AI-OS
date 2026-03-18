from __future__ import annotations

import pytest

from infrastructure.builtin_connectors import FileIntelConnector, build_default_connector_registry


@pytest.mark.asyncio
async def test_file_intel_index_and_acl_summary(tmp_path) -> None:
    root = tmp_path / "intel"
    root.mkdir(parents=True, exist_ok=True)
    (root / "notes.txt").write_text("Project alpha status update with milestones and blockers.", encoding="utf-8")

    conn = FileIntelConnector(base_dir=str(root))
    indexed = await conn.invoke(
        "index_file",
        {"path": "notes.txt", "acl_tags": ["team-a", "confidential"]},
    )
    assert indexed["indexed"] == 1
    doc_id = indexed["record"]["doc_id"]

    denied = await conn.invoke("list_index", {"actor_acl_tags": ["team-b"]})
    assert denied["count"] == 0

    allowed = await conn.invoke("list_index", {"actor_acl_tags": ["team-a"]})
    assert allowed["count"] == 1

    with pytest.raises(PermissionError):
        await conn.invoke(
            "summarize_indexed",
            {"doc_id": doc_id, "actor_acl_tags": ["team-b"]},
        )

    summary = await conn.invoke(
        "summarize_indexed",
        {"doc_id": doc_id, "actor_acl_tags": ["team-a"], "max_chars": 400},
    )
    assert summary["doc_id"] == doc_id
    assert summary["confidence"] >= 0.6
    assert "freshness_ts" in summary


@pytest.mark.asyncio
async def test_file_intel_registry_scopes(tmp_path) -> None:
    registry = build_default_connector_registry(str(tmp_path))
    file_base = tmp_path / "file_intel"
    file_base.mkdir(parents=True, exist_ok=True)
    (file_base / "doc.txt").write_text("hello world", encoding="utf-8")

    with pytest.raises(PermissionError):
        await registry.invoke(
            "file_intel",
            "index_file",
            {"path": "doc.txt", "acl_tags": ["x"]},
            actor_scopes=set(),
        )

    result = await registry.invoke(
        "file_intel",
        "index_file",
        {"path": "doc.txt", "acl_tags": ["x"]},
        actor_scopes={"connector:file_intel:index"},
    )
    assert result["indexed"] == 1

    listed = await registry.invoke(
        "file_intel",
        "list_index",
        {"actor_acl_tags": ["x"]},
        actor_scopes={"connector:file_intel:read"},
    )
    assert listed["count"] >= 1
