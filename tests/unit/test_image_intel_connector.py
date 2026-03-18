from __future__ import annotations

import pytest

from infrastructure.builtin_connectors import ImageIntelConnector, build_default_connector_registry


@pytest.mark.asyncio
async def test_image_intel_preview_apply_undo(tmp_path) -> None:
    root = tmp_path / "images"
    root.mkdir(parents=True, exist_ok=True)
    (root / "trip_1.jpg").write_bytes(b"a")
    (root / "trip_2.jpg").write_bytes(b"b")
    (root / "work_1.png").write_bytes(b"c")

    conn = ImageIntelConnector(base_dir=str(tmp_path))

    grouped = await conn.invoke(
        "group_images",
        {"path": "images", "strategy": "by_prefix", "recursive": False},
    )
    assert grouped["group_count"] >= 2

    plan = await conn.invoke(
        "preview_organize",
        {
            "path": "images",
            "strategy": "by_prefix",
            "target_root": "organized",
            "recursive": False,
        },
    )
    assert plan["status"] == "preview"
    assert plan["move_count"] == 3

    applied = await conn.invoke("apply_plan", {"plan_id": plan["plan_id"]})
    assert applied["applied"] is True
    assert applied["applied_count"] == 3

    undone = await conn.invoke("undo_plan", {"plan_id": plan["plan_id"]})
    assert undone["undone"] is True
    assert undone["undone_count"] == 3


@pytest.mark.asyncio
async def test_image_intel_registry_scopes(tmp_path) -> None:
    registry = build_default_connector_registry(str(tmp_path))
    image_base = tmp_path / "image_intel" / "src"
    image_base.mkdir(parents=True, exist_ok=True)
    (image_base / "img1.jpg").write_bytes(b"x")

    with pytest.raises(PermissionError):
        await registry.invoke(
            "image_intel",
            "preview_organize",
            {"path": "src", "target_root": "organized"},
            actor_scopes=set(),
        )

    preview = await registry.invoke(
        "image_intel",
        "preview_organize",
        {"path": "src", "target_root": "organized", "recursive": False},
        actor_scopes={"connector:image_intel:plan"},
    )
    assert preview["move_count"] == 1

    applied = await registry.invoke(
        "image_intel",
        "apply_plan",
        {"plan_id": preview["plan_id"]},
        actor_scopes={"connector:image_intel:write"},
    )
    assert applied["applied"] is True
