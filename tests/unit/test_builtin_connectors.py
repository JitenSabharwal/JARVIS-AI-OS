from __future__ import annotations

import pytest

from infrastructure.builtin_connectors import build_default_connector_registry


@pytest.mark.asyncio
async def test_default_connectors_registered_and_health(tmp_path) -> None:
    registry = build_default_connector_registry(str(tmp_path))
    info = registry.list_info()
    names = {i["name"] for i in info}
    assert {"calendar", "mail", "files_notifications"}.issubset(names)

    health = await registry.health_all()
    assert set(health.keys()) >= {"calendar", "mail", "files_notifications"}
    assert all(bool(health[name]["healthy"]) for name in ("calendar", "mail", "files_notifications"))


@pytest.mark.asyncio
async def test_files_notifications_write_read_scope_enforced(tmp_path) -> None:
    registry = build_default_connector_registry(str(tmp_path))

    with pytest.raises(PermissionError):
        await registry.invoke(
            "files_notifications",
            "write_note",
            {"path": "notes/test.txt", "content": "hello"},
            actor_scopes=set(),
        )

    await registry.invoke(
        "files_notifications",
        "write_note",
        {"path": "notes/test.txt", "content": "hello"},
        actor_scopes={"connector:files:write"},
    )
    read_result = await registry.invoke(
        "files_notifications",
        "read_note",
        {"path": "notes/test.txt"},
        actor_scopes={"connector:files:read"},
    )
    assert read_result["content"] == "hello"
