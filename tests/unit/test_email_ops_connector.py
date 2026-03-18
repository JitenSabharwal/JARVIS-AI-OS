from __future__ import annotations

import pytest

from infrastructure.builtin_connectors import EmailOpsConnector


@pytest.mark.asyncio
async def test_email_ops_oauth_and_triage_undo_flow() -> None:
    conn = EmailOpsConnector(max_ops_per_minute=50)

    await conn.invoke(
        "oauth_connect",
        {
            "account_id": "acc-1",
            "provider": "gmail",
            "access_token": "tok-1",
            "refresh_token": "ref-1",
            "scopes": ["mail.read", "mail.send"],
            "expires_in_sec": 3600,
        },
    )

    ingest = await conn.invoke(
        "ingest_inbox",
        {
            "account_id": "acc-1",
            "messages": [
                {
                    "message_id": "m-1",
                    "from": "alice@example.com",
                    "subject": "Need update",
                    "body": "Can you share status?",
                }
            ],
        },
    )
    assert ingest["inserted"] == 1

    classified = await conn.invoke(
        "classify",
        {"account_id": "acc-1", "message_id": "m-1", "label": "important"},
    )
    assert "action_id" in classified

    prioritized = await conn.invoke(
        "prioritize",
        {"account_id": "acc-1", "message_id": "m-1", "priority": "high"},
    )
    assert "action_id" in prioritized

    listed = await conn.invoke("list_inbox", {"account_id": "acc-1", "limit": 10})
    assert listed["messages"][-1]["label"] == "important"
    assert listed["messages"][-1]["priority"] == "high"

    undone = await conn.invoke("undo", {"account_id": "acc-1", "action_id": prioritized["action_id"]})
    assert undone["undone"] is True

    listed_after = await conn.invoke("list_inbox", {"account_id": "acc-1", "limit": 10})
    assert listed_after["messages"][-1]["priority"] == "normal"


@pytest.mark.asyncio
async def test_email_ops_expired_token_auto_refresh() -> None:
    conn = EmailOpsConnector(max_ops_per_minute=50)

    await conn.invoke(
        "oauth_connect",
        {
            "account_id": "acc-2",
            "provider": "outlook",
            "access_token": "tok-2",
            "refresh_token": "ref-2",
            "expires_in_sec": 0,
        },
    )

    # Triggers auth check and auto-refresh path.
    listed = await conn.invoke("list_inbox", {"account_id": "acc-2"})
    assert listed["count"] == 0


@pytest.mark.asyncio
async def test_email_ops_rate_limit_enforced() -> None:
    conn = EmailOpsConnector(max_ops_per_minute=1)
    await conn.invoke(
        "oauth_connect",
        {
            "account_id": "acc-3",
            "provider": "imap",
            "access_token": "tok-3",
            "refresh_token": "ref-3",
            "expires_in_sec": 3600,
        },
    )
    await conn.invoke(
        "ingest_inbox",
        {
            "account_id": "acc-3",
            "messages": [{"from": "a@b.com", "subject": "s", "body": "b"}],
        },
    )
    with pytest.raises(RuntimeError, match="rate_limited"):
        await conn.invoke(
            "ingest_inbox",
            {
                "account_id": "acc-3",
                "messages": [{"from": "a@b.com", "subject": "s2", "body": "b2"}],
            },
        )
