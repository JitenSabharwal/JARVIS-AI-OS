"""
Built-in connector set for production-profile API mode.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from infrastructure.connectors import BaseConnector, ConnectorPolicy, ConnectorRegistry


class CalendarConnector(BaseConnector):
    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "calendar"

    @property
    def description(self) -> str:
        return "Calendar connector (create/list event)."

    async def invoke(self, operation: str, params: Dict[str, Any]) -> Any:
        if operation == "create_event":
            title = str(params.get("title", "")).strip()
            start = str(params.get("start", "")).strip()
            end = str(params.get("end", "")).strip()
            if not title or not start or not end:
                raise ValueError("create_event requires title/start/end")
            event = {
                "event_id": f"evt-{int(time.time() * 1000)}-{len(self._events) + 1}",
                "title": title,
                "start": start,
                "end": end,
                "created_at": time.time(),
            }
            self._events.append(event)
            return event
        if operation == "list_events":
            limit = int(params.get("limit", 20))
            limit = max(1, min(500, limit))
            return {"events": self._events[-limit:], "count": min(limit, len(self._events))}
        raise ValueError(f"Unsupported operation for calendar connector: {operation}")

    async def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "connector": self.name, "events_cached": len(self._events)}


class MailConnector(BaseConnector):
    def __init__(self) -> None:
        self._sent: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "mail"

    @property
    def description(self) -> str:
        return "Mail connector (send/list sent messages)."

    async def invoke(self, operation: str, params: Dict[str, Any]) -> Any:
        if operation == "send_mail":
            to = str(params.get("to", "")).strip()
            subject = str(params.get("subject", "")).strip()
            body = str(params.get("body", "")).strip()
            if not to or "@" not in to:
                raise ValueError("send_mail requires a valid 'to' email")
            if not subject:
                raise ValueError("send_mail requires 'subject'")
            if not body:
                raise ValueError("send_mail requires 'body'")
            msg = {
                "message_id": f"msg-{int(time.time() * 1000)}-{len(self._sent) + 1}",
                "to": to,
                "subject": subject,
                "body_preview": body[:80],
                "sent_at": time.time(),
            }
            self._sent.append(msg)
            return msg
        if operation == "list_sent":
            limit = int(params.get("limit", 20))
            limit = max(1, min(500, limit))
            return {"messages": self._sent[-limit:], "count": min(limit, len(self._sent))}
        raise ValueError(f"Unsupported operation for mail connector: {operation}")

    async def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "connector": self.name, "messages_cached": len(self._sent)}


class EmailOpsConnector(BaseConnector):
    """
    Provider-aware email operations connector with OAuth lifecycle, rate limits,
    triage actions, and reversible mutations.
    """

    def __init__(self, *, max_ops_per_minute: int = 60) -> None:
        self._accounts: Dict[str, Dict[str, Any]] = {}
        self._inbox: Dict[str, List[Dict[str, Any]]] = {}
        self._drafts: Dict[str, List[Dict[str, Any]]] = {}
        self._scheduled: Dict[str, List[Dict[str, Any]]] = {}
        self._follow_ups: Dict[str, List[Dict[str, Any]]] = {}
        self._actions: Dict[str, List[Dict[str, Any]]] = {}
        self._op_windows: Dict[str, List[float]] = {}
        self._max_ops_per_minute = max(1, int(max_ops_per_minute))

    @property
    def name(self) -> str:
        return "email_ops"

    @property
    def description(self) -> str:
        return (
            "Email operations connector (oauth connect/refresh, inbox triage, "
            "draft/schedule/follow-up, undo)."
        )

    async def invoke(self, operation: str, params: Dict[str, Any]) -> Any:
        if operation == "oauth_connect":
            return self._oauth_connect(params)
        if operation == "oauth_refresh":
            return self._oauth_refresh(params)
        if operation == "ingest_inbox":
            account_id = self._require_account(params)
            self._ensure_auth(account_id)
            self._check_rate_limit(account_id)
            return self._ingest_inbox(account_id, params)
        if operation == "list_inbox":
            account_id = self._require_account(params)
            self._ensure_auth(account_id)
            return self._list_inbox(account_id, params)
        if operation == "draft_reply":
            account_id = self._require_account(params)
            self._ensure_auth(account_id)
            self._check_rate_limit(account_id)
            return self._draft_reply(account_id, params)
        if operation == "classify":
            account_id = self._require_account(params)
            self._ensure_auth(account_id)
            self._check_rate_limit(account_id)
            return self._classify(account_id, params)
        if operation == "prioritize":
            account_id = self._require_account(params)
            self._ensure_auth(account_id)
            self._check_rate_limit(account_id)
            return self._prioritize(account_id, params)
        if operation == "schedule_send":
            account_id = self._require_account(params)
            self._ensure_auth(account_id)
            self._check_rate_limit(account_id)
            return self._schedule_send(account_id, params)
        if operation == "send_draft":
            account_id = self._require_account(params)
            self._ensure_auth(account_id)
            self._check_rate_limit(account_id)
            return self._send_draft(account_id, params)
        if operation == "follow_up":
            account_id = self._require_account(params)
            self._ensure_auth(account_id)
            self._check_rate_limit(account_id)
            return self._follow_up(account_id, params)
        if operation == "undo":
            account_id = self._require_account(params)
            self._ensure_auth(account_id)
            self._check_rate_limit(account_id)
            return self._undo(account_id, params)
        if operation == "list_actions":
            account_id = self._require_account(params)
            self._ensure_auth(account_id)
            return self._list_actions(account_id, params)
        raise ValueError(f"Unsupported operation for email_ops connector: {operation}")

    async def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "connector": self.name,
            "accounts": len(self._accounts),
            "inbox_messages": sum(len(v) for v in self._inbox.values()),
            "drafts": sum(len(v) for v in self._drafts.values()),
            "scheduled": sum(len(v) for v in self._scheduled.values()),
            "follow_ups": sum(len(v) for v in self._follow_ups.values()),
        }

    @staticmethod
    def _message_id() -> str:
        return f"em-{int(time.time() * 1000)}"

    def _require_account(self, params: Dict[str, Any]) -> str:
        account_id = str(params.get("account_id", "")).strip()
        if not account_id:
            raise ValueError("account_id is required")
        return account_id

    def _check_rate_limit(self, account_id: str) -> None:
        now = time.time()
        window = self._op_windows.setdefault(account_id, [])
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.pop(0)
        if len(window) >= self._max_ops_per_minute:
            raise RuntimeError("rate_limited: too many email operations in 60s window")
        window.append(now)

    def _ensure_auth(self, account_id: str) -> None:
        account = self._accounts.get(account_id)
        if account is None:
            raise PermissionError(f"oauth account not connected: {account_id}")
        expires_at = float(account.get("expires_at", 0.0))
        now = time.time()
        if expires_at > now:
            return
        refresh_token = str(account.get("refresh_token", "")).strip()
        if not refresh_token:
            raise PermissionError(f"oauth token expired for account: {account_id}")
        # Simulated refresh lifecycle.
        account["access_token"] = f"refreshed-{int(now)}"
        account["expires_at"] = now + 3600.0
        account["refreshed_at"] = now

    def _oauth_connect(self, params: Dict[str, Any]) -> Dict[str, Any]:
        account_id = self._require_account(params)
        provider = str(params.get("provider", "")).strip().lower()
        if provider not in {"gmail", "outlook", "imap"}:
            raise ValueError("provider must be one of: gmail,outlook,imap")
        access_token = str(params.get("access_token", "")).strip()
        if not access_token:
            raise ValueError("access_token is required")
        refresh_token = str(params.get("refresh_token", "")).strip()
        scopes = params.get("scopes", [])
        if not isinstance(scopes, list):
            raise ValueError("scopes must be list[str]")
        expires_in_sec = params.get("expires_in_sec", 3600)
        try:
            expires_in = max(0, int(expires_in_sec))
        except (TypeError, ValueError):
            raise ValueError("expires_in_sec must be integer >= 0")
        now = time.time()
        self._accounts[account_id] = {
            "account_id": account_id,
            "provider": provider,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "scopes": [str(s) for s in scopes],
            "connected_at": now,
            "expires_at": now + float(expires_in),
        }
        self._inbox.setdefault(account_id, [])
        self._drafts.setdefault(account_id, [])
        self._scheduled.setdefault(account_id, [])
        self._follow_ups.setdefault(account_id, [])
        self._actions.setdefault(account_id, [])
        return {
            "account_id": account_id,
            "provider": provider,
            "connected": True,
            "expires_at": self._accounts[account_id]["expires_at"],
        }

    def _oauth_refresh(self, params: Dict[str, Any]) -> Dict[str, Any]:
        account_id = self._require_account(params)
        account = self._accounts.get(account_id)
        if account is None:
            raise PermissionError(f"oauth account not connected: {account_id}")
        refresh_token = str(account.get("refresh_token", "")).strip()
        if not refresh_token:
            raise PermissionError("oauth refresh_token missing")
        now = time.time()
        account["access_token"] = f"manual-refresh-{int(now)}"
        account["expires_at"] = now + 3600.0
        account["refreshed_at"] = now
        return {"account_id": account_id, "refreshed": True, "expires_at": account["expires_at"]}

    def _ingest_inbox(self, account_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        items = params.get("messages", [])
        if not isinstance(items, list):
            raise ValueError("messages must be list[object]")
        inbox = self._inbox.setdefault(account_id, [])
        inserted = 0
        for raw in items:
            if not isinstance(raw, dict):
                continue
            sender = str(raw.get("from", "")).strip()
            subject = str(raw.get("subject", "")).strip()
            body = str(raw.get("body", "")).strip()
            if not sender or not subject:
                continue
            msg = {
                "message_id": str(raw.get("message_id", "")).strip() or self._message_id(),
                "from": sender,
                "subject": subject,
                "body": body,
                "label": str(raw.get("label", "inbox")).strip() or "inbox",
                "priority": str(raw.get("priority", "normal")).strip() or "normal",
                "received_at": float(raw.get("received_at", time.time())),
            }
            inbox.append(msg)
            inserted += 1
        self._record_action(account_id, "ingest_inbox", {"inserted": inserted}, reversible=False)
        return {"account_id": account_id, "inserted": inserted, "count": len(inbox)}

    def _list_inbox(self, account_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        inbox = self._inbox.setdefault(account_id, [])
        limit = int(params.get("limit", 20))
        limit = max(1, min(500, limit))
        label_filter = str(params.get("label", "")).strip()
        items = list(inbox)
        if label_filter:
            items = [m for m in items if str(m.get("label", "")) == label_filter]
        return {"account_id": account_id, "messages": items[-limit:], "count": min(limit, len(items))}

    def _find_message(self, account_id: str, message_id: str) -> Dict[str, Any]:
        inbox = self._inbox.setdefault(account_id, [])
        for msg in inbox:
            if str(msg.get("message_id")) == message_id:
                return msg
        raise KeyError(f"message not found: {message_id}")

    def _draft_reply(self, account_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        message_id = str(params.get("message_id", "")).strip()
        body = str(params.get("body", "")).strip()
        if not message_id or not body:
            raise ValueError("draft_reply requires message_id and body")
        msg = self._find_message(account_id, message_id)
        draft = {
            "draft_id": f"dr-{int(time.time() * 1000)}-{len(self._drafts[account_id]) + 1}",
            "message_id": message_id,
            "to": msg.get("from", ""),
            "subject": f"Re: {msg.get('subject', '')}",
            "body": body,
            "status": "draft",
            "created_at": time.time(),
        }
        self._drafts[account_id].append(draft)
        self._record_action(account_id, "draft_reply", {"draft_id": draft["draft_id"]}, reversible=False)
        return draft

    def _classify(self, account_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        message_id = str(params.get("message_id", "")).strip()
        label = str(params.get("label", "")).strip()
        if not message_id or not label:
            raise ValueError("classify requires message_id and label")
        msg = self._find_message(account_id, message_id)
        previous = str(msg.get("label", ""))
        msg["label"] = label
        action = self._record_action(
            account_id,
            "classify",
            {"message_id": message_id, "previous": previous, "current": label},
            reversible=True,
        )
        return {"message_id": message_id, "label": label, "action_id": action["action_id"]}

    def _prioritize(self, account_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        message_id = str(params.get("message_id", "")).strip()
        priority = str(params.get("priority", "")).strip().lower()
        if priority not in {"low", "normal", "high", "urgent"}:
            raise ValueError("priority must be one of low,normal,high,urgent")
        msg = self._find_message(account_id, message_id)
        previous = str(msg.get("priority", "normal"))
        msg["priority"] = priority
        action = self._record_action(
            account_id,
            "prioritize",
            {"message_id": message_id, "previous": previous, "current": priority},
            reversible=True,
        )
        return {"message_id": message_id, "priority": priority, "action_id": action["action_id"]}

    def _schedule_send(self, account_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        draft_id = str(params.get("draft_id", "")).strip()
        send_at = str(params.get("send_at", "")).strip()
        if not draft_id or not send_at:
            raise ValueError("schedule_send requires draft_id and send_at")
        drafts = self._drafts.setdefault(account_id, [])
        draft = next((d for d in drafts if str(d.get("draft_id")) == draft_id), None)
        if draft is None:
            raise KeyError(f"draft not found: {draft_id}")
        previous = str(draft.get("status", "draft"))
        draft["status"] = "scheduled"
        item = {"draft_id": draft_id, "send_at": send_at, "scheduled_at": time.time()}
        self._scheduled[account_id].append(item)
        action = self._record_action(
            account_id,
            "schedule_send",
            {"draft_id": draft_id, "previous": previous, "current": "scheduled", "send_at": send_at},
            reversible=True,
        )
        return {"draft_id": draft_id, "status": "scheduled", "action_id": action["action_id"]}

    def _send_draft(self, account_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        draft_id = str(params.get("draft_id", "")).strip()
        if not draft_id:
            raise ValueError("send_draft requires draft_id")
        drafts = self._drafts.setdefault(account_id, [])
        draft = next((d for d in drafts if str(d.get("draft_id")) == draft_id), None)
        if draft is None:
            raise KeyError(f"draft not found: {draft_id}")
        draft["status"] = "sent"
        draft["sent_at"] = time.time()
        self._record_action(account_id, "send_draft", {"draft_id": draft_id}, reversible=False)
        return {"draft_id": draft_id, "status": "sent"}

    def _follow_up(self, account_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        message_id = str(params.get("message_id", "")).strip()
        due_at = str(params.get("due_at", "")).strip()
        note = str(params.get("note", "")).strip()
        if not message_id or not due_at:
            raise ValueError("follow_up requires message_id and due_at")
        item = {
            "follow_up_id": f"fu-{int(time.time() * 1000)}-{len(self._follow_ups[account_id]) + 1}",
            "message_id": message_id,
            "due_at": due_at,
            "note": note,
            "created_at": time.time(),
        }
        self._follow_ups[account_id].append(item)
        action = self._record_action(
            account_id,
            "follow_up",
            {"follow_up_id": item["follow_up_id"]},
            reversible=True,
        )
        item["action_id"] = action["action_id"]
        return item

    def _undo(self, account_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        action_id = str(params.get("action_id", "")).strip()
        actions = self._actions.setdefault(account_id, [])
        candidates = [a for a in actions if bool(a.get("reversible", False))]
        if action_id:
            candidates = [a for a in candidates if str(a.get("action_id")) == action_id]
        if not candidates:
            raise KeyError("reversible action not found")
        action = candidates[-1]
        op = str(action.get("operation", ""))
        data = dict(action.get("data", {}))
        if op == "classify":
            msg = self._find_message(account_id, str(data.get("message_id", "")))
            msg["label"] = data.get("previous", "inbox")
        elif op == "prioritize":
            msg = self._find_message(account_id, str(data.get("message_id", "")))
            msg["priority"] = data.get("previous", "normal")
        elif op == "schedule_send":
            draft_id = str(data.get("draft_id", ""))
            drafts = self._drafts.setdefault(account_id, [])
            draft = next((d for d in drafts if str(d.get("draft_id")) == draft_id), None)
            if draft is not None:
                draft["status"] = data.get("previous", "draft")
            self._scheduled[account_id] = [
                s for s in self._scheduled.setdefault(account_id, [])
                if str(s.get("draft_id")) != draft_id
            ]
        elif op == "follow_up":
            follow_up_id = str(data.get("follow_up_id", ""))
            self._follow_ups[account_id] = [
                f for f in self._follow_ups.setdefault(account_id, [])
                if str(f.get("follow_up_id")) != follow_up_id
            ]
        else:
            raise ValueError(f"undo unsupported for operation: {op}")
        undo_action = self._record_action(
            account_id,
            "undo",
            {"undone_action_id": action.get("action_id"), "operation": op},
            reversible=False,
        )
        return {"undone": True, "operation": op, "action_id": undo_action["action_id"]}

    def _list_actions(self, account_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        limit = int(params.get("limit", 50))
        limit = max(1, min(1000, limit))
        items = self._actions.setdefault(account_id, [])
        return {"actions": items[-limit:], "count": min(limit, len(items))}

    def _record_action(
        self,
        account_id: str,
        operation: str,
        data: Dict[str, Any],
        *,
        reversible: bool,
    ) -> Dict[str, Any]:
        item = {
            "action_id": f"act-{int(time.time() * 1000)}-{len(self._actions.setdefault(account_id, [])) + 1}",
            "operation": operation,
            "data": dict(data),
            "reversible": reversible,
            "created_at": time.time(),
        }
        self._actions.setdefault(account_id, []).append(item)
        return item


class FilesNotificationsConnector(BaseConnector):
    def __init__(self, base_dir: str = "data/connector_files") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._notifications: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "files_notifications"

    @property
    def description(self) -> str:
        return "Files + notifications connector (write/read notes, emit notifications)."

    def _resolve_safe(self, relative_path: str) -> Path:
        rel = relative_path.strip().lstrip("/")
        if not rel:
            raise ValueError("path is required")
        target = (self._base_dir / rel).resolve()
        base = self._base_dir.resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError("path escapes connector base directory")
        return target

    async def invoke(self, operation: str, params: Dict[str, Any]) -> Any:
        if operation == "write_note":
            path = self._resolve_safe(str(params.get("path", "")).strip())
            content = str(params.get("content", ""))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {"path": str(path), "bytes": len(content.encode("utf-8"))}
        if operation == "read_note":
            path = self._resolve_safe(str(params.get("path", "")).strip())
            if not path.exists():
                raise FileNotFoundError(f"File does not exist: {path}")
            text = path.read_text(encoding="utf-8")
            return {"path": str(path), "content": text}
        if operation == "notify":
            channel = str(params.get("channel", "")).strip() or "general"
            message = str(params.get("message", "")).strip()
            if not message:
                raise ValueError("notify requires 'message'")
            item = {
                "notification_id": f"ntf-{int(time.time() * 1000)}-{len(self._notifications) + 1}",
                "channel": channel,
                "message": message,
                "timestamp": time.time(),
            }
            self._notifications.append(item)
            return item
        if operation == "list_notifications":
            limit = int(params.get("limit", 20))
            limit = max(1, min(500, limit))
            return {
                "notifications": self._notifications[-limit:],
                "count": min(limit, len(self._notifications)),
            }
        raise ValueError(f"Unsupported operation for files_notifications connector: {operation}")

    async def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "connector": self.name,
            "base_dir": str(self._base_dir),
            "notifications_cached": len(self._notifications),
        }


class FileIntelConnector(BaseConnector):
    """
    Secure file indexing and summarization connector with simple ACL tags.

    ACL model:
    - Each indexed document is tagged with one or more ACL labels.
    - Read/summarize operations must provide actor_acl_tags and intersect.
    """

    def __init__(self, base_dir: str = "data/file_intel") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "file_intel"

    @property
    def description(self) -> str:
        return "File intelligence connector (index/list/summarize) with ACL-aware retrieval."

    def _resolve_safe(self, relative_path: str) -> Path:
        rel = relative_path.strip().lstrip("/")
        if not rel:
            raise ValueError("path is required")
        target = (self._base_dir / rel).resolve()
        base = self._base_dir.resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError("path escapes file_intel base directory")
        return target

    @staticmethod
    def _sanitize_acl_tags(raw: Any) -> List[str]:
        if not isinstance(raw, list):
            return []
        return [str(t).strip() for t in raw if str(t).strip()]

    @staticmethod
    def _check_acl(actor_acl_tags: List[str], doc_acl_tags: List[str]) -> bool:
        if not doc_acl_tags:
            return True
        actor = {t for t in actor_acl_tags}
        doc = {t for t in doc_acl_tags}
        return bool(actor.intersection(doc))

    def _summarize_text(self, text: str, max_chars: int = 1200) -> Dict[str, Any]:
        cleaned = " ".join(text.split())
        clipped = cleaned[: max(100, min(10000, int(max_chars)))]
        words = clipped.split()
        summary = " ".join(words[: min(80, len(words))])
        confidence = 0.9 if len(words) >= 20 else 0.65
        return {
            "summary": summary,
            "word_count": len(cleaned.split()),
            "char_count": len(text),
            "confidence": confidence,
        }

    def _index_file(self, *, path: Path, acl_tags: List[str]) -> Dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        stat = path.stat()
        rel_path = str(path.relative_to(self._base_dir.resolve()))
        doc_id = f"doc-{int(stat.st_mtime * 1000)}-{abs(hash(rel_path)) % 1000000}"
        summary = self._summarize_text(text)
        record = {
            "doc_id": doc_id,
            "path": rel_path,
            "acl_tags": list(acl_tags),
            "indexed_at": time.time(),
            "modified_at": float(stat.st_mtime),
            "size_bytes": int(stat.st_size),
            "summary_preview": summary["summary"][:240],
            "word_count": summary["word_count"],
            "confidence": summary["confidence"],
        }
        self._index[doc_id] = record
        return record

    async def invoke(self, operation: str, params: Dict[str, Any]) -> Any:
        if operation == "index_file":
            path = self._resolve_safe(str(params.get("path", "")))
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"file not found: {path}")
            acl_tags = self._sanitize_acl_tags(params.get("acl_tags", []))
            record = self._index_file(path=path, acl_tags=acl_tags)
            return {"indexed": 1, "record": record}

        if operation == "index_directory":
            root = self._resolve_safe(str(params.get("path", "")))
            if not root.exists() or not root.is_dir():
                raise FileNotFoundError(f"directory not found: {root}")
            recursive = bool(params.get("recursive", True))
            max_files = max(1, min(5000, int(params.get("max_files", 500))))
            ext_filter = self._sanitize_acl_tags(params.get("extensions", []))
            ext_filter = [e.lower().lstrip(".") for e in ext_filter]
            acl_tags = self._sanitize_acl_tags(params.get("acl_tags", []))
            indexed: List[Dict[str, Any]] = []
            iterator = root.rglob("*") if recursive else root.glob("*")
            for p in iterator:
                if len(indexed) >= max_files:
                    break
                if not p.is_file():
                    continue
                if ext_filter and p.suffix.lower().lstrip(".") not in ext_filter:
                    continue
                try:
                    indexed.append(self._index_file(path=p.resolve(), acl_tags=acl_tags))
                except Exception:  # noqa: BLE001
                    continue
            return {"indexed": len(indexed), "records": indexed}

        if operation == "list_index":
            actor_acl_tags = self._sanitize_acl_tags(params.get("actor_acl_tags", []))
            limit = max(1, min(2000, int(params.get("limit", 200))))
            docs = [
                rec for rec in self._index.values()
                if self._check_acl(actor_acl_tags, list(rec.get("acl_tags", [])))
            ]
            docs = sorted(docs, key=lambda r: float(r.get("indexed_at", 0.0)), reverse=True)
            return {"documents": docs[:limit], "count": min(limit, len(docs))}

        if operation == "summarize_file":
            path = self._resolve_safe(str(params.get("path", "")))
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"file not found: {path}")
            text = path.read_text(encoding="utf-8")
            max_chars = int(params.get("max_chars", 1200))
            result = self._summarize_text(text, max_chars=max_chars)
            result["path"] = str(path.relative_to(self._base_dir.resolve()))
            result["freshness_ts"] = float(path.stat().st_mtime)
            return result

        if operation == "summarize_indexed":
            doc_id = str(params.get("doc_id", "")).strip()
            if not doc_id:
                raise ValueError("doc_id is required")
            actor_acl_tags = self._sanitize_acl_tags(params.get("actor_acl_tags", []))
            rec = self._index.get(doc_id)
            if rec is None:
                raise KeyError(f"document not found: {doc_id}")
            if not self._check_acl(actor_acl_tags, list(rec.get("acl_tags", []))):
                raise PermissionError("acl denied for document")
            path = self._resolve_safe(str(rec.get("path", "")))
            if not path.exists():
                raise FileNotFoundError(f"file no longer exists: {path}")
            text = path.read_text(encoding="utf-8")
            result = self._summarize_text(text, max_chars=int(params.get("max_chars", 1200)))
            result["doc_id"] = doc_id
            result["path"] = str(rec.get("path", ""))
            result["freshness_ts"] = float(path.stat().st_mtime)
            result["acl_tags"] = list(rec.get("acl_tags", []))
            return result

        if operation == "remove_index":
            doc_id = str(params.get("doc_id", "")).strip()
            if not doc_id:
                raise ValueError("doc_id is required")
            removed = self._index.pop(doc_id, None) is not None
            return {"removed": removed, "doc_id": doc_id}

        raise ValueError(f"Unsupported operation for file_intel connector: {operation}")

    async def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "connector": self.name,
            "base_dir": str(self._base_dir),
            "indexed_docs": len(self._index),
        }


class ImageIntelConnector(BaseConnector):
    """
    Image-folder organization connector with preview/apply/undo flows.
    """

    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".heic"}

    def __init__(self, base_dir: str = "data/image_intel") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._plans: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "image_intel"

    @property
    def description(self) -> str:
        return (
            "Image intelligence connector (scan/group/preview/apply/undo) "
            "for relevance-based folder organization."
        )

    def _resolve_safe(self, relative_path: str) -> Path:
        rel = relative_path.strip().lstrip("/")
        if not rel:
            raise ValueError("path is required")
        target = (self._base_dir / rel).resolve()
        base = self._base_dir.resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError("path escapes image_intel base directory")
        return target

    @staticmethod
    def _stem_tokens(path: Path) -> List[str]:
        stem = path.stem.lower()
        for ch in ("-", "_", ".", "(", ")", "[", "]"):
            stem = stem.replace(ch, " ")
        parts = [p for p in stem.split() if p]
        cleaned: List[str] = []
        for p in parts:
            if p.isdigit():
                continue
            cleaned.append(p)
        return cleaned

    def _scan_images(self, root: Path, *, recursive: bool, limit: int) -> List[Path]:
        iterator = root.rglob("*") if recursive else root.glob("*")
        images: List[Path] = []
        for p in iterator:
            if len(images) >= limit:
                break
            if not p.is_file():
                continue
            if p.suffix.lower() not in self._IMAGE_EXTS:
                continue
            images.append(p.resolve())
        return images

    def _group_images(self, images: List[Path], strategy: str) -> Dict[str, List[Path]]:
        groups: Dict[str, List[Path]] = {}
        for p in images:
            if strategy == "by_extension":
                key = p.suffix.lower().lstrip(".") or "unknown"
            elif strategy == "by_prefix":
                tokens = self._stem_tokens(p)
                key = tokens[0] if tokens else "misc"
            else:  # by_relevance (default)
                tokens = self._stem_tokens(p)
                key = "_".join(tokens[:2]) if tokens else "misc"
            groups.setdefault(key or "misc", []).append(p)
        return groups

    def _relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self._base_dir.resolve()))

    def _safe_unique_dest(self, destination: Path) -> Path:
        if not destination.exists():
            return destination
        stem = destination.stem
        suffix = destination.suffix
        parent = destination.parent
        counter = 1
        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    async def invoke(self, operation: str, params: Dict[str, Any]) -> Any:
        if operation == "scan_images":
            root = self._resolve_safe(str(params.get("path", "")))
            if not root.exists() or not root.is_dir():
                raise FileNotFoundError(f"directory not found: {root}")
            recursive = bool(params.get("recursive", True))
            limit = max(1, min(10000, int(params.get("limit", 2000))))
            images = self._scan_images(root, recursive=recursive, limit=limit)
            return {
                "path": self._relative(root),
                "count": len(images),
                "images": [self._relative(p) for p in images],
            }

        if operation == "group_images":
            root = self._resolve_safe(str(params.get("path", "")))
            if not root.exists() or not root.is_dir():
                raise FileNotFoundError(f"directory not found: {root}")
            recursive = bool(params.get("recursive", True))
            limit = max(1, min(10000, int(params.get("limit", 2000))))
            strategy = str(params.get("strategy", "by_relevance")).strip() or "by_relevance"
            images = self._scan_images(root, recursive=recursive, limit=limit)
            grouped = self._group_images(images, strategy)
            payload = {
                k: {
                    "count": len(v),
                    "items": [self._relative(p) for p in v],
                }
                for k, v in grouped.items()
            }
            return {"path": self._relative(root), "strategy": strategy, "group_count": len(payload), "groups": payload}

        if operation == "preview_organize":
            root = self._resolve_safe(str(params.get("path", "")))
            if not root.exists() or not root.is_dir():
                raise FileNotFoundError(f"directory not found: {root}")
            recursive = bool(params.get("recursive", True))
            limit = max(1, min(10000, int(params.get("limit", 2000))))
            strategy = str(params.get("strategy", "by_relevance")).strip() or "by_relevance"
            target_root_rel = str(params.get("target_root", "organized")).strip() or "organized"
            target_root = self._resolve_safe(target_root_rel)
            images = self._scan_images(root, recursive=recursive, limit=limit)
            grouped = self._group_images(images, strategy)

            moves: List[Dict[str, Any]] = []
            for group_name, files in grouped.items():
                bucket = target_root / group_name
                for src in files:
                    candidate = self._safe_unique_dest(bucket / src.name)
                    moves.append(
                        {
                            "from": self._relative(src),
                            "to": self._relative(candidate),
                            "group": group_name,
                        }
                    )
            plan_id = f"img-plan-{int(time.time() * 1000)}-{len(self._plans) + 1}"
            plan = {
                "plan_id": plan_id,
                "status": "preview",
                "created_at": time.time(),
                "strategy": strategy,
                "source_path": self._relative(root),
                "target_root": self._relative(target_root),
                "move_count": len(moves),
                "moves": moves,
                "applied_at": None,
                "undone_at": None,
            }
            self._plans[plan_id] = plan
            return plan

        if operation == "apply_plan":
            plan_id = str(params.get("plan_id", "")).strip()
            if not plan_id:
                raise ValueError("plan_id is required")
            plan = self._plans.get(plan_id)
            if plan is None:
                raise KeyError(f"plan not found: {plan_id}")
            if plan.get("status") == "applied":
                return {"plan_id": plan_id, "applied": True, "already_applied": True}
            applied = 0
            for move in list(plan.get("moves", [])):
                src = self._resolve_safe(str(move.get("from", "")))
                dst = self._resolve_safe(str(move.get("to", "")))
                if not src.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                applied += 1
            plan["status"] = "applied"
            plan["applied_at"] = time.time()
            return {"plan_id": plan_id, "applied": True, "applied_count": applied}

        if operation == "undo_plan":
            plan_id = str(params.get("plan_id", "")).strip()
            if not plan_id:
                raise ValueError("plan_id is required")
            plan = self._plans.get(plan_id)
            if plan is None:
                raise KeyError(f"plan not found: {plan_id}")
            if plan.get("status") != "applied":
                return {"plan_id": plan_id, "undone": False, "reason": "plan_not_applied"}
            undone = 0
            for move in list(plan.get("moves", [])):
                src = self._resolve_safe(str(move.get("to", "")))
                dst = self._resolve_safe(str(move.get("from", "")))
                if not src.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                undone += 1
            plan["status"] = "undone"
            plan["undone_at"] = time.time()
            return {"plan_id": plan_id, "undone": True, "undone_count": undone}

        if operation == "list_plans":
            limit = max(1, min(1000, int(params.get("limit", 200))))
            items = sorted(
                self._plans.values(),
                key=lambda p: float(p.get("created_at", 0.0)),
                reverse=True,
            )
            return {"plans": items[:limit], "count": min(limit, len(items))}

        raise ValueError(f"Unsupported operation for image_intel connector: {operation}")

    async def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "connector": self.name,
            "base_dir": str(self._base_dir),
            "plans": len(self._plans),
        }


def build_default_connector_registry(data_dir: str = "data") -> ConnectorRegistry:
    """
    Build a connector registry with three production-profile connectors:
    calendar, mail, files_notifications.
    """
    registry = ConnectorRegistry()
    registry.register(
        CalendarConnector(),
        policy=ConnectorPolicy(
            required_scopes_by_operation={
                "create_event": {"connector:calendar:write"},
                "list_events": {"connector:calendar:read"},
            },
            failure_threshold=3,
            recovery_timeout_seconds=30.0,
        ),
    )
    registry.register(
        MailConnector(),
        policy=ConnectorPolicy(
            required_scopes_by_operation={
                "send_mail": {"connector:mail:send"},
                "list_sent": {"connector:mail:read"},
            },
            failure_threshold=3,
            recovery_timeout_seconds=30.0,
        ),
    )
    registry.register(
        EmailOpsConnector(),
        policy=ConnectorPolicy(
            required_scopes_by_operation={
                "oauth_connect": {"connector:email:oauth:write"},
                "oauth_refresh": {"connector:email:oauth:write"},
                "ingest_inbox": {"connector:email:write"},
                "list_inbox": {"connector:email:read"},
                "draft_reply": {"connector:email:draft"},
                "classify": {"connector:email:triage"},
                "prioritize": {"connector:email:triage"},
                "schedule_send": {"connector:email:schedule"},
                "send_draft": {"connector:email:send"},
                "follow_up": {"connector:email:triage"},
                "undo": {"connector:email:undo"},
                "list_actions": {"connector:email:audit"},
            },
            failure_threshold=3,
            recovery_timeout_seconds=30.0,
        ),
    )
    registry.register(
        FileIntelConnector(base_dir=str(Path(data_dir) / "file_intel")),
        policy=ConnectorPolicy(
            required_scopes_by_operation={
                "index_file": {"connector:file_intel:index"},
                "index_directory": {"connector:file_intel:index"},
                "list_index": {"connector:file_intel:read"},
                "summarize_file": {"connector:file_intel:read"},
                "summarize_indexed": {"connector:file_intel:read"},
                "remove_index": {"connector:file_intel:write"},
            },
            failure_threshold=3,
            recovery_timeout_seconds=30.0,
        ),
    )
    registry.register(
        ImageIntelConnector(base_dir=str(Path(data_dir) / "image_intel")),
        policy=ConnectorPolicy(
            required_scopes_by_operation={
                "scan_images": {"connector:image_intel:read"},
                "group_images": {"connector:image_intel:read"},
                "preview_organize": {"connector:image_intel:plan"},
                "apply_plan": {"connector:image_intel:write"},
                "undo_plan": {"connector:image_intel:write"},
                "list_plans": {"connector:image_intel:audit"},
            },
            failure_threshold=3,
            recovery_timeout_seconds=30.0,
        ),
    )
    registry.register(
        FilesNotificationsConnector(base_dir=str(Path(data_dir) / "connector_files")),
        policy=ConnectorPolicy(
            required_scopes_by_operation={
                "write_note": {"connector:files:write"},
                "read_note": {"connector:files:read"},
                "notify": {"connector:notify:send"},
                "list_notifications": {"connector:notify:read"},
            },
            failure_threshold=3,
            recovery_timeout_seconds=30.0,
        ),
    )
    return registry
