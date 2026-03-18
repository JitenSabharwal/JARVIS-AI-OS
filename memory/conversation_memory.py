"""
Short-term conversation memory for JARVIS AI OS.

Stores the messages exchanged during a session, provides a sliding context
window for LLM prompts, and supports per-session isolation.

Usage::

    from memory.conversation_memory import ConversationMemory

    mem = ConversationMemory(session_id="user-42")
    mem.add_message("user", "What is the weather in Paris?")
    mem.add_message("assistant", "It is 18°C and sunny.")
    context = mem.get_context_window(max_messages=10)
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from infrastructure.logger import get_logger
from utils.exceptions import MemoryStorageError
from utils.helpers import generate_id, timestamp_now

logger = get_logger(__name__)

# Default maximum messages per session before automatic pruning.
_DEFAULT_MAX_MESSAGES = 500
# How many messages to keep after pruning (keep the most recent).
_DEFAULT_PRUNE_TARGET = 400


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single conversation message.

    Attributes:
        role: Sender role, typically ``"user"``, ``"assistant"``, or
              ``"system"``.
        content: The textual content of the message.
        timestamp: ISO-8601 UTC timestamp string.
        metadata: Arbitrary key-value pairs (model name, token count, etc.).
        message_id: Unique identifier for this message.
    """

    role: str
    content: str
    timestamp: str = field(default_factory=timestamp_now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: generate_id("msg"))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserialise from a plain dict."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", timestamp_now()),
            metadata=data.get("metadata", {}),
            message_id=data.get("message_id", generate_id("msg")),
        )

    def __repr__(self) -> str:
        snippet = self.content[:60].replace("\n", " ")
        return f"<Message role={self.role!r} id={self.message_id!r} content={snippet!r}>"


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


class ConversationMemory:
    """In-memory conversation history with optional file persistence.

    Each :class:`ConversationMemory` instance represents one conversation
    session.  Multiple sessions can run concurrently without interference.

    Args:
        session_id: Unique identifier for this conversation session.
        max_messages: Maximum messages to retain before pruning.
        prune_target: Number of messages to keep after pruning.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_messages: int = _DEFAULT_MAX_MESSAGES,
        prune_target: int = _DEFAULT_PRUNE_TARGET,
    ) -> None:
        self.session_id: str = session_id or generate_id("session")
        self._max_messages = max_messages
        self._prune_target = min(prune_target, max_messages)
        self._messages: List[Message] = []
        self._lock = threading.RLock()
        logger.debug("ConversationMemory created for session '%s'.", self.session_id)

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Append a new message to the conversation.

        Args:
            role: Sender role (``"user"``, ``"assistant"``, ``"system"``).
            content: Message text.
            metadata: Optional extra context (model, tokens, latency, etc.).

        Returns:
            The created :class:`Message` instance.
        """
        if not content.strip():
            raise ValueError("Message content must not be empty.")
        if not role.strip():
            raise ValueError("Message role must not be empty.")

        msg = Message(role=role.strip(), content=content, metadata=metadata or {})
        with self._lock:
            self._messages.append(msg)
            if len(self._messages) > self._max_messages:
                self._prune()
        logger.debug(
            "Session '%s': added %s message (id=%s).",
            self.session_id,
            role,
            msg.message_id,
        )
        return msg

    def get_history(
        self,
        roles: Optional[List[str]] = None,
        since: Optional[str] = None,
    ) -> List[Message]:
        """Return the full message history, optionally filtered.

        Args:
            roles: If provided, return only messages with matching roles.
            since: ISO-8601 timestamp; return only messages after this time.

        Returns:
            List of :class:`Message` objects (oldest first).
        """
        with self._lock:
            messages = list(self._messages)

        if roles:
            role_set = {r.lower() for r in roles}
            messages = [m for m in messages if m.role.lower() in role_set]

        if since:
            messages = [m for m in messages if m.timestamp >= since]

        return messages

    def get_context_window(
        self,
        max_messages: int = 20,
        include_system: bool = True,
    ) -> List[Dict[str, str]]:
        """Return the last *max_messages* messages as dicts for LLM context.

        System messages are always prepended when *include_system* is ``True``.

        Args:
            max_messages: Maximum number of messages in the window.
            include_system: Whether to include system-role messages.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        with self._lock:
            all_msgs = list(self._messages)

        if include_system:
            system_msgs = [m for m in all_msgs if m.role.lower() == "system"]
            non_system = [m for m in all_msgs if m.role.lower() != "system"]
        else:
            system_msgs = []
            non_system = [m for m in all_msgs]

        # Take the most recent non-system messages
        recent = non_system[-max_messages:] if max_messages else non_system

        window = system_msgs + recent
        return [{"role": m.role, "content": m.content} for m in window]

    def clear(self) -> None:
        """Remove all messages from this session."""
        with self._lock:
            count = len(self._messages)
            self._messages.clear()
        logger.info("Session '%s': cleared %d messages.", self.session_id, count)

    def delete_message(self, message_id: str) -> bool:
        """Delete a specific message by its ID.

        Returns:
            ``True`` if the message was found and removed, ``False`` otherwise.
        """
        with self._lock:
            before = len(self._messages)
            self._messages = [m for m in self._messages if m.message_id != message_id]
            return len(self._messages) < before

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        case_sensitive: bool = False,
        max_results: int = 20,
    ) -> List[Message]:
        """Search messages by content substring.

        Args:
            query: Text to search for in message content.
            case_sensitive: Whether the search is case-sensitive.
            max_results: Maximum number of matching messages to return.

        Returns:
            List of matching :class:`Message` objects (most recent first).
        """
        if not query:
            return []

        compare_query = query if case_sensitive else query.lower()
        results = []
        with self._lock:
            for msg in reversed(self._messages):
                compare_content = msg.content if case_sensitive else msg.content.lower()
                if compare_query in compare_content:
                    results.append(msg)
                    if len(results) >= max_results:
                        break
        return results

    # ------------------------------------------------------------------
    # Summarisation (extractive)
    # ------------------------------------------------------------------

    def summarize(
        self,
        max_sentences: int = 5,
        roles: Optional[List[str]] = None,
    ) -> str:
        """Produce a simple extractive summary of the conversation.

        Selects the longest sentences from the conversation (a heuristic for
        informativeness) and returns them joined as a paragraph.

        Args:
            max_sentences: Maximum number of sentences in the summary.
            roles: If given, only consider messages from these roles.

        Returns:
            A plain-text summary string.
        """
        messages = self.get_history(roles=roles)
        if not messages:
            return "No messages in this conversation."

        # Collect all sentences (crude split on '.', '!', '?')
        import re
        sentences: List[str] = []
        for msg in messages:
            for s in re.split(r"(?<=[.!?])\s+", msg.content):
                s = s.strip()
                if len(s) > 20:
                    sentences.append(s)

        if not sentences:
            # Fallback: just truncate the last user message
            last = messages[-1]
            return f"Last message ({last.role}): {last.content[:200]}"

        # Rank by length (longer = more informative heuristic)
        ranked = sorted(sentences, key=len, reverse=True)
        # Take top sentences but keep original order
        top_set = set(ranked[:max_sentences])
        ordered = [s for s in sentences if s in top_set]
        seen: set = set()
        unique_ordered = []
        for s in ordered:
            if s not in seen:
                seen.add(s)
                unique_ordered.append(s)

        return " ".join(unique_ordered[:max_sentences])

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return metadata and usage statistics for this session."""
        with self._lock:
            messages = list(self._messages)

        role_counts: Dict[str, int] = {}
        total_chars = 0
        for msg in messages:
            role_counts[msg.role] = role_counts.get(msg.role, 0) + 1
            total_chars += len(msg.content)

        return {
            "session_id": self.session_id,
            "message_count": len(messages),
            "total_chars": total_chars,
            "role_breakdown": role_counts,
            "oldest_message": messages[0].timestamp if messages else None,
            "newest_message": messages[-1].timestamp if messages else None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def export_to_dict(self) -> Dict[str, Any]:
        """Export the entire session to a serialisable dict."""
        with self._lock:
            messages = [m.to_dict() for m in self._messages]
        return {
            "session_id": self.session_id,
            "exported_at": timestamp_now(),
            "message_count": len(messages),
            "messages": messages,
        }

    def export_to_json(self, indent: int = 2) -> str:
        """Serialise the session to a JSON string."""
        return json.dumps(self.export_to_dict(), indent=indent, default=str)

    @classmethod
    def import_from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        """Reconstruct a :class:`ConversationMemory` from an exported dict.

        Args:
            data: Dict as returned by :meth:`export_to_dict`.

        Returns:
            A new :class:`ConversationMemory` instance.

        Raises:
            :exc:`~utils.exceptions.MemoryStorageError`: On malformed input.
        """
        try:
            mem = cls(session_id=data.get("session_id"))
            for msg_data in data.get("messages", []):
                msg = Message.from_dict(msg_data)
                mem._messages.append(msg)
            return mem
        except (KeyError, TypeError, ValueError) as exc:
            raise MemoryStorageError(
                operation="import",
                reason=str(exc),
            ) from exc

    @classmethod
    def import_from_json(cls, json_str: str) -> "ConversationMemory":
        """Deserialise from a JSON string produced by :meth:`export_to_json`."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise MemoryStorageError(operation="import", reason=f"Invalid JSON: {exc}") from exc
        return cls.import_from_dict(data)

    def save_to_file(self, filepath: str) -> None:
        """Persist the session to a JSON file.

        Args:
            filepath: Destination file path.

        Raises:
            :exc:`~utils.exceptions.MemoryStorageError`: On I/O error.
        """
        try:
            with open(filepath, "w", encoding="utf-8") as fh:
                fh.write(self.export_to_json())
            logger.info("Session '%s' saved to '%s'.", self.session_id, filepath)
        except OSError as exc:
            raise MemoryStorageError(operation="save", reason=str(exc)) from exc

    @classmethod
    def load_from_file(cls, filepath: str) -> "ConversationMemory":
        """Load a session from a JSON file.

        Args:
            filepath: Path to a file written by :meth:`save_to_file`.

        Returns:
            A new :class:`ConversationMemory` instance.

        Raises:
            :exc:`~utils.exceptions.MemoryStorageError`: On I/O or parse error.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                json_str = fh.read()
        except OSError as exc:
            raise MemoryStorageError(operation="load", reason=str(exc)) from exc
        return cls.import_from_json(json_str)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Trim message list to _prune_target, keeping system messages and most recent."""
        system_msgs = [m for m in self._messages if m.role.lower() == "system"]
        non_system = [m for m in self._messages if m.role.lower() != "system"]

        keep_target = max(0, self._prune_target - len(system_msgs))
        trimmed_non_system = non_system[-keep_target:] if keep_target else []
        removed = len(self._messages) - len(system_msgs) - len(trimmed_non_system)

        self._messages = system_msgs + trimmed_non_system
        logger.debug(
            "Session '%s': pruned %d messages (kept %d).",
            self.session_id,
            removed,
            len(self._messages),
        )

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"<ConversationMemory session_id={self.session_id!r} "
            f"messages={len(self._messages)}>"
        )


# ---------------------------------------------------------------------------
# Session manager (multi-session support)
# ---------------------------------------------------------------------------


class SessionManager:
    """Thread-safe registry of active :class:`ConversationMemory` sessions.

    Usage::

        manager = SessionManager()
        mem = manager.get_or_create("user-123")
        mem.add_message("user", "Hello JARVIS")
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, ConversationMemory] = {}
        self._lock = threading.RLock()

    def get_or_create(
        self,
        session_id: str,
        max_messages: int = _DEFAULT_MAX_MESSAGES,
    ) -> ConversationMemory:
        """Return an existing session or create a new one."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationMemory(
                    session_id=session_id,
                    max_messages=max_messages,
                )
            return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[ConversationMemory]:
        """Return session by ID, or ``None`` if not found."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Remove a session. Returns ``True`` if found and deleted."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> List[str]:
        """Return all active session IDs."""
        return list(self._sessions.keys())

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Return stats for all sessions."""
        return [session.get_stats() for session in self._sessions.values()]

    def __len__(self) -> int:
        return len(self._sessions)


__all__ = [
    "Message",
    "ConversationMemory",
    "SessionManager",
]
