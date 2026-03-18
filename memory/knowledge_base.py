"""
Long-term persistent knowledge storage for JARVIS AI OS.

The :class:`KnowledgeBase` provides an in-memory key/value store with
optional JSON-file persistence.  Entries can be tagged, categorised, and
set to expire via a TTL.

Usage::

    from memory.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(persist_path="data/knowledge.json")
    kb.store("weather_paris", "18°C sunny", category="facts", tags=["weather", "paris"])
    entry = kb.retrieve("weather_paris")
    print(entry.value)
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from infrastructure.logger import get_logger
from utils.exceptions import MemoryStorageError
from utils.helpers import generate_id, timestamp_now

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeEntry:
    """A single knowledge base record.

    Attributes:
        id: Unique entry identifier.
        key: Human-readable lookup key (unique within the KB).
        value: The stored value (any JSON-serialisable type).
        category: Logical grouping (e.g. ``"facts"``, ``"preferences"``).
        tags: List of searchable tag strings.
        created_at: ISO-8601 timestamp when the entry was first stored.
        updated_at: ISO-8601 timestamp of the last update.
        access_count: Number of times this entry has been retrieved.
        ttl_seconds: Optional TTL in seconds. ``None`` means no expiry.
        expires_at: Unix epoch time when the entry expires (computed from TTL).
    """

    id: str
    key: str
    value: Any
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=timestamp_now)
    updated_at: str = field(default_factory=timestamp_now)
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    expires_at: Optional[float] = None  # Unix epoch float

    def is_expired(self) -> bool:
        """Return ``True`` if the entry has passed its expiry time."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        """Deserialise from a plain dict."""
        return cls(
            id=data.get("id", generate_id("ke")),
            key=data["key"],
            value=data["value"],
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            created_at=data.get("created_at", timestamp_now()),
            updated_at=data.get("updated_at", timestamp_now()),
            access_count=data.get("access_count", 0),
            ttl_seconds=data.get("ttl_seconds"),
            expires_at=data.get("expires_at"),
        )

    def __repr__(self) -> str:
        val_preview = str(self.value)[:60]
        return (
            f"<KnowledgeEntry key={self.key!r} category={self.category!r} "
            f"value={val_preview!r} access={self.access_count}>"
        )


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """In-memory knowledge store with optional JSON-file persistence and TTL.

    Args:
        persist_path: If provided, the KB is loaded from this file on init
            and auto-saved on every mutation.
        auto_persist: Whether to write to *persist_path* after every change.
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        auto_persist: bool = True,
    ) -> None:
        self._entries: Dict[str, KnowledgeEntry] = {}  # key -> entry
        self._lock = threading.RLock()
        self._persist_path = persist_path
        self._auto_persist = auto_persist

        if persist_path:
            try:
                self.load_from_file(persist_path)
                logger.info("KnowledgeBase loaded from '%s'.", persist_path)
            except MemoryStorageError:
                logger.debug("No existing KB file at '%s'; starting fresh.", persist_path)

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def store(
        self,
        key: str,
        value: Any,
        category: str = "general",
        tags: Optional[List[str]] = None,
        ttl_seconds: Optional[float] = None,
        overwrite: bool = True,
    ) -> KnowledgeEntry:
        """Store a key-value pair in the knowledge base.

        If an entry with *key* already exists and *overwrite* is ``True``,
        the value, category, and tags are updated; the ``created_at``
        timestamp and ``access_count`` are preserved.

        Args:
            key: Unique lookup key.
            value: Any JSON-serialisable value.
            category: Logical category string.
            tags: Optional list of tag strings for search.
            ttl_seconds: Optional TTL in seconds; entry expires after this.
            overwrite: Replace existing entry when ``True`` (default).

        Returns:
            The created or updated :class:`KnowledgeEntry`.

        Raises:
            ValueError: If *key* exists and *overwrite* is ``False``.
        """
        if not key.strip():
            raise ValueError("Knowledge key must not be empty.")

        expires_at: Optional[float] = None
        if ttl_seconds is not None:
            expires_at = time.time() + ttl_seconds

        with self._lock:
            existing = self._entries.get(key)
            if existing is not None and not overwrite:
                raise ValueError(f"Key '{key}' already exists. Use overwrite=True.")

            if existing is not None:
                existing.value = value
                existing.category = category
                existing.tags = tags or []
                existing.updated_at = timestamp_now()
                existing.ttl_seconds = ttl_seconds
                existing.expires_at = expires_at
                entry = existing
            else:
                entry = KnowledgeEntry(
                    id=generate_id("ke"),
                    key=key,
                    value=value,
                    category=category,
                    tags=tags or [],
                    ttl_seconds=ttl_seconds,
                    expires_at=expires_at,
                )
                self._entries[key] = entry

        logger.debug("KnowledgeBase: stored key='%s' category='%s'.", key, category)
        if self._auto_persist and self._persist_path:
            self._safe_persist()
        return entry

    def retrieve(self, key: str) -> Optional[KnowledgeEntry]:
        """Retrieve an entry by key, incrementing its access count.

        Returns ``None`` when the key is absent or the entry has expired.
        Expired entries are removed automatically.

        Args:
            key: The lookup key.

        Returns:
            A :class:`KnowledgeEntry` or ``None``.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._entries[key]
                logger.debug("KnowledgeBase: entry '%s' expired and removed.", key)
                return None
            entry.access_count += 1
            entry.updated_at = timestamp_now()
        return entry

    def update(
        self,
        key: str,
        value: Any,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        ttl_seconds: Optional[float] = None,
    ) -> KnowledgeEntry:
        """Update an existing entry.

        Args:
            key: Key of the entry to update.
            value: New value.
            category: New category (unchanged if ``None``).
            tags: New tags list (unchanged if ``None``).
            ttl_seconds: New TTL (unchanged if ``None``).

        Returns:
            The updated :class:`KnowledgeEntry`.

        Raises:
            KeyError: If *key* does not exist in the KB.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise KeyError(f"Knowledge key '{key}' not found.")
            entry.value = value
            entry.updated_at = timestamp_now()
            if category is not None:
                entry.category = category
            if tags is not None:
                entry.tags = tags
            if ttl_seconds is not None:
                entry.ttl_seconds = ttl_seconds
                entry.expires_at = time.time() + ttl_seconds

        if self._auto_persist and self._persist_path:
            self._safe_persist()
        return entry

    def delete(self, key: str) -> bool:
        """Remove an entry by key.

        Returns:
            ``True`` if the entry existed and was removed.
        """
        with self._lock:
            existed = key in self._entries
            self._entries.pop(key, None)

        if existed:
            logger.debug("KnowledgeBase: deleted key='%s'.", key)
            if self._auto_persist and self._persist_path:
                self._safe_persist()
        return existed

    # ------------------------------------------------------------------
    # Search and listing
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        case_sensitive: bool = False,
        max_results: int = 20,
        search_values: bool = True,
    ) -> List[KnowledgeEntry]:
        """Keyword search across keys, values, and tags.

        Args:
            query: Search term.
            case_sensitive: Whether to use case-sensitive matching.
            max_results: Maximum entries to return.
            search_values: Whether to include the value string in the search.

        Returns:
            Matching entries, sorted by access count (descending).
        """
        compare = lambda s: s if case_sensitive else s.lower()  # noqa: E731
        cq = compare(query)

        results: List[KnowledgeEntry] = []
        with self._lock:
            for entry in self._entries.values():
                if entry.is_expired():
                    continue
                haystack = compare(entry.key) + " " + compare(entry.category)
                haystack += " " + " ".join(compare(t) for t in entry.tags)
                if search_values:
                    haystack += " " + compare(str(entry.value))
                if cq in haystack:
                    results.append(entry)

        results.sort(key=lambda e: e.access_count, reverse=True)
        return results[:max_results]

    def search_semantic(
        self,
        query: str,
        *,
        max_results: int = 10,
        min_score: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Return relevance-ranked matches using lexical similarity heuristics.

        This is a lightweight fallback to provide semantic-ish ranking without
        external embedding services.
        """
        if not query.strip():
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scored: List[tuple[float, KnowledgeEntry]] = []
        with self._lock:
            entries = list(self._entries.values())

        now = time.time()
        for entry in entries:
            if entry.is_expired():
                continue

            text = " ".join(
                [
                    entry.key,
                    entry.category,
                    " ".join(entry.tags),
                    str(entry.value),
                ]
            )
            entry_terms = self._tokenize(text)
            if not entry_terms:
                continue

            overlap = len(query_terms & entry_terms) / max(1, len(query_terms | entry_terms))
            contains_boost = 0.15 if query.lower() in text.lower() else 0.0
            access_boost = min(0.1, entry.access_count * 0.01)
            age_seconds = max(0.0, now - self._to_epoch(entry.updated_at))
            recency_boost = max(0.0, 0.1 - min(0.1, age_seconds / (86400 * 30)))
            score = overlap + contains_boost + access_boost + recency_boost

            if score >= min_score:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:max_results]
        return [
            {
                "score": round(score, 4),
                "key": entry.key,
                "category": entry.category,
                "value": entry.value,
                "tags": entry.tags,
                "updated_at": entry.updated_at,
                "access_count": entry.access_count,
            }
            for score, entry in top
        ]

    def list_by_category(self, category: str) -> List[KnowledgeEntry]:
        """Return all non-expired entries in *category*."""
        with self._lock:
            return [
                e for e in self._entries.values()
                if e.category == category and not e.is_expired()
            ]

    def list_by_tags(
        self,
        tags: List[str],
        require_all: bool = False,
    ) -> List[KnowledgeEntry]:
        """Return entries matching the given tags.

        Args:
            tags: List of tag strings to match.
            require_all: When ``True``, entry must have all tags.
                When ``False`` (default), any tag match suffices.

        Returns:
            Matching non-expired entries.
        """
        tag_set = set(tags)
        with self._lock:
            results = []
            for entry in self._entries.values():
                if entry.is_expired():
                    continue
                entry_tags = set(entry.tags)
                if require_all:
                    if tag_set.issubset(entry_tags):
                        results.append(entry)
                else:
                    if tag_set & entry_tags:
                        results.append(entry)
        return results

    def get_recent(self, n: int = 10) -> List[KnowledgeEntry]:
        """Return the *n* most recently updated entries."""
        with self._lock:
            valid = [e for e in self._entries.values() if not e.is_expired()]
        valid.sort(key=lambda e: e.updated_at, reverse=True)
        return valid[:n]

    def get_most_accessed(self, n: int = 10) -> List[KnowledgeEntry]:
        """Return the *n* most frequently retrieved entries."""
        with self._lock:
            valid = [e for e in self._entries.values() if not e.is_expired()]
        valid.sort(key=lambda e: e.access_count, reverse=True)
        return valid[:n]

    def get_all_categories(self) -> List[str]:
        """Return a sorted list of unique category names."""
        with self._lock:
            return sorted({e.category for e in self._entries.values() if not e.is_expired()})

    def get_all_tags(self) -> List[str]:
        """Return a sorted list of all unique tags across all entries."""
        tags: set = set()
        with self._lock:
            for entry in self._entries.values():
                if not entry.is_expired():
                    tags.update(entry.tags)
        return sorted(tags)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def purge_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            expired_keys = [k for k, e in self._entries.items() if e.is_expired()]
            for k in expired_keys:
                del self._entries[k]
        if expired_keys:
            logger.info("KnowledgeBase: purged %d expired entries.", len(expired_keys))
            if self._auto_persist and self._persist_path:
                self._safe_persist()
        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the knowledge base."""
        with self._lock:
            all_entries = list(self._entries.values())

        valid = [e for e in all_entries if not e.is_expired()]
        expired = [e for e in all_entries if e.is_expired()]
        categories: Dict[str, int] = {}
        total_accesses = 0
        for e in valid:
            categories[e.category] = categories.get(e.category, 0) + 1
            total_accesses += e.access_count

        return {
            "total_entries": len(valid),
            "expired_entries": len(expired),
            "categories": categories,
            "total_accesses": total_accesses,
            "has_persistence": self._persist_path is not None,
            "persist_path": self._persist_path,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_to_file(self, filepath: Optional[str] = None) -> None:
        """Write the KB to a JSON file.

        Args:
            filepath: Destination path. Falls back to *persist_path*.

        Raises:
            :exc:`~utils.exceptions.MemoryStorageError`: On I/O error.
            ValueError: If no filepath is provided or configured.
        """
        path = filepath or self._persist_path
        if not path:
            raise ValueError("No persist_path configured and no filepath provided.")
        try:
            with self._lock:
                data = {
                    "version": "1.0",
                    "saved_at": timestamp_now(),
                    "entries": [e.to_dict() for e in self._entries.values()],
                }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            logger.info("KnowledgeBase persisted to '%s' (%d entries).", path, len(data["entries"]))
        except OSError as exc:
            raise MemoryStorageError(operation="persist", reason=str(exc)) from exc

    def load_from_file(self, filepath: Optional[str] = None) -> int:
        """Load entries from a JSON file, merging with existing data.

        Args:
            filepath: Source path. Falls back to *persist_path*.

        Returns:
            Number of entries loaded.

        Raises:
            :exc:`~utils.exceptions.MemoryStorageError`: On I/O or parse error.
        """
        path = filepath or self._persist_path
        if not path:
            raise ValueError("No persist_path configured and no filepath provided.")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError as exc:
            raise MemoryStorageError(operation="load", reason=f"File not found: {path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryStorageError(operation="load", reason=str(exc)) from exc

        loaded = 0
        with self._lock:
            for entry_data in data.get("entries", []):
                try:
                    entry = KnowledgeEntry.from_dict(entry_data)
                    if not entry.is_expired():
                        self._entries[entry.key] = entry
                        loaded += 1
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("KnowledgeBase: skipping malformed entry: %s", exc)
        return loaded

    def _safe_persist(self) -> None:
        """Persist without raising; log errors instead."""
        try:
            self.persist_to_file()
        except (MemoryStorageError, ValueError) as exc:
            logger.warning("KnowledgeBase auto-persist failed: %s", exc)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())}

    @staticmethod
    def _to_epoch(iso_timestamp: str) -> float:
        try:
            from datetime import datetime

            return datetime.fromisoformat(iso_timestamp).timestamp()
        except Exception:
            return time.time()

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries.values() if not e.is_expired())

    def __contains__(self, key: str) -> bool:
        entry = self._entries.get(key)
        return entry is not None and not entry.is_expired()

    def __repr__(self) -> str:
        return f"<KnowledgeBase entries={len(self)} persist={self._persist_path!r}>"


__all__ = [
    "KnowledgeEntry",
    "KnowledgeBase",
]
