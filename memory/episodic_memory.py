"""
Episodic memory for task history and learning in JARVIS AI OS.

Stores records of completed tasks (episodes), supports similarity search,
and extracts learnings and success patterns from historical episodes.

Usage::

    from memory.episodic_memory import EpisodicMemory, Episode

    em = EpisodicMemory(persist_path="data/episodes.json")
    ep = em.record_episode(
        task_description="Search for Python tutorials",
        actions_taken=["web_search", "url_fetch"],
        outcome="Found 5 tutorials on python.org",
        success=True,
        duration=2.3,
        learned_facts=["python.org has official tutorials"],
    )
    similar = em.get_similar_episodes("find Python learning resources")
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from infrastructure.logger import get_logger
from utils.exceptions import MemoryStorageError
from utils.helpers import generate_id, timestamp_now

logger = get_logger(__name__)

_DEFAULT_MAX_EPISODES = 10_000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """A recorded task execution episode.

    Attributes:
        id: Unique episode identifier.
        task_description: Natural-language description of the task attempted.
        actions_taken: Ordered list of skill/action names executed.
        outcome: Human-readable description of the result.
        success: Whether the task completed successfully.
        duration: Wall-clock execution time in seconds.
        learned_facts: Key facts or insights extracted from this episode.
        timestamp: ISO-8601 UTC timestamp when the episode was recorded.
        metadata: Arbitrary extra context (agent ID, model, error trace, etc.).
        error: Error message if the episode failed.
    """

    id: str
    task_description: str
    actions_taken: List[str]
    outcome: str
    success: bool
    duration: float = 0.0
    learned_facts: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=timestamp_now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Deserialise from a plain dict."""
        return cls(
            id=data.get("id", generate_id("ep")),
            task_description=data["task_description"],
            actions_taken=data.get("actions_taken", []),
            outcome=data.get("outcome", ""),
            success=bool(data.get("success", False)),
            duration=float(data.get("duration", 0.0)),
            learned_facts=data.get("learned_facts", []),
            timestamp=data.get("timestamp", timestamp_now()),
            metadata=data.get("metadata", {}),
            error=data.get("error"),
        )

    def get_keywords(self) -> List[str]:
        """Extract meaningful keywords from the task description."""
        text = self.task_description.lower()
        # Remove punctuation and split
        words = re.findall(r"\b[a-z]{3,}\b", text)
        # Exclude common stop words
        stopwords = {
            "the", "and", "for", "that", "with", "this", "from", "have",
            "are", "was", "were", "has", "had", "been", "will", "can",
            "could", "would", "should", "may", "might", "shall", "does",
            "did", "get", "got", "set", "let", "try", "use", "make",
            "take", "give", "show", "tell", "find", "look", "come", "go",
            "run", "put", "see", "know", "think", "want", "need", "call",
        }
        return [w for w in words if w not in stopwords]

    def similarity_score(self, query_keywords: List[str]) -> float:
        """Compute a simple Jaccard-like similarity against *query_keywords*.

        Returns a float in [0, 1].
        """
        if not query_keywords:
            return 0.0
        episode_keywords = set(self.get_keywords())
        query_set = set(query_keywords)
        if not episode_keywords and not query_set:
            return 1.0
        intersection = episode_keywords & query_set
        union = episode_keywords | query_set
        return len(intersection) / len(union) if union else 0.0

    def __repr__(self) -> str:
        snippet = self.task_description[:60]
        return (
            f"<Episode id={self.id!r} success={self.success} "
            f"duration={self.duration:.1f}s task={snippet!r}>"
        )


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------


class EpisodicMemory:
    """Stores and queries task history (episodes) with learning extraction.

    Args:
        persist_path: Optional JSON file path for persistence.
        max_episodes: Maximum episodes to keep; oldest are pruned first.
        auto_persist: Automatically save after each episode is recorded.
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        max_episodes: int = _DEFAULT_MAX_EPISODES,
        auto_persist: bool = True,
    ) -> None:
        self._episodes: List[Episode] = []
        self._lock = threading.RLock()
        self._max_episodes = max_episodes
        self._persist_path = persist_path
        self._auto_persist = auto_persist

        if persist_path:
            try:
                self.load_from_file(persist_path)
                logger.info(
                    "EpisodicMemory loaded %d episodes from '%s'.",
                    len(self._episodes),
                    persist_path,
                )
            except MemoryStorageError:
                logger.debug("No existing episode file at '%s'; starting fresh.", persist_path)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_episode(
        self,
        task_description: str,
        actions_taken: List[str],
        outcome: str,
        success: bool,
        duration: float = 0.0,
        learned_facts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Episode:
        """Record a new episode of task execution.

        Args:
            task_description: What the agent was trying to do.
            actions_taken: Ordered list of skill/action names used.
            outcome: Textual description of what happened.
            success: Whether the task succeeded.
            duration: Execution time in seconds.
            learned_facts: New facts or insights from this execution.
            metadata: Extra context (agent ID, model version, etc.).
            error: Error message for failed episodes.

        Returns:
            The created :class:`Episode` instance.
        """
        episode = Episode(
            id=generate_id("ep"),
            task_description=task_description.strip(),
            actions_taken=actions_taken or [],
            outcome=outcome.strip(),
            success=success,
            duration=max(0.0, duration),
            learned_facts=learned_facts or [],
            metadata=metadata or {},
            error=error,
        )

        with self._lock:
            self._episodes.append(episode)
            if len(self._episodes) > self._max_episodes:
                # Remove oldest episodes
                removed = len(self._episodes) - self._max_episodes
                self._episodes = self._episodes[removed:]
                logger.debug("EpisodicMemory: pruned %d old episodes.", removed)

        logger.debug(
            "EpisodicMemory: recorded episode %s (success=%s, duration=%.2fs).",
            episode.id,
            success,
            duration,
        )
        if self._auto_persist and self._persist_path:
            self._safe_persist()
        return episode

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Return a specific episode by ID."""
        with self._lock:
            for ep in self._episodes:
                if ep.id == episode_id:
                    return ep
        return None

    def get_recent(self, n: int = 10, success_only: bool = False) -> List[Episode]:
        """Return the *n* most recent episodes.

        Args:
            n: Number of episodes to return.
            success_only: If ``True``, return only successful episodes.

        Returns:
            Episodes ordered newest-first.
        """
        with self._lock:
            episodes = list(self._episodes)

        if success_only:
            episodes = [e for e in episodes if e.success]
        return list(reversed(episodes[-n:]))

    def search_episodes(
        self,
        query: str,
        max_results: int = 10,
        success_only: bool = False,
    ) -> List[Episode]:
        """Full-text search across task descriptions and outcomes.

        Args:
            query: Search text.
            max_results: Maximum results to return.
            success_only: Restrict to successful episodes.

        Returns:
            Matching episodes, most recent first.
        """
        query_lower = query.lower()
        with self._lock:
            episodes = list(self._episodes)

        results = []
        for ep in reversed(episodes):
            if success_only and not ep.success:
                continue
            searchable = (
                ep.task_description.lower()
                + " "
                + ep.outcome.lower()
                + " "
                + " ".join(ep.learned_facts).lower()
            )
            if query_lower in searchable:
                results.append(ep)
            if len(results) >= max_results:
                break
        return results

    def get_similar_episodes(
        self,
        task_description: str,
        max_results: int = 5,
        min_similarity: float = 0.1,
        success_only: bool = False,
    ) -> List[Tuple[float, Episode]]:
        """Find episodes with similar task descriptions using keyword matching.

        Args:
            task_description: The new task to find history for.
            max_results: Maximum episodes to return.
            min_similarity: Minimum Jaccard similarity score (0–1).
            success_only: Restrict to successful episodes.

        Returns:
            List of ``(similarity_score, Episode)`` tuples, sorted by score desc.
        """
        # Build query keywords
        query_ep = Episode(
            id="__query__",
            task_description=task_description,
            actions_taken=[],
            outcome="",
            success=True,
        )
        query_keywords = query_ep.get_keywords()

        scored: List[Tuple[float, Episode]] = []
        with self._lock:
            episodes = list(self._episodes)

        for ep in episodes:
            if success_only and not ep.success:
                continue
            score = ep.similarity_score(query_keywords)
            if score >= min_similarity:
                scored.append((score, ep))

        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[:max_results]

    # ------------------------------------------------------------------
    # Learning extraction
    # ------------------------------------------------------------------

    def extract_learnings(
        self,
        category_filter: Optional[str] = None,
        max_facts: int = 20,
    ) -> List[str]:
        """Aggregate unique learned facts across all (or filtered) episodes.

        Args:
            category_filter: If set, only use episodes whose metadata has
                ``category == category_filter``.
            max_facts: Maximum number of unique facts to return.

        Returns:
            Deduplicated list of fact strings, most frequently seen first.
        """
        fact_counts: Counter = Counter()

        with self._lock:
            episodes = list(self._episodes)

        for ep in episodes:
            if category_filter:
                if ep.metadata.get("category") != category_filter:
                    continue
            for fact in ep.learned_facts:
                fact = fact.strip()
                if fact:
                    fact_counts[fact] += 1

        return [fact for fact, _ in fact_counts.most_common(max_facts)]

    def get_success_patterns(
        self,
        min_occurrences: int = 2,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Identify action sequences that frequently lead to success.

        Analyses successful episodes to find the most common skill/action
        combinations used.

        Args:
            min_occurrences: Minimum times a sequence must appear to qualify.
            top_n: Maximum number of patterns to return.

        Returns:
            List of dicts with ``actions``, ``count``, and ``success_rate``.
        """
        # Count action sequences (as tuples) for success/failure
        success_seqs: Counter = Counter()
        failure_seqs: Counter = Counter()

        with self._lock:
            episodes = list(self._episodes)

        for ep in episodes:
            seq = tuple(ep.actions_taken)
            if not seq:
                continue
            if ep.success:
                success_seqs[seq] += 1
            else:
                failure_seqs[seq] += 1

        patterns = []
        for seq, count in success_seqs.most_common():
            if count < min_occurrences:
                continue
            total = count + failure_seqs.get(seq, 0)
            patterns.append({
                "actions": list(seq),
                "success_count": count,
                "total_count": total,
                "success_rate": round(count / total, 3),
            })
            if len(patterns) >= top_n:
                break

        return patterns

    def get_failure_patterns(
        self,
        min_occurrences: int = 2,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Identify action sequences associated with failures.

        Args:
            min_occurrences: Minimum failure count to qualify.
            top_n: Maximum patterns to return.

        Returns:
            List of dicts with ``actions``, ``failure_count``, and ``failure_rate``.
        """
        success_seqs: Counter = Counter()
        failure_seqs: Counter = Counter()

        with self._lock:
            episodes = list(self._episodes)

        for ep in episodes:
            seq = tuple(ep.actions_taken)
            if not seq:
                continue
            if ep.success:
                success_seqs[seq] += 1
            else:
                failure_seqs[seq] += 1

        patterns = []
        for seq, count in failure_seqs.most_common():
            if count < min_occurrences:
                continue
            total = count + success_seqs.get(seq, 0)
            patterns.append({
                "actions": list(seq),
                "failure_count": count,
                "total_count": total,
                "failure_rate": round(count / total, 3),
            })
            if len(patterns) >= top_n:
                break

        return patterns

    def get_agent_capability_success_rate(
        self,
        *,
        agent_id: str,
        capability: str,
        window: int = 300,
    ) -> float:
        """Return recent success rate for a specific (agent, capability) pair."""
        with self._lock:
            episodes = list(self._episodes)[-max(1, int(window)) :]

        matched = [
            ep
            for ep in episodes
            if ep.metadata.get("agent_id") == agent_id
            and ep.metadata.get("capability") == capability
        ]
        if not matched:
            return 0.5
        successes = sum(1 for ep in matched if ep.success)
        return round(successes / len(matched), 4)

    def recommend_actions_for_task(
        self,
        task_description: str,
        *,
        top_n: int = 3,
    ) -> List[str]:
        """Recommend actions using similar successful episodes."""
        similar = self.get_similar_episodes(
            task_description,
            max_results=max(10, top_n * 4),
            min_similarity=0.05,
            success_only=True,
        )
        if not similar:
            return []

        action_scores: Counter = Counter()
        for score, ep in similar:
            for action in ep.actions_taken:
                if action:
                    action_scores[action] += max(1, int(round(score * 10)))
        return [a for a, _ in action_scores.most_common(top_n)]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics across all episodes.

        Returns:
            Dict with success_rate, avg_duration, most_common_tasks, etc.
        """
        with self._lock:
            episodes = list(self._episodes)

        if not episodes:
            return {
                "total_episodes": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "most_common_tasks": [],
                "most_used_actions": [],
            }

        total = len(episodes)
        successful = sum(1 for e in episodes if e.success)
        durations = [e.duration for e in episodes]
        avg_duration = sum(durations) / total

        # Most common task keywords
        all_keywords: Counter = Counter()
        for ep in episodes:
            for kw in ep.get_keywords():
                all_keywords[kw] += 1

        # Most used actions
        action_counts: Counter = Counter()
        for ep in episodes:
            for action in ep.actions_taken:
                action_counts[action] += 1

        # Recent success rate (last 50 episodes)
        recent = episodes[-50:]
        recent_success_rate = sum(1 for e in recent if e.success) / len(recent)

        return {
            "total_episodes": total,
            "successful_episodes": successful,
            "failed_episodes": total - successful,
            "success_rate": round(successful / total, 3),
            "recent_success_rate": round(recent_success_rate, 3),
            "avg_duration": round(avg_duration, 3),
            "min_duration": round(min(durations), 3),
            "max_duration": round(max(durations), 3),
            "most_common_task_keywords": all_keywords.most_common(10),
            "most_used_actions": action_counts.most_common(10),
            "total_learned_facts": sum(len(e.learned_facts) for e in episodes),
        }

    def get_task_history(
        self,
        task_pattern: str,
        max_results: int = 20,
    ) -> List[Episode]:
        """Return episodes whose task description matches *task_pattern*.

        Args:
            task_pattern: Regex or substring to match against task descriptions.
            max_results: Maximum results to return.

        Returns:
            Matching episodes, newest first.
        """
        try:
            compiled = re.compile(task_pattern, re.IGNORECASE)
        except re.error:
            compiled = re.compile(re.escape(task_pattern), re.IGNORECASE)

        with self._lock:
            episodes = list(self._episodes)

        results = [ep for ep in reversed(episodes) if compiled.search(ep.task_description)]
        return results[:max_results]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_to_file(self, filepath: Optional[str] = None) -> None:
        """Write all episodes to a JSON file.

        Args:
            filepath: Destination path.  Falls back to *persist_path*.

        Raises:
            :exc:`~utils.exceptions.MemoryStorageError`: On I/O error.
        """
        path = filepath or self._persist_path
        if not path:
            raise ValueError("No persist_path configured and no filepath provided.")
        try:
            with self._lock:
                data = {
                    "version": "1.0",
                    "saved_at": timestamp_now(),
                    "episode_count": len(self._episodes),
                    "episodes": [e.to_dict() for e in self._episodes],
                }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            logger.info(
                "EpisodicMemory persisted %d episodes to '%s'.",
                data["episode_count"],
                path,
            )
        except OSError as exc:
            raise MemoryStorageError(operation="persist", reason=str(exc)) from exc

    def load_from_file(self, filepath: Optional[str] = None) -> int:
        """Load episodes from a JSON file.

        Merges with existing episodes (deduplicates by episode ID).

        Args:
            filepath: Source path.  Falls back to *persist_path*.

        Returns:
            Number of episodes loaded.

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
            existing_ids = {e.id for e in self._episodes}
            for ep_data in data.get("episodes", []):
                try:
                    ep = Episode.from_dict(ep_data)
                    if ep.id not in existing_ids:
                        self._episodes.append(ep)
                        existing_ids.add(ep.id)
                        loaded += 1
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("EpisodicMemory: skipping malformed episode: %s", exc)

            # Sort by timestamp after load
            self._episodes.sort(key=lambda e: e.timestamp)

        return loaded

    def _safe_persist(self) -> None:
        """Persist without raising; log errors instead."""
        try:
            self.persist_to_file()
        except (MemoryStorageError, ValueError) as exc:
            logger.warning("EpisodicMemory auto-persist failed: %s", exc)

    def clear(self) -> None:
        """Remove all episodes from memory (does not delete the persist file)."""
        with self._lock:
            count = len(self._episodes)
            self._episodes.clear()
        logger.info("EpisodicMemory: cleared %d episodes.", count)

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"<EpisodicMemory episodes={stats['total_episodes']} "
            f"success_rate={stats['success_rate']:.1%}>"
        )


__all__ = [
    "Episode",
    "EpisodicMemory",
]
