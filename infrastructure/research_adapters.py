"""
Research source adapter abstractions.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseResearchAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def description(self) -> str:
        return ""

    @abstractmethod
    def fetch_sources(self, *, topic: str, max_items: int = 10) -> List[Dict[str, Any]]:
        pass


class StaticResearchAdapter(BaseResearchAdapter):
    """Simple deterministic adapter for tests/local demos."""

    def __init__(self, *, name: str = "static", items: Optional[List[Dict[str, Any]]] = None) -> None:
        self._name = name
        self._items = list(items or [])

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Static in-memory research source adapter"

    def fetch_sources(self, *, topic: str, max_items: int = 10) -> List[Dict[str, Any]]:
        top = topic.strip().lower()
        out = []
        for item in self._items:
            itopic = str(item.get("topic", "")).strip().lower()
            text = f"{item.get('title', '')} {item.get('content', '')}".lower()
            if top and top not in itopic and top not in text:
                continue
            out.append(dict(item))
            if len(out) >= max(1, int(max_items)):
                break
        return out


class DuckDuckGoAdapter(BaseResearchAdapter):
    """
    Lightweight no-key adapter using DuckDuckGo Instant Answer API.
    """

    def __init__(self, *, name: str = "duckduckgo", timeout_seconds: int = 12) -> None:
        self._name = name
        self._timeout = max(1, int(timeout_seconds))

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "DuckDuckGo Instant Answer adapter"

    def fetch_sources(self, *, topic: str, max_items: int = 10) -> List[Dict[str, Any]]:
        query = topic.strip()
        if not query:
            return []
        encoded = urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        )
        url = f"https://api.duckduckgo.com/?{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "JARVIS-AI-OS/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
            data = json.loads(body)
        except Exception:  # noqa: BLE001
            return []

        items: List[Dict[str, Any]] = []
        abstract = str(data.get("AbstractText", "") or "").strip()
        abstract_url = str(data.get("AbstractURL", "") or "").strip()
        if abstract and abstract_url:
            items.append(
                {
                    "title": f"{query} — abstract",
                    "url": abstract_url,
                    "content": abstract,
                    "topic": query,
                    "source_type": "news",
                    "metadata": {"adapter": self.name},
                }
            )
        for topic_item in data.get("RelatedTopics", []) or []:
            if not isinstance(topic_item, dict):
                continue
            text = str(topic_item.get("Text", "") or "").strip()
            t_url = str(topic_item.get("FirstURL", "") or "").strip()
            if not text or not t_url:
                continue
            items.append(
                {
                    "title": text[:120],
                    "url": t_url,
                    "content": text,
                    "topic": query,
                    "source_type": "blog",
                    "metadata": {"adapter": self.name},
                }
            )
            if len(items) >= max(1, int(max_items)):
                break
        return items[: max(1, int(max_items))]
