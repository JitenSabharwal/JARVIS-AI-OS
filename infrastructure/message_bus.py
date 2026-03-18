"""
Async inter-agent message bus for JARVIS AI OS.

Provides pub/sub, request/reply, and broadcast messaging with dead-letter queue,
audit logging, and delivery metrics — all in-memory with no external broker.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Awaitable

from infrastructure.logger import get_logger

logger = get_logger("message_bus")


# ---------------------------------------------------------------------------
# Enums & core data structures
# ---------------------------------------------------------------------------

class MessagePriority(IntEnum):
    """Delivery priority — higher values are processed first."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class MessageEnvelope:
    """Wrapper that carries a message through the bus."""
    id: str
    topic: str
    payload: Any
    sender: str
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: str | None = None          # for request/reply pairing
    reply_to: str | None = None                # one-shot reply topic
    timestamp: float = field(default_factory=time.time)
    ttl: float | None = None                   # seconds; None = no expiry
    headers: dict[str, str] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Return True if the message has lived longer than its TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl

    @classmethod
    def create(
        cls,
        topic: str,
        payload: Any,
        sender: str = "system",
        *,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        ttl: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> "MessageEnvelope":
        return cls(
            id=str(uuid.uuid4()),
            topic=topic,
            payload=payload,
            sender=sender,
            priority=priority,
            correlation_id=correlation_id,
            reply_to=reply_to,
            ttl=ttl,
            headers=headers or {},
        )


# Handler type: async callable that receives an envelope
MessageHandler = Callable[[MessageEnvelope], Awaitable[None]]
FilterFn = Callable[[MessageEnvelope], bool]


@dataclass
class Subscription:
    """Represents a single topic subscription."""
    id: str
    topic: str
    handler: MessageHandler
    filter_fn: FilterFn | None = None          # optional message filter
    subscriber_id: str = "anonymous"

    def matches(self, envelope: MessageEnvelope) -> bool:
        """Return True when this subscription should receive *envelope*."""
        if self.filter_fn is None:
            return True
        try:
            return bool(self.filter_fn(envelope))
        except Exception:
            return True                         # fail-open: deliver anyway


@dataclass
class DeadLetter:
    """A message that could not be delivered."""
    envelope: MessageEnvelope
    reason: str
    failed_at: float = field(default_factory=time.time)
    subscription_id: str | None = None


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------

class MessageBus:
    """
    In-memory async message bus supporting:
    - Topic-based pub/sub
    - Request/reply (RPC-style with timeout)
    - Broadcast to all subscribers
    - Dead-letter queue for failed deliveries
    - Bounded audit log
    - Delivery metrics
    """

    _MAX_HISTORY = 1_000          # max audit-log entries kept
    _DLQ_MAX = 500                # max dead-letter entries

    def __init__(self, max_history: int = _MAX_HISTORY) -> None:
        # topic -> list[Subscription]
        self._subscriptions: dict[str, list[Subscription]] = {}
        self._sub_index: dict[str, Subscription] = {}       # id -> Subscription

        # Pending reply futures keyed by correlation_id
        self._pending_replies: dict[str, asyncio.Future[MessageEnvelope]] = {}

        self._dead_letters: deque[DeadLetter] = deque(maxlen=self._DLQ_MAX)
        self._history: deque[MessageEnvelope] = deque(maxlen=max_history)

        # Metrics
        self._metrics: dict[str, int] = {
            "messages_sent": 0,
            "messages_received": 0,
            "failed_deliveries": 0,
        }

        self._running = False
        self._lock = asyncio.Lock()
        logger.debug("MessageBus created (in-memory)")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Mark the bus as active."""
        self._running = True
        logger.info("MessageBus started")

    async def stop(self) -> None:
        """Drain pending reply futures and shut down."""
        self._running = False
        # Cancel all outstanding request/reply waiters
        for fut in self._pending_replies.values():
            if not fut.done():
                fut.cancel()
        self._pending_replies.clear()
        logger.info("MessageBus stopped")

    # ------------------------------------------------------------------
    # Subscribe / Unsubscribe
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        *,
        filter_fn: FilterFn | None = None,
        subscriber_id: str = "anonymous",
    ) -> str:
        """
        Register *handler* for *topic*.

        Returns the subscription ID that can be used to unsubscribe.
        Supports glob-style wildcards: ``agent.*`` or ``*``.
        """
        sub = Subscription(
            id=str(uuid.uuid4()),
            topic=topic,
            handler=handler,
            filter_fn=filter_fn,
            subscriber_id=subscriber_id,
        )
        async with self._lock:
            self._subscriptions.setdefault(topic, []).append(sub)
            self._sub_index[sub.id] = sub
        logger.debug("Subscribed %s to topic '%s' (sub_id=%s)", subscriber_id, topic, sub.id)
        return sub.id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription by its ID. Returns True if found."""
        async with self._lock:
            sub = self._sub_index.pop(subscription_id, None)
            if sub is None:
                return False
            subs = self._subscriptions.get(sub.topic, [])
            self._subscriptions[sub.topic] = [s for s in subs if s.id != subscription_id]
        logger.debug("Unsubscribed sub_id=%s", subscription_id)
        return True

    # ------------------------------------------------------------------
    # Publish / Broadcast
    # ------------------------------------------------------------------

    async def publish(
        self,
        topic: str,
        payload: Any,
        *,
        sender: str = "system",
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        ttl: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> MessageEnvelope:
        """
        Publish a message on *topic* and deliver to all matching subscribers.

        Returns the envelope so the caller can inspect its ID / correlation_id.
        """
        envelope = MessageEnvelope.create(
            topic=topic,
            payload=payload,
            sender=sender,
            priority=priority,
            correlation_id=correlation_id,
            reply_to=reply_to,
            ttl=ttl,
            headers=headers,
        )
        self._metrics["messages_sent"] += 1
        self._history.append(envelope)

        # Resolve matching subscribers (exact + wildcard)
        subscribers = self._resolve_subscribers(topic)
        await self._dispatch(envelope, subscribers)

        # If this envelope is a reply to a pending request, resolve the future
        if correlation_id and correlation_id in self._pending_replies:
            fut = self._pending_replies.get(correlation_id)
            if fut and not fut.done():
                fut.set_result(envelope)

        return envelope

    async def broadcast(
        self,
        payload: Any,
        *,
        sender: str = "system",
        priority: MessagePriority = MessagePriority.NORMAL,
        exclude_topics: list[str] | None = None,
    ) -> int:
        """
        Send *payload* to every registered subscriber across all topics.

        Returns the number of subscribers notified.
        """
        exclude = set(exclude_topics or [])
        async with self._lock:
            all_topics = list(self._subscriptions.keys())

        count = 0
        for topic in all_topics:
            if topic in exclude:
                continue
            envelope = MessageEnvelope.create(
                topic=topic, payload=payload, sender=sender, priority=priority
            )
            self._metrics["messages_sent"] += 1
            self._history.append(envelope)
            subs = self._resolve_subscribers(topic)
            await self._dispatch(envelope, subs)
            count += len(subs)

        logger.debug("Broadcast delivered to %d subscribers across %d topics", count, len(all_topics))
        return count

    # ------------------------------------------------------------------
    # Request / Reply (RPC)
    # ------------------------------------------------------------------

    async def request_reply(
        self,
        topic: str,
        payload: Any,
        *,
        sender: str = "system",
        timeout: float = 10.0,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> MessageEnvelope:
        """
        Publish a request on *topic* and await a correlated reply.

        The responder should call ``publish()`` with the same ``correlation_id``
        (or on the ``reply_to`` topic if provided) to fulfill the future.

        Raises ``asyncio.TimeoutError`` if no reply arrives within *timeout* seconds.
        """
        correlation_id = str(uuid.uuid4())
        reply_topic = f"__reply__.{correlation_id}"

        # Register a future before sending so we cannot miss an instant reply
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[MessageEnvelope] = loop.create_future()
        self._pending_replies[correlation_id] = fut

        # Subscribe to the ephemeral reply topic
        sub_id = await self.subscribe(
            reply_topic,
            self._make_reply_handler(correlation_id),
            subscriber_id="__rpc__",
        )

        try:
            await self.publish(
                topic,
                payload,
                sender=sender,
                priority=priority,
                correlation_id=correlation_id,
                reply_to=reply_topic,
            )
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            await self.unsubscribe(sub_id)
            self._pending_replies.pop(correlation_id, None)

    def _make_reply_handler(self, correlation_id: str) -> MessageHandler:
        """Return a handler that resolves the future for *correlation_id*."""
        async def _handler(envelope: MessageEnvelope) -> None:
            fut = self._pending_replies.get(correlation_id)
            if fut and not fut.done():
                fut.set_result(envelope)
        return _handler

    # ------------------------------------------------------------------
    # Dead-letter queue
    # ------------------------------------------------------------------

    def get_dead_letters(self, limit: int | None = None) -> list[DeadLetter]:
        """Return dead-letter entries (newest first)."""
        items = list(reversed(self._dead_letters))
        return items[:limit] if limit else items

    def clear_dead_letters(self) -> int:
        """Discard all dead-letter entries and return how many were removed."""
        count = len(self._dead_letters)
        self._dead_letters.clear()
        return count

    # ------------------------------------------------------------------
    # Audit log / history
    # ------------------------------------------------------------------

    def get_history(
        self,
        topic: str | None = None,
        limit: int | None = 100,
    ) -> list[MessageEnvelope]:
        """
        Return recent message history, optionally filtered by *topic*.
        Newest messages are first.
        """
        items = list(reversed(self._history))
        if topic:
            items = [m for m in items if m.topic == topic]
        return items[:limit] if limit else items

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, int]:
        """Return a snapshot of delivery metrics."""
        return dict(self._metrics)

    def reset_metrics(self) -> None:
        """Reset all counters to zero."""
        for key in self._metrics:
            self._metrics[key] = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_subscribers(self, topic: str) -> list[Subscription]:
        """Collect all subscriptions that match *topic* (exact + wildcards)."""
        matched: list[Subscription] = []
        for sub_topic, subs in self._subscriptions.items():
            if self._topic_matches(sub_topic, topic):
                matched.extend(subs)
        return matched

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        """
        Simple wildcard matching:
        - ``*`` matches any single segment
        - ``**`` matches any number of segments
        """
        if pattern == topic or pattern == "*" or pattern == "**":
            return True
        pattern_parts = pattern.split(".")
        topic_parts = topic.split(".")
        return _wildcard_match(pattern_parts, topic_parts)

    async def _dispatch(
        self, envelope: MessageEnvelope, subscribers: list[Subscription]
    ) -> None:
        """Deliver *envelope* to each subscriber, recording failures to DLQ."""
        if envelope.is_expired():
            logger.debug("Dropping expired message id=%s topic=%s", envelope.id, envelope.topic)
            return

        for sub in subscribers:
            if not sub.matches(envelope):
                continue
            self._metrics["messages_received"] += 1
            try:
                await sub.handler(envelope)
            except Exception as exc:  # noqa: BLE001
                self._metrics["failed_deliveries"] += 1
                dl = DeadLetter(
                    envelope=envelope,
                    reason=str(exc),
                    subscription_id=sub.id,
                )
                self._dead_letters.append(dl)
                logger.warning(
                    "Delivery failed for sub_id=%s topic=%s: %s",
                    sub.id, envelope.topic, exc,
                )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_topics(self) -> list[str]:
        """Return all topics that have at least one subscriber."""
        return [t for t, subs in self._subscriptions.items() if subs]

    def subscriber_count(self, topic: str | None = None) -> int:
        """Count total subscribers, or subscribers for a specific topic."""
        if topic:
            return len(self._subscriptions.get(topic, []))
        return sum(len(s) for s in self._subscriptions.values())

    def __repr__(self) -> str:
        return (
            f"<MessageBus topics={len(self.list_topics())} "
            f"subscribers={self.subscriber_count()} "
            f"sent={self._metrics['messages_sent']}>"
        )


# ---------------------------------------------------------------------------
# Wildcard helper (module-level for clarity)
# ---------------------------------------------------------------------------

def _wildcard_match(pattern_parts: list[str], topic_parts: list[str]) -> bool:
    """Recursive wildcard match for dotted topic segments."""
    if not pattern_parts and not topic_parts:
        return True
    if not pattern_parts:
        return False
    if pattern_parts[0] == "**":
        # '**' can consume 0 or more segments
        for i in range(len(topic_parts) + 1):
            if _wildcard_match(pattern_parts[1:], topic_parts[i:]):
                return True
        return False
    if not topic_parts:
        return False
    if pattern_parts[0] == "*" or pattern_parts[0] == topic_parts[0]:
        return _wildcard_match(pattern_parts[1:], topic_parts[1:])
    return False
