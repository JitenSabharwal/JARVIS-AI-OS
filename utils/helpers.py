"""
General-purpose helper utilities for the JARVIS AI OS system.

All public helpers are free functions except for decorators, which are returned
by factory functions so they can be used with or without arguments.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Identity / time helpers
# ---------------------------------------------------------------------------


def generate_id(prefix: str = "") -> str:
    """Return a unique identifier string.

    Args:
        prefix: Optional string prepended to the UUID, separated by ``-``.

    Returns:
        A unique ID string, e.g. ``"task-4b3c1a..."``.
    """
    uid = uuid.uuid4().hex
    return f"{prefix}-{uid}" if prefix else uid


def timestamp_now() -> str:
    """Return the current UTC time as an ISO-8601 string with timezone info.

    Returns:
        String like ``"2024-01-15T12:34:56.789012+00:00"``.
    """
    return datetime.now(tz=timezone.utc).isoformat()


def timestamp_epoch() -> float:
    """Return the current time as a Unix timestamp (seconds since epoch)."""
    return time.time()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds (may be fractional).

    Returns:
        A string such as ``"2h 5m 3s"`` or ``"450ms"``.

    Examples:
        >>> format_duration(7503.0)
        '2h 5m 3s'
        >>> format_duration(0.45)
        '450ms'
    """
    if seconds < 0:
        return f"-{format_duration(-seconds)}"
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"

    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def truncate_string(text: str, max_length: int = 100, ellipsis: str = "...") -> str:
    """Truncate *text* to *max_length* characters, appending *ellipsis* if cut.

    Args:
        text: The string to (potentially) truncate.
        max_length: Maximum allowed length of the returned string (including
            the ellipsis if appended).
        ellipsis: Suffix appended when the string is truncated.

    Returns:
        The (possibly truncated) string.
    """
    if len(text) <= max_length:
        return text
    cut = max(0, max_length - len(ellipsis))
    return text[:cut] + ellipsis


# ---------------------------------------------------------------------------
# Data-structure helpers
# ---------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Nested dicts are merged recursively; all other value types in *override*
    replace those in *base*.  Neither input dict is mutated.

    Args:
        base: The starting dictionary.
        override: Values that take precedence over *base*.

    Returns:
        A new merged dictionary.
    """
    result: dict[str, Any] = dict(base)
    for key, override_val in override.items():
        base_val = result.get(key)
        if isinstance(base_val, dict) and isinstance(override_val, dict):
            result[key] = deep_merge(base_val, override_val)
        else:
            result[key] = override_val
    return result


def flatten_list(nested: Iterable[Any]) -> list[Any]:
    """Recursively flatten a (potentially deeply) nested list.

    Args:
        nested: An iterable that may contain other iterables.

    Returns:
        A flat list of all leaf values.

    Examples:
        >>> flatten_list([1, [2, [3, 4]], 5])
        [1, 2, 3, 4, 5]
    """
    result: list[Any] = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    """Split *items* into consecutive sublists of at most *size* elements.

    Args:
        items: The source list.
        size: Maximum number of elements per chunk.  Must be >= 1.

    Returns:
        A list of sublists.

    Raises:
        ValueError: If *size* is less than 1.
    """
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def safe_json_dumps(obj: Any, *, default: Any = None, **kwargs: Any) -> str:
    """Serialize *obj* to a JSON string, never raising on non-serializable types.

    Non-serializable values are replaced with *default* (``None`` by default).

    Args:
        obj: The Python object to serialize.
        default: Fallback value for objects that are not JSON-serializable.
        **kwargs: Forwarded to :func:`json.dumps`.

    Returns:
        A JSON string.
    """

    def _fallback(o: Any) -> Any:
        if isinstance(o, (datetime,)):
            return o.isoformat()
        if isinstance(o, (set, frozenset)):
            return list(o)
        if hasattr(o, "__dict__"):
            return o.__dict__
        return default

    try:
        return json.dumps(obj, default=_fallback, **kwargs)
    except (TypeError, ValueError) as exc:
        logger.debug("safe_json_dumps fallback triggered: %s", exc)
        return json.dumps(default)


def safe_json_loads(text: str, *, default: Any = None) -> Any:
    """Deserialize a JSON string, returning *default* on any parse error.

    Args:
        text: The JSON-encoded string.
        default: Value returned when parsing fails.

    Returns:
        The parsed Python object, or *default*.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.debug("safe_json_loads failed: %s", exc)
        return default


# ---------------------------------------------------------------------------
# Input sanitisation
# ---------------------------------------------------------------------------

# Characters that should be stripped from user-supplied strings before they
# are processed further (control characters, null bytes, etc.).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_input(
    text: str,
    *,
    max_length: Optional[int] = None,
    strip_html: bool = True,
    strip_control: bool = True,
) -> str:
    """Return a sanitized copy of *text*.

    The function does **not** escape HTML for rendering; it removes potentially
    dangerous characters to reduce injection surface area.

    Args:
        text: Raw input string.
        max_length: If provided, truncate to this length (before other ops).
        strip_html: Remove ``<...>`` HTML/XML tags when ``True``.
        strip_control: Remove ASCII control characters when ``True``.

    Returns:
        Sanitized string.
    """
    if not isinstance(text, str):
        text = str(text)

    if max_length is not None:
        text = text[:max_length]

    if strip_html:
        text = re.sub(r"<[^>]+>", "", text)

    if strip_control:
        text = _CONTROL_CHAR_RE.sub("", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Async decorators
# ---------------------------------------------------------------------------


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator factory: retry an async function on failure.

    Args:
        max_attempts: Total number of attempts (including the first try).
        delay: Seconds to wait before the first retry.
        backoff: Multiplier applied to *delay* after each failure.
        exceptions: Tuple of exception types that trigger a retry.

    Returns:
        A decorator that wraps an async callable with retry logic.

    Example::

        @retry_async(max_attempts=5, delay=0.5, backoff=2.0)
        async def call_external_api() -> dict:
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt < max_attempts:
                        logger.debug(
                            "retry_async: attempt %d/%d for %s failed (%s); "
                            "retrying in %.2fs",
                            attempt,
                            max_attempts,
                            func.__qualname__,
                            exc,
                            current_delay,
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.warning(
                            "retry_async: %s exhausted %d attempts",
                            func.__qualname__,
                            max_attempts,
                        )
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


def timed_async(func: F) -> F:
    """Decorator: log the wall-clock execution time of an async function.

    The elapsed time is logged at DEBUG level with the function's qualified
    name.

    Example::

        @timed_async
        async def expensive_operation() -> None:
            ...
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(
                "timed_async: %s completed in %s",
                func.__qualname__,
                format_duration(elapsed),
            )

    return wrapper  # type: ignore[return-value]


__all__ = [
    "generate_id",
    "timestamp_now",
    "timestamp_epoch",
    "format_duration",
    "truncate_string",
    "deep_merge",
    "flatten_list",
    "chunk_list",
    "safe_json_dumps",
    "safe_json_loads",
    "sanitize_input",
    "retry_async",
    "timed_async",
]
