"""
Centralized colored logging for the JARVIS AI OS system.

Usage::

    from infrastructure.logger import get_logger

    log = get_logger(__name__)
    log.info("System initialized")
    log.warning("Low memory")
    log.error("Agent crashed")

The :func:`setup_logger` function configures the root ``jarvis`` logger
with console (and optionally file) handlers.  Each call to
:func:`get_logger` returns a child of this root logger so that all
JARVIS log records share the same handlers and formatting.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"

_LEVEL_COLOURS: dict[int, str] = {
    logging.DEBUG: "\033[34m",       # blue
    logging.INFO: "\033[32m",        # green
    logging.WARNING: "\033[33m",     # yellow
    logging.ERROR: "\033[31m",       # red
    logging.CRITICAL: "\033[35m",    # magenta
}

_LEVEL_LABELS: dict[int, str] = {
    logging.DEBUG: "DEBUG   ",
    logging.INFO: "INFO    ",
    logging.WARNING: "WARNING ",
    logging.ERROR: "ERROR   ",
    logging.CRITICAL: "CRITICAL",
}


def _supports_colour(stream: object) -> bool:
    """Return ``True`` when *stream* appears to support ANSI escape codes."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    isatty = getattr(stream, "isatty", None)
    return isatty is not None and isatty()


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

_CONSOLE_FORMAT = "[%(asctime)s] %(levelname_coloured)s %(name)s — %(message)s"
_FILE_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """A :class:`logging.Formatter` that adds ANSI colour codes to the output.

    The level name is rendered in the colour associated with the log level;
    the message itself is left uncoloured.  On streams that do not support
    colour (detected via :meth:`sys.stderr.isatty`) the formatter degrades
    gracefully to plain text.

    Args:
        use_colour: Override colour detection.  When ``None`` (default),
            colour is used only when writing to a TTY.
    """

    def __init__(
        self,
        fmt: str = _CONSOLE_FORMAT,
        datefmt: str = _DATE_FORMAT,
        use_colour: Optional[bool] = None,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._use_colour: Optional[bool] = use_colour

    def _colour_enabled(self, record: logging.LogRecord) -> bool:
        if self._use_colour is not None:
            return self._use_colour
        # Try to detect from the handler's stream, if reachable
        return True  # default; overridden per-handler below

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelno, "")
        label = _LEVEL_LABELS.get(record.levelno, record.levelname)

        if self._colour_enabled(record) and colour:
            record.levelname_coloured = f"{_BOLD}{colour}{label}{_RESET}"
        else:
            record.levelname_coloured = label  # type: ignore[attr-defined]

        return super().format(record)


class PlainFormatter(logging.Formatter):
    """Plain-text formatter for file handlers (no ANSI codes)."""

    def __init__(self) -> None:
        super().__init__(fmt=_FILE_FORMAT, datefmt=_DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        # Ensure levelname_coloured attribute exists for unified fmt strings
        record.levelname_coloured = record.levelname  # type: ignore[attr-defined]
        return super().format(record)


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

_ROOT_LOGGER_NAME = "jarvis"
_setup_done = False


def setup_logger(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
    force: bool = False,
) -> logging.Logger:
    """Configure the root JARVIS logger.

    This function is idempotent unless *force* is ``True``.  It should
    normally be called once at application startup.

    Args:
        level: The minimum log level for the console handler.
        log_file: Optional path to a rotating log file.  If omitted (or
            ``None``), file logging is disabled.  The directory is created
            automatically if it does not exist.
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of rotated log files to keep.
        force: Re-configure even if the logger has already been set up.

    Returns:
        The configured root JARVIS :class:`logging.Logger`.
    """
    global _setup_done

    root_logger = logging.getLogger(_ROOT_LOGGER_NAME)

    if _setup_done and not force:
        return root_logger

    # Remove existing handlers to avoid duplicates on re-call
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)  # handlers filter to their own levels
    root_logger.propagate = False

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    colour = _supports_colour(sys.stderr)
    fmt = ColoredFormatter(use_colour=colour)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # --- File handler (optional) ---
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(PlainFormatter())
        root_logger.addHandler(file_handler)

    _setup_done = True
    root_logger.debug(
        "JARVIS logger configured (level=%s, file=%s)",
        logging.getLevelName(level),
        log_file or "none",
    )
    return root_logger


def get_logger(name: str) -> "JARVISLogger":
    """Return a :class:`JARVISLogger` child of the root JARVIS logger.

    If :func:`setup_logger` has not been called yet this function calls it
    with default settings so that logging is always functional.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`JARVISLogger` instance.
    """
    if not _setup_done:
        setup_logger()

    if name.startswith(_ROOT_LOGGER_NAME + ".") or name == _ROOT_LOGGER_NAME:
        child_name = name
    else:
        child_name = f"{_ROOT_LOGGER_NAME}.{name}"

    return JARVISLogger(logging.getLogger(child_name))


# ---------------------------------------------------------------------------
# JARVISLogger wrapper
# ---------------------------------------------------------------------------


class JARVISLogger:
    """A thin, typed wrapper around :class:`logging.Logger`.

    Provides the familiar ``debug`` / ``info`` / ``warning`` / ``error`` /
    ``critical`` interface together with a few convenience helpers.

    Args:
        logger: The underlying :class:`logging.Logger` to wrap.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    # ------------------------------------------------------------------
    # Standard log-level methods
    # ------------------------------------------------------------------

    def debug(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log *msg* at DEBUG level."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log *msg* at INFO level."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log *msg* at WARNING level."""
        self._logger.warning(msg, *args, **kwargs)

    # Alias
    warn = warning

    def error(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log *msg* at ERROR level."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log *msg* at CRITICAL level."""
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log *msg* at ERROR level, appending the current exception traceback."""
        self._logger.exception(msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args: object, **kwargs: object) -> None:
        """Log *msg* at an arbitrary numeric *level*."""
        self._logger.log(level, msg, *args, **kwargs)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def success(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log a success message (rendered at INFO level with a ✓ prefix)."""
        self._logger.info("✓ " + msg, *args, **kwargs)

    def banner(self, title: str, width: int = 60) -> None:
        """Log a visual banner useful for startup / shutdown messages."""
        border = "=" * width
        padded = f"  {title}  "
        self._logger.info(border)
        self._logger.info(padded.center(width))
        self._logger.info(border)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The name of the underlying logger."""
        return self._logger.name

    @property
    def level(self) -> int:
        """The effective log level of the underlying logger."""
        return self._logger.getEffectiveLevel()

    def set_level(self, level: int) -> None:
        """Set the minimum log level for this logger."""
        self._logger.setLevel(level)

    def is_enabled_for(self, level: int) -> bool:
        """Return ``True`` if the logger will emit records at *level*."""
        return self._logger.isEnabledFor(level)

    def child(self, suffix: str) -> "JARVISLogger":
        """Return a child :class:`JARVISLogger` with name ``<this.name>.<suffix>``."""
        return JARVISLogger(self._logger.getChild(suffix))

    # Allow the wrapper to be used as a plain logging.Logger in third-party code
    def __getattr__(self, name: str) -> object:
        return getattr(self._logger, name)


__all__ = [
    "ColoredFormatter",
    "PlainFormatter",
    "setup_logger",
    "get_logger",
    "JARVISLogger",
]
