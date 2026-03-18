"""
JARVIS AI OS - Configuration Settings Loader
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Re-export the core config machinery so callers can do:
#   from config.settings import load_settings, Settings
from core.config import ConfigManager, JARVISConfig, get_config  # noqa: F401

Settings = JARVISConfig


def load_settings(config_path: str | None = None, env_file: str | None = None) -> JARVISConfig:
    """Load JARVIS settings from YAML config and/or .env file.

    Args:
        config_path: Optional path to a YAML configuration file.
        env_file: Optional path to a .env file.

    Returns:
        A fully populated :class:`JARVISConfig` instance.
    """
    manager = ConfigManager(
        config_file=config_path,
        env_file=env_file or str(Path(__file__).parent.parent / ".env"),
    )
    return manager.load()


def get_setting(key: str, default: Any = None) -> Any:
    """Retrieve a top-level setting value by dot-separated key.

    Example::

        api_port = get_setting("api.port", 8080)

    Args:
        key: Dot-separated key path (e.g. ``"agent.max_agents"``).
        default: Value returned when the key is not found.

    Returns:
        The resolved setting value or *default*.
    """
    config = get_config()
    parts = key.split(".")
    obj: Any = config
    for part in parts:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif isinstance(obj, dict):
            obj = obj.get(part, default)
            if obj is default:
                return default
        else:
            return default
    return obj
