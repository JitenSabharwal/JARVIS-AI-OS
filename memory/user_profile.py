"""
User profile memory for personalization.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from utils.helpers import timestamp_now


@dataclass
class UserProfile:
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    traits: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=timestamp_now)
    updated_at: str = field(default_factory=timestamp_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UserProfileStore:
    """Thread-safe in-memory user profile store."""

    def __init__(self) -> None:
        self._profiles: Dict[str, UserProfile] = {}
        self._lock = threading.RLock()

    def get_or_create(self, user_id: str) -> UserProfile:
        with self._lock:
            profile = self._profiles.get(user_id)
            if profile is None:
                profile = UserProfile(user_id=user_id)
                self._profiles[user_id] = profile
            return profile

    def get(self, user_id: str) -> Optional[UserProfile]:
        with self._lock:
            return self._profiles.get(user_id)

    def update_preferences(self, user_id: str, **preferences: Any) -> UserProfile:
        with self._lock:
            profile = self.get_or_create(user_id)
            profile.preferences.update(preferences)
            profile.updated_at = timestamp_now()
            return profile

    def update_traits(self, user_id: str, **traits: Any) -> UserProfile:
        with self._lock:
            profile = self.get_or_create(user_id)
            profile.traits.update(traits)
            profile.updated_at = timestamp_now()
            return profile

    def summary(self, user_id: str) -> str:
        profile = self.get(user_id)
        if profile is None:
            return ""
        parts = []
        if profile.preferences:
            pref_items = ", ".join(f"{k}={v}" for k, v in sorted(profile.preferences.items()))
            parts.append(f"Preferences: {pref_items}")
        if profile.traits:
            trait_items = ", ".join(f"{k}={v}" for k, v in sorted(profile.traits.items()))
            parts.append(f"Traits: {trait_items}")
        return " | ".join(parts)

