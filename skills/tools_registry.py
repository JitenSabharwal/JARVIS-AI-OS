"""
Centralized skills registry for the JARVIS AI OS system.

The :class:`ToolsRegistry` is a thread-safe singleton that owns the lifecycle
of all registered :class:`~skills.base_skill.BaseSkill` instances.  Agents
and the orchestrator interact with skills exclusively through this registry.

Usage::

    from skills.tools_registry import ToolsRegistry, SkillCategory

    registry = ToolsRegistry.get_instance()
    registry.load_builtin_skills()

    result = await registry.execute_skill("web_search", {"query": "Python docs"})
"""

from __future__ import annotations

import asyncio
import importlib
import threading
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from infrastructure.logger import get_logger
from skills.base_skill import BaseSkill, SkillResult, SkillStatus
from utils.exceptions import SkillExecutionError, SkillNotFoundError
from utils.helpers import timestamp_now

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Skill categories
# ---------------------------------------------------------------------------


class SkillCategory(str, Enum):
    """Logical groupings for registered skills."""

    WEB = "web"
    FILE = "file"
    SYSTEM = "system"
    DATA = "data"
    CUSTOM = "custom"
    COMMUNICATION = "communication"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolsRegistry:
    """Thread-safe singleton registry for all JARVIS skills.

    Skills are stored by name and can be looked up by name, category, or via
    a natural-language task description match.

    The registry is a singleton; always obtain it via :meth:`get_instance`.
    """

    _instance: Optional["ToolsRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        # Private constructor – use get_instance().
        self._skills: Dict[str, BaseSkill] = {}
        self._registry_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "ToolsRegistry":
        """Return the singleton :class:`ToolsRegistry` instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.debug("ToolsRegistry singleton created.")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton (useful for testing)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_skill(self, skill: BaseSkill, *, overwrite: bool = False) -> None:
        """Register *skill* in the registry.

        Args:
            skill: An instance of :class:`~skills.base_skill.BaseSkill`.
            overwrite: When ``True``, silently replace an existing skill with
                the same name.  When ``False`` (default), raise
                :exc:`ValueError` on name collision.

        Raises:
            ValueError: If a skill with the same name is already registered
                and *overwrite* is ``False``.
            TypeError: If *skill* is not a :class:`BaseSkill` instance.
        """
        if not isinstance(skill, BaseSkill):
            raise TypeError(f"Expected BaseSkill, got {type(skill).__name__}")

        with self._registry_lock:
            if skill.name in self._skills and not overwrite:
                raise ValueError(
                    f"Skill '{skill.name}' is already registered. "
                    "Use overwrite=True to replace it."
                )
            self._skills[skill.name] = skill
            logger.info(
                "Registered skill '%s' v%s (category=%s)",
                skill.name,
                skill.version,
                skill.category,
            )

    def unregister_skill(self, skill_name: str) -> None:
        """Remove a skill from the registry.

        Args:
            skill_name: The :attr:`~skills.base_skill.BaseSkill.name` to remove.

        Raises:
            :exc:`~utils.exceptions.SkillNotFoundError`: If the skill is not registered.
        """
        with self._registry_lock:
            if skill_name not in self._skills:
                raise SkillNotFoundError(skill_name)
            del self._skills[skill_name]
            logger.info("Unregistered skill '%s'.", skill_name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_skill(self, skill_name: str) -> BaseSkill:
        """Return the skill registered under *skill_name*.

        Raises:
            :exc:`~utils.exceptions.SkillNotFoundError`: If not found.
        """
        skill = self._skills.get(skill_name)
        if skill is None:
            raise SkillNotFoundError(skill_name)
        return skill

    def find_skills_by_category(self, category: SkillCategory) -> List[BaseSkill]:
        """Return all active skills belonging to *category*.

        Args:
            category: A :class:`SkillCategory` enum member.

        Returns:
            List of matching :class:`~skills.base_skill.BaseSkill` instances
            (only those with status :attr:`~skills.base_skill.SkillStatus.ACTIVE`).
        """
        cat_value = category.value if isinstance(category, SkillCategory) else str(category)
        return [
            s
            for s in self._skills.values()
            if s.category == cat_value and s.status == SkillStatus.ACTIVE
        ]

    def find_skills_for_task(self, task_description: str) -> List[BaseSkill]:
        """Return skills whose name or description matches *task_description*.

        Performs a simple case-insensitive keyword search across skill names
        and descriptions.  Returns active skills only, ordered by relevance
        (number of matching keywords).

        Args:
            task_description: Natural-language description of the task.

        Returns:
            Sorted list of matching :class:`~skills.base_skill.BaseSkill` instances.
        """
        keywords = [w.lower() for w in task_description.split() if len(w) > 2]
        scored: List[tuple[int, BaseSkill]] = []
        for skill in self._skills.values():
            if skill.status != SkillStatus.ACTIVE:
                continue
            text = f"{skill.name} {skill.description}".lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [s for _, s in scored]

    def list_all_skills(self) -> List[BaseSkill]:
        """Return all registered skills regardless of status."""
        return list(self._skills.values())

    def list_active_skills(self) -> List[BaseSkill]:
        """Return all skills with :attr:`~skills.base_skill.SkillStatus.ACTIVE` status."""
        return [s for s in self._skills.values() if s.status == SkillStatus.ACTIVE]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_skill(
        self,
        skill_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> SkillResult:
        """Execute a registered skill by name.

        This is the primary execution path for agents and the orchestrator.
        Uses :meth:`~skills.base_skill.BaseSkill.safe_execute` so errors are
        always captured in the returned :class:`~skills.base_skill.SkillResult`.

        Args:
            skill_name: Name of the registered skill.
            params: Parameter dict passed to the skill.

        Returns:
            A :class:`~skills.base_skill.SkillResult` (never raises).
        """
        try:
            skill = self.get_skill(skill_name)
        except SkillNotFoundError as exc:
            logger.error("execute_skill: %s", exc)
            return SkillResult.failure(
                error=str(exc),
                metadata={"skill_name": skill_name},
            )

        logger.debug("Executing skill '%s' with params=%s", skill_name, params)
        return await skill.safe_execute(params or {})

    # ------------------------------------------------------------------
    # Schema / introspection
    # ------------------------------------------------------------------

    def get_all_skills_info(self) -> List[Dict[str, Any]]:
        """Return full schema dicts for all active skills.

        Suitable for passing to an LLM as its ``tools`` / ``functions`` list.

        Returns:
            List of JSON-schema-compatible dicts, one per active skill.
        """
        return [s.get_schema() for s in self.list_active_skills()]

    def get_registry_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the registry."""
        by_category: Dict[str, int] = {}
        for skill in self._skills.values():
            by_category[skill.category] = by_category.get(skill.category, 0) + 1

        return {
            "total_skills": len(self._skills),
            "active_skills": len(self.list_active_skills()),
            "by_category": by_category,
            "skill_names": sorted(self._skills.keys()),
            "timestamp": timestamp_now(),
        }

    # ------------------------------------------------------------------
    # Built-in skill auto-discovery
    # ------------------------------------------------------------------

    def load_builtin_skills(self) -> int:
        """Discover and register all built-in skill modules.

        Imports ``skills.web_skills``, ``skills.file_skills``,
        ``skills.system_skills``, and ``skills.data_skills``, then
        instantiates and registers every exported skill class.

        Returns:
            Number of skills successfully registered.
        """
        builtin_modules = [
            "skills.web_skills",
            "skills.file_skills",
            "skills.system_skills",
            "skills.data_skills",
        ]

        registered = 0
        for module_path in builtin_modules:
            try:
                module = importlib.import_module(module_path)
            except ImportError as exc:
                logger.warning("Could not import built-in module '%s': %s", module_path, exc)
                continue

            skill_classes: List[Type[BaseSkill]] = []
            exported = getattr(module, "__all__", None)
            names = exported if exported else dir(module)

            for attr_name in names:
                attr = getattr(module, attr_name, None)
                if (
                    attr is not None
                    and isinstance(attr, type)
                    and issubclass(attr, BaseSkill)
                    and attr is not BaseSkill
                ):
                    skill_classes.append(attr)

            for skill_cls in skill_classes:
                try:
                    instance = skill_cls()
                    self.register_skill(instance, overwrite=True)
                    registered += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to instantiate/register skill %s: %s",
                        skill_cls.__name__,
                        exc,
                    )

        logger.info("load_builtin_skills: registered %d skill(s).", registered)
        return registered

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, skill_name: str) -> bool:
        return skill_name in self._skills

    def __repr__(self) -> str:
        return f"<ToolsRegistry skills={len(self._skills)}>"


__all__ = [
    "SkillCategory",
    "ToolsRegistry",
]
