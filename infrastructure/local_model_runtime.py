"""
Local model runtime manager with RAM-budget-aware load/unload behavior.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LocalModelSpec:
    name: str
    modality: str = "text"
    size_gb: float = 0.0
    backend: str = "ollama"


@dataclass
class LoadedModelState:
    name: str
    size_gb: float
    modality: str
    backend: str
    loaded_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    in_use_count: int = 0


class LocalModelRuntimeManager:
    """
    Tracks local model residency under a configurable memory budget.

    This manager is backend-agnostic and can support Ollama now and MLX later.
    """

    def __init__(
        self,
        *,
        memory_budget_gb: float = 35.0,
        total_memory_gb: float = 48.0,
        max_parallel_models: int = 3,
        large_model_threshold_gb: float = 18.0,
        single_large_model_mode: bool = True,
        auto_unload: bool = True,
    ) -> None:
        self._memory_budget_gb = max(1.0, float(memory_budget_gb))
        self._total_memory_gb = max(self._memory_budget_gb, float(total_memory_gb))
        self._max_parallel_models = max(1, int(max_parallel_models))
        self._large_model_threshold_gb = max(1.0, float(large_model_threshold_gb))
        self._single_large_model_mode = bool(single_large_model_mode)
        self._auto_unload = bool(auto_unload)
        self._loaded: Dict[str, LoadedModelState] = {}
        self._lock = threading.RLock()

    @property
    def memory_budget_gb(self) -> float:
        return self._memory_budget_gb

    @property
    def total_memory_gb(self) -> float:
        return self._total_memory_gb

    def used_memory_gb(self) -> float:
        with self._lock:
            return round(sum(m.size_gb for m in self._loaded.values()), 3)

    def can_fit(self, size_gb: float) -> bool:
        return (self.used_memory_gb() + max(0.0, float(size_gb))) <= self._memory_budget_gb

    def ensure_capacity(self, required_size_gb: float) -> List[str]:
        """
        Ensure room for a model; optionally unload least-recently-used idle models.

        Returns the list of unloaded model names.
        """
        required = max(0.0, float(required_size_gb))
        unloaded: List[str] = []
        with self._lock:
            if self.used_memory_gb() + required <= self._memory_budget_gb:
                return unloaded

            if not self._auto_unload:
                raise RuntimeError("Insufficient local model budget and auto_unload disabled")

            # LRU by last_used_at among idle models only.
            candidates = sorted(
                (m for m in self._loaded.values() if m.in_use_count == 0),
                key=lambda m: m.last_used_at,
            )
            for state in candidates:
                if self.used_memory_gb() + required <= self._memory_budget_gb:
                    break
                del self._loaded[state.name]
                unloaded.append(state.name)
                logger.info(
                    "Unloaded local model '%s' (%.2fGB) to free memory budget",
                    state.name,
                    state.size_gb,
                )

            if self.used_memory_gb() + required > self._memory_budget_gb:
                raise RuntimeError(
                    "Cannot satisfy local model memory budget even after unloading idle models"
                )
        return unloaded

    def mark_loaded(self, spec: LocalModelSpec) -> None:
        with self._lock:
            state = self._loaded.get(spec.name)
            if state:
                state.last_used_at = time.time()
                return
            self._loaded[spec.name] = LoadedModelState(
                name=spec.name,
                size_gb=max(0.0, float(spec.size_gb)),
                modality=spec.modality,
                backend=spec.backend,
            )
            logger.info(
                "Loaded local model '%s' modality=%s size_gb=%.2f backend=%s",
                spec.name,
                spec.modality,
                spec.size_gb,
                spec.backend,
            )

    def mark_in_use(self, name: str) -> None:
        with self._lock:
            state = self._loaded.get(name)
            if not state:
                return
            if self._single_large_model_mode:
                current_active = [m for m in self._loaded.values() if m.in_use_count > 0]
                active_large = [m for m in current_active if m.size_gb >= self._large_model_threshold_gb]
                candidate_is_large = state.size_gb >= self._large_model_threshold_gb
                if candidate_is_large and current_active:
                    raise RuntimeError(
                        "Large model concurrency policy: a large model must run exclusively"
                    )
                if not candidate_is_large and active_large:
                    raise RuntimeError(
                        "Large model concurrency policy: cannot run small model while large model is active"
                    )
            active_model_count = sum(1 for m in self._loaded.values() if m.in_use_count > 0)
            if state.in_use_count == 0 and active_model_count >= self._max_parallel_models:
                raise RuntimeError("Local model concurrency limit reached")
            state.in_use_count += 1
            state.last_used_at = time.time()

    def mark_released(self, name: str) -> None:
        with self._lock:
            state = self._loaded.get(name)
            if not state:
                return
            state.in_use_count = max(0, state.in_use_count - 1)
            state.last_used_at = time.time()

    def can_run_parallel(self, model_sizes_gb: List[float]) -> bool:
        sizes = [max(0.0, float(s)) for s in model_sizes_gb]
        if len(sizes) > self._max_parallel_models:
            return False
        if self._single_large_model_mode:
            large_count = sum(1 for s in sizes if s >= self._large_model_threshold_gb)
            if large_count > 1:
                return False
            if large_count == 1 and len(sizes) > 1:
                return False
        return sum(sizes) <= self._memory_budget_gb

    def status(self) -> Dict[str, object]:
        with self._lock:
            loaded = [
                {
                    "name": s.name,
                    "size_gb": s.size_gb,
                    "modality": s.modality,
                    "backend": s.backend,
                    "in_use_count": s.in_use_count,
                    "last_used_at": s.last_used_at,
                }
                for s in sorted(self._loaded.values(), key=lambda x: x.name)
            ]
        return {
            "memory_budget_gb": self._memory_budget_gb,
            "total_memory_gb": self._total_memory_gb,
            "used_memory_gb": self.used_memory_gb(),
            "available_memory_gb": round(self._memory_budget_gb - self.used_memory_gb(), 3),
            "max_parallel_models": self._max_parallel_models,
            "large_model_threshold_gb": self._large_model_threshold_gb,
            "single_large_model_mode": self._single_large_model_mode,
            "loaded_models": loaded,
        }
