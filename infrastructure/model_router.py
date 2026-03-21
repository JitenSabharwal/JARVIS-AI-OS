"""
Hybrid model routing for local and API providers.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from infrastructure.logger import get_logger

logger = get_logger(__name__)


class PrivacyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ModelRequest:
    prompt: str
    task_type: str = "general"
    modality: str = "text"
    media: Dict[str, Any] = field(default_factory=dict)
    privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM
    max_latency_ms: int | None = None
    prefer_local: bool | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    text: str
    provider_name: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteDecision:
    primary: str
    chain: List[str]
    reason: str
    task_type: str
    privacy_level: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary": self.primary,
            "chain": self.chain,
            "reason": self.reason,
            "task_type": self.task_type,
            "privacy_level": self.privacy_level,
        }


class ModelProvider(ABC):
    """Model provider contract for router-managed generation."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Expected values: 'local' or 'api'."""
        pass

    def is_available(self) -> bool:
        return True

    @property
    def supported_modalities(self) -> set[str]:
        return {"text"}

    def can_handle(self, request: ModelRequest) -> bool:
        if request.modality == "voice" and "text" in self.supported_modalities:
            # Voice requests already carry transcribed text prompt.
            return True
        return request.modality in self.supported_modalities

    @abstractmethod
    async def generate(self, request: ModelRequest) -> str:
        pass


class CallableModelProvider(ModelProvider):
    """Provider adapter over an async callable."""

    def __init__(
        self,
        *,
        name: str,
        provider_type: str,
        handler: Callable[[ModelRequest], Awaitable[str]],
        available_fn: Optional[Callable[[], bool]] = None,
        supported_modalities: Optional[set[str]] = None,
    ) -> None:
        self._name = name
        self._provider_type = provider_type
        self._handler = handler
        self._available_fn = available_fn
        self._supported_modalities = supported_modalities or {"text"}

    @property
    def name(self) -> str:
        return self._name

    @property
    def provider_type(self) -> str:
        return self._provider_type

    def is_available(self) -> bool:
        if self._available_fn:
            return bool(self._available_fn())
        return True

    @property
    def supported_modalities(self) -> set[str]:
        return set(self._supported_modalities)

    async def generate(self, request: ModelRequest) -> str:
        return await self._handler(request)


class ModelRouter:
    """Task/privacy-aware router with local/API fallback chains."""

    _API_PREFERRED_TASKS = {
        "information_query",
        "memory_query",
        "analysis",
        "coding",
        "writing",
    }

    _LOCAL_PREFERRED_TASKS = {
        "status_query",
        "time_query",
        "greeting",
        "farewell",
        "acknowledgement",
        "summarization",
    }

    def __init__(
        self,
        *,
        local_provider: Optional[ModelProvider] = None,
        api_provider: Optional[ModelProvider] = None,
        fallback_enabled: bool = True,
        shadow_mode: bool = False,
    ) -> None:
        self._local_provider = local_provider
        self._api_provider = api_provider
        self._fallback_enabled = fallback_enabled
        self._shadow_mode = shadow_mode

    def has_provider(self) -> bool:
        return bool(self._local_provider or self._api_provider)

    def route(self, request: ModelRequest) -> RouteDecision:
        available_local = bool(
            self._local_provider
            and self._local_provider.is_available()
            and self._local_provider.can_handle(request)
        )
        available_api = bool(
            self._api_provider
            and self._api_provider.is_available()
            and self._api_provider.can_handle(request)
        )

        if not available_local and not available_api:
            raise RuntimeError("No model providers available")

        policy_decision = request.metadata.get("policy_decision", {})
        if isinstance(policy_decision, dict):
            allowed = policy_decision.get("allowed_providers", [])
            if isinstance(allowed, list) and allowed:
                allowed_set = {str(a).strip().lower() for a in allowed if str(a).strip()}
                if "local" not in allowed_set:
                    available_local = False
                if "api" not in allowed_set:
                    available_api = False
            if policy_decision.get("prefer_local", None) is not None:
                request.prefer_local = bool(policy_decision.get("prefer_local"))

        if request.privacy_level == PrivacyLevel.HIGH:
            if available_local:
                return RouteDecision(
                    primary=self._local_provider.name,  # type: ignore[union-attr]
                    chain=[self._local_provider.name],  # type: ignore[union-attr]
                    reason="high_privacy_local_only",
                    task_type=request.task_type,
                    privacy_level=request.privacy_level.value,
                )
            raise RuntimeError("High-privacy request requires local provider")

        if request.prefer_local is True and available_local:
            chain = [self._local_provider.name]  # type: ignore[union-attr]
            if self._fallback_enabled and available_api:
                chain.append(self._api_provider.name)  # type: ignore[union-attr]
            return RouteDecision(
                primary=chain[0],
                chain=chain,
                reason="prefer_local",
                task_type=request.task_type,
                privacy_level=request.privacy_level.value,
            )

        if request.prefer_local is False and available_api:
            chain = [self._api_provider.name]  # type: ignore[union-attr]
            if self._fallback_enabled and available_local:
                chain.append(self._local_provider.name)  # type: ignore[union-attr]
            return RouteDecision(
                primary=chain[0],
                chain=chain,
                reason="prefer_api",
                task_type=request.task_type,
                privacy_level=request.privacy_level.value,
            )

        task = request.task_type.strip().lower()
        complexity = None
        plan_meta = request.metadata.get("response_plan")
        if isinstance(plan_meta, dict):
            try:
                complexity = float(plan_meta.get("complexity"))
            except (TypeError, ValueError):
                complexity = None

        if complexity is not None and available_api and available_local:
            if complexity >= 0.72:
                return RouteDecision(
                    primary=self._api_provider.name,  # type: ignore[union-attr]
                    chain=[self._api_provider.name, self._local_provider.name],  # type: ignore[union-attr]
                    reason="complexity_prefers_api",
                    task_type=request.task_type,
                    privacy_level=request.privacy_level.value,
                )
            if complexity <= 0.42:
                return RouteDecision(
                    primary=self._local_provider.name,  # type: ignore[union-attr]
                    chain=[self._local_provider.name, self._api_provider.name],  # type: ignore[union-attr]
                    reason="complexity_prefers_local",
                    task_type=request.task_type,
                    privacy_level=request.privacy_level.value,
                )

        budget_usd = None
        if isinstance(policy_decision, dict):
            try:
                budget_usd = float(policy_decision.get("budget_usd"))
            except Exception:
                budget_usd = None
        if budget_usd is not None and budget_usd <= 0.002 and available_local:
            chain = [self._local_provider.name]  # type: ignore[union-attr]
            if self._fallback_enabled and available_api:
                chain.append(self._api_provider.name)  # type: ignore[union-attr]
            return RouteDecision(
                primary=chain[0],
                chain=chain,
                reason="policy_budget_prefers_local",
                task_type=request.task_type,
                privacy_level=request.privacy_level.value,
            )

        if task in self._LOCAL_PREFERRED_TASKS and available_local:
            chain = [self._local_provider.name]  # type: ignore[union-attr]
            if self._fallback_enabled and available_api:
                chain.append(self._api_provider.name)  # type: ignore[union-attr]
            return RouteDecision(
                primary=chain[0],
                chain=chain,
                reason="task_prefers_local",
                task_type=request.task_type,
                privacy_level=request.privacy_level.value,
            )

        if task in self._API_PREFERRED_TASKS and available_api:
            chain = [self._api_provider.name]  # type: ignore[union-attr]
            if self._fallback_enabled and available_local:
                chain.append(self._local_provider.name)  # type: ignore[union-attr]
            return RouteDecision(
                primary=chain[0],
                chain=chain,
                reason="task_prefers_api",
                task_type=request.task_type,
                privacy_level=request.privacy_level.value,
            )

        if available_local and available_api:
            return RouteDecision(
                primary=self._local_provider.name,  # type: ignore[union-attr]
                chain=[self._local_provider.name, self._api_provider.name],  # type: ignore[union-attr]
                reason="default_local_then_api",
                task_type=request.task_type,
                privacy_level=request.privacy_level.value,
            )
        if available_local:
            return RouteDecision(
                primary=self._local_provider.name,  # type: ignore[union-attr]
                chain=[self._local_provider.name],  # type: ignore[union-attr]
                reason="only_local_available",
                task_type=request.task_type,
                privacy_level=request.privacy_level.value,
            )
        return RouteDecision(
            primary=self._api_provider.name,  # type: ignore[union-attr]
            chain=[self._api_provider.name],  # type: ignore[union-attr]
            reason="only_api_available",
            task_type=request.task_type,
            privacy_level=request.privacy_level.value,
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        decision = self.route(request)
        providers = self._providers_map()
        last_error = ""
        for provider_name in decision.chain:
            provider = providers.get(provider_name)
            if provider is None:
                continue
            if not provider.is_available():
                last_error = f"provider_unavailable:{provider_name}"
                continue
            if not provider.can_handle(request):
                last_error = f"provider_cannot_handle_modality:{provider_name}:{request.modality}"
                continue
            started = time.time()
            try:
                text = await provider.generate(request)
                text_out = str(text or "").strip()
                if not text_out:
                    last_error = f"empty_output:{provider.name}"
                    logger.warning("Provider '%s' returned empty output", provider.name)
                    continue
                if self._looks_like_prompt_echo(text_out, request):
                    last_error = f"invalid_output_prompt_echo:{provider.name}"
                    logger.warning("Provider '%s' returned prompt-echo output; trying fallback", provider.name)
                    continue
                latency_ms = round((time.time() - started) * 1000.0, 2)
                logger.info(
                    "model_route provider=%s task=%s privacy=%s latency_ms=%.2f reason=%s",
                    provider.name,
                    request.task_type,
                    request.privacy_level.value,
                    latency_ms,
                    decision.reason,
                )
                metadata: Dict[str, Any] = {"route_decision": decision.to_dict()}
                if self._shadow_mode and len(decision.chain) > 1:
                    shadow_result = await self._shadow_invoke(
                        request=request,
                        primary_provider=provider.name,
                        chain=decision.chain,
                    )
                    metadata["shadow"] = shadow_result
                return ModelResponse(
                    text=text_out,
                    provider_name=provider.name,
                    latency_ms=latency_ms,
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.warning("Provider '%s' failed: %s", provider.name, exc)
                continue
        raise RuntimeError(f"All providers failed. last_error={last_error}")

    async def _shadow_invoke(
        self,
        *,
        request: ModelRequest,
        primary_provider: str,
        chain: List[str],
    ) -> Dict[str, Any]:
        providers = self._providers_map()
        shadow_provider_name = next((p for p in chain if p != primary_provider), "")
        if not shadow_provider_name:
            return {"enabled": True, "used": False}
        provider = providers.get(shadow_provider_name)
        if provider is None or not provider.is_available() or not provider.can_handle(request):
            return {
                "enabled": True,
                "used": False,
                "provider_name": shadow_provider_name,
                "reason": "unavailable_or_incompatible",
            }

        started = time.time()
        try:
            _ = await provider.generate(request)
            latency_ms = round((time.time() - started) * 1000.0, 2)
            logger.info(
                "model_route_shadow provider=%s task=%s latency_ms=%.2f",
                shadow_provider_name,
                request.task_type,
                latency_ms,
            )
            return {
                "enabled": True,
                "used": True,
                "provider_name": shadow_provider_name,
                "latency_ms": latency_ms,
                "status": "ok",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "enabled": True,
                "used": True,
                "provider_name": shadow_provider_name,
                "status": "failed",
                "error": str(exc),
            }

    def _providers_map(self) -> Dict[str, ModelProvider]:
        providers: Dict[str, ModelProvider] = {}
        if self._local_provider:
            providers[self._local_provider.name] = self._local_provider
        if self._api_provider:
            providers[self._api_provider.name] = self._api_provider
        return providers

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        s = str(text or "").lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _looks_like_prompt_echo(text: str, request: ModelRequest) -> bool:
        low = str(text or "").strip().lower()
        if not low:
            return True
        if str(request.task_type or "").strip().lower() == "summarization":
            return False
        if low.startswith(("user request:", "assistant draft response:", "analyze the request:", "thinking process")):
            return True
        user_input = str(request.metadata.get("user_input", "")).strip()
        if user_input:
            t_norm = ModelRouter._normalize_for_match(text)
            q_norm = ModelRouter._normalize_for_match(user_input)
            if t_norm == q_norm:
                return True
            t_stripped = re.sub(r"^(user request|request|query)\s*:\s*", "", t_norm).strip()
            if t_stripped and t_stripped == q_norm:
                return True
        return False
