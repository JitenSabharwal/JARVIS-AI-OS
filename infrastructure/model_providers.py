"""
Provider adapters for hybrid model routing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]

from infrastructure.local_model_runtime import LocalModelRuntimeManager, LocalModelSpec
from infrastructure.model_router import ModelProvider, ModelRequest


class CohereProvider(ModelProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "command-r-plus",
        base_url: str = "https://api.cohere.com/v2",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "cohere"

    @property
    def provider_type(self) -> str:
        return "api"

    @property
    def supported_modalities(self) -> set[str]:
        return {"text"}

    def is_available(self) -> bool:
        return bool(self._api_key and aiohttp is not None)

    async def generate(self, request: ModelRequest) -> str:
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for CohereProvider")
        url = f"{self._base_url}/chat"
        payload: Dict[str, Any] = {
            "model": self._model,
            "message": request.prompt,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                data = await response.json()
                if response.status >= 400:
                    raise RuntimeError(f"Cohere error {response.status}: {data}")
                text = data.get("text")
                if not text and isinstance(data.get("message"), dict):
                    text = data["message"].get("content", [{}])[0].get("text")
                if not text:
                    raise RuntimeError("Cohere response did not contain text")
                return str(text).strip()


class OllamaProvider(ModelProvider):
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        text_model: str = "qwen2.5:7b",
        image_model: str = "",
        audio_model: str = "",
        timeout_seconds: float = 45.0,
        runtime_manager: Optional[LocalModelRuntimeManager] = None,
        model_sizes_gb: Optional[Dict[str, float]] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._text_model = text_model
        self._image_model = image_model
        self._audio_model = audio_model
        self._timeout_seconds = timeout_seconds
        self._runtime = runtime_manager
        self._model_sizes_gb = model_sizes_gb or {}

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def provider_type(self) -> str:
        return "local"

    @property
    def supported_modalities(self) -> set[str]:
        mods = {"text"}
        if self._image_model:
            mods.add("image")
        if self._audio_model:
            mods.add("audio")
        return mods

    def is_available(self) -> bool:
        return aiohttp is not None

    def _select_model(self, modality: str) -> str:
        if modality == "text":
            return self._text_model
        if modality == "image" and self._image_model:
            return self._image_model
        if modality == "audio" and self._audio_model:
            return self._audio_model
        raise RuntimeError(f"Ollama model not configured for modality '{modality}'")

    async def generate(self, request: ModelRequest) -> str:
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for OllamaProvider")

        model_name = self._select_model(request.modality)
        if self._runtime:
            model_size = float(self._model_sizes_gb.get(model_name, 0.0))
            self._runtime.ensure_capacity(model_size)
            self._runtime.mark_loaded(
                LocalModelSpec(
                    name=model_name,
                    modality=request.modality,
                    size_gb=model_size,
                    backend="ollama",
                )
            )
            self._runtime.mark_in_use(model_name)

        try:
            url = f"{self._base_url}/api/generate"
            payload: Dict[str, Any] = {
                "model": model_name,
                "prompt": request.prompt,
                "stream": False,
            }
            if request.modality == "image":
                images = request.media.get("images", [])
                if images:
                    payload["images"] = images
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    data = await response.json()
                    if response.status >= 400:
                        raise RuntimeError(f"Ollama error {response.status}: {data}")
                    text = data.get("response", "")
                    if not text:
                        raise RuntimeError("Ollama response did not contain text")
                    return str(text).strip()
        finally:
            if self._runtime:
                self._runtime.mark_released(model_name)


class MLXProvider(ModelProvider):
    """
    Placeholder for future MLX/VLMX backend integration.

    Keep this provider disabled in config until MLX runtime wiring is added.
    """

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "mlx"

    @property
    def provider_type(self) -> str:
        return "local"

    @property
    def supported_modalities(self) -> set[str]:
        return {"text", "audio", "image"}

    def is_available(self) -> bool:
        return self._enabled

    async def generate(self, request: ModelRequest) -> str:
        raise NotImplementedError("MLX provider not implemented yet")
