"""
Provider adapters for hybrid model routing.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
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
    """MLX local provider with task-aware model selection and CLI runners."""

    _SMALL_TASKS = {
        "status_query",
        "time_query",
        "greeting",
        "farewell",
        "acknowledgement",
    }
    _CODING_HINTS = ("coding", "code", "programming", "software_delivery")
    _REASONING_HINTS = ("analysis", "reason", "planning", "strategy", "architecture")
    _DEEP_HINTS = ("research", "deep", "long_form")

    def __init__(
        self,
        *,
        enabled: bool = False,
        python_executable: str = "python3",
        text_runner_module: str = "mlx_lm.generate",
        image_runner_module: str = "mlx_vlm.generate",
        audio_runner_module: str = "mlx_whisper.transcribe",
        timeout_seconds: float = 180.0,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        dry_run: bool = False,
        runtime_manager: Optional[LocalModelRuntimeManager] = None,
        model_sizes_gb: Optional[Dict[str, float]] = None,
        text_model: str = "",
        text_model_small: str = "",
        text_model_coding: str = "",
        text_model_reasoning: str = "",
        text_model_deep_research: str = "",
        image_model: str = "",
        audio_model: str = "",
        enable_reasoning_model: bool = True,
        enable_deep_research_model: bool = True,
    ) -> None:
        self._enabled = bool(enabled)
        self._python = str(python_executable or "python3").strip()
        self._text_runner_module = str(text_runner_module or "").strip()
        self._image_runner_module = str(image_runner_module or "").strip()
        self._audio_runner_module = str(audio_runner_module or "").strip()
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._temperature = float(temperature)
        self._max_tokens = max(64, int(max_tokens))
        self._dry_run = bool(dry_run)
        self._runtime = runtime_manager
        self._model_sizes_gb = model_sizes_gb or {}
        self._text_model = str(text_model or "").strip()
        self._text_model_small = str(text_model_small or "").strip()
        self._text_model_coding = str(text_model_coding or "").strip()
        self._text_model_reasoning = str(text_model_reasoning or "").strip()
        self._text_model_deep_research = str(text_model_deep_research or "").strip()
        self._image_model = str(image_model or "").strip()
        self._audio_model = str(audio_model or "").strip()
        self._enable_reasoning_model = bool(enable_reasoning_model)
        self._enable_deep_research_model = bool(enable_deep_research_model)

    @property
    def name(self) -> str:
        return "mlx"

    @property
    def provider_type(self) -> str:
        return "local"

    @property
    def supported_modalities(self) -> set[str]:
        mods = {"text"}
        if self._image_model and self._image_runner_module:
            mods.add("image")
        if self._audio_model and self._audio_runner_module:
            mods.add("audio")
        return mods

    def is_available(self) -> bool:
        if not self._enabled:
            return False
        if self._python and "/" in self._python:
            return Path(self._python).exists()
        return shutil.which(self._python) is not None

    async def generate(self, request: ModelRequest) -> str:
        model_name = self._select_model(request)
        model_size = float(self._model_sizes_gb.get(model_name, 0.0))
        if self._runtime:
            self._runtime.ensure_capacity(model_size)
            self._runtime.mark_loaded(
                LocalModelSpec(
                    name=model_name,
                    modality=request.modality,
                    size_gb=model_size,
                    backend="mlx",
                )
            )
            self._runtime.mark_in_use(model_name)

        try:
            if self._dry_run:
                return f"[mlx dry-run:{model_name}] {request.prompt[:200]}".strip()
            if request.modality == "text":
                cmd = self._build_text_command(model_name=model_name, prompt=request.prompt)
            elif request.modality == "image":
                cmd = self._build_image_command(model_name=model_name, request=request)
            elif request.modality == "audio":
                cmd = self._build_audio_command(model_name=model_name, request=request)
            else:
                raise RuntimeError(f"Unsupported modality for MLXProvider: {request.modality}")
            output = await self._run_subprocess(cmd)
            normalized = self._normalize_output(output)
            if not normalized:
                raise RuntimeError("MLX command returned empty output")
            return normalized
        finally:
            if self._runtime:
                self._runtime.mark_released(model_name)

    def _select_model(self, request: ModelRequest) -> str:
        modality = request.modality.strip().lower()
        if modality == "image":
            if not self._image_model:
                raise RuntimeError("MLX image model not configured")
            return self._image_model
        if modality == "audio":
            if not self._audio_model:
                raise RuntimeError("MLX audio model not configured")
            return self._audio_model

        task = request.task_type.strip().lower()
        if task in self._SMALL_TASKS and self._text_model_small:
            return self._text_model_small
        if any(h in task for h in self._CODING_HINTS) and self._text_model_coding:
            return self._text_model_coding
        if (
            self._enable_deep_research_model
            and any(h in task for h in self._DEEP_HINTS)
            and self._text_model_deep_research
        ):
            return self._text_model_deep_research
        if (
            self._enable_reasoning_model
            and any(h in task for h in self._REASONING_HINTS)
            and self._text_model_reasoning
        ):
            return self._text_model_reasoning
        if self._text_model:
            return self._text_model
        raise RuntimeError("MLX text model not configured")

    def _build_text_command(self, *, model_name: str, prompt: str) -> list[str]:
        if not self._text_runner_module:
            raise RuntimeError("MLX text runner module not configured")
        return [
            self._python,
            "-m",
            self._text_runner_module,
            "--model",
            model_name,
            "--prompt",
            prompt,
            "--max-tokens",
            str(self._max_tokens),
            "--temp",
            str(self._temperature),
        ]

    def _build_image_command(self, *, model_name: str, request: ModelRequest) -> list[str]:
        if not self._image_runner_module:
            raise RuntimeError("MLX image runner module not configured")
        image_path = self._extract_media_path(request, keys=("image_path", "image_file", "image"))
        return [
            self._python,
            "-m",
            self._image_runner_module,
            "--model",
            model_name,
            "--prompt",
            request.prompt,
            "--image",
            image_path,
            "--max-tokens",
            str(self._max_tokens),
            "--temp",
            str(self._temperature),
        ]

    def _build_audio_command(self, *, model_name: str, request: ModelRequest) -> list[str]:
        if not self._audio_runner_module:
            raise RuntimeError("MLX audio runner module not configured")
        audio_path = self._extract_media_path(request, keys=("audio_path", "audio_file", "audio"))
        return [
            self._python,
            "-m",
            self._audio_runner_module,
            "--model",
            model_name,
            audio_path,
        ]

    def _extract_media_path(self, request: ModelRequest, *, keys: tuple[str, ...]) -> str:
        for key in keys:
            candidate = request.media.get(key)
            if isinstance(candidate, str) and candidate.strip():
                p = Path(candidate.strip()).expanduser()
                if p.exists() and p.is_file():
                    return str(p)
        for list_key in ("images", "audio_files"):
            candidate = request.media.get(list_key)
            if isinstance(candidate, list) and candidate:
                first = candidate[0]
                if isinstance(first, str) and first.strip():
                    p = Path(first.strip()).expanduser()
                    if p.exists() and p.is_file():
                        return str(p)
        raise RuntimeError(f"MLX media input missing. expected keys={','.join(keys)}")

    async def _run_subprocess(self, cmd: list[str]) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise RuntimeError(f"MLX command timeout ({self._timeout_seconds}s)") from exc
        if proc.returncode != 0:
            err = (stderr or b"").decode("utf-8", errors="ignore").strip()
            out = (stdout or b"").decode("utf-8", errors="ignore").strip()
            hint = err or out or f"exit_code={proc.returncode}"
            raise RuntimeError(f"MLX command failed: {hint[-600:]}")
        return (stdout or b"").decode("utf-8", errors="ignore")

    @staticmethod
    def _normalize_output(raw: str) -> str:
        lines = [ln.strip() for ln in str(raw or "").splitlines() if ln.strip()]
        if not lines:
            return ""
        if len(lines) == 1:
            return lines[0]
        # Many MLX CLIs print banners/metadata before the answer.
        return lines[-1]
