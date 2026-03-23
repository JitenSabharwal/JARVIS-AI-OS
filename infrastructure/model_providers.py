"""
Provider adapters for hybrid model routing.
"""

from __future__ import annotations

import asyncio
import json
import re
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
            self._runtime.ensure_capacity(model_size, target_model_name=model_name)
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
        "summarization",
    }
    _CODING_HINTS = ("coding", "code", "programming", "software_delivery")
    _REASONING_HINTS = ("analysis", "reason", "planning", "strategy", "architecture")
    _DEEP_HINTS = ("research", "deep", "long_form")
    _HEAVY_HINTS = (
        "analyze",
        "analysis",
        "tradeoff",
        "architecture",
        "design",
        "benchmark",
        "investigate",
        "root cause",
        "research",
        "compare",
    )

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
        persistent_enabled: bool = False,
        persistent_base_url: str = "http://127.0.0.1:8004",
        persistent_endpoint: str = "/v1/chat/completions",
        persistent_api_key: str = "",
        persistent_fallback_cli: bool = True,
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
        self._persistent_enabled = bool(persistent_enabled)
        self._persistent_base_url = str(persistent_base_url or "http://127.0.0.1:8004").rstrip("/")
        endpoint = str(persistent_endpoint or "/v1/chat/completions").strip()
        if endpoint and not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        self._persistent_endpoint = endpoint or "/v1/chat/completions"
        self._persistent_api_key = str(persistent_api_key or "").strip()
        self._persistent_fallback_cli = bool(persistent_fallback_cli)

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
        # Voice requests are transcribed to text upstream; execute as text generation.
        effective_modality = "text" if request.modality == "voice" else request.modality
        model_name = self._select_model(request)
        model_size = float(self._model_sizes_gb.get(model_name, 0.0))
        if self._runtime:
            self._runtime.ensure_capacity(model_size, target_model_name=model_name)
            self._runtime.mark_loaded(
                LocalModelSpec(
                    name=model_name,
                    modality=effective_modality,
                    size_gb=model_size,
                    backend="mlx",
                )
            )
            self._runtime.mark_in_use(model_name)

        try:
            if self._dry_run:
                return f"[mlx dry-run:{model_name}] {request.prompt[:200]}".strip()
            max_tokens = self._max_tokens_for_request(request)
            if effective_modality == "text" and self._persistent_enabled:
                try:
                    out = await self._run_persistent_text_request(
                        model_name=model_name,
                        prompt=request.prompt,
                        max_tokens=max_tokens,
                    )
                    normalized = self._normalize_output(out)
                    if normalized:
                        return normalized
                    raise RuntimeError("MLX persistent response was empty")
                except Exception:
                    if not self._persistent_fallback_cli:
                        raise
            if effective_modality == "text":
                cmd = self._build_text_command(
                    model_name=model_name,
                    prompt=request.prompt,
                    max_tokens=max_tokens,
                )
            elif effective_modality == "image":
                cmd = self._build_image_command(model_name=model_name, request=request)
            elif effective_modality == "audio":
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

        tier = str((request.metadata or {}).get("model_tier", "")).strip().lower()
        if tier == "small" and self._text_model_small:
            return self._text_model_small
        if tier in {"large", "base"} and self._text_model:
            return self._text_model

        task = request.task_type.strip().lower()
        if task in self._SMALL_TASKS and self._text_model_small:
            return self._text_model_small
        if any(h in task for h in self._CODING_HINTS) and self._text_model_coding:
            return self._text_model_coding
        if (
            self._enable_deep_research_model
            and (any(h in task for h in self._DEEP_HINTS) or self._is_deep_text_request(request))
            and self._text_model_deep_research
        ):
            return self._text_model_deep_research
        if (
            self._enable_reasoning_model
            and (any(h in task for h in self._REASONING_HINTS) or self._is_heavy_text_request(request))
            and self._text_model_reasoning
        ):
            return self._text_model_reasoning
        if self._text_model_small and self._is_lightweight_text_request(request):
            return self._text_model_small
        if self._text_model:
            return self._text_model
        raise RuntimeError("MLX text model not configured")

    def _build_text_command(self, *, model_name: str, prompt: str, max_tokens: int | None = None) -> list[str]:
        if not self._text_runner_module:
            raise RuntimeError("MLX text runner module not configured")
        resolved_model = self._resolve_mlx_model_arg(model_name)
        out_tokens = max(64, int(max_tokens if max_tokens is not None else self._max_tokens))
        if self._text_runner_module == "mlx_lm.generate":
            # Upstream deprecates: python -m mlx_lm.generate ...
            # Preferred form: python -m mlx_lm generate ...
            return [
                self._python,
                "-m",
                "mlx_lm",
                "generate",
                "--model",
                resolved_model,
                "--prompt",
                prompt,
                "--max-tokens",
                str(out_tokens),
                "--temp",
                str(self._temperature),
            ]
        return [
            self._python,
            "-m",
            self._text_runner_module,
            "--model",
            resolved_model,
            "--prompt",
            prompt,
            "--max-tokens",
            str(out_tokens),
            "--temp",
            str(self._temperature),
        ]

    @staticmethod
    def _request_user_text(request: ModelRequest) -> str:
        md = request.metadata if isinstance(request.metadata, dict) else {}
        user_input = str(md.get("user_input", "")).strip()
        if user_input:
            return user_input
        return str(request.prompt or "").strip()

    def _is_lightweight_text_request(self, request: ModelRequest) -> bool:
        if request.modality.strip().lower() != "text":
            return False
        task = request.task_type.strip().lower()
        if task in {"weather_query", "help_request", "confirmation", "negation", "cancel"}:
            return True
        user_text = self._request_user_text(request)
        if not user_text:
            return False
        word_count = len(user_text.split())
        if word_count <= 14 and not any(h in user_text.lower() for h in self._HEAVY_HINTS):
            return True
        return False

    def _is_heavy_text_request(self, request: ModelRequest) -> bool:
        if request.modality.strip().lower() != "text":
            return False
        user_text = self._request_user_text(request).lower()
        word_count = len(user_text.split())
        if word_count >= 40:
            return True
        return any(h in user_text for h in self._HEAVY_HINTS)

    def _is_deep_text_request(self, request: ModelRequest) -> bool:
        user_text = self._request_user_text(request).lower()
        return any(h in user_text for h in ("deep research", "long form", "comprehensive report"))

    def _max_tokens_for_request(self, request: ModelRequest) -> int:
        task = request.task_type.strip().lower()
        if task == "summarization":
            # Give the light summarizer enough room to avoid truncation when models
            # emit brief preambles before the final answer.
            return min(self._max_tokens, 256)
        if task in {"weather_query", "status_query", "time_query", "greeting", "farewell", "acknowledgement"}:
            return min(self._max_tokens, 192)
        if self._is_lightweight_text_request(request):
            return min(self._max_tokens, 256)
        return self._max_tokens

    def _build_image_command(self, *, model_name: str, request: ModelRequest) -> list[str]:
        if not self._image_runner_module:
            raise RuntimeError("MLX image runner module not configured")
        resolved_model = self._resolve_mlx_model_arg(model_name)
        image_path = self._extract_media_path(request, keys=("image_path", "image_file", "image"))
        return [
            self._python,
            "-m",
            self._image_runner_module,
            "--model",
            resolved_model,
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
        resolved_model = self._resolve_mlx_model_arg(model_name)
        audio_path = self._extract_media_path(request, keys=("audio_path", "audio_file", "audio"))
        return [
            self._python,
            "-m",
            self._audio_runner_module,
            "--model",
            resolved_model,
            audio_path,
        ]

    @staticmethod
    def _resolve_mlx_model_arg(model_name: str) -> str:
        """
        Accept HF cache folder names (models--org--repo) and convert them to repo ids (org/repo)
        for mlx_* CLIs that validate model ids through huggingface_hub.
        """
        text = str(model_name or "").strip()
        if text.startswith("models--"):
            # Keep only the first org/repo separator conversion.
            return text.replace("models--", "", 1).replace("--", "/", 1)
        return text

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

    async def _run_persistent_text_request(self, *, model_name: str, prompt: str, max_tokens: int) -> str:
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for MLX persistent mode")
        url = f"{self._persistent_base_url}{self._persistent_endpoint}"
        resolved_model = self._resolve_mlx_model_arg(model_name)
        headers = {"Content-Type": "application/json"}
        if self._persistent_api_key:
            headers["Authorization"] = f"Bearer {self._persistent_api_key}"
        endpoint_low = self._persistent_endpoint.lower()
        if endpoint_low.endswith("/chat/completions"):
            payload: Dict[str, Any] = {
                "model": resolved_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(max_tokens),
                "temperature": float(self._temperature),
                "stream": False,
            }
        else:
            payload = {
                "model": resolved_model,
                "prompt": prompt,
                "max_tokens": int(max_tokens),
                "temperature": float(self._temperature),
                "stream": False,
            }

        timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                raw_text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"MLX persistent error {response.status}: {raw_text[-600:]}")
                data: Dict[str, Any] | None = None
                try:
                    data = json.loads(raw_text)
                except Exception:
                    data = None
                if data is None:
                    return str(raw_text).strip()
                parsed = self._extract_persistent_text(data)
                if parsed:
                    return parsed
                return str(raw_text).strip()

    @staticmethod
    def _extract_persistent_text(data: Dict[str, Any]) -> str:
        txt = str(data.get("response", "")).strip()
        if txt:
            return txt
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            msg = first.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            part = str(item.get("text", "")).strip()
                            if part:
                                parts.append(part)
                    joined = " ".join(parts).strip()
                    if joined:
                        return joined
            txt = str(first.get("text", "")).strip()
            if txt:
                return txt
            delta = first.get("delta")
            if isinstance(delta, dict):
                txt = str(delta.get("content", "")).strip()
                if txt:
                    return txt
        txt = str(data.get("text", "")).strip()
        if txt:
            return txt
        result = data.get("result")
        if isinstance(result, dict):
            txt = str(result.get("text", "")).strip()
            if txt:
                return txt
        return ""

    @staticmethod
    def _normalize_output(raw: str) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        text = MLXProvider._sanitize_assistant_output(text)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        def _is_separator(line: str) -> bool:
            return bool(re.fullmatch(r"[-=_*~`]{3,}", line))

        def _is_telemetry(line: str) -> bool:
            lower = line.lower()
            prefixes = (
                "prompt:",
                "prompt tokens:",
                "completion tokens:",
                "generation tokens:",
                "tokens/sec",
                "tokens per second",
                "latency:",
                "eval time:",
                "total time:",
                "peak memory:",
                "memory:",
                "temperature:",
            )
            return lower.startswith(prefixes)

        content = [ln for ln in lines if not _is_separator(ln) and not _is_telemetry(ln)]
        if not content:
            return ""
        return "\n".join(content).strip()

    @staticmethod
    def _sanitize_assistant_output(text: str) -> str:
        cleaned = text.strip()
        if re.match(r"(?is)\A\s*Thinking Process\s*:", cleaned):
            # Handle verbose chain-of-thought style dumps.
            final_match = re.search(
                r'(?im)^\s*(?:Final Final|Final Decision|JARVIS Response|Response)\s*:\s*(.+?)\s*$',
                cleaned,
            )
            if final_match:
                candidate = final_match.group(1).strip().strip("\"' ")
                if candidate:
                    return candidate
            quoted_candidates = re.findall(r'(?m)^\s*"([^"\n]{8,400})"\s*$', cleaned)
            if quoted_candidates:
                return quoted_candidates[-1].strip()
            return ""
        cleaned = re.sub(
            r"^Calling `python -m mlx_lm\.generate\.\.\.` directly is deprecated\..*$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        # Never expose reasoning traces if model emits them.
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(
            r"(?ims)^\s*#{0,3}\s*Reasoning/Explanation\s*$.*",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"(?is)\A\s*Let me analyze this .*?step by step:.*?(?:\s*</think>|\Z)",
            "",
            cleaned,
        )
        cleaned = MLXProvider._strip_leading_analysis_preamble(cleaned)
        cleaned = re.sub(r"(?im)^\s*</think>\s*$", "", cleaned)
        cleaned = re.sub(
            r"(?is)\*\*Analysis of Context Fusion Data:\*\*.*?(?=\*\*JARVIS Response:\*\*)",
            "",
            cleaned,
        )
        cleaned = re.sub(r"(?im)^\s*\*\*JARVIS Response:\*\*\s*", "", cleaned)
        cleaned = re.sub(
            r"(?im)^\s*Generation:\s*\d+\s+tokens,.*$",
            "",
            cleaned,
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _strip_leading_analysis_preamble(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        if not re.match(r"(?is)\A\s*Let me analy[sz]e\b", raw):
            return raw
        lines = [ln.rstrip() for ln in raw.splitlines()]
        if not lines:
            return raw

        reasoning_patterns = (
            r"^\s*Let me analy[sz]e\b",
            r"^\s*\d+\.\s+",
            r"^\s*[-*]\s+",
            r"^\s*(the user|since this|i should|i need to|i can|this is|to be|and ask|keep it)\b",
        )

        start_idx = -1
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if any(re.search(pat, stripped, flags=re.IGNORECASE) for pat in reasoning_patterns):
                continue
            if stripped.endswith(":"):
                continue
            if re.match(r"(?i)^(absolutely|sure|yes|certainly|of course|here|great|hello|hi)\b", stripped):
                start_idx = idx
                break
            if len(stripped.split()) >= 6:
                start_idx = idx
                break

        if start_idx < 0:
            return ""
        kept = "\n".join(lines[start_idx:]).strip()
        return kept or ""
