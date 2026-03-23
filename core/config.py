"""
Advanced configuration management for the JARVIS AI OS system.

Configuration is assembled from three sources, in ascending priority order:

1. Built-in defaults (defined in the dataclasses below).
2. A YAML configuration file (``config.yaml`` by default, or the path set
   via the ``JARVIS_CONFIG_FILE`` environment variable).
3. A ``.env`` file (``".env"`` by default, or ``JARVIS_ENV_FILE``).
4. Process environment variables (highest priority).

Usage::

    from core.config import get_config

    cfg = get_config()
    print(cfg.agent.max_agents)

Calling :func:`get_config` is safe from multiple threads; it initialises the
singleton on the first call and returns the same instance thereafter.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies (yaml, dotenv)
# ---------------------------------------------------------------------------

try:
    import yaml  # type: ignore

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    logger.debug("PyYAML not installed; YAML config file support disabled")

try:
    from dotenv import load_dotenv  # type: ignore

    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False
    logger.debug("python-dotenv not installed; .env file support disabled")

# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Configuration for the agent subsystem."""

    max_agents: int = 10
    """Maximum number of concurrent agents."""

    default_timeout: float = 30.0
    """Default task timeout in seconds."""

    heartbeat_interval: float = 5.0
    """How often (seconds) agents send heartbeat signals."""

    max_retries: int = 3
    """Default number of retries for failed agent tasks."""

    retry_delay: float = 1.0
    """Initial delay (seconds) between retries."""

    retry_backoff: float = 2.0
    """Backoff multiplier applied to *retry_delay* on each retry."""

    default_agent_type: str = "reactive"
    """Agent type used when none is specified: reactive | deliberative | hybrid."""

    registry_backend: str = "in_memory"
    """Agent registry backend: ``in_memory`` or ``redis``."""

    coordinator_pool_size: int = 1
    """Number of coordinator agents to register at startup."""

    analyst_pool_size: int = 1
    """Number of analyst agents to register at startup."""

    developer_pool_size: int = 1
    """Number of developer agents to register at startup."""

    manager_pool_size: int = 1
    """Number of manager agents to register at startup."""


@dataclass
class SkillsConfig:
    """Configuration for the skills subsystem."""

    skill_directories: List[str] = field(default_factory=lambda: ["skills"])
    """Directories scanned for skill modules at start-up."""

    auto_discover: bool = True
    """Automatically discover and register skills on start-up."""

    max_execution_time: float = 60.0
    """Maximum time (seconds) allowed for a single skill execution."""

    sandbox_enabled: bool = False
    """Run skills in a sandboxed subprocess (experimental)."""


@dataclass
class MemoryConfig:
    """Configuration for the memory subsystem."""

    backend: str = "in_memory"
    """Storage backend: ``in_memory``, ``redis``, or ``sqlite``."""

    redis_url: str = "redis://localhost:6379/0"
    """Redis connection URL (used when *backend* is ``redis``)."""

    sqlite_path: str = "data/jarvis_memory.db"
    """SQLite database file path (used when *backend* is ``sqlite``)."""

    max_short_term_items: int = 1000
    """Maximum entries kept in short-term memory."""

    ttl_short_term: int = 3600
    """TTL (seconds) for short-term memory entries."""

    ttl_long_term: int = 86400 * 30
    """TTL (seconds) for long-term memory entries (30 days)."""

    enable_semantic_search: bool = False
    """Enable vector-based semantic search over stored memories."""

    embedding_model: str = "text-embedding-ada-002"
    """Model name used for embedding generation."""


@dataclass
class InfrastructureConfig:
    """Configuration for infrastructure components (logging, message bus, etc.)."""

    log_level: str = "INFO"
    """Root log level: DEBUG | INFO | WARNING | ERROR | CRITICAL."""

    log_file: Optional[str] = None
    """Path to a rotating log file.  ``None`` disables file logging."""

    log_max_bytes: int = 10 * 1024 * 1024
    """Maximum log-file size in bytes before rotation (default 10 MB)."""

    log_backup_count: int = 5
    """Number of rotated log-file backups to keep."""

    message_bus_backend: str = "in_memory"
    """Message bus backend: ``in_memory`` or ``redis``."""

    redis_url: str = "redis://localhost:6379/1"
    """Redis URL for the message bus (when *message_bus_backend* is ``redis``)."""

    max_queue_size: int = 1000
    """Maximum number of messages held in any single queue."""

    event_loop_policy: str = "default"
    """asyncio event loop policy: ``default`` or ``uvloop``."""

    metrics_enabled: bool = False
    """Enable Prometheus-compatible metrics collection."""

    metrics_port: int = 9090
    """Port on which the metrics HTTP endpoint listens."""

    slo_api_p95_ms: float = 1500.0
    """Warn threshold for API request p95 latency in milliseconds."""

    slo_api_error_rate_pct: float = 5.0
    """Warn threshold for API error rate percentage (4xx+5xx over responses)."""

    slo_run_command_p95_ms: float = 30000.0
    """Warn threshold for run_command p95 latency in milliseconds."""

    slo_connectors_unhealthy_count: float = 1.0
    """Warn threshold for count of unhealthy connectors."""

    slo_automation_dead_letters_backlog: float = 20.0
    """Warn threshold for pending automation dead letters."""

    slo_min_samples: int = 20
    """Minimum samples before enforcing SLO thresholds."""


@dataclass
class VoiceConfig:
    """Configuration for the voice interface."""

    enabled: bool = False
    """Whether the voice interface is active at startup."""

    stt_engine: str = "whisper"
    """Speech-to-text engine: ``whisper``, ``google``, ``azure``."""

    tts_engine: str = "pyttsx3"
    """Text-to-speech engine: ``pyttsx3``, ``google``, ``azure``, ``elevenlabs``."""

    wake_word: str = "jarvis"
    """Wake-word that activates listening mode."""

    language: str = "en-US"
    """BCP-47 language tag used for both STT and TTS."""

    sample_rate: int = 16000
    """Audio sample rate in Hz."""

    silence_threshold: float = 0.03
    """Amplitude below which audio is treated as silence."""

    silence_duration: float = 1.5
    """Seconds of silence required to end an utterance."""

    tts_voice: str = ""
    """Voice identifier passed to the TTS engine (engine-specific)."""

    tts_rate: int = 150
    """Speech rate in words per minute."""

    tts_volume: float = 1.0
    """TTS volume: 0.0 (silent) to 1.0 (full)."""


@dataclass
class APIConfig:
    """Configuration for external API integrations."""

    host: str = "0.0.0.0"
    """Bind host for the internal REST API interface."""

    port: int = 8080
    """Bind port for the internal REST API interface."""

    token: str = ""
    """Optional bearer token for REST API authentication."""

    openai_api_key: str = ""
    """OpenAI API key.  Also read from ``OPENAI_API_KEY`` env var."""

    openai_model: str = "gpt-4o"
    """Default OpenAI chat model."""

    openai_max_tokens: int = 4096
    """Maximum tokens per OpenAI completion request."""

    openai_temperature: float = 0.7
    """Sampling temperature for OpenAI completions."""

    anthropic_api_key: str = ""
    """Anthropic API key.  Also read from ``ANTHROPIC_API_KEY`` env var."""

    anthropic_model: str = "claude-3-5-sonnet-20241022"
    """Default Anthropic model."""

    google_api_key: str = ""
    """Google API key for Gemini / other Google services."""

    google_model: str = "gemini-pro"
    """Default Google Gemini model."""

    serper_api_key: str = ""
    """Serper.dev API key for web search skills."""

    weather_api_key: str = ""
    """OpenWeatherMap (or compatible) API key."""

    request_timeout: float = 30.0
    """Default HTTP request timeout in seconds."""

    max_retries: int = 3
    """Number of retries for failed API calls."""


@dataclass
class ModelRuntimeConfig:
    """Configuration for hybrid model provider routing/runtime."""

    enabled: bool = True
    """Enable hybrid router/provider integration."""

    local_provider: str = "mlx"
    """Local provider name: ``ollama`` or ``mlx``."""

    api_provider: str = "cohere"
    """API provider name: ``cohere`` or empty."""

    fallback_enabled: bool = True
    """Allow failover between local and API providers."""

    shadow_mode: bool = False
    """When enabled, also invoke secondary route for telemetry-only comparison."""

    # Local runtime budget controls
    memory_budget_gb: float = 35.0
    """RAM budget reserved for active local models."""

    total_memory_gb: float = 48.0
    """Total system RAM for planning/reference."""

    max_parallel_models: int = 3
    """Maximum local models intended to run in parallel."""

    large_model_threshold_gb: float = 18.0
    """Models at/above this size are treated as large for concurrency policy."""

    single_large_model_mode: bool = True
    """When enabled, large models run exclusively (no parallel local model execution)."""

    auto_unload: bool = True
    """Auto-unload idle local models to stay under budget."""

    keep_base_model_loaded: bool = True
    """Keep the configured base local model resident to reduce load/unload churn."""

    base_model_name: str = ""
    """Optional explicit base model to pin in memory; defaults to provider text model."""

    unload_base_when_required: bool = True
    """Allow unloading pinned base model only when required to fit a heavier model."""

    local_timeout_seconds: float = 45.0
    api_timeout_seconds: float = 30.0

    # Ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_text_model: str = "qwen2.5:7b"
    ollama_text_model_size_gb: float = 8.0
    ollama_image_model: str = ""
    ollama_image_model_size_gb: float = 12.0
    ollama_audio_model: str = ""
    ollama_audio_model_size_gb: float = 8.0

    # Cohere
    cohere_api_key: str = ""
    cohere_base_url: str = "https://api.cohere.com/v2"
    cohere_text_model: str = "command-r-plus"

    # MLX / VLMX local runtime
    mlx_enabled: bool = True
    mlx_timeout_seconds: float = 180.0
    mlx_command_python: str = "python3"
    mlx_text_runner_module: str = "mlx_lm.generate"
    mlx_image_runner_module: str = "mlx_vlm.generate"
    mlx_audio_runner_module: str = "mlx_whisper.transcribe"
    mlx_temperature: float = 0.2
    mlx_max_tokens: int = 2048
    mlx_dry_run: bool = False
    mlx_enable_reasoning_model: bool = True
    mlx_enable_deep_research_model: bool = True
    mlx_text_model: str = "models--Mxfp4-Lab--Qwen3.5-35B-A3B-Claude-4.6-Opus-Distilled-MXFP4-MLX"
    mlx_text_model_size_gb: float = 24.0
    mlx_text_model_small: str = "models--mlx-community--Qwen3.5-4B-MLX-4bit"
    mlx_text_model_small_size_gb: float = 4.0
    mlx_text_model_coding: str = "models--mlx-community--Qwen2.5-Coder-14B-Instruct-4bit"
    mlx_text_model_coding_size_gb: float = 10.0
    mlx_text_model_reasoning: str = "models--TheCluster--Qwen3.5-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-MLX-mxfp4"
    mlx_text_model_reasoning_size_gb: float = 28.0
    mlx_text_model_deep_research: str = "models--lmstudio-community--Qwen3-Next-80B-A3B-Instruct-MLX-4bit"
    mlx_text_model_deep_research_size_gb: float = 34.0
    mlx_audio_model: str = "models--mlx-community--whisper-large-v3-turbo"
    mlx_audio_model_size_gb: float = 3.0
    mlx_image_model: str = "models--mlx-community--Qwen2-VL-2B-Instruct-4bit"
    mlx_image_model_size_gb: float = 3.0
    mlx_reranker_model_small: str = "models--Qwen--Qwen3-Reranker-0.6B"
    mlx_reranker_model: str = "models--Qwen--Qwen3-Reranker-4B"
    mlx_persistent_enabled: bool = False
    mlx_persistent_base_url: str = "http://127.0.0.1:8004"
    mlx_persistent_endpoint: str = "/v1/chat/completions"
    mlx_persistent_api_key: str = ""
    mlx_persistent_fallback_cli: bool = True

    prewarm_enabled: bool = True
    prewarm_prompt: str = "Reply with just: ready"
    prewarm_timeout_seconds: float = 8.0
    prewarm_dual_tier: bool = False


@dataclass
class DeliveryConfig:
    """Configuration for production software delivery execution."""

    command_execution_enabled: bool = True
    """Allow subprocess execution for gate/deploy commands."""

    command_timeout_seconds: float = 120.0
    """Default timeout for gate/deploy commands."""

    max_output_chars: int = 2000
    """Maximum captured output chars per command."""

    deploy_max_retries: int = 1
    """Retry count for retryable deploy adapter failures."""

    deploy_retry_backoff_seconds: float = 1.0
    """Backoff base (seconds) between deploy retries."""

    allowed_deploy_targets: List[str] = field(
        default_factory=lambda: ["local", "aws", "gcp", "vercel"]
    )
    """Allowed deploy targets enforced by the delivery engine."""

    default_working_dir: str = ""
    """Optional default working directory for command execution."""

    local_deploy_command: str = ""
    aws_deploy_command: str = ""
    gcp_deploy_command: str = ""
    vercel_deploy_command: str = ""


@dataclass
class ResearchConfig:
    """Configuration for hierarchical RAG, graph persistence, and LangGraph."""

    hierarchical_rag_enabled: bool = True
    """Enable hierarchical tree-based retrieval augmentation."""

    rag_neighbor_expansion: bool = True
    """Include parent/sibling/child context expansion in retrieval."""

    embedding_backend: str = "local_deterministic"
    """Embedding backend: local_deterministic | mlx_clip."""

    embedding_dim: int = 64
    """Embedding vector dimension for local deterministic backend."""

    embedding_multimodal_backend: str = "mlx_clip"
    """Secondary embedding backend for multimodal (image-text aligned) retrieval."""

    embedding_multimodal_dim: int = 64
    """Embedding vector dimension for multimodal backend."""

    rag_fusion_text_weight: float = 0.65
    """Fusion weight for text embedding similarity."""

    rag_fusion_multimodal_weight: float = 0.35
    """Fusion weight for multimodal embedding similarity."""

    rag_reranker_enabled: bool = True
    """Enable a reranker pass over top-k RAG candidates."""

    rag_reranker_top_k: int = 24
    """Top-k candidates considered in reranker pass."""

    vector_store: str = "memory"
    """RAG node store backend: memory | chroma."""

    chroma_path: str = "data/research/chroma"
    """Persistent path for ChromaDB when vector_store=chroma."""

    chroma_collection: str = "jarvis_rag_nodes"
    """ChromaDB collection name for RAG nodes."""

    state_path: str = "data/research/state.json"
    """JSON persistence path for research sources/watchlists metadata."""

    mlx_embedding_text_model: str = "models--mlx-community--Qwen3-Embedding-4B-mxfp8"
    """Preferred MLX text embedding model repo/cache id."""

    mlx_embedding_text_model_fallback: str = "models--mlx-community--Qwen3-Embedding-0.6B-mxfp8"
    """Fallback MLX text embedding model when memory is constrained."""

    mlx_embedding_image_model: str = "models--mlx-community--clip-vit-large-patch14"
    """MLX image embedding model (CLIP family)."""

    mlx_reranker_model: str = "models--Qwen--Qwen3-Reranker-4B"
    """Preferred reranker model for retrieval scoring."""

    mlx_reranker_model_fallback: str = "models--Qwen--Qwen3-Reranker-0.6B"
    """Fallback reranker model for low-latency/low-memory routes."""

    neo4j_enabled: bool = False
    """Enable Neo4j relationship persistence for research/doc nodes."""

    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    langgraph_enabled: bool = False
    """Enable LangGraph-assisted workflow wave planning when dependency exists."""

    langgraph_max_wave_size: int = 0
    """Optional cap on parallel steps per execution wave (0 = uncapped)."""


# ---------------------------------------------------------------------------
# Root config dataclass
# ---------------------------------------------------------------------------


@dataclass
class JARVISConfig:
    """Top-level configuration object for the JARVIS AI OS system.

    Instantiate directly for testing or call :func:`get_config` for the
    application-wide singleton.
    """

    # Identity
    app_name: str = "JARVIS AI OS"
    version: str = "0.1.0"
    environment: str = "development"
    """Runtime environment: ``development``, ``staging``, or ``production``."""

    debug: bool = False
    """Enable verbose debug output across all subsystems."""

    data_dir: str = "data"
    """Root directory for persistent data (databases, caches, etc.)."""

    # Subsystem configs
    agent: AgentConfig = field(default_factory=AgentConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    dual_tier_response_enabled: bool = False

    def is_production(self) -> bool:
        """Return ``True`` when *environment* is ``"production"``."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Return ``True`` when *environment* is ``"development"``."""
        return self.environment.lower() == "development"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV_PREFIX = "JARVIS_"
_SECTION_PREFIXES: dict[str, str] = {
    "agent": "JARVIS_AGENT_",
    "skills": "JARVIS_SKILLS_",
    "memory": "JARVIS_MEMORY_",
    "infrastructure": "JARVIS_INFRA_",
    "voice": "JARVIS_VOICE_",
    "api": "JARVIS_API_",
    "model": "JARVIS_MODEL_",
    "delivery": "JARVIS_DELIVERY_",
    "research": "JARVIS_RESEARCH_",
}
# Direct env-var → config-path mappings for well-known keys
_DIRECT_ENV_MAP: dict[str, tuple[str, ...]] = {
    "OPENAI_API_KEY": ("api", "openai_api_key"),
    "ANTHROPIC_API_KEY": ("api", "anthropic_api_key"),
    "GOOGLE_API_KEY": ("api", "google_api_key"),
    "COHERE_API_KEY": ("model", "cohere_api_key"),
    "SERPER_API_KEY": ("api", "serper_api_key"),
    "WEATHER_API_KEY": ("api", "weather_api_key"),
    "LOG_LEVEL": ("infrastructure", "log_level"),
    "DEBUG": ("debug",),
}


_STR_TO_TYPE: dict[str, type] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "list": list,
}


def _resolve_type(target_type: Any) -> Any:
    """Resolve *target_type* to a concrete type, handling string annotations.

    Dataclass ``field.type`` may be a plain string (e.g. ``'int'``) when
    ``from __future__ import annotations`` is active or when Python stores
    annotations lazily.  This helper converts those strings to real types.
    """
    if isinstance(target_type, str):
        # Strip common generic wrappers stored as strings: "Optional[str]", "List[str]"
        stripped = target_type.strip()
        # Optional[X] → X
        if stripped.startswith("Optional[") and stripped.endswith("]"):
            inner = stripped[len("Optional["):-1]
            return _resolve_type(inner)
        if stripped.startswith("List[") and stripped.endswith("]"):
            return list
        if stripped.startswith("list[") and stripped.endswith("]"):
            return list
        return _STR_TO_TYPE.get(stripped, str)
    return target_type


def _coerce(value: str, target_type: Any) -> Any:
    """Coerce *value* (a string from env vars) to *target_type*."""
    target_type = _resolve_type(target_type)

    origin = getattr(target_type, "__origin__", None)
    # Handle Optional[X] → X  (when target_type is already a real typing object)
    if hasattr(target_type, "__args__"):
        args = [a for a in target_type.__args__ if a is not type(None)]
        if args:
            target_type = args[0]
            origin = getattr(target_type, "__origin__", None)

    if target_type is bool:
        return value.lower() in ("1", "true", "yes", "on")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is list or origin is list:
        return [v.strip() for v in value.split(",") if v.strip()]
    return value  # str or unknown → pass through


def _apply_env_to_section(section_obj: Any, prefix: str) -> None:
    """Overwrite fields on *section_obj* from env vars starting with *prefix*."""
    for f in fields(section_obj):
        env_key = f"{prefix}{f.name.upper()}"
        raw = os.environ.get(env_key)
        if raw is not None:
            try:
                setattr(section_obj, f.name, _coerce(raw, f.type))  # type: ignore[arg-type]
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Could not coerce env var %s=%r to %s: %s",
                    env_key,
                    raw,
                    f.type,
                    exc,
                )


def _apply_dict_to_section(section_obj: Any, data: dict[str, Any]) -> None:
    """Apply a mapping of field-name → value onto *section_obj*."""
    valid_fields = {f.name for f in fields(section_obj)}
    for key, value in data.items():
        if key in valid_fields:
            setattr(section_obj, key, value)
        else:
            logger.debug(
                "Unknown config key '%s' for %s; ignoring",
                key,
                type(section_obj).__name__,
            )


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------


class ConfigManager:
    """Loads and assembles a :class:`JARVISConfig` from multiple sources.

    Sources are applied in this order (last wins):
    1. Dataclass defaults.
    2. YAML config file.
    3. ``.env`` file.
    4. Environment variables.

    Args:
        config_file: Path to a YAML config file.  Defaults to the value of
            the ``JARVIS_CONFIG_FILE`` env var, then ``"config/config.yaml"``.
        env_file: Path to a ``.env`` file.  Defaults to ``JARVIS_ENV_FILE``
            env var, then ``".env"``.
    """

    def __init__(
        self,
        config_file: Optional[str] = None,
        env_file: Optional[str] = None,
    ) -> None:
        self._config_file = config_file or os.environ.get(
            "JARVIS_CONFIG_FILE", "config/config.yaml"
        )
        self._env_file = env_file or os.environ.get("JARVIS_ENV_FILE", ".env")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> JARVISConfig:
        """Build and return a fully resolved :class:`JARVISConfig`.

        Returns:
            A :class:`JARVISConfig` assembled from all available sources.
        """
        cfg = JARVISConfig()

        self._load_env_file()

        yaml_data = self._load_yaml()
        if yaml_data:
            self._apply_yaml(cfg, yaml_data)

        self._apply_environment(cfg)

        # Propagate debug flag to infrastructure
        if cfg.debug:
            cfg.infrastructure.log_level = "DEBUG"

        return cfg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_env_file(self) -> None:
        env_path = Path(self._env_file)
        if not env_path.exists():
            return
        if _DOTENV_AVAILABLE:
            load_dotenv(env_path, override=False)
            logger.debug("Loaded .env file: %s", env_path)
        else:
            # Manual minimal .env parser
            self._parse_env_file(env_path)

    @staticmethod
    def _parse_env_file(path: Path) -> None:
        """Parse a .env file and set variables into os.environ (no override)."""
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except OSError as exc:
            logger.warning("Could not read .env file %s: %s", path, exc)

    def _load_yaml(self) -> dict[str, Any]:
        if not _YAML_AVAILABLE:
            return {}
        path = Path(self._config_file)
        if not path.exists():
            logger.debug("No YAML config file found at %s", path)
            return {}
        try:
            with path.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            logger.debug("Loaded YAML config from %s", path)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("Failed to parse YAML config %s: %s", path, exc)
            return {}

    @staticmethod
    def _apply_yaml(cfg: JARVISConfig, data: dict[str, Any]) -> None:
        """Apply a parsed YAML dict onto *cfg*."""
        top_fields = {f.name for f in fields(cfg)}
        section_map: dict[str, Any] = {
            "agent": cfg.agent,
            "skills": cfg.skills,
            "memory": cfg.memory,
            "infrastructure": cfg.infrastructure,
            "voice": cfg.voice,
            "api": cfg.api,
            "model": cfg.model,
            "delivery": cfg.delivery,
            "research": cfg.research,
        }
        for key, value in data.items():
            if key in section_map and isinstance(value, dict):
                _apply_dict_to_section(section_map[key], value)
            elif key in top_fields:
                setattr(cfg, key, value)
            else:
                logger.debug("Unknown top-level YAML key '%s'; ignoring", key)

    @staticmethod
    def _apply_environment(cfg: JARVISConfig) -> None:
        """Apply environment variable overrides to *cfg*."""
        # Direct well-known mappings
        for env_key, path in _DIRECT_ENV_MAP.items():
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            if len(path) == 1:
                # Top-level field
                f_name = path[0]
                target_field = next(
                    (f for f in fields(cfg) if f.name == f_name), None
                )
                if target_field:
                    try:
                        setattr(
                            cfg,
                            f_name,
                            _coerce(raw, target_field.type),  # type: ignore[arg-type]
                        )
                    except (ValueError, TypeError) as exc:
                        logger.warning("Could not apply %s: %s", env_key, exc)
            else:
                section_name, field_name = path[0], path[1]
                section_obj = getattr(cfg, section_name, None)
                if section_obj is not None:
                    target_field = next(
                        (f for f in fields(section_obj) if f.name == field_name),
                        None,
                    )
                    if target_field:
                        try:
                            setattr(
                                section_obj,
                                field_name,
                                _coerce(raw, target_field.type),  # type: ignore[arg-type]
                            )
                        except (ValueError, TypeError) as exc:
                            logger.warning("Could not apply %s: %s", env_key, exc)

        # Section-level JARVIS_<SECTION>_<FIELD> env vars
        section_map: dict[str, Any] = {
            "agent": cfg.agent,
            "skills": cfg.skills,
            "memory": cfg.memory,
            "infrastructure": cfg.infrastructure,
            "voice": cfg.voice,
            "api": cfg.api,
            "model": cfg.model,
            "delivery": cfg.delivery,
            "research": cfg.research,
        }
        for section_name, prefix in _SECTION_PREFIXES.items():
            _apply_env_to_section(section_map[section_name], prefix)

        # Top-level JARVIS_ env vars (e.g. JARVIS_DEBUG, JARVIS_ENVIRONMENT)
        top_map = {f.name: f for f in fields(cfg)}
        for env_key, raw in os.environ.items():
            if not env_key.startswith(_ENV_PREFIX):
                continue
            suffix = env_key[len(_ENV_PREFIX):].lower()
            if suffix in top_map:
                f = top_map[suffix]
                try:
                    setattr(cfg, suffix, _coerce(raw, f.type))  # type: ignore[arg-type]
                except (ValueError, TypeError) as exc:
                    logger.warning("Could not apply %s: %s", env_key, exc)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_config_lock = threading.Lock()
_config_instance: Optional[JARVISConfig] = None


def get_config(
    *,
    config_file: Optional[str] = None,
    env_file: Optional[str] = None,
    reload: bool = False,
) -> JARVISConfig:
    """Return the application-wide :class:`JARVISConfig` singleton.

    Thread-safe.  The singleton is initialised on the first call and reused
    thereafter unless *reload* is ``True``.

    Args:
        config_file: Override the YAML config file path (only used on the
            first call or when *reload* is ``True``).
        env_file: Override the .env file path (only used on first call or
            when *reload* is ``True``).
        reload: If ``True``, discard the cached config and reload from disk.

    Returns:
        The :class:`JARVISConfig` singleton.
    """
    global _config_instance

    with _config_lock:
        if _config_instance is None or reload:
            manager = ConfigManager(config_file=config_file, env_file=env_file)
            _config_instance = manager.load()
            logger.debug(
                "Config loaded: env=%s debug=%s",
                _config_instance.environment,
                _config_instance.debug,
            )

    return _config_instance


def reset_config() -> None:
    """Reset the config singleton (intended for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "AgentConfig",
    "SkillsConfig",
    "MemoryConfig",
    "InfrastructureConfig",
    "VoiceConfig",
    "APIConfig",
    "ModelRuntimeConfig",
    "DeliveryConfig",
    "ResearchConfig",
    "JARVISConfig",
    "ConfigManager",
    "get_config",
    "reset_config",
]
