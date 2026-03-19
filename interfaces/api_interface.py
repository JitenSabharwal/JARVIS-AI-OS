"""
REST API interface for JARVIS AI OS.

Provides HTTP endpoints for querying JARVIS, managing agents/tasks/skills,
and checking system health. Uses aiohttp when available, falls back to
http.server for basic functionality.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

try:
    import aiohttp
    from aiohttp import web
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]

    class _WebStub:
        class _StubResponse(dict):
            def __init__(self, payload: Any | None = None, *, status: int = 200) -> None:
                super().__init__(payload or {})
                self.status = status
                self.headers: dict[str, str] = {}

        @staticmethod
        def middleware(func: Any) -> Any:
            return func

        def json_response(self, data: Any, status: int = 200) -> "_WebStub._StubResponse":
            return self._StubResponse(data, status=status)

        class HTTPException(Exception):
            pass

        class Request:  # pragma: no cover - typing fallback only
            pass

        class Response(_StubResponse):  # pragma: no cover - typing fallback only
            def __init__(self, status: int = 200) -> None:
                super().__init__({}, status=status)

        class Application:  # pragma: no cover - typing fallback only
            pass

    web = _WebStub()  # type: ignore[assignment]

from infrastructure.logger import get_logger
from infrastructure.connectors import ConnectorRegistry
from infrastructure.automation import AutomationEngine
from infrastructure.approval import ApprovalManager
from infrastructure.audit import AuditEvent, AuditLogger
from infrastructure.automation_actions import register_research_automation_actions
from infrastructure.proactive_engine import ProactiveEventEngine
from infrastructure.research_intelligence import ResearchIntelligenceEngine
from infrastructure.software_delivery import SoftwareDeliveryEngine
from infrastructure.slo_metrics import SLOMetrics, evaluate_slo_snapshot, get_slo_metrics

logger = get_logger("api_interface")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

@dataclass
class QueryRequest:
    query: str
    session_id: str | None = None
    user_id: str = "api_user"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRequest:
    description: str
    required_capabilities: list[str] = field(default_factory=list)
    priority: int = 5
    payload: dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0


@dataclass
class SkillExecuteRequest:
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class APIResponse:
    success: bool
    data: Any = None
    error: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Rate limiter (token-bucket, in-memory)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple per-IP token-bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        self._rpm = requests_per_minute
        self._buckets: dict[str, dict[str, Any]] = {}

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        bucket = self._buckets.get(client_id)
        if bucket is None:
            self._buckets[client_id] = {"tokens": self._rpm - 1, "last": now}
            return True
        elapsed = now - bucket["last"]
        refill = elapsed * (self._rpm / 60.0)
        bucket["tokens"] = min(self._rpm, bucket["tokens"] + refill)
        bucket["last"] = now
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False

    def cleanup(self) -> None:
        """Remove stale buckets (older than 5 minutes)."""
        cutoff = time.time() - 300
        self._buckets = {k: v for k, v in self._buckets.items() if v["last"] > cutoff}


# ---------------------------------------------------------------------------
# APIInterface
# ---------------------------------------------------------------------------

class APIInterface:
    """
    REST API interface for JARVIS AI OS.

    Injects optional service references (conversation_manager, orchestrator,
    skills_registry, monitor) after construction via ``set_*`` methods or
    directly setting attributes.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        auth_token: str | None = None,
        requests_per_minute: int = 60,
        cors_origins: list[str] | None = None,
        slo_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._auth_token = auth_token
        self._rate_limiter = RateLimiter(requests_per_minute)
        self._cors_origins = cors_origins or ["*"]
        self._slo_thresholds = slo_thresholds or {}

        # Injected service references (optional)
        self.conversation_manager: Any = None
        self.orchestrator: Any = None
        self.skills_registry: Any = None
        self.monitor: Any = None
        self.audit_logger: AuditLogger = AuditLogger()
        self.approval_manager: ApprovalManager = ApprovalManager.get_instance()
        self.connector_registry: ConnectorRegistry = ConnectorRegistry()
        self.automation_engine: AutomationEngine = AutomationEngine()
        self.research_engine: ResearchIntelligenceEngine = ResearchIntelligenceEngine()
        self.software_delivery_engine: SoftwareDeliveryEngine = SoftwareDeliveryEngine()
        self.proactive_engine: ProactiveEventEngine = ProactiveEventEngine()
        register_research_automation_actions(self.automation_engine, self.research_engine)
        self.slo_metrics: SLOMetrics = get_slo_metrics()

        # In-memory task status store
        self._tasks: dict[str, dict[str, Any]] = {}

        self._app: Any = None   # aiohttp.web.Application
        self._runner: Any = None
        self._site: Any = None
        self._running = False

        logger.info("APIInterface configured (%s:%d aiohttp=%s)", host, port, _AIOHTTP_AVAILABLE)

    # ------------------------------------------------------------------
    # Service injection helpers
    # ------------------------------------------------------------------

    def set_conversation_manager(self, cm: Any) -> None:
        self.conversation_manager = cm

    def set_orchestrator(self, orch: Any) -> None:
        self.orchestrator = orch

    def set_skills_registry(self, registry: Any) -> None:
        self.skills_registry = registry

    def set_monitor(self, monitor: Any) -> None:
        self.monitor = monitor

    def set_audit_logger(self, audit_logger: AuditLogger) -> None:
        self.audit_logger = audit_logger

    def set_approval_manager(self, approval_manager: ApprovalManager) -> None:
        self.approval_manager = approval_manager

    def set_connector_registry(self, connector_registry: ConnectorRegistry) -> None:
        self.connector_registry = connector_registry

    def set_automation_engine(self, automation_engine: AutomationEngine) -> None:
        self.automation_engine = automation_engine
        register_research_automation_actions(self.automation_engine, self.research_engine)

    def set_slo_metrics(self, slo_metrics: SLOMetrics) -> None:
        self.slo_metrics = slo_metrics

    def set_research_engine(self, research_engine: ResearchIntelligenceEngine) -> None:
        self.research_engine = research_engine
        register_research_automation_actions(self.automation_engine, self.research_engine)

    def set_software_delivery_engine(self, software_delivery_engine: SoftwareDeliveryEngine) -> None:
        self.software_delivery_engine = software_delivery_engine

    def set_proactive_engine(self, proactive_engine: ProactiveEventEngine) -> None:
        self.proactive_engine = proactive_engine

    def set_slo_thresholds(self, slo_thresholds: dict[str, float]) -> None:
        self._slo_thresholds = dict(slo_thresholds)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the HTTP server."""
        if not _AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available — API interface will not start. Install aiohttp.")
            return
        self._app = self._build_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        self._running = True
        logger.info("API server listening on http://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Gracefully shut down the HTTP server."""
        self._running = False
        if self._runner:
            await self._runner.cleanup()
        logger.info("API server stopped")

    # ------------------------------------------------------------------
    # aiohttp app construction
    # ------------------------------------------------------------------

    def _build_app(self) -> "web.Application":
        app = web.Application(middlewares=[
            self._request_context_middleware,
            self._auth_middleware,
            self._rate_limit_middleware,
            self._cors_middleware,
            self._error_middleware,
        ])
        app.router.add_post("/api/v1/query", self._handle_query)
        # OpenAI-compatible compatibility endpoints for IDE clients (e.g. Continue).
        app.router.add_get("/v1/models", self._handle_openai_models)
        app.router.add_post("/v1/chat/completions", self._handle_openai_chat_completions)
        app.router.add_get("/api/v1/agents", self._handle_list_agents)
        app.router.add_post("/api/v1/tasks", self._handle_submit_task)
        app.router.add_get("/api/v1/tasks/{task_id}", self._handle_get_task)
        app.router.add_post("/api/v1/tasks/{task_id}/retry", self._handle_retry_task)
        app.router.add_post("/api/v1/tasks/{task_id}/replan", self._handle_replan_task)
        app.router.add_post("/api/v1/plans", self._handle_submit_plan)
        app.router.add_get("/api/v1/plans/{plan_id}", self._handle_get_plan)
        app.router.add_post("/api/v1/research/ingest", self._handle_research_ingest)
        app.router.add_post("/api/v1/research/query", self._handle_research_query)
        app.router.add_get("/api/v1/research/tree/{source_id}", self._handle_research_tree)
        app.router.add_get("/api/v1/research/graph/health", self._handle_research_graph_health)
        app.router.add_get("/api/v1/research/adapters", self._handle_research_adapters_list)
        app.router.add_post("/api/v1/research/adapters/run", self._handle_research_adapters_run)
        app.router.add_get("/api/v1/research/watchlists", self._handle_research_watchlists)
        app.router.add_post("/api/v1/research/watchlists", self._handle_research_watchlist_create)
        app.router.add_post("/api/v1/research/watchlists/{watchlist_id}/digest", self._handle_research_digest)
        app.router.add_post("/api/v1/research/digests/run-due", self._handle_research_run_due_digests)
        app.router.add_get("/api/v1/research/quarantine", self._handle_research_quarantine_list)
        app.router.add_post("/api/v1/research/quarantine/{source_id}/review", self._handle_research_quarantine_review)
        app.router.add_get("/api/v1/delivery/templates", self._handle_delivery_templates)
        app.router.add_post("/api/v1/delivery/bootstrap", self._handle_delivery_bootstrap)
        app.router.add_post("/api/v1/delivery/pipelines/run", self._handle_delivery_pipeline_run)
        app.router.add_post("/api/v1/delivery/releases", self._handle_delivery_release_create)
        app.router.add_post("/api/v1/delivery/releases/{release_id}/post-deploy", self._handle_delivery_post_deploy)
        app.router.add_get("/api/v1/delivery/releases/{release_id}", self._handle_delivery_release_get)
        app.router.add_get("/api/v1/delivery/metrics/lead-time", self._handle_delivery_lead_time_metrics)
        app.router.add_get("/api/v1/delivery/capabilities", self._handle_delivery_capabilities)
        app.router.add_post("/api/v1/delivery/releases/run", self._handle_delivery_release_run)
        app.router.add_get("/api/v1/skills", self._handle_list_skills)
        app.router.add_post("/api/v1/skills/{skill_name}/execute", self._handle_execute_skill)
        app.router.add_get("/api/v1/health", self._handle_health)
        app.router.add_get("/api/v1/status", self._handle_status)
        app.router.add_get("/api/v1/metrics", self._handle_metrics)
        app.router.add_get("/api/v1/audit", self._handle_audit)
        app.router.add_get("/api/v1/connectors", self._handle_connectors_list)
        app.router.add_get("/api/v1/connectors/health", self._handle_connectors_health)
        app.router.add_get("/api/v1/connectors/{connector_name}/health", self._handle_connector_health)
        app.router.add_post("/api/v1/connectors/{connector_name}/invoke", self._handle_connector_invoke)
        app.router.add_post("/api/v1/email/{operation}", self._handle_email_operation)
        app.router.add_post("/api/v1/files/intel/{operation}", self._handle_file_intel_operation)
        app.router.add_post("/api/v1/images/intel/{operation}", self._handle_image_intel_operation)
        app.router.add_get("/api/v1/automation/rules", self._handle_automation_rules_list)
        app.router.add_post("/api/v1/automation/rules", self._handle_automation_rule_create)
        app.router.add_post("/api/v1/automation/events", self._handle_automation_event)
        app.router.add_get("/api/v1/automation/history", self._handle_automation_history)
        app.router.add_get("/api/v1/automation/dead-letters", self._handle_automation_dead_letters)
        app.router.add_post("/api/v1/automation/dead-letters/replay", self._handle_automation_dead_letter_replay)
        app.router.add_post("/api/v1/automation/dead-letters/{dead_letter_id}/resolve", self._handle_automation_dead_letter_resolve)
        app.router.add_post("/api/v1/proactive/events", self._handle_proactive_event)
        app.router.add_post("/api/v1/proactive/preferences", self._handle_proactive_preferences)
        app.router.add_get("/api/v1/proactive/suggestions", self._handle_proactive_suggestions)
        app.router.add_get("/api/v1/proactive/profile/{user_id}", self._handle_proactive_profile)
        app.router.add_post("/api/v1/proactive/suggestions/{suggestion_id}/ack", self._handle_proactive_suggestion_ack)
        app.router.add_post("/api/v1/proactive/suggestions/{suggestion_id}/dismiss", self._handle_proactive_suggestion_dismiss)
        app.router.add_post("/api/v1/proactive/suggestions/{suggestion_id}/snooze", self._handle_proactive_suggestion_snooze)
        app.router.add_post("/api/v1/proactive/actions/execute", self._handle_proactive_execute_action)
        app.router.add_post("/api/v1/approvals/request", self._handle_approval_request)
        app.router.add_post("/api/v1/approvals/{approval_id}/approve", self._handle_approval_approve)
        app.router.add_post("/api/v1/approvals/{approval_id}/reject", self._handle_approval_reject)
        app.router.add_get("/api/v1/approvals/{approval_id}", self._handle_approval_get)
        # OPTIONS for CORS preflight
        app.router.add_route("OPTIONS", "/{path_info:.*}", self._handle_options)
        return app

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    @web.middleware
    async def _request_context_middleware(self, request: "web.Request", handler: Any) -> "web.Response":
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request["request_id"] = request_id
        started_at = time.time()
        try:
            response = await handler(request)
        finally:
            elapsed_ms = round((time.time() - started_at) * 1000, 2)
            if self.slo_metrics:
                path = request.path
                method = request.method
                route_label = f"{method} {path}"
                self.slo_metrics.inc("api_requests_total", label=route_label)
                self.slo_metrics.observe_latency(
                    "api_request_latency_ms", elapsed_ms, label=route_label
                )
            logger.info(
                "request_id=%s method=%s path=%s duration_ms=%s",
                request_id,
                request.method,
                request.path,
                elapsed_ms,
            )
        if self.slo_metrics:
            status_label = f"{request.method} {request.path} {response.status}"
            self.slo_metrics.inc("api_responses_total", label=status_label)
            if response.status >= 500:
                self.slo_metrics.inc("api_errors_total", label="5xx")
            elif response.status >= 400:
                self.slo_metrics.inc("api_errors_total", label="4xx")
        response.headers["X-Request-ID"] = request_id
        return response

    @web.middleware
    async def _auth_middleware(self, request: "web.Request", handler: Any) -> "web.Response":
        # Health endpoint is always public
        if request.path in ("/api/v1/health",) or request.method == "OPTIONS":
            return await handler(request)
        if self._auth_token:
            auth_header = request.headers.get("Authorization", "")
            token = auth_header.removeprefix("Bearer ").strip()
            if not self._constant_time_compare(token, self._auth_token):
                self._record_audit(
                    request,
                    event_type="auth",
                    action="api_auth",
                    success=False,
                    decision="deny",
                    reason="Unauthorized token",
                )
                return self._error_response(request, "Unauthorized", status=401)
        return await handler(request)

    @web.middleware
    async def _rate_limit_middleware(self, request: "web.Request", handler: Any) -> "web.Response":
        client_ip = request.remote or "unknown"
        if not self._rate_limiter.is_allowed(client_ip):
            return self._error_response(request, "Rate limit exceeded", status=429)
        return await handler(request)

    @web.middleware
    async def _cors_middleware(self, request: "web.Request", handler: Any) -> "web.Response":
        response = await handler(request)
        origin = request.headers.get("Origin", "*")
        allowed = "*" in self._cors_origins or origin in self._cors_origins
        if allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type,Authorization,X-Scopes,X-User-ID,X-Approver-ID,X-Request-ID"
            )
        return response

    @web.middleware
    async def _error_middleware(self, request: "web.Request", handler: Any) -> "web.Response":
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Unhandled API error: %s", exc)
            self._record_audit(
                request,
                event_type="error",
                action="request_exception",
                success=False,
                decision="deny",
                reason=str(exc),
                metadata={"path": request.path, "method": request.method},
            )
            return self._error_response(
                request,
                f"Internal server error: {exc}",
                status=500,
            )

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    async def _handle_options(self, _request: "web.Request") -> "web.Response":
        return web.Response(status=204)

    async def _handle_query(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/query — submit a natural-language query to JARVIS."""
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")

        query = body.get("query", "").strip()
        if not query:
            return self._bad_request(request, "'query' field is required")

        session_id = body.get("session_id")
        user_id = body.get("user_id", "api_user")
        modality = str(body.get("modality", "text"))
        media = body.get("media", {})
        context = body.get("context", {})
        if not isinstance(media, dict):
            return self._bad_request(request, "'media' must be an object")
        if not isinstance(context, dict):
            return self._bad_request(request, "'context' must be an object")

        if self.conversation_manager:
            if not session_id:
                session_id = self.conversation_manager.get_or_create_session(user_id)
            response_text = await self.conversation_manager.process_input(
                session_id,
                query,
                modality=modality,
                media=media,
                context=context,
            )
            self._record_audit(
                request,
                event_type="query",
                action="conversation_query",
                success=True,
                metadata={"session_id": session_id, "user_id": user_id},
            )
            return self._ok_response(
                request,
                {"response": response_text, "session_id": session_id},
            )

        self._record_audit(
            request,
            event_type="query",
            action="conversation_query_echo",
            success=True,
            metadata={"user_id": user_id},
        )
        return self._ok_response(
            request,
            {"response": f"Echo: {query}", "session_id": session_id or "none"},
        )

    async def _handle_openai_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/models — lightweight OpenAI-compatible model list."""
        model_ids: list[str] = []
        try:
            from core.config import get_config

            cfg = get_config()
            model_ids = [
                str(cfg.model.mlx_text_model or "").strip(),
                str(cfg.model.mlx_text_model_coding or "").strip(),
                str(cfg.model.mlx_text_model_small or "").strip(),
                str(cfg.model.ollama_text_model or "").strip(),
                str(cfg.model.cohere_text_model or "").strip(),
            ]
        except Exception:  # noqa: BLE001
            model_ids = []
        deduped = [m for m in dict.fromkeys([m for m in model_ids if m])]
        if not deduped:
            deduped = ["jarvis-default"]
        payload = {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "jarvis",
                }
                for model_id in deduped
            ],
        }
        return web.json_response(payload, status=200)

    async def _handle_openai_chat_completions(self, request: "web.Request") -> "web.Response":
        """
        POST /v1/chat/completions — OpenAI-compatible chat endpoint.

        Supports a non-streaming response and a minimal SSE stream mode.
        """
        req_started = time.time()
        parse_started = time.time()
        body = await self._parse_json(request)
        parse_latency_ms = round((time.time() - parse_started) * 1000.0, 2)
        if body is None:
            return web.json_response({"error": {"message": "Invalid JSON body"}}, status=400)
        messages = body.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return web.json_response({"error": {"message": "'messages' must be a non-empty list"}}, status=400)

        stream = bool(body.get("stream", False))
        include_debug = bool(body.get("jarvis_debug", False)) or str(
            request.headers.get("X-JARVIS-Debug", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        model = str(body.get("model", "") or "jarvis-default")
        user_id = str(body.get("user", "") or request.headers.get("X-User-ID", "continue_user"))

        prompt_build_started = time.time()
        prompt_parts: list[str] = []
        last_user = ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user")).strip() or "user"
            content = msg.get("content", "")
            text = self._coerce_chat_message_content(content)
            if not text:
                continue
            prompt_parts.append(f"{role}: {text}")
            if role == "user":
                last_user = text
        if not last_user and prompt_parts:
            last_user = prompt_parts[-1]
        if not last_user:
            return web.json_response({"error": {"message": "No valid message content found"}}, status=400)
        prompt_build_latency_ms = round((time.time() - prompt_build_started) * 1000.0, 2)

        response_text = ""
        session_id = ""
        conversation_latency_ms = 0.0
        if self.conversation_manager:
            session_id = self.conversation_manager.get_or_create_session(user_id)
            conversation_started = time.time()
            chat_ctx: dict[str, Any] = {}
            if len(prompt_parts[:-1]) > 4:
                chat_ctx = {"chat_history": prompt_parts[:-1]}
            response_text = await self.conversation_manager.process_input(
                session_id,
                last_user,
                modality="text",
                media={},
                context=chat_ctx,
            )
            conversation_latency_ms = round((time.time() - conversation_started) * 1000.0, 2)
        else:
            response_text = f"Echo: {last_user}"

        ctx_latency: dict[str, Any] = {}
        route_latency: dict[str, Any] = {}
        summary_stage: dict[str, Any] = {}
        if self.conversation_manager and session_id and hasattr(self.conversation_manager, "get_context"):
            try:
                ctx = self.conversation_manager.get_context(session_id)
                if ctx is not None and isinstance(getattr(ctx, "metadata", None), dict):
                    md = ctx.metadata
                    if isinstance(md.get("latency_ms"), dict):
                        ctx_latency = dict(md.get("latency_ms", {}))
                    if isinstance(md.get("model_route"), dict):
                        route_latency = dict(md.get("model_route", {}))
                    if isinstance(md.get("summary_stage"), dict):
                        summary_stage = dict(md.get("summary_stage", {}))
            except Exception:  # noqa: BLE001
                pass
        response_build_latency_ms = round((time.time() - req_started) * 1000.0, 2)
        stage_latency_ms = {
            "parse": parse_latency_ms,
            "prompt_build": prompt_build_latency_ms,
            "conversation": conversation_latency_ms,
            "response_build": response_build_latency_ms,
            "total": round((time.time() - req_started) * 1000.0, 2),
        }
        logger.info(
            "chat_completion request_id=%s user=%s model=%s latencies_ms=%s",
            self._request_id(request),
            user_id,
            model,
            stage_latency_ms,
        )
        if self.slo_metrics:
            self.slo_metrics.observe_latency(
                "chat_completion_total_latency_ms",
                stage_latency_ms["total"],
                label=model,
            )
            self.slo_metrics.observe_latency(
                "chat_completion_conversation_latency_ms",
                stage_latency_ms["conversation"],
                label=model,
            )
            if summary_stage:
                source = str(summary_stage.get("source", "unknown"))
                self.slo_metrics.inc("summary_stage_total", label=source)
                reject_reason = str(summary_stage.get("reject_reason", "")).strip()
                if reject_reason:
                    self.slo_metrics.inc("summary_reject_total", label=reject_reason)

        if not stream:
            payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": str(response_text)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": max(1, len(last_user.split())),
                    "completion_tokens": max(1, len(str(response_text).split())),
                    "total_tokens": max(1, len(last_user.split())) + max(1, len(str(response_text).split())),
                },
            }
            if include_debug:
                payload["jarvis_debug"] = {
                    "session_id": session_id or "",
                    "stage_latency_ms": stage_latency_ms,
                    "conversation_latency_ms": ctx_latency,
                    "model_route": route_latency,
                    "summary_stage": summary_stage,
                }
            return web.json_response(payload, status=200)

        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": str(response_text)},
                    "finish_reason": "stop",
                }
            ],
        }
        if include_debug:
            chunk["jarvis_debug"] = {
                "session_id": session_id or "",
                "stage_latency_ms": stage_latency_ms,
                "conversation_latency_ms": ctx_latency,
                "model_route": route_latency,
                "summary_stage": summary_stage,
            }
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def _handle_list_agents(self, _request: "web.Request") -> "web.Response":
        """GET /api/v1/agents — list registered agents."""
        if self.orchestrator and hasattr(self.orchestrator, "get_system_status"):
            status = self.orchestrator.get_system_status()
            agents = status.get("agents", [])
        else:
            agents = []
        return self._ok_response(_request, {"agents": agents})

    async def _handle_submit_task(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/tasks — submit a task for execution."""
        op_started = time.time()
        body = await self._parse_json(request)
        if body is None:
            if self.slo_metrics:
                self.slo_metrics.inc("task_submit_total", label="bad_request")
            return self._bad_request(request, "Invalid JSON body")

        description = body.get("description", "").strip()
        if not description:
            if self.slo_metrics:
                self.slo_metrics.inc("task_submit_total", label="bad_request")
            return self._bad_request(request, "'description' field is required")

        required_capabilities = body.get("required_capabilities", [])
        if not isinstance(required_capabilities, list) or not required_capabilities:
            if self.slo_metrics:
                self.slo_metrics.inc("task_submit_total", label="bad_request")
            return self._bad_request(
                request,
                "'required_capabilities' must be a non-empty list",
            )

        task_id = str(uuid.uuid4())
        task_entry: dict[str, Any] = {
            "task_id": task_id,
            "description": description,
            "required_capabilities": required_capabilities,
            "payload": body.get("payload", {}),
            "status": "pending",
            "submitted_at": time.time(),
            "result": None,
            "error": None,
        }
        self._tasks[task_id] = task_entry

        if self.orchestrator and hasattr(self.orchestrator, "submit_task"):
            try:
                requires_human = bool(body.get("requires_human", False))
                approval_token = str(body.get("approval_token", "")).strip() or None
                metadata = body.get("metadata", {})
                if metadata is None:
                    metadata = {}
                if not isinstance(metadata, dict):
                    return self._bad_request(request, "'metadata' must be an object")
                orch_task_id = await self.orchestrator.submit_task(
                    description=description,
                    required_capabilities=required_capabilities,
                    priority=body.get("priority", 5),
                    payload=body.get("payload", {}),
                    requires_human=requires_human,
                    approval_token=approval_token,
                    metadata=metadata,
                )
                task_entry["orchestrator_task_id"] = orch_task_id
                task_entry["status"] = "submitted"
                task_entry["requires_human"] = requires_human
                if hasattr(self.orchestrator, "get_task_status"):
                    orch_task = self.orchestrator.get_task_status(orch_task_id)
                    if orch_task is not None:
                        task_entry["status"] = orch_task.status.value
                        task_entry["error"] = orch_task.error
                self._record_audit(
                    request,
                    event_type="task",
                    action="submit_task",
                    success=True,
                    metadata={
                        "task_id": task_id,
                        "orchestrator_task_id": orch_task_id,
                        "required_capabilities": required_capabilities,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                task_entry["status"] = "failed"
                task_entry["error"] = str(exc)
                if self.slo_metrics:
                    self.slo_metrics.inc("task_submit_total", label="orchestrator_failed")
                self._record_audit(
                    request,
                    event_type="task",
                    action="submit_task",
                    success=False,
                    decision="deny",
                    reason=str(exc),
                    metadata={"task_id": task_id},
                )
        if self.slo_metrics:
            self.slo_metrics.inc("task_submit_total", label="accepted")
            self.slo_metrics.observe_latency(
                "task_submit_latency_ms", (time.time() - op_started) * 1000, label="submit_task"
            )

        return self._ok_response(request, task_entry, status=202)

    async def _handle_get_task(self, request: "web.Request") -> "web.Response":
        """GET /api/v1/tasks/{task_id} — get task status."""
        task_id = request.match_info.get("task_id", "")
        task = self._tasks.get(task_id)
        if task is None:
            return self._error_response(
                request,
                f"Task not found: {task_id}",
                status=404,
            )
        # Optionally sync status from orchestrator
        if self.orchestrator and hasattr(self.orchestrator, "get_task_status"):
            try:
                orch_task_id = task.get("orchestrator_task_id")
                if orch_task_id:
                    orch_task = self.orchestrator.get_task_status(orch_task_id)
                    if orch_task is not None:
                        task["status"] = orch_task.status.value
                        task["result"] = orch_task.result
                        task["error"] = orch_task.error
            except Exception:  # noqa: BLE001
                pass
        return self._ok_response(request, task)

    async def _handle_retry_task(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/tasks/{task_id}/retry — retry failed/cancelled task."""
        task_id = request.match_info.get("task_id", "")
        task = self._tasks.get(task_id)
        if task is None:
            return self._error_response(request, f"Task not found: {task_id}", status=404)
        if not self.orchestrator or not hasattr(self.orchestrator, "retry_task"):
            return self._error_response(request, "Task retry unsupported by orchestrator", status=503)
        orch_task_id = str(task.get("orchestrator_task_id") or "")
        if not orch_task_id:
            return self._error_response(request, "Task has no orchestrator mapping", status=400)

        body = await self._parse_json(request) or {}
        approval_token = str(body.get("approval_token", "")).strip() or None
        try:
            retried = await self.orchestrator.retry_task(orch_task_id, approval_token=approval_token)
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=500)
        if not retried:
            return self._error_response(request, f"Task not retryable: {task_id}", status=409)
        self._record_audit(
            request,
            event_type="task",
            action="retry_task",
            success=True,
            metadata={"task_id": task_id, "orchestrator_task_id": orch_task_id},
        )
        task["status"] = "pending"
        task["error"] = None
        return self._ok_response(request, {"task_id": task_id, "retried": True})

    async def _handle_replan_task(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/tasks/{task_id}/replan — create replacement task with fallback capabilities."""
        task_id = request.match_info.get("task_id", "")
        task = self._tasks.get(task_id)
        if task is None:
            return self._error_response(request, f"Task not found: {task_id}", status=404)
        if not self.orchestrator or not hasattr(self.orchestrator, "replan_task"):
            return self._error_response(request, "Task replan unsupported by orchestrator", status=503)
        orch_task_id = str(task.get("orchestrator_task_id") or "")
        if not orch_task_id:
            return self._error_response(request, "Task has no orchestrator mapping", status=400)

        body = await self._parse_json(request) or {}
        fallback_capabilities = body.get("fallback_capabilities", [])
        if not isinstance(fallback_capabilities, list) or any(not isinstance(c, str) for c in fallback_capabilities):
            return self._bad_request(request, "'fallback_capabilities' must be list[str]")
        if not fallback_capabilities:
            fallback_capabilities = list(task.get("required_capabilities") or [])
        payload_override = body.get("payload_override", None)
        if payload_override is not None and not isinstance(payload_override, dict):
            return self._bad_request(request, "'payload_override' must be an object")
        description_suffix = str(body.get("description_suffix", "replan")).strip() or "replan"
        approval_token = str(body.get("approval_token", "")).strip() or None
        try:
            new_orch_task_id = await self.orchestrator.replan_task(
                orch_task_id,
                fallback_capabilities=fallback_capabilities,
                payload_override=payload_override,
                description_suffix=description_suffix,
                approval_token=approval_token,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=500)
        if not new_orch_task_id:
            return self._error_response(request, "Unable to replan task", status=409)

        new_task_id = str(uuid.uuid4())
        self._tasks[new_task_id] = {
            "task_id": new_task_id,
            "description": f"{task.get('description', '')} ({description_suffix})",
            "status": "submitted",
            "submitted_at": time.time(),
            "result": None,
            "error": None,
            "required_capabilities": fallback_capabilities,
            "payload": payload_override if payload_override is not None else task.get("payload", {}),
            "orchestrator_task_id": new_orch_task_id,
            "replanned_from_task_id": task_id,
        }
        self._record_audit(
            request,
            event_type="task",
            action="replan_task",
            success=True,
            metadata={
                "task_id": task_id,
                "new_task_id": new_task_id,
                "orchestrator_task_id": orch_task_id,
                "new_orchestrator_task_id": new_orch_task_id,
            },
        )
        return self._ok_response(
            request,
            {
                "task_id": new_task_id,
                "orchestrator_task_id": new_orch_task_id,
                "replanned_from_task_id": task_id,
            },
            status=202,
        )

    async def _handle_submit_plan(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/plans — submit task plan to orchestrator."""
        if not self.orchestrator or not hasattr(self.orchestrator, "submit_task_plan"):
            return self._error_response(request, "Plan submission unsupported by orchestrator", status=503)
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        description = str(body.get("description", "")).strip()
        steps = body.get("steps", [])
        try:
            priority = int(body.get("priority", 0))
        except (TypeError, ValueError):
            return self._bad_request(request, "'priority' must be an integer")
        metadata = body.get("metadata", {})
        if not description:
            return self._bad_request(request, "'description' is required")
        if not isinstance(steps, list) or not steps:
            return self._bad_request(request, "'steps' must be a non-empty list")
        if not isinstance(metadata, dict):
            return self._bad_request(request, "'metadata' must be an object")
        try:
            result = await self.orchestrator.submit_task_plan(
                description=description,
                steps=steps,
                priority=priority,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=400)
        self._record_audit(
            request,
            event_type="task",
            action="submit_plan",
            success=True,
            metadata={"plan_id": result.get("plan_id", ""), "step_count": result.get("step_count", 0)},
        )
        return self._ok_response(request, result, status=202)

    async def _handle_get_plan(self, request: "web.Request") -> "web.Response":
        """GET /api/v1/plans/{plan_id} — get plan status."""
        plan_id = request.match_info.get("plan_id", "")
        if not self.orchestrator or not hasattr(self.orchestrator, "get_plan_status"):
            return self._error_response(request, "Plan status unsupported by orchestrator", status=503)
        try:
            plan = self.orchestrator.get_plan_status(plan_id)
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=500)
        if plan is None:
            return self._error_response(request, f"Plan not found: {plan_id}", status=404)
        return self._ok_response(request, plan)

    async def _handle_research_ingest(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/research/ingest — ingest research/news/blog source items."""
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        items = body.get("items", [])
        if not isinstance(items, list):
            return self._bad_request(request, "'items' must be a list")
        result = self.research_engine.ingest_sources(items)
        self._record_audit(
            request,
            event_type="research",
            action="ingest_sources",
            success=True,
            metadata={"inserted": result.get("inserted", 0), "skipped_duplicates": result.get("skipped_duplicates", 0)},
        )
        return self._ok_response(request, result)

    async def _handle_research_query(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/research/query — ranked research query with citations."""
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        topic = str(body.get("topic", "")).strip()
        if not topic:
            return self._bad_request(request, "'topic' is required")
        max_results = body.get("max_results", 5)
        freshness_days = body.get("freshness_days", 30)
        min_trust = body.get("min_trust", 0.0)
        try:
            max_results = max(1, min(50, int(max_results)))
            freshness_days = max(1, min(365, int(freshness_days)))
            min_trust = max(0.0, min(1.0, float(min_trust)))
        except (TypeError, ValueError):
            return self._bad_request(request, "'max_results' and 'freshness_days' must be integers and 'min_trust' number")
        result = self.research_engine.query(
            topic,
            max_results=max_results,
            freshness_days=freshness_days,
            min_trust=min_trust,
        )
        self._record_audit(
            request,
            event_type="research",
            action="query_research",
            success=True,
            metadata={"topic": topic, "result_count": result.get("result_count", 0)},
        )
        return self._ok_response(request, result)

    async def _handle_research_tree(self, request: "web.Request") -> "web.Response":
        source_id = str(request.match_info.get("source_id", "")).strip()
        if not source_id:
            return self._bad_request(request, "'source_id' is required")
        tree = self.research_engine.get_source_tree(source_id)
        return self._ok_response(request, tree)

    async def _handle_research_graph_health(self, request: "web.Request") -> "web.Response":
        health = self.research_engine.graph_health()
        return self._ok_response(request, health)

    async def _handle_research_adapters_list(self, request: "web.Request") -> "web.Response":
        adapters = self.research_engine.list_adapters()
        return self._ok_response(request, {"adapters": adapters, "count": len(adapters)})

    async def _handle_research_adapters_run(self, request: "web.Request") -> "web.Response":
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        topic = str(body.get("topic", "")).strip()
        if not topic:
            return self._bad_request(request, "'topic' is required")
        max_items_per_adapter = body.get("max_items_per_adapter", 10)
        try:
            max_items_per_adapter = max(1, min(100, int(max_items_per_adapter)))
        except (TypeError, ValueError):
            return self._bad_request(request, "'max_items_per_adapter' must be an integer")
        result = self.research_engine.run_adapters(
            topic=topic,
            max_items_per_adapter=max_items_per_adapter,
        )
        self._record_audit(
            request,
            event_type="research",
            action="run_source_adapters",
            success=True,
            metadata={
                "topic": topic,
                "adapter_count": result.get("adapter_count", 0),
                "inserted_total": result.get("inserted_total", 0),
            },
        )
        return self._ok_response(request, result)

    async def _handle_research_watchlists(self, request: "web.Request") -> "web.Response":
        watchlists = self.research_engine.list_watchlists()
        return self._ok_response(request, {"watchlists": watchlists, "count": len(watchlists)})

    async def _handle_research_watchlist_create(self, request: "web.Request") -> "web.Response":
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        name = str(body.get("name", "")).strip()
        topics = body.get("topics", [])
        cadence = str(body.get("cadence", "daily")).strip() or "daily"
        metadata = body.get("metadata", {})
        if not isinstance(topics, list):
            return self._bad_request(request, "'topics' must be a list")
        if not isinstance(metadata, dict):
            return self._bad_request(request, "'metadata' must be an object")
        try:
            watch = self.research_engine.create_watchlist(
                name=name,
                topics=[str(t) for t in topics],
                cadence=cadence,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=400)
        self._record_audit(
            request,
            event_type="research",
            action="create_watchlist",
            success=True,
            metadata={"watchlist_id": watch.get("watchlist_id", ""), "topic_count": len(watch.get("topics", []))},
        )
        return self._ok_response(request, watch, status=201)

    async def _handle_research_digest(self, request: "web.Request") -> "web.Response":
        watchlist_id = request.match_info.get("watchlist_id", "")
        body = await self._parse_json(request) or {}
        max_per_topic = body.get("max_per_topic", 3)
        try:
            max_per_topic = max(1, min(20, int(max_per_topic)))
        except (TypeError, ValueError):
            return self._bad_request(request, "'max_per_topic' must be an integer")
        try:
            digest = self.research_engine.generate_digest(
                watchlist_id,
                max_per_topic=max_per_topic,
            )
        except KeyError:
            return self._error_response(request, f"Watchlist not found: {watchlist_id}", status=404)
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=500)
        self._record_audit(
            request,
            event_type="research",
            action="generate_digest",
            success=True,
            metadata={"watchlist_id": watchlist_id, "sections": len(digest.get("sections", []))},
        )
        return self._ok_response(request, digest)

    async def _handle_research_run_due_digests(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/research/digests/run-due — run cadence-based due watchlist digests."""
        body = await self._parse_json(request) or {}
        max_per_topic = body.get("max_per_topic", 3)
        try:
            max_per_topic = max(1, min(20, int(max_per_topic)))
        except (TypeError, ValueError):
            return self._bad_request(request, "'max_per_topic' must be an integer")
        result = self.research_engine.run_due_digests(max_per_topic=max_per_topic)
        self._record_audit(
            request,
            event_type="research",
            action="run_due_digests",
            success=True,
            metadata={
                "generated_count": result.get("generated_count", 0),
                "skipped_count": result.get("skipped_count", 0),
            },
        )
        return self._ok_response(request, result)

    async def _handle_research_quarantine_list(self, request: "web.Request") -> "web.Response":
        limit_raw = request.query.get("limit", "100")
        try:
            limit = max(1, min(1000, int(limit_raw)))
        except (TypeError, ValueError):
            return self._bad_request(request, "'limit' must be an integer")
        result = self.research_engine.list_quarantined_sources(limit=limit)
        return self._ok_response(request, result)

    async def _handle_research_quarantine_review(self, request: "web.Request") -> "web.Response":
        source_id = str(request.match_info.get("source_id", "")).strip()
        if not source_id:
            return self._bad_request(request, "'source_id' is required")
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        action = str(body.get("action", "")).strip().lower()
        reviewer = str(body.get("reviewer", "")).strip()
        reason = str(body.get("reason", "")).strip()
        try:
            result = self.research_engine.review_quarantined_source(
                source_id,
                action=action,
                reviewer=reviewer,
                reason=reason,
            )
        except KeyError:
            return self._error_response(request, f"Source not found: {source_id}", status=404)
        except ValueError as exc:
            return self._bad_request(request, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=500)
        self._record_audit(
            request,
            event_type="research",
            action="review_quarantined_source",
            success=True,
            metadata={"source_id": source_id, "review_action": action},
        )
        return self._ok_response(request, result)

    async def _handle_delivery_templates(self, request: "web.Request") -> "web.Response":
        templates = self.software_delivery_engine.list_templates()
        profiles = self.software_delivery_engine.list_profiles()
        return self._ok_response(
            request,
            {
                "templates": templates,
                "profiles": profiles,
                "template_count": len(templates),
                "profile_count": len(profiles),
            },
        )

    async def _handle_delivery_bootstrap(self, request: "web.Request") -> "web.Response":
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        template_id = str(body.get("template_id", "")).strip()
        project_name = str(body.get("project_name", "")).strip()
        cloud_target = str(body.get("cloud_target", "local")).strip() or "local"
        include_ci = bool(body.get("include_ci", True))
        if not template_id:
            return self._bad_request(request, "'template_id' is required")
        if not project_name:
            return self._bad_request(request, "'project_name' is required")
        try:
            result = self.software_delivery_engine.bootstrap_project(
                template_id=template_id,
                project_name=project_name,
                cloud_target=cloud_target,
                include_ci=include_ci,
            )
        except KeyError as exc:
            return self._error_response(request, str(exc), status=404)
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=400)
        self._record_audit(
            request,
            event_type="delivery",
            action="bootstrap_project",
            success=True,
            metadata={
                "project_name": result.get("project_name", ""),
                "template_id": result.get("template_id", ""),
                "file_count": result.get("file_count", 0),
            },
        )
        return self._ok_response(request, result, status=201)

    async def _handle_delivery_pipeline_run(self, request: "web.Request") -> "web.Response":
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        project_name = str(body.get("project_name", "")).strip()
        gate_inputs = body.get("gate_inputs", {})
        required_gates = body.get("required_gates", None)
        if not project_name:
            return self._bad_request(request, "'project_name' is required")
        if not isinstance(gate_inputs, dict):
            return self._bad_request(request, "'gate_inputs' must be an object")
        if required_gates is not None:
            if not isinstance(required_gates, list) or any(not isinstance(g, str) for g in required_gates):
                return self._bad_request(request, "'required_gates' must be list[str]")
        try:
            result = self.software_delivery_engine.run_pipeline(
                project_name=project_name,
                gate_inputs=gate_inputs,
                required_gates=required_gates,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=400)
        self._record_audit(
            request,
            event_type="delivery",
            action="run_pipeline",
            success=True,
            metadata={
                "project_name": project_name,
                "all_passed": result.get("all_passed", False),
                "failed_gates": result.get("failed_gates", []),
            },
        )
        return self._ok_response(request, result)

    async def _handle_delivery_release_create(self, request: "web.Request") -> "web.Response":
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        project_name = str(body.get("project_name", "")).strip()
        profile = str(body.get("profile", "dev")).strip() or "dev"
        pipeline_result = body.get("pipeline_result", {})
        post_deploy = body.get("post_deploy", None)
        approved = bool(body.get("approved", False))
        metadata = body.get("metadata", {})
        build_started_at = body.get("build_started_at", None)
        if not project_name:
            return self._bad_request(request, "'project_name' is required")
        if not isinstance(pipeline_result, dict):
            return self._bad_request(request, "'pipeline_result' must be an object")
        if post_deploy is not None and not isinstance(post_deploy, dict):
            return self._bad_request(request, "'post_deploy' must be an object")
        if not isinstance(metadata, dict):
            return self._bad_request(request, "'metadata' must be an object")
        if build_started_at is not None:
            try:
                build_started_at = float(build_started_at)
            except (TypeError, ValueError):
                return self._bad_request(request, "'build_started_at' must be numeric")
        try:
            result = self.software_delivery_engine.create_release(
                project_name=project_name,
                profile=profile,
                pipeline_result=pipeline_result,
                post_deploy=post_deploy,
                approved=approved,
                metadata=metadata,
                build_started_at=build_started_at,
            )
        except KeyError as exc:
            return self._error_response(request, str(exc), status=404)
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=400)
        self._record_audit(
            request,
            event_type="delivery",
            action="create_release",
            success=True,
            metadata={
                "project_name": result.get("project_name", ""),
                "release_id": result.get("release_id", ""),
                "status": result.get("status", ""),
                "profile": result.get("profile", ""),
            },
        )
        return self._ok_response(request, result, status=201)

    async def _handle_delivery_post_deploy(self, request: "web.Request") -> "web.Response":
        release_id = request.match_info.get("release_id", "")
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        post_deploy = body.get("post_deploy", {})
        if not isinstance(post_deploy, dict):
            return self._bad_request(request, "'post_deploy' must be an object")
        try:
            result = self.software_delivery_engine.evaluate_post_deploy(
                release_id=release_id,
                post_deploy=post_deploy,
            )
        except KeyError as exc:
            return self._error_response(request, str(exc), status=404)
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=400)
        self._record_audit(
            request,
            event_type="delivery",
            action="evaluate_post_deploy",
            success=True,
            metadata={
                "release_id": result.get("release_id", ""),
                "status": result.get("status", ""),
                "rollback_reason": result.get("rollback_reason"),
            },
        )
        return self._ok_response(request, result)

    async def _handle_delivery_release_get(self, request: "web.Request") -> "web.Response":
        release_id = request.match_info.get("release_id", "")
        result = self.software_delivery_engine.get_release(release_id)
        if result is None:
            return self._error_response(request, f"Release not found: {release_id}", status=404)
        return self._ok_response(request, result)

    async def _handle_delivery_lead_time_metrics(self, request: "web.Request") -> "web.Response":
        result = self.software_delivery_engine.get_lead_time_summary()
        return self._ok_response(request, result)

    async def _handle_delivery_capabilities(self, request: "web.Request") -> "web.Response":
        return self._ok_response(
            request,
            {
                "gate_runners": self.software_delivery_engine.list_gate_runners(),
                "deploy_adapters": self.software_delivery_engine.list_deploy_adapters(),
                "deploy_adapter_specs": self.software_delivery_engine.get_deploy_adapter_specs(),
                "runtime_config": self.software_delivery_engine.get_runtime_config(),
                "ci_gate_templates": self.software_delivery_engine.list_ci_gate_templates(),
            },
        )

    async def _handle_delivery_release_run(self, request: "web.Request") -> "web.Response":
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        project_name = str(body.get("project_name", "")).strip()
        profile = str(body.get("profile", "dev")).strip() or "dev"
        deploy_target = str(body.get("deploy_target", "local")).strip() or "local"
        approved = bool(body.get("approved", False))
        required_gates = body.get("required_gates", None)
        context = body.get("context", {})
        post_deploy = body.get("post_deploy", None)
        metadata = body.get("metadata", {})

        if not project_name:
            return self._bad_request(request, "'project_name' is required")
        if required_gates is not None and (
            not isinstance(required_gates, list) or any(not isinstance(g, str) for g in required_gates)
        ):
            return self._bad_request(request, "'required_gates' must be list[str]")
        if not isinstance(context, dict):
            return self._bad_request(request, "'context' must be an object")
        if post_deploy is not None and not isinstance(post_deploy, dict):
            return self._bad_request(request, "'post_deploy' must be an object")
        if not isinstance(metadata, dict):
            return self._bad_request(request, "'metadata' must be an object")

        try:
            result = self.software_delivery_engine.run_release_pipeline(
                project_name=project_name,
                profile=profile,
                deploy_target=deploy_target,
                approved=approved,
                required_gates=required_gates,
                context=context,
                post_deploy=post_deploy,
                metadata=metadata,
            )
        except KeyError as exc:
            return self._error_response(request, str(exc), status=404)
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=400)

        release_data = result.get("release", {})
        self._record_audit(
            request,
            event_type="delivery",
            action="run_release_pipeline",
            success=True,
            metadata={
                "project_name": project_name,
                "profile": profile,
                "deploy_target": deploy_target,
                "release_status": release_data.get("status", ""),
                "release_id": release_data.get("release_id", ""),
            },
        )
        return self._ok_response(request, result, status=201)

    async def _handle_list_skills(self, _request: "web.Request") -> "web.Response":
        """GET /api/v1/skills — list available skills."""
        if self.skills_registry and hasattr(self.skills_registry, "get_all_skills_info"):
            skills = self.skills_registry.get_all_skills_info()
        else:
            skills = []
        return self._ok_response(_request, {"skills": skills})

    async def _handle_execute_skill(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/skills/{skill_name}/execute — execute a skill."""
        op_started = time.time()
        skill_name = request.match_info.get("skill_name", "")
        body = await self._parse_json(request) or {}
        params = body.get("params", {})

        if not self.skills_registry:
            if self.slo_metrics:
                self.slo_metrics.inc("skill_execute_total", label=f"{skill_name}:unavailable")
            return self._error_response(
                request,
                "Skills registry not available",
                status=503,
            )

        try:
            result = await self.skills_registry.execute_skill(skill_name, params)
            if hasattr(result, "success"):
                if result.success:
                    if self.slo_metrics:
                        self.slo_metrics.inc("skill_execute_total", label=f"{skill_name}:success")
                        self.slo_metrics.observe_latency(
                            "skill_execute_latency_ms",
                            (time.time() - op_started) * 1000,
                            label=skill_name,
                        )
                    self._record_audit(
                        request,
                        event_type="skill",
                        action=f"execute_skill:{skill_name}",
                        success=True,
                    )
                    return self._ok_response(request, result.data)
                if self.slo_metrics:
                    self.slo_metrics.inc("skill_execute_total", label=f"{skill_name}:failed")
                self._record_audit(
                    request,
                    event_type="skill",
                    action=f"execute_skill:{skill_name}",
                    success=False,
                    decision="deny",
                    reason=result.error or "Skill execution failed",
                )
                return self._error_response(request, result.error or "Skill execution failed", status=400)
            if self.slo_metrics:
                self.slo_metrics.inc("skill_execute_total", label=f"{skill_name}:success")
                self.slo_metrics.observe_latency(
                    "skill_execute_latency_ms",
                    (time.time() - op_started) * 1000,
                    label=skill_name,
                )
            self._record_audit(
                request,
                event_type="skill",
                action=f"execute_skill:{skill_name}",
                success=True,
            )
            return self._ok_response(request, result)
        except KeyError:
            if self.slo_metrics:
                self.slo_metrics.inc("skill_execute_total", label=f"{skill_name}:not_found")
            self._record_audit(
                request,
                event_type="skill",
                action=f"execute_skill:{skill_name}",
                success=False,
                decision="deny",
                reason="Skill not found",
            )
            return self._error_response(
                request,
                f"Skill not found: {skill_name}",
                status=404,
            )
        except Exception as exc:  # noqa: BLE001
            if self.slo_metrics:
                self.slo_metrics.inc("skill_execute_total", label=f"{skill_name}:error")
            self._record_audit(
                request,
                event_type="skill",
                action=f"execute_skill:{skill_name}",
                success=False,
                decision="deny",
                reason=str(exc),
            )
            return self._error_response(request, str(exc), status=500)

    async def _handle_health(self, _request: "web.Request") -> "web.Response":
        """GET /api/v1/health — lightweight health check."""
        if self.monitor:
            report = await self.monitor.get_health_report()
            return self._ok_response(
                _request,
                {
                    "status": report.overall_status.value,
                    "summary": report.summary,
                },
            )
        return self._ok_response(
            _request,
            {"status": "healthy"},
        )

    async def _handle_status(self, _request: "web.Request") -> "web.Response":
        """GET /api/v1/status — full system status."""
        data: dict[str, Any] = {
            "api": {"running": self._running, "host": self.host, "port": self.port},
            "active_tasks": len([t for t in self._tasks.values() if t["status"] in ("pending", "submitted", "running")]),
        }
        if self.monitor:
            report = await self.monitor.get_health_report()
            data["health"] = report.to_dict()
        if self.orchestrator and hasattr(self.orchestrator, "get_system_status"):
            data["orchestrator"] = self.orchestrator.get_system_status()
        return self._ok_response(_request, data)

    async def _handle_metrics(self, request: "web.Request") -> "web.Response":
        metrics = self.slo_metrics.snapshot() if self.slo_metrics else {}
        slo = evaluate_slo_snapshot(metrics, thresholds=self._slo_thresholds)
        return self._ok_response(request, {"metrics": metrics, "slo": slo})

    async def _handle_audit(self, request: "web.Request") -> "web.Response":
        """GET /api/v1/audit — recent audit events."""
        limit_raw = request.query.get("limit", "100")
        try:
            limit = max(1, min(1000, int(limit_raw)))
        except ValueError:
            return self._bad_request(request, "'limit' must be an integer")

        events = self.audit_logger.recent(limit=limit) if self.audit_logger else []
        return self._ok_response(request, {"events": events, "count": len(events)})

    async def _handle_connectors_list(self, request: "web.Request") -> "web.Response":
        """GET /api/v1/connectors — list registered connectors."""
        connectors = self.connector_registry.list_info() if self.connector_registry else []
        return self._ok_response(request, {"connectors": connectors, "count": len(connectors)})

    async def _handle_connectors_health(self, request: "web.Request") -> "web.Response":
        """GET /api/v1/connectors/health — health report for all connectors."""
        if not self.connector_registry:
            return self._ok_response(request, {"connectors": {}, "count": 0})
        report = await self.connector_registry.health_all()
        unhealthy = [name for name, health in report.items() if not bool(health.get("healthy", False))]
        if self.slo_metrics:
            self.slo_metrics.set_gauge("connectors_unhealthy_count", float(len(unhealthy)))
            self.slo_metrics.set_gauge("connectors_total_count", float(len(report)))
        return self._ok_response(
            request,
            {
                "connectors": report,
                "count": len(report),
                "unhealthy": unhealthy,
            },
        )

    async def _handle_connector_health(self, request: "web.Request") -> "web.Response":
        """GET /api/v1/connectors/{connector_name}/health — health report for one connector."""
        connector_name = request.match_info.get("connector_name", "")
        try:
            health = await self.connector_registry.health(connector_name)
        except KeyError:
            return self._error_response(request, f"Connector not found: {connector_name}", status=404)
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=500)
        return self._ok_response(request, {"connector": connector_name, "health": health})

    async def _handle_connector_invoke(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/connectors/{connector_name}/invoke — invoke connector operation."""
        op_started = time.time()
        connector_name = request.match_info.get("connector_name", "")
        body = await self._parse_json(request)
        if body is None:
            if self.slo_metrics:
                self.slo_metrics.inc("connector_invoke_total", label=f"{connector_name}:bad_request")
            return self._bad_request(request, "Invalid JSON body")

        operation = str(body.get("operation", "")).strip()
        params = body.get("params", {})
        if not operation:
            if self.slo_metrics:
                self.slo_metrics.inc("connector_invoke_total", label=f"{connector_name}:bad_request")
            return self._bad_request(request, "'operation' is required")
        if not isinstance(params, dict):
            if self.slo_metrics:
                self.slo_metrics.inc("connector_invoke_total", label=f"{connector_name}:bad_request")
            return self._bad_request(request, "'params' must be an object")
        body_scopes = body.get("actor_scopes", [])
        actor_scopes = self._extract_actor_scopes(request, body_scopes)
        if actor_scopes is None:
            if self.slo_metrics:
                self.slo_metrics.inc("connector_invoke_total", label=f"{connector_name}:bad_request")
            return self._bad_request(request, "'actor_scopes' must be a list of strings")
        return await self._invoke_connector_operation(
            request=request,
            connector_name=connector_name,
            operation=operation,
            params=params,
            actor_scopes=actor_scopes,
            op_started=op_started,
        )

    async def _handle_email_operation(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/email/{operation} — typed email ops endpoint."""
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        params = body.get("params", body)
        if not isinstance(params, dict):
            return self._bad_request(request, "'params' must be an object")
        actor_scopes = self._extract_actor_scopes(request, body.get("actor_scopes", []))
        if actor_scopes is None:
            return self._bad_request(request, "'actor_scopes' must be a list of strings")
        return await self._invoke_connector_operation(
            request=request,
            connector_name="email_ops",
            operation=request.match_info.get("operation", ""),
            params=params,
            actor_scopes=actor_scopes,
            op_started=time.time(),
        )

    async def _handle_file_intel_operation(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/files/intel/{operation} — typed file intel endpoint."""
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        params = body.get("params", body)
        if not isinstance(params, dict):
            return self._bad_request(request, "'params' must be an object")
        actor_scopes = self._extract_actor_scopes(request, body.get("actor_scopes", []))
        if actor_scopes is None:
            return self._bad_request(request, "'actor_scopes' must be a list of strings")
        return await self._invoke_connector_operation(
            request=request,
            connector_name="file_intel",
            operation=request.match_info.get("operation", ""),
            params=params,
            actor_scopes=actor_scopes,
            op_started=time.time(),
        )

    async def _handle_image_intel_operation(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/images/intel/{operation} — typed image intel endpoint."""
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        params = body.get("params", body)
        if not isinstance(params, dict):
            return self._bad_request(request, "'params' must be an object")
        actor_scopes = self._extract_actor_scopes(request, body.get("actor_scopes", []))
        if actor_scopes is None:
            return self._bad_request(request, "'actor_scopes' must be a list of strings")
        return await self._invoke_connector_operation(
            request=request,
            connector_name="image_intel",
            operation=request.match_info.get("operation", ""),
            params=params,
            actor_scopes=actor_scopes,
            op_started=time.time(),
        )

    @staticmethod
    def _extract_actor_scopes(request: "web.Request", raw_body_scopes: Any) -> set[str] | None:
        body_scopes = raw_body_scopes
        if body_scopes is None:
            body_scopes = []
        if not isinstance(body_scopes, list) or any(not isinstance(s, str) for s in body_scopes):
            return None
        header_scopes = request.headers.get("X-Scopes", "")
        parsed_header_scopes = [s.strip() for s in header_scopes.split(",") if s.strip()]
        return set(body_scopes) | set(parsed_header_scopes)

    async def _invoke_connector_operation(
        self,
        *,
        request: "web.Request",
        connector_name: str,
        operation: str,
        params: dict[str, Any],
        actor_scopes: set[str],
        op_started: float,
    ) -> "web.Response":
        if not operation:
            if self.slo_metrics:
                self.slo_metrics.inc("connector_invoke_total", label=f"{connector_name}:bad_request")
            return self._bad_request(request, "'operation' is required")
        try:
            result = await self.connector_registry.invoke(
                connector_name,
                operation,
                params,
                actor_scopes=actor_scopes,
            )
        except KeyError:
            if self.slo_metrics:
                self.slo_metrics.inc("connector_invoke_total", label=f"{connector_name}:not_found")
            return self._error_response(
                request,
                f"Connector not found: {connector_name}",
                status=404,
            )
        except PermissionError as exc:
            if self.slo_metrics:
                self.slo_metrics.inc("connector_invoke_total", label=f"{connector_name}:forbidden")
            self._record_audit(
                request,
                event_type="connector",
                action=f"invoke_connector:{connector_name}",
                success=False,
                decision="deny",
                reason=str(exc),
                metadata={"operation": operation, "actor_scopes": sorted(actor_scopes)},
            )
            return self._error_response(request, str(exc), status=403)
        except RuntimeError as exc:
            if self.slo_metrics:
                self.slo_metrics.inc("connector_invoke_total", label=f"{connector_name}:circuit_open")
            self._record_audit(
                request,
                event_type="connector",
                action=f"invoke_connector:{connector_name}",
                success=False,
                decision="deny",
                reason=str(exc),
                metadata={"operation": operation},
            )
            return self._error_response(request, str(exc), status=503)
        except Exception as exc:  # noqa: BLE001
            if self.slo_metrics:
                self.slo_metrics.inc("connector_invoke_total", label=f"{connector_name}:error")
            return self._error_response(request, str(exc), status=500)

        if self.slo_metrics:
            self.slo_metrics.inc("connector_invoke_total", label=f"{connector_name}:success")
            self.slo_metrics.observe_latency(
                "connector_invoke_latency_ms",
                (time.time() - op_started) * 1000,
                label=connector_name,
            )
        self._record_audit(
            request,
            event_type="connector",
            action=f"invoke_connector:{connector_name}",
            success=True,
            metadata={"operation": operation, "actor_scopes": sorted(actor_scopes)},
        )
        return self._ok_response(request, {"connector": connector_name, "operation": operation, "result": result})

    async def _handle_automation_rules_list(self, request: "web.Request") -> "web.Response":
        rules = self.automation_engine.list_rules() if self.automation_engine else []
        return self._ok_response(request, {"rules": rules, "count": len(rules)})

    async def _handle_automation_rule_create(self, request: "web.Request") -> "web.Response":
        op_started = time.time()
        body = await self._parse_json(request)
        if body is None:
            if self.slo_metrics:
                self.slo_metrics.inc("automation_rule_create_total", label="bad_request")
            return self._bad_request(request, "Invalid JSON body")

        name = str(body.get("name", "")).strip()
        event_type = str(body.get("event_type", "")).strip()
        action_name = str(body.get("action_name", "")).strip()
        match = body.get("match", {})
        enabled = bool(body.get("enabled", True))
        max_retries = body.get("max_retries", 0)
        retry_backoff_seconds = body.get("retry_backoff_seconds", 0.0)

        if not name:
            if self.slo_metrics:
                self.slo_metrics.inc("automation_rule_create_total", label="bad_request")
            return self._bad_request(request, "'name' is required")
        if not event_type:
            if self.slo_metrics:
                self.slo_metrics.inc("automation_rule_create_total", label="bad_request")
            return self._bad_request(request, "'event_type' is required")
        if not action_name:
            if self.slo_metrics:
                self.slo_metrics.inc("automation_rule_create_total", label="bad_request")
            return self._bad_request(request, "'action_name' is required")
        if not isinstance(match, dict):
            if self.slo_metrics:
                self.slo_metrics.inc("automation_rule_create_total", label="bad_request")
            return self._bad_request(request, "'match' must be an object")
        try:
            max_retries = int(max_retries)
            retry_backoff_seconds = float(retry_backoff_seconds)
        except (TypeError, ValueError):
            if self.slo_metrics:
                self.slo_metrics.inc("automation_rule_create_total", label="bad_request")
            return self._bad_request(
                request,
                "'max_retries' must be int and 'retry_backoff_seconds' must be number",
            )
        if max_retries < 0:
            if self.slo_metrics:
                self.slo_metrics.inc("automation_rule_create_total", label="bad_request")
            return self._bad_request(request, "'max_retries' must be >= 0")
        if retry_backoff_seconds < 0:
            if self.slo_metrics:
                self.slo_metrics.inc("automation_rule_create_total", label="bad_request")
            return self._bad_request(request, "'retry_backoff_seconds' must be >= 0")

        try:
            rule = self.automation_engine.create_rule(
                name=name,
                event_type=event_type,
                action_name=action_name,
                match=match,
                enabled=enabled,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            if self.slo_metrics:
                self.slo_metrics.inc("automation_rule_create_total", label="failed")
            return self._error_response(request, str(exc), status=400)

        if self.slo_metrics:
            self.slo_metrics.inc("automation_rule_create_total", label="success")
            self.slo_metrics.observe_latency(
                "automation_rule_create_latency_ms",
                (time.time() - op_started) * 1000,
                label=event_type,
            )
        self._record_audit(
            request,
            event_type="automation",
            action="create_rule",
            success=True,
            metadata={
                "rule_id": rule.rule_id,
                "event_type": event_type,
                "action_name": action_name,
                "max_retries": max_retries,
                "retry_backoff_seconds": retry_backoff_seconds,
            },
        )
        return self._ok_response(request, rule.to_dict(), status=201)

    async def _handle_automation_event(self, request: "web.Request") -> "web.Response":
        op_started = time.time()
        body = await self._parse_json(request)
        if body is None:
            if self.slo_metrics:
                self.slo_metrics.inc("automation_event_total", label="bad_request")
            return self._bad_request(request, "Invalid JSON body")

        event_type = str(body.get("event_type", "")).strip()
        payload = body.get("payload", {})
        if not event_type:
            if self.slo_metrics:
                self.slo_metrics.inc("automation_event_total", label="bad_request")
            return self._bad_request(request, "'event_type' is required")
        if not isinstance(payload, dict):
            if self.slo_metrics:
                self.slo_metrics.inc("automation_event_total", label="bad_request")
            return self._bad_request(request, "'payload' must be an object")

        timeout_seconds = float(body.get("timeout_seconds", 10.0))
        result = await self.automation_engine.process_event(
            event_type,
            payload,
            timeout_seconds=timeout_seconds,
        )
        if self.slo_metrics:
            self.slo_metrics.inc("automation_event_total", label="success")
            self.slo_metrics.observe_latency(
                "automation_event_latency_ms",
                (time.time() - op_started) * 1000,
                label=event_type,
            )
        self._record_audit(
            request,
            event_type="automation",
            action="process_event",
            success=True,
            metadata={"event_type": event_type, "matched_rules": result.get("matched_rules", 0)},
        )
        return self._ok_response(request, result)

    async def _handle_automation_history(self, request: "web.Request") -> "web.Response":
        limit_raw = request.query.get("limit", "100")
        try:
            limit = max(1, min(1000, int(limit_raw)))
        except ValueError:
            return self._bad_request(request, "'limit' must be an integer")
        history = self.automation_engine.get_history(limit=limit)
        return self._ok_response(request, {"history": history, "count": len(history)})

    async def _handle_automation_dead_letters(self, request: "web.Request") -> "web.Response":
        limit_raw = request.query.get("limit", "100")
        try:
            limit = max(1, min(1000, int(limit_raw)))
        except ValueError:
            return self._bad_request(request, "'limit' must be an integer")
        dead_letters = self.automation_engine.get_dead_letters(limit=limit)
        if self.slo_metrics:
            self.slo_metrics.set_gauge(
                "automation_dead_letters_backlog",
                float(self.automation_engine.dead_letter_count()),
            )
        return self._ok_response(
            request,
            {"dead_letters": dead_letters, "count": len(dead_letters)},
        )

    async def _handle_automation_dead_letter_replay(self, request: "web.Request") -> "web.Response":
        op_started = time.time()
        body = await self._parse_json(request)
        if body is None:
            if self.slo_metrics:
                self.slo_metrics.inc("automation_dead_letter_replay_total", label="bad_request")
            return self._bad_request(request, "Invalid JSON body")

        dead_letter_id = str(body.get("dead_letter_id", "")).strip()
        if not dead_letter_id:
            if self.slo_metrics:
                self.slo_metrics.inc("automation_dead_letter_replay_total", label="bad_request")
            return self._bad_request(request, "'dead_letter_id' is required")
        timeout_seconds = body.get("timeout_seconds", 10.0)
        remove_on_success = bool(body.get("remove_on_success", False))
        payload_override = body.get("payload_override", None)
        if payload_override is not None and not isinstance(payload_override, dict):
            if self.slo_metrics:
                self.slo_metrics.inc("automation_dead_letter_replay_total", label="bad_request")
            return self._bad_request(request, "'payload_override' must be an object")

        try:
            timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError):
            if self.slo_metrics:
                self.slo_metrics.inc("automation_dead_letter_replay_total", label="bad_request")
            return self._bad_request(request, "'timeout_seconds' must be a number")

        try:
            result = await self.automation_engine.replay_dead_letter(
                dead_letter_id,
                timeout_seconds=max(0.1, timeout_seconds),
                remove_on_success=remove_on_success,
                payload_override=payload_override,
            )
        except KeyError:
            if self.slo_metrics:
                self.slo_metrics.inc("automation_dead_letter_replay_total", label="not_found")
            return self._error_response(request, f"Dead letter not found: {dead_letter_id}", status=404)
        except Exception as exc:  # noqa: BLE001
            if self.slo_metrics:
                self.slo_metrics.inc("automation_dead_letter_replay_total", label="failed")
            return self._error_response(request, str(exc), status=400)

        if self.slo_metrics:
            self.slo_metrics.inc(
                "automation_dead_letter_replay_total",
                label="success" if result.get("succeeded") else "replayed_failed",
            )
            self.slo_metrics.observe_latency(
                "automation_dead_letter_replay_latency_ms",
                (time.time() - op_started) * 1000,
                label="replay",
            )
        self._record_audit(
            request,
            event_type="automation",
            action="replay_dead_letter",
            success=bool(result.get("succeeded", False)),
            metadata={
                "dead_letter_id": dead_letter_id,
                "removed": bool(result.get("removed", False)),
            },
        )
        return self._ok_response(request, result)

    async def _handle_automation_dead_letter_resolve(self, request: "web.Request") -> "web.Response":
        dead_letter_id = request.match_info.get("dead_letter_id", "")
        body = await self._parse_json(request) or {}
        reason = str(body.get("reason", "manual_resolve")).strip() or "manual_resolve"
        try:
            result = self.automation_engine.resolve_dead_letter(dead_letter_id, reason=reason)
        except KeyError:
            return self._error_response(request, f"Dead letter not found: {dead_letter_id}", status=404)
        self._record_audit(
            request,
            event_type="automation",
            action="resolve_dead_letter",
            success=True,
            metadata={"dead_letter_id": dead_letter_id, "reason": reason},
        )
        return self._ok_response(request, result)

    async def _handle_proactive_event(self, request: "web.Request") -> "web.Response":
        op_started = time.time()
        body = await self._parse_json(request)
        if body is None:
            if self.slo_metrics:
                self.slo_metrics.inc("proactive_event_total", label="bad_request")
            return self._bad_request(request, "Invalid JSON body")
        event_type = str(body.get("event_type", "")).strip()
        payload = body.get("payload", {})
        if not event_type:
            if self.slo_metrics:
                self.slo_metrics.inc("proactive_event_total", label="bad_request")
            return self._bad_request(request, "'event_type' is required")
        if not isinstance(payload, dict):
            if self.slo_metrics:
                self.slo_metrics.inc("proactive_event_total", label="bad_request")
            return self._bad_request(request, "'payload' must be an object")

        result = self.proactive_engine.ingest_event(event_type=event_type, payload=payload)
        if self.slo_metrics:
            self.slo_metrics.inc("proactive_event_total", label="success")
            self.slo_metrics.observe_latency(
                "proactive_event_latency_ms",
                (time.time() - op_started) * 1000,
                label=event_type,
            )
        self._record_audit(
            request,
            event_type="proactive",
            action="ingest_event",
            success=True,
            metadata={"event_type": event_type, "generated_count": result.get("generated_count", 0)},
        )
        return self._ok_response(request, result, status=201)

    async def _handle_proactive_preferences(self, request: "web.Request") -> "web.Response":
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        user_id = str(body.get("user_id", "default")).strip()
        preferences = body.get("preferences", {})
        if not user_id:
            return self._bad_request(request, "'user_id' is required")
        if not isinstance(preferences, dict):
            return self._bad_request(request, "'preferences' must be an object")
        result = self.proactive_engine.set_user_preferences(user_id=user_id, preferences=preferences)
        self._record_audit(
            request,
            event_type="proactive",
            action="set_preferences",
            success=True,
            metadata={"user_id": user_id, "keys": sorted(preferences.keys())},
        )
        return self._ok_response(request, result)

    async def _handle_proactive_suggestions(self, request: "web.Request") -> "web.Response":
        user_id = str(request.query.get("user_id", "default")).strip()
        if not user_id:
            return self._bad_request(request, "'user_id' is required")
        max_items_raw = request.query.get("max_items", "20")
        include_low_priority_raw = str(request.query.get("include_low_priority", "false")).strip().lower()
        include_low_priority = include_low_priority_raw in {"1", "true", "yes", "on"}
        try:
            max_items = int(max_items_raw)
        except ValueError:
            return self._bad_request(request, "'max_items' must be an integer")
        result = self.proactive_engine.list_suggestions(
            user_id=user_id,
            max_items=max_items,
            include_low_priority=include_low_priority,
        )
        return self._ok_response(request, result)

    async def _handle_proactive_profile(self, request: "web.Request") -> "web.Response":
        user_id = str(request.match_info.get("user_id", "default")).strip()
        if not user_id:
            return self._bad_request(request, "'user_id' is required")
        result = self.proactive_engine.get_user_profile(user_id=user_id)
        return self._ok_response(request, result)

    async def _handle_proactive_suggestion_ack(self, request: "web.Request") -> "web.Response":
        suggestion_id = str(request.match_info.get("suggestion_id", "")).strip()
        if not suggestion_id:
            return self._bad_request(request, "'suggestion_id' is required")
        try:
            result = self.proactive_engine.acknowledge_suggestion(suggestion_id=suggestion_id)
        except KeyError:
            return self._error_response(request, f"Suggestion not found: {suggestion_id}", status=404)
        self._record_audit(
            request,
            event_type="proactive",
            action="ack_suggestion",
            success=True,
            metadata={"suggestion_id": suggestion_id},
        )
        return self._ok_response(request, result)

    async def _handle_proactive_suggestion_dismiss(self, request: "web.Request") -> "web.Response":
        suggestion_id = str(request.match_info.get("suggestion_id", "")).strip()
        if not suggestion_id:
            return self._bad_request(request, "'suggestion_id' is required")
        try:
            result = self.proactive_engine.dismiss_suggestion(suggestion_id=suggestion_id)
        except KeyError:
            return self._error_response(request, f"Suggestion not found: {suggestion_id}", status=404)
        self._record_audit(
            request,
            event_type="proactive",
            action="dismiss_suggestion",
            success=True,
            metadata={"suggestion_id": suggestion_id},
        )
        return self._ok_response(request, result)

    async def _handle_proactive_suggestion_snooze(self, request: "web.Request") -> "web.Response":
        suggestion_id = str(request.match_info.get("suggestion_id", "")).strip()
        if not suggestion_id:
            return self._bad_request(request, "'suggestion_id' is required")
        body = await self._parse_json(request) or {}
        try:
            seconds = int(body.get("seconds", 600))
        except (TypeError, ValueError):
            return self._bad_request(request, "'seconds' must be an integer")
        try:
            result = self.proactive_engine.snooze_suggestion(suggestion_id=suggestion_id, seconds=max(1, seconds))
        except KeyError:
            return self._error_response(request, f"Suggestion not found: {suggestion_id}", status=404)
        self._record_audit(
            request,
            event_type="proactive",
            action="snooze_suggestion",
            success=True,
            metadata={"suggestion_id": suggestion_id, "seconds": max(1, seconds)},
        )
        return self._ok_response(request, result)

    async def _handle_proactive_execute_action(self, request: "web.Request") -> "web.Response":
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        user_id = str(body.get("user_id", "default")).strip() or "default"
        action_name = str(body.get("action_name", "")).strip()
        category = str(body.get("category", "general")).strip() or "general"
        priority = str(body.get("priority", "normal")).strip() or "normal"
        capability = str(body.get("capability", "")).strip()
        payload = body.get("payload", {})
        approval_token = str(body.get("approval_token", "")).strip() or None
        if not action_name:
            return self._bad_request(request, "'action_name' is required")
        if not capability:
            return self._bad_request(request, "'capability' is required")
        if not isinstance(payload, dict):
            return self._bad_request(request, "'payload' must be an object")
        if not self.orchestrator or not hasattr(self.orchestrator, "submit_task"):
            return self._error_response(request, "Proactive execution requires orchestrator", status=503)

        decision = self.proactive_engine.evaluate_autonomous_action(
            user_id=user_id,
            action_name=action_name,
            category=category,
            priority=priority,
        )
        requires_human = bool(decision.get("requires_approval", False))
        approval_action = str(body.get("approval_action", f"orchestrator:execute:{capability}")).strip()
        if requires_human:
            if not approval_token:
                return self._error_response(
                    request,
                    "Approval token required for this proactive action",
                    status=403,
                )
            if not self.approval_manager.validate_token(approval_token, expected_action=approval_action):
                return self._error_response(
                    request,
                    "Invalid or not approved token for proactive action",
                    status=403,
                )

        orch_task_id = await self.orchestrator.submit_task(
            description=f"proactive_action:{action_name}",
            required_capabilities=[capability],
            payload=payload,
            requires_human=requires_human,
            approval_token=approval_token,
            metadata={
                "category": "proactive_action",
                "proactive_user_id": user_id,
                "proactive_action_name": action_name,
                "approval_action": approval_action,
                "autonomous_decision": decision,
            },
        )
        status = "submitted"
        if hasattr(self.orchestrator, "get_task_status"):
            orch_task = self.orchestrator.get_task_status(orch_task_id)
            if orch_task is not None:
                status = orch_task.status.value
        self._record_audit(
            request,
            event_type="proactive",
            action="execute_action",
            success=True,
            metadata={
                "orchestrator_task_id": orch_task_id,
                "action_name": action_name,
                "requires_human": requires_human,
                "decision_reason": decision.get("reason", ""),
            },
        )
        return self._ok_response(
            request,
            {
                "orchestrator_task_id": orch_task_id,
                "status": status,
                "decision": decision,
            },
            status=202,
        )

    async def _handle_approval_request(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/approvals/request — create an approval request."""
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")

        action = str(body.get("action", "")).strip()
        reason = str(body.get("reason", "")).strip()
        resource = str(body.get("resource", "")).strip()
        ttl_seconds = int(body.get("ttl_seconds", 900))
        requested_by = request.headers.get("X-User-ID", str(body.get("requested_by", "api_user")))

        if not action:
            return self._bad_request(request, "'action' is required")
        if not reason:
            return self._bad_request(request, "'reason' is required")

        approval = self.approval_manager.create_request(
            action=action,
            requested_by=requested_by,
            reason=reason,
            resource=resource,
            ttl_seconds=ttl_seconds,
            metadata={"request_id": self._request_id(request)},
        )
        self._record_audit(
            request,
            event_type="approval",
            action="create_approval_request",
            success=True,
            metadata={"approval_id": approval.approval_id, "approval_action": action},
        )
        return self._ok_response(request, approval.to_dict(), status=201)

    async def _handle_approval_approve(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/approvals/{approval_id}/approve — approve a request."""
        approval_id = request.match_info.get("approval_id", "")
        body = await self._parse_json(request) or {}
        approver = request.headers.get("X-Approver-ID", str(body.get("approver", "approver")))
        note = str(body.get("note", "")).strip()

        approval = self.approval_manager.approve(approval_id, approver=approver, note=note)
        if approval is None:
            return self._error_response(request, f"Approval not found: {approval_id}", status=404)

        self._record_audit(
            request,
            event_type="approval",
            action="approve_request",
            success=True,
            metadata={"approval_id": approval_id, "approved_by": approver},
        )
        return self._ok_response(request, approval.to_dict())

    async def _handle_approval_reject(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/approvals/{approval_id}/reject — reject a request."""
        approval_id = request.match_info.get("approval_id", "")
        body = await self._parse_json(request) or {}
        approver = request.headers.get("X-Approver-ID", str(body.get("approver", "approver")))
        reason = str(body.get("reason", "")).strip()

        approval = self.approval_manager.reject(approval_id, approver=approver, reason=reason)
        if approval is None:
            return self._error_response(request, f"Approval not found: {approval_id}", status=404)

        self._record_audit(
            request,
            event_type="approval",
            action="reject_request",
            success=True,
            metadata={"approval_id": approval_id, "approved_by": approver, "reason": reason},
        )
        return self._ok_response(request, approval.to_dict())

    async def _handle_approval_get(self, request: "web.Request") -> "web.Response":
        """GET /api/v1/approvals/{approval_id} — approval status."""
        approval_id = request.match_info.get("approval_id", "")
        approval = self.approval_manager.get(approval_id)
        if approval is None:
            return self._error_response(request, f"Approval not found: {approval_id}", status=404)
        return self._ok_response(request, approval.to_dict())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _parse_json(request: "web.Request") -> dict[str, Any] | None:
        try:
            return await request.json()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _coerce_chat_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text = str(item.get("text", "")).strip()
                        if text:
                            parts.append(text)
                elif isinstance(item, str):
                    text = item.strip()
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(content or "").strip()

    @staticmethod
    def _request_id(request: "web.Request") -> str:
        request_id = request.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id
        return str(uuid.uuid4())

    def _ok_response(
        self,
        request: "web.Request",
        data: Any,
        *,
        status: int = 200,
    ) -> "web.Response":
        return web.json_response(
            APIResponse(
                success=True,
                data=data,
                request_id=self._request_id(request),
            ).to_dict(),
            status=status,
        )

    def _error_response(
        self,
        request: "web.Request",
        message: str,
        *,
        status: int = 400,
    ) -> "web.Response":
        return web.json_response(
            APIResponse(
                success=False,
                error=message,
                request_id=self._request_id(request),
            ).to_dict(),
            status=status,
        )

    def _bad_request(self, request: "web.Request", message: str) -> "web.Response":
        return self._error_response(request, message, status=400)

    def _record_audit(
        self,
        request: "web.Request",
        *,
        event_type: str,
        action: str,
        success: bool,
        decision: str = "allow",
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.audit_logger:
            return
        actor = request.headers.get("X-User-ID", "api_user")
        event = AuditEvent(
            event_type=event_type,
            action=action,
            request_id=self._request_id(request),
            actor=actor,
            resource=request.path,
            decision=decision,
            success=success,
            reason=reason,
            metadata=metadata or {},
        )
        self.audit_logger.record(event)

    @staticmethod
    def _constant_time_compare(a: str, b: str) -> bool:
        """Timing-safe string comparison to prevent token-oracle attacks."""
        ha = hashlib.sha256(a.encode()).digest()
        hb = hashlib.sha256(b.encode()).digest()
        return hmac.compare_digest(ha, hb)

    def __repr__(self) -> str:
        return f"<APIInterface {self.host}:{self.port} running={self._running}>"
