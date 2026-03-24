"""
REST API interface for JARVIS AI OS.

Provides HTTP endpoints for querying JARVIS, managing agents/tasks/skills,
and checking system health. Uses aiohttp when available, falls back to
http.server for basic functionality.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import hashlib
import os
import hmac
import json
import os
import re
import shlex
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from core.response_contracts import (
    CodeAssistRequest,
    CodeWorkflowRequest,
    RepoUnderstandRequest,
    RequestEnvelope,
    VerifiedResponse,
)
from core.response_fallbacks import get_fallback
from core.response_governance import apply_response_governance

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
from infrastructure.policy_cost_engine import PolicyContext, PolicyCostEngine, PolicyDecision
from infrastructure.research_intelligence import ResearchIntelligenceEngine
from infrastructure.software_delivery import SoftwareDeliveryEngine
from infrastructure.slo_metrics import SLOMetrics, evaluate_slo_snapshot, get_slo_metrics
from infrastructure.ingress_control import IngressController
from infrastructure.tool_isolation import ToolIsolationPolicy
from infrastructure.live_stream_ingest import LiveStreamIngestService
from infrastructure.realtime_stt import RealtimeSTTService
from infrastructure.person_identity_registry import PersonIdentityRegistry
from infrastructure.world_knowledge import WorldKnowledgeService
from infrastructure.neo4j_graph_store import Neo4jGraphStore

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

    def __init__(self, requests_per_minute: int = 600) -> None:
        self._rpm = max(1, int(requests_per_minute or 1))
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
        requests_per_minute: int | None = None,
        cors_origins: list[str] | None = None,
        slo_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._auth_token = auth_token
        rpm = requests_per_minute
        if rpm is None:
            try:
                rpm = int(os.getenv("JARVIS_API_RATE_LIMIT_RPM", "600") or 600)
            except Exception:
                rpm = 600
        self._rate_limiter = RateLimiter(rpm)
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
        self.policy_cost_engine: PolicyCostEngine = PolicyCostEngine.from_env()
        self.ingress_controller: IngressController = IngressController.from_env()
        self.tool_isolation_policy: ToolIsolationPolicy = ToolIsolationPolicy.from_env()
        self.live_stream_ingest: LiveStreamIngestService = LiveStreamIngestService()
        self.realtime_stt: RealtimeSTTService = RealtimeSTTService()
        self.person_identity_registry: PersonIdentityRegistry = PersonIdentityRegistry.from_env()
        self.world_knowledge: WorldKnowledgeService = WorldKnowledgeService.from_env()
        self.profile_graph_store: Neo4jGraphStore = Neo4jGraphStore(
            enabled=str(os.getenv("JARVIS_RESEARCH_NEO4J_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"},
            uri=str(os.getenv("JARVIS_RESEARCH_NEO4J_URI", "bolt://127.0.0.1:7687")).strip(),
            username=str(os.getenv("JARVIS_RESEARCH_NEO4J_USERNAME", "neo4j")).strip(),
            password=str(os.getenv("JARVIS_RESEARCH_NEO4J_PASSWORD", "")).strip(),
            database=str(os.getenv("JARVIS_RESEARCH_NEO4J_DATABASE", "neo4j")).strip(),
        )
        register_research_automation_actions(self.automation_engine, self.research_engine)
        self.slo_metrics: SLOMetrics = get_slo_metrics()

        # In-memory task status store
        self._tasks: dict[str, dict[str, Any]] = {}
        self._workspace_by_user: dict[str, str] = {}
        self._workspace_by_session: dict[str, str] = {}
        self._world_enrichment_jobs: dict[str, dict[str, Any]] = {}
        self._world_enrichment_tasks: dict[str, asyncio.Task[Any]] = {}
        self._session_notifications: dict[str, list[dict[str, Any]]] = {}
        self._realtime_ws_subscribers: dict[str, set[Any]] = {}
        self._realtime_visual_summary_tasks: dict[str, asyncio.Task[Any]] = {}
        self._realtime_detection_allow_web = str(
            os.getenv("JARVIS_REALTIME_DETECTION_ALLOW_WEB", "false")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._realtime_async_visual_summary = str(
            os.getenv("JARVIS_REALTIME_ASYNC_VISUAL_SUMMARY", "true")
        ).strip().lower() in {"1", "true", "yes", "on"}
        try:
            self._query_timeout_s = max(5.0, float(os.getenv("JARVIS_QUERY_TIMEOUT_SECONDS", "45") or 45))
        except Exception:
            self._query_timeout_s = 45.0
        try:
            self._realtime_turn_timeout_s = max(
                3.0,
                float(os.getenv("JARVIS_REALTIME_TURN_TIMEOUT_SECONDS", "20") or 20),
            )
        except Exception:
            self._realtime_turn_timeout_s = 20.0
        try:
            self._stream_default_interval_ms = max(
                500,
                min(10000, int(os.getenv("JARVIS_STREAM_DEFAULT_INTERVAL_MS", "2500") or 2500)),
            )
        except Exception:
            self._stream_default_interval_ms = 2500

        self._app: Any = None   # aiohttp.web.Application
        self._runner: Any = None
        self._site: Any = None
        self._running = False
        self._perf_log_enabled = str(os.getenv("JARVIS_API_PERF_LOG_ENABLED", "true")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            self._perf_slow_ms = float(os.getenv("JARVIS_API_PERF_SLOW_MS", "1500") or 1500)
        except Exception:
            self._perf_slow_ms = 1500.0

        logger.info("APIInterface configured (%s:%d aiohttp=%s)", host, port, _AIOHTTP_AVAILABLE)

    # ------------------------------------------------------------------
    # Service injection helpers
    # ------------------------------------------------------------------

    def set_conversation_manager(self, cm: Any) -> None:
        self.conversation_manager = cm
        try:
            self.live_stream_ingest.set_conversation_manager(cm)
        except Exception:
            pass

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
        for task in list(self._realtime_visual_summary_tasks.values()):
            if task and not task.done():
                task.cancel()
        self._realtime_visual_summary_tasks.clear()
        with contextlib.suppress(Exception):
            await self.live_stream_ingest.shutdown()
        with contextlib.suppress(Exception):
            self.profile_graph_store.close()
        if self._runner:
            await self._runner.cleanup()
        logger.info("API server stopped")

    # ------------------------------------------------------------------
    # aiohttp app construction
    # ------------------------------------------------------------------

    def _build_app(self) -> "web.Application":
        # Raise request body limit above aiohttp default (~1MB) so bulk research
        # ingest payloads don't fail with misleading JSON parse errors.
        app = web.Application(client_max_size=32 * 1024**2, middlewares=[
            self._request_context_middleware,
            self._cors_middleware,
            self._auth_middleware,
            self._rate_limit_middleware,
            self._error_middleware,
        ])
        app.router.add_post("/api/v1/query", self._handle_query)
        app.router.add_post("/api/v1/realtime/sessions/start", self._handle_realtime_start)
        app.router.add_post("/api/v1/realtime/sessions/{session_id}/media", self._handle_realtime_media)
        app.router.add_post("/api/v1/realtime/sessions/{session_id}/interrupt", self._handle_realtime_interrupt)
        app.router.add_post("/api/v1/realtime/sessions/{session_id}/turn", self._handle_realtime_turn)
        app.router.add_get("/api/v1/realtime/sessions/{session_id}/ws", self._handle_realtime_ws)
        app.router.add_get("/api/v1/realtime/sessions/{session_id}/notifications", self._handle_realtime_notifications)
        app.router.add_get("/api/v1/realtime/sessions/{session_id}/social/timeline", self._handle_realtime_social_timeline)
        app.router.add_get("/api/v1/realtime/sessions/{session_id}/social/explain", self._handle_realtime_social_explain)
        app.router.get("/api/v1/realtime/sessions/{session_id}/streams", self._handle_realtime_streams_list)
        app.router.add_post("/api/v1/realtime/sessions/{session_id}/streams/start", self._handle_realtime_stream_start)
        app.router.add_post("/api/v1/realtime/sessions/{session_id}/streams/{stream_id}/stop", self._handle_realtime_stream_stop)
        app.router.add_get("/api/v1/vision/identities", self._handle_vision_identity_list)
        app.router.add_get("/api/v1/vision/identities/{person_id}/samples", self._handle_vision_identity_samples_list)
        app.router.add_post("/api/v1/vision/identities/enroll", self._handle_vision_identity_enroll)
        app.router.add_post("/api/v1/vision/identities/recognize", self._handle_vision_identity_recognize)
        app.router.add_delete("/api/v1/vision/identities/{person_id}", self._handle_vision_identity_delete)
        app.router.add_delete("/api/v1/vision/identities/{person_id}/samples/{sample_id}", self._handle_vision_identity_sample_delete)
        app.router.add_get("/api/v1/world/concepts", self._handle_world_concepts_list)
        app.router.add_get("/api/v1/world/concepts/{concept_id}", self._handle_world_concept_get)
        app.router.add_get("/api/v1/world/concepts/{concept_id}/profile-graph", self._handle_world_concept_profile_graph)
        app.router.add_patch("/api/v1/world/concepts/{concept_id}", self._handle_world_concept_update)
        app.router.add_delete("/api/v1/world/concepts/{concept_id}", self._handle_world_concept_delete)
        app.router.add_post("/api/v1/world/teach", self._handle_world_teach)
        app.router.add_post("/api/v1/world/concepts/{concept_id}/links", self._handle_world_concept_link_add)
        app.router.add_delete("/api/v1/world/concepts/{concept_id}/links/{link_id}", self._handle_world_concept_link_delete)
        app.router.add_post(
            "/api/v1/world/concepts/{concept_id}/links/{link_id}/interactions",
            self._handle_world_concept_link_interaction,
        )
        app.router.add_post(
            "/api/v1/world/concepts/{concept_id}/links/{link_id}/browser-use/run",
            self._handle_world_concept_link_browser_use_run,
        )
        app.router.add_post("/api/v1/world/concepts/{concept_id}/enrich", self._handle_world_concept_enrich)
        app.router.add_post("/api/v1/world/detections/enrich", self._handle_world_detections_enrich)
        app.router.add_get("/api/v1/world/enrichment/jobs/{job_id}", self._handle_world_enrichment_job_get)
        # OpenAI-compatible compatibility endpoints for IDE clients (e.g. Continue).
        app.router.add_get("/v1/models", self._handle_openai_models)
        app.router.add_post("/v1/chat/completions", self._handle_openai_chat_completions)
        app.router.add_get("/api/v1/agents", self._handle_list_agents)
        app.router.add_post("/api/v1/tasks", self._handle_submit_task)
        app.router.add_post("/api/v1/code/assist", self._handle_code_assist)
        app.router.add_post("/api/v1/code/understand", self._handle_code_understand)
        app.router.add_post("/api/v1/code/workflow", self._handle_code_workflow)
        app.router.add_get("/api/v1/tasks/{task_id}", self._handle_get_task)
        app.router.add_post("/api/v1/tasks/{task_id}/retry", self._handle_retry_task)
        app.router.add_post("/api/v1/tasks/{task_id}/replan", self._handle_replan_task)
        app.router.add_post("/api/v1/plans", self._handle_submit_plan)
        app.router.add_get("/api/v1/plans/{plan_id}", self._handle_get_plan)
        app.router.add_get("/api/v1/workflows/{workflow_id}/checkpoint", self._handle_get_workflow_checkpoint)
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
        app.router.add_get("/metrics", self._handle_prometheus_metrics)
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
        if request.path in ("/api/v1/health", "/metrics") or request.method == "OPTIONS":
            return await handler(request)
        if self._auth_token:
            token = self._extract_request_token(request)
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

    def _extract_request_token(self, request: "web.Request") -> str:
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        if token:
            return token
        # Browser WS clients cannot set custom Authorization headers reliably;
        # allow query-token auth on realtime websocket endpoints only.
        if "/api/v1/realtime/sessions/" in request.path and request.path.endswith("/ws"):
            query_token = str(request.query.get("access_token", "")).strip()
            if query_token:
                return query_token
            query_token = str(request.query.get("token", "")).strip()
            if query_token:
                return query_token
        return ""

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
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type,Authorization,X-Scopes,X-User-ID,X-Approver-ID,X-Request-ID,"
                "X-Jarvis-Workspace,X-Workspace-Path,X-Jarvis-Active-File,X-Jarvis-Selection"
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

    async def _ingress_acquire_or_response(
        self,
        request: "web.Request",
        *,
        route_label: str,
    ) -> tuple[bool, "web.Response | None", float]:
        decision = await self.ingress_controller.acquire()
        if not decision.allowed:
            if self.slo_metrics:
                self.slo_metrics.inc("ingress_reject_total", label=decision.reason or route_label)
            return (
                False,
                self._error_response(
                    request,
                    f"Request rejected at ingress: {decision.reason or 'unavailable'}",
                    status=429,
                ),
                0.0,
            )
        if self.slo_metrics:
            self.slo_metrics.observe_latency(
                "ingress_queue_wait_ms",
                float(decision.queue_wait_ms or 0.0),
                label=route_label,
            )
            self.slo_metrics.set_gauge(
                "ingress_inflight",
                float(self.ingress_controller.snapshot().get("inflight", 0) or 0),
            )
            self.slo_metrics.set_gauge(
                "ingress_queued",
                float(self.ingress_controller.snapshot().get("queued", 0) or 0),
            )
        return True, None, float(decision.queue_wait_ms or 0.0)

    async def _ingress_release(self) -> None:
        await self.ingress_controller.release()

    async def _run_realtime_visual_summary_task(
        self,
        *,
        session_id: str,
        source: str,
        image_url: str,
        image_b64: str,
        note: str,
        metadata: dict[str, Any],
        ts_val: float | None,
    ) -> None:
        cm = self.conversation_manager
        if cm is None or not hasattr(cm, "summarize_visual_observation") or not hasattr(cm, "ingest_realtime_frame"):
            return
        try:
            summary = await cm.summarize_visual_observation(
                session_id,
                source=source,
                image_url=image_url,
                image_b64=image_b64,
                note=note,
                metadata=metadata,
            )
            summary = str(summary or "").strip()
            if not summary:
                return
            enrich_meta = dict(metadata or {})
            enrich_meta["async_visual_summary"] = True
            cm.ingest_realtime_frame(
                session_id,
                source=f"{source}:vision",
                summary=summary,
                metadata=enrich_meta,
                ts=ts_val,
            )
        except Exception:
            pass
        finally:
            self._realtime_visual_summary_tasks.pop(str(session_id or "").strip(), None)

    def _queue_realtime_visual_summary(
        self,
        *,
        session_id: str,
        source: str,
        image_url: str,
        image_b64: str,
        note: str,
        metadata: dict[str, Any],
        ts_val: float | None,
    ) -> bool:
        sid = str(session_id or "").strip()
        if not sid:
            return False
        prev = self._realtime_visual_summary_tasks.get(sid)
        if prev and not prev.done():
            return False
        task = asyncio.create_task(
            self._run_realtime_visual_summary_task(
                session_id=sid,
                source=source,
                image_url=image_url,
                image_b64=image_b64,
                note=note,
                metadata=dict(metadata or {}),
                ts_val=ts_val,
            ),
            name=f"rt_visual_summary:{sid}",
        )
        self._realtime_visual_summary_tasks[sid] = task
        return True

    @staticmethod
    def _extract_url(text: str) -> str:
        m = re.search(r"(https?://[^\s,;]+)", str(text or ""), re.IGNORECASE)
        return str(m.group(1)).strip() if m else ""

    def _derive_profile_topic_for_user(self, *, user_id: str) -> str:
        uid = str(user_id or "").strip() or "api_user"
        cm = self.conversation_manager
        if cm is not None and hasattr(cm, "get_user_profile_snapshot"):
            try:
                snap = cm.get_user_profile_snapshot(uid)
                if isinstance(snap, dict):
                    traits = snap.get("traits", {})
                    if isinstance(traits, dict):
                        name = str(traits.get("display_name", "")).strip()
                        if name and name.lower() not in {"api_user", "anonymous", "user", "default", "me"}:
                            return name
            except Exception:
                pass
        # Fallback: if exactly one enrolled identity exists, assume it is the current user.
        try:
            identities = self.person_identity_registry.list_identities()
            if isinstance(identities, list) and len(identities) == 1:
                only = identities[0] if isinstance(identities[0], dict) else {}
                name = str(only.get("display_name", "")).strip()
                if name:
                    return name
        except Exception:
            pass
        return uid

    def _derive_profile_linkedin_url_for_user(self, *, user_id: str) -> str:
        uid = str(user_id or "").strip() or "api_user"
        cm = self.conversation_manager
        if cm is not None and hasattr(cm, "get_user_profile_snapshot"):
            try:
                snap = cm.get_user_profile_snapshot(uid)
                if isinstance(snap, dict):
                    traits = snap.get("traits", {})
                    if isinstance(traits, dict):
                        url = str(traits.get("linkedin_url", "")).strip()
                        if "linkedin.com" in url.lower():
                            return url
            except Exception:
                pass
        return ""

    @staticmethod
    def _extract_professional_traits_from_result(result: dict[str, Any], *, fallback_url: str = "") -> dict[str, Any]:
        concept = dict(result.get("concept", {}) if isinstance(result, dict) else {})
        web_facts = [x for x in list(concept.get("web_facts", [])) if isinstance(x, dict)]
        refs = [x for x in list(concept.get("reference_links", [])) if isinstance(x, dict)]
        snippets: list[str] = []
        for row in web_facts[:12]:
            for key in ("snippet", "title"):
                txt = str(row.get(key, "")).strip()
                if txt:
                    snippets.append(txt)
        latest_note = str(concept.get("latest_note", "")).strip()
        if latest_note:
            snippets.append(latest_note)
        pool = " ".join(snippets)
        out: dict[str, Any] = {}
        linkedin_url = ""
        for r in refs:
            url = str(r.get("url", "")).strip()
            if "linkedin.com" in url.lower():
                linkedin_url = url
                break
        if not linkedin_url:
            for wf in web_facts:
                url = str(wf.get("url", "")).strip()
                if "linkedin.com" in url.lower():
                    linkedin_url = url
                    break
        if not linkedin_url and "linkedin.com" in str(fallback_url or "").lower():
            linkedin_url = str(fallback_url or "").strip()
        if linkedin_url:
            out["linkedin_url"] = linkedin_url

        role_match = re.search(
            r"\b(founder|co-founder|ceo|cto|cfo|coo|engineer|developer|manager|director|consultant|scientist|designer|analyst)\b",
            pool,
            re.IGNORECASE,
        )
        if role_match:
            out["occupation_title"] = role_match.group(1).strip()
        work_at = re.search(r"\b(?:works?|working)\s+at\s+([A-Z][A-Za-z0-9&.,'() \-]{1,80})", pool)
        if work_at:
            out["employer"] = work_at.group(1).strip(" .,!?\t\r\n")
        if not out.get("employer"):
            at_match = re.search(r"\bat\s+([A-Z][A-Za-z0-9&.,'() \-]{1,80})", pool)
            if at_match and str(out.get("occupation_title", "")).strip():
                out["employer"] = at_match.group(1).strip(" .,!?\t\r\n")
        summary = ""
        for s in snippets:
            if len(s.split()) >= 6:
                summary = s.strip()
                break
        if summary:
            out["professional_summary"] = summary[:240]
        return out

    def _parse_enrichment_request(self, *, query: str, context: dict[str, Any], user_id: str) -> dict[str, Any] | None:
        text = str(query or "").strip()
        if not text:
            return None
        low = text.lower()
        if not re.search(r"\b(enrich|enrichment|research|learn|fetch|get|lookup|crawl|scrape|find|information)\b", low):
            return None
        if not (
            "profile" in low
            or "about" in low
            or "from website" in low
            or "from web" in low
            or "from internet" in low
            or "background" in low
            or "linkedin" in low
        ):
            return None
        linkedin_requested = "linkedin" in low
        github_requested = "github" in low
        google_requested = bool(re.search(r"\bgoogle(?:\s+search)?\b", low))
        about_me = bool(
            re.search(r"\b(about me|for me|my profile|my linkedin|me on linkedin)\b", low)
        )
        concept_id = str(context.get("concept_id", "")).strip() if isinstance(context, dict) else ""
        topic = ""
        quoted = re.search(r"['\"]([^'\"]{2,120})['\"]", text)
        if quoted:
            topic = str(quoted.group(1)).strip()
        if not topic:
            m = re.search(r"\b(?:for|about|on)\s+([a-z0-9][a-z0-9\s.'-]{1,100})", text, re.IGNORECASE)
            if m:
                topic = str(m.group(1)).strip()
        if not topic:
            m = re.search(r"\bprofile\s+(?:for|of)\s+([a-z0-9][a-z0-9\s.'-]{1,100})", text, re.IGNORECASE)
            if m:
                topic = str(m.group(1)).strip()
        topic = re.sub(
            r"\s+(?:from|using|with)\s+(?:https?://\S+|website|web|internet).*$",
            "",
            topic,
            flags=re.IGNORECASE,
        ).strip()
        topic = re.sub(r"\s+", " ", topic).strip()
        topic_low = str(topic or "").strip().lower()
        if topic_low in {
            "me",
            "about me",
            "for me",
            "my profile",
            "my linkedin",
            "me on linkedin",
            "me on linked in",
            "linkedin profile",
            "my linkedin profile",
        }:
            topic = ""
        if about_me and not topic:
            topic = self._derive_profile_topic_for_user(user_id=str(user_id or "").strip())
        if not topic and re.search(r"\bme\s+on\s+linkedin\b", low, re.IGNORECASE):
            topic = self._derive_profile_topic_for_user(user_id=str(user_id or "").strip())
        url = self._extract_url(text)
        if about_me and linkedin_requested and not url:
            url = self._derive_profile_linkedin_url_for_user(user_id=str(user_id or "").strip())
        topic_low = str(topic or "").strip().lower()
        if about_me and linkedin_requested and not url and topic_low in {"", "api_user", "anonymous", "user", "me"}:
            return {
                "needs_identity": True,
                "message": "I need your name or direct LinkedIn profile URL to target the correct profile.",
            }
        if linkedin_requested and topic:
            topic = re.sub(r"\blinked\s*in\b", "", topic, flags=re.IGNORECASE)
            topic = re.sub(r"\bprofile\b", "", topic, flags=re.IGNORECASE)
            topic = re.sub(r"\s+", " ", topic).strip()
        if github_requested and topic and "github" not in topic.lower():
            topic = f"{topic} github"
        if google_requested and topic:
            topic = re.sub(r"\bgoogle(?:\s+search)?\b", "", topic, flags=re.IGNORECASE).strip() or topic
        if not concept_id and not topic:
            return None
        target_source = "web"
        if linkedin_requested:
            target_source = "linkedin"
        elif github_requested:
            target_source = "github"
        elif google_requested:
            target_source = "google"
        max_items_val = 6
        if isinstance(context, dict):
            try:
                max_items_val = max(1, min(12, int(context.get("max_items", 6) or 6)))
            except Exception:
                max_items_val = 6
        return {
            "concept_id": concept_id,
            "topic": topic,
            "url": url,
            "about_me": about_me,
            "target_source": target_source,
            "run_adapters": bool(context.get("run_adapters", True)) if isinstance(context, dict) else True,
            "max_items": max_items_val,
        }

    def _resolve_concept_id_for_enrichment(
        self,
        *,
        topic: str,
        concept_id: str,
        session_id: str,
        user_id: str,
    ) -> str:
        cid = str(concept_id or "").strip()
        if cid:
            return cid
        clean_topic = str(topic or "").strip()
        if not clean_topic:
            raise ValueError("topic is required")
        topic_norm = clean_topic.lower()
        for row in self.world_knowledge.list_concepts(limit=500):
            if not isinstance(row, dict):
                continue
            if str(row.get("topic", "")).strip().lower() == topic_norm:
                found = str(row.get("concept_id", "")).strip()
                if found:
                    return found
        concept = self.world_knowledge.teach_concept(
            topic=clean_topic,
            notes="Queued via conversation enrichment request.",
            tags=["profile", "conversation"],
            metadata={"source": "conversation", "session_id": session_id, "user_id": user_id},
        )
        return str(concept.get("concept_id", "")).strip()

    def _push_session_notification(self, session_id: str, payload: dict[str, Any]) -> None:
        sid = str(session_id or "").strip()
        if not sid:
            return
        rows = self._session_notifications.get(sid, [])
        item = dict(payload)
        rows.insert(0, item)
        self._session_notifications[sid] = rows[:120]
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._fanout_realtime_notification(sid, item))
        except Exception:
            pass

    async def _fanout_realtime_notification(self, session_id: str, payload: dict[str, Any]) -> None:
        sid = str(session_id or "").strip()
        if not sid:
            return
        listeners = list(self._realtime_ws_subscribers.get(sid, set()))
        if not listeners:
            return
        stale: list[Any] = []
        for ws in listeners:
            try:
                await ws.send_json({"type": "notification", "session_id": sid, "notification": dict(payload)})
            except Exception:
                stale.append(ws)
        if stale:
            cur = self._realtime_ws_subscribers.get(sid, set())
            for ws in stale:
                cur.discard(ws)
            if cur:
                self._realtime_ws_subscribers[sid] = cur
            else:
                self._realtime_ws_subscribers.pop(sid, None)

    def _drain_session_notifications(self, session_id: str, *, limit: int = 6, drain: bool = True) -> list[dict[str, Any]]:
        sid = str(session_id or "").strip()
        if not sid:
            return []
        rows = list(self._session_notifications.get(sid, []))
        lim = max(1, min(50, int(limit or 6)))
        out = rows[:lim]
        if drain and out:
            self._session_notifications[sid] = rows[lim:]
        return out

    @staticmethod
    def _format_notification_lines(notifications: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for item in notifications:
            if not isinstance(item, dict):
                continue
            txt = str(item.get("message", "")).strip()
            if txt:
                lines.append(f"[Update] {txt}")
        return "\n".join(lines[:5]).strip()

    def _prepend_notifications(self, *, base_response: str, notifications: list[dict[str, Any]]) -> str:
        note_text = self._format_notification_lines(notifications)
        body = str(base_response or "").strip()
        if note_text and body:
            return f"{note_text}\n\n{body}"
        if note_text:
            return note_text
        return body

    def _queue_world_enrichment_job(
        self,
        *,
        session_id: str,
        user_id: str,
        query: str,
        concept_id: str,
        topic: str,
        url: str,
        target_source: str,
        max_items: int,
        run_adapters: bool,
    ) -> dict[str, Any]:
        sid = str(session_id or "").strip()
        job_id = f"wej_{uuid.uuid4().hex[:12]}"
        job = {
            "job_id": job_id,
            "session_id": sid,
            "user_id": str(user_id or "api_user").strip() or "api_user",
            "status": "queued",
            "query": str(query or "").strip(),
            "concept_id": str(concept_id or "").strip(),
            "topic": str(topic or "").strip(),
            "url": str(url or "").strip(),
            "target_source": str(target_source or "web").strip().lower() or "web",
            "max_items": max(1, min(12, int(max_items or 6))),
            "run_adapters": bool(run_adapters),
            "created_at": time.time(),
            "started_at": 0.0,
            "completed_at": 0.0,
            "result": None,
            "error": None,
        }
        self._world_enrichment_jobs[job_id] = job
        task = asyncio.create_task(
            self._run_world_enrichment_job(job_id),
            name=f"world_enrich:{job_id}",
        )
        self._world_enrichment_tasks[job_id] = task
        return job

    async def _run_world_enrichment_job(self, job_id: str) -> None:
        job = self._world_enrichment_jobs.get(str(job_id or "").strip())
        if not isinstance(job, dict):
            return
        session_id = str(job.get("session_id", "")).strip()
        try:
            job["status"] = "running"
            job["started_at"] = time.time()
            concept_id = self._resolve_concept_id_for_enrichment(
                topic=str(job.get("topic", "")).strip(),
                concept_id=str(job.get("concept_id", "")).strip(),
                session_id=session_id,
                user_id=str(job.get("user_id", "api_user")),
            )
            result: dict[str, Any]
            url = str(job.get("url", "")).strip()
            target_source = str(job.get("target_source", "web")).strip().lower() or "web"
            used_linkedin_mcp = False
            used_source_mcp = False
            if target_source == "linkedin" and self.connector_registry is not None:
                try:
                    mcp_payload = await self.connector_registry.invoke(
                        "linkedin_mcp",
                        "enrich_profile",
                        {
                            "query": str(job.get("topic", "")).strip(),
                            "user_id": str(job.get("user_id", "api_user")).strip(),
                            "profile_url": url,
                        },
                        actor_scopes={"connector:linkedin:read"},
                    )
                    result = self._build_world_result_from_linkedin_mcp(
                        concept_id=concept_id,
                        topic=str(job.get("topic", "")).strip(),
                        payload=dict(mcp_payload or {}),
                    )
                    used_linkedin_mcp = True
                    used_source_mcp = True
                except Exception as exc:
                    job["linkedin_mcp_error"] = str(exc)
            if target_source == "github" and self.connector_registry is not None:
                try:
                    mcp_payload = await self.connector_registry.invoke(
                        "github_mcp",
                        "enrich_profile",
                        {
                            "query": str(job.get("topic", "")).strip(),
                            "user_id": str(job.get("user_id", "api_user")).strip(),
                            "url": url,
                        },
                        actor_scopes={"connector:github:read"},
                    )
                    result = self._build_world_result_from_github_mcp(
                        concept_id=concept_id,
                        topic=str(job.get("topic", "")).strip(),
                        payload=dict(mcp_payload or {}),
                    )
                    used_source_mcp = True
                except Exception as exc:
                    job["github_mcp_error"] = str(exc)
            if target_source == "google" and self.connector_registry is not None:
                try:
                    mcp_payload = await self.connector_registry.invoke(
                        "google_search_mcp",
                        "enrich_profile",
                        {
                            "query": str(job.get("topic", "")).strip(),
                            "user_id": str(job.get("user_id", "api_user")).strip(),
                            "max_results": max(1, min(10, int(job.get("max_items", 6) or 6))),
                        },
                        actor_scopes={"connector:google:read"},
                    )
                    result = self._build_world_result_from_google_mcp(
                        concept_id=concept_id,
                        topic=str(job.get("topic", "")).strip(),
                        payload=dict(mcp_payload or {}),
                    )
                    used_source_mcp = True
                except Exception as exc:
                    job["google_mcp_error"] = str(exc)
            if not url and target_source == "linkedin":
                topic = str(job.get("topic", "")).strip()
                if topic:
                    # Target people search rather than scraping the LinkedIn homepage.
                    url = f"https://www.linkedin.com/search/results/people/?keywords={quote_plus(topic)}"
                    job["url"] = url
            if not url and target_source == "github":
                topic = str(job.get("topic", "")).strip()
                if topic:
                    url = f"https://github.com/search?q={quote_plus(topic)}"
                    job["url"] = url
            if not used_source_mcp and url:
                concept = self.world_knowledge.add_reference_link(
                    concept_id=concept_id,
                    url=url,
                    title="Conversation-requested profile source" if target_source != "linkedin" else "LinkedIn people search source",
                    notes="Queued from conversational enrichment request.",
                    source_type="conversation_linkedin" if target_source == "linkedin" else "conversation",
                    tags=["profile", "enrichment", target_source],
                )
                link_id = ""
                for ref in list(concept.get("reference_links", [])):
                    if not isinstance(ref, dict):
                        continue
                    if str(ref.get("url", "")).strip().lower() == url.lower():
                        link_id = str(ref.get("link_id", "")).strip()
                        break
                if not link_id:
                    raise RuntimeError("Failed to resolve reference link for enrichment")
                result = self.world_knowledge.run_link_learning(
                    concept_id=concept_id,
                    link_id=link_id,
                    research_engine=self.research_engine,
                    max_items=max(1, min(12, int(job.get("max_items", 6) or 6))),
                    run_adapters=bool(job.get("run_adapters", True)),
                )
            elif not used_source_mcp:
                result = self.world_knowledge.enrich_concept_from_web(
                    concept_id=concept_id,
                    research_engine=self.research_engine,
                    max_items=max(1, min(12, int(job.get("max_items", 6) or 6))),
                    run_adapters=bool(job.get("run_adapters", True)),
                )
            topic = str((result.get("concept", {}) if isinstance(result, dict) else {}).get("topic", "")).strip() or str(
                job.get("topic", "")
            ).strip() or "requested profile"
            job["concept_id"] = concept_id
            job["status"] = "completed"
            job["completed_at"] = time.time()
            job["result"] = result
            self._sync_enrichment_to_user_profile(job=job, result=result)
            self._push_session_notification(
                session_id,
                {
                    "id": f"ntf_{uuid.uuid4().hex[:12]}",
                    "type": "world_enrichment_completed",
                    "job_id": str(job.get("job_id", "")),
                    "at": time.time(),
                    "message": f"Enrichment finished for {topic}. Say 'show enrichment result' to review details.",
                },
            )
        except Exception as exc:  # noqa: BLE001
            job["status"] = "failed"
            job["completed_at"] = time.time()
            job["error"] = str(exc)
            self._push_session_notification(
                session_id,
                {
                    "id": f"ntf_{uuid.uuid4().hex[:12]}",
                    "type": "world_enrichment_failed",
                    "job_id": str(job.get("job_id", "")),
                    "at": time.time(),
                    "message": f"Enrichment failed for {str(job.get('topic', '')).strip() or 'requested profile'}: {exc}",
                },
            )
        finally:
            self._world_enrichment_tasks.pop(str(job_id or "").strip(), None)

    def _sync_enrichment_to_user_profile(self, *, job: dict[str, Any], result: dict[str, Any]) -> None:
        cm = self.conversation_manager
        if cm is None or not hasattr(cm, "remember_user_profile_traits"):
            return
        user_id = str(job.get("user_id", "api_user")).strip() or "api_user"
        traits = self._extract_professional_traits_from_result(
            dict(result or {}),
            fallback_url=str(job.get("url", "")).strip(),
        )
        topic = str(job.get("topic", "")).strip()
        if topic and not traits.get("display_name"):
            if topic.lower() not in {"profile", "my profile", "requested profile"}:
                traits["display_name"] = topic
        if not traits:
            return
        try:
            cm.remember_user_profile_traits(
                user_id,
                traits=traits,
                source="world_enrichment_job",
            )
        except Exception:
            pass

    @staticmethod
    def _collect_named_values(obj: Any, keys: set[str], out: list[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k or "").strip().lower()
                if key in keys and isinstance(v, str) and v.strip():
                    out.append(v.strip())
                APIInterface._collect_named_values(v, keys, out)
            return
        if isinstance(obj, list):
            for v in obj:
                APIInterface._collect_named_values(v, keys, out)

    def _ingest_profile_document_to_research(
        self,
        *,
        concept_id: str,
        topic: str,
        profile_url: str,
        name: str,
        headline: str,
        company: str,
        summary: str,
        raw_payload: dict[str, Any],
    ) -> dict[str, Any]:
        concept_key = str(concept_id or "").strip()
        clean_topic = str(topic or "").strip() or str(name or "").strip() or "profile"
        clean_name = str(name or "").strip() or clean_topic
        clean_headline = str(headline or "").strip()
        clean_company = str(company or "").strip()
        clean_summary = str(summary or "").strip()
        clean_url = str(profile_url or "").strip() or f"jarvis://profile/{concept_key or clean_name.lower().replace(' ', '_')}"

        sections: list[str] = [f"Profile name: {clean_name}"]
        if clean_headline:
            sections.append(f"Headline: {clean_headline}")
        if clean_company:
            sections.append(f"Company: {clean_company}")
        if clean_summary:
            sections.append(f"Summary: {clean_summary}")
        sections.append(f"Source URL: {clean_url}")
        content = "\n".join(sections).strip()

        metadata: dict[str, Any] = {
            "source_kind": "profile_linkedin",
            "concept_id": concept_key,
            "profile_name": clean_name,
            "profile_headline": clean_headline,
            "profile_company": clean_company,
            "profile_url": clean_url,
        }
        if isinstance(raw_payload, dict) and raw_payload:
            # Keep payload in metadata as compact JSON for traceability in retrieval.
            metadata["linkedin_payload_json"] = json.dumps(raw_payload, ensure_ascii=True, sort_keys=True)[:8000]

        try:
            ingest = self.research_engine.ingest_sources(
                [
                    {
                        "title": f"{clean_name} LinkedIn Profile",
                        "url": clean_url,
                        "content": content,
                        "topic": clean_topic,
                        "source_type": "official",
                        "metadata": metadata,
                    }
                ]
            )
            return {
                "ok": True,
                "inserted": int(ingest.get("inserted", 0) or 0),
                "skipped_duplicates": int(ingest.get("skipped_duplicates", 0) or 0),
                "total_sources": int(ingest.get("total_sources", 0) or 0),
                "topic": clean_topic,
                "url": clean_url,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "error": str(exc),
                "topic": clean_topic,
                "url": clean_url,
            }

    def _build_world_result_from_linkedin_mcp(
        self,
        *,
        concept_id: str,
        topic: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        data = dict(payload or {})
        raw_result = data.get("result", data)
        urls: list[str] = []
        names: list[str] = []
        headlines: list[str] = []
        companies: list[str] = []
        summaries: list[str] = []
        self._collect_named_values(raw_result, {"linkedin_url", "profile_url", "public_profile_url", "url"}, urls)
        self._collect_named_values(raw_result, {"name", "full_name", "display_name"}, names)
        self._collect_named_values(raw_result, {"headline", "title", "position", "occupation"}, headlines)
        self._collect_named_values(raw_result, {"company", "employer", "organization"}, companies)
        self._collect_named_values(raw_result, {"summary", "about", "bio", "description"}, summaries)

        profile_url = next((u for u in urls if "linkedin.com" in u.lower()), urls[0] if urls else "")
        name = names[0] if names else str(topic or "").strip()
        headline = headlines[0] if headlines else ""
        company = companies[0] if companies else ""
        summary = summaries[0] if summaries else ""
        note_parts = [p for p in [name, headline, company, summary] if str(p).strip()]
        note = " | ".join(note_parts[:3]).strip()
        if not note:
            note = f"LinkedIn MCP enrichment captured for {str(topic or '').strip() or 'profile'}."
        concept = self.world_knowledge.update_concept(
            concept_id=concept_id,
            notes=note,
            metadata={
                "linkedin_mcp_last_run_at": time.time(),
                "linkedin_mcp_name": name,
                "linkedin_mcp_headline": headline,
                "linkedin_mcp_company": company,
            },
        )
        if profile_url:
            concept = self.world_knowledge.add_reference_link(
                concept_id=concept_id,
                url=profile_url,
                title=f"{name} LinkedIn Profile".strip(),
                notes=headline or summary,
                source_type="linkedin_mcp",
                tags=["linkedin", "mcp", "profile"],
            )
        ingest_result = self._ingest_profile_document_to_research(
            concept_id=concept_id,
            topic=topic,
            profile_url=profile_url,
            name=name,
            headline=headline,
            company=company,
            summary=summary,
            raw_payload=data,
        )
        concept = self.world_knowledge.update_concept(
            concept_id=concept_id,
            metadata={
                "linkedin_profile_vector_ingest_ok": bool(ingest_result.get("ok", False)),
                "linkedin_profile_vector_inserted": int(ingest_result.get("inserted", 0) or 0),
                "linkedin_profile_vector_skipped_duplicates": int(ingest_result.get("skipped_duplicates", 0) or 0),
                "linkedin_profile_vector_total_sources": int(ingest_result.get("total_sources", 0) or 0),
            },
        )
        graph = self._sync_profile_graph(concept)
        return {
            "concept": concept,
            "added_facts": len([x for x in [name, headline, company, summary, profile_url] if str(x).strip()]),
            "query_result_count": 1,
            "source": "linkedin_mcp",
            "mcp_payload": data,
            "profile_graph": graph,
            "profile_vector_ingest": ingest_result,
        }

    def _build_world_result_from_github_mcp(
        self,
        *,
        concept_id: str,
        topic: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        data = dict(payload or {})
        raw_result = data.get("result", data)
        urls: list[str] = []
        names: list[str] = []
        descs: list[str] = []
        langs: list[str] = []
        self._collect_named_values(raw_result, {"html_url", "url", "repo_url", "repository_url"}, urls)
        self._collect_named_values(raw_result, {"name", "full_name", "login", "repo"}, names)
        self._collect_named_values(raw_result, {"description", "summary", "bio"}, descs)
        self._collect_named_values(raw_result, {"language"}, langs)
        repo_name = names[0] if names else str(topic or "").strip()
        description = descs[0] if descs else ""
        language = langs[0] if langs else ""
        note_parts = [x for x in [repo_name, description, language] if str(x).strip()]
        note = " | ".join(note_parts[:3]).strip() or f"GitHub MCP enrichment captured for {str(topic or '').strip()}."
        concept = self.world_knowledge.update_concept(
            concept_id=concept_id,
            notes=note,
            metadata={
                "github_mcp_last_run_at": time.time(),
                "github_mcp_name": repo_name,
                "github_mcp_language": language,
            },
        )
        added_links = 0
        for u in urls[:5]:
            if "github.com" not in str(u).lower():
                continue
            concept = self.world_knowledge.add_reference_link(
                concept_id=concept_id,
                url=u,
                title=f"{repo_name} GitHub".strip(),
                notes=description,
                source_type="github_mcp",
                tags=["github", "mcp", "repository"],
            )
            added_links += 1
        graph = self._sync_profile_graph(concept)
        return {
            "concept": concept,
            "added_facts": len([x for x in [repo_name, description, language] if str(x).strip()]) + added_links,
            "query_result_count": max(1, added_links),
            "source": "github_mcp",
            "mcp_payload": data,
            "profile_graph": graph,
        }

    def _build_world_result_from_google_mcp(
        self,
        *,
        concept_id: str,
        topic: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        data = dict(payload or {})
        raw_result = data.get("result", data)
        urls: list[str] = []
        titles: list[str] = []
        snippets: list[str] = []
        self._collect_named_values(raw_result, {"url", "link", "href"}, urls)
        self._collect_named_values(raw_result, {"title", "name"}, titles)
        self._collect_named_values(raw_result, {"snippet", "summary", "description"}, snippets)
        top_title = titles[0] if titles else str(topic or "").strip()
        top_snippet = snippets[0] if snippets else ""
        note_parts = [x for x in [top_title, top_snippet] if str(x).strip()]
        note = " | ".join(note_parts[:2]).strip() or f"Google MCP enrichment captured for {str(topic or '').strip()}."
        concept = self.world_knowledge.update_concept(
            concept_id=concept_id,
            notes=note,
            metadata={
                "google_mcp_last_run_at": time.time(),
                "google_mcp_query": str(topic or "").strip(),
            },
        )
        added_links = 0
        for idx, u in enumerate(urls[:6]):
            if not str(u).strip():
                continue
            title = titles[idx] if idx < len(titles) and str(titles[idx]).strip() else f"Google result {idx + 1}"
            note_item = snippets[idx] if idx < len(snippets) and str(snippets[idx]).strip() else ""
            concept = self.world_knowledge.add_reference_link(
                concept_id=concept_id,
                url=u,
                title=title,
                notes=note_item,
                source_type="google_mcp",
                tags=["google", "mcp", "search"],
            )
            added_links += 1
        graph = self._sync_profile_graph(concept)
        return {
            "concept": concept,
            "added_facts": len([x for x in [top_title, top_snippet] if str(x).strip()]) + added_links,
            "query_result_count": max(1, added_links),
            "source": "google_mcp",
            "mcp_payload": data,
            "profile_graph": graph,
        }

    def _maybe_queue_enrichment_from_query(
        self,
        *,
        session_id: str,
        user_id: str,
        query: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        req = self._parse_enrichment_request(query=query, context=context, user_id=user_id)
        if not req:
            return None
        if bool(req.get("needs_identity")):
            msg = str(req.get("message", "")).strip() or "I need more details to target the right profile."
            return {"job": {}, "ack": msg}
        try:
            cid = self._resolve_concept_id_for_enrichment(
                topic=str(req.get("topic", "")).strip(),
                concept_id=str(req.get("concept_id", "")).strip(),
                session_id=session_id,
                user_id=user_id,
            )
        except Exception:
            cid = str(req.get("concept_id", "")).strip()
        job = self._queue_world_enrichment_job(
            session_id=session_id,
            user_id=user_id,
            query=query,
            concept_id=cid,
            topic=str(req.get("topic", "")).strip(),
            url=str(req.get("url", "")).strip(),
            target_source=str(req.get("target_source", "web")).strip().lower() or "web",
            max_items=max(1, min(12, int(req.get("max_items", 6) or 6))),
            run_adapters=bool(req.get("run_adapters", True)),
        )
        topic = str(req.get("topic", "")).strip() or "requested profile"
        src = " with the provided website source" if str(req.get("url", "")).strip() else ""
        ack = (
            f"Queued enrichment for {topic}{src}. "
            f"I will update you when it finishes. (job {str(job.get('job_id', ''))})"
        )
        self._push_session_notification(
            session_id,
            {
                "id": f"ntf_{uuid.uuid4().hex[:12]}",
                "type": "world_enrichment_queued",
                "job_id": str(job.get("job_id", "")),
                "at": time.time(),
                "message": ack,
            },
        )
        return {"job": job, "ack": ack}

    def _maybe_enrichment_status_reply(self, *, session_id: str, query: str) -> dict[str, Any] | None:
        sid = str(session_id or "").strip()
        low = str(query or "").strip().lower()
        if not sid or not low:
            return None
        if not (
            "enrichment status" in low
            or "enrichment result" in low
            or "job status" in low
            or "what did you enrich" in low
            or "show enrichment" in low
        ):
            return None
        rows = [
            dict(v)
            for v in self._world_enrichment_jobs.values()
            if isinstance(v, dict) and str(v.get("session_id", "")).strip() == sid
        ]
        if not rows:
            return {"response": "No enrichment jobs have been queued in this session yet."}
        rows.sort(key=lambda r: float(r.get("created_at", 0.0) or 0.0), reverse=True)
        latest = rows[0]
        status = str(latest.get("status", "unknown")).strip().lower()
        topic = str(latest.get("topic", "")).strip() or "requested profile"
        if status in {"queued", "running"}:
            return {
                "response": f"Enrichment for {topic} is {status}. I will update you when it completes.",
                "job": latest,
            }
        if status == "failed":
            err = str(latest.get("error", "")).strip() or "unknown error"
            return {"response": f"Latest enrichment for {topic} failed: {err}", "job": latest}
        result = latest.get("result", {})
        added_facts = int((result.get("added_facts", 0) if isinstance(result, dict) else 0) or 0)
        query_count = int((result.get("query_result_count", 0) if isinstance(result, dict) else 0) or 0)
        return {
            "response": (
                f"Latest enrichment for {topic} is complete. "
                f"Added {added_facts} fact(s) from {query_count} web result(s)."
            ),
            "job": latest,
        }

    async def _handle_query(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/query — submit a natural-language query to JARVIS."""
        perf_started = time.time()
        perf_stages: dict[str, float] = {}
        allowed, reject_response, queue_wait_ms = await self._ingress_acquire_or_response(
            request, route_label="query"
        )
        perf_stages["ingress_wait"] = round(float(queue_wait_ms or 0.0), 2)
        if not allowed:
            perf_stages["total"] = round((time.time() - perf_started) * 1000.0, 2)
            self._log_perf_breakdown(
                request=request,
                route="query",
                outcome="ingress_reject",
                stages_ms=perf_stages,
            )
            return reject_response or self._error_response(request, "Ingress rejected", status=429)
        try:
            parse_started = time.time()
            body = await self._parse_json(request)
            perf_stages["parse_json"] = round((time.time() - parse_started) * 1000.0, 2)
            if body is None:
                perf_stages["total"] = round((time.time() - perf_started) * 1000.0, 2)
                self._log_perf_breakdown(
                    request=request,
                    route="query",
                    outcome="bad_request_invalid_json",
                    stages_ms=perf_stages,
                )
                return self._bad_request(request, "Invalid JSON body")

            query = body.get("query", "").strip()
            if not query:
                perf_stages["total"] = round((time.time() - perf_started) * 1000.0, 2)
                self._log_perf_breakdown(
                    request=request,
                    route="query",
                    outcome="bad_request_missing_query",
                    stages_ms=perf_stages,
                )
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
                status_reply = self._maybe_enrichment_status_reply(session_id=str(session_id or "").strip(), query=str(query))
                if status_reply:
                    return self._ok_response(
                        request,
                        {
                            "response": str(status_reply.get("response", "")).strip(),
                            "session_id": session_id,
                            "background_job": status_reply.get("job", {}),
                        },
                    )
                queued = self._maybe_queue_enrichment_from_query(
                    session_id=str(session_id or "").strip(),
                    user_id=str(user_id or "api_user"),
                    query=str(query),
                    context=context if isinstance(context, dict) else {},
                )
                if queued:
                    ack = str(queued.get("ack", "")).strip()
                    return self._ok_response(
                        request,
                        {
                            "response": ack,
                            "session_id": session_id,
                            "background_job": queued.get("job", {}),
                        },
                        status=202,
                    )
                policy_started = time.time()
                policy_decision = self.policy_cost_engine.decide(
                    PolicyContext(
                        route="query",
                        task_type="general_query",
                        user_id=str(user_id),
                        sla_tier=str(body.get("sla_tier", "standard")),
                        latency_sensitive=bool(body.get("latency_sensitive", False)),
                        max_latency_ms=(
                            int(body.get("max_latency_ms")) if body.get("max_latency_ms") is not None else None
                        ),
                        budget_usd=(
                            float(body.get("budget_usd")) if body.get("budget_usd") is not None else None
                        ),
                        privacy_level=str(body.get("privacy_level", "medium")),
                    )
                )
                perf_stages["policy"] = round((time.time() - policy_started) * 1000.0, 2)
                if not policy_decision.allow:
                    if self.slo_metrics:
                        self.slo_metrics.inc("policy_decision_total", label=policy_decision.reason or "deny")
                        self.slo_metrics.inc("policy_deny_total", label=policy_decision.reason or "deny")
                    perf_stages["total"] = round((time.time() - perf_started) * 1000.0, 2)
                    self._log_perf_breakdown(
                        request=request,
                        route="query",
                        outcome="policy_deny",
                        stages_ms=perf_stages,
                        extra={
                            "session_id": str(session_id or ""),
                            "modality": modality,
                            "reason": str(policy_decision.reason or ""),
                        },
                    )
                    return self._error_response(
                        request,
                        f"Request denied by policy: {policy_decision.reason}",
                        status=429,
                    )
                context = dict(context)
                context["policy_decision"] = policy_decision.to_dict()
                conversation_started = time.time()
                response_text = await self._call_conversation_manager(
                    session_id=session_id,
                    query=query,
                    modality=modality,
                    media=media,
                    context=context,
                )
                perf_stages["conversation"] = round((time.time() - conversation_started) * 1000.0, 2)
                governance_hints: dict[str, Any] = {"user_input": query}
                try:
                    if session_id and self.conversation_manager and hasattr(self.conversation_manager, "get_context"):
                        ctx = self.conversation_manager.get_context(session_id)
                        if ctx is not None and isinstance(getattr(ctx, "metadata", None), dict):
                            md = ctx.metadata
                            if isinstance(md.get("response_plan"), dict):
                                governance_hints["response_plan"] = dict(md.get("response_plan", {}))
                            if isinstance(md.get("query_understanding"), dict):
                                governance_hints["query_understanding"] = dict(md.get("query_understanding", {}))
                except Exception:  # noqa: BLE001
                    pass
                governance_started = time.time()
                governed = apply_response_governance(
                    str(response_text),
                    route="chat",
                    hints=governance_hints,
                )
                perf_stages["governance"] = round((time.time() - governance_started) * 1000.0, 2)
                notifications = self._drain_session_notifications(str(session_id or ""), limit=6, drain=True)
                response_text = self._prepend_notifications(base_response=governed.text, notifications=notifications)
                if self.slo_metrics:
                    self.slo_metrics.inc("response_governance_total", label=governed.route)
                    self.slo_metrics.inc(
                        "response_governance_tier_total",
                        label=f"{governed.route}:{governed.verbosity_tier}",
                    )
                    self.slo_metrics.set_gauge(
                        "response_governance_min_words_last",
                        float(governed.min_words),
                        label=governed.route,
                    )
                    self.slo_metrics.set_gauge(
                        "response_governance_max_words_last",
                        float(governed.max_words),
                        label=governed.route,
                    )
                    self.slo_metrics.set_gauge(
                        "response_governance_words_last",
                        float(governed.word_count),
                        label=governed.route,
                    )
                    if governed.changed:
                        self.slo_metrics.inc("response_governance_changed_total", label=governed.route)
                    if governed.rejected:
                        self.slo_metrics.inc(
                            "response_governance_rejected_total",
                            label=f"{governed.route}:{governed.reason or 'unknown'}",
                        )
                self._record_audit(
                    request,
                    event_type="query",
                    action="conversation_query",
                    success=True,
                    metadata={"session_id": session_id, "user_id": user_id},
                )
                perf_stages["total"] = round((time.time() - perf_started) * 1000.0, 2)
                self._log_perf_breakdown(
                    request=request,
                    route="query",
                    outcome="ok",
                    stages_ms=perf_stages,
                    extra={"session_id": str(session_id or ""), "modality": modality},
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
            perf_stages["total"] = round((time.time() - perf_started) * 1000.0, 2)
            self._log_perf_breakdown(
                request=request,
                route="query",
                outcome="ok_echo",
                stages_ms=perf_stages,
                extra={"modality": modality},
            )
            return self._ok_response(
                request,
                {"response": f"Echo: {query}", "session_id": session_id or "none"},
            )
        finally:
            await self._ingress_release()

    async def _handle_realtime_start(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/realtime/sessions/start — create/activate realtime loop session."""
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        user_id = str(body.get("user_id", "api_user")).strip() or "api_user"
        session_id = str(body.get("session_id", "")).strip() or None
        max_frames_raw = body.get("max_frames", 12)
        try:
            max_frames = max(1, min(64, int(max_frames_raw)))
        except Exception:
            max_frames = 12
        cm = self.conversation_manager
        if cm is None or not hasattr(cm, "start_realtime_session"):
            return self._error_response(request, "Realtime session support is unavailable", status=501)
        sid = cm.start_realtime_session(user_id=user_id, session_id=session_id, max_frames=max_frames)
        snap = cm.get_realtime_session(sid) if hasattr(cm, "get_realtime_session") else {"session_id": sid}
        return self._ok_response(request, {"session_id": sid, "realtime": snap})

    async def _handle_realtime_media(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/realtime/sessions/{session_id}/media — append camera/screen grounding frame."""
        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return self._bad_request(request, "session_id is required")
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        source = str(body.get("source", "screen")).strip() or "screen"
        summary = str(body.get("summary", "")).strip()
        image_url = (
            str(body.get("image_url", "")).strip()
            or str(body.get("frame_url", "")).strip()
            or str(body.get("snapshot_url", "")).strip()
        )
        image_b64 = str(body.get("image_b64", "")).strip()
        metadata = body.get("metadata", {})
        if not isinstance(metadata, dict):
            return self._bad_request(request, "'metadata' must be an object")
        note = str(body.get("note", "")).strip()
        ts_raw = body.get("ts")
        ts_val = None
        if ts_raw is not None:
            try:
                ts_val = float(ts_raw)
            except Exception:
                ts_val = None
        cm = self.conversation_manager
        if cm is None or not hasattr(cm, "ingest_realtime_frame"):
            return self._error_response(request, "Realtime media ingestion support is unavailable", status=501)
        queued_visual_summary = False
        if not summary:
            if (image_url or image_b64) and hasattr(cm, "summarize_visual_observation"):
                if self._realtime_async_visual_summary:
                    queued_visual_summary = self._queue_realtime_visual_summary(
                        session_id=session_id,
                        source=source,
                        image_url=image_url,
                        image_b64=image_b64,
                        note=note,
                        metadata=metadata,
                        ts_val=ts_val,
                    )
                    summary = note or f"Live {source} frame queued for async vision summary."
                else:
                    try:
                        summary = await asyncio.wait_for(
                            cm.summarize_visual_observation(
                                session_id,
                                source=source,
                                image_url=image_url,
                                image_b64=image_b64,
                                note=note,
                                metadata=metadata,
                            ),
                            timeout=self._realtime_turn_timeout_s,
                        )
                    except Exception:
                        summary = note
            if not summary:
                return self._bad_request(
                    request,
                    "Either 'summary' or one of ('image_url','frame_url','snapshot_url','image_b64') is required",
                )
        if image_url:
            metadata = dict(metadata)
            metadata["image_url"] = image_url
        if image_b64:
            metadata = dict(metadata)
            metadata["image_b64_size"] = len(image_b64)
        detections = metadata.get("detections", [])
        if isinstance(detections, list) and detections:
            labels = []
            for d in detections:
                if not isinstance(d, dict):
                    continue
                lbl = str(d.get("label", "")).strip().lower()
                if lbl:
                    labels.append(lbl)
            if labels:
                try:
                    world_ctx = self.world_knowledge.enrich_detection_labels(
                        labels=labels,
                        research_engine=self.research_engine,
                        max_items_per_label=2,
                        allow_web=self._realtime_detection_allow_web,
                    )
                    if world_ctx:
                        metadata = dict(metadata)
                        metadata["web_context"] = world_ctx
                except Exception:
                    pass
        snap = cm.ingest_realtime_frame(
            session_id,
            source=source,
            summary=summary,
            metadata=metadata,
            ts=ts_val,
        )
        return self._ok_response(
            request,
            {
                "session_id": session_id,
                "realtime": snap,
                "visual_summary_queued": queued_visual_summary,
            },
        )

    async def _handle_realtime_social_timeline(self, request: "web.Request") -> "web.Response":
        """GET /api/v1/realtime/sessions/{session_id}/social/timeline — social scene events and coverage."""
        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return self._bad_request(request, "session_id is required")
        limit_raw = str(request.query.get("limit", "40")).strip() or "40"
        try:
            limit = max(1, min(200, int(limit_raw)))
        except Exception:
            limit = 40
        cm = self.conversation_manager
        if cm is None or not hasattr(cm, "get_realtime_social_timeline"):
            return self._error_response(request, "Realtime social timeline support is unavailable", status=501)
        out = cm.get_realtime_social_timeline(session_id, limit=limit)
        return self._ok_response(request, out if isinstance(out, dict) else {"session_id": session_id, "items": []})

    async def _handle_realtime_social_explain(self, request: "web.Request") -> "web.Response":
        """GET /api/v1/realtime/sessions/{session_id}/social/explain — explain latest or selected social event."""
        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return self._bad_request(request, "session_id is required")
        event_id = str(request.query.get("event_id", "")).strip()
        cm = self.conversation_manager
        if cm is None or not hasattr(cm, "explain_realtime_social_event"):
            return self._error_response(request, "Realtime social explain support is unavailable", status=501)
        out = cm.explain_realtime_social_event(session_id, event_id=event_id)
        return self._ok_response(request, out if isinstance(out, dict) else {"session_id": session_id, "found": False})

    async def _handle_realtime_notifications(self, request: "web.Request") -> "web.Response":
        """GET /api/v1/realtime/sessions/{session_id}/notifications — background updates for this session."""
        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return self._bad_request(request, "session_id is required")
        limit_raw = str(request.query.get("limit", "20")).strip() or "20"
        try:
            limit = max(1, min(100, int(limit_raw)))
        except Exception:
            limit = 20
        drain_raw = str(request.query.get("drain", "true")).strip().lower()
        drain = drain_raw in {"1", "true", "yes", "on"}
        rows = self._drain_session_notifications(session_id, limit=limit, drain=drain)
        return self._ok_response(
            request,
            {
                "session_id": session_id,
                "count": len(rows),
                "drained": drain,
                "items": rows,
            },
        )

    async def _handle_world_enrichment_job_get(self, request: "web.Request") -> "web.Response":
        """GET /api/v1/world/enrichment/jobs/{job_id} — fetch background enrichment job status."""
        job_id = str(request.match_info.get("job_id", "")).strip()
        if not job_id:
            return self._bad_request(request, "job_id is required")
        job = self._world_enrichment_jobs.get(job_id)
        if not isinstance(job, dict):
            return self._error_response(request, "Enrichment job not found", status=404)
        return self._ok_response(request, {"job": dict(job)})

    async def _handle_realtime_streams_list(self, request: "web.Request") -> "web.Response":
        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return self._bad_request(request, "session_id is required")
        items = self.live_stream_ingest.list_streams(session_id=session_id)
        return self._ok_response(request, {"session_id": session_id, "streams": items})

    async def _handle_realtime_stream_start(self, request: "web.Request") -> "web.Response":
        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return self._bad_request(request, "session_id is required")
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        source_type = str(body.get("source_type", "http")).strip().lower() or "http"
        source_url = str(body.get("source_url", "")).strip()
        if not source_url:
            return self._bad_request(request, "'source_url' is required")
        interval_ms_raw = body.get("interval_ms", self._stream_default_interval_ms)
        try:
            interval_ms = max(500, min(10000, int(interval_ms_raw)))
        except Exception:
            interval_ms = self._stream_default_interval_ms
        note = str(body.get("note", "")).strip()
        metadata = body.get("metadata", {})
        if not isinstance(metadata, dict):
            return self._bad_request(request, "'metadata' must be an object")
        if self.conversation_manager and hasattr(self.conversation_manager, "start_realtime_session"):
            user_id = str(body.get("user_id", "api_user")).strip() or "api_user"
            self.conversation_manager.start_realtime_session(user_id=user_id, session_id=session_id)
        try:
            stream = await self.live_stream_ingest.start_stream(
                session_id=session_id,
                source_type=source_type,
                source_url=source_url,
                interval_ms=interval_ms,
                note=note,
                metadata=metadata,
            )
        except Exception as exc:
            return self._error_response(request, f"Failed to start stream: {exc}", status=500)
        return self._ok_response(request, {"session_id": session_id, "stream": stream})

    async def _handle_realtime_stream_stop(self, request: "web.Request") -> "web.Response":
        session_id = str(request.match_info.get("session_id", "")).strip()
        stream_id = str(request.match_info.get("stream_id", "")).strip()
        if not session_id:
            return self._bad_request(request, "session_id is required")
        if not stream_id:
            return self._bad_request(request, "stream_id is required")
        snap = await self.live_stream_ingest.stop_stream(stream_id)
        if snap is None:
            return self._error_response(request, "Stream not found", status=404)
        return self._ok_response(request, {"session_id": session_id, "stream": snap})

    async def _handle_vision_identity_list(self, request: "web.Request") -> "web.Response":
        rows = self.person_identity_registry.list_identities()
        return self._ok_response(
            request,
            {
                "count": len(rows),
                "identities": rows,
            },
        )

    async def _handle_vision_identity_enroll(self, request: "web.Request") -> "web.Response":
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        display_name = str(body.get("name", "")).strip()
        if not display_name:
            return self._bad_request(request, "'name' is required")
        metadata = body.get("metadata", {})
        if not isinstance(metadata, dict):
            return self._bad_request(request, "'metadata' must be an object")
        raw_samples = body.get("samples")
        samples: list[str] = []
        if isinstance(raw_samples, list):
            samples.extend(str(s).strip() for s in raw_samples if str(s).strip())
        single = str(body.get("image_b64", "")).strip()
        if single:
            samples.append(single)
        if not samples:
            return self._bad_request(request, "At least one sample image is required")
        try:
            enrolled = self.person_identity_registry.enroll(
                display_name=display_name,
                sample_images_b64=samples,
                metadata=metadata,
            )
        except ValueError as exc:
            return self._bad_request(request, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, f"Enrollment failed: {exc}", status=500)
        return self._ok_response(
            request,
            {
                "identity": enrolled,
                "registered_identities": self.person_identity_registry.count(),
            },
            status=201,
        )

    async def _handle_vision_identity_samples_list(self, request: "web.Request") -> "web.Response":
        person_id = str(request.match_info.get("person_id", "")).strip()
        if not person_id:
            return self._bad_request(request, "person_id is required")
        try:
            rows = self.person_identity_registry.list_samples(person_id)
        except KeyError:
            return self._error_response(request, "Identity not found", status=404)
        return self._ok_response(
            request,
            {
                "person_id": person_id,
                "count": len(rows),
                "samples": rows,
            },
        )

    async def _handle_vision_identity_recognize(self, request: "web.Request") -> "web.Response":
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        raw_samples = body.get("samples", [])
        samples: list[dict[str, Any]] = []
        if isinstance(raw_samples, list):
            for idx, item in enumerate(raw_samples):
                if not isinstance(item, dict):
                    continue
                image_b64 = str(item.get("image_b64", "")).strip()
                if not image_b64:
                    continue
                samples.append(
                    {
                        "sample_id": str(item.get("sample_id", f"s{idx}")).strip(),
                        "detection_index": item.get("detection_index"),
                        "image_b64": image_b64,
                    }
                )
        if not samples:
            image_b64 = str(body.get("image_b64", "")).strip()
            if image_b64:
                samples = [{"sample_id": "single", "detection_index": 0, "image_b64": image_b64}]
        if not samples:
            return self._bad_request(request, "At least one sample image is required")
        matches = self.person_identity_registry.recognize_samples(samples)
        return self._ok_response(
            request,
            {
                "count": len(matches),
                "matches": matches,
                "registered_identities": self.person_identity_registry.count(),
            },
        )

    async def _handle_vision_identity_delete(self, request: "web.Request") -> "web.Response":
        person_id = str(request.match_info.get("person_id", "")).strip()
        if not person_id:
            return self._bad_request(request, "person_id is required")
        deleted = self.person_identity_registry.delete(person_id)
        if not deleted:
            return self._error_response(request, "Identity not found", status=404)
        return self._ok_response(
            request,
            {
                "deleted": True,
                "person_id": person_id,
                "registered_identities": self.person_identity_registry.count(),
            },
        )

    async def _handle_vision_identity_sample_delete(self, request: "web.Request") -> "web.Response":
        person_id = str(request.match_info.get("person_id", "")).strip()
        sample_id = str(request.match_info.get("sample_id", "")).strip()
        if not person_id:
            return self._bad_request(request, "person_id is required")
        if not sample_id:
            return self._bad_request(request, "sample_id is required")
        try:
            identity = self.person_identity_registry.delete_sample(person_id, sample_id)
        except KeyError as exc:
            msg = str(exc).lower()
            if "sample" in msg:
                return self._error_response(request, "Sample not found", status=404)
            return self._error_response(request, "Identity not found", status=404)
        except ValueError as exc:
            return self._bad_request(request, str(exc))
        return self._ok_response(
            request,
            {
                "deleted": True,
                "person_id": person_id,
                "sample_id": sample_id,
                "identity": identity,
            },
        )

    def _active_profile_graph_store(self) -> Neo4jGraphStore | None:
        engine_store = getattr(self.research_engine, "_graph_store", None)
        if isinstance(engine_store, Neo4jGraphStore) and bool(getattr(engine_store, "enabled", False)):
            return engine_store
        if bool(getattr(self.profile_graph_store, "enabled", False)):
            return self.profile_graph_store
        return None

    @staticmethod
    def _collect_profile_entities_from_concept(concept: dict[str, Any]) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        md = dict(concept.get("metadata", {}) or {}) if isinstance(concept, dict) else {}
        person_id = str(md.get("person_id", "")).strip()
        if person_id:
            entities.append(
                {
                    "entity_type": "identity",
                    "value": person_id,
                    "relation": "HAS_IDENTITY",
                    "confidence": 1.0,
                    "source": "enrollment",
                    "metadata": {"kind": "person_id"},
                }
            )
        display_name = str(md.get("identity_name", "")).strip() or str(concept.get("topic", "")).strip()
        if display_name:
            entities.append(
                {
                    "entity_type": "person",
                    "value": display_name,
                    "relation": "IS_PERSON",
                    "confidence": 1.0,
                    "source": "concept_topic",
                    "metadata": {},
                }
            )
        role = str(md.get("linkedin_mcp_headline", "")).strip()
        if role:
            entities.append(
                {
                    "entity_type": "role",
                    "value": role,
                    "relation": "HAS_ROLE",
                    "confidence": 0.84,
                    "source": "linkedin_mcp",
                    "metadata": {},
                }
            )
        company = str(md.get("linkedin_mcp_company", "")).strip()
        if company:
            entities.append(
                {
                    "entity_type": "organization",
                    "value": company,
                    "relation": "WORKS_AT",
                    "confidence": 0.86,
                    "source": "linkedin_mcp",
                    "metadata": {},
                }
            )
        for row in list(concept.get("reference_links", [])):
            if not isinstance(row, dict):
                continue
            url = str(row.get("url", "")).strip()
            if not url:
                continue
            entities.append(
                {
                    "entity_type": "source",
                    "value": url,
                    "relation": "EVIDENCED_BY",
                    "confidence": 0.72,
                    "source": str(row.get("source_type", "")).strip() or "reference_link",
                    "metadata": {"title": str(row.get("title", "")).strip()},
                }
            )
        return entities

    def _sync_profile_graph(self, concept: dict[str, Any]) -> dict[str, Any]:
        concept_id = str(concept.get("concept_id", "")).strip()
        if not concept_id:
            return {"enabled": False, "nodes": [], "edges": [], "reason": "missing_concept_id"}
        profile_id = f"profile:{concept_id}"
        graph_store = self._active_profile_graph_store()
        entities = self._collect_profile_entities_from_concept(concept)
        if graph_store is not None:
            graph_store.upsert_profile(
                profile_id=profile_id,
                concept_id=concept_id,
                display_name=str(concept.get("topic", "")).strip(),
                person_id=str((concept.get("metadata", {}) or {}).get("person_id", "")).strip(),
            )
            for ent in entities:
                etype = str(ent.get("entity_type", "")).strip()
                value = str(ent.get("value", "")).strip()
                relation = str(ent.get("relation", "")).strip()
                if not etype or not value or not relation:
                    continue
                if relation == "EVIDENCED_BY":
                    graph_store.upsert_profile_source(
                        profile_id=profile_id,
                        url=value,
                        title=str((ent.get("metadata", {}) or {}).get("title", "")).strip(),
                        source_type=str(ent.get("source", "")).strip() or "reference",
                    )
                    continue
                graph_store.upsert_profile_entity(
                    profile_id=profile_id,
                    entity_type=etype,
                    value=value,
                    relation=relation,
                    confidence=float(ent.get("confidence", 0.8) or 0.8),
                    source=str(ent.get("source", "")).strip() or "profile_enrichment",
                    metadata=dict(ent.get("metadata", {}) or {}),
                )
            return graph_store.get_profile_graph(profile_id=profile_id, limit=180)

        # Fallback local graph payload when Neo4j is disabled/unavailable.
        node_map: dict[str, dict[str, Any]] = {
            profile_id: {"id": profile_id, "label": "Profile", "value": str(concept.get("topic", "")).strip() or concept_id}
        }
        edges: list[dict[str, Any]] = []
        for ent in entities:
            value = str(ent.get("value", "")).strip()
            if not value:
                continue
            node_id = f"{str(ent.get('entity_type', 'entity')).strip().lower()}:{value.lower()}"
            if node_id not in node_map:
                node_map[node_id] = {
                    "id": node_id,
                    "label": str(ent.get("entity_type", "Entity")).strip().title(),
                    "value": value,
                    "source_type": str(ent.get("source", "")).strip(),
                }
            edges.append(
                {
                    "from": profile_id,
                    "to": node_id,
                    "relation": str(ent.get("relation", "")).strip(),
                    "confidence": float(ent.get("confidence", 0.0) or 0.0),
                    "source": str(ent.get("source", "")).strip(),
                }
            )
        return {
            "enabled": False,
            "profile_id": profile_id,
            "display_name": str(concept.get("topic", "")).strip(),
            "nodes": list(node_map.values()),
            "edges": edges,
            "reason": "neo4j_not_enabled",
        }

    async def _handle_world_concept_profile_graph(self, request: "web.Request") -> "web.Response":
        concept_id = str(request.match_info.get("concept_id", "")).strip()
        if not concept_id:
            return self._bad_request(request, "concept_id is required")
        try:
            concept = self.world_knowledge.get_concept(concept_id=concept_id)
        except KeyError:
            return self._error_response(request, "Concept not found", status=404)
        graph = self._sync_profile_graph(concept)
        return self._ok_response(request, {"concept_id": concept_id, "graph": graph})

    async def _handle_world_concepts_list(self, request: "web.Request") -> "web.Response":
        limit_raw = request.query.get("limit", "100")
        try:
            limit = max(1, min(500, int(limit_raw)))
        except Exception:
            limit = 100
        rows = self.world_knowledge.list_concepts(limit=limit)
        return self._ok_response(request, {"count": len(rows), "items": rows})

    async def _handle_world_concept_get(self, request: "web.Request") -> "web.Response":
        concept_id = str(request.match_info.get("concept_id", "")).strip()
        if not concept_id:
            return self._bad_request(request, "concept_id is required")
        try:
            row = self.world_knowledge.get_concept(concept_id=concept_id)
        except KeyError:
            return self._error_response(request, "Concept not found", status=404)
        return self._ok_response(request, {"concept": row})

    async def _handle_world_concept_update(self, request: "web.Request") -> "web.Response":
        concept_id = str(request.match_info.get("concept_id", "")).strip()
        if not concept_id:
            return self._bad_request(request, "concept_id is required")
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        topic_raw = body.get("topic")
        topic = str(topic_raw).strip() if topic_raw is not None else None
        notes_raw = body.get("notes")
        notes = str(notes_raw).strip() if notes_raw is not None else None
        tags: list[str] | None = None
        if "tags" in body:
            tags_raw = body.get("tags", [])
            if isinstance(tags_raw, list):
                tags = [str(t).strip() for t in tags_raw if str(t).strip()]
            elif isinstance(tags_raw, str):
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            else:
                return self._bad_request(request, "'tags' must be an array or csv string")
        detections: list[dict[str, Any]] | None = None
        if "detections" in body:
            raw = body.get("detections", [])
            if not isinstance(raw, list):
                return self._bad_request(request, "'detections' must be an array")
            detections = [d for d in raw if isinstance(d, dict)]
        metadata: dict[str, Any] | None = None
        if "metadata" in body:
            raw = body.get("metadata", {})
            if not isinstance(raw, dict):
                return self._bad_request(request, "'metadata' must be an object")
            metadata = raw
        try:
            concept = self.world_knowledge.update_concept(
                concept_id=concept_id,
                topic=topic,
                notes=notes,
                tags=tags,
                detections=detections,
                metadata=metadata,
            )
        except KeyError:
            return self._error_response(request, "Concept not found", status=404)
        except ValueError as exc:
            return self._bad_request(request, str(exc))
        graph = self._sync_profile_graph(concept)
        return self._ok_response(request, {"concept": concept, "profile_graph": graph})

    async def _handle_world_concept_delete(self, request: "web.Request") -> "web.Response":
        concept_id = str(request.match_info.get("concept_id", "")).strip()
        if not concept_id:
            return self._bad_request(request, "concept_id is required")
        deleted = self.world_knowledge.delete_concept(concept_id=concept_id)
        if not deleted:
            return self._error_response(request, "Concept not found", status=404)
        return self._ok_response(request, {"deleted": True, "concept_id": concept_id})

    async def _handle_world_teach(self, request: "web.Request") -> "web.Response":
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        topic = str(body.get("topic", "")).strip()
        if not topic:
            return self._bad_request(request, "'topic' is required")
        notes = str(body.get("notes", "")).strip()
        tags_raw = body.get("tags", [])
        tags: list[str] = []
        if isinstance(tags_raw, list):
            tags = [str(t).strip() for t in tags_raw if str(t).strip()]
        elif isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        detections = body.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        metadata = body.get("metadata", {})
        if not isinstance(metadata, dict):
            return self._bad_request(request, "'metadata' must be an object")
        try:
            concept = self.world_knowledge.teach_concept(
                topic=topic,
                notes=notes,
                tags=tags,
                detections=detections,
                metadata=metadata,
            )
        except ValueError as exc:
            return self._bad_request(request, str(exc))
        enrich = bool(body.get("enrich_web", True))
        enrich_result: dict[str, Any] = {}
        if enrich:
            try:
                enrich_result = self.world_knowledge.enrich_concept_from_web(
                    concept_id=str(concept.get("concept_id", "")),
                    research_engine=self.research_engine,
                    max_items=max(1, min(8, int(body.get("max_items", 4) or 4))),
                    run_adapters=True,
                )
                concept = dict(enrich_result.get("concept", concept))
            except Exception as exc:  # noqa: BLE001
                enrich_result = {"error": str(exc)}
        graph = self._sync_profile_graph(concept)
        return self._ok_response(
            request,
            {
                "concept": concept,
                "profile_graph": graph,
                "enrich_web": enrich,
                "enrich_result": enrich_result,
            },
            status=201,
        )

    async def _handle_world_concept_enrich(self, request: "web.Request") -> "web.Response":
        concept_id = str(request.match_info.get("concept_id", "")).strip()
        if not concept_id:
            return self._bad_request(request, "concept_id is required")
        body = await self._parse_json(request)
        if body is None:
            body = {}
        target_source = str(body.get("target_source", "web")).strip().lower() or "web"
        query = str(body.get("query", "")).strip()
        url = str(body.get("url", "")).strip()
        user_id = str(body.get("user_id", "api_user")).strip() or "api_user"
        max_items_raw = body.get("max_items", 5)
        try:
            max_items = max(1, min(12, int(max_items_raw)))
        except Exception:
            max_items = 5
        run_adapters = bool(body.get("run_adapters", True))
        if target_source in {"linkedin", "github", "google"}:
            concept = {}
            try:
                concept = self.world_knowledge.get_concept(concept_id=concept_id)
            except KeyError:
                return self._error_response(request, "Concept not found", status=404)
            if not query:
                query = str(concept.get("topic", "")).strip()
            if not query and target_source != "linkedin":
                return self._bad_request(request, "'query' is required for selected target_source")
            if self.connector_registry is None:
                return self._error_response(request, "Connector registry unavailable", status=503)
            try:
                if target_source == "linkedin":
                    payload = await self.connector_registry.invoke(
                        "linkedin_mcp",
                        "enrich_profile",
                        {
                            "query": query,
                            "user_id": user_id,
                            "profile_url": url,
                        },
                        actor_scopes={"connector:linkedin:read"},
                    )
                    out = self._build_world_result_from_linkedin_mcp(
                        concept_id=concept_id,
                        topic=query or str(concept.get("topic", "")).strip(),
                        payload=dict(payload or {}),
                    )
                    return self._ok_response(request, out)
                if target_source == "github":
                    payload = await self.connector_registry.invoke(
                        "github_mcp",
                        "enrich_profile",
                        {
                            "query": query,
                            "user_id": user_id,
                            "url": url,
                        },
                        actor_scopes={"connector:github:read"},
                    )
                    out = self._build_world_result_from_github_mcp(
                        concept_id=concept_id,
                        topic=query,
                        payload=dict(payload or {}),
                    )
                    return self._ok_response(request, out)
                payload = await self.connector_registry.invoke(
                    "google_search_mcp",
                    "enrich_profile",
                    {
                        "query": query,
                        "user_id": user_id,
                        "max_results": max(1, min(10, int(max_items))),
                    },
                    actor_scopes={"connector:google:read"},
                )
                out = self._build_world_result_from_google_mcp(
                    concept_id=concept_id,
                    topic=query,
                    payload=dict(payload or {}),
                )
                return self._ok_response(request, out)
            except Exception as exc:  # noqa: BLE001
                return self._error_response(request, f"MCP enrich failed: {exc}", status=500)
        try:
            out = self.world_knowledge.enrich_concept_from_web(
                concept_id=concept_id,
                research_engine=self.research_engine,
                max_items=max_items,
                run_adapters=run_adapters,
            )
        except KeyError:
            return self._error_response(request, "Concept not found", status=404)
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, f"World enrich failed: {exc}", status=500)
        concept = dict(out.get("concept", {}) or {})
        graph = self._sync_profile_graph(concept) if concept else {}
        out["profile_graph"] = graph
        return self._ok_response(request, out)

    async def _handle_world_concept_link_add(self, request: "web.Request") -> "web.Response":
        concept_id = str(request.match_info.get("concept_id", "")).strip()
        if not concept_id:
            return self._bad_request(request, "concept_id is required")
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        url = str(body.get("url", "")).strip()
        if not url:
            return self._bad_request(request, "'url' is required")
        tags_raw = body.get("tags", [])
        tags: list[str] = []
        if isinstance(tags_raw, list):
            tags = [str(t).strip() for t in tags_raw if str(t).strip()]
        elif isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        interaction = body.get("interaction")
        if interaction is not None and not isinstance(interaction, dict):
            return self._bad_request(request, "'interaction' must be an object")
        try:
            concept = self.world_knowledge.add_reference_link(
                concept_id=concept_id,
                url=url,
                title=str(body.get("title", "")).strip(),
                notes=str(body.get("notes", "")).strip(),
                source_type=str(body.get("source_type", "manual")).strip() or "manual",
                tags=tags,
                interaction=interaction if isinstance(interaction, dict) else None,
            )
        except KeyError:
            return self._error_response(request, "Concept not found", status=404)
        except ValueError as exc:
            return self._bad_request(request, str(exc))
        graph = self._sync_profile_graph(concept)
        return self._ok_response(request, {"concept": concept, "profile_graph": graph}, status=201)

    async def _handle_world_concept_link_delete(self, request: "web.Request") -> "web.Response":
        concept_id = str(request.match_info.get("concept_id", "")).strip()
        link_id = str(request.match_info.get("link_id", "")).strip()
        if not concept_id or not link_id:
            return self._bad_request(request, "concept_id and link_id are required")
        try:
            deleted = self.world_knowledge.remove_reference_link(concept_id=concept_id, link_id=link_id)
        except KeyError:
            return self._error_response(request, "Concept not found", status=404)
        if not deleted:
            return self._error_response(request, "Link not found", status=404)
        return self._ok_response(request, {"deleted": True, "concept_id": concept_id, "link_id": link_id})

    async def _handle_world_concept_link_interaction(self, request: "web.Request") -> "web.Response":
        concept_id = str(request.match_info.get("concept_id", "")).strip()
        link_id = str(request.match_info.get("link_id", "")).strip()
        if not concept_id or not link_id:
            return self._bad_request(request, "concept_id and link_id are required")
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        summary = str(body.get("summary", "")).strip()
        if not summary:
            return self._bad_request(request, "'summary' is required")
        facts_raw = body.get("extracted_facts", [])
        facts: list[str] = []
        if isinstance(facts_raw, list):
            facts = [str(x).strip() for x in facts_raw if str(x).strip()]
        elif isinstance(facts_raw, str):
            facts = [x.strip() for x in facts_raw.split("\n") if x.strip()]
        try:
            concept = self.world_knowledge.log_link_interaction(
                concept_id=concept_id,
                link_id=link_id,
                summary=summary,
                extracted_facts=facts,
                pattern_hint=str(body.get("pattern_hint", "")).strip(),
                outcome=str(body.get("outcome", "")).strip(),
            )
        except KeyError:
            return self._error_response(request, "Concept not found", status=404)
        except ValueError as exc:
            return self._bad_request(request, str(exc))
        graph = self._sync_profile_graph(concept)
        return self._ok_response(request, {"concept": concept, "profile_graph": graph}, status=201)

    async def _handle_world_concept_link_browser_use_run(self, request: "web.Request") -> "web.Response":
        concept_id = str(request.match_info.get("concept_id", "")).strip()
        link_id = str(request.match_info.get("link_id", "")).strip()
        if not concept_id or not link_id:
            return self._bad_request(request, "concept_id and link_id are required")
        body = await self._parse_json(request)
        if body is None:
            body = {}
        max_items_raw = body.get("max_items", 6)
        try:
            max_items = max(1, min(12, int(max_items_raw)))
        except Exception:
            max_items = 6
        run_adapters = bool(body.get("run_adapters", True))
        try:
            out = self.world_knowledge.run_link_learning(
                concept_id=concept_id,
                link_id=link_id,
                research_engine=self.research_engine,
                max_items=max_items,
                run_adapters=run_adapters,
            )
        except KeyError as exc:
            msg = str(exc)
            if "link" in msg.lower():
                return self._error_response(request, "Link not found", status=404)
            return self._error_response(request, "Concept not found", status=404)
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, f"Browser-use learning run failed: {exc}", status=500)
        concept = dict(out.get("concept", {}) or {})
        graph = self._sync_profile_graph(concept) if concept else {}
        out["profile_graph"] = graph
        return self._ok_response(request, out)

    async def _handle_world_detections_enrich(self, request: "web.Request") -> "web.Response":
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        labels_raw = body.get("labels", [])
        labels: list[str] = []
        if isinstance(labels_raw, list):
            labels = [str(x).strip() for x in labels_raw if str(x).strip()]
        elif isinstance(labels_raw, str):
            labels = [x.strip() for x in labels_raw.split(",") if x.strip()]
        detections = body.get("detections", [])
        if isinstance(detections, list):
            for d in detections:
                if not isinstance(d, dict):
                    continue
                lbl = str(d.get("label", "")).strip()
                if lbl:
                    labels.append(lbl)
        if not labels:
            return self._bad_request(request, "At least one label or detection is required")
        allow_web = bool(body.get("allow_web", True))
        max_items_raw = body.get("max_items_per_label", 2)
        try:
            max_items = max(1, min(5, int(max_items_raw)))
        except Exception:
            max_items = 2
        context_rows = self.world_knowledge.enrich_detection_labels(
            labels=labels,
            research_engine=self.research_engine,
            max_items_per_label=max_items,
            allow_web=allow_web,
        )
        return self._ok_response(request, {"count": len(context_rows), "items": context_rows})

    async def _handle_realtime_interrupt(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/realtime/sessions/{session_id}/interrupt — cancel stale in-flight turn."""
        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return self._bad_request(request, "session_id is required")
        body = await self._parse_json(request)
        reason = ""
        if isinstance(body, dict):
            reason = str(body.get("reason", "")).strip()
        cm = self.conversation_manager
        if cm is None or not hasattr(cm, "interrupt_realtime_session"):
            return self._error_response(request, "Realtime interrupt support is unavailable", status=501)
        snap = cm.interrupt_realtime_session(session_id, reason=reason)
        return self._ok_response(request, {"session_id": session_id, "realtime": snap})

    async def _handle_realtime_turn(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/realtime/sessions/{session_id}/turn — low-latency multimodal turn."""
        perf_started = time.time()
        perf_stages: dict[str, float] = {}
        allowed, reject_response, queue_wait_ms = await self._ingress_acquire_or_response(
            request, route_label="realtime_turn"
        )
        perf_stages["ingress_wait"] = round(float(queue_wait_ms or 0.0), 2)
        if not allowed:
            perf_stages["total"] = round((time.time() - perf_started) * 1000.0, 2)
            self._log_perf_breakdown(
                request=request,
                route="realtime_turn",
                outcome="ingress_reject",
                stages_ms=perf_stages,
            )
            return reject_response or self._error_response(request, "Ingress rejected", status=429)
        try:
            session_id = str(request.match_info.get("session_id", "")).strip()
            if not session_id:
                perf_stages["total"] = round((time.time() - perf_started) * 1000.0, 2)
                self._log_perf_breakdown(
                    request=request,
                    route="realtime_turn",
                    outcome="bad_request_missing_session",
                    stages_ms=perf_stages,
                )
                return self._bad_request(request, "session_id is required")
            parse_started = time.time()
            body = await self._parse_json(request)
            perf_stages["parse_json"] = round((time.time() - parse_started) * 1000.0, 2)
            if body is None:
                perf_stages["total"] = round((time.time() - perf_started) * 1000.0, 2)
                self._log_perf_breakdown(
                    request=request,
                    route="realtime_turn",
                    outcome="bad_request_invalid_json",
                    stages_ms=perf_stages,
                    extra={"session_id": session_id},
                )
                return self._bad_request(request, "Invalid JSON body")
            query = str(body.get("text", "")).strip() or str(body.get("query", "")).strip()
            if not query:
                perf_stages["total"] = round((time.time() - perf_started) * 1000.0, 2)
                self._log_perf_breakdown(
                    request=request,
                    route="realtime_turn",
                    outcome="bad_request_missing_text",
                    stages_ms=perf_stages,
                    extra={"session_id": session_id},
                )
                return self._bad_request(request, "'text' (or 'query') is required")
            modality = str(body.get("modality", "voice")).strip() or "voice"
            media = body.get("media", {})
            context = body.get("context", {})
            if not isinstance(media, dict):
                return self._bad_request(request, "'media' must be an object")
            if not isinstance(context, dict):
                return self._bad_request(request, "'context' must be an object")
            context = dict(context)
            context["realtime_mode"] = True
            status_reply = self._maybe_enrichment_status_reply(session_id=session_id, query=query)
            if status_reply:
                return self._ok_response(
                    request,
                    {
                        "session_id": session_id,
                        "response": str(status_reply.get("response", "")).strip(),
                        "background_job": status_reply.get("job", {}),
                    },
                )
            queued = self._maybe_queue_enrichment_from_query(
                session_id=session_id,
                user_id=str(body.get("user_id", "api_user")).strip() or "api_user",
                query=query,
                context=context,
            )
            if queued:
                ack = str(queued.get("ack", "")).strip()
                return self._ok_response(
                    request,
                    {
                        "session_id": session_id,
                        "response": ack,
                        "background_job": queued.get("job", {}),
                    },
                    status=202,
                )
            quick_social = self._fast_social_scene_reply(query=query, modality=modality, context=context)
            if quick_social:
                governed = apply_response_governance(
                    quick_social,
                    route="chat",
                    hints={"user_input": query},
                )
                perf_stages["total"] = round((time.time() - perf_started) * 1000.0, 2)
                self._log_perf_breakdown(
                    request=request,
                    route="realtime_turn",
                    outcome="ok_social_fastpath",
                    stages_ms=perf_stages,
                    extra={"session_id": session_id, "modality": modality},
                )
                return self._ok_response(
                    request,
                    {"session_id": session_id, "response": governed.text, "response_governance": governed.to_dict()},
                )
            cm = self.conversation_manager
            if cm is None or not hasattr(cm, "process_input"):
                perf_stages["total"] = round((time.time() - perf_started) * 1000.0, 2)
                self._log_perf_breakdown(
                    request=request,
                    route="realtime_turn",
                    outcome="cm_unavailable",
                    stages_ms=perf_stages,
                    extra={"session_id": session_id},
                )
                return self._error_response(request, "Conversation manager unavailable", status=503)
            if hasattr(cm, "start_realtime_session"):
                user_id = str(body.get("user_id", "api_user")).strip() or "api_user"
                cm.start_realtime_session(user_id=user_id, session_id=session_id)
            conversation_started = time.time()
            try:
                response_text = await asyncio.wait_for(
                    self._call_conversation_manager(
                        session_id=session_id,
                        query=query,
                        modality=modality,
                        media=media,
                        context=context,
                    ),
                    timeout=self._realtime_turn_timeout_s,
                )
            except asyncio.TimeoutError:
                if hasattr(cm, "interrupt_realtime_session"):
                    with contextlib.suppress(Exception):
                        cm.interrupt_realtime_session(session_id, reason="turn_timeout")
                return self._ok_response(
                    request,
                    {
                        "session_id": session_id,
                        "response": "Request timed out. I cleared the previous turn. Please continue speaking.",
                        "timeout": True,
                    },
                    status=504,
                )
            perf_stages["conversation"] = round((time.time() - conversation_started) * 1000.0, 2)
            governance_started = time.time()
            governed = apply_response_governance(
                str(response_text),
                route="chat",
                hints={"user_input": query},
            )
            notifications = self._drain_session_notifications(session_id, limit=6, drain=True)
            perf_stages["governance"] = round((time.time() - governance_started) * 1000.0, 2)
            perf_stages["total"] = round((time.time() - perf_started) * 1000.0, 2)
            self._log_perf_breakdown(
                request=request,
                route="realtime_turn",
                outcome="ok",
                stages_ms=perf_stages,
                extra={"session_id": session_id, "modality": modality},
            )
            return self._ok_response(
                request,
                {
                    "session_id": session_id,
                    "response": self._prepend_notifications(base_response=governed.text, notifications=notifications),
                    "response_governance": governed.to_dict(),
                },
            )
        finally:
            await self._ingress_release()

    async def _handle_realtime_ws(self, request: "web.Request") -> "web.StreamResponse":
        """GET /api/v1/realtime/sessions/{session_id}/ws — duplex realtime channel."""
        if not _AIOHTTP_AVAILABLE:
            return self._error_response(request, "WebSocket support unavailable", status=501)
        session_id = str(request.match_info.get("session_id", "")).strip()
        if not session_id:
            return self._bad_request(request, "session_id is required")
        cm = self.conversation_manager
        if cm is None or not hasattr(cm, "process_input"):
            return self._error_response(request, "Conversation manager unavailable", status=503)

        ws = web.WebSocketResponse(heartbeat=25)
        await ws.prepare(request)
        listeners = self._realtime_ws_subscribers.get(session_id, set())
        listeners.add(ws)
        self._realtime_ws_subscribers[session_id] = listeners

        async def _send(payload: dict[str, Any]) -> None:
            try:
                await ws.send_json(payload)
            except Exception:
                pass

        user_id = str(request.query.get("user_id", "api_user")).strip() or "api_user"
        if hasattr(cm, "start_realtime_session"):
            cm.start_realtime_session(user_id=user_id, session_id=session_id)

        await _send({"type": "ready", "session_id": session_id, "transport": "ws"})

        async def _handle_turn(cmd: dict[str, Any]) -> None:
            query = str(cmd.get("text", "")).strip() or str(cmd.get("query", "")).strip()
            if not query:
                await _send({"type": "error", "error": "'text' (or 'query') is required", "code": "bad_request"})
                return
            modality = str(cmd.get("modality", "voice")).strip() or "voice"
            media = cmd.get("media", {})
            context = cmd.get("context", {})
            if not isinstance(media, dict) or not isinstance(context, dict):
                await _send({"type": "error", "error": "'media' and 'context' must be objects", "code": "bad_request"})
                return
            context = dict(context)
            context["realtime_mode"] = True
            context.setdefault("response_shape", "sectioned")
            context.setdefault("latency_target", "low")
            corr_id = str(cmd.get("id", "")).strip() or str(uuid.uuid4())
            status_reply = self._maybe_enrichment_status_reply(session_id=session_id, query=query)
            if status_reply:
                await _send(
                    {
                        "type": "response",
                        "id": corr_id,
                        "session_id": session_id,
                        "response": str(status_reply.get("response", "")).strip(),
                        "background_job": status_reply.get("job", {}),
                    }
                )
                return
            queued = self._maybe_queue_enrichment_from_query(
                session_id=session_id,
                user_id=user_id,
                query=query,
                context=context,
            )
            if queued:
                await _send(
                    {
                        "type": "response",
                        "id": corr_id,
                        "session_id": session_id,
                        "response": str(queued.get("ack", "")).strip(),
                        "background_job": queued.get("job", {}),
                    }
                )
                return
            quick_social = self._fast_social_scene_reply(query=query, modality=modality, context=context)
            if quick_social:
                governed = apply_response_governance(
                    quick_social,
                    route="chat",
                    hints={"user_input": query},
                )
                await _send(
                    {
                        "type": "response",
                        "id": corr_id,
                        "session_id": session_id,
                        "response": governed.text,
                        "response_governance": governed.to_dict(),
                    }
                )
                return
            stream_hints = self._infer_streaming_hints_from_query(query)

            allowed, reject_response, _ = await self._ingress_acquire_or_response(
                request,
                route_label="realtime_ws_turn",
            )
            if not allowed:
                msg = "Ingress rejected"
                if reject_response is not None:
                    try:
                        payload = json.loads(reject_response.text)
                        msg = str(payload.get("error") or msg)
                    except Exception:
                        msg = "Ingress rejected"
                await _send({"type": "error", "id": corr_id, "error": msg, "code": "rate_limited"})
                return
            try:
                turn_started = time.time()
                await _send(
                    {
                        "type": "progress",
                        "id": corr_id,
                        "session_id": session_id,
                        "stage": "accepted",
                        "message": "Request accepted. Preparing context.",
                        "elapsed_ms": 0,
                    }
                )
                progress_stop = asyncio.Event()

                async def _progress_pulse() -> None:
                    while not progress_stop.is_set():
                        await asyncio.sleep(1.5)
                        if progress_stop.is_set():
                            break
                        elapsed_ms = int((time.time() - turn_started) * 1000)
                        await _send(
                            {
                                "type": "progress",
                                "id": corr_id,
                                "session_id": session_id,
                                "stage": "generating",
                                "message": "Generating response...",
                                "elapsed_ms": elapsed_ms,
                            }
                        )

                pulse_task = asyncio.create_task(_progress_pulse())
                try:
                    await _send(
                        {
                            "type": "progress",
                            "id": corr_id,
                            "session_id": session_id,
                            "stage": "routing",
                            "message": "Routing to model and executing turn.",
                            "elapsed_ms": int((time.time() - turn_started) * 1000),
                        }
                    )
                    response_text = await asyncio.wait_for(
                        self._call_conversation_manager(
                            session_id=session_id,
                            query=query,
                            modality=modality,
                            media=media,
                            context=context,
                        ),
                        timeout=self._realtime_turn_timeout_s,
                    )
                finally:
                    progress_stop.set()
                    with contextlib.suppress(Exception):
                        await pulse_task
                await _send(
                    {
                        "type": "progress",
                        "id": corr_id,
                        "session_id": session_id,
                        "stage": "governance",
                        "message": "Applying response policy.",
                        "elapsed_ms": int((time.time() - turn_started) * 1000),
                    }
                )
                governed = apply_response_governance(
                    str(response_text),
                    route="chat",
                    hints={"user_input": query},
                )
                notifications = self._drain_session_notifications(session_id, limit=6, drain=True)
                response_final = self._prepend_notifications(base_response=governed.text, notifications=notifications)
                sectioned = bool(cmd.get("sectioned", stream_hints.get("sectioned", True)))
                if sectioned:
                    max_sections_raw = cmd.get("max_sections", 6)
                    if "max_sections" not in cmd and isinstance(stream_hints.get("max_sections"), int):
                        max_sections_raw = stream_hints["max_sections"]
                    try:
                        max_sections = max(1, min(16, int(max_sections_raw)))
                    except Exception:
                        max_sections = 6
                    sections = self._split_response_sections(governed.text, max_sections=max_sections)
                    await _send(
                        {
                            "type": "response_started",
                            "id": corr_id,
                            "session_id": session_id,
                            "section_count": len(sections),
                        }
                    )
                    for idx, sec in enumerate(sections):
                        await _send(
                            {
                                "type": "response_section",
                                "id": corr_id,
                                "session_id": session_id,
                                "section_index": idx,
                                "section_count": len(sections),
                                "section_text": sec,
                            }
                        )
                    await _send(
                        {
                            "type": "response_done",
                            "id": corr_id,
                            "session_id": session_id,
                            "response": response_final,
                            "response_governance": governed.to_dict(),
                        }
                    )
                else:
                    await _send(
                        {
                            "type": "response",
                            "id": corr_id,
                            "session_id": session_id,
                            "response": response_final,
                            "response_governance": governed.to_dict(),
                        }
                    )
            except asyncio.TimeoutError:
                if hasattr(cm, "interrupt_realtime_session"):
                    with contextlib.suppress(Exception):
                        cm.interrupt_realtime_session(session_id, reason="turn_timeout")
                await _send(
                    {
                        "type": "error",
                        "id": corr_id,
                        "error": "Turn timed out. Cleared stale turn; you can continue speaking.",
                        "code": "turn_timeout",
                    }
                )
            except Exception as exc:
                await _send({"type": "error", "id": corr_id, "error": str(exc), "code": "turn_failed"})
            finally:
                await self._ingress_release()

        async def _handle_media(cmd: dict[str, Any]) -> None:
            source = str(cmd.get("source", "screen")).strip() or "screen"
            summary = str(cmd.get("summary", "")).strip()
            image_url = (
                str(cmd.get("image_url", "")).strip()
                or str(cmd.get("frame_url", "")).strip()
                or str(cmd.get("snapshot_url", "")).strip()
            )
            image_b64 = str(cmd.get("image_b64", "")).strip()
            metadata = cmd.get("metadata", {})
            if not isinstance(metadata, dict):
                await _send({"type": "error", "error": "'metadata' must be an object", "code": "bad_request"})
                return
            note = str(cmd.get("note", "")).strip()
            ts_raw = cmd.get("ts")
            ts_val = None
            if ts_raw is not None:
                try:
                    ts_val = float(ts_raw)
                except Exception:
                    ts_val = None
            queued_visual_summary = False
            if not summary and (image_url or image_b64) and hasattr(cm, "summarize_visual_observation"):
                if self._realtime_async_visual_summary:
                    queued_visual_summary = self._queue_realtime_visual_summary(
                        session_id=session_id,
                        source=source,
                        image_url=image_url,
                        image_b64=image_b64,
                        note=note,
                        metadata=metadata,
                        ts_val=ts_val,
                    )
                    summary = note or f"Live {source} frame queued for async vision summary."
                else:
                    try:
                        summary = await asyncio.wait_for(
                            cm.summarize_visual_observation(
                                session_id,
                                source=source,
                                image_url=image_url,
                                image_b64=image_b64,
                                note=note,
                                metadata=metadata,
                            ),
                            timeout=self._realtime_turn_timeout_s,
                        )
                    except Exception:
                        summary = note
            if not summary:
                await _send(
                    {
                        "type": "error",
                        "error": "Either 'summary' or one of ('image_url','frame_url','snapshot_url','image_b64') is required",
                        "code": "bad_request",
                    }
                )
                return
            if image_url:
                metadata = dict(metadata)
                metadata["image_url"] = image_url
            if image_b64:
                metadata = dict(metadata)
                metadata["image_b64_size"] = len(image_b64)
            snap = cm.ingest_realtime_frame(
                session_id,
                source=source,
                summary=summary,
                metadata=metadata,
                ts=ts_val,
            )
            await _send(
                {
                    "type": "media_ack",
                    "session_id": session_id,
                    "realtime": snap,
                    "visual_summary_queued": queued_visual_summary,
                }
            )

        async def _handle_interrupt(cmd: dict[str, Any]) -> None:
            reason = str(cmd.get("reason", "")).strip()
            snap = cm.interrupt_realtime_session(session_id, reason=reason)
            await _send({"type": "interrupt_ack", "session_id": session_id, "realtime": snap})

        async def _handle_stream_start(cmd: dict[str, Any]) -> None:
            source_type = str(cmd.get("source_type", "http")).strip().lower() or "http"
            source_url = str(cmd.get("source_url", "")).strip()
            if not source_url:
                await _send({"type": "error", "error": "'source_url' is required", "code": "bad_request"})
                return
            interval_ms_raw = cmd.get("interval_ms", self._stream_default_interval_ms)
            try:
                interval_ms = max(500, min(10000, int(interval_ms_raw)))
            except Exception:
                interval_ms = self._stream_default_interval_ms
            note = str(cmd.get("note", "")).strip()
            metadata = cmd.get("metadata", {})
            if not isinstance(metadata, dict):
                await _send({"type": "error", "error": "'metadata' must be an object", "code": "bad_request"})
                return
            stream = await self.live_stream_ingest.start_stream(
                session_id=session_id,
                source_type=source_type,
                source_url=source_url,
                interval_ms=interval_ms,
                note=note,
                metadata=metadata,
            )
            await _send({"type": "stream_started", "session_id": session_id, "stream": stream})

        async def _handle_stream_stop(cmd: dict[str, Any]) -> None:
            stream_id = str(cmd.get("stream_id", "")).strip()
            if not stream_id:
                await _send({"type": "error", "error": "'stream_id' is required", "code": "bad_request"})
                return
            snap = await self.live_stream_ingest.stop_stream(stream_id)
            if snap is None:
                await _send({"type": "error", "error": "Stream not found", "code": "not_found"})
                return
            await _send({"type": "stream_stopped", "session_id": session_id, "stream": snap})

        async def _handle_audio_chunk(cmd: dict[str, Any]) -> None:
            pcm16_b64 = str(cmd.get("pcm16_b64", "")).strip() or str(cmd.get("pcm_b64", "")).strip()
            if not pcm16_b64:
                await _send({"type": "error", "error": "'pcm16_b64' is required", "code": "bad_request"})
                return
            sample_rate_raw = cmd.get("sample_rate", 16000)
            try:
                sample_rate = max(8000, min(96000, int(sample_rate_raw)))
            except Exception:
                sample_rate = 16000
            try:
                snap = self.realtime_stt.ingest_pcm16_chunk(
                    session_id,
                    pcm16_b64=pcm16_b64,
                    sample_rate=sample_rate,
                )
            except Exception as exc:
                await _send({"type": "error", "error": str(exc), "code": "bad_audio"})
                return
            await _send({"type": "audio_ack", "session_id": session_id, **snap})

        async def _handle_audio_commit(cmd: dict[str, Any]) -> None:
            language = str(cmd.get("language", "en-IN")).strip() or "en-IN"
            auto_turn = bool(cmd.get("auto_turn", True))
            try:
                transcript = await self.realtime_stt.transcribe_and_reset_async(session_id, language=language)
            except Exception as exc:
                await _send({"type": "error", "error": str(exc), "code": "stt_failed"})
                return
            await _send(
                {
                    "type": "transcript",
                    "session_id": session_id,
                    "text": transcript,
                    "final": True,
                    "language": language,
                }
            )
            if auto_turn and transcript:
                turn_id = str(cmd.get("id", "")).strip() or str(uuid.uuid4())
                asyncio.create_task(
                    _handle_turn(
                        {
                            "id": turn_id,
                            "text": transcript,
                            "modality": "voice",
                            "context": {"realtime_mode": True, "source": "ws_audio_commit"},
                        }
                    )
                )
                await _send(
                    {
                        "type": "turn_queued",
                        "id": turn_id,
                        "session_id": session_id,
                        "message": "Voice turn queued for processing.",
                    }
                )

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    text = (msg.data or "").strip()
                    if not text:
                        continue
                    try:
                        cmd = json.loads(text)
                    except Exception:
                        await _send({"type": "error", "error": "Invalid JSON payload", "code": "bad_json"})
                        continue
                    if not isinstance(cmd, dict):
                        await _send({"type": "error", "error": "Payload must be a JSON object", "code": "bad_json"})
                        continue
                    cmd_type = str(cmd.get("type", "")).strip().lower()
                    if cmd_type in {"ping", "health"}:
                        await _send({"type": "pong", "session_id": session_id, "ts": time.time()})
                    elif cmd_type in {"turn", "query"}:
                        await _handle_turn(cmd)
                    elif cmd_type == "media":
                        await _handle_media(cmd)
                    elif cmd_type == "interrupt":
                        await _handle_interrupt(cmd)
                    elif cmd_type == "start_stream":
                        await _handle_stream_start(cmd)
                    elif cmd_type == "stop_stream":
                        await _handle_stream_stop(cmd)
                    elif cmd_type == "audio_chunk":
                        await _handle_audio_chunk(cmd)
                    elif cmd_type in {"audio_commit", "audio_end"}:
                        await _handle_audio_commit(cmd)
                    else:
                        await _send(
                            {
                                "type": "error",
                                "error": f"Unsupported command type: {cmd_type or '<empty>'}",
                                "code": "unsupported",
                            }
                        )
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.warning("Realtime WS closed with exception: %s", ws.exception())
                    break
        finally:
            listeners = self._realtime_ws_subscribers.get(session_id, set())
            listeners.discard(ws)
            if listeners:
                self._realtime_ws_subscribers[session_id] = listeners
            else:
                self._realtime_ws_subscribers.pop(session_id, None)
            with contextlib.suppress(Exception):
                await ws.close()
        return ws

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
        allowed, reject_response, _ = await self._ingress_acquire_or_response(
            request, route_label="chat_completions"
        )
        if not allowed:
            return reject_response or self._error_response(request, "Ingress rejected", status=429)
        try:
            return await self._handle_openai_chat_completions_impl(request)
        finally:
            await self._ingress_release()

    async def _handle_openai_chat_completions_impl(self, request: "web.Request") -> "web.Response":
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
        body_session_id = str(body.get("session_id", "")).strip()
        workspace_hint = self._resolve_workspace_hint(
            request=request,
            body=body,
            user_id=user_id,
            session_id=body_session_id,
        )
        envelope = RequestEnvelope.from_any(
            {
                "request_id": self._request_id(request),
                "user_id": user_id,
                "session_id": body_session_id,
                "model": model,
                "modality": "text",
                "workspace_path": workspace_hint,
                "route": "chat_completions",
            }
        )
        try:
            parsed_max_latency = int(body.get("max_latency_ms")) if body.get("max_latency_ms") is not None else None
        except Exception:
            parsed_max_latency = None
        try:
            parsed_budget = float(body.get("budget_usd")) if body.get("budget_usd") is not None else None
        except Exception:
            parsed_budget = None
        policy_decision = self.policy_cost_engine.decide(
            PolicyContext(
                route="chat_completions",
                task_type="general_query",
                user_id=user_id,
                sla_tier=str(body.get("sla_tier", request.headers.get("X-SLA-Tier", "standard"))),
                latency_sensitive=bool(body.get("latency_sensitive", False)),
                max_latency_ms=parsed_max_latency,
                budget_usd=parsed_budget,
                privacy_level=str(body.get("privacy_level", "medium")),
                metadata={"model": model},
            )
        )
        if not policy_decision.allow:
            if self.slo_metrics:
                self.slo_metrics.inc("policy_decision_total", label=policy_decision.reason or "deny")
                self.slo_metrics.inc("policy_deny_total", label=policy_decision.reason or "deny")
            return web.json_response(
                {"error": {"message": f"Request denied by policy: {policy_decision.reason}"}},
                status=429,
            )

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

        tool_req = self._parse_chat_tool_request(
            last_user=last_user,
            body=body,
            workspace_hint=workspace_hint,
        )
        contract_error = self._validate_chat_tool_request(tool_req)
        if contract_error:
            tool_req = {"type": "command_error", "error": contract_error}

        response_text = ""
        response_route = "chat"
        session_id = ""
        conversation_latency_ms = 0.0
        if tool_req and tool_req.get("type") == "command_error":
            response_text = str(tool_req.get("error", "Invalid command")).strip() or "Invalid command."
        elif tool_req and tool_req.get("type") == "code_assist":
            response_route = "code_assist"
            req_obj, req_err = CodeAssistRequest.from_any(tool_req)
            if req_err or req_obj is None:
                response_text = f"Invalid /code request: {req_err or 'invalid payload'}"
                conversation_latency_ms = 0.0
                tool_req = None
            else:
                conversation_started = time.time()
                code_result = await self._submit_code_assist_task(
                    instruction=req_obj.instruction,
                    workspace_path=req_obj.workspace_path,
                    dry_run=req_obj.dry_run,
                    run_checks=req_obj.run_checks,
                    wait=True,
                    timeout_s=float(tool_req.get("timeout_seconds", 240)),
                    priority=int(tool_req.get("priority", 6)),
                    max_files=req_obj.max_files,
                )
                conversation_latency_ms = round((time.time() - conversation_started) * 1000.0, 2)
                if code_result.get("ok"):
                    payload = code_result.get("data", {}) if isinstance(code_result.get("data"), dict) else {}
                    status = str(payload.get("status", "unknown"))
                    result = payload.get("result", {}) if isinstance(payload.get("result"), dict) else {}
                    touched = result.get("files_touched", []) if isinstance(result.get("files_touched", []), list) else []
                    applied = int(result.get("applied_count", 0) or 0)
                    response_text = (
                        f"Code assist {status}. Applied {applied} edit(s)"
                        + (f" across {len(touched)} file(s): {', '.join(touched[:8])}" if touched else ".")
                    )
                else:
                    response_text = str(code_result.get("error", "Code assist failed")).strip()
        elif tool_req and tool_req.get("type") == "repo_understand":
            response_route = "repo"
            req_obj, req_err = RepoUnderstandRequest.from_any(tool_req)
            if req_err or req_obj is None:
                response_text = f"Invalid /repo request: {req_err or 'invalid payload'}"
                conversation_latency_ms = 0.0
                tool_req = None
            else:
                conversation_started = time.time()
                repo_result = await self._submit_repo_understand_task(
                    workspace_path=req_obj.workspace_path,
                    question=req_obj.question,
                    wait=True,
                    timeout_s=float(tool_req.get("timeout_seconds", 240)),
                    priority=int(tool_req.get("priority", 5)),
                    max_files=req_obj.max_files,
                    include_tree=req_obj.include_tree,
                    depth=req_obj.depth,
                )
                conversation_latency_ms = round((time.time() - conversation_started) * 1000.0, 2)
                if repo_result.get("ok"):
                    payload = repo_result.get("data", {}) if isinstance(repo_result.get("data"), dict) else {}
                    status = str(payload.get("status", "unknown")).strip().lower()
                    task_error = str(payload.get("error", "")).strip()
                    result = payload.get("result", {}) if isinstance(payload.get("result"), dict) else {}
                    verified = VerifiedResponse.from_repo_result(result)
                    verified_text = verified.to_user_text()
                    if verified_text:
                        response_text = verified_text
                    elif task_error:
                        response_text = f"Repository analysis {status}: {task_error}"
                    elif status and status != "completed":
                        response_text = f"Repository analysis status: {status}. Try again in a few seconds."
                    else:
                        response_text = (
                            "Repository analysis completed, but no summary was returned. "
                            "Please retry with a more specific question."
                        )
                else:
                    response_text = str(repo_result.get("error", "Repository analysis failed")).strip()
        elif tool_req and tool_req.get("type") == "code_workflow":
            response_route = "code_workflow"
            req_obj, req_err = CodeWorkflowRequest.from_any(tool_req)
            if req_err or req_obj is None:
                response_text = f"Invalid /workflow request: {req_err or 'invalid payload'}"
                conversation_latency_ms = 0.0
                tool_req = None
            else:
                conversation_started = time.time()
                wf_result = await self._submit_code_multi_agent_workflow(
                    workspace_path=req_obj.workspace_path,
                    goal=req_obj.goal,
                    dry_run=req_obj.dry_run,
                    run_checks=req_obj.run_checks,
                    wait=True,
                    timeout_s=float(tool_req.get("timeout_seconds", 420)),
                    max_workers=req_obj.max_workers,
                )
                conversation_latency_ms = round((time.time() - conversation_started) * 1000.0, 2)
                if wf_result.get("ok"):
                    payload = wf_result.get("data", {}) if isinstance(wf_result.get("data"), dict) else {}
                    status = str(payload.get("status", "unknown")).lower()
                    plan_id = str(payload.get("plan_id", "")).strip()
                    steps = payload.get("steps", {}) if isinstance(payload.get("steps"), dict) else {}
                    completed = sum(1 for _, st in steps.items() if str(st) == "completed")
                    response_text = (
                        f"Workflow {status}. "
                        f"Completed {completed}/{len(steps)} step(s). "
                        + (f"Plan ID: {plan_id}." if plan_id else "")
                    ).strip()
                else:
                    response_text = str(wf_result.get("error", "Code workflow failed")).strip()
        elif self.conversation_manager:
            session_id = self.conversation_manager.get_or_create_session(user_id)
            if workspace_hint:
                self._workspace_by_user[user_id] = workspace_hint
                self._workspace_by_session[session_id] = workspace_hint
            conversation_started = time.time()
            chat_ctx: dict[str, Any] = {"request_envelope": envelope.to_dict()}
            chat_ctx["policy_decision"] = policy_decision.to_dict()
            if len(prompt_parts[:-1]) > 4:
                # Keep only a small rolling window to avoid prompt bloat for short turns.
                chat_ctx = {"chat_history": prompt_parts[-5:-1]}
                chat_ctx["request_envelope"] = envelope.to_dict()
                chat_ctx["policy_decision"] = policy_decision.to_dict()
            response_text = await self._call_conversation_manager(
                session_id=session_id,
                query=last_user,
                modality="text",
                media={},
                context=chat_ctx,
            )
            conversation_latency_ms = round((time.time() - conversation_started) * 1000.0, 2)
        else:
            response_text = f"Echo: {last_user}"
        governance_hints: dict[str, Any] = {"user_input": last_user}
        if self.conversation_manager and session_id and hasattr(self.conversation_manager, "get_context"):
            try:
                ctx = self.conversation_manager.get_context(session_id)
                if ctx is not None and isinstance(getattr(ctx, "metadata", None), dict):
                    md = ctx.metadata
                    if isinstance(md.get("response_plan"), dict):
                        governance_hints["response_plan"] = dict(md.get("response_plan", {}))
                    if isinstance(md.get("query_understanding"), dict):
                        governance_hints["query_understanding"] = dict(md.get("query_understanding", {}))
            except Exception:  # noqa: BLE001
                pass
        governed = apply_response_governance(
            str(response_text),
            route=response_route,
            hints=governance_hints,
        )
        response_text = governed.text

        ctx_latency: dict[str, Any] = {}
        route_latency: dict[str, Any] = {}
        summary_stage: dict[str, Any] = {}
        response_plan: dict[str, Any] = {}
        query_understanding: dict[str, Any] = {}
        route_failures_turn: list[dict[str, Any]] = []
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
                    if isinstance(md.get("response_plan"), dict):
                        response_plan = dict(md.get("response_plan", {}))
                    if isinstance(md.get("query_understanding"), dict):
                        query_understanding = dict(md.get("query_understanding", {}))
                    if isinstance(md.get("route_failures_turn"), list):
                        route_failures_turn = [
                            rf for rf in md.get("route_failures_turn", [])
                            if isinstance(rf, dict)
                        ]
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
        self._log_perf_breakdown(
            request=request,
            route="chat_completions",
            outcome="ok",
            stages_ms=stage_latency_ms,
            extra={
                "model": model,
                "session_id": session_id or "",
                "response_route": response_route,
            },
        )
        prompt_tokens = max(1, len(last_user.split()))
        completion_tokens = max(1, len(str(response_text).split()))
        total_tokens = prompt_tokens + completion_tokens
        # Rough local cost estimate for budget accounting.
        est_cost_usd = float(total_tokens) * 0.000002
        ledger_entry = self.policy_cost_engine.record_usage(
            user_id=user_id,
            cost_usd=est_cost_usd,
            tokens_total=total_tokens,
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
            self.slo_metrics.inc("chat_tokens_total", label=f"{model}:prompt", value=prompt_tokens)
            self.slo_metrics.inc("chat_tokens_total", label=f"{model}:completion", value=completion_tokens)
            self.slo_metrics.inc("chat_tokens_total", label=f"{model}:total", value=total_tokens)
            self.slo_metrics.inc("modality_usage_total", label="text")
            provider = str(route_latency.get("provider_name", "")).strip()
            if provider:
                self.slo_metrics.inc("model_route_total", label=provider)
            if summary_stage:
                source = str(summary_stage.get("source", "unknown"))
                self.slo_metrics.inc("summary_stage_total", label=source)
                reject_reason = str(summary_stage.get("reject_reason", "")).strip()
                if reject_reason:
                    self.slo_metrics.inc("summary_reject_total", label=reject_reason)
            if response_plan:
                task_type = str(response_plan.get("task_type", "unknown")).strip() or "unknown"
                target_length = str(response_plan.get("target_length", "unknown")).strip() or "unknown"
                handler_key = str(response_plan.get("handler_key", "none")).strip() or "none"
                self.slo_metrics.inc(
                    "planner_decision_total",
                    label=f"{task_type}:{target_length}:{handler_key}",
                )
                self.slo_metrics.inc("planner_target_length_total", label=target_length)
                self.slo_metrics.inc("planner_handler_total", label=handler_key)
                prefer_local = response_plan.get("prefer_local", None)
                prefer_label = "auto"
                if prefer_local is True:
                    prefer_label = "local"
                elif prefer_local is False:
                    prefer_label = "api"
                self.slo_metrics.inc("planner_preference_total", label=prefer_label)
                try:
                    complexity = float(response_plan.get("complexity", 0.0))
                except Exception:
                    complexity = 0.0
                self.slo_metrics.set_gauge("planner_complexity_last", complexity, label=task_type)
            if isinstance(policy_decision, PolicyDecision):
                decision_label = str(policy_decision.reason or "unknown")
                self.slo_metrics.inc("policy_decision_total", label=decision_label)
                if policy_decision.max_latency_ms is not None:
                    self.slo_metrics.set_gauge(
                        "policy_max_latency_ms_last",
                        float(policy_decision.max_latency_ms),
                    )
                if policy_decision.budget_usd is not None:
                    self.slo_metrics.set_gauge(
                        "policy_budget_usd_last",
                        float(policy_decision.budget_usd),
                    )
                self.slo_metrics.set_gauge(
                    "policy_spent_usd_last",
                    float(ledger_entry.get("spent_usd", 0.0) or 0.0),
                )
            if route_failures_turn:
                self.slo_metrics.inc("route_failure_turn_total", label="any", value=len(route_failures_turn))
                for failure in route_failures_turn:
                    stage = str(failure.get("stage", "unknown")).strip() or "unknown"
                    self.slo_metrics.inc("route_failure_stage_total", label=stage)
            self.slo_metrics.inc("response_governance_total", label=governed.route)
            self.slo_metrics.inc(
                "response_governance_tier_total",
                label=f"{governed.route}:{governed.verbosity_tier}",
            )
            self.slo_metrics.set_gauge(
                "response_governance_min_words_last",
                float(governed.min_words),
                label=governed.route,
            )
            self.slo_metrics.set_gauge(
                "response_governance_max_words_last",
                float(governed.max_words),
                label=governed.route,
            )
            self.slo_metrics.set_gauge(
                "response_governance_words_last",
                float(governed.word_count),
                label=governed.route,
            )
            if governed.changed:
                self.slo_metrics.inc("response_governance_changed_total", label=governed.route)
            if governed.rejected:
                self.slo_metrics.inc(
                    "response_governance_rejected_total",
                    label=f"{governed.route}:{governed.reason or 'unknown'}",
                )

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
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            if include_debug:
                payload["jarvis_debug"] = {
                    "session_id": session_id or "",
                    "stage_latency_ms": stage_latency_ms,
                    "conversation_latency_ms": ctx_latency,
                    "model_route": route_latency,
                    "summary_stage": summary_stage,
                    "response_plan": response_plan,
                    "query_understanding": query_understanding,
                    "route_failures_turn": route_failures_turn,
                    "response_governance": governed.to_dict(),
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
            "choices": [],
        }
        if include_debug:
            chunk["jarvis_debug"] = {
                "session_id": session_id or "",
                "stage_latency_ms": stage_latency_ms,
                "conversation_latency_ms": ctx_latency,
                "model_route": route_latency,
                "summary_stage": summary_stage,
                "response_plan": response_plan,
                "route_failures_turn": route_failures_turn,
                "response_governance": governed.to_dict(),
            }
        response_chunks = self._chunk_text_for_stream(str(response_text))
        for idx, part in enumerate(response_chunks):
            piece = dict(chunk)
            piece["choices"] = [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": part},
                    "finish_reason": None if idx < (len(response_chunks) - 1) else "stop",
                }
            ]
            await resp.write(f"data: {json.dumps(piece)}\n\n".encode("utf-8"))
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    @staticmethod
    def _chunk_text_for_stream(text: str) -> list[str]:
        raw = str(text or "")
        if not raw:
            return [""]
        parts = [p for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip()]
        if len(parts) <= 1:
            max_chunk = 120
            chunks = [raw[i : i + max_chunk] for i in range(0, len(raw), max_chunk)]
            return chunks or [raw]
        out: list[str] = []
        for sentence in parts:
            s = sentence.strip()
            if not s:
                continue
            if len(s) <= 200:
                out.append(s + " ")
            else:
                out.extend([s[i : i + 180] for i in range(0, len(s), 180)])
        return out or [raw]

    async def _handle_list_agents(self, _request: "web.Request") -> "web.Response":
        """GET /api/v1/agents — list registered agents."""
        if self.orchestrator and hasattr(self.orchestrator, "get_system_status"):
            status = self.orchestrator.get_system_status()
            agents = status.get("agents", [])
        else:
            agents = []
        if self.slo_metrics:
            self.slo_metrics.set_gauge("agents_total_count", float(len(agents)))
            by_state: dict[str, int] = {}
            for a in agents:
                state = str((a or {}).get("state", "unknown")).strip().lower() or "unknown"
                by_state[state] = by_state.get(state, 0) + 1
            for st, count in by_state.items():
                self.slo_metrics.set_gauge("agents_state_count", float(count), label=st)
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
                envelope = RequestEnvelope.from_any(
                    {
                        "request_id": str(getattr(request, "request_id", "") or ""),
                        "user_id": str(request.headers.get("X-User-Id", "api_user")),
                        "session_id": str(body.get("session_id", "")).strip(),
                        "model": str(body.get("model", "jarvis-default")),
                        "workspace_path": str(body.get("workspace_path", "")).strip(),
                        "route": "/api/v1/tasks",
                        "metadata": {
                            "source": "task_submit",
                            "required_capabilities": required_capabilities[:8],
                        },
                    }
                )
                metadata = dict(metadata)
                metadata["request_envelope"] = envelope.to_dict()
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
            self._update_task_gauges()

        return self._ok_response(request, task_entry, status=202)

    async def _handle_code_assist(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/code/assist — run repository code update task."""
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")

        instruction = str(body.get("instruction", "")).strip()
        user_id = str(request.headers.get("X-User-ID", "api_user"))
        workspace_path = self._resolve_workspace_hint(
            request=request,
            body=body,
            user_id=user_id,
            session_id="",
        ) or str(os.getcwd())
        req_obj, req_err = CodeAssistRequest.from_any(
            {
                "workspace_path": workspace_path,
                "instruction": instruction,
                "dry_run": bool(body.get("dry_run", False)),
                "run_checks": bool(body.get("run_checks", True)),
                "max_files": int(body.get("max_files", 10) or 10),
            }
        )
        if req_err or req_obj is None:
            return self._bad_request(request, req_err or "Invalid code assist request")
        envelope = RequestEnvelope.from_any(
            {
                "request_id": self._request_id(request),
                "user_id": user_id,
                "session_id": "",
                "model": "jarvis-default",
                "modality": "text",
                "workspace_path": workspace_path,
                "route": "code_assist",
            }
        )

        result = await self._submit_code_assist_task(
            instruction=req_obj.instruction,
            workspace_path=req_obj.workspace_path,
            dry_run=req_obj.dry_run,
            run_checks=req_obj.run_checks,
            wait=bool(body.get("wait", True)),
            timeout_s=float(body.get("timeout_seconds", 240) or 240),
            priority=int(body.get("priority", 6) or 6),
            max_files=req_obj.max_files,
        )
        if result.get("ok"):
            status = 202 if not result.get("waited", True) else 200
            payload = result.get("data")
            if isinstance(payload, dict):
                payload["request_envelope"] = envelope.to_dict()
            return self._ok_response(request, payload, status=status)
        return self._error_response(request, str(result.get("error", "Code assist failed")), status=500)

    async def _handle_code_understand(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/code/understand — analyze repository architecture and flows."""
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        user_id = str(request.headers.get("X-User-ID", "api_user"))
        workspace_path = self._resolve_workspace_hint(
            request=request,
            body=body,
            user_id=user_id,
            session_id="",
        ) or str(os.getcwd())
        question = str(body.get("question", "")).strip() or "Explain this repository."
        req_obj, req_err = RepoUnderstandRequest.from_any(
            {
                "workspace_path": workspace_path,
                "question": question,
                "max_files": int(body.get("max_files", 30) or 30),
                "include_tree": bool(body.get("include_tree", True)),
                "depth": str(body.get("depth", "medium") or "medium"),
            }
        )
        if req_err or req_obj is None:
            return self._bad_request(request, req_err or "Invalid repository analysis request")
        envelope = RequestEnvelope.from_any(
            {
                "request_id": self._request_id(request),
                "user_id": user_id,
                "session_id": "",
                "model": "jarvis-default",
                "modality": "text",
                "workspace_path": workspace_path,
                "route": "code_understand",
            }
        )
        result = await self._submit_repo_understand_task(
            workspace_path=req_obj.workspace_path,
            question=req_obj.question,
            wait=bool(body.get("wait", True)),
            timeout_s=float(body.get("timeout_seconds", 240) or 240),
            priority=int(body.get("priority", 5) or 5),
            max_files=req_obj.max_files,
            include_tree=req_obj.include_tree,
            depth=req_obj.depth,
        )
        if result.get("ok"):
            status = 202 if not result.get("waited", True) else 200
            payload = result.get("data")
            if isinstance(payload, dict):
                payload["request_envelope"] = envelope.to_dict()
            return self._ok_response(request, payload, status=status)
        return self._error_response(request, str(result.get("error", "Repository analysis failed")), status=500)

    async def _handle_code_workflow(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/code/workflow — run codex-like multi-agent workflow."""
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request(request, "Invalid JSON body")
        user_id = str(request.headers.get("X-User-ID", "api_user"))
        workspace_path = self._resolve_workspace_hint(
            request=request,
            body=body,
            user_id=user_id,
            session_id="",
        ) or str(os.getcwd())
        goal = str(body.get("goal", "")).strip() or str(body.get("instruction", "")).strip()
        req_obj, req_err = CodeWorkflowRequest.from_any(
            {
                "workspace_path": workspace_path,
                "goal": goal,
                "dry_run": bool(body.get("dry_run", False)),
                "run_checks": bool(body.get("run_checks", True)),
                "max_workers": int(body.get("max_workers", 3) or 3),
            }
        )
        if req_err or req_obj is None:
            return self._bad_request(request, req_err or "Invalid code workflow request")
        envelope = RequestEnvelope.from_any(
            {
                "request_id": self._request_id(request),
                "user_id": user_id,
                "session_id": "",
                "model": "jarvis-default",
                "modality": "text",
                "workspace_path": workspace_path,
                "route": "code_workflow",
            }
        )
        result = await self._submit_code_multi_agent_workflow(
            workspace_path=req_obj.workspace_path,
            goal=req_obj.goal,
            dry_run=req_obj.dry_run,
            run_checks=req_obj.run_checks,
            wait=bool(body.get("wait", True)),
            timeout_s=float(body.get("timeout_seconds", 420) or 420),
            max_workers=req_obj.max_workers,
        )
        if result.get("ok"):
            status = 202 if not result.get("waited", True) else 200
            payload = result.get("data")
            if isinstance(payload, dict):
                payload["request_envelope"] = envelope.to_dict()
            return self._ok_response(request, payload, status=status)
        return self._error_response(request, str(result.get("error", "Code workflow failed")), status=500)

    async def _submit_code_assist_task(
        self,
        *,
        instruction: str,
        workspace_path: str,
        dry_run: bool,
        run_checks: bool,
        wait: bool,
        timeout_s: float,
        priority: int,
        max_files: int = 10,
    ) -> dict[str, Any]:
        op_started = time.time()
        if not self.orchestrator or not hasattr(self.orchestrator, "submit_task"):
            return {"ok": False, "error": "Code assist unavailable: orchestrator not configured"}
        ws_decision = self.tool_isolation_policy.validate_workspace(workspace_path)
        if not ws_decision.allowed:
            return {
                "ok": False,
                "error": f"Workspace rejected by isolation policy: {ws_decision.reason}",
            }
        payload = {
            "workspace_path": ws_decision.workspace_path or workspace_path,
            "instruction": instruction,
            "dry_run": dry_run,
            "run_checks": run_checks,
            "max_files": max_files,
        }
        try:
            orch_task_id = await self.orchestrator.submit_task(
                description=f"Code assist: {instruction[:96]}",
                required_capabilities=["update_codebase"],
                payload=payload,
                priority=priority,
                timeout=timeout_s,
                metadata={"source": "api_code_assist"},
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Failed to submit code assist task: {exc}"}

        if self.slo_metrics:
            self.slo_metrics.inc("code_assist_total", label="submitted")

        if not wait or not hasattr(self.orchestrator, "wait_for_task"):
            return {
                "ok": True,
                "waited": False,
                "data": {"task_id": orch_task_id, "status": "submitted", "wait": False},
            }

        try:
            task = await self.orchestrator.wait_for_task(orch_task_id, timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001
            if self.slo_metrics:
                self.slo_metrics.inc("code_assist_total", label="timeout_or_error")
            return {"ok": False, "error": f"Code assist task wait failed: {exc}"}

        status_value = getattr(task.status, "value", str(task.status))
        if self.slo_metrics:
            self.slo_metrics.inc("code_assist_total", label=status_value)
            self.slo_metrics.observe_latency(
                "code_assist_latency_ms",
                (time.time() - op_started) * 1000.0,
                label=status_value,
            )
        return {
            "ok": True,
            "waited": True,
            "data": {
                "task_id": task.id,
                "status": status_value,
                "result": task.result,
                "error": task.error,
                "assigned_agent_id": task.assigned_agent_id,
            },
        }

    @staticmethod
    def _parse_chat_tool_request(
        *,
        last_user: str,
        body: dict[str, Any],
        workspace_hint: str = "",
    ) -> dict[str, Any] | None:
        mode = str(body.get("jarvis_mode", "")).strip().lower()
        if mode == "code_assist":
            workspace = str(body.get("workspace_path", "")).strip() or str(workspace_hint or "").strip()
            if not workspace:
                return None
            instruction = str(body.get("instruction", "")).strip() or str(last_user or "").strip()
            if not instruction:
                return None
            return {
                "workspace_path": workspace,
                "instruction": instruction,
                "dry_run": bool(body.get("dry_run", False)),
                "run_checks": bool(body.get("run_checks", True)),
                "timeout_seconds": float(body.get("timeout_seconds", 240) or 240),
                "priority": int(body.get("priority", 6) or 6),
                "type": "code_assist",
            }
        if mode in {"repo_understand", "understand_repo", "code_understand"}:
            workspace = str(body.get("workspace_path", "")).strip() or str(workspace_hint or "").strip()
            if not workspace:
                return None
            question = str(body.get("question", "")).strip() or str(last_user or "").strip() or "Explain this repository."
            return {
                "workspace_path": workspace,
                "question": question,
                "max_files": int(body.get("max_files", 30) or 30),
                "include_tree": bool(body.get("include_tree", True)),
                "depth": str(body.get("depth", "medium") or "medium"),
                "timeout_seconds": float(body.get("timeout_seconds", 240) or 240),
                "priority": int(body.get("priority", 5) or 5),
                "type": "repo_understand",
            }
        if mode in {"code_workflow", "workflow"}:
            workspace = str(body.get("workspace_path", "")).strip() or str(workspace_hint or "").strip()
            if not workspace:
                return None
            goal = str(body.get("goal", "")).strip() or str(last_user or "").strip()
            if not goal:
                return None
            return {
                "workspace_path": workspace,
                "goal": goal,
                "dry_run": bool(body.get("dry_run", False)),
                "run_checks": bool(body.get("run_checks", True)),
                "timeout_seconds": float(body.get("timeout_seconds", 420) or 420),
                "max_workers": int(body.get("max_workers", 3) or 3),
                "type": "code_workflow",
            }
        raw = str(last_user or "").strip()
        if raw and not raw.startswith("/"):
            workspace_auto = str(workspace_hint or "").strip()
            repo_intent = bool(
                re.search(r"\b(repo|repository|codebase|project)\b", raw, re.IGNORECASE)
                and re.search(
                    r"\b(read|scan|analy[sz]e|understand|explain|summari[sz]e|review)\b",
                    raw,
                    re.IGNORECASE,
                )
            )
            if workspace_auto and repo_intent:
                return {
                    "workspace_path": workspace_auto,
                    "question": raw,
                    "max_files": int(body.get("max_files", 40) or 40),
                    "include_tree": bool(body.get("include_tree", True)),
                    "depth": str(body.get("depth", "medium") or "medium"),
                    "timeout_seconds": float(body.get("timeout_seconds", 240) or 240),
                    "priority": int(body.get("priority", 5) or 5),
                    "type": "repo_understand",
                }
        if not (raw.startswith("/code") or raw.startswith("/repo") or raw.startswith("/workflow")):
            return None
        # Chat command syntax:
        # /code --workspace /abs/path --dry-run --no-checks your instruction
        # /repo --workspace /abs/path [--max-files 40] [--depth low|medium|high] [--no-tree] [question]
        # /workflow --workspace /abs/path [--dry-run] [--max-workers 3] [--no-checks] goal
        try:
            tokens = shlex.split(raw)
        except ValueError:
            return {
                "type": "command_error",
                "error": "I couldn't parse that slash command. Check quotes and try again.",
            }
        cmd = tokens[0] if tokens else ""
        workspace = ""
        dry_run = False
        run_checks = True
        include_tree = True
        max_files = 30
        depth = "medium"
        max_workers = 3
        idx = 1
        while idx < len(tokens):
            tok = tokens[idx]
            if tok == "--workspace" and idx + 1 < len(tokens):
                workspace = tokens[idx + 1]
                idx += 2
                continue
            if tok == "--dry-run":
                dry_run = True
                idx += 1
                continue
            if tok == "--no-checks":
                run_checks = False
                idx += 1
                continue
            if tok == "--max-files" and idx + 1 < len(tokens):
                try:
                    max_files = int(tokens[idx + 1])
                except Exception:
                    max_files = 30
                idx += 2
                continue
            if tok == "--no-tree":
                include_tree = False
                idx += 1
                continue
            if tok == "--depth" and idx + 1 < len(tokens):
                cand = str(tokens[idx + 1]).strip().lower()
                if cand in {"low", "medium", "high"}:
                    depth = cand
                idx += 2
                continue
            if tok == "--max-workers" and idx + 1 < len(tokens):
                try:
                    max_workers = int(tokens[idx + 1])
                except Exception:
                    max_workers = 3
                idx += 2
                continue
            break
        tail = " ".join(tokens[idx:]).strip()
        if not workspace:
            workspace = str(workspace_hint or "").strip()
        if not workspace:
            return {
                "type": "command_error",
                "error": (
                    "I couldn't detect your workspace path. "
                    "Make sure Jarvis Bridge is running, or pass "
                    "`--workspace /absolute/path`."
                ),
            }
        if cmd == "/code":
            if not tail:
                return None
            return {
                "workspace_path": workspace,
                "instruction": tail,
                "dry_run": dry_run,
                "run_checks": run_checks,
                "timeout_seconds": 240.0,
                "priority": 6,
                "type": "code_assist",
            }
        if cmd == "/workflow":
            if not tail:
                return None
            return {
                "workspace_path": workspace,
                "goal": tail,
                "dry_run": dry_run,
                "run_checks": run_checks,
                "timeout_seconds": 420.0,
                "max_workers": max_workers,
                "type": "code_workflow",
            }
        return {
            "workspace_path": workspace,
            "question": tail or "Explain this repository.",
            "max_files": max_files,
            "include_tree": include_tree,
            "depth": depth,
            "timeout_seconds": 240.0,
            "priority": 5,
            "type": "repo_understand",
        }

    @staticmethod
    def _validate_chat_tool_request(tool_req: dict[str, Any] | None) -> str:
        if not isinstance(tool_req, dict):
            return ""
        t = str(tool_req.get("type", "")).strip().lower()
        if t in {"", "command_error"}:
            return ""
        if t == "code_assist":
            _, err = CodeAssistRequest.from_any(tool_req)
            if err:
                return err
        elif t == "repo_understand":
            _, err = RepoUnderstandRequest.from_any(tool_req)
            if err:
                return err
        elif t == "code_workflow":
            _, err = CodeWorkflowRequest.from_any(tool_req)
            if err:
                return err
        return ""

    @staticmethod
    def _parse_code_assist_chat_request(
        *,
        last_user: str,
        body: dict[str, Any],
    ) -> dict[str, Any] | None:
        parsed = APIInterface._parse_chat_tool_request(last_user=last_user, body=body, workspace_hint="")
        if parsed and parsed.get("type") == "code_assist":
            out = dict(parsed)
            out.pop("type", None)
            return out
        return None

    def _resolve_workspace_hint(
        self,
        *,
        request: "web.Request",
        body: dict[str, Any],
        user_id: str,
        session_id: str,
    ) -> str:
        body_context = body.get("context", {}) if isinstance(body.get("context", {}), dict) else {}
        body_meta = body.get("metadata", {}) if isinstance(body.get("metadata", {}), dict) else {}
        candidates = [
            str(body.get("workspace_path", "")).strip(),
            str(body.get("jarvis_workspace", "")).strip(),
            str(body.get("workspace", "")).strip(),
            str(body.get("workspacePath", "")).strip(),
            str(body.get("project_path", "")).strip(),
            str(body.get("projectPath", "")).strip(),
            str(body.get("cwd", "")).strip(),
            str(body_context.get("workspace_path", "")).strip(),
            str(body_context.get("workspace", "")).strip(),
            str(body_context.get("cwd", "")).strip(),
            str(body_meta.get("workspace_path", "")).strip(),
            str(body_meta.get("workspace", "")).strip(),
            str(request.headers.get("X-Jarvis-Workspace", "")).strip(),
            str(request.headers.get("X-Workspace-Path", "")).strip(),
            str(request.headers.get("X-Continue-Workspace", "")).strip(),
            str(request.headers.get("X-Continue-Workspace-Path", "")).strip(),
            str(request.headers.get("X-Project-Path", "")).strip(),
            str(request.headers.get("X-Cwd", "")).strip(),
        ]
        if session_id:
            candidates.append(str(self._workspace_by_session.get(session_id, "")).strip())
        if user_id:
            candidates.append(str(self._workspace_by_user.get(user_id, "")).strip())
        env_default_workspace = str(os.getenv("JARVIS_DEFAULT_WORKSPACE_PATH", "")).strip()
        if env_default_workspace:
            candidates.append(env_default_workspace)
        candidates.append(str(Path.cwd()))
        for value in candidates:
            if not value:
                continue
            if user_id:
                self._workspace_by_user[user_id] = value
            if session_id:
                self._workspace_by_session[session_id] = value
            return value
        return ""

    async def _submit_repo_understand_task(
        self,
        *,
        workspace_path: str,
        question: str,
        wait: bool,
        timeout_s: float,
        priority: int,
        max_files: int,
        include_tree: bool,
        depth: str = "medium",
    ) -> dict[str, Any]:
        op_started = time.time()
        if not self.orchestrator or not hasattr(self.orchestrator, "submit_task"):
            return {"ok": False, "error": "Repository analysis unavailable: orchestrator not configured"}
        ws_decision = self.tool_isolation_policy.validate_workspace(workspace_path)
        if not ws_decision.allowed:
            return {
                "ok": False,
                "error": f"Workspace rejected by isolation policy: {ws_decision.reason}",
            }
        payload = {
            "workspace_path": ws_decision.workspace_path or workspace_path,
            "question": question,
            "max_files": max_files,
            "include_tree": include_tree,
            "depth": str(depth or "medium").strip().lower(),
        }
        try:
            orch_task_id = await self.orchestrator.submit_task(
                description=f"Repo understand: {question[:96]}",
                required_capabilities=["understand_codebase"],
                payload=payload,
                priority=priority,
                timeout=timeout_s,
                metadata={"source": "api_repo_understand"},
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Failed to submit repository analysis task: {exc}"}
        if self.slo_metrics:
            self.slo_metrics.inc("repo_understand_total", label="submitted")
        if not wait or not hasattr(self.orchestrator, "wait_for_task"):
            return {
                "ok": True,
                "waited": False,
                "data": {"task_id": orch_task_id, "status": "submitted", "wait": False},
            }
        try:
            task = await self.orchestrator.wait_for_task(orch_task_id, timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001
            if self.slo_metrics:
                self.slo_metrics.inc("repo_understand_total", label="timeout_or_error")
            return {"ok": False, "error": f"Repository analysis wait failed: {exc}"}
        status_value = getattr(task.status, "value", str(task.status))
        if self.slo_metrics:
            self.slo_metrics.inc("repo_understand_total", label=status_value)
            self.slo_metrics.observe_latency(
                "repo_understand_latency_ms",
                (time.time() - op_started) * 1000.0,
                label=status_value,
            )
        return {
            "ok": True,
            "waited": True,
            "data": {
                "task_id": task.id,
                "status": status_value,
                "result": task.result,
                "error": task.error,
                "assigned_agent_id": task.assigned_agent_id,
            },
        }

    async def _submit_code_multi_agent_workflow(
        self,
        *,
        workspace_path: str,
        goal: str,
        dry_run: bool,
        run_checks: bool,
        wait: bool,
        timeout_s: float,
        max_workers: int,
    ) -> dict[str, Any]:
        op_started = time.time()
        if not self.orchestrator or not hasattr(self.orchestrator, "submit_task_plan"):
            return {"ok": False, "error": "Code workflow unavailable: orchestrator plan API not configured"}
        ws_decision = self.tool_isolation_policy.validate_workspace(workspace_path)
        if not ws_decision.allowed:
            return {
                "ok": False,
                "error": f"Workspace rejected by isolation policy: {ws_decision.reason}",
            }
        workspace_path = ws_decision.workspace_path or workspace_path
        subtasks = self._split_goal_into_subtasks(goal, max_items=max_workers)
        plan_steps: list[dict[str, Any]] = [
            {
                "name": "understand_repo",
                "capability": "understand_codebase",
                "payload": {
                    "workspace_path": workspace_path,
                    "question": f"Understand this repository before implementing: {goal}",
                    "max_files": 40,
                    "include_tree": True,
                },
                "depends_on": [],
            }
        ]
        apply_step_names: list[str] = []
        for idx, sub in enumerate(subtasks):
            step_name = f"apply_change_{idx + 1}"
            apply_step_names.append(step_name)
            plan_steps.append(
                {
                    "name": step_name,
                    "capability": "update_codebase",
                    "payload": {
                        "workspace_path": workspace_path,
                        "instruction": sub,
                        "dry_run": dry_run,
                        "run_checks": False,
                        "max_files": 12,
                    },
                    "depends_on": ["understand_repo"],
                }
            )
        plan_steps.append(
            {
                "name": "verify_outcome",
                "capability": "understand_codebase",
                "payload": {
                    "workspace_path": workspace_path,
                    "question": (
                        "Summarize what changed, likely impact, and any risks after these edits: "
                        + "; ".join(subtasks)
                    ),
                    "max_files": 40,
                    "include_tree": False,
                },
                "depends_on": apply_step_names,
            }
        )
        if run_checks:
            plan_steps.append(
                {
                    "name": "final_checks",
                    "capability": "update_codebase",
                    "payload": {
                        "workspace_path": workspace_path,
                        "instruction": "Run project checks only; no source edits.",
                        "dry_run": True,
                        "run_checks": True,
                        "max_files": 1,
                    },
                    "depends_on": ["verify_outcome"],
                }
            )
        try:
            result = await self.orchestrator.submit_task_plan(
                description=f"code_workflow:{goal[:120]}",
                steps=plan_steps,
                priority=7,
                metadata={"source": "api_code_workflow", "workspace_path": workspace_path},
            )
            plan_id = str(result.get("plan_id", "")).strip()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Failed to submit workflow plan: {exc}"}

        if self.slo_metrics:
            self.slo_metrics.inc("code_workflow_total", label="submitted")

        if not wait or not plan_id:
            return {"ok": True, "waited": False, "data": {"plan_id": plan_id, "status": "submitted"}}

        try:
            status = await self._wait_for_plan_status(plan_id=plan_id, timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001
            if self.slo_metrics:
                self.slo_metrics.inc("code_workflow_total", label="timeout_or_error")
            return {"ok": False, "error": f"Workflow wait failed: {exc}"}

        overall = str(status.get("status", "unknown")).strip().lower()
        if self.slo_metrics:
            self.slo_metrics.inc("code_workflow_total", label=overall or "unknown")
            self.slo_metrics.observe_latency(
                "code_workflow_latency_ms",
                (time.time() - op_started) * 1000.0,
                label=overall or "unknown",
            )
        return {"ok": True, "waited": True, "data": status}

    async def _wait_for_plan_status(self, *, plan_id: str, timeout_s: float) -> dict[str, Any]:
        if not self.orchestrator or not hasattr(self.orchestrator, "get_plan_status"):
            raise RuntimeError("Orchestrator plan status API unavailable")
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            status = self.orchestrator.get_plan_status(plan_id) or {}
            overall = str(status.get("status", "")).lower()
            if overall in {"completed", "failed"}:
                return status
            await asyncio.sleep(0.25)
        raise TimeoutError(f"Plan {plan_id} did not complete before timeout")

    @staticmethod
    def _split_goal_into_subtasks(goal: str, *, max_items: int) -> list[str]:
        text = str(goal or "").strip()
        if not text:
            return ["Implement requested change."]
        chunks = [
            c.strip(" .")
            for c in re.split(r"\s+(?:and|then)\s+|[,;]\s*", text)
            if c.strip(" .")
        ]
        if not chunks:
            chunks = [text]
        max_items = max(1, min(8, int(max_items or 1)))
        return chunks[:max_items]

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
        if self.slo_metrics:
            state = str(task.get("status", "unknown")).strip().lower() or "unknown"
            self.slo_metrics.inc("task_read_total", label=state)
            self._update_task_gauges()
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
            replan_fn = self.orchestrator.replan_task
            try:
                sig = inspect.signature(replan_fn)
                params = sig.parameters
            except Exception:
                params = {}
            if "approval_token" in params:
                new_orch_task_id = await replan_fn(
                    orch_task_id,
                    fallback_capabilities=fallback_capabilities,
                    payload_override=payload_override,
                    description_suffix=description_suffix,
                    approval_token=approval_token,
                )
            else:
                new_orch_task_id = await replan_fn(
                    orch_task_id,
                    fallback_capabilities=fallback_capabilities,
                    payload_override=payload_override,
                    description_suffix=description_suffix,
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
            envelope = RequestEnvelope.from_any(
                {
                    "request_id": str(getattr(request, "request_id", "") or ""),
                    "user_id": str(request.headers.get("X-User-Id", "api_user")),
                    "session_id": str(body.get("session_id", "")).strip(),
                    "model": str(body.get("model", "jarvis-default")),
                    "workspace_path": str(body.get("workspace_path", "")).strip(),
                    "route": "/api/v1/plans",
                    "metadata": {
                        "source": "plan_submit",
                        "step_count": len(steps),
                    },
                }
            )
            metadata = dict(metadata)
            metadata["request_envelope"] = envelope.to_dict()
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
        if self.slo_metrics:
            self.slo_metrics.inc("plan_submit_total", label="accepted")
            try:
                self.slo_metrics.set_gauge("plans_step_count_last", float(result.get("step_count", 0)))
            except Exception:
                pass
            self._update_orchestrator_gauges()
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

    async def _handle_get_workflow_checkpoint(self, request: "web.Request") -> "web.Response":
        """GET /api/v1/workflows/{workflow_id}/checkpoint — get workflow checkpoint."""
        workflow_id = request.match_info.get("workflow_id", "")
        if not self.orchestrator or not hasattr(self.orchestrator, "get_workflow_checkpoint"):
            return self._error_response(request, "Workflow checkpoints unsupported by orchestrator", status=503)
        try:
            checkpoint = self.orchestrator.get_workflow_checkpoint(workflow_id)
        except Exception as exc:  # noqa: BLE001
            return self._error_response(request, str(exc), status=500)
        if checkpoint is None:
            return self._error_response(request, f"Workflow checkpoint not found: {workflow_id}", status=404)
        return self._ok_response(request, checkpoint)

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
        if self.slo_metrics:
            self.slo_metrics.inc("skill_execute_total", label="list:success")
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
            "ingress": self.ingress_controller.snapshot(),
            "query_understanding": self._conversation_understanding_snapshot(),
            "realtime": self._conversation_realtime_snapshot(),
            "realtime_streams": {
                "active_streams": len(self.live_stream_ingest.list_streams()),
                "samples": self.live_stream_ingest.list_streams()[:5],
            },
            "tool_isolation": {
                "enabled": bool(self.tool_isolation_policy.enabled),
                "allowed_roots": list(self.tool_isolation_policy.allowed_roots),
            },
            "fallback_contracts": {
                "chat": get_fallback("chat"),
                "repo": get_fallback("repo"),
                "code_assist": get_fallback("code_assist"),
                "code_workflow": get_fallback("code_workflow"),
            },
        }
        if self.monitor:
            report = await self.monitor.get_health_report()
            data["health"] = report.to_dict()
        if self.orchestrator and hasattr(self.orchestrator, "get_system_status"):
            data["orchestrator"] = self.orchestrator.get_system_status()
        return self._ok_response(_request, data)

    def _conversation_understanding_snapshot(self) -> dict[str, Any]:
        cm = self.conversation_manager
        if cm is None:
            return {"enabled": False, "active_sessions": 0, "sessions_with_understanding": 0, "samples": []}
        if not hasattr(cm, "list_active_sessions"):
            return {"enabled": False, "active_sessions": 0, "sessions_with_understanding": 0, "samples": []}

        try:
            sessions = cm.list_active_sessions()
        except Exception:
            sessions = []
        if not isinstance(sessions, list):
            sessions = []

        samples: list[dict[str, Any]] = []
        sessions_with_understanding = 0
        for row in sessions:
            if not isinstance(row, dict):
                continue
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            q = metadata.get("query_understanding", {})
            if not isinstance(q, dict) or not q:
                continue
            sessions_with_understanding += 1
            samples.append(
                {
                    "session_id": str(row.get("session_id", "")),
                    "intent": str(row.get("intent", "")),
                    "topic": str(row.get("current_topic", "")),
                    "turn_count": int(row.get("turn_count", 0) or 0),
                    "understanding": {
                        "inferred_intent": str(q.get("inferred_intent", "")),
                        "user_goal": str(q.get("user_goal", "")),
                        "missing_constraints": (
                            list(q.get("missing_constraints", []))
                            if isinstance(q.get("missing_constraints", []), list)
                            else []
                        ),
                        "confidence": float(q.get("confidence", 0.0) or 0.0),
                        "should_ask_clarification": bool(q.get("should_ask_clarification", False)),
                    },
                }
            )
        samples.sort(key=lambda x: int(x.get("turn_count", 0) or 0), reverse=True)
        return {
            "enabled": True,
            "active_sessions": len(sessions),
            "sessions_with_understanding": sessions_with_understanding,
            "samples": samples[:5],
        }

    def _conversation_realtime_snapshot(self) -> dict[str, Any]:
        cm = self.conversation_manager
        if cm is None or not hasattr(cm, "list_realtime_sessions"):
            return {"enabled": False, "active_realtime_sessions": 0, "samples": []}
        try:
            sessions = cm.list_realtime_sessions()
        except Exception:
            sessions = []
        if not isinstance(sessions, list):
            sessions = []
        samples: list[dict[str, Any]] = []
        for row in sessions[:5]:
            if not isinstance(row, dict):
                continue
            samples.append(
                {
                    "session_id": str(row.get("session_id", "")),
                    "user_id": str(row.get("user_id", "")),
                    "interrupt_epoch": int(row.get("interrupt_epoch", 0) or 0),
                    "frame_count": int(row.get("frame_count", 0) or 0),
                    "last_interrupt_reason": str(row.get("last_interrupt_reason", "")),
                }
            )
        return {
            "enabled": True,
            "active_realtime_sessions": len(sessions),
            "samples": samples,
        }

    async def _call_conversation_manager(
        self,
        *,
        session_id: str,
        query: str,
        modality: str,
        media: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        cm = self.conversation_manager
        if cm is None or not hasattr(cm, "process_input"):
            return f"Echo: {query}"
        fn = cm.process_input
        try:
            sig = inspect.signature(fn)
            params = sig.parameters
        except Exception:
            params = {}
        supports_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()) if params else False
        if supports_kwargs or ("modality" in params and "media" in params and "context" in params):
            return await fn(session_id, query, modality=modality, media=media, context=context)
        return await fn(session_id, query)

    @staticmethod
    def _split_response_sections(text: str, *, max_sections: int = 6) -> list[str]:
        body = str(text or "").strip()
        if not body:
            return []
        max_sections = max(1, min(16, int(max_sections or 6)))
        blocks = [b.strip() for b in re.split(r"\n{2,}", body) if b.strip()]
        if len(blocks) >= 2:
            if len(blocks) <= max_sections:
                return blocks
            merged: list[str] = []
            chunk_size = max(1, len(blocks) // max_sections)
            for i in range(0, len(blocks), chunk_size):
                merged.append("\n\n".join(blocks[i : i + chunk_size]).strip())
            return merged[:max_sections]
        sentences = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", body) if s.strip()]
        if len(sentences) <= 1:
            return [body]
        sections: list[str] = []
        per = max(1, (len(sentences) + max_sections - 1) // max_sections)
        for i in range(0, len(sentences), per):
            sections.append(" ".join(sentences[i : i + per]).strip())
        return sections[:max_sections]

    @staticmethod
    def _infer_streaming_hints_from_query(query: str) -> dict[str, Any]:
        low = str(query or "").strip().lower()
        if not low:
            return {}
        hints: dict[str, Any] = {}
        if re.search(r"\b(stream|section|part|chunk)\b", low):
            hints["sectioned"] = True
        m = re.search(r"\b(?:in|with)\s+(\d{1,2})\s+(?:sections|parts|chunks)\b", low)
        if m:
            try:
                hints["max_sections"] = max(1, min(16, int(m.group(1))))
                hints["sectioned"] = True
            except Exception:
                pass
        return hints

    @staticmethod
    def _fast_social_scene_reply(*, query: str, modality: str, context: dict[str, Any]) -> str:
        def _sanitize_greet_name(raw: str) -> str:
            value = re.split(r"[,.;:!?]", str(raw or "").strip(), maxsplit=1)[0].strip()
            value = re.sub(r"\s+", " ", value)
            if not value:
                return ""
            # Strip trailing style/instruction words so we only keep person identity.
            trail = re.compile(
                r"\s+(?:naturally|socially|briefly|politely|casually|warmly|kindly|concisely|"
                r"in\s+one\s+sentence|in\s+1\s+sentence)$",
                re.IGNORECASE,
            )
            while True:
                next_value = trail.sub("", value).strip()
                if next_value == value:
                    break
                value = next_value
            value = re.sub(r"\s+(?:using|with|in)\s+.*$", "", value, flags=re.IGNORECASE).strip()
            return value

        txt = str(query or "").strip()
        if not txt:
            return ""
        low = txt.lower()
        source = str((context or {}).get("source", "")).strip().lower()
        text_social_prefixed = low.startswith("[scene assistant]")
        social_phrase = "social room update" in low or "greet " in low
        is_voice = str(modality or "text").strip().lower() == "voice"
        if not (source == "social_scene_orchestrator" or text_social_prefixed or (is_voice and social_phrase)):
            return ""
        if text_social_prefixed:
            txt = re.sub(r"^\s*\[scene assistant\]\s*", "", txt, flags=re.IGNORECASE).strip()
            low = txt.lower()
        if "greet" in low:
            m = re.search(r"\bgreet\s+([a-z][a-z\s.'-]{1,60})", txt, re.IGNORECASE)
            if m:
                name = _sanitize_greet_name(str(m.group(1) or ""))
                if name:
                    return f"{name} is in the room now. Hi {name}, good to see you."
            return "Someone is in the room now. Hi, good to see you."
        if "unidentified" in low or "still unidentified" in low:
            return "I can see people in the room and at least one is still unidentified. Want me to learn them now?"
        if "room update" in low or "social room update" in low:
            return "Quick update: I am tracking the room and will notify you when someone is identified."
        return ""

    async def _handle_metrics(self, request: "web.Request") -> "web.Response":
        if self.slo_metrics:
            self._update_task_gauges()
            self._update_orchestrator_gauges()
            self._update_ingress_gauges()
        metrics = self.slo_metrics.snapshot() if self.slo_metrics else {}
        slo = evaluate_slo_snapshot(metrics, thresholds=self._slo_thresholds)
        return self._ok_response(request, {"metrics": metrics, "slo": slo})

    def _update_ingress_gauges(self) -> None:
        if not self.slo_metrics:
            return
        snap = self.ingress_controller.snapshot()
        self.slo_metrics.set_gauge("ingress_enabled", 1.0 if snap.get("enabled") else 0.0)
        self.slo_metrics.set_gauge("ingress_max_inflight", float(snap.get("max_inflight", 0) or 0))
        self.slo_metrics.set_gauge("ingress_max_queue", float(snap.get("max_queue", 0) or 0))
        self.slo_metrics.set_gauge("ingress_inflight", float(snap.get("inflight", 0) or 0))
        self.slo_metrics.set_gauge("ingress_queued", float(snap.get("queued", 0) or 0))

    def _update_task_gauges(self) -> None:
        if not self.slo_metrics:
            return
        total = len(self._tasks)
        self.slo_metrics.set_gauge("tasks_total_count", float(total))
        by_state: dict[str, int] = {}
        for t in self._tasks.values():
            st = str((t or {}).get("status", "unknown")).strip().lower() or "unknown"
            by_state[st] = by_state.get(st, 0) + 1
        for st, count in by_state.items():
            self.slo_metrics.set_gauge("tasks_state_count", float(count), label=st)

    def _update_orchestrator_gauges(self) -> None:
        if not (self.slo_metrics and self.orchestrator and hasattr(self.orchestrator, "get_system_status")):
            return
        try:
            status = self.orchestrator.get_system_status()
        except Exception:
            return
        if not isinstance(status, dict):
            return
        orch = status.get("orchestrator", {}) if isinstance(status.get("orchestrator", {}), dict) else {}
        self.slo_metrics.set_gauge(
            "orchestrator_registered_agents",
            float(orch.get("registered_agents", 0) or 0),
        )
        self.slo_metrics.set_gauge(
            "orchestrator_running_tasks",
            float(orch.get("running_tasks", 0) or 0),
        )
        self.slo_metrics.set_gauge(
            "orchestrator_workflow_checkpoints",
            float(orch.get("workflow_checkpoints", 0) or 0),
        )
        task_counts = status.get("task_counts", {}) if isinstance(status.get("task_counts", {}), dict) else {}
        for key, val in task_counts.items():
            try:
                self.slo_metrics.set_gauge("orchestrator_task_count", float(val), label=str(key))
            except Exception:
                continue
        plan_counts = status.get("plan_counts", {}) if isinstance(status.get("plan_counts", {}), dict) else {}
        for key, val in plan_counts.items():
            try:
                self.slo_metrics.set_gauge("orchestrator_plan_count", float(val), label=str(key))
            except Exception:
                continue
        wf_stats = orch.get("workflow_stats", {}) if isinstance(orch.get("workflow_stats", {}), dict) else {}
        for key, val in wf_stats.items():
            try:
                self.slo_metrics.set_gauge("orchestrator_workflow_stat", float(val), label=str(key))
            except Exception:
                continue
        pools = orch.get("resource_pools", {}) if isinstance(orch.get("resource_pools", {}), dict) else {}
        for key, val in pools.items():
            try:
                self.slo_metrics.set_gauge("orchestrator_resource_pool", float(val), label=str(key))
            except Exception:
                continue

    async def _handle_prometheus_metrics(self, _request: "web.Request") -> "web.Response":
        snapshot = self.slo_metrics.snapshot() if self.slo_metrics else {}
        text = self._to_prometheus_metrics(snapshot)
        return web.Response(
            text=text,
            content_type="text/plain",
            headers={"Cache-Control": "no-store"},
        )

    @staticmethod
    def _sanitize_metric_name(name: str) -> str:
        raw = str(name or "").strip().lower()
        raw = re.sub(r"[^a-z0-9_:]", "_", raw)
        raw = re.sub(r"_+", "_", raw).strip("_")
        if not raw:
            raw = "jarvis_metric"
        if raw[0].isdigit():
            raw = f"jarvis_{raw}"
        return raw

    @staticmethod
    def _sanitize_label_value(value: str) -> str:
        txt = str(value or "")
        txt = txt.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
        return txt

    @classmethod
    def _prom_line(cls, name: str, value: float, labels: dict[str, str] | None = None) -> str:
        metric = cls._sanitize_metric_name(name)
        if labels:
            label_body = ",".join(
                f'{cls._sanitize_metric_name(k)}="{cls._sanitize_label_value(v)}"'
                for k, v in sorted(labels.items())
            )
            return f"{metric}{{{label_body}}} {value}"
        return f"{metric} {value}"

    @classmethod
    def _to_prometheus_metrics(cls, snapshot: dict[str, Any]) -> str:
        lines: list[str] = []
        uptime = float(snapshot.get("uptime_seconds", 0.0) or 0.0)
        lines.append(cls._prom_line("jarvis_slo_uptime_seconds", uptime))

        for key, raw_val in (snapshot.get("counters", {}) or {}).items():
            parts = str(key).split(":", 1)
            mname = f"jarvis_{parts[0]}"
            label = parts[1] if len(parts) > 1 else "default"
            try:
                val = float(raw_val)
            except Exception:
                continue
            lines.append(cls._prom_line(mname, val, {"label": label}))

        for key, raw_val in (snapshot.get("gauges", {}) or {}).items():
            parts = str(key).split(":", 1)
            mname = f"jarvis_{parts[0]}"
            label = parts[1] if len(parts) > 1 else "default"
            try:
                val = float(raw_val)
            except Exception:
                continue
            lines.append(cls._prom_line(mname, val, {"label": label}))

        for key, series in (snapshot.get("latency", {}) or {}).items():
            if not isinstance(series, dict):
                continue
            parts = str(key).split(":", 1)
            base = f"jarvis_{parts[0]}"
            label = parts[1] if len(parts) > 1 else "default"
            for field in ("count", "avg_ms", "p50_ms", "p95_ms", "p99_ms"):
                if field not in series:
                    continue
                try:
                    val = float(series.get(field, 0.0))
                except Exception:
                    continue
                lines.append(cls._prom_line(f"{base}_{field}", val, {"label": label}))

        return "\n".join(lines) + "\n"

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
            err_text = str(exc)
            is_circuit_open = "circuit is open" in err_text.lower()
            if self.slo_metrics:
                self.slo_metrics.inc(
                    "connector_invoke_total",
                    label=f"{connector_name}:{'circuit_open' if is_circuit_open else 'error'}",
                )
            self._record_audit(
                request,
                event_type="connector",
                action=f"invoke_connector:{connector_name}",
                success=False,
                decision="deny",
                reason=err_text,
                metadata={"operation": operation},
            )
            return self._error_response(request, err_text, status=503 if is_circuit_open else 500)
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

    def _log_perf_breakdown(
        self,
        *,
        request: "web.Request",
        route: str,
        outcome: str,
        stages_ms: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self._perf_log_enabled:
            return
        rid = self._request_id(request)
        clean_stages: dict[str, float] = {}
        for key, val in (stages_ms or {}).items():
            try:
                clean_stages[str(key)] = round(float(val), 2)
            except Exception:
                continue
        total_ms = clean_stages.get("total")
        if total_ms is None:
            total_ms = round(sum(v for v in clean_stages.values() if v >= 0), 2)
            clean_stages["total"] = total_ms
        stage_str = ", ".join(f"{k}={v}ms" for k, v in clean_stages.items())
        extra_str = ""
        if isinstance(extra, dict) and extra:
            pairs: list[str] = []
            for k, v in extra.items():
                sval = str(v).strip()
                if not sval:
                    continue
                pairs.append(f"{k}={sval}")
            if pairs:
                extra_str = " " + " ".join(pairs)
        msg = (
            f"PERF request_id={rid} route={route} outcome={outcome} "
            f"total={clean_stages.get('total', total_ms)}ms "
            f"breakdown=[{stage_str}]"
            f"{extra_str}"
        )
        if float(total_ms) >= float(self._perf_slow_ms):
            logger.warning("%s SLOW_THRESHOLD=%sms", msg, round(self._perf_slow_ms, 2))
        else:
            logger.info(msg)

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
