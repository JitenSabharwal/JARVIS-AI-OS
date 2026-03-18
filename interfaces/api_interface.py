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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

try:
    import aiohttp
    from aiohttp import web
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False

from infrastructure.logger import get_logger

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
    ) -> None:
        self.host = host
        self.port = port
        self._auth_token = auth_token
        self._rate_limiter = RateLimiter(requests_per_minute)
        self._cors_origins = cors_origins or ["*"]

        # Injected service references (optional)
        self.conversation_manager: Any = None
        self.orchestrator: Any = None
        self.skills_registry: Any = None
        self.monitor: Any = None

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
            self._auth_middleware,
            self._rate_limit_middleware,
            self._cors_middleware,
            self._error_middleware,
        ])
        app.router.add_post("/api/v1/query", self._handle_query)
        app.router.add_get("/api/v1/agents", self._handle_list_agents)
        app.router.add_post("/api/v1/tasks", self._handle_submit_task)
        app.router.add_get("/api/v1/tasks/{task_id}", self._handle_get_task)
        app.router.add_get("/api/v1/skills", self._handle_list_skills)
        app.router.add_post("/api/v1/skills/{skill_name}/execute", self._handle_execute_skill)
        app.router.add_get("/api/v1/health", self._handle_health)
        app.router.add_get("/api/v1/status", self._handle_status)
        # OPTIONS for CORS preflight
        app.router.add_route("OPTIONS", "/{path_info:.*}", self._handle_options)
        return app

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    @web.middleware
    async def _auth_middleware(self, request: "web.Request", handler: Any) -> "web.Response":
        # Health endpoint is always public
        if request.path in ("/api/v1/health",) or request.method == "OPTIONS":
            return await handler(request)
        if self._auth_token:
            auth_header = request.headers.get("Authorization", "")
            token = auth_header.removeprefix("Bearer ").strip()
            if not self._constant_time_compare(token, self._auth_token):
                return web.json_response(
                    APIResponse(success=False, error="Unauthorized").to_dict(),
                    status=401,
                )
        return await handler(request)

    @web.middleware
    async def _rate_limit_middleware(self, request: "web.Request", handler: Any) -> "web.Response":
        client_ip = request.remote or "unknown"
        if not self._rate_limiter.is_allowed(client_ip):
            return web.json_response(
                APIResponse(success=False, error="Rate limit exceeded").to_dict(),
                status=429,
            )
        return await handler(request)

    @web.middleware
    async def _cors_middleware(self, request: "web.Request", handler: Any) -> "web.Response":
        response = await handler(request)
        origin = request.headers.get("Origin", "*")
        allowed = "*" in self._cors_origins or origin in self._cors_origins
        if allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        return response

    @web.middleware
    async def _error_middleware(self, request: "web.Request", handler: Any) -> "web.Response":
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Unhandled API error: %s", exc)
            return web.json_response(
                APIResponse(success=False, error=f"Internal server error: {exc}").to_dict(),
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
            return self._bad_request("Invalid JSON body")

        query = body.get("query", "").strip()
        if not query:
            return self._bad_request("'query' field is required")

        session_id = body.get("session_id")
        user_id = body.get("user_id", "api_user")

        if self.conversation_manager:
            if not session_id:
                session_id = self.conversation_manager.get_or_create_session(user_id)
            response_text = await self.conversation_manager.process_input(session_id, query)
            return web.json_response(APIResponse(
                success=True,
                data={"response": response_text, "session_id": session_id},
            ).to_dict())

        return web.json_response(APIResponse(
            success=True,
            data={"response": f"Echo: {query}", "session_id": session_id or "none"},
        ).to_dict())

    async def _handle_list_agents(self, _request: "web.Request") -> "web.Response":
        """GET /api/v1/agents — list registered agents."""
        if self.orchestrator and hasattr(self.orchestrator, "get_system_status"):
            status = self.orchestrator.get_system_status()
            agents = status.get("agents", [])
        else:
            agents = []
        return web.json_response(APIResponse(success=True, data={"agents": agents}).to_dict())

    async def _handle_submit_task(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/tasks — submit a task for execution."""
        body = await self._parse_json(request)
        if body is None:
            return self._bad_request("Invalid JSON body")

        description = body.get("description", "").strip()
        if not description:
            return self._bad_request("'description' field is required")

        task_id = str(uuid.uuid4())
        task_entry: dict[str, Any] = {
            "task_id": task_id,
            "description": description,
            "status": "pending",
            "submitted_at": time.time(),
            "result": None,
            "error": None,
        }
        self._tasks[task_id] = task_entry

        if self.orchestrator and hasattr(self.orchestrator, "submit_task"):
            try:
                orch_task_id = await self.orchestrator.submit_task(
                    description=description,
                    required_capabilities=body.get("required_capabilities", []),
                    priority=body.get("priority", 5),
                    payload=body.get("payload", {}),
                )
                task_entry["orchestrator_task_id"] = orch_task_id
                task_entry["status"] = "submitted"
            except Exception as exc:  # noqa: BLE001
                task_entry["status"] = "failed"
                task_entry["error"] = str(exc)

        return web.json_response(
            APIResponse(success=True, data=task_entry).to_dict(), status=202
        )

    async def _handle_get_task(self, request: "web.Request") -> "web.Response":
        """GET /api/v1/tasks/{task_id} — get task status."""
        task_id = request.match_info.get("task_id", "")
        task = self._tasks.get(task_id)
        if task is None:
            return web.json_response(
                APIResponse(success=False, error=f"Task not found: {task_id}").to_dict(),
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
            except Exception:  # noqa: BLE001
                pass
        return web.json_response(APIResponse(success=True, data=task).to_dict())

    async def _handle_list_skills(self, _request: "web.Request") -> "web.Response":
        """GET /api/v1/skills — list available skills."""
        if self.skills_registry and hasattr(self.skills_registry, "get_all_skills_info"):
            skills = self.skills_registry.get_all_skills_info()
        else:
            skills = []
        return web.json_response(APIResponse(success=True, data={"skills": skills}).to_dict())

    async def _handle_execute_skill(self, request: "web.Request") -> "web.Response":
        """POST /api/v1/skills/{skill_name}/execute — execute a skill."""
        skill_name = request.match_info.get("skill_name", "")
        body = await self._parse_json(request) or {}
        params = body.get("params", {})

        if not self.skills_registry:
            return web.json_response(
                APIResponse(success=False, error="Skills registry not available").to_dict(),
                status=503,
            )

        try:
            result = await self.skills_registry.execute_skill(skill_name, params)
            if hasattr(result, "success"):
                return web.json_response(APIResponse(
                    success=result.success,
                    data=result.data if result.success else None,
                    error=result.error if not result.success else None,
                ).to_dict())
            return web.json_response(APIResponse(success=True, data=result).to_dict())
        except KeyError:
            return web.json_response(
                APIResponse(success=False, error=f"Skill not found: {skill_name}").to_dict(),
                status=404,
            )
        except Exception as exc:  # noqa: BLE001
            return web.json_response(
                APIResponse(success=False, error=str(exc)).to_dict(), status=500
            )

    async def _handle_health(self, _request: "web.Request") -> "web.Response":
        """GET /api/v1/health — lightweight health check."""
        if self.monitor:
            report = await self.monitor.get_health_report()
            return web.json_response(APIResponse(
                success=True,
                data={
                    "status": report.overall_status.value,
                    "summary": report.summary,
                },
            ).to_dict())
        return web.json_response(
            APIResponse(success=True, data={"status": "healthy"}).to_dict()
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
        return web.json_response(APIResponse(success=True, data=data).to_dict())

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
    def _bad_request(message: str) -> "web.Response":
        return web.json_response(
            APIResponse(success=False, error=message).to_dict(), status=400
        )

    @staticmethod
    def _constant_time_compare(a: str, b: str) -> bool:
        """Timing-safe string comparison to prevent token-oracle attacks."""
        ha = hashlib.sha256(a.encode()).digest()
        hb = hashlib.sha256(b.encode()).digest()
        return hmac.compare_digest(ha, hb)

    def __repr__(self) -> str:
        return f"<APIInterface {self.host}:{self.port} running={self._running}>"
