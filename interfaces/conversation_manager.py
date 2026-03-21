"""
Conversation state management for JARVIS AI OS.

Handles multi-turn sessions, intent extraction (rule-based + LLM), contextual
response generation, and integration with ConversationMemory and KnowledgeBase.
"""

from __future__ import annotations

import json
import re
import socket
import time
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable
from urllib.parse import quote
from urllib.request import Request, urlopen

from infrastructure.logger import get_logger
from infrastructure.model_router import ModelRequest, ModelRouter, PrivacyLevel
from memory.conversation_memory import ConversationMemory, SessionManager
from memory.episodic_memory import EpisodicMemory
from memory.knowledge_base import KnowledgeBase
from memory.user_profile import UserProfileStore

logger = get_logger("conversation_manager")


# ---------------------------------------------------------------------------
# Enums & data structures
# ---------------------------------------------------------------------------

class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    ERROR = "error"


@dataclass
class ConversationContext:
    """Mutable context for an active conversation session."""
    session_id: str
    state: ConversationState = ConversationState.IDLE
    user_id: str = "anonymous"
    current_topic: str = ""
    intent: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    turn_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_activity = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "user_id": self.user_id,
            "current_topic": self.current_topic,
            "intent": self.intent,
            "entities": self.entities,
            "confidence": round(self.confidence, 3),
            "turn_count": self.turn_count,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "metadata": self.metadata,
        }


@dataclass
class ResponsePlan:
    """Per-turn plan used to choose handlers, models, and output controls."""
    intent: str
    task_type: str
    complexity: float
    target_length: str
    handler_key: str = ""
    preserve_format: bool = False
    use_model_router: bool = True
    allow_kb_lookup: bool = True
    require_context_fusion: bool = False
    prefer_local: bool | None = None
    max_latency_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "task_type": self.task_type,
            "complexity": round(float(self.complexity), 3),
            "target_length": self.target_length,
            "handler_key": self.handler_key,
            "preserve_format": self.preserve_format,
            "use_model_router": self.use_model_router,
            "allow_kb_lookup": self.allow_kb_lookup,
            "require_context_fusion": self.require_context_fusion,
            "prefer_local": self.prefer_local,
            "max_latency_ms": self.max_latency_ms,
        }


# ---------------------------------------------------------------------------
# Intent rules
# ---------------------------------------------------------------------------

# Each entry: (regex_pattern, intent_name, confidence_boost)
_INTENT_RULES: list[tuple[str, str, float]] = [
    # Greetings
    (r"\b(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))\b", "greeting", 0.95),
    # Farewells
    (r"\b(bye|goodbye|see\s+you|farewell|take\s+care)\b", "farewell", 0.95),
    # Help / capability enquiries
    (r"\b(help|what\s+can\s+you\s+do|capabilities|commands)\b", "help_request", 0.90),
    # Status / health
    (r"\b(status|health|how\s+are\s+you|system\s+status)\b", "status_query", 0.85),
    # Task execution
    (r"\b(run|execute|perform|start|launch|write|draft|compose|create|generate)\b", "task_execution", 0.80),
    (r"\b(can|could|would)\s+you\s+(read|scan|analy[sz]e|understand|explain|write|create|generate|help)\b", "task_execution", 0.84),
    # Information retrieval
    (r"\b(what|who|where|when|why|how|tell\s+me|explain|describe|define)\b", "information_query", 0.70),
    # Memory / recall
    (r"\b(remember|recall|do\s+you\s+know|what\s+is|lookup)\b", "memory_query", 0.72),
    # Settings / configuration
    (r"\b(set|configure|change|update|adjust|settings?)\b", "configuration", 0.80),
    # Stop / cancel
    (r"\b(stop|cancel|abort|quit|exit|terminate)\b", "cancel", 0.90),
    # Thanks
    (r"\b(thanks|thank\s+you|cheers|appreciate)\b", "acknowledgement", 0.90),
    # Confirmation
    (r"^\s*(yes|yep|yeah|sure|ok|okay|confirm|correct|right)\s*$", "confirmation", 0.95),
    # Negation
    (r"^\s*(no|nope|nah|cancel|negative|wrong|incorrect)\s*$", "negation", 0.95),
    # Weather
    (r"\b(weather|forecast|temperature|rain|snow|sunny)\b", "weather_query", 0.88),
    # Time / date
    (r"\b(time|date|day|today|tomorrow|yesterday|clock)\b", "time_query", 0.85),
]

# Compiled patterns cached at module load
_COMPILED_RULES: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(pat, re.IGNORECASE), intent, conf)
    for pat, intent, conf in _INTENT_RULES
]


# ---------------------------------------------------------------------------
# Simple entity extractors
# ---------------------------------------------------------------------------

_ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "number": re.compile(r"\b\d+(?:\.\d+)?\b"),
    "url": re.compile(r"https?://\S+"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.\w+\b"),
    "time": re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b", re.IGNORECASE),
    "date": re.compile(
        r"\b(?:\d{4}-\d{2}-\d{2}|(?:today|tomorrow|yesterday|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
        re.IGNORECASE,
    ),
}


def _extract_entities(text: str) -> dict[str, list[str]]:
    """Return a dict of entity_type -> list of matched strings."""
    entities: dict[str, list[str]] = {}
    for entity_type, pattern in _ENTITY_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            entities[entity_type] = matches
    return entities


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------

class ConversationManager:
    """
    Manages multi-turn conversations.

    Each session gets its own :class:`ConversationContext` and a
    :class:`~memory.conversation_memory.ConversationMemory` slice managed by
    the shared :class:`~memory.conversation_memory.SessionManager`.

    An optional :class:`~memory.knowledge_base.KnowledgeBase` is used for
    grounding responses in stored facts.
    """

    _SESSION_TIMEOUT = 3600     # seconds of inactivity before a session expires
    _FAST_PATH_INTENTS = {
        "greeting",
        "farewell",
        "acknowledgement",
        "status_query",
        "time_query",
        "weather_query",
        "help_request",
        "confirmation",
        "negation",
    }

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        session_manager: SessionManager | None = None,
        user_profile_store: UserProfileStore | None = None,
        episodic_memory: EpisodicMemory | None = None,
        llm_handler: Any | None = None,  # optional async fn(prompt) -> str
        model_router: ModelRouter | None = None,
        kb_min_confidence: float = 0.20,
        kb_max_age_days: int = 90,
    ) -> None:
        self._kb: KnowledgeBase = knowledge_base or KnowledgeBase()
        self._session_mgr: SessionManager = session_manager or SessionManager()
        self._profile_store: UserProfileStore = user_profile_store or UserProfileStore()
        self._episodic_memory: EpisodicMemory = episodic_memory or EpisodicMemory()
        self._llm_handler = llm_handler     # injected post-construction if needed
        self._model_router = model_router
        self._kb_min_confidence = max(0.0, float(kb_min_confidence))
        self._kb_max_age_days = max(1, int(kb_max_age_days))

        # session_id -> ConversationContext
        self._contexts: dict[str, ConversationContext] = {}
        self._response_handlers: dict[str, Callable[[ConversationContext, str], Awaitable[str]]] = {
            "time_query": self._handle_time_query,
            "weather_query": self._handle_weather_query,
            "email_draft": self._handle_email_draft,
            "code_draft": self._handle_code_draft,
        }

        # Simple response templates keyed by intent
        self._response_templates: dict[str, list[str]] = self._build_response_templates()

        logger.info("ConversationManager initialised (kb=%s)", bool(knowledge_base))

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(
        self,
        user_id: str = "anonymous",
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Open a new conversation session.

        Returns the session ID.
        """
        sid = session_id or str(uuid.uuid4())
        ctx = ConversationContext(
            session_id=sid,
            user_id=user_id,
            state=ConversationState.IDLE,
            metadata=metadata or {},
        )
        self._contexts[sid] = ctx
        self._session_mgr.get_or_create(sid)     # ensure memory slice exists
        self._profile_store.get_or_create(user_id)
        logger.info("Session started: %s (user=%s)", sid, user_id)
        return sid

    def end_session(self, session_id: str) -> bool:
        """Close a session and clear its in-memory state."""
        if session_id not in self._contexts:
            return False
        del self._contexts[session_id]
        self._session_mgr.delete(session_id)
        logger.info("Session ended: %s", session_id)
        return True

    def get_or_create_session(self, user_id: str = "anonymous") -> str:
        """Return an existing idle session for *user_id*, or create one."""
        self._expire_idle_sessions()
        for ctx in self._contexts.values():
            if ctx.user_id == user_id and ctx.state != ConversationState.ERROR:
                return ctx.session_id
        return self.start_session(user_id)

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    async def process_input(
        self,
        session_id: str,
        user_input: str,
        *,
        modality: str = "text",
        media: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Full pipeline: update context → extract intent → generate response.

        Returns the response string.
        """
        self._expire_idle_sessions()
        turn_started = time.time()
        ctx = self._get_context(session_id)
        ctx.state = ConversationState.PROCESSING
        ctx.turn_count += 1
        ctx.touch()
        self._update_modality_context(
            ctx=ctx,
            modality=modality,
            media=media or {},
            context=context or {},
        )
        if isinstance(context, dict):
            req_env = context.get("request_envelope")
            if isinstance(req_env, dict):
                ctx.metadata["request_envelope_last"] = req_env

        # Store in memory
        memory = self._session_mgr.get_or_create(session_id)
        memory.add_message(role="user", content=user_input)

        try:
            # Reset turn-local routing failure buffer.
            ctx.metadata["route_failures_turn"] = []

            # 1. Intent extraction
            intent_started = time.time()
            intent, entities, confidence = self.extract_intent(user_input)
            intent_latency_ms = round((time.time() - intent_started) * 1000.0, 2)
            ctx.intent = intent
            ctx.entities = entities
            ctx.confidence = confidence
            logger.debug(
                "Session %s — intent='%s' confidence=%.2f", session_id, intent, confidence
            )

            # 2. Topic tracking (simple heuristic: use the first noun-like word)
            words = user_input.split()
            if words:
                ctx.current_topic = words[0].lower()

            # 2.1 Learning loop: extract explicit user preferences.
            self._learn_user_profile(ctx.user_id, user_input)

            # 2.2 Dynamic response plan.
            plan = self._plan_response(
                ctx=ctx,
                user_input=user_input,
                modality=modality,
                media=media or {},
                context=context or {},
            )
            ctx.metadata["response_plan"] = plan.to_dict()

            # 3. Response generation
            ctx.state = ConversationState.RESPONDING
            gen_started = time.time()
            response = await self.generate_response(
                ctx,
                user_input,
                plan=plan,
                modality=modality,
                media=media or {},
                context=context or {},
            )
            gen_latency_ms = round((time.time() - gen_started) * 1000.0, 2)
            summary_started = time.time()
            response = await self._summarize_response_for_chat(
                ctx=ctx,
                user_input=user_input,
                response=response,
                plan=plan,
            )
            summary_latency_ms = round((time.time() - summary_started) * 1000.0, 2)
            total_latency_ms = round((time.time() - turn_started) * 1000.0, 2)
            ctx.metadata["latency_ms"] = {
                "intent_extract": intent_latency_ms,
                "response_generate": gen_latency_ms,
                "response_summarize": summary_latency_ms,
                "total_turn": total_latency_ms,
            }

            # 4. Persist response
            memory.add_message(role="assistant", content=response)
            ctx.state = ConversationState.IDLE
            self._episodic_memory.record_episode(
                task_description=user_input,
                actions_taken=self._extract_actions_from_context(ctx),
                outcome=response[:300],
                success=True,
                duration=0.0,
                learned_facts=[],
                metadata={
                    "category": "conversation_turn",
                    "session_id": session_id,
                    "user_id": ctx.user_id,
                    "intent": ctx.intent,
                    "modality": modality,
                },
            )

            return response

        except Exception as exc:  # noqa: BLE001
            ctx.state = ConversationState.ERROR
            self._episodic_memory.record_episode(
                task_description=user_input,
                actions_taken=self._extract_actions_from_context(ctx),
                outcome=f"error: {exc}",
                success=False,
                duration=0.0,
                learned_facts=[f"Conversation failure for intent {ctx.intent}"],
                metadata={
                    "category": "conversation_turn",
                    "session_id": session_id,
                    "user_id": ctx.user_id,
                    "intent": ctx.intent,
                    "modality": modality,
                },
                error=str(exc),
            )
            logger.error("process_input error (session=%s): %s", session_id, exc)
            return "I encountered an error processing your request. Please try again."

    # ------------------------------------------------------------------
    # Intent extraction
    # ------------------------------------------------------------------

    def extract_intent(
        self, text: str
    ) -> tuple[str, dict[str, list[str]], float]:
        """
        Rule-based intent classification.

        Returns (intent, entities, confidence). Falls back to
        ``"general_query"`` when no rule matches.
        """
        best_intent = "general_query"
        best_confidence = 0.4

        for pattern, intent, base_conf in _COMPILED_RULES:
            if pattern.search(text):
                if base_conf > best_confidence:
                    best_intent = intent
                    best_confidence = base_conf

        entities = _extract_entities(text)
        return best_intent, entities, best_confidence

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    async def generate_response(
        self,
        ctx: ConversationContext,
        user_input: str,
        *,
        plan: ResponsePlan | None = None,
        modality: str = "text",
        media: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Build a contextual reply for *ctx.intent* using a dynamic per-turn plan.

        Priority:
        1. Planned direct handler (time/weather/code/email)
        2. Hybrid model router (task-aware)
        3. LLM handler (fallback)
        4. Knowledge base (when allowed)
        5. Template or generic fallback
        """
        turn_plan = plan or self._plan_response(
            ctx=ctx,
            user_input=user_input,
            modality=modality,
            media=media or {},
            context=context or {},
        )

        direct = await self._run_direct_handler(turn_plan, ctx, user_input)
        if direct:
            return direct

        # 1. Hybrid model-router path
        if turn_plan.use_model_router and self._model_router and self._model_router.has_provider():
            try:
                memory = self._session_mgr.get(ctx.session_id)
                history_text = ""
                if memory:
                    window = memory.get_context_window(max_messages=8)
                    history_text = "\n".join(
                        f"{m['role']}: {m['content']}" for m in window[:-1]
                    )
                prompt = self._build_llm_prompt(
                    ctx=ctx,
                    user_input=user_input,
                    history=history_text,
                    plan=turn_plan,
                )
                fused_context: dict[str, Any] | None = None
                if turn_plan.require_context_fusion:
                    profile_summary = self.get_user_profile_summary(ctx.user_id)
                    if profile_summary:
                        prompt = f"{prompt}\nUser profile context: {profile_summary}"
                    profile_hints = self._build_profile_prompt_hints(ctx.user_id)
                    if profile_hints:
                        prompt = f"{prompt}\nPersonalization hints: {profile_hints}"
                    fused_context = self._build_context_fusion(
                        ctx=ctx,
                        user_input=user_input,
                        modality=modality,
                        media=media or {},
                        context=context or {},
                    )
                    prompt = f"{prompt}\nContext fusion: {fused_context}"

                request = ModelRequest(
                    prompt=prompt,
                    task_type=turn_plan.task_type,
                    modality=modality,
                    media=media or {},
                    privacy_level=self._infer_privacy_level(ctx, user_input),
                    max_latency_ms=turn_plan.max_latency_ms,
                    prefer_local=turn_plan.prefer_local,
                    metadata={
                        "session_id": ctx.session_id,
                        "user_id": ctx.user_id,
                        "context_fusion": fused_context or {},
                        "user_input": user_input,
                        "fast_path": bool(turn_plan.intent in self._FAST_PATH_INTENTS),
                        "response_plan": turn_plan.to_dict(),
                    },
                )
                policy_decision = (context or {}).get("policy_decision", {}) if isinstance(context, dict) else {}
                if isinstance(policy_decision, dict):
                    if policy_decision.get("prefer_local", None) is not None:
                        request.prefer_local = bool(policy_decision.get("prefer_local"))
                    if policy_decision.get("max_latency_ms", None) is not None:
                        try:
                            request.max_latency_ms = int(policy_decision.get("max_latency_ms"))
                        except Exception:
                            pass
                    request.metadata["policy_decision"] = dict(policy_decision)
                routed = await self._model_router.generate(request)
                if routed.text:
                    route_decision = routed.metadata.get("route_decision", {})
                    ctx.metadata["model_route"] = {
                        "provider_name": routed.provider_name,
                        "latency_ms": routed.latency_ms,
                        "decision": route_decision,
                    }
                    if "shadow" in routed.metadata:
                        ctx.metadata["route_shadow"] = routed.metadata["shadow"]
                    if fused_context is not None:
                        ctx.metadata["context_fusion"] = fused_context
                    return str(routed.text).strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hybrid model router failed: %s", exc)
                self._record_route_failure(ctx, stage="model_router", error=exc, plan=turn_plan)

        # 2. LLM path
        if self._llm_handler:
            try:
                memory = self._session_mgr.get(ctx.session_id)
                history_text = ""
                if memory:
                    window = memory.get_context_window(max_messages=8)
                    history_text = "\n".join(
                        f"{m['role']}: {m['content']}" for m in window[:-1]
                    )
                prompt = self._build_llm_prompt(
                    ctx=ctx,
                    user_input=user_input,
                    history=history_text,
                    plan=turn_plan,
                )
                profile_summary = self.get_user_profile_summary(ctx.user_id)
                if profile_summary:
                    prompt = f"{prompt}\nUser profile context: {profile_summary}"
                profile_hints = self._build_profile_prompt_hints(ctx.user_id)
                if profile_hints:
                    prompt = f"{prompt}\nPersonalization hints: {profile_hints}"
                planning_hints = self._episodic_memory.recommend_actions_for_task(user_input, top_n=3)
                if planning_hints:
                    prompt = f"{prompt}\nLearned planning hints: {', '.join(planning_hints)}"
                response = await self._llm_handler(prompt)
                if response:
                    return str(response).strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM handler failed: %s", exc)
                self._record_route_failure(ctx, stage="llm_handler", error=exc, plan=turn_plan)

        # 3. Knowledge base lookup
        if turn_plan.allow_kb_lookup and ctx.intent in ("memory_query", "information_query"):
            kb_response = self._kb_lookup(user_input, ctx=ctx)
            if kb_response:
                return kb_response

        # 4. Template-based responses
        templates = self._response_templates.get(ctx.intent, [])
        if templates and turn_plan.target_length == "short":
            import random
            return random.choice(templates)

        # 5. Fallback
        profile_hint = self.get_user_profile_summary(ctx.user_id)
        if profile_hint and ctx.intent in ("general_query", "help_request"):
            return f"{profile_hint}. {self._generic_fallback(ctx, user_input)}"
        return self._generic_fallback(ctx, user_input)

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def update_context(
        self,
        session_id: str,
        **kwargs: Any,
    ) -> ConversationContext:
        """
        Update arbitrary fields on the session context.

        Valid keys: ``state``, ``current_topic``, ``intent``, ``entities``,
        ``confidence``, ``metadata``.
        """
        ctx = self._get_context(session_id)
        for key, value in kwargs.items():
            if hasattr(ctx, key):
                setattr(ctx, key, value)
        ctx.touch()
        return ctx

    def get_context(self, session_id: str) -> ConversationContext | None:
        return self._contexts.get(session_id)

    # ------------------------------------------------------------------
    # Session history
    # ------------------------------------------------------------------

    def get_session_history(
        self,
        session_id: str,
        *,
        n: int = 20,
    ) -> list[dict[str, Any]]:
        """Return the last *n* messages for *session_id*."""
        memory = self._session_mgr.get(session_id)
        if memory is None:
            return []
        # get_history returns Message objects; slice to last n
        messages = memory.get_history()[-n:]
        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "message_id": m.message_id,
            }
            for m in messages
        ]

    def list_active_sessions(self) -> list[dict[str, Any]]:
        """Return summary info for all open sessions."""
        self._expire_idle_sessions()
        return [ctx.to_dict() for ctx in self._contexts.values()]

    def session_count(self) -> int:
        return len(self._contexts)

    # ------------------------------------------------------------------
    # LLM injection
    # ------------------------------------------------------------------

    def set_llm_handler(self, handler: Any) -> None:
        """
        Provide an async callable ``(prompt: str) -> str`` used for response
        generation when no rule-based answer is available.
        """
        self._llm_handler = handler

    def set_model_router(self, model_router: ModelRouter | None) -> None:
        """Configure the hybrid local/API model router."""
        self._model_router = model_router

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_context(self, session_id: str) -> ConversationContext:
        ctx = self._contexts.get(session_id)
        if ctx is None:
            # Auto-create a session for unknown IDs (convenience)
            logger.debug("Auto-creating session for id=%s", session_id)
            self.start_session(session_id=session_id)
            ctx = self._contexts[session_id]
        return ctx

    def _kb_lookup(self, text: str, *, ctx: ConversationContext | None = None) -> str | None:
        """Try a keyword search in the knowledge base and format a reply."""
        try:
            ranked = self._kb.search_semantic(text, max_results=5)
            ranked = [
                r for r in ranked
                if float(r.get("score", 0.0)) >= self._kb_min_confidence
                and self._is_fresh(r.get("updated_at"))
            ]
            if ranked:
                facts = "; ".join(
                    f"{r['key']} (conf={r['score']:.2f}): {str(r['value'])[:120]}"
                    for r in ranked
                )
                if ctx is not None:
                    ctx.metadata["kb_match_count"] = len(ranked)
                    ctx.metadata["kb_thresholds"] = {
                        "min_confidence": self._kb_min_confidence,
                        "max_age_days": self._kb_max_age_days,
                    }
                return f"Based on what I know: {facts}."

            results = self._kb.search(text, max_results=3)
            if results:
                facts = "; ".join(
                    f"{e.key}: {str(e.value)[:120]}" for e in results
                )
                return f"Based on what I know: {facts}."
        except Exception as exc:  # noqa: BLE001
            logger.debug("KB lookup error: %s", exc)
        return None

    def _build_context_fusion(
        self,
        *,
        ctx: ConversationContext,
        user_input: str,
        modality: str,
        media: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        continuity = ctx.metadata.get("cross_modal", {})
        if not isinstance(continuity, dict):
            continuity = {}
        return {
            "intent": ctx.intent,
            "topic": ctx.current_topic,
            "turn": ctx.turn_count,
            "modality": modality,
            "has_media": bool(media),
            "media_keys": sorted(list(media.keys())),
            "entities": ctx.entities,
            "external_context_keys": sorted(list(context.keys())),
            "profile": self.get_user_profile_summary(ctx.user_id),
            "continuity": continuity,
        }

    def _is_fresh(self, updated_at: Any) -> bool:
        if not isinstance(updated_at, str) or not updated_at:
            return False
        try:
            ts = datetime.fromisoformat(updated_at)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds() / 86400.0
            return age_days <= float(self._kb_max_age_days)
        except ValueError:
            return False

    @staticmethod
    def _extract_actions_from_context(ctx: ConversationContext) -> list[str]:
        actions: list[str] = []
        route = ctx.metadata.get("model_route", {})
        if isinstance(route, dict):
            provider = route.get("provider_name")
            if isinstance(provider, str) and provider:
                actions.append(f"model:{provider}")
        if ctx.intent:
            actions.append(f"intent:{ctx.intent}")
        return actions

    @staticmethod
    def _build_llm_prompt(
        ctx: ConversationContext,
        user_input: str,
        history: str,
        plan: ResponsePlan | None = None,
    ) -> str:
        if plan is None:
            inferred_intent = str(ctx.intent or "general_query")
            inferred_target = "short" if inferred_intent in {"time_query", "weather_query", "greeting", "farewell"} else "medium"
            plan = ResponsePlan(
                intent=inferred_intent,
                task_type=inferred_intent,
                complexity=0.35,
                target_length=inferred_target,
            )
        style_rules = (
            "Sound natural and human.",
            "Reply with the final answer only.",
            "Use natural conversational text.",
            "Do not include internal reasoning.",
            "Never output internal analysis, planning steps, or labels like 'Thinking Process'.",
            "Avoid markdown headings/lists unless the user explicitly asks for them.",
        )
        length_rule = (
            "Keep replies concise by default (1-3 sentences)."
            if plan.target_length == "short"
            else "Use a complete answer with practical detail."
        )
        parts = [
            "You are JARVIS, an intelligent AI assistant.",
            *style_rules,
            length_rule,
            f"Conversation intent: {ctx.intent}",
            f"Planned task type: {plan.task_type}",
        ]
        if plan.task_type == "weather_query":
            parts.append(
                "Weather guidance: reply in 1-2 sentences with a direct answer; "
                "ask at most one brief follow-up only if location or date is missing."
            )
        if ctx.current_topic:
            parts.append(f"Current topic: {ctx.current_topic}")
        if history:
            parts.append(f"Recent conversation:\n{history}")
        parts.append(f"User message: {user_input}")
        parts.append("Assistant response:")
        return "\n".join(parts)

    def _plan_response(
        self,
        *,
        ctx: ConversationContext,
        user_input: str,
        modality: str,
        media: dict[str, Any],
        context: dict[str, Any],
    ) -> ResponsePlan:
        intent = str(ctx.intent or "general_query")
        low = str(user_input or "").lower()
        complexity = self._estimate_request_complexity(
            user_input=user_input,
            modality=modality,
            media=media,
            context=context,
        )
        task_type = intent
        handler_key = ""
        preserve_format = False
        require_context_fusion = bool(media or context)
        allow_kb_lookup = intent in {"memory_query", "information_query"}
        prefer_local: bool | None = None
        max_latency_ms: int | None = None

        wants_code = bool(
            re.search(r"\b(write|create|generate|build|make|draft|compose)\b", low)
            and re.search(r"\b(code|react|reactjs|javascript|python|component|function|class|jsx|tsx)\b", low)
        )
        wants_email = bool(
            re.search(r"\b(write|draft|compose|create|generate)\b", low)
            and ("email" in low or "mail" in low)
        )
        if intent in {"time_query", "weather_query"}:
            handler_key = intent
            task_type = intent
            prefer_local = True
            max_latency_ms = 4500
        elif wants_email:
            handler_key = "email_draft"
            task_type = "writing"
            preserve_format = True
            allow_kb_lookup = False
            complexity = max(complexity, 0.58)
        elif wants_code:
            handler_key = "code_draft"
            task_type = "coding"
            preserve_format = True
            allow_kb_lookup = False
            complexity = max(complexity, 0.62)
        elif intent == "task_execution":
            task_type = "writing" if re.search(r"\b(write|draft|compose)\b", low) else "general"
            allow_kb_lookup = False
            require_context_fusion = True
            complexity = max(complexity, 0.56)
        elif intent in {"greeting", "farewell", "acknowledgement", "confirmation", "negation", "help_request"}:
            prefer_local = True
            max_latency_ms = 8000

        target_length = self._target_length_for_request(
            user_input=user_input,
            intent=intent,
            complexity=complexity,
            preserve_format=preserve_format,
        )
        if target_length == "short" and max_latency_ms is None:
            max_latency_ms = 8000
        use_model_router = True
        if handler_key in {"time_query", "weather_query"}:
            use_model_router = False

        return ResponsePlan(
            intent=intent,
            task_type=task_type,
            complexity=complexity,
            target_length=target_length,
            handler_key=handler_key,
            preserve_format=preserve_format,
            use_model_router=use_model_router,
            allow_kb_lookup=allow_kb_lookup,
            require_context_fusion=require_context_fusion,
            prefer_local=prefer_local,
            max_latency_ms=max_latency_ms,
        )

    @staticmethod
    def _estimate_request_complexity(
        *,
        user_input: str,
        modality: str,
        media: dict[str, Any],
        context: dict[str, Any],
    ) -> float:
        text = str(user_input or "")
        low = text.lower()
        score = 0.20
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        score += min(0.30, len(words) / 100.0)
        score += min(0.10, text.count("\n") * 0.03)
        if any(k in low for k in ("analyze", "compare", "design", "architecture", "plan", "strategy")):
            score += 0.25
        if any(k in low for k in ("code", "implement", "refactor", "debug", "optimize", "react", "python")):
            score += 0.22
        if modality != "text" or media or context:
            score += 0.18
        return max(0.0, min(1.0, score))

    @staticmethod
    def _target_length_for_request(
        *,
        user_input: str,
        intent: str,
        complexity: float,
        preserve_format: bool,
    ) -> str:
        low = str(user_input or "").lower()
        if preserve_format:
            return "medium"
        if any(k in low for k in ("detailed", "step by step", "explain", "why", "how")):
            return "long"
        if intent in {"greeting", "farewell", "acknowledgement", "confirmation", "negation", "time_query", "weather_query"}:
            return "short"
        if complexity >= 0.75:
            return "long"
        if complexity >= 0.50:
            return "medium"
        return "short"

    async def _run_direct_handler(
        self,
        plan: ResponsePlan,
        ctx: ConversationContext,
        user_input: str,
    ) -> str:
        if not plan.handler_key:
            return ""
        handler = self._response_handlers.get(plan.handler_key)
        if not handler:
            return ""
        try:
            return str(await handler(ctx, user_input)).strip()
        except Exception as exc:  # noqa: BLE001
            self._record_route_failure(ctx, stage=f"direct_handler:{plan.handler_key}", error=exc, plan=plan)
            return ""

    async def _handle_time_query(self, _ctx: ConversationContext, user_input: str) -> str:
        low = user_input.lower()
        now = datetime.now()
        if re.search(r"\b(day|weekday)\b", low):
            if "tomorrow" in low:
                return f"Tomorrow is {(now + timedelta(days=1)).strftime('%A')}."
            if "yesterday" in low:
                return f"Yesterday was {(now - timedelta(days=1)).strftime('%A')}."
            return f"Today is {now.strftime('%A')}."
        if re.search(r"\b(date|today|tomorrow|yesterday)\b", low):
            if "tomorrow" in low:
                return f"Tomorrow is {(now + timedelta(days=1)).strftime('%A, %B %d, %Y')}."
            if "yesterday" in low:
                return f"Yesterday was {(now - timedelta(days=1)).strftime('%A, %B %d, %Y')}."
            return f"Today is {now.strftime('%A, %B %d, %Y')}."
        return f"It is currently {now.strftime('%H:%M')}."

    async def _handle_weather_query(self, _ctx: ConversationContext, user_input: str) -> str:
        location = self._extract_weather_location(user_input)
        if not location:
            return "Please share a city so I can check the weather."
        weather_line = await self._quick_weather_lookup(location)
        if weather_line:
            return weather_line
        if self._weather_network_unavailable():
            return (
                f"I can't access live weather from this environment right now, "
                f"so I can't fetch the current conditions for {location} yet."
            )
        return f"I couldn't fetch live weather for {location} at the moment. Please try again shortly."

    async def _handle_email_draft(self, ctx: ConversationContext, user_input: str) -> str:
        drafted = self._rule_based_email_draft(user_input)
        return drafted if drafted else self._generic_fallback(ctx, user_input)

    async def _handle_code_draft(self, ctx: ConversationContext, user_input: str) -> str:
        drafted = self._rule_based_code_draft(user_input)
        return drafted if drafted else self._generic_fallback(ctx, user_input)

    @staticmethod
    def _record_route_failure(
        ctx: ConversationContext,
        *,
        stage: str,
        error: Exception,
        plan: ResponsePlan,
    ) -> None:
        failures = ctx.metadata.get("route_failures", [])
        if not isinstance(failures, list):
            failures = []
        turn_failures = ctx.metadata.get("route_failures_turn", [])
        if not isinstance(turn_failures, list):
            turn_failures = []
        failure_record = {
            "stage": stage,
            "error": str(error),
            "task_type": plan.task_type,
            "handler_key": plan.handler_key,
            "ts": time.time(),
        }
        failures.append(
            failure_record
        )
        turn_failures.append(failure_record)
        ctx.metadata["route_failures"] = failures[-12:]
        ctx.metadata["route_failures_turn"] = turn_failures[-8:]

    def get_user_profile_summary(self, user_id: str) -> str:
        return self._profile_store.summary(user_id)

    def _learn_user_profile(self, user_id: str, user_input: str) -> None:
        text = user_input.strip()
        low = text.lower()
        if not text:
            return

        # Simple pattern extraction for personalization.
        m = re.search(r"\bmy favorite (\w+)\s+is\s+([a-zA-Z0-9 _-]+)", low)
        if m:
            pref_key = f"favorite_{m.group(1)}"
            self._profile_store.update_preferences(user_id, **{pref_key: m.group(2).strip()})
            return

        m2 = re.search(r"\bi prefer\s+([a-zA-Z0-9 _-]+)", low)
        if m2:
            self._profile_store.update_preferences(user_id, preferred_style=m2.group(1).strip())
            return

        m4 = re.search(r"\b(?:use|set)\s+(?:a\s+)?(concise|detailed|formal|casual)\s+tone\b", low)
        if m4:
            self._profile_store.update_preferences(user_id, tone=m4.group(1).strip())
            return

        m5 = re.search(r"\b(?:notify|remind)\s+me\s+(hourly|daily|weekly)\b", low)
        if m5:
            self._profile_store.update_preferences(user_id, cadence=m5.group(1).strip())
            return

        m6 = re.search(r"\brisk tolerance\s+(?:is|to)\s+(low|medium|high)\b", low)
        if m6:
            self._profile_store.update_preferences(user_id, risk_tolerance=m6.group(1).strip())
            return

        m7 = re.search(r"\bmy routine is\s+([a-zA-Z0-9 ,:_-]+)", text, re.IGNORECASE)
        if m7:
            self._profile_store.update_preferences(user_id, routine=m7.group(1).strip())
            return

        m3 = re.search(r"\bcall me\s+([a-zA-Z0-9 _-]+)", text, re.IGNORECASE)
        if m3:
            self._profile_store.update_traits(user_id, display_name=m3.group(1).strip())

    def _generic_fallback(self, ctx: ConversationContext, user_input: str) -> str:
        direct_email = self._rule_based_email_draft(user_input)
        if direct_email:
            return direct_email
        direct_code = self._rule_based_code_draft(user_input)
        if direct_code:
            return direct_code
        low = str(user_input or "").lower()
        if re.search(r"\b(repo|repository|codebase|project)\b", low) and re.search(
            r"\b(read|scan|analy[sz]e|understand|explain|summari[sz]e|review)\b",
            low,
        ):
            return (
                "I can analyze the current repository. "
                "Ask with `/repo --workspace /absolute/path <question>` or include workspace context."
            )
        profile = self._profile_store.get_or_create(ctx.user_id)
        tone = str(profile.preferences.get("tone", "")).strip().lower()
        if len(user_input) < 5:
            if tone == "formal":
                return "Could you please elaborate a bit more?"
            return "Could you elaborate a bit more?"
        if "?" in user_input:
            if tone == "concise":
                return "Good question. I need more context to answer accurately."
            return (
                "That's a good question. I don't have a specific answer right now, "
                "but I'm always learning."
            )
        if tone == "concise":
            return f"Understood. You are asking about '{ctx.current_topic or 'this'}'. How should I proceed?"
        return (
            f"I understand you're talking about '{ctx.current_topic or 'something'}'. "
            "How can I help you with that?"
        )

    @staticmethod
    def _rule_based_email_draft(user_input: str) -> str:
        text = str(user_input or "").strip()
        low = text.lower()
        if not re.search(r"\b(write|draft|compose|create|generate)\b", low):
            return ""
        if "email" not in low and "mail" not in low:
            return ""

        item_match = re.search(
            r"\b(?:buy|purchase|get)\s+(?:a|an|the)?\s*([a-z][a-z0-9\s\-]{1,50}?)(?:\s+for\s+me\b|[.!?,]|$)",
            low,
            re.IGNORECASE,
        )
        item = "item"
        if item_match:
            item = item_match.group(1).strip(" .,!?\t\r\n")

        personal_recipients = {
            "mom", "mother", "dad", "father", "brother", "sister", "wife", "husband",
            "friend", "uncle", "aunt", "grandma", "grandmother", "grandpa", "grandfather",
            "parents", "family",
        }
        work_recipients = {"manager", "boss", "team", "procurement", "finance", "hr", "lead", "supervisor"}
        personal_pat = r"|".join(sorted(personal_recipients, key=len, reverse=True))
        work_pat = r"|".join(sorted(work_recipients, key=len, reverse=True))

        recipient = ""
        personal_match = re.search(rf"\bto\s+(?:my\s+)?({personal_pat})\b", low, re.IGNORECASE)
        work_match = re.search(rf"\bto\s+(?:my\s+)?({work_pat})\b", low, re.IGNORECASE)
        if personal_match:
            recipient = str(personal_match.group(1)).strip().lower()
        elif work_match:
            recipient = str(work_match.group(1)).strip().lower()
        else:
            generic_recipient_match = re.search(
                r"\bto\s+(?:my\s+)?([a-z][a-z0-9'\-]{1,20})\b",
                low,
                re.IGNORECASE,
            )
            if generic_recipient_match:
                recipient = str(generic_recipient_match.group(1)).strip().lower()

        is_personal = bool(recipient and any(token in personal_recipients for token in recipient.split()))
        is_work = bool(recipient and any(token in work_recipients for token in recipient.split()))

        item_title = " ".join(part.capitalize() for part in item.split()) if item else "Item"
        if is_personal:
            greet = "Mom" if recipient in {"mom", "mother"} else " ".join(part.capitalize() for part in recipient.split())
            return (
                f"Subject: Can You Help Me Buy {item_title}?\n\n"
                f"Hi {greet},\n\n"
                f"I hope you're doing well. I wanted to ask if you could please help me buy a {item}. "
                "It would really help me with my work and learning.\n\n"
                "Please let me know what you think.\n\n"
                "Love,\n"
                "[Your Name]"
            )

        if is_work:
            return (
                f"Subject: Request to Purchase {item_title}\n\n"
                "Hi [Manager Name],\n\n"
                f"I would like to request approval to purchase {item} for our use. "
                "This will help improve day-to-day work and productivity.\n\n"
                "Please let me know if I can proceed, and I can share options within budget.\n\n"
                "Thanks,\n"
                "[Your Name]"
            )

        return (
            f"Subject: Request to Purchase {item_title}\n\n"
            "Hi there,\n\n"
            f"I wanted to ask if you could help me buy a {item}. "
            "It would really help me, and I would appreciate your support.\n\n"
            "Please let me know if this is possible.\n\n"
            "Thanks,\n"
            "[Your Name]"
        )

    @staticmethod
    def _rule_based_code_draft(user_input: str) -> str:
        text = str(user_input or "").strip()
        low = text.lower()
        asks_to_generate = bool(re.search(r"\b(write|create|generate|build|make|draft|compose)\b", low))
        asks_react_component = bool(
            re.search(r"\b(react|reactjs|jsx|tsx)\b", low)
            and re.search(r"\b(component|functional component|function component)\b", low)
        )
        if not (asks_to_generate and asks_react_component):
            return ""

        name = "MyComponent"
        m = re.search(r"\b(?:called|named)\s+([A-Za-z][A-Za-z0-9_]*)\b", text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if candidate:
                name = candidate[0].upper() + candidate[1:]
        return (
            "```jsx\n"
            "import React from \"react\";\n\n"
            f"function {name}() {{\n"
            "  return <div>Hello from React</div>;\n"
            "}\n\n"
            f"export default {name};\n"
            "```"
        )

    async def _summarize_response_for_chat(
        self,
        *,
        ctx: ConversationContext,
        user_input: str,
        response: str,
        plan: ResponsePlan | None = None,
    ) -> str:
        raw = str(response or "").strip()
        if not raw:
            ctx.metadata["summary_stage"] = {"used": False, "source": "empty", "latency_ms": 0.0}
            return ""
        summarize_started = time.time()
        if self._should_preserve_response_format(ctx=ctx, user_input=user_input, response=raw):
            ctx.metadata["summary_stage"] = {
                "used": False,
                "source": "passthrough_preserve_format",
                "latency_ms": round((time.time() - summarize_started) * 1000.0, 2),
            }
            return raw
        if plan and plan.target_length == "long" and not self._looks_like_meta_response(raw):
            ctx.metadata["summary_stage"] = {
                "used": False,
                "source": "passthrough_target_long",
                "latency_ms": round((time.time() - summarize_started) * 1000.0, 2),
            }
            return raw
        if not self._needs_response_summary(raw):
            ctx.metadata["summary_stage"] = {
                "used": False,
                "source": "passthrough",
                "latency_ms": round((time.time() - summarize_started) * 1000.0, 2),
            }
            return raw
        reject_reason = ""
        attempts = 0
        for strict_retry in (False, True):
            model_summary = await self._summarize_response_with_light_model(
                ctx=ctx,
                user_input=user_input,
                response=raw,
                strict_retry=strict_retry,
            )
            if not model_summary:
                continue
            attempts += 1
            extracted = self._extract_summary_payload(str(model_summary))
            reject_reason = self._summary_reject_reason(extracted, user_input=user_input)
            if not reject_reason:
                ctx.metadata["summary_stage"] = {
                    "used": True,
                    "source": "model",
                    "attempts": attempts,
                    "latency_ms": round((time.time() - summarize_started) * 1000.0, 2),
                }
                return extracted

        salvaged = self._summarize_response_for_chat_rule(raw)
        if salvaged and salvaged != raw:
            ctx.metadata["summary_stage"] = {
                "used": False,
                "source": "fallback_rule",
                "attempts": attempts,
                "reject_reason": reject_reason or "model_unavailable",
                "latency_ms": round((time.time() - summarize_started) * 1000.0, 2),
            }
            return salvaged
        ctx.metadata["summary_stage"] = {
            "used": False,
            "source": "fallback_generic",
            "attempts": attempts,
            "reject_reason": reject_reason or "model_unavailable",
            "latency_ms": round((time.time() - summarize_started) * 1000.0, 2),
        }
        return self._generic_fallback(ctx, user_input)

    async def _summarize_response_with_light_model(
        self,
        *,
        ctx: ConversationContext,
        user_input: str,
        response: str,
        strict_retry: bool = False,
    ) -> str:
        if not (self._model_router and self._model_router.has_provider()):
            return ""
        prompt = (
            "You are a response summarizer. Return JSON only.\n"
            "Rewrite the assistant output into a short, natural reply.\n"
            "Constraints:\n"
            "- Keep factual meaning.\n"
            "- 1-2 sentences max.\n"
            "- Plain text only.\n"
            "- No lists, tables, headings, or meta-commentary.\n"
            "- Do not include internal reasoning or analysis.\n"
            "- Output strictly as JSON: {\"final\":\"...\"}\n"
            f"User request: {user_input}\n"
            f"Assistant draft response:\n{response}\n"
            "JSON:"
        )
        if strict_retry:
            prompt = (
                "Previous output was invalid. Output JSON only with key 'final'. "
                "Do not include any extra text.\n" + prompt
            )
        try:
            routed = await self._model_router.generate(
                ModelRequest(
                    prompt=prompt,
                    task_type="summarization",
                    modality="text",
                    privacy_level=self._infer_privacy_level(ctx, user_input),
                    prefer_local=True,
                    max_latency_ms=2500,
                    metadata={
                        "session_id": ctx.session_id,
                        "user_id": ctx.user_id,
                        "stage": "response_summary",
                    },
                )
            )
            return str(routed.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Light-model summarization skipped: %s", exc)
            return ""

    @staticmethod
    def _extract_summary_payload(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        if re.match(r"(?is)^\s*user request\s*:", raw):
            return ""
        parsed = ConversationManager._parse_summary_json(raw)
        if parsed:
            return ConversationManager._trim_trailing_meta_noise(parsed)
        marker = "Final concise response:"
        if marker in raw:
            tail = raw.split(marker)[-1].strip()
            if tail:
                return tail.strip("\"' ")
        if re.match(r"(?is)^\s*Thinking Process(?:\s*:)?\s*$", raw):
            return ""
        if re.match(r"(?is)^\s*Thinking Process\s*:", raw):
            quoted = re.findall(r'"([^"\n]{12,400})"', raw)
            if quoted:
                return quoted[-1].strip()
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            candidates = [
                ln for ln in lines
                if not re.match(r"^\*+\s*", ln)
                and not re.match(r"^\d+\.\s+", ln)
                and ":" not in ln[:40]
            ]
            if candidates:
                return candidates[-1].strip("\"' ")
            return ""
        return ConversationManager._trim_trailing_meta_noise(raw)

    @staticmethod
    def _parse_summary_json(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        candidates = [raw]
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            candidates.append(m.group(0))
        for c in candidates:
            try:
                obj = json.loads(c)
            except Exception:
                continue
            if isinstance(obj, dict):
                final = str(obj.get("final", "")).strip()
                if final:
                    return final
        return ""

    @staticmethod
    def _trim_trailing_meta_noise(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return ""

        noise_markers = (
            "thinking process",
            "analyze the request",
            "response policy",
            "refine the output",
            "constraint check",
            "current intent",
            "user request",
            "assistant draft response",
            "assistant response",
        )
        kept: list[str] = []
        for ln in lines:
            low = ln.strip().lower()
            if any(m in low for m in noise_markers):
                break
            if re.match(r"^\s*\d+\.\s+", ln):
                # section/list indicator after answer usually means reasoning tail
                break
            kept.append(ln.strip())
        if not kept:
            first_low = lines[0].strip().lower()
            if any(m in first_low for m in noise_markers):
                return ""
        cleaned = " ".join(kept).strip() if kept else raw
        cleaned = re.sub(r"\s+", " ", cleaned)
        # Keep at most first 2 sentence-like chunks for natural chat style.
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
        if parts:
            cleaned = " ".join(parts[:2]).strip()
        return cleaned

    @staticmethod
    def _needs_response_summary(text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        meta_noise = any(
            tok in raw
            for tok in (
                "Thinking Process:",
                "Thinking Process",
                "Reasoning/Explanation",
                "Analyze the Request:",
                "Response Policy:",
                "User request:",
                "Assistant draft response:",
            )
        )
        return (
            meta_noise
            or len(raw) > 360
            or raw.count("\n") >= 8
        )

    @staticmethod
    def _should_preserve_response_format(
        *,
        ctx: ConversationContext,
        user_input: str,
        response: str,
    ) -> bool:
        low_q = str(user_input or "").lower()
        low_r = str(response or "").lower()
        is_email_request = bool(
            re.search(r"\b(write|draft|compose|create|generate)\b", low_q)
            and ("email" in low_q or "mail" in low_q)
        )
        is_code_request = bool(
            re.search(r"\b(write|draft|compose|create|generate|build|make)\b", low_q)
            and re.search(r"\b(code|component|react|reactjs|javascript|js|python|function|class|jsx|tsx)\b", low_q)
        )
        looks_structured_email = (
            "subject:" in low_r
            and re.search(r"(?im)^\s*(hi|hello|dear)\b", response) is not None
            and ("\n" in response)
        )
        looks_structured_code = (
            "```" in response
            or bool(re.search(r"(?m)^\s*import\s+\w+", response))
            or bool(re.search(r"(?m)^\s*export\s+default\s+", response))
            or bool(re.search(r"(?m)^\s*(const|function)\s+[A-Za-z_][A-Za-z0-9_]*\s*(=|\()", response))
        )
        if is_email_request and looks_structured_email:
            return True
        if ctx.intent == "task_execution" and looks_structured_email:
            return True
        if is_code_request and looks_structured_code:
            return True
        if ctx.intent == "task_execution" and looks_structured_code:
            return True
        return False

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        s = str(text or "").lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _is_bad_summary_candidate(candidate: str, *, user_input: str) -> bool:
        return bool(ConversationManager._summary_reject_reason(candidate, user_input=user_input))

    @staticmethod
    def _summary_reject_reason(candidate: str, *, user_input: str) -> str:
        c = str(candidate or "").strip()
        if not c:
            return "empty_candidate"
        c_norm = ConversationManager._normalize_for_match(c)
        q_norm = ConversationManager._normalize_for_match(user_input)
        if not c_norm:
            return "blank_normalized"
        prefixed_q = re.sub(r"^(user request|request|query)\s*:\s*", "", c_norm).strip()
        if q_norm and prefixed_q and prefixed_q == q_norm:
            return "echo_question_prefixed"
        # Reject only near-exact echo of the user question.
        if q_norm and c_norm == q_norm:
            return "echo_question_exact"
        if q_norm and c_norm.rstrip(" ?.!") == q_norm.rstrip(" ?.!"):
            return "echo_question_normalized"
        meta_markers = (
            "thinking process",
            "analyze the request",
            "response policy",
            "current intent",
            "constraint check",
            "user request",
            "assistant draft response",
            "assistant response",
        )
        if any(m in c_norm for m in meta_markers):
            return "meta_marker"
        return ""

    @staticmethod
    def _extract_weather_location(user_input: str) -> str:
        text = str(user_input or "").strip()
        if not text:
            return ""
        m = re.search(r"\b(?:in|at|for)\s+([A-Za-z][A-Za-z\s\-'.,]{1,60})\??\s*$", text, re.IGNORECASE)
        if not m:
            return ""
        location = re.sub(r"[?.!,]+$", "", m.group(1)).strip()
        location = re.sub(
            r"\b(right\s+now|now|today|tonight|this\s+morning|this\s+evening)\b\s*$",
            "",
            location,
            flags=re.IGNORECASE,
        ).strip(" ,.-")
        return location

    async def _quick_weather_lookup(self, location: str) -> str:
        def _fetch() -> str:
            headers = {
                "User-Agent": "JARVIS-AI-OS/1.0 (+weather-fetch)",
                "Accept": "application/json",
            }
            wttr = self._fetch_weather_wttr(location, headers=headers)
            if wttr:
                return wttr
            return self._fetch_weather_open_meteo(location, headers=headers)

        try:
            return await asyncio.to_thread(_fetch)
        except Exception:
            return ""

    def _fetch_weather_wttr(self, location: str, *, headers: dict[str, str]) -> str:
        encoded = quote(location)
        urls = [
            f"https://wttr.in/{encoded}?format=j1",
            f"http://wttr.in/{encoded}?format=j1",
        ]
        for url in urls:
            try:
                req = Request(url, headers=headers)
                with urlopen(req, timeout=5.0) as resp:  # nosec B310
                    payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
                current = (payload.get("current_condition") or [{}])[0]
                temp_c = str(current.get("temp_C", "")).strip()
                desc = ""
                desc_list = current.get("weatherDesc") or []
                if isinstance(desc_list, list) and desc_list:
                    desc = str((desc_list[0] or {}).get("value", "")).strip()
                if temp_c and desc:
                    return f"In {location}, it is {temp_c}°C with {desc.lower()} right now."
                if temp_c:
                    return f"In {location}, it is {temp_c}°C right now."
            except Exception:
                continue
        return ""

    def _fetch_weather_open_meteo(self, location: str, *, headers: dict[str, str]) -> str:
        encoded = quote(location)
        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
            f"?name={encoded}&count=1&language=en&format=json"
        )
        try:
            with urlopen(Request(geo_url, headers=headers), timeout=5.0) as resp:  # nosec B310
                geo_payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            results = geo_payload.get("results") or []
            if not results:
                return ""
            first = results[0] or {}
            lat = first.get("latitude")
            lon = first.get("longitude")
            if lat is None or lon is None:
                return ""
            name = str(first.get("name", "")).strip() or location
            country = str(first.get("country", "")).strip()
            display = f"{name}, {country}" if country and country.lower() not in name.lower() else name

            weather_url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&current=temperature_2m,weather_code&timezone=auto"
            )
            with urlopen(Request(weather_url, headers=headers), timeout=5.0) as resp:  # nosec B310
                weather_payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            current = weather_payload.get("current") or {}
            temp_val = current.get("temperature_2m")
            code = current.get("weather_code")
            if temp_val is None:
                return ""
            temp_num = float(temp_val)
            temp = str(int(round(temp_num))) if abs(temp_num - round(temp_num)) < 0.05 else f"{temp_num:.1f}"
            desc = self._open_meteo_code_to_desc(code)
            if desc:
                return f"In {display}, it is {temp}°C with {desc} right now."
            return f"In {display}, it is {temp}°C right now."
        except Exception:
            return ""

    @staticmethod
    def _weather_network_unavailable() -> bool:
        hosts = ("wttr.in", "geocoding-api.open-meteo.com", "api.open-meteo.com")
        ok = 0
        for host in hosts:
            try:
                socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
                ok += 1
            except Exception:
                continue
        return ok == 0

    @staticmethod
    def _open_meteo_code_to_desc(code: Any) -> str:
        try:
            c = int(code)
        except Exception:
            return ""
        mapping = {
            0: "clear skies",
            1: "mostly clear skies",
            2: "partly cloudy skies",
            3: "overcast skies",
            45: "foggy conditions",
            48: "rime fog",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",
            56: "light freezing drizzle",
            57: "dense freezing drizzle",
            61: "light rain",
            63: "moderate rain",
            65: "heavy rain",
            66: "light freezing rain",
            67: "heavy freezing rain",
            71: "light snowfall",
            73: "moderate snowfall",
            75: "heavy snowfall",
            77: "snow grains",
            80: "light rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",
            85: "light snow showers",
            86: "heavy snow showers",
            95: "a thunderstorm",
            96: "thunderstorms with light hail",
            99: "thunderstorms with heavy hail",
        }
        return mapping.get(c, "")

    @staticmethod
    def _looks_like_meta_response(text: str) -> bool:
        s = str(text or "").strip().lower()
        if not s:
            return True
        meta_markers = (
            "analyze the request:",
            "response policy:",
            "role: jarvis",
            "current intent:",
            "thinking process:",
            "reasoning/explanation",
            "internal reasoning",
            "meta-commentary",
            "user request:",
            "assistant draft response:",
        )
        return any(m in s for m in meta_markers)

    @staticmethod
    def _summarize_response_for_chat_rule(
        text: str,
        *,
        max_sentences: int = 2,
        max_chars: int = 320,
    ) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""

        needs_summary = (
            len(raw) > max_chars
            or raw.count("\n") >= 3
            or any(
                tok in raw
                for tok in (
                    "Thinking Process:",
                    "Reasoning/Explanation",
                    "Analyze the Request:",
                    "Response Policy:",
                    "User request:",
                    "Assistant draft response:",
                    "|---",
                    "**",
                )
            )
        )
        if not needs_summary:
            return raw

        lines = []
        for ln in raw.splitlines():
            s = ln.strip()
            if not s:
                continue
            if re.match(
                r"(?i)^(thinking process|reasoning/explanation|analyze the request|response policy|role|current intent|user request|assistant draft response)\s*:?",
                s,
            ):
                continue
            if s.startswith(("#", "|")):
                continue
            if re.match(r"^\s*[-*]\s+", s):
                s = re.sub(r"^\s*[-*]\s+", "", s).strip()
            if re.match(r"^\s*\d+\.\s+", s):
                s = re.sub(r"^\s*\d+\.\s+", "", s).strip()
            s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
            lines.append(s)

        if not lines:
            return ""

        cleaned = " ".join(lines).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            return ""

        sentence_parts = [
            seg.strip() for seg in re.split(r"(?<=[.!?])\s+", cleaned) if seg.strip()
        ]
        summary = " ".join(sentence_parts[: max(1, int(max_sentences))]).strip()
        if not summary:
            summary = cleaned
        if len(summary) > max_chars:
            summary = summary[: max_chars - 1].rstrip() + "…"
        return summary

    def _update_modality_context(
        self,
        *,
        ctx: ConversationContext,
        modality: str,
        media: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        continuity = ctx.metadata.get("cross_modal")
        if not isinstance(continuity, dict):
            continuity = {
                "last_modality": "none",
                "modality_history": [],
                "switch_count": 0,
                "last_media_keys": [],
            }
        last_modality = str(continuity.get("last_modality", "none"))
        if last_modality != "none" and last_modality != modality:
            continuity["switch_count"] = int(continuity.get("switch_count", 0)) + 1
            continuity["last_transition"] = f"{last_modality}->{modality}"
        history = continuity.get("modality_history", [])
        if not isinstance(history, list):
            history = []
        history.append(modality)
        continuity["modality_history"] = history[-8:]
        continuity["last_modality"] = modality
        continuity["last_media_keys"] = sorted(list(media.keys()))
        continuity["external_context_keys"] = sorted(list(context.keys()))
        ctx.metadata["cross_modal"] = continuity

    def _build_profile_prompt_hints(self, user_id: str) -> str:
        profile = self._profile_store.get(user_id)
        if profile is None or not profile.preferences:
            return ""
        preferred_keys = ("tone", "cadence", "risk_tolerance", "routine", "preferred_style")
        hints: list[str] = []
        for key in preferred_keys:
            if key in profile.preferences:
                hints.append(f"{key}={profile.preferences[key]}")
        return ", ".join(hints)

    @staticmethod
    def _infer_privacy_level(ctx: ConversationContext, user_input: str) -> PrivacyLevel:
        low = user_input.lower()
        if ctx.intent in {"status_query", "greeting", "acknowledgement"}:
            return PrivacyLevel.LOW
        sensitive_markers = (
            "password",
            "ssn",
            "social security",
            "bank account",
            "private",
            "confidential",
            "secret",
            "token",
            "api key",
        )
        if any(marker in low for marker in sensitive_markers):
            return PrivacyLevel.HIGH
        return PrivacyLevel.MEDIUM

    def _expire_idle_sessions(self) -> None:
        """Remove sessions that have been idle longer than ``_SESSION_TIMEOUT``."""
        now = time.time()
        expired = [
            sid for sid, ctx in self._contexts.items()
            if (now - ctx.last_activity) > self._SESSION_TIMEOUT
        ]
        for sid in expired:
            logger.debug("Expiring idle session %s", sid)
            self.end_session(sid)

    @staticmethod
    def _build_response_templates() -> dict[str, list[str]]:
        return {
            "greeting": [
                "Hello! How can I assist you today?",
                "Hi there! What can I do for you?",
                "Hey! I'm JARVIS. What do you need?",
            ],
            "farewell": [
                "Goodbye! Have a great day.",
                "See you later!",
                "Farewell! Don't hesitate to return if you need anything.",
            ],
            "help_request": [
                "I can help you search the web, manage files, run system commands, "
                "answer questions, and much more. Just ask!",
                "My capabilities include information retrieval, task execution, "
                "system monitoring, and conversational assistance.",
            ],
            "status_query": [
                "All systems are operational.",
                "I'm functioning normally and ready to assist.",
            ],
            "acknowledgement": [
                "You're welcome!",
                "Happy to help!",
                "Anytime!",
            ],
            "confirmation": ["Understood, proceeding.", "Got it!"],
            "negation": ["Okay, I'll cancel that.", "Understood, stopping."],
            "time_query": [
                f"The current time is {__import__('datetime').datetime.now().strftime('%H:%M')}."
            ],
            "cancel": [
                "Operation cancelled.",
                "Understood, I'll stop that.",
            ],
        }

    def __repr__(self) -> str:
        return f"<ConversationManager sessions={len(self._contexts)} llm={'yes' if self._llm_handler else 'no'}>"
