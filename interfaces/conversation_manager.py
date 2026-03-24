"""
Conversation state management for JARVIS AI OS.

Handles multi-turn sessions, intent extraction (rule-based + LLM), contextual
response generation, and integration with ConversationMemory and KnowledgeBase.
"""

from __future__ import annotations

import json
import os
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
from infrastructure.slo_metrics import get_slo_metrics
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


@dataclass
class QueryUnderstanding:
    """Structured per-turn understanding beyond keyword intent."""
    inferred_intent: str = "general_query"
    user_goal: str = ""
    constraints: list[str] = field(default_factory=list)
    missing_constraints: list[str] = field(default_factory=list)
    ambiguity_score: float = 0.0
    requires_retrieval: bool = False
    recommended_route: str = "direct"
    response_depth: str = "medium"
    confidence: float = 0.0
    should_ask_clarification: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "inferred_intent": self.inferred_intent,
            "user_goal": self.user_goal,
            "constraints": list(self.constraints),
            "missing_constraints": list(self.missing_constraints),
            "ambiguity_score": round(float(self.ambiguity_score), 3),
            "requires_retrieval": bool(self.requires_retrieval),
            "recommended_route": str(self.recommended_route or "direct"),
            "response_depth": str(self.response_depth or "medium"),
            "confidence": round(float(self.confidence), 3),
            "should_ask_clarification": bool(self.should_ask_clarification),
        }


@dataclass
class QueryDecomposition:
    """Structured decomposition for multi-part user queries."""
    should_decompose: bool = False
    sub_questions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_decompose": bool(self.should_decompose),
            "sub_questions": [str(s).strip() for s in self.sub_questions if str(s).strip()],
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
        }


@dataclass
class RealtimeFrame:
    source: str
    summary: str
    ts: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "summary": self.summary,
            "ts": float(self.ts),
            "metadata": dict(self.metadata),
        }


@dataclass
class RealtimeSessionState:
    session_id: str
    user_id: str
    active: bool = True
    interrupt_epoch: int = 0
    frames: list[RealtimeFrame] = field(default_factory=list)
    max_frames: int = 12
    last_activity: float = field(default_factory=time.time)
    last_interrupt_reason: str = ""

    def touch(self) -> None:
        self.last_activity = time.time()

    def add_frame(self, frame: RealtimeFrame) -> None:
        self.frames.append(frame)
        if len(self.frames) > int(self.max_frames):
            self.frames = self.frames[-int(self.max_frames):]
        self.touch()

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "active": bool(self.active),
            "interrupt_epoch": int(self.interrupt_epoch),
            "last_activity": float(self.last_activity),
            "last_interrupt_reason": self.last_interrupt_reason,
            "frame_count": len(self.frames),
            "recent_frames": [f.to_dict() for f in self.frames[-5:]],
        }


@dataclass
class ReferenceResolution:
    """Resolved follow-up reference for deictic turns (that/those/it/them)."""
    used: bool = False
    resolved_query: str = ""
    antecedent: str = ""
    confidence: float = 0.0
    source: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "used": bool(self.used),
            "resolved_query": self.resolved_query,
            "antecedent": self.antecedent,
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
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
_KNOWN_INTENTS: set[str] = {intent for _, intent, _ in _INTENT_RULES} | {"general_query"}
_REFERENCE_FOLLOWUP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(and|also|so|then)\b", re.IGNORECASE),
    re.compile(r"\b(that|those|it|them|this|these|same)\b", re.IGNORECASE),
    re.compile(r"\b(tell me more|more about|go deeper|continue|elaborate)\b", re.IGNORECASE),
    re.compile(r"^\s*(ok|okay|sure|yes|yep|yeah)\b", re.IGNORECASE),
)

_TOPIC_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "to", "of", "for", "in", "on", "at", "by",
    "with", "from", "as", "is", "are", "was", "were", "be", "being", "been", "can", "could", "would", "should",
    "will", "shall", "may", "might", "must", "do", "does", "did", "have", "has", "had", "tell", "explain",
    "about", "please", "you", "me", "my", "your", "our", "we", "i", "it", "this", "that", "these", "those",
    "what", "why", "how", "when", "where", "who", "whom", "which", "hi", "hello", "hey",
    "learn", "understand", "know",
    "other", "another", "something", "anything", "thing", "things", "stuff", "library", "libraries",
    "ok", "okay", "sure", "yes", "yep", "yeah",
    "research", "investigate", "analyze", "analyse", "study",
}

_KNOWLEDGE_FALLBACK_SNIPPETS: dict[str, str] = {
    "ai": (
        "AI is the field of building systems that can learn patterns, reason over data, "
        "and automate tasks such as language, vision, and decision support."
    ),
    "artificialintelligence": (
        "Artificial intelligence focuses on creating models and agents that perform tasks "
        "requiring perception, prediction, planning, and language understanding."
    ),
    "react": (
        "React is a JavaScript library for building user interfaces with reusable components "
        "and state-driven rendering."
    ),
    "reactjs": (
        "ReactJS is a JavaScript library for building user interfaces with reusable components "
        "and state-driven rendering."
    ),
    "nextjs": (
        "Next.js is a React framework for full-stack web apps with routing, server rendering, "
        "and API endpoints."
    ),
    "nodejs": (
        "Node.js is a JavaScript runtime for running backend services and scripts outside the browser."
    ),
    "python": (
        "Python is a high-level programming language used across web development, automation, "
        "data engineering, and AI."
    ),
}

_LIBRARY_DETAIL_SNIPPETS: dict[str, str] = {
    "react": "React is best for component-driven UIs and stateful frontend apps.",
    "next.js": "Next.js adds routing, server rendering, and full-stack patterns on top of React.",
    "tailwind": "Tailwind CSS is a utility-first styling system that speeds up UI development.",
    "node.js": "Node.js is used for backend APIs, real-time services, and JavaScript tooling.",
    "fastapi": "FastAPI is a high-performance Python framework for typed API services.",
    "pandas": "Pandas is the standard Python library for tabular data analysis and ETL tasks.",
    "langgraph": "LangGraph helps build stateful, multi-step agent workflows with explicit control flow.",
}


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
        self._slo_metrics = get_slo_metrics()
        self._understanding_enabled = str(
            os.getenv("JARVIS_QUERY_UNDERSTANDING_ENABLED", "true")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._adaptive_planning_enabled = str(
            os.getenv("JARVIS_ADAPTIVE_PLANNING_ENABLED", "true")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._confidence_retrieval_enabled = str(
            os.getenv("JARVIS_CONFIDENCE_RETRIEVAL_ENABLED", "true")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._memory_semantics_enabled = str(
            os.getenv("JARVIS_MEMORY_SEMANTICS_ENABLED", "true")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._eval_telemetry_enabled = str(
            os.getenv("JARVIS_EVAL_TELEMETRY_ENABLED", "true")
        ).strip().lower() in {"1", "true", "yes", "on"}

        # session_id -> ConversationContext
        self._contexts: dict[str, ConversationContext] = {}
        self._realtime_sessions: dict[str, RealtimeSessionState] = {}
        self._response_handlers: dict[str, Callable[[ConversationContext, str], Awaitable[str]]] = {
            "time_query": self._handle_time_query,
            "weather_query": self._handle_weather_query,
            "email_draft": self._handle_email_draft,
            "code_draft": self._handle_code_draft,
        }
        self._dual_tier_response_enabled = str(
            os.getenv("JARVIS_DUAL_TIER_RESPONSE_ENABLED", "false")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._single_pass_summary_enabled = str(
            os.getenv("JARVIS_RESPONSE_SINGLE_PASS_ENABLED", "true")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._history_window_short = max(
            2,
            int(str(os.getenv("JARVIS_RESPONSE_HISTORY_WINDOW_SHORT", "4")).strip() or "4"),
        )
        self._history_window_medium = max(
            self._history_window_short,
            int(str(os.getenv("JARVIS_RESPONSE_HISTORY_WINDOW_MEDIUM", "6")).strip() or "6"),
        )
        self._history_window_long = max(
            self._history_window_medium,
            int(str(os.getenv("JARVIS_RESPONSE_HISTORY_WINDOW_LONG", "8")).strip() or "8"),
        )

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
        self._realtime_sessions.pop(session_id, None)
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
        media_payload: dict[str, Any] = dict(media or {})
        context_payload: dict[str, Any] = dict(context or {})
        rt_state = self._realtime_sessions.get(session_id)
        rt_epoch_at_start = int(rt_state.interrupt_epoch) if rt_state else 0
        self._attach_realtime_live_context(
            session_id=session_id,
            media=media_payload,
            context=context_payload,
        )
        ctx.state = ConversationState.PROCESSING
        ctx.turn_count += 1
        ctx.touch()
        self._update_modality_context(
            ctx=ctx,
            modality=modality,
            media=media_payload,
            context=context_payload,
        )
        ctx.metadata["low_latency_turn"] = bool(
            self._is_low_latency_turn(modality=modality, context=context_payload)
        )
        if isinstance(context_payload, dict):
            req_env = context_payload.get("request_envelope")
            if isinstance(req_env, dict):
                ctx.metadata["request_envelope_last"] = req_env

        # Store in memory
        memory = self._session_mgr.get_or_create(session_id)
        memory.add_message(role="user", content=user_input)

        try:
            # Reset turn-local routing failure buffer.
            ctx.metadata["route_failures_turn"] = []

            # 1. Resolve follow-up references using turn memory semantics.
            resolution_started = time.time()
            reference_resolution = await self._resolve_reference_from_history(
                ctx=ctx,
                user_input=user_input,
            )
            ref_resolution_latency_ms = round((time.time() - resolution_started) * 1000.0, 2)
            ctx.metadata["reference_resolution"] = reference_resolution.to_dict()
            analysis_input = (
                reference_resolution.resolved_query
                if reference_resolution.used and reference_resolution.resolved_query
                else user_input
            )

            # 2. Intent extraction
            intent_started = time.time()
            intent, entities, confidence = self.extract_intent(analysis_input)
            intent_latency_ms = round((time.time() - intent_started) * 1000.0, 2)
            ctx.intent = intent
            ctx.entities = entities
            ctx.confidence = confidence
            logger.debug(
                "Session %s — intent='%s' confidence=%.2f", session_id, intent, confidence
            )
            quick_response = self._fast_intent_short_circuit(
                ctx=ctx,
                user_input=user_input,
                analysis_input=analysis_input,
                modality=modality,
                media=media_payload,
                context=context_payload,
            )
            if quick_response:
                memory.add_message(role="assistant", content=quick_response)
                ctx.state = ConversationState.IDLE
                return quick_response
            if self._is_affirmative_continuation(analysis_input.lower()):
                directives = self._extract_output_directives(analysis_input)
                if directives:
                    ctx.metadata["response_preferences"] = directives
                pending_reply = self._consume_pending_action(ctx, analysis_input, directives=directives)
                if pending_reply:
                    memory.add_message(role="assistant", content=pending_reply)
                    ctx.state = ConversationState.IDLE
                    return pending_reply

            # 2. Query understanding (model-assisted with heuristic fallback).
            understanding = await self._infer_query_understanding(
                ctx=ctx,
                user_input=analysis_input,
                rule_intent=intent,
                entities=entities,
                modality=modality,
                media=media_payload,
                context=context_payload,
            )
            ctx.metadata["query_understanding"] = understanding.to_dict()
            if (
                understanding.inferred_intent in _KNOWN_INTENTS
                and understanding.confidence >= 0.72
                and (
                    intent == "general_query"
                    or (understanding.inferred_intent != intent and understanding.confidence >= (confidence + 0.12))
                )
            ):
                ctx.intent = understanding.inferred_intent
                ctx.confidence = max(ctx.confidence, understanding.confidence)

            # 3. Topic tracking (salient token/phrase heuristic).
            inferred_topic = self._infer_topic_from_text(analysis_input)
            if inferred_topic:
                ctx.current_topic = inferred_topic

            # 3.05 Query decomposition for multi-part prompts.
            decomposition = await self._infer_query_decomposition(
                ctx=ctx,
                user_input=analysis_input,
                understanding=understanding,
                intent=ctx.intent or intent,
            )
            ctx.metadata["query_decomposition"] = decomposition.to_dict()
            if self._memory_semantics_enabled:
                self._update_turn_memory_semantics(
                    ctx=ctx,
                    user_input=analysis_input,
                    understanding=understanding,
                    decomposition=decomposition,
                )

            # 3.1 Learning loop: extract explicit user preferences.
            self._learn_user_profile(ctx.user_id, user_input)

            # 3.2 Dynamic response plan.
            plan = self._plan_response(
                ctx=ctx,
                user_input=analysis_input,
                modality=modality,
                media=media_payload,
                context=context_payload,
            )
            ctx.metadata["response_plan"] = plan.to_dict()
            ctx.metadata["response_policy"] = self._derive_response_policy(
                user_input=analysis_input,
                plan=plan,
                query_understanding=ctx.metadata.get("query_understanding", {}),
            )

            # 4. Clarification-first path when understanding is low-confidence/incomplete.
            clarification = self._build_clarification_prompt(
                understanding=understanding,
                user_input=user_input,
            )
            if clarification:
                missing = set(understanding.missing_constraints or [])
                if "level" in missing or "goal" in missing:
                    ctx.metadata["learning_plan_pending"] = True
                    ctx.metadata.setdefault("learning_plan_slots", {})
                self._record_turn_telemetry(
                    ctx=ctx,
                    outcome="clarification",
                    user_input=analysis_input,
                )
                memory.add_message(role="assistant", content=clarification)
                ctx.state = ConversationState.IDLE
                return clarification

            # 5. Response generation
            ctx.state = ConversationState.RESPONDING
            gen_started = time.time()
            response = await self.generate_response(
                ctx,
                analysis_input,
                plan=plan,
                modality=modality,
                media=media_payload,
                context=context_payload,
            )
            if self._is_realtime_turn_interrupted(session_id, rt_epoch_at_start):
                interrupted = "Understood. Interrupted the previous response. Please continue."
                memory.add_message(role="assistant", content=interrupted)
                ctx.state = ConversationState.IDLE
                return interrupted
            gen_latency_ms = round((time.time() - gen_started) * 1000.0, 2)
            summary_started = time.time()
            if self._should_single_pass_response(ctx=ctx, plan=plan, response=response):
                ctx.metadata["summary_stage"] = {
                    "used": False,
                    "source": "single_pass",
                    "latency_ms": 0.0,
                }
            else:
                response = await self._summarize_response_for_chat(
                    ctx=ctx,
                    user_input=user_input,
                    response=response,
                    plan=plan,
                )
            self._refresh_pending_action_from_response(
                ctx=ctx,
                user_input=analysis_input,
                response=response,
            )
            summary_latency_ms = round((time.time() - summary_started) * 1000.0, 2)
            total_latency_ms = round((time.time() - turn_started) * 1000.0, 2)
            ctx.metadata["latency_ms"] = {
                "intent_extract": intent_latency_ms,
                "reference_resolution": ref_resolution_latency_ms,
                "response_generate": gen_latency_ms,
                "response_summarize": summary_latency_ms,
                "total_turn": total_latency_ms,
            }

            # 6. Persist response
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
            self._record_turn_telemetry(
                ctx=ctx,
                outcome="ok",
                user_input=analysis_input,
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
            self._record_turn_telemetry(
                ctx=ctx,
                outcome="error",
                user_input=user_input,
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

        # Demote greeting when the same utterance carries an actionable ask.
        if best_intent == "greeting":
            reconciled_intent, reconciled_conf = self._reconcile_greeting_with_request(text=text)
            if reconciled_intent != best_intent:
                best_intent = reconciled_intent
                best_confidence = max(best_confidence, reconciled_conf)

        entities = _extract_entities(text)
        return best_intent, entities, best_confidence

    @staticmethod
    def _reconcile_greeting_with_request(*, text: str) -> tuple[str, float]:
        raw = str(text or "").strip()
        if not raw:
            return "greeting", 0.95
        remainder = re.sub(
            r"^\s*(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening))[\s,!.:;-]*",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip()
        if not remainder:
            return "greeting", 0.95
        if len(remainder.split()) < 3:
            return "greeting", 0.95
        low = remainder.lower()
        if re.search(
            r"\b(can|could|would)\s+you\b|\b(help|learn|teach|explain|show|build|write|create|generate|debug|fix)\b",
            low,
        ):
            return "task_execution", 0.88
        if re.search(r"\b(what|why|how|when|where|who)\b", low):
            return "information_query", 0.80
        return "greeting", 0.95

    async def _infer_query_understanding(
        self,
        *,
        ctx: ConversationContext,
        user_input: str,
        rule_intent: str,
        entities: dict[str, list[str]],
        modality: str,
        media: dict[str, Any],
        context: dict[str, Any],
    ) -> QueryUnderstanding:
        heuristic = self._infer_query_understanding_heuristic(
            user_input=user_input,
            rule_intent=rule_intent,
            entities=entities,
        )
        if bool(ctx.metadata.get("low_latency_turn")):
            return heuristic
        if rule_intent in {"time_query", "weather_query"}:
            return heuristic
        if not (self._model_router and self._model_router.has_provider()):
            return heuristic
        if not self._understanding_enabled:
            return heuristic
        prompt = (
            "Extract query understanding as strict JSON only.\n"
            "Return keys: inferred_intent, user_goal, constraints, missing_constraints, ambiguity_score, "
            "requires_retrieval, recommended_route, response_depth, confidence, should_ask_clarification.\n"
            "Rules:\n"
            "- inferred_intent must be one of: greeting,farewell,help_request,status_query,task_execution,information_query,"
            "memory_query,configuration,cancel,acknowledgement,confirmation,negation,weather_query,time_query,general_query.\n"
            "- constraints/missing_constraints must be JSON string arrays.\n"
            "- ambiguity_score is 0..1 (higher means user intent/constraints are underspecified).\n"
            "- recommended_route must be one of: direct,clarify,retrieve,decompose,plan.\n"
            "- response_depth must be one of: short,medium,long.\n"
            "- confidence is 0..1.\n"
            "- should_ask_clarification true only when missing info blocks a good answer.\n"
            f"User message: {user_input}\n"
            f"Rule intent: {rule_intent}\n"
            f"Known entities: {json.dumps(entities, ensure_ascii=False)}\n"
            f"Modality: {modality}\n"
            f"Has media: {bool(media)}\n"
            f"Context keys: {sorted(list((context or {}).keys()))}\n"
            "JSON:"
        )
        try:
            routed = await self._model_router.generate(
                ModelRequest(
                    prompt=prompt,
                    task_type="summarization",
                    modality="text",
                    privacy_level=self._infer_privacy_level(ctx, user_input),
                    prefer_local=True,
                    max_latency_ms=1800,
                    metadata={
                        "session_id": ctx.session_id,
                        "user_id": ctx.user_id,
                        "stage": "query_understanding",
                    },
                )
            )
            parsed = self._parse_query_understanding(str(routed.text or ""))
            if parsed is None:
                return heuristic
            # Conservative merge: keep model when confident, otherwise heuristic.
            if parsed.confidence >= max(0.70, heuristic.confidence + 0.05):
                return parsed
            return heuristic
        except Exception:
            return heuristic

    @staticmethod
    def _parse_query_understanding(text: str) -> QueryUnderstanding | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        candidates = [raw]
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            candidates.append(m.group(0))
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            intent = str(obj.get("inferred_intent", "general_query")).strip() or "general_query"
            if intent not in _KNOWN_INTENTS:
                intent = "general_query"
            goal = str(obj.get("user_goal", "")).strip()
            constraints = obj.get("constraints", [])
            missing = obj.get("missing_constraints", [])
            if not isinstance(constraints, list):
                constraints = []
            if not isinstance(missing, list):
                missing = []
            try:
                ambiguity = float(obj.get("ambiguity_score", 0.0))
            except Exception:
                ambiguity = 0.0
            requires_retrieval = bool(obj.get("requires_retrieval", False))
            recommended_route = str(obj.get("recommended_route", "direct")).strip().lower() or "direct"
            if recommended_route not in {"direct", "clarify", "retrieve", "decompose", "plan"}:
                recommended_route = "direct"
            response_depth = str(obj.get("response_depth", "medium")).strip().lower() or "medium"
            if response_depth not in {"short", "medium", "long"}:
                response_depth = "medium"
            try:
                conf = float(obj.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            ask = bool(obj.get("should_ask_clarification", False))
            return QueryUnderstanding(
                inferred_intent=intent,
                user_goal=goal,
                constraints=[str(x).strip() for x in constraints if str(x).strip()],
                missing_constraints=[str(x).strip() for x in missing if str(x).strip()],
                ambiguity_score=max(0.0, min(1.0, ambiguity)),
                requires_retrieval=requires_retrieval,
                recommended_route=recommended_route,
                response_depth=response_depth,
                confidence=max(0.0, min(1.0, conf)),
                should_ask_clarification=ask,
            )
        return None

    @staticmethod
    def _infer_query_understanding_heuristic(
        *,
        user_input: str,
        rule_intent: str,
        entities: dict[str, list[str]],
    ) -> QueryUnderstanding:
        low = str(user_input or "").strip().lower()
        goal = ""
        constraints: list[str] = []
        missing: list[str] = []
        ask = False
        conf = 0.55 if rule_intent != "general_query" else 0.42
        requires_retrieval = False
        recommended_route = "direct"
        response_depth = "medium"
        if re.search(r"\b(for|toward|to get)\s+(job|interview|project)\b", low):
            m = re.search(r"\b(for|toward|to get)\s+(job|interview|project)\b", low)
            if m:
                goal = m.group(2)
                constraints.append(f"goal={goal}")
                conf += 0.08
        if re.search(r"\b(beginner|intermediate|advanced)\b", low):
            m = re.search(r"\b(beginner|intermediate|advanced)\b", low)
            if m:
                constraints.append(f"level={m.group(1)}")
                conf += 0.07
        if re.search(r"\b(teach me|how can i learn|learning path|roadmap)\b", low):
            goal = goal or "learn"
            if not any(c.startswith("level=") for c in constraints):
                missing.append("level")
            if not any(c.startswith("goal=") for c in constraints):
                missing.append("goal")
            ask = bool(missing)
            conf += 0.10
        if ConversationManager._is_why_reasoning_query(low):
            goal = goal or "causal_explanation"
            if not any(c == "mode=why_reasoning" for c in constraints):
                constraints.append("mode=why_reasoning")
            conf += 0.10
            response_depth = "long"

        if rule_intent in {"information_query", "memory_query"} and not ask:
            requires_retrieval = bool(
                re.search(r"\b(latest|recent|source|research|paper|compare|benchmark|trend|news|cite)\b", low)
            )
            if requires_retrieval:
                recommended_route = "retrieve"
                response_depth = "long" if "compare" in low or "benchmark" in low else "medium"

        if ask:
            recommended_route = "clarify"
        elif any(tok in low for tok in (" and ", " also ", " then ", ", and ")) and len(low.split()) >= 12:
            recommended_route = "decompose"
            response_depth = "long"

        ambiguity = 0.0
        ambiguity += 0.35 if rule_intent == "general_query" else 0.15
        ambiguity += min(0.45, 0.16 * len(missing))
        ambiguity += 0.10 if len(low.split()) <= 4 else 0.0
        ambiguity += 0.12 if low in {"okay", "yes", "sure", "go ahead"} else 0.0
        ambiguity = max(0.0, min(1.0, ambiguity))
        return QueryUnderstanding(
            inferred_intent=rule_intent or "general_query",
            user_goal=goal,
            constraints=constraints,
            missing_constraints=missing,
            ambiguity_score=ambiguity,
            requires_retrieval=requires_retrieval,
            recommended_route=recommended_route,
            response_depth=response_depth,
            confidence=max(0.0, min(1.0, conf)),
            should_ask_clarification=ask,
        )

    async def _infer_query_decomposition(
        self,
        *,
        ctx: ConversationContext,
        user_input: str,
        understanding: QueryUnderstanding,
        intent: str,
    ) -> QueryDecomposition:
        heuristic = self._infer_query_decomposition_heuristic(
            user_input=user_input,
            understanding=understanding,
            intent=intent,
        )
        if bool(ctx.metadata.get("low_latency_turn")):
            return heuristic
        if not heuristic.should_decompose:
            return heuristic
        if not (self._model_router and self._model_router.has_provider()):
            return heuristic
        parsed = await self._infer_query_decomposition_via_model(
            ctx=ctx,
            user_input=user_input,
            understanding=understanding,
        )
        if parsed and parsed.should_decompose and parsed.confidence >= max(0.72, heuristic.confidence + 0.05):
            return parsed
        return heuristic

    async def _infer_query_decomposition_via_model(
        self,
        *,
        ctx: ConversationContext,
        user_input: str,
        understanding: QueryUnderstanding,
    ) -> QueryDecomposition | None:
        prompt = (
            "Decompose the query into sub-questions when it has multiple parts.\n"
            "Return strict JSON only with keys: should_decompose, sub_questions, confidence.\n"
            "Rules:\n"
            "- should_decompose true only for multi-part/multi-goal requests.\n"
            "- sub_questions max 5, each concise and standalone.\n"
            "- confidence is 0..1.\n"
            f"Inferred intent: {understanding.inferred_intent}\n"
            f"Inferred goal: {understanding.user_goal}\n"
            f"User query: {user_input}\n"
            "JSON:"
        )
        try:
            routed = await self._model_router.generate(
                ModelRequest(
                    prompt=prompt,
                    task_type="summarization",
                    modality="text",
                    privacy_level=self._infer_privacy_level(ctx, user_input),
                    prefer_local=True,
                    max_latency_ms=1500,
                    metadata={
                        "session_id": ctx.session_id,
                        "user_id": ctx.user_id,
                        "stage": "query_decomposition",
                    },
                )
            )
        except Exception:
            return None
        return self._parse_query_decomposition(str(routed.text or ""), source="model")

    @staticmethod
    def _parse_query_decomposition(text: str, *, source: str) -> QueryDecomposition | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        candidates = [raw]
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            candidates.append(m.group(0))
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            should = bool(obj.get("should_decompose", False))
            raw_sub = obj.get("sub_questions", [])
            if not isinstance(raw_sub, list):
                raw_sub = []
            sub_questions = [str(s).strip() for s in raw_sub if str(s).strip()]
            try:
                conf = float(obj.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            if should and not sub_questions:
                continue
            return QueryDecomposition(
                should_decompose=should and bool(sub_questions),
                sub_questions=sub_questions[:5],
                confidence=max(0.0, min(1.0, conf)),
                source=source,
            )
        return None

    @staticmethod
    def _infer_query_decomposition_heuristic(
        *,
        user_input: str,
        understanding: QueryUnderstanding,
        intent: str,
    ) -> QueryDecomposition:
        text = str(user_input or "").strip()
        low = text.lower()
        if not text:
            return QueryDecomposition(should_decompose=False, confidence=0.0, source="heuristic")
        if intent in {"greeting", "farewell", "acknowledgement", "confirmation", "negation", "time_query", "weather_query"}:
            return QueryDecomposition(should_decompose=False, confidence=0.15, source="heuristic")
        if understanding.should_ask_clarification:
            return QueryDecomposition(should_decompose=False, confidence=0.35, source="heuristic")

        sequenced = bool(re.search(r"\b(then|after that|next|finally|also)\b", low))
        list_like = bool(re.search(r",\s*[^,]+,\s*(and|&)\s+[^,]+", low))
        question_count = low.count("?")
        multi_clause = len(re.split(r"[;?\n]+", text)) >= 3
        verb_count = len(re.findall(r"\b(explain|compare|design|build|write|summarize|teach|plan|debug|review|evaluate|recommend)\b", low))
        if not (sequenced or list_like or question_count > 1 or multi_clause or verb_count >= 2):
            return QueryDecomposition(should_decompose=False, confidence=0.28, source="heuristic")

        pieces = re.split(r"(?:\s+(?:and then|then|after that|next)\s+|[;?\n]+)", text)
        sub_questions: list[str] = []
        for p in pieces:
            s = re.sub(r"\s+", " ", str(p or "").strip(" .,:-")).strip()
            if len(s.split()) < 3:
                continue
            if not s.endswith("?") and re.match(
                r"^(explain|compare|design|build|write|summarize|teach|plan|debug|review|evaluate|recommend)\b",
                s,
                flags=re.IGNORECASE,
            ):
                s = f"{s}?"
            if s and s not in sub_questions:
                sub_questions.append(s)

        if len(sub_questions) < 2 and list_like:
            topics = [t.strip() for t in re.split(r",| and ", text) if t and len(t.strip().split()) <= 5]
            topics = [t for t in topics if t.lower() not in _TOPIC_STOPWORDS][:5]
            for t in topics:
                q = f"Explain {t}."
                if q not in sub_questions:
                    sub_questions.append(q)

        if len(sub_questions) >= 2:
            conf = 0.66 + min(0.22, 0.05 * len(sub_questions))
            return QueryDecomposition(
                should_decompose=True,
                sub_questions=sub_questions[:5],
                confidence=min(0.95, conf),
                source="heuristic",
            )
        return QueryDecomposition(should_decompose=False, confidence=0.33, source="heuristic")

    @staticmethod
    def _build_clarification_prompt(
        *,
        understanding: QueryUnderstanding,
        user_input: str,
    ) -> str:
        if not understanding.should_ask_clarification:
            return ""
        missing = [m for m in understanding.missing_constraints if m]
        if not missing:
            return ""
        low = str(user_input or "").lower()
        if any(m in {"level", "goal"} for m in missing) and re.search(r"\b(learn|teach|roadmap|path)\b", low):
            parts: list[str] = []
            if "level" in missing:
                parts.append("level (beginner/intermediate/advanced)")
            if "goal" in missing:
                parts.append("goal (job/project/interview)")
            if parts:
                joined = " and ".join(parts)
                return f"To tailor this well, share your {joined}."
        if len(missing) == 1:
            return f"I can answer better if you share one detail: {missing[0]}."
        return f"I can answer better if you share: {', '.join(missing)}."

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

        profile_name_set_reply = self._handle_profile_name_set(ctx=ctx, user_input=user_input)
        if profile_name_set_reply:
            return profile_name_set_reply

        profile_name_reply = self._handle_profile_name_query(ctx=ctx, user_input=user_input)
        if profile_name_reply:
            return profile_name_reply

        # 1. Hybrid model-router path
        if turn_plan.use_model_router and self._model_router and self._model_router.has_provider():
            try:
                memory = self._session_mgr.get(ctx.session_id)
                history_text = ""
                if memory:
                    window = memory.get_context_window(
                        max_messages=self._history_window_size(plan=turn_plan, modality=modality, user_input=user_input)
                    )
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

                dual_tier_text = await self._maybe_generate_dual_tier_response(
                    ctx=ctx,
                    user_input=user_input,
                    plan=turn_plan,
                    modality=modality,
                    media=media or {},
                    prompt=prompt,
                    fused_context=fused_context,
                )
                if dual_tier_text:
                    if fused_context is not None:
                        ctx.metadata["context_fusion"] = fused_context
                    return dual_tier_text

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
                        "response_policy": (
                            dict(ctx.metadata.get("response_policy", {}))
                            if isinstance(ctx.metadata.get("response_policy", {}), dict)
                            else {}
                        ),
                        "query_understanding": (
                            dict(ctx.metadata.get("query_understanding", {}))
                            if isinstance(ctx.metadata.get("query_understanding", {}), dict)
                            else {}
                        ),
                        "query_decomposition": (
                            dict(ctx.metadata.get("query_decomposition", {}))
                            if isinstance(ctx.metadata.get("query_decomposition", {}), dict)
                            else {}
                        ),
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
                    window = memory.get_context_window(
                        max_messages=self._history_window_size(plan=turn_plan, modality=modality, user_input=user_input)
                    )
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

    async def _maybe_generate_dual_tier_response(
        self,
        *,
        ctx: ConversationContext,
        user_input: str,
        plan: ResponsePlan,
        modality: str,
        media: dict[str, Any],
        prompt: str,
        fused_context: dict[str, Any] | None,
    ) -> str:
        if not self._dual_tier_response_enabled:
            return ""
        if str(modality or "text").strip().lower() != "text":
            return ""
        if media:
            return ""
        if bool(ctx.metadata.get("low_latency_turn")):
            return ""
        if plan.task_type in {"time_query", "weather_query", "summarization"}:
            return ""
        if plan.handler_key:
            return ""

        base_metadata = {
            "session_id": ctx.session_id,
            "user_id": ctx.user_id,
            "context_fusion": fused_context or {},
            "user_input": user_input,
            "fast_path": bool(plan.intent in self._FAST_PATH_INTENTS),
            "response_plan": plan.to_dict(),
            "response_policy": (
                dict(ctx.metadata.get("response_policy", {}))
                if isinstance(ctx.metadata.get("response_policy", {}), dict)
                else {}
            ),
            "query_understanding": (
                dict(ctx.metadata.get("query_understanding", {}))
                if isinstance(ctx.metadata.get("query_understanding", {}), dict)
                else {}
            ),
            "query_decomposition": (
                dict(ctx.metadata.get("query_decomposition", {}))
                if isinstance(ctx.metadata.get("query_decomposition", {}), dict)
                else {}
            ),
        }

        small_latency = min(6000, int(plan.max_latency_ms or 12000))
        large_prompt = self._build_large_tier_prompt(prompt)

        async def _run_tier(*, tier: str, tier_prompt: str, max_latency_ms: int | None, stage: str) -> str:
            try:
                resp = await self._model_router.generate(
                    ModelRequest(
                        prompt=tier_prompt,
                        task_type=plan.task_type,
                        modality="text",
                        media={},
                        privacy_level=self._infer_privacy_level(ctx, user_input),
                        max_latency_ms=max_latency_ms,
                        prefer_local=True,
                        metadata={**base_metadata, "stage": stage, "model_tier": tier},
                    )
                )
                return str(resp.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dual-tier %s stage failed: %s", tier, exc)
                return ""

        small_task = asyncio.create_task(
            _run_tier(
                tier="small",
                tier_prompt=prompt,
                max_latency_ms=small_latency,
                stage="dual_tier_small",
            )
        )
        large_task = asyncio.create_task(
            _run_tier(
                tier="large",
                tier_prompt=large_prompt,
                max_latency_ms=plan.max_latency_ms,
                stage="dual_tier_large",
            )
        )

        small_text = await small_task
        if not small_text:
            try:
                return await large_task
            finally:
                if not large_task.done():
                    large_task.cancel()

        # Keep UX responsive: if large tier is not ready soon, return small result.
        large_wait_s = 2.5
        if plan.max_latency_ms:
            large_wait_s = max(1.0, min(4.0, float(plan.max_latency_ms) / 1000.0 * 0.35))
        try:
            large_text = await asyncio.wait_for(large_task, timeout=large_wait_s)
        except asyncio.TimeoutError:
            if not large_task.done():
                large_task.cancel()
            return small_text

        if not large_text:
            return small_text
        delta = self._remove_sentence_overlap(base=small_text, candidate=large_text)
        if not delta:
            return small_text
        return f"{small_text}\n\nMore depth:\n{delta}".strip()

    @staticmethod
    def _build_large_tier_prompt(prompt: str) -> str:
        return (
            f"{str(prompt or '').strip()}\n\n"
            "Return a richer second-pass answer with deeper explanation, tradeoffs, and concrete next steps.\n"
            "Avoid repeating introductory lines."
        )

    @staticmethod
    def _remove_sentence_overlap(*, base: str, candidate: str) -> str:
        def _sentences(text: str) -> list[str]:
            parts = re.split(r"(?<=[.!?])\s+|\n+", str(text or "").strip())
            out = [p.strip() for p in parts if p and p.strip()]
            return out

        def _norm(text: str) -> str:
            s = str(text or "").lower()
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        base_set = {_norm(s) for s in _sentences(base)}
        kept: list[str] = []
        for sent in _sentences(candidate):
            n = _norm(sent)
            if not n:
                continue
            if n in base_set:
                continue
            if any(len(n) > 20 and n in b for b in base_set):
                continue
            kept.append(sent)
        return " ".join(kept).strip()

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
    # Realtime session control
    # ------------------------------------------------------------------

    def start_realtime_session(
        self,
        *,
        user_id: str = "anonymous",
        session_id: str | None = None,
        max_frames: int = 12,
    ) -> str:
        sid = session_id or self.get_or_create_session(user_id)
        if sid not in self._contexts:
            self.start_session(user_id=user_id, session_id=sid)
        state = self._realtime_sessions.get(sid)
        if state is None:
            state = RealtimeSessionState(
                session_id=sid,
                user_id=user_id,
                max_frames=max(1, int(max_frames)),
            )
            self._realtime_sessions[sid] = state
        state.active = True
        state.user_id = user_id
        state.max_frames = max(1, int(max_frames))
        state.touch()
        ctx = self._contexts.get(sid)
        if ctx is not None:
            ctx.metadata["realtime"] = state.snapshot()
        return sid

    def stop_realtime_session(self, session_id: str) -> bool:
        state = self._realtime_sessions.get(session_id)
        if state is None:
            return False
        state.active = False
        state.touch()
        ctx = self._contexts.get(session_id)
        if ctx is not None:
            ctx.metadata["realtime"] = state.snapshot()
        return True

    def interrupt_realtime_session(self, session_id: str, *, reason: str = "") -> dict[str, Any]:
        state = self._realtime_sessions.get(session_id)
        if state is None:
            user_id = self._contexts.get(session_id).user_id if session_id in self._contexts else "anonymous"
            self.start_realtime_session(user_id=user_id, session_id=session_id)
            state = self._realtime_sessions[session_id]
        state.interrupt_epoch += 1
        state.last_interrupt_reason = str(reason or "").strip()
        state.touch()
        snap = state.snapshot()
        ctx = self._contexts.get(session_id)
        if ctx is not None:
            ctx.metadata["realtime"] = snap
        return snap

    def ingest_realtime_frame(
        self,
        session_id: str,
        *,
        source: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> dict[str, Any]:
        state = self._realtime_sessions.get(session_id)
        if state is None:
            user_id = self._contexts.get(session_id).user_id if session_id in self._contexts else "anonymous"
            self.start_realtime_session(user_id=user_id, session_id=session_id)
            state = self._realtime_sessions[session_id]
        frame = RealtimeFrame(
            source=str(source or "unknown").strip() or "unknown",
            summary=str(summary or "").strip(),
            ts=float(ts if ts is not None else time.time()),
            metadata=dict(metadata or {}),
        )
        state.add_frame(frame)
        snap = state.snapshot()
        ctx = self._contexts.get(session_id)
        if ctx is not None:
            ctx.metadata["realtime"] = snap
        return snap

    async def summarize_visual_observation(
        self,
        session_id: str,
        *,
        source: str = "camera",
        image_url: str = "",
        image_b64: str = "",
        note: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Summarize a live visual frame using the model router when available.

        Falls back to provided note or a generic placeholder if image analysis
        is unavailable in the current runtime.
        """
        ctx = self._get_context(session_id)
        image_url = str(image_url or "").strip()
        image_b64 = str(image_b64 or "").strip()
        note = str(note or "").strip()
        if not image_url and not image_b64:
            return note or f"Live {source} frame received."

        if not (self._model_router and self._model_router.has_provider()):
            return note or f"Live {source} frame received (no visual model available)."

        prompt = (
            "Summarize this live visual frame for ongoing conversation context.\n"
            "Keep it factual and concise in 1-2 sentences.\n"
            "Mention visible UI/state/signals and likely user focus.\n"
            "Do not include internal reasoning."
        )
        media_payload: dict[str, Any] = {}
        if image_url:
            media_payload["image_url"] = image_url
        if image_b64:
            media_payload["image_b64"] = image_b64
        if isinstance(metadata, dict) and metadata:
            media_payload["frame_metadata"] = dict(metadata)
        try:
            routed = await self._model_router.generate(
                ModelRequest(
                    prompt=prompt,
                    task_type="analysis",
                    modality="image",
                    media=media_payload,
                    privacy_level=PrivacyLevel.MEDIUM,
                    max_latency_ms=1800,
                    metadata={
                        "session_id": session_id,
                        "user_id": ctx.user_id,
                        "stage": "realtime_visual_summary",
                        "source": source,
                    },
                )
            )
            summary = str(routed.text or "").strip()
            if summary:
                return summary
        except Exception:
            pass
        return note or f"Live {source} frame received (visual summary unavailable)."

    def get_realtime_session(self, session_id: str) -> dict[str, Any] | None:
        state = self._realtime_sessions.get(session_id)
        return state.snapshot() if state else None

    def list_realtime_sessions(self) -> list[dict[str, Any]]:
        return [s.snapshot() for s in self._realtime_sessions.values() if s.active]

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

    def _attach_realtime_live_context(
        self,
        *,
        session_id: str,
        media: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        state = self._realtime_sessions.get(session_id)
        if state is None or not state.active:
            return
        state.touch()
        if not state.frames:
            context["realtime"] = state.snapshot()
            return
        latest = state.frames[-1]
        summaries = [f.summary for f in state.frames[-3:] if f.summary]
        context["live_visual_grounding"] = {
            "latest_source": latest.source,
            "latest_summary": latest.summary,
            "recent_summaries": summaries,
            "frame_count": len(state.frames),
        }
        context["realtime"] = state.snapshot()
        media["live_frame_available"] = True

    def _is_realtime_turn_interrupted(self, session_id: str, start_epoch: int) -> bool:
        state = self._realtime_sessions.get(session_id)
        if state is None or not state.active:
            return False
        return int(state.interrupt_epoch) > int(start_epoch)

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
            ranked = self._kb.search_semantic(text, max_results=8)
            selected: list[dict[str, Any]] = []
            dynamic_threshold = float(self._kb_min_confidence)
            if self._confidence_retrieval_enabled and ctx is not None:
                understanding = ctx.metadata.get("query_understanding", {})
                if isinstance(understanding, dict):
                    try:
                        ambiguity = float(understanding.get("ambiguity_score", 0.0) or 0.0)
                    except Exception:
                        ambiguity = 0.0
                    # Harder evidence requirement on ambiguous queries.
                    dynamic_threshold = min(0.72, max(self._kb_min_confidence, self._kb_min_confidence + (ambiguity * 0.25)))
            for r in ranked:
                freshness_ok = self._is_fresh(r.get("updated_at"))
                relevance = float(r.get("score", 0.0) or 0.0)
                trust = self._kb_entry_trust_score(r)
                confidence = (relevance * 0.70) + (trust * 0.30)
                if freshness_ok and confidence >= dynamic_threshold:
                    selected.append(
                        {
                            **r,
                            "retrieval_confidence": round(confidence, 4),
                            "trust_score": round(trust, 4),
                        }
                    )
            if selected:
                facts = "; ".join(
                    f"{r['key']} (conf={r['retrieval_confidence']:.2f}): {str(r['value'])[:120]}"
                    for r in selected[:4]
                )
                if ctx is not None:
                    ctx.metadata["kb_match_count"] = len(selected)
                    ctx.metadata["kb_thresholds"] = {
                        "min_confidence": dynamic_threshold,
                        "max_age_days": self._kb_max_age_days,
                    }
                    ctx.metadata["kb_selected"] = [
                        {
                            "key": str(r.get("key", "")),
                            "category": str(r.get("category", "")),
                            "score": float(r.get("score", 0.0) or 0.0),
                            "trust_score": float(r.get("trust_score", 0.0) or 0.0),
                            "retrieval_confidence": float(r.get("retrieval_confidence", 0.0) or 0.0),
                        }
                        for r in selected[:4]
                    ]
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

    @staticmethod
    def _kb_entry_trust_score(entry: dict[str, Any]) -> float:
        category = str(entry.get("category", "")).strip().lower()
        tags = [str(t).strip().lower() for t in (entry.get("tags") or []) if str(t).strip()]
        score = 0.45
        if category in {"verified", "trusted", "official"}:
            score += 0.35
        elif category in {"community", "unverified", "draft"}:
            score -= 0.15
        if any(t in {"verified", "hf", "official", "trusted"} for t in tags):
            score += 0.20
        if any(t in {"unverified", "non_hf", "community"} for t in tags):
            score -= 0.10
        return max(0.05, min(0.99, score))

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
        rich_context = self._extract_rich_context(context)
        return {
            "intent": ctx.intent,
            "topic": ctx.current_topic,
            "turn": ctx.turn_count,
            "modality": modality,
            "has_media": bool(media),
            "media_keys": sorted(list(media.keys())),
            "entities": ctx.entities,
            "query_understanding": (
                dict(ctx.metadata.get("query_understanding", {}))
                if isinstance(ctx.metadata.get("query_understanding", {}), dict)
                else {}
            ),
            "query_decomposition": (
                dict(ctx.metadata.get("query_decomposition", {}))
                if isinstance(ctx.metadata.get("query_decomposition", {}), dict)
                else {}
            ),
            "external_context_keys": sorted(list(rich_context.keys())),
            "external_context": rich_context,
            "profile": self.get_user_profile_summary(ctx.user_id),
            "continuity": continuity,
        }

    @staticmethod
    def _extract_rich_context(context: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(context, dict):
            return {}
        ignored = {"request_envelope", "policy_decision", "chat_history", "latency_target", "response_shape"}
        out: dict[str, Any] = {}
        for key, value in context.items():
            k = str(key).strip()
            if not k or k in ignored:
                continue
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                out[k] = value
                continue
            if isinstance(value, (list, tuple)):
                out[k] = list(value)[:6]
                continue
            if isinstance(value, dict):
                out[k] = {str(subk): value[subk] for subk in list(value.keys())[:8]}
                continue
            out[k] = str(value)
        return out

    def _has_rich_context(self, context: dict[str, Any]) -> bool:
        return bool(self._extract_rich_context(context))

    @staticmethod
    def _is_low_latency_turn(*, modality: str, context: dict[str, Any]) -> bool:
        mod = str(modality or "").strip().lower()
        if mod == "voice":
            return True
        if not isinstance(context, dict):
            return False
        realtime_flag = context.get("realtime_mode")
        if isinstance(realtime_flag, bool):
            return realtime_flag
        return str(realtime_flag or "").strip().lower() in {"1", "true", "yes", "on"}

    def _history_window_size(self, *, plan: ResponsePlan, modality: str, user_input: str) -> int:
        if modality == "voice" and plan.target_length == "short":
            return self._history_window_short
        word_count = len(str(user_input or "").split())
        if plan.target_length == "short" or word_count <= 16:
            return self._history_window_short
        if plan.target_length == "medium":
            return self._history_window_medium
        return self._history_window_long

    def _fast_intent_short_circuit(
        self,
        *,
        ctx: ConversationContext,
        user_input: str,
        analysis_input: str,
        modality: str,
        media: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        intent = str(ctx.intent or "").strip().lower()
        if intent not in self._FAST_PATH_INTENTS:
            return ""
        if str(modality or "text").strip().lower() != "voice":
            return ""
        if media:
            return ""
        if not isinstance(context, dict):
            return ""
        # Do not short-circuit when external rich context is attached beyond known lightweight keys.
        allowed_ctx_keys = {"request_envelope", "policy_decision", "realtime_mode", "response_shape", "latency_target"}
        if any(str(k) not in allowed_ctx_keys for k in context.keys()):
            return ""
        text = str(analysis_input or user_input or "").strip()
        if not text:
            return ""
        if len(text.split()) > 16:
            return ""
        # Weather/time have dedicated direct handlers and should keep full path.
        if intent in {"time_query", "weather_query"}:
            return ""
        # For voice greetings/acks/confirms/help, return immediately to avoid model latency.
        templates = self._response_templates.get(intent, [])
        if templates:
            return str(templates[0]).strip()
        if intent == "help_request":
            return (
                "I can help with questions, coding, writing, planning, and realtime voice/chat tasks. "
                "Tell me what you want to do."
            )
        if intent == "confirmation":
            return "Understood, proceeding."
        if intent == "acknowledgement":
            return "You are welcome."
        if intent == "greeting":
            return "Hello. How can I help you?"
        if intent == "farewell":
            return "Goodbye."
        if intent == "negation":
            return "Okay, I will pause here."
        return ""

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

    def _update_turn_memory_semantics(
        self,
        *,
        ctx: ConversationContext,
        user_input: str,
        understanding: QueryUnderstanding,
        decomposition: QueryDecomposition,
    ) -> None:
        turn_memory = ctx.metadata.get("turn_memory")
        if not isinstance(turn_memory, dict):
            turn_memory = {}
        slots = turn_memory.get("slots")
        if not isinstance(slots, dict):
            slots = {}
        for item in understanding.constraints:
            token = str(item or "").strip()
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key and value:
                slots[key] = value
        unresolved = [str(x).strip() for x in understanding.missing_constraints if str(x).strip()]
        followups = turn_memory.get("followups")
        if not isinstance(followups, list):
            followups = []
        if decomposition.should_decompose and decomposition.sub_questions:
            followups = [str(x).strip() for x in decomposition.sub_questions[:5] if str(x).strip()]
        turn_memory.update(
            {
                "last_user_input": str(user_input or "").strip(),
                "last_intent": str(ctx.intent or "").strip(),
                "last_topic": str(ctx.current_topic or "").strip(),
                "last_user_goal": str(understanding.user_goal or "").strip(),
                "slots": slots,
                "unresolved_constraints": unresolved,
                "followups": followups,
                "ambiguity_score": round(float(understanding.ambiguity_score), 3),
                "updated_turn": int(ctx.turn_count),
            }
        )
        ctx.metadata["turn_memory"] = turn_memory

    def _record_turn_telemetry(
        self,
        *,
        ctx: ConversationContext,
        outcome: str,
        user_input: str,
    ) -> None:
        if not self._eval_telemetry_enabled or self._slo_metrics is None:
            return
        label_intent = str(ctx.intent or "unknown")
        self._slo_metrics.inc("conversation_turn_total", label=label_intent)
        self._slo_metrics.inc("conversation_turn_outcome_total", label=str(outcome or "unknown"))
        understanding = ctx.metadata.get("query_understanding", {})
        if isinstance(understanding, dict):
            try:
                confidence = float(understanding.get("confidence", 0.0) or 0.0)
                self._slo_metrics.set_gauge("query_understanding_confidence_last", confidence, label=label_intent)
            except Exception:
                pass
            try:
                ambiguity = float(understanding.get("ambiguity_score", 0.0) or 0.0)
                self._slo_metrics.set_gauge("query_understanding_ambiguity_last", ambiguity, label=label_intent)
            except Exception:
                pass
            if bool(understanding.get("should_ask_clarification", False)):
                self._slo_metrics.inc("query_understanding_clarification_total", label=label_intent)
            route = str(understanding.get("recommended_route", "direct") or "direct")
            self._slo_metrics.inc("query_understanding_route_total", label=route)
        if bool(ctx.metadata.get("kb_match_count", 0)):
            self._slo_metrics.inc("kb_retrieval_hit_total", label=label_intent)
        elif str(ctx.intent or "") in {"information_query", "memory_query"}:
            self._slo_metrics.inc("kb_retrieval_miss_total", label=label_intent)
        lats = ctx.metadata.get("latency_ms", {})
        if isinstance(lats, dict):
            for name, value in lats.items():
                try:
                    self._slo_metrics.observe_latency(
                        f"conversation_{name}_ms",
                        float(value or 0.0),
                        label=label_intent,
                    )
                except Exception:
                    continue
        if self._is_question_like(user_input):
            self._slo_metrics.inc("conversation_question_total", label=label_intent)

    async def _resolve_reference_from_history(
        self,
        *,
        ctx: ConversationContext,
        user_input: str,
    ) -> ReferenceResolution:
        text = str(user_input or "").strip()
        if not text or not self._looks_like_reference_followup(text):
            return ReferenceResolution(used=False, resolved_query=text, source="none")
        if bool(ctx.metadata.get("low_latency_turn")):
            return ReferenceResolution(used=False, resolved_query=text, source="heuristic_low_latency")

        memory = self._session_mgr.get(ctx.session_id)
        recent_turns: list[dict[str, Any]] = []
        if memory:
            try:
                recent_turns = memory.get_context_window(max_messages=8)[:-1]
            except Exception:
                recent_turns = []

        heuristic = self._resolve_reference_heuristic(text=text, recent_turns=recent_turns, ctx=ctx)
        if not (self._model_router and self._model_router.has_provider()):
            return heuristic
        parsed = await self._resolve_reference_via_model(ctx=ctx, user_input=text, recent_turns=recent_turns)
        if parsed and parsed.used and parsed.confidence >= max(0.64, heuristic.confidence + 0.05):
            return parsed
        return heuristic

    async def _resolve_reference_via_model(
        self,
        *,
        ctx: ConversationContext,
        user_input: str,
        recent_turns: list[dict[str, Any]],
    ) -> ReferenceResolution | None:
        history_lines: list[str] = []
        for m in recent_turns[-6:]:
            role = str(m.get("role", "")).strip().lower()
            content = str(m.get("content", "")).strip()
            if role and content:
                history_lines.append(f"{role}: {content}")
        prompt = (
            "Resolve follow-up references in the user message using recent conversation.\n"
            "Return JSON only with keys: used, resolved_query, antecedent, confidence.\n"
            "Rules:\n"
            "- used: true only when message relies on previous turns.\n"
            "- resolved_query: standalone user query with references expanded.\n"
            "- antecedent: short phrase that was referenced.\n"
            "- confidence: number 0..1.\n"
            "Recent conversation:\n"
            + "\n".join(history_lines)
            + f"\nUser message: {user_input}\nJSON:"
        )
        try:
            routed = await self._model_router.generate(
                ModelRequest(
                    prompt=prompt,
                    task_type="summarization",
                    modality="text",
                    privacy_level=self._infer_privacy_level(ctx, user_input),
                    prefer_local=True,
                    max_latency_ms=1200,
                    metadata={
                        "session_id": ctx.session_id,
                        "user_id": ctx.user_id,
                        "stage": "reference_resolution",
                    },
                )
            )
        except Exception:
            return None
        return self._parse_reference_resolution(str(routed.text or ""), fallback_query=user_input, source="model")

    @staticmethod
    def _parse_reference_resolution(
        text: str,
        *,
        fallback_query: str,
        source: str,
    ) -> ReferenceResolution | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        candidates = [raw]
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            candidates.append(m.group(0))
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            used = bool(obj.get("used", False))
            resolved_query = str(obj.get("resolved_query", "")).strip() or str(fallback_query or "").strip()
            antecedent = str(obj.get("antecedent", "")).strip()
            try:
                conf = float(obj.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            return ReferenceResolution(
                used=used and bool(resolved_query),
                resolved_query=resolved_query,
                antecedent=antecedent,
                confidence=max(0.0, min(1.0, conf)),
                source=source,
            )
        return None

    @staticmethod
    def _looks_like_reference_followup(text: str) -> bool:
        s = str(text or "").strip().lower()
        if not s:
            return False
        return any(p.search(s) for p in _REFERENCE_FOLLOWUP_PATTERNS)

    def _resolve_reference_heuristic(
        self,
        *,
        text: str,
        recent_turns: list[dict[str, Any]],
        ctx: ConversationContext,
    ) -> ReferenceResolution:
        antecedent = self._extract_recent_antecedent(recent_turns=recent_turns, ctx=ctx)
        if not antecedent:
            return ReferenceResolution(used=False, resolved_query=text, source="heuristic", confidence=0.25)
        # Keep user wording, but inject antecedent for downstream intent/planning.
        lowered = str(text or "").strip().lower()
        if re.search(r"\babout\b", lowered):
            resolved = f"{text.strip()} {antecedent}".strip()
        else:
            resolved = f"{text.strip()} about {antecedent}".strip()
        return ReferenceResolution(
            used=True,
            resolved_query=resolved,
            antecedent=antecedent,
            confidence=0.68,
            source="heuristic",
        )

    def _extract_recent_antecedent(
        self,
        *,
        recent_turns: list[dict[str, Any]],
        ctx: ConversationContext,
    ) -> str:
        tm = ctx.metadata.get("turn_memory", {})
        if isinstance(tm, dict):
            goal = str(tm.get("last_user_goal", "")).strip()
            if goal:
                return goal
            slot_map = tm.get("slots", {})
            if isinstance(slot_map, dict) and slot_map:
                for key in ("topic", "framework", "library", "goal"):
                    value = str(slot_map.get(key, "")).strip()
                    if value:
                        return value
        # Prefer explicit understanding/topic from previous turn.
        q = ctx.metadata.get("query_understanding", {})
        if isinstance(q, dict):
            goal = str(q.get("user_goal", "")).strip()
            if goal:
                return goal
        topic = str(ctx.current_topic or "").strip()
        if topic and topic not in _TOPIC_STOPWORDS:
            return topic

        # Fall back to recent assistant text.
        for m in reversed(recent_turns):
            role = str(m.get("role", "")).strip().lower()
            if role != "assistant":
                continue
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            inferred = self._infer_topic_from_text(content)
            if inferred and inferred not in _TOPIC_STOPWORDS:
                return inferred
        return ""

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
        response_policy = ConversationManager._derive_response_policy(
            user_input=user_input,
            plan=plan,
            query_understanding=ctx.metadata.get("query_understanding", {}),
        )
        length_rule = str(response_policy.get("length_rule", "Use a complete answer with practical detail."))
        parts = [
            "You are JARVIS, an intelligent AI assistant.",
            *style_rules,
            length_rule,
            f"Conversation intent: {ctx.intent}",
            f"Planned task type: {plan.task_type}",
            "Response policy contract: "
            f"style={response_policy.get('style','natural')}, "
            f"verbosity={response_policy.get('verbosity','medium')}, "
            f"sections={response_policy.get('sections','auto')}, "
            f"reasoning_depth={response_policy.get('reasoning_depth','standard')}.",
        ]
        if plan.task_type == "weather_query":
            parts.append(
                "Weather guidance: reply in 1-2 sentences with a direct answer; "
                "ask at most one brief follow-up only if location or date is missing."
            )
        if plan.task_type == "reasoning_why":
            parts.append(
                "Reasoning guidance: explain the causal chain clearly, include tradeoffs, "
                "and provide 1-2 practical alternatives when relevant."
            )
        if ctx.current_topic:
            parts.append(f"Current topic: {ctx.current_topic}")
        q_understanding = ctx.metadata.get("query_understanding", {})
        if isinstance(q_understanding, dict) and q_understanding:
            inferred_goal = str(q_understanding.get("user_goal", "")).strip()
            constraints = q_understanding.get("constraints", [])
            missing = q_understanding.get("missing_constraints", [])
            if inferred_goal:
                parts.append(f"Inferred user goal: {inferred_goal}")
            if isinstance(constraints, list) and constraints:
                parts.append(f"Known constraints: {constraints}")
            if isinstance(missing, list) and missing:
                parts.append(f"Missing constraints: {missing}")
        q_decomp = ctx.metadata.get("query_decomposition", {})
        if isinstance(q_decomp, dict) and q_decomp.get("should_decompose"):
            sub_questions = q_decomp.get("sub_questions", [])
            if isinstance(sub_questions, list) and sub_questions:
                parts.append(
                    "Decomposed sub-questions (cover all, then synthesize): "
                    + " | ".join(str(s).strip() for s in sub_questions[:5] if str(s).strip())
                )
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
        require_context_fusion = bool(media) or self._has_rich_context(context)
        allow_kb_lookup = intent in {"memory_query", "information_query"}
        prefer_local: bool | None = None
        max_latency_ms: int | None = None
        understanding = ctx.metadata.get("query_understanding", {})
        if not isinstance(understanding, dict):
            understanding = {}
        recommended_route = str(understanding.get("recommended_route", "")).strip().lower()
        response_depth = str(understanding.get("response_depth", "")).strip().lower()
        requires_retrieval = bool(understanding.get("requires_retrieval", False))
        ambiguity = 0.0
        try:
            ambiguity = float(understanding.get("ambiguity_score", 0.0) or 0.0)
        except Exception:
            ambiguity = 0.0

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
        elif self._is_why_reasoning_query(low):
            task_type = "reasoning_why"
            require_context_fusion = True
            complexity = max(complexity, 0.74)
            prefer_local = None
            max_latency_ms = 12000
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
        q_decomp = ctx.metadata.get("query_decomposition", {})
        if isinstance(q_decomp, dict) and bool(q_decomp.get("should_decompose")):
            sub_questions = q_decomp.get("sub_questions", [])
            sub_count = len(sub_questions) if isinstance(sub_questions, list) else 0
            if sub_count >= 2:
                complexity = max(complexity, min(0.92, 0.58 + 0.06 * sub_count))
                require_context_fusion = True
                if target_length == "short":
                    target_length = "medium"
                elif sub_count >= 3 and target_length == "medium":
                    target_length = "long"
                if task_type in {"general_query", "task_execution"}:
                    task_type = "analysis"
        if self._adaptive_planning_enabled:
            if recommended_route == "clarify":
                task_type = "summarization"
                prefer_local = True if prefer_local is None else prefer_local
                max_latency_ms = min(int(max_latency_ms or 9000), 5000)
            elif recommended_route in {"decompose", "plan"}:
                task_type = "analysis"
                complexity = max(complexity, 0.72)
                require_context_fusion = True
                if max_latency_ms is None:
                    max_latency_ms = 14000
            elif recommended_route == "retrieve":
                allow_kb_lookup = True
                task_type = "information_query"
                complexity = max(complexity, 0.58)
            if ambiguity >= 0.68 and recommended_route not in {"clarify"}:
                require_context_fusion = True
                complexity = max(complexity, 0.65)

        if self._confidence_retrieval_enabled and requires_retrieval:
            allow_kb_lookup = True

        if target_length == "short" and max_latency_ms is None:
            max_latency_ms = 8000
        if response_depth == "short":
            target_length = "short"
        elif response_depth == "long":
            target_length = "long"

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

    def _estimate_request_complexity(
        self,
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
        if modality != "text" or media or self._has_rich_context(context):
            score += 0.18
        return max(0.0, min(1.0, score))

    def _should_single_pass_response(
        self,
        *,
        ctx: ConversationContext,
        plan: ResponsePlan | None,
        response: str,
    ) -> bool:
        if not self._single_pass_summary_enabled:
            return False
        raw = str(response or "").strip()
        if not raw:
            return True
        if self._looks_like_meta_response(raw):
            return False
        if self._needs_response_summary(raw):
            return False
        current_plan = plan or None
        if current_plan is None:
            return True
        if current_plan.preserve_format:
            return True
        if current_plan.target_length == "short":
            return True
        if current_plan.complexity <= 0.44 and ctx.intent in self._FAST_PATH_INTENTS:
            return True
        return False

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
        if ConversationManager._is_why_reasoning_query(low):
            return "long"
        if any(k in low for k in ("tell me more", "more about", "all of those", "go deeper", "in detail")):
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

    @staticmethod
    def _derive_response_policy(
        *,
        user_input: str,
        plan: ResponsePlan,
        query_understanding: dict[str, Any] | Any,
    ) -> dict[str, Any]:
        low = str(user_input or "").strip().lower()
        understanding = query_understanding if isinstance(query_understanding, dict) else {}
        depth = str(understanding.get("response_depth", "")).strip().lower()
        ambiguity = 0.0
        try:
            ambiguity = float(understanding.get("ambiguity_score", 0.0) or 0.0)
        except Exception:
            ambiguity = 0.0

        verbosity = plan.target_length
        if depth in {"short", "medium", "long"}:
            verbosity = depth
        style = "natural"
        sections = "auto"
        reasoning_depth = "standard"
        length_rule = "Use a complete answer with practical detail."
        if verbosity == "short":
            length_rule = "Keep replies concise by default (1-3 sentences)."
        elif verbosity == "long":
            length_rule = "Provide a thorough answer with clear structure and practical detail."
            sections = "recommended"
            reasoning_depth = "high"
        if re.search(r"\b(step by step|compare|tradeoff|why)\b", low):
            reasoning_depth = "high"
            sections = "recommended"
            if verbosity == "short":
                verbosity = "medium"
                length_rule = "Use a complete answer with practical detail."
        if ambiguity >= 0.66:
            style = "clarifying"
        if plan.preserve_format:
            style = "format_preserving"
            sections = "none"
        return {
            "style": style,
            "verbosity": verbosity,
            "sections": sections,
            "reasoning_depth": reasoning_depth,
            "length_rule": length_rule,
        }

    @staticmethod
    def _is_why_reasoning_query(low_text: str) -> bool:
        low = str(low_text or "")
        if not low:
            return False
        if re.search(r"\bwhy\b", low):
            return True
        if re.search(r"\b(reason|root cause|because|tradeoff|pros and cons|advantages and disadvantages)\b", low):
            return True
        return False

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
            return

        m8 = re.search(r"\bmy name is\s+([a-zA-Z][a-zA-Z0-9 _-]{0,40})\b", text, re.IGNORECASE)
        if m8:
            self._profile_store.update_traits(user_id, display_name=m8.group(1).strip())
            return

        m9 = re.search(r"\bi am\s+([a-zA-Z][a-zA-Z0-9 _-]{0,40})\b", text, re.IGNORECASE)
        if m9:
            value = m9.group(1).strip()
            # Avoid storing non-name phrases.
            if not re.search(r"\b(learning|looking|trying|working|doing|building)\b", value, re.IGNORECASE):
                self._profile_store.update_traits(user_id, display_name=value)

    def _handle_profile_name_query(self, *, ctx: ConversationContext, user_input: str) -> str:
        low = str(user_input or "").strip().lower()
        if not low:
            return ""
        if not self._is_name_recall_query(low):
            return ""
        profile = self._profile_store.get_or_create(ctx.user_id)
        name = str(profile.traits.get("display_name", "")).strip()
        if name:
            return f"Your name is {name}."
        return "I don't have your name yet. You can tell me by saying: My name is <your name>."

    def _handle_profile_name_set(self, *, ctx: ConversationContext, user_input: str) -> str:
        name = self._extract_name_from_intro(user_input)
        if not name:
            return ""
        self._profile_store.update_traits(ctx.user_id, display_name=name)
        return f"Nice to meet you, {name}. I will remember your name."

    @staticmethod
    def _is_name_recall_query(low_text: str) -> bool:
        low = str(low_text or "")
        if not low:
            return False
        patterns = (
            r"\b(what(?:'s| is)\s+my\s+name)\b",
            r"\b(tell\s+my\s+name)\b",
            r"\b(tell\s+me\s+my\s+name)\b",
            r"\b(do\s+you\s+know\s+my\s+name)\b",
        )
        return any(re.search(p, low, re.IGNORECASE) for p in patterns)

    @staticmethod
    def _extract_name_from_intro(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        m = re.search(r"\bmy name is\s+([a-zA-Z][a-zA-Z0-9 _-]{0,40})\b", raw, re.IGNORECASE)
        if not m:
            m = re.search(r"\bi am\s+([a-zA-Z][a-zA-Z0-9 _-]{0,40})\b", raw, re.IGNORECASE)
        if not m:
            return ""
        candidate = m.group(1).strip(" .,!?\t\r\n")
        if not candidate:
            return ""
        if re.search(r"\b(learning|looking|trying|working|doing|building)\b", candidate, re.IGNORECASE):
            return ""
        return candidate

    def _generic_fallback(self, ctx: ConversationContext, user_input: str) -> str:
        low = str(user_input or "").lower()
        if self._is_affirmative_continuation(low):
            pending = self._consume_pending_action(ctx, user_input)
            if pending:
                return pending
        if ctx.metadata.get("learning_plan_pending"):
            slots = dict(ctx.metadata.get("learning_plan_slots", {}))
            level, goal = self._extract_learning_slots(user_input)
            if level:
                slots["level"] = level
            if goal:
                slots["goal"] = goal
            ctx.metadata["learning_plan_slots"] = slots
            level = str(slots.get("level", "")).strip().lower()
            goal = str(slots.get("goal", "")).strip().lower()
            if level and goal:
                ctx.metadata["learning_plan_pending"] = False
                return self._build_learning_plan(level=level, goal=goal)
            if level and not goal:
                return "Got it. What is your goal: job, project, or interview?"
            if goal and not level:
                return "Got it. What is your level: beginner, intermediate, or advanced?"

        direct_email = self._rule_based_email_draft(user_input)
        if direct_email:
            return direct_email
        direct_code = self._rule_based_code_draft(user_input)
        if direct_code:
            return direct_code
        if self._is_affirmative_continuation(low):
            follow = self._affirmative_followup_from_recent_history(ctx.session_id)
            if follow:
                return follow
            return "Sure. Tell me what you want me to continue with."
        topic_hint = self._infer_topic_from_text(user_input) or str(ctx.current_topic or "").strip().lower()
        question_like = self._is_question_like(user_input)
        if self._is_why_reasoning_query(low):
            topic = topic_hint or self._infer_topic_from_text(user_input) or "this"
            ctx.metadata["pending_action"] = {
                "type": "compare_alternatives",
                "topic": topic,
                "created_turn": int(ctx.turn_count),
                "source": "generic_fallback_why",
            }
            return (
                f"The main reason behind {topic} is usually a tradeoff between speed, reliability, and complexity. "
                "Teams pick the option that best fits their constraints, even if it is not perfect on every axis. "
                "If you want, I can compare two concrete alternatives and explain when each is better."
            )
        if question_like and re.search(r"\b(all of those|those|them|these)\b", low):
            follow_up = self._followup_from_recent_history(ctx.session_id)
            if follow_up:
                return follow_up
        if self._is_capability_learning_query(low):
            ctx.metadata["learning_plan_pending"] = True
            ctx.metadata["learning_plan_slots"] = {}
            return (
                "I can teach frontend libraries like React, Next.js, and Tailwind, backend stacks like Node.js and FastAPI, "
                "and data/AI tools like Pandas and LangGraph. "
                "Tell me your level and goal, and I will give you a step-by-step learning path."
            )
        if self._is_research_request(low):
            topic = self._resolve_research_topic(topic_hint=topic_hint, user_input=user_input)
            return (
                f"Yes, I can research {topic}. I can give you a structured brief with fundamentals, current trends, "
                "top tools/frameworks, and a practical roadmap based on your goal."
            )
        if topic_hint and re.search(r"\b(learn|study|understand)\b", low):
            clean_topic = topic_hint.strip()
            if clean_topic in {"react", "react js", "reactjs"}:
                return (
                    "Start with components, props, state, and hooks, then build 2-3 small projects "
                    "like a todo app and a simple dashboard. "
                    "After that, learn routing, data fetching, and testing to become job-ready."
                )
            return (
                f"Start with the basics of {clean_topic}, then build small hands-on projects and gradually add "
                "real-world patterns like testing and deployment."
            )
        if topic_hint:
            compact_topic = topic_hint.replace(" ", "")
            snippet = _KNOWLEDGE_FALLBACK_SNIPPETS.get(compact_topic) or _KNOWLEDGE_FALLBACK_SNIPPETS.get(topic_hint)
            if snippet:
                return snippet
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
        if "?" in user_input or question_like:
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
    def _is_research_request(low: str) -> bool:
        text = str(low or "").strip().lower()
        if not text:
            return False
        if re.search(r"\b(research|investigate|analy[sz]e|study)\b", text):
            return True
        return False

    @staticmethod
    def _extract_research_topic(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        m = re.search(r"\b(?:on|about|into)\s+([a-zA-Z0-9][a-zA-Z0-9 ._\-+/]{1,80})", raw, re.IGNORECASE)
        if not m:
            return ""
        topic = re.sub(r"\s+", " ", m.group(1)).strip(" .!?")
        return topic

    @staticmethod
    def _resolve_research_topic(*, topic_hint: str, user_input: str) -> str:
        hint = str(topic_hint or "").strip().lower()
        if hint in {"research", "investigate", "analyze", "analyse", "study"}:
            hint = ""
        extracted = ConversationManager._extract_research_topic(user_input)
        topic = extracted or hint or "this topic"
        return topic

    @staticmethod
    def _infer_topic_from_text(text: str) -> str:
        raw = str(text or "").strip().lower()
        if not raw:
            return ""
        phrase_patterns = (
            r"\b(?:tell me about|explain|what is|what's|define)\s+([a-z0-9][a-z0-9 ._\-+/]{1,60})",
            r"\b(?:learn|study|understand)\s+([a-z0-9][a-z0-9 ._\-+/]{1,60})",
            r"\babout\s+([a-z0-9][a-z0-9 ._\-+/]{1,60})",
        )
        for pat in phrase_patterns:
            m = re.search(pat, raw, re.IGNORECASE)
            if not m:
                continue
            phrase = re.sub(r"\s+", " ", m.group(1)).strip(" .!?")
            phrase = re.sub(r"\b(?:can|could|would|will)\s+you\b.*$", "", phrase).strip(" .!?")
            phrase = re.sub(r"\b(?:teach|help)\s+me\b.*$", "", phrase).strip(" .!?")
            if phrase:
                return phrase
        tokens = re.findall(r"[a-z0-9][a-z0-9._\-+/]{1,40}", raw)
        for token in tokens:
            if token in _TOPIC_STOPWORDS:
                continue
            return token
        return ""

    @staticmethod
    def _is_question_like(text: str) -> bool:
        low = str(text or "").strip().lower()
        if not low:
            return False
        if "?" in low:
            return True
        return bool(
            re.match(
                r"^(what|why|how|when|where|who|which|can|could|would|should|is|are|do|does|did|tell me|explain)\b",
                low,
            )
        )

    @staticmethod
    def _is_capability_learning_query(low_text: str) -> bool:
        low = str(low_text or "")
        if not re.search(r"\b(learn|teach|teaching|study|help)\b", low):
            return False

        # Direct capability prompts.
        if re.search(r"\b(what can you teach|what do you teach|what can i learn)\b", low):
            return True

        # Domain-bucket prompts even without explicit "libraries/frameworks".
        if re.search(r"\b(frontend|backend|fullstack|ai|ml|tooling|devops|data)\b", low):
            return True

        # Generic library/framework learning prompts.
        if ("library" in low or "libraries" in low or "framework" in low or "frameworks" in low) and re.search(
            r"\b(what|which|other|can you)\b", low
        ):
            return True

        return False

    @staticmethod
    def _extract_learning_slots(text: str) -> tuple[str, str]:
        low = str(text or "").strip().lower()
        if not low:
            return "", ""
        level = ""
        goal = ""
        if re.search(r"\b(beginner|intermediate|advanced|advance)\b", low):
            m = re.search(r"\b(beginner|intermediate|advanced|advance)\b", low)
            if m:
                level = "advanced" if m.group(1) == "advance" else m.group(1)
        if re.search(r"\b(job|project|interview)\b", low):
            m = re.search(r"\b(job|project|interview)\b", low)
            if m:
                goal = m.group(1)
        m_goal = re.search(r"\bgoal\s*(?:is|=|:)?\s*(job|project|interview)\b", low)
        if m_goal:
            goal = m_goal.group(1)
        m_level = re.search(r"\blevel\s*(?:is|=|:)?\s*(beginner|intermediate|advanced|advance)\b", low)
        if m_level:
            level = "advanced" if m_level.group(1) == "advance" else m_level.group(1)
        return level, goal

    @staticmethod
    def _build_learning_plan(*, level: str, goal: str) -> str:
        lvl = str(level or "").strip().lower() or "intermediate"
        g = str(goal or "").strip().lower() or "project"
        if lvl == "advanced" and g == "job":
            return (
                "Great. For an advanced job track, focus on architecture, performance, and production systems. "
                "Build one end-to-end project with testing, CI/CD, observability, and cloud deployment, then prepare "
                "for system design and deep debugging interviews with weekly mock rounds."
            )
        return (
            f"Great. For a {lvl} {g} track, I suggest a weekly plan with core concepts, hands-on projects, "
            "and interview/readiness checkpoints."
        )

    def _followup_from_recent_history(self, session_id: str) -> str:
        memory = self._session_mgr.get(session_id)
        if memory is None:
            return ""
        recent = memory.get_context_window(max_messages=8)
        if not recent:
            return ""
        assistant_text = ""
        for msg in reversed(recent):
            if str(msg.get("role", "")).strip().lower() == "assistant":
                assistant_text = str(msg.get("content", "") or "").lower()
                break
        if not assistant_text:
            return ""
        ordered = ["react", "next.js", "tailwind", "node.js", "fastapi", "pandas", "langgraph"]
        picked = [name for name in ordered if name in assistant_text]
        if not picked:
            return ""
        details = [f"{name}: {_LIBRARY_DETAIL_SNIPPETS[name]}" for name in picked]
        return " ".join(details)

    def _affirmative_followup_from_recent_history(self, session_id: str) -> str:
        memory = self._session_mgr.get(session_id)
        if memory is None:
            return ""
        recent = memory.get_context_window(max_messages=8)
        if not recent:
            return ""
        assistant_text = ""
        for msg in reversed(recent):
            if str(msg.get("role", "")).strip().lower() == "assistant":
                assistant_text = str(msg.get("content", "") or "").lower()
                break
        if not assistant_text:
            return ""
        if "level and goal" in assistant_text:
            return (
                "Great. Tell me your level (beginner/intermediate/advanced) and your goal "
                "(job, project, or interview), and I will give you a step-by-step plan."
            )
        if "compare two concrete alternatives" in assistant_text or "if you want, i can compare" in assistant_text:
            topic = ""
            for msg in reversed(recent):
                if str(msg.get("role", "")).strip().lower() != "user":
                    continue
                user_text = str(msg.get("content", "") or "").strip().lower()
                if not user_text:
                    continue
                if "react" in user_text:
                    topic = "react"
                    break
                inferred = self._infer_topic_from_text(user_text)
                if inferred and inferred not in _TOPIC_STOPWORDS:
                    topic = inferred
                    break
            if topic in {"react", "reactjs", "react js"}:
                return (
                    "For frontend apps, React is usually faster to ship than Angular because its component model is lighter and "
                    "its ecosystem is less opinionated, while Angular gives stronger built-in structure for large enterprise teams. "
                    "Compared with Vue, React has a larger job market and tooling ecosystem, while Vue is often simpler to learn and faster for small teams."
                )
            if topic:
                return (
                    f"Here are two alternatives for {topic}: option one optimizes for faster delivery and flexibility, "
                    "while option two optimizes for stricter structure and long-term consistency. "
                    "If you want, I can map both options to your exact project constraints."
                )
        return ""

    @staticmethod
    def _is_affirmative_continuation(low_text: str) -> bool:
        low = str(low_text or "").strip().lower()
        if not low:
            return False
        words = [w for w in re.split(r"\s+", low) if w]
        if re.match(r"^\s*(ok|okay|sure|yes|yep|yeah)\b", low) and re.search(r"\b(give|show|tell|share|send)\b", low):
            return True
        # Handle short follow-ups like "yes please do", "sure", "go ahead", "do it".
        if re.match(r"^\s*(ok|okay|sure|yes|yep|yeah)(\s+please)?(\s+do)?\s*[.!?]*\s*$", low):
            return True
        if re.match(r"^\s*(please\s+)?(do\s+it|go\s+ahead|continue)\s*[.!?]*\s*$", low):
            return True
        # Accept short affirmative continuations even when reference resolution appends extra context.
        if re.match(r"^\s*(ok|okay|sure|yes|yep|yeah)\b", low):
            if re.search(r"\b(but|not|later|wait|hold on)\b", low):
                return False
            if len(words) <= 8 and not re.search(r"\b(why|how|what|when|where|who)\b", low):
                return True
        return False

    def _consume_pending_action(
        self,
        ctx: ConversationContext,
        user_input: str,
        *,
        directives: dict[str, Any] | None = None,
    ) -> str:
        pending = ctx.metadata.get("pending_action")
        if not isinstance(pending, dict):
            return ""
        action_type = str(pending.get("type", "")).strip().lower()
        if not action_type:
            return ""
        prefs = directives if isinstance(directives, dict) and directives else self._extract_output_directives(user_input)

        ctx.metadata.pop("pending_action", None)
        if action_type == "compare_alternatives":
            topic = str(pending.get("topic", "")).strip().lower()
            if not topic:
                topic = self._infer_topic_from_text(user_input) or str(ctx.current_topic or "").strip().lower()
            if topic in {"react", "reactjs", "react js"}:
                text = (
                    "Compared with Angular, React is usually faster to ship because it has less framework ceremony and a more flexible component stack, "
                    "while Angular is better when you want strict conventions and built-in enterprise patterns. "
                    "Compared with Vue, React has a larger hiring/tooling ecosystem, while Vue often feels simpler and faster for smaller teams."
                )
                return self._apply_output_directives(text, prefs)
            if topic:
                text = (
                    f"For {topic}, a flexible option is usually faster to deliver, while a structured option is usually easier to scale and govern long-term. "
                    "Tell me your constraints and I will recommend one."
                )
                return self._apply_output_directives(text, prefs)
            text = "Sure. Share the topic you want compared, and I will break down two concrete alternatives."
            return self._apply_output_directives(text, prefs)

        if action_type == "collect_learning_profile":
            ctx.metadata["learning_plan_pending"] = True
            ctx.metadata.setdefault("learning_plan_slots", {})
            text = (
                "Great. Tell me your level (beginner/intermediate/advanced) and goal "
                "(job/project/interview), and I will give you a tailored plan."
            )
            return self._apply_output_directives(text, prefs)
        return ""

    def _refresh_pending_action_from_response(
        self,
        *,
        ctx: ConversationContext,
        user_input: str,
        response: str,
    ) -> None:
        low_resp = str(response or "").strip().lower()
        if not low_resp:
            return
        if "if you want, i can compare" in low_resp or "compare two concrete alternatives" in low_resp:
            topic = self._infer_topic_from_text(user_input) or str(ctx.current_topic or "").strip().lower() or "this"
            ctx.metadata["pending_action"] = {
                "type": "compare_alternatives",
                "topic": topic,
                "created_turn": int(ctx.turn_count),
                "source": "response_offer",
            }
            return
        if "tell me your level and goal" in low_resp or "share your current level and goal" in low_resp:
            ctx.metadata["pending_action"] = {
                "type": "collect_learning_profile",
                "created_turn": int(ctx.turn_count),
                "source": "response_offer",
            }

    @staticmethod
    def _extract_output_directives(text: str) -> dict[str, Any]:
        low = str(text or "").strip().lower()
        if not low:
            return {}
        prefs: dict[str, Any] = {}
        if re.search(r"\b(stream|section|part|chunk)\b", low):
            prefs["sectioned"] = True
        m_sec = re.search(r"\b(?:in|with)\s+(\d{1,2})\s+(?:sections|parts|chunks)\b", low)
        if m_sec:
            try:
                prefs["max_sections"] = max(1, min(12, int(m_sec.group(1))))
                prefs["sectioned"] = True
            except Exception:
                pass
        if re.search(r"\b(concise|brief|short)\b", low):
            prefs["target_length"] = "short"
        elif re.search(r"\b(detailed|detail|deep|long)\b", low):
            prefs["target_length"] = "long"
        if re.search(r"\b(points|bullet|list)\b", low):
            prefs["format"] = "list"
        return prefs

    @staticmethod
    def _apply_output_directives(text: str, directives: dict[str, Any] | None) -> str:
        body = str(text or "").strip()
        if not body:
            return body
        prefs = directives if isinstance(directives, dict) else {}
        target_length = str(prefs.get("target_length", "")).strip().lower()
        sectioned = bool(prefs.get("sectioned"))
        max_sections = prefs.get("max_sections", 0)
        format_hint = str(prefs.get("format", "")).strip().lower()
        if sectioned and isinstance(max_sections, int) and max_sections > 1:
            sentences = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", body) if s.strip()]
            if len(sentences) >= 2:
                effective_sections = max_sections
                if target_length == "short":
                    effective_sections = min(effective_sections, 2)
                per = max(1, (len(sentences) + effective_sections - 1) // effective_sections)
                blocks: list[str] = []
                for i in range(0, len(sentences), per):
                    blocks.append(" ".join(sentences[i : i + per]).strip())
                if format_hint == "list":
                    return "\n\n".join(f"{idx + 1}. {blk}" for idx, blk in enumerate(blocks[:effective_sections]))
                return "\n\n".join(blocks[:effective_sections])
        if target_length == "short":
            first_sentence = re.split(r"(?<=[\.\!\?])\s+", body, maxsplit=1)[0].strip()
            if first_sentence:
                body = first_sentence
        return body

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
        if plan and plan.target_length in {"medium", "long"} and not self._looks_like_meta_response(raw):
            ctx.metadata["summary_stage"] = {
                "used": False,
                "source": "passthrough_target_nonshort",
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
        if bool(ctx.metadata.get("low_latency_turn")):
            salvaged = self._summarize_response_for_chat_rule(raw)
            if salvaged and salvaged != raw:
                ctx.metadata["summary_stage"] = {
                    "used": False,
                    "source": "fallback_rule_low_latency",
                    "latency_ms": round((time.time() - summarize_started) * 1000.0, 2),
                }
                return salvaged
            ctx.metadata["summary_stage"] = {
                "used": False,
                "source": "passthrough_low_latency",
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
