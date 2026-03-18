"""
Conversation state management for JARVIS AI OS.

Handles multi-turn sessions, intent extraction (rule-based + LLM), contextual
response generation, and integration with ConversationMemory and KnowledgeBase.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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
    (r"\b(run|execute|perform|do|start|launch)\b", "task_execution", 0.75),
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

        # Store in memory
        memory = self._session_mgr.get_or_create(session_id)
        memory.add_message(role="user", content=user_input)

        try:
            # 1. Intent extraction
            intent, entities, confidence = self.extract_intent(user_input)
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

            # 3. Response generation
            ctx.state = ConversationState.RESPONDING
            response = await self.generate_response(
                ctx,
                user_input,
                modality=modality,
                media=media or {},
                context=context or {},
            )

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
        modality: str = "text",
        media: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Build a contextual reply for *ctx.intent*.

        Priority:
        1. LLM handler (if wired up)
        2. Knowledge base lookup
        3. Rule-based template
        4. Generic fallback
        """
        # 1. Hybrid model-router path
        if self._model_router and self._model_router.has_provider():
            try:
                memory = self._session_mgr.get(ctx.session_id)
                history_text = ""
                if memory:
                    window = memory.get_context_window(max_messages=6)
                    history_text = "\n".join(
                        f"{m['role']}: {m['content']}" for m in window[:-1]
                    )
                prompt = self._build_llm_prompt(ctx, user_input, history_text)
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
                    task_type=ctx.intent or "general",
                    modality=modality,
                    media=media or {},
                    privacy_level=self._infer_privacy_level(ctx, user_input),
                    metadata={
                        "session_id": ctx.session_id,
                        "user_id": ctx.user_id,
                        "context_fusion": fused_context,
                    },
                )
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
                    ctx.metadata["context_fusion"] = fused_context
                    return str(routed.text).strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hybrid model router failed: %s", exc)

        # 2. LLM path
        if self._llm_handler:
            try:
                memory = self._session_mgr.get(ctx.session_id)
                history_text = ""
                if memory:
                    window = memory.get_context_window(max_messages=6)
                    history_text = "\n".join(
                        f"{m['role']}: {m['content']}" for m in window[:-1]  # exclude last (current)
                    )
                prompt = self._build_llm_prompt(ctx, user_input, history_text)
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

        # 3. Knowledge base lookup for memory/information queries
        if ctx.intent in ("memory_query", "information_query"):
            kb_response = self._kb_lookup(user_input, ctx=ctx)
            if kb_response:
                return kb_response

        # Personalized response hint for general intent.
        profile_hint = self.get_user_profile_summary(ctx.user_id)
        if profile_hint and ctx.intent in ("general_query", "help_request"):
            return f"{profile_hint}. {self._generic_fallback(ctx, user_input)}"

        # 4. Template-based responses
        templates = self._response_templates.get(ctx.intent, [])
        if templates:
            import random
            return random.choice(templates)

        # 5. Fallback
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
        ctx: ConversationContext, user_input: str, history: str
    ) -> str:
        parts = [
            "You are JARVIS, an intelligent AI assistant.",
            f"Current intent: {ctx.intent}",
        ]
        if ctx.current_topic:
            parts.append(f"Current topic: {ctx.current_topic}")
        if history:
            parts.append(f"Recent conversation:\n{history}")
        parts.append(f"User: {user_input}")
        parts.append("Assistant:")
        return "\n".join(parts)

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
