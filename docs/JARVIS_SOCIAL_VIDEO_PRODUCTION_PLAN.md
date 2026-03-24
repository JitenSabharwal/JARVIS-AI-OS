# Jarvis Social Voice + Video Production Plan

Status date: 2026-03-24

## 1. Product Goal
Build a production-grade multimodal assistant that:
1. Understands live video context continuously.
2. Speaks in socially appropriate, non-repetitive turns.
3. Integrates voice call + camera + chat as one coherent interaction loop.
4. Scales to multiple people with identity continuity and efficient compute use.

## 2. North-Star Experience
1. Jarvis notices meaningful changes (not every frame).
2. Jarvis tracks who is present and remembers identity context.
3. Jarvis speaks briefly, only when useful or requested.
4. Jarvis asks clarifying questions on uncertainty.
5. Users can audit why Jarvis said something (event timeline + confidence).

## 3. Production Architecture
1. Perception Layer
- Person/object detection and per-person track continuity.
- Identity match service with sticky per-track identity.
- Compute policy: prioritize unmatched tracks.

2. Scene Event Layer
- Convert frame detections into event stream (`person_entered`, `person_identified`, `person_left`, `object_spotted`).
- Event cooldowns and dedupe windows.

3. Social Orchestrator Layer
- Turn-taking policy (interrupt vs wait).
- Anti-loop policy (cooldowns, novelty thresholds).
- Prompt policy for concise social narration.

4. Conversation Layer
- Voice + chat unification.
- Scene-grounded response context via realtime media summary.
- User-facing controls: auto mode, ask-now, mute, interrupt.

5. Trust + Governance Layer
- Confidence thresholds and fallback behaviors.
- Transparent timeline with source metadata.
- Deletion and retention controls for identity data.

## 4. SLOs and KPIs
1. Median social event-to-assistant-latency: <= 2.5s.
2. False repetitive narration rate: < 5% per 10-minute call.
3. Identity rematch compute savings vs naive loop: >= 40%.
4. User-confirmed relevance of proactive social prompts: >= 70%.
5. Session stability (no UI lockups/crashes): >= 99.5%.

## 5. Rollout Phases
## Phase 1 (Implemented in this cycle)
1. Multi-person sticky identity tracking with unmatched-first recognition budget.
2. Social scene event director in web runtime.
3. Coverage stats (`Tracked`, `Matched`, `Scanning`).
4. Scene timeline panel and social prompt pipeline.
5. Optional social auto mode with cooldowns.

## Phase 2
1. Backend event bus for scene events (durable, queryable).
2. Session-level social memory with uncertainty handling.
3. Prompt strategy with style/persona controls.
4. Scene explanation endpoint (why a prompt was generated).

## Phase 3
1. Advanced social cues: gaze proxy, interaction proximity, engagement score.
2. Multi-camera fusion with shared identity graph.
3. Background learning and domain-specific social behaviors.
4. Production observability dashboards + alert thresholds.

## 6. Risk Controls
1. Never auto-speak continuously on unchanged scene.
2. Confidence-aware messaging for uncertain identity.
3. Hard cooldowns per event type and per identity.
4. Force user control over auto mode.
5. Keep conversational overrides (interrupt/stop speaking) always available.

## 7. Immediate Next Engineering Tasks
1. Move social scene director logic to backend service (`infrastructure/social_scene_orchestrator.py`).
2. Add API route for event timeline retrieval per session.
3. Add unit tests for event cooldown and prompt policy.
4. Add integration tests for multi-person scene transitions.
