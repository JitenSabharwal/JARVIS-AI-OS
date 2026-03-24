# JARVIS AI OS - Development Plan (Engineer Handoff)

This file is the single source of truth for current project state, completed scope, pending work, and how to continue implementation safely.

## 1. Project Objective
Build a reliable, production-grade AI assistant platform with:
1. Hybrid model routing (local + API)
2. Agentic execution with policy/approval controls
3. Retrieval + memory + graph-backed intelligence
4. Multimodal support
5. Operational readiness (CI, observability, rollback)

## 2. Current Snapshot
Status date: 2026-03-23

Overall state:
1. Core platform is functionally built.
2. Most major feature phases are implemented.
3. Runtime validation is green in local dev environment; remaining work is CI/release closure and phase-status reconciliation.

Validation note (latest run):
1. `source ~/ai-envs/mlx-env/bin/activate && pytest -q` -> `208 passed, 1 warning` (2026-03-21 baseline).
2. Targeted regression updates added for conversational continuation + streaming directives (2026-03-23).

## 3. Architecture at a Glance
Primary layers:
1. Interface Layer
- OpenAI-compatible `/v1/chat/completions`
- API endpoints for tasks, plans, research, delivery, proactive flows
- Request normalization and workspace/session inference

2. Orchestration Layer
- Task submission and dependency-aware plans
- Retry/replan/approval-gated execution
- Optional LangGraph workflow adapter integration

3. Agent Layer
- Specialized agents for coding, repository understanding, and execution
- High-depth bounded analysis pipeline for deep repo understanding

4. Model Routing Layer
- Policy-based provider routing
- Fallback handling and shadow-mode telemetry paths

5. Retrieval/Memory Layer
- Hierarchical RAG index
- Optional Neo4j relationship graph
- Research ingest/query/watchlist/digest flows

6. Verification/Response Layer
- Evidence consistency checks
- Contradiction detection and confidence scoring
- Response contracts to prevent reasoning/noise leakage

7. Observability/Ops Layer
- SLO metrics and threshold checks
- CI quality gates
- Audit logging and approval traceability

## 4. What Is Implemented
## 4.1 Foundation and platform reliability
1. API/orchestrator baseline stabilized.
2. Serialization and request handling normalized.
3. Security and approval-gating scaffolding is in place.

## 4.2 Hybrid intelligence and routing
1. Local/API providers integrated via model router.
2. Route context + fallback logic wired in core flows.
3. Modality-aware routing is implemented in core request handling.

## 4.3 Agentic reliability
1. Plan graph submission with dependencies implemented.
2. Retry/replan and approval-based escalation paths implemented.
3. Verifier-capability hooks and fail-closed behavior implemented.

## 4.4 Research + retrieval + graph
1. Hierarchical RAG index and query context integration implemented.
2. Optional Neo4j graph persistence and health checks implemented.
3. Research adapters, watchlists, and digest workflows implemented.

## 4.5 Personal/productivity and proactive features
1. Email/File/Image intelligence connectors implemented with scoped permissions.
2. Proactive event/suggestion lifecycle implemented with safety checks.
3. Cross-modal continuity tracking and preference modeling implemented.

## 4.6 Delivery engine
1. Bootstrap templates and quality-gate pipeline implemented.
2. Deployment profiles, post-deploy checks, and rollback decisions implemented.
3. One-shot release run API and command-backed gate/deploy adapters implemented.

## 4.7 Response architecture hardening (Phase 14)
Tracks completed:
1. Track A: Typed contracts + request envelope propagation
2. Track B: Graph-enriched retrieval signals
3. Track C: Contradiction detection + evidence policy hardening
4. Track D: Repo quality benchmark + CI release gate with thresholds
5. Track E: High-depth bounded planner -> analyst -> reviewer -> verifier pipeline

## 4.8 Conversational Intelligence Upgrade (Phase 15)
Implemented in this cycle:
1. Intent understanding upgraded to include ambiguity score, retrieval need, recommended execution route, and response depth.
2. Adaptive routing/planning now consumes understanding signals (`clarify/retrieve/decompose/plan`) instead of relying only on keyword intent.
3. Retrieval path upgraded to confidence-aware selection (relevance + source trust + freshness) with dynamic thresholding on ambiguous queries.
4. Structured turn-memory semantics added (`slots`, unresolved constraints, follow-up sub-questions, last goal/topic) for better continuation.
5. Response policy shaping contract added to prompt construction and propagated via model metadata.
6. Evaluation telemetry hooks added for turn outcomes, understanding quality, retrieval hit/miss, and per-stage latency series.

## 5. Current Pending Work (Actionable)
## 5.1 Validation and closure
1. Confirm CI quality gate results on live branch state (local suite is already passing).
2. Capture and link quality-gate artifacts in this plan (`eval_repo_quality`, release readiness outputs).
3. Close residual status markers that still show "In Progress" where implementation is already complete.

## 5.0 Architecture Layer Hardening (Completed in this cycle)
1. Orchestration Layer
- Added workflow checkpoint persistence in-memory with resume support (`__workflow_checkpoint` + stored checkpoints).
- Added `get_workflow_checkpoint(workflow_id)` for recovery/introspection.
2. Execution Layer (lane-aware)
- Added lane-aware wave batching with configurable per-lane caps via `JARVIS_AGENT_WORKFLOW_LANE_CAPS`.
- Added flow metadata propagation to each workflow task (`flow_lane`, `flow_wave_index`, `flow_fan_in`).
3. Verification Layer
- Added centralized response finalizer utility (`core/response_finalizer.py`) to enforce no meta reasoning leakage.
4. Response Composer Layer
- Wired canonical finalization through `/api/v1/query` and `/v1/chat/completions`.
5. Observability/Eval Layer
- Added cross-route quality evaluator (`scripts/eval_response_quality.py`) and CI gate.
- Added deterministic case datasets for chat/code/workflow response checks.
6. Ops Closure Layer
- Extended `scripts/release_readiness_check.py` with quality-gate asset checks and lane-cap env contract checks.
- Added persistent workflow-checkpoint contract checks and quality gate coverage checks for cross-route response governance.
7. Policy/Strategy/Resource Upgrade Layer
- Added policy+cost awareness engine and routing influence (`infrastructure/policy_cost_engine.py`).
- Added adaptive strategy engine for lane scheduling (`core/strategy_engine.py`).
- Added CPU/GPU resource pool admission control (`infrastructure/resource_pool_manager.py`).
- Added synthetic load + strategy evaluation scripts (`scripts/simulate_load.py`, `scripts/eval_scheduling_strategy.py`).

## 5.2 Productization hardening
1. Finalize observability dashboard completeness and alert tuning.
2. Complete release-readiness checklist execution and artifact capture.
3. Confirm rollback runbooks with rehearsal evidence.

## 5.3 Phase-specific pending status
1. Phase 9: marked "In Progress (Slice 1 Implemented)"; needs final validation closure.
2. Phases 7/8/10/11/12: marked "Implementation Complete / Validation Pending"; need validation sign-off.
3. Phase 13: Wave A in progress; Waves B/C/D remain planned scope.
4. Phase 14: implementation complete; needs final status flip after validation evidence.

## 5.4 Conversational Continuation and Streaming Control (New)
Completed in this cycle:
1. Added generic `pending_action` continuation path for short affirmatives (`yes/sure/go ahead/do it`) so "offer -> continue" works across flows.
2. Added pre-template continuation execution in `process_input` to avoid short template replies (for example `Got it!`) masking pending actions.
3. Added follow-up output directive parsing from user text (for example `in 3 sections`, `concise`, `points`) and applied shaping for pending-action replies.
4. Added realtime WS query-level streaming hints fallback (`stream`, `in N sections`) when client does not explicitly send `max_sections`.
5. Added regression tests for affirmative continuation, pending-action consume, directive parsing/shaping, and WS streaming-hint parsing.

Pending next-step upgrades:
1. Promote `pending_action` from heuristic phrase-detection to typed action intents set by planner/model output contracts.
2. Persist per-session response preferences (`sectioned`, `max_sections`, verbosity) with TTL and explicit reset semantics.
3. Extend pending-action handlers beyond compare/learning profile into repo analysis, code generation, and research workflows.
4. Add end-to-end tests for websocket section streaming with affirmative + directives in one utterance.
5. Expose a small diagnostics endpoint to inspect current session `pending_action` + `response_preferences` for UI debugging.

## 6. Engineering Handoff - Where to Start
For a new engineer, execute in this order:
1. Environment + dependency validation
2. Full test execution and failure triage
3. Phase status reconciliation in this file
4. Remaining Phase 13 roadmap implementation

Recommended entry points:
1. `interfaces/api_interface.py`
2. `core/orchestrator.py`
3. `agents/specialized_agents.py`
4. `infrastructure/model_router.py`
5. `infrastructure/research_intelligence.py`
6. `infrastructure/software_delivery.py`

## 7. Validation Checklist (Required Before Marking Complete)
1. Compile checks pass. - Pending evidence capture
2. Unit tests pass. - Complete (`pytest -q` on 2026-03-21)
3. Integration tests pass. - Complete (`pytest -q` on 2026-03-21)
4. Critical smoke tests pass. - Complete (`tests/integration/test_api_interface.py::test_api_smoke_flow`)
5. Repo quality gate passes at configured thresholds. - Pending
6. Metrics endpoints and health endpoints are healthy. - Pending latest artifact link
7. Approval-gated high-risk action tests are fail-closed. - Complete (`test_api_policy_gated_high_risk_smoke` within full run)

## 8. Standard Commands
Run from repo root.

1. Compile check
```bash
python -m compileall -q core infrastructure interfaces skills tests
```

2. Unit tests
```bash
pytest tests/unit -q
```

3. Integration tests
```bash
pytest tests/integration -q
```

4. Critical smoke gates
```bash
pytest tests/integration/test_api_interface.py::test_api_smoke_flow -q
pytest tests/integration/test_api_interface.py::test_api_policy_gated_high_risk_smoke -q
```

5. Repo quality gate
```bash
python scripts/eval_repo_quality.py \
  --cases config/evals/repo_quality_cases.json \
  --responses-file config/evals/repo_quality_sample_responses.json \
  --min-pass-rate 0.95 \
  --min-case-score 0.70 \
  --out runtime/evals/repo_quality_ci_report.json
```

## 9. Definition of Done for This Plan
Mark this plan "Complete" only when:
1. Validation checklist in section 7 is fully green.
2. Phase statuses are reconciled with evidence.
3. Remaining roadmap items are explicitly moved to next-phase backlog with owners.

## 10. Next Iteration Backlog
Priority P0:
1. CI parity with local runtime and deterministic failure triage docs.
2. Repo quality gate + readiness artifact capture and linking.
3. Final status reconciliation in this plan.
4. E2E validation for "offer -> yes + directives -> sectioned response" in websocket and `/v1/chat/completions`.

Priority P1:
1. Phase 13 Wave B implementation.
2. Dashboard and alert depth improvements.
3. Additional regression suites for high-depth multi-agent analysis.
4. Typed pending-action contract and planner-originated continuation intents.

Priority P2:
1. Cross-phase performance/cost optimization passes.
2. Extended benchmark datasets and gold-answer coverage.

## 11. Ownership Guidance
1. Keep this document current after every merged milestone.
2. Do not mark phases complete without command output evidence.
3. Link PRs/tests/reports when changing status lines.
