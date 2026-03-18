# JARVIS AI OS Development Plan Sheet

## 1) Purpose
Build a stable, secure, hybrid (local + API) JARVIS OS foundation, then evolve it into a real-time multimodal, deeply integrated assistant platform.

This sheet is the execution source of truth for roadmap, scope, and completion criteria.

## 2) Strategic Goals
1. Stabilize the current platform and remove runtime contract breakage.
2. Introduce a production-grade hybrid intelligence layer (local + API routing).
3. Build a trustworthy action system with policy guardrails and approvals.
4. Add durable memory + personalization + learning loops.
5. Deliver real-time multimodal interaction (voice first, then vision/context).
6. Expand deep OS and external system integrations safely.
7. Reach operational maturity with observability, SLOs, and reliable CI/CD.

## 3) Scope Map (Gaps Covered)
1. Core intelligence layer
2. Reliable action system
3. Real-time multimodal interaction
4. World model + memory
5. System integration depth
6. Security and trust
7. Architecture consistency
8. Operational maturity
9. Product UX layer (added)
10. Data governance/compliance (added)
11. Deployment profiles and benchmarking (added)

## 4) Workstreams
1. Platform Core: API/orchestrator contracts, config, packaging, boot flow.
2. Intelligence: model providers, routing policy, planning/execution loop.
3. Agentic Actions: tool contracts, risk scoring, approvals, rollback patterns.
4. Memory: semantic retrieval, user profile, episode learning integration.
5. Multimodal: low-latency voice stack, interruption handling, context fusion.
6. Integrations: connectors, event automation, permissions/scopes.
7. Security & Governance: auth hardening, secrets, auditing, policy engine.
8. Reliability & Ops: tests, observability, SLOs, release process.

## 5) Phase Plan

### Phase 0: Stabilize Baseline (Weeks 1-2)
Deliverables:
1. Fix API/orchestrator sync-async and task status contract mismatches.
2. Fix JSON serialization for skills/tasks/status responses.
3. Fix packaging and runtime entrypoint issues.
4. Repair broken tests and establish passing baseline.

Exit Criteria:
1. Local boot + API health/status endpoints work end-to-end.
2. Task submit/query and skill list/execute flows work reliably.
3. CI passes unit + integration smoke tests.

### Phase 1: Security + Operational Foundation (Weeks 3-5)
Deliverables:
1. Harden auth/token compare and request validation.
2. Introduce capability-based permission model.
3. Restrict dangerous system skill surfaces and enforce policy checks.
4. Add structured audit logs + trace IDs across request/task/tool execution.
5. Define SLOs and monitoring dashboards.

Exit Criteria:
1. All sensitive operations are policy-gated and audited.
2. p95 latency and error budget tracking available.

### Phase 2: Hybrid Intelligence Layer (Weeks 6-9)
Deliverables:
1. Add `ModelProvider` abstraction and provider adapters (local + API).
2. Add routing engine by task type/privacy/latency/cost.
3. Add fallback chain local->API and API->local when eligible.
4. Integrate router into conversation and agent execution paths.

Exit Criteria:
1. Router decisions are explainable and logged.
2. Hybrid execution works in production-like scenarios.

### Phase 3: Reliable Agentic Execution (Weeks 10-13)
Deliverables:
1. Standardized typed tool contracts and strict validation.
2. Pre-check/execute/verify pattern for high-impact actions.
3. Approval workflow for risky actions.
4. Retry, compensation, and rollback conventions.

Exit Criteria:
1. High-risk tasks cannot execute without policy/approval.
2. Task reliability metrics improve and regressions are caught by tests.

### Phase 4: Memory + Personalization (Weeks 14-18)
Deliverables:
1. Semantic memory indexing and ranked retrieval pipeline.
2. User profile graph (preferences, routines, constraints).
3. Learning loop from episodic outcomes into planning heuristics.
4. Freshness/confidence metadata for retrieved knowledge.

Exit Criteria:
1. Responses/actions show measurable personalization gain.
2. Retrieval quality and memory hit-rate meet target thresholds.

### Phase 5: Multimodal Assistant (Weeks 19-24)
Deliverables:
1. Full-duplex voice with barge-in and interruption recovery.
2. Context fusion layer for text/voice/(future)vision signals.
3. Latency optimization and graceful fallback behavior.

Exit Criteria:
1. Real-time conversation quality is stable under normal load.
2. Voice interaction meets p95 latency targets.

### Phase 6: Deep Integration + Automation (Weeks 25-30)
Deliverables:
1. Connector framework (calendar, mail, files, notifications, automation).
2. Event-driven automation rules with scoped permissions.
3. Connector health checks and failure containment.

Exit Criteria:
1. At least 3 critical connectors production-ready with guardrails.
2. Event automation is auditable, reversible, and stable.

## 6) Prioritized Backlog (Now)
P0:
1. API/orchestrator contract fixes.
2. Serialization fixes.
3. Packaging + entrypoint fixes.
4. Test suite alignment with current code.

P1:
1. Security hardening for auth and command execution.
2. Audit trail and traceability.
3. Model provider + routing interfaces.

P2:
1. Hybrid routing productionization.
2. Planner/executor reliability patterns.
3. Memory retrieval and personalization loop.

## 7) Definition of Done (Global)
1. Code merged with tests and docs updated.
2. Observability added for new critical path behavior.
3. Security/policy impact reviewed.
4. Backward compatibility considered and migration notes included.
5. Rollback plan documented for risky changes.

## 8) Metrics to Track
1. Reliability: task success rate, workflow completion rate, MTTR.
2. Latency: p50/p95 for query, task execution, voice response.
3. Quality: user correction rate, plan success rate, tool-call accuracy.
4. Cost: API spend per task class, local-vs-API ratio, fallback frequency.
5. Safety: unauthorized action attempts blocked, approval bypass count (target 0).

## 9) Risks and Mitigations
1. Scope overload -> enforce phase gates and strict P0/P1/P2 prioritization.
2. Security regressions -> mandatory policy tests + audit assertions in CI.
3. Hybrid routing instability -> progressive rollout and route shadow mode first.
4. Connector fragility -> sandboxed adapters and circuit breakers.

## 10) Execution Cadence
1. Weekly planning: lock sprint goals and acceptance criteria.
2. Daily triage: blockers, incidents, and dependency resolution.
3. Weekly review: demo completed milestones + metrics trend.
4. Monthly roadmap check: re-prioritize based on risk, usage, and reliability data.

## 11) Immediate Next Sprint (Sprint 1)
1. Fix API/orchestrator task/status contract end-to-end.
2. Normalize API response serialization for tasks/skills/status.
3. Fix setup entrypoint and packaging/install path.
4. Repair tests and add smoke tests for `/query`, `/tasks`, `/skills`, `/status`.
5. Add basic request/task trace IDs in logs.

---

## 12) Sprint Progress Log
- Sprint 1 complete: API/orchestrator contract alignment, response normalization, entrypoint fixes, smoke coverage.
- Sprint 2 complete: auth hardening, policy-gated command execution, audit logging, approval workflow.
- Sprint 3 complete: skills contract validation and stricter execution checks.
- Sprint 4 complete: memory personalization baseline and semantic retrieval integration.
- Sprint 5 complete: voice realtime baseline with barge-in and callback timeout controls.
- Sprint 6 complete: connectors + automation framework with scoped permissions, circuit breaking, retries, and dead-letter endpoint.
- Sprint 7 complete: hybrid model routing baseline (local/API providers, policy-based routing, fallback chain, conversation integration).
- Sprint 8 complete: provider adapters and runtime controls (Cohere API, Ollama local, global config wiring, RAM-budget-aware local model lifecycle manager, MLX multimodal placeholders).

## 13) Sprint 9 (All Remaining Gaps Consolidated)
Objective: Close all remaining plan gaps and move Phases 0-6 to completed state.

Scope:
1. CI and baseline closure (Phase 0)
2. SLO/observability closure (Phase 1)
3. Hybrid routing productionization across agents (Phase 2)
4. Reliable agentic execution hardening (Phase 3)
5. Memory learning-loop completion (Phase 4)
6. Multimodal context fusion + latency validation (Phase 5)
7. Production-grade connector rollout and automation governance (Phase 6)
8. Global Definition of Done and metrics closure

Sprint 9 Work Items:
1. Baseline + CI
   - Install/lock test dependencies in CI and local dev.
   - Ensure unit + integration suites run green end-to-end.
   - Add explicit smoke gate for API, orchestrator, router, approvals, connectors, automation.
2. SLO + Dashboards
   - Define p50/p95 latency, error-rate, and availability SLOs per critical path.
   - Emit required metrics and wire dashboard panels.
   - Add alert thresholds for budget burn and elevated failure rates.
3. Hybrid Routing Completion
   - Route all agent execution paths (not only conversation manager) through model router where applicable.
   - Add route shadow-mode toggle, decision auditing, and fallback telemetry.
   - Add production-like scenario tests (provider outage, high privacy tasks, modality mismatch).
4. Agentic Reliability Patterns
   - Implement pre-check/execute/verify flow for high-impact actions.
   - Implement retry + compensation + rollback conventions with audit traceability.
   - Enforce policy/approval gating at all high-risk execution surfaces.
5. Memory + Personalization Completion
   - Integrate episodic outcomes into planning heuristics.
   - Add freshness/confidence thresholds for retrieval usage.
   - Add measurable retrieval/personalization evaluation tests and targets.
6. Multimodal Completion
   - Add context fusion layer across text + voice + image/audio request modalities.
   - Wire modality-aware routing from interfaces into model router.
   - Validate voice realtime latency against p95 target under representative load.
7. Integrations Completion
   - Deliver 3 production-ready connectors (calendar, mail, files/notifications) with scoped permissions.
   - Add health probes, isolation, and failure containment runbooks for connectors.
   - Ensure automation runs are auditable and reversible (including dead-letter replay workflow).
8. DoD + Governance Closure
   - Add backward compatibility notes and migration notes for changed interfaces.
   - Add rollback playbooks for high-risk feature toggles.
   - Complete security/policy review checklist for all Sprint 9 merges.
9. Metrics Closure
   - Reliability: success rate, completion rate, MTTR.
   - Latency: p50/p95 for query/task/voice.
   - Quality: correction rate, plan success rate, tool-call accuracy.
   - Cost: local-vs-API ratio, fallback frequency, spend by task class.
   - Safety: blocked unauthorized actions, approval bypass count.

Sprint 9 Exit Criteria:
1. All Phase 0-6 deliverables and exit criteria are demonstrably satisfied.
2. CI passes all unit/integration suites and smoke gates.
3. SLO dashboards and alerts are live and validated.
4. Required connectors are production-ready with guardrails.
5. Global DoD items are completed and documented.

## 14) Sprint 9 Execution in 3 Parts

Progress:
1. Part 1: Complete
2. Part 2: Complete
3. Part 3: In Progress
4. Part 1 checkpoints completed:
   - Operation-level SLO metrics endpoint and instrumentation for API/task/skill/connector/automation flows.
   - CI smoke coverage includes policy-gated high-risk flow (approval required for `run_command:ps`).
   - SLO threshold evaluation payloads and violation reporting available via `/api/v1/metrics`.
5. Part 2 checkpoints completed:
   - Modality-aware routing context wired across API query, voice callback payloads, and conversation manager.
   - Context fusion metadata persisted per turn and propagated into model-router requests.
   - Episodic learning hooks added for orchestrator task outcomes and conversation turn traces.
   - Agent execution paths now use model-router generation where applicable (developer/manager capability handlers).
   - Route shadow-mode support added for secondary provider telemetry comparison.
   - Production-like routing tests expanded for provider outage, modality mismatch fallback, and high-privacy modality gating.
   - Synthetic voice p95 latency validation harness added with configurable load profile and target threshold checks.
6. Part 3 checkpoints completed (initial):
   - Added production-profile connector pack (calendar, mail, files/notifications) with scoped permissions.
   - Added connector health probe endpoints and circuit/failure visibility in API responses.
   - Added automation dead-letter replay + manual resolve workflow with audit lineage.
   - Added migration notes, rollback playbook, and connector failure-containment runbook.
   - Added SLO gauges/thresholds for connector health and dead-letter backlog.
   - Added executable release-readiness checker script for repeatable go/no-go validation.
7. Part 3 final closure pending:
   - Full CI unit/integration execution after test dependencies are installed in runtime.

### Part 1: Stability + Governance Foundation
Goal: Close baseline reliability, CI, policy consistency, and observability foundations.

Scope:
1. Baseline + CI
2. SLO + dashboards instrumentation baseline
3. Agentic reliability guardrails (pre-check/verify + policy enforcement skeleton)

Tasks:
1. Lock and validate test/runtime dependencies; make unit + integration pipeline consistently runnable.
2. Add smoke gates for API/orchestrator/router/approvals/connectors/automation in CI.
3. Define and emit core SLO metrics (query/task/voice latency, error-rate, success-rate).
4. Add route/audit metrics for approval-gated and high-risk operations.
5. Implement pre-check/execute/verify scaffolding for high-impact tool paths.
6. Add policy/approval assertion tests for all high-risk action surfaces.

Acceptance Criteria:
1. CI passes baseline unit/integration + smoke suites.
2. Core SLO panels are populated from running system telemetry.
3. High-risk operations consistently fail closed without policy/approval.

---

### Part 2: Intelligence + Memory + Multimodal Core
Goal: Complete hybrid intelligence behavior, memory quality loop, and modality-aware routing.

Scope:
1. Hybrid routing productionization across agent paths
2. Memory learning-loop completion
3. Multimodal context-fusion core

Tasks:
1. Route agent execution paths through model router where applicable.
2. Add shadow-mode routing toggle and decision/fallback telemetry.
3. Add production-like routing tests (provider outage, privacy routing, modality mismatch).
4. Integrate episodic outcomes into planning heuristics.
5. Add freshness/confidence thresholds and enforcement in retrieval path.
6. Wire modality-aware requests (text/voice/image/audio) into router request model.
7. Add context fusion layer linking conversation, voice, and modality context metadata.
8. Validate voice p95 latency target under representative load tests.

Acceptance Criteria:
1. Hybrid router is active across conversation + agent paths with fallback evidence.
2. Memory retrieval uses freshness/confidence gates with measurable quality gains.
3. Modality-aware routing works and voice latency target is met in test profile.

---

### Part 3: Production Integrations + Final Closure
Goal: Complete connector production readiness, reversibility, and full DoD closure.

Scope:
1. Deep integration completion
2. Automation governance and reversibility
3. Global DoD and metrics closure

Tasks:
1. Deliver 3 production-ready connectors (calendar, mail, files/notifications) with scoped permissions.
2. Add connector health probes, isolation behavior, and failure containment runbooks.
3. Add automation reversal/replay workflow from dead-letter queue with audit lineage.
4. Add migration/backward-compat notes for changed interfaces.
5. Add rollback playbooks and release controls for high-risk toggles.
6. Finalize reliability/latency/quality/cost/safety metrics and alert thresholds.
7. Run final end-to-end validation and produce release-readiness checklist.

Acceptance Criteria:
1. Three connectors are production-ready with guardrails and observability.
2. Automation flows are auditable and reversible.
3. All Phase 0-6 criteria and global DoD items are marked complete.

## Status Tracker
- Phase 0: In Progress
- Phase 1: In Progress
- Phase 2: In Progress
- Phase 3: In Progress
- Phase 4: In Progress
- Phase 5: In Progress
- Phase 6: In Progress

---

## 15) Productivity Expansion Roadmap (Post Sprint 9)

Objective: Evolve JARVIS from feature-complete foundation into a high-productivity autonomous assistant for research, engineering execution, personal operations, and multimodal system organization.

### Phase 7: Autonomous Planning + Verification (Sprint 10)
Goal: Make multi-agent execution dependable for long-horizon tasks.

Scope:
1. Hierarchical planner (goal -> milestones -> tasks -> subtasks)
2. Verifier agent lane (tests, security, quality, deployment checks)
3. Recovery control loop (retry/replan/fallback/escalate)

Tasks:
1. Implement plan graph with dependency tracking and resumability.
2. Add explicit verifier roles for code quality, security, and release checks.
3. Add task-level confidence scoring and “require-human” thresholds.
4. Add durable project memory for reusable playbooks and prior outcomes.

Acceptance Criteria:
1. End-to-end multi-step workflows can pause/resume without losing context.
2. Critical actions require verifier pass or human approval.
3. Agent success rate and completion rate improve measurably.
Status: Complete (Implementation) / Validation Pending
Progress:
1. Added plan-graph submission API (`submit_task_plan`) with `PlanStep` dependency mapping.
2. Added confidence-based task escalation to `WAITING_APPROVAL` + runtime `approve_task(...)`.
3. Added post-execution verifier capability hook (`verifier_capability`) with fail-closed behavior.
4. Added plan status/recovery APIs (`get_plan_status`, `retry_task`, `replan_task`) for resumable execution.
5. Added optional plan-record persistence (`plan_persist_path`) and plan-level counters in system status.
6. Added API endpoints for plan submission/status and task retry/replan flows.
7. Added controlled auto-replan policy hook for failed tasks (`auto_replan_*` metadata).

### Phase 8: Research Intelligence + News/Blog Monitoring (Sprint 11)
Goal: Deliver trustworthy, source-grounded research at scale.

Scope:
1. Retrieval pipeline for latest web/news/blog sources
2. Source ranking, dedupe, contradiction detection
3. Topic subscriptions and periodic digests

Tasks:
1. Build source adapters with freshness metadata and citation requirements.
2. Add relevance scoring + duplicate collapse + source diversity constraints.
3. Add conflict detector for contradictory claims across sources.
4. Add user topic watchlists, daily/weekly digests, and alert triggers.

Acceptance Criteria:
1. Research responses include citations and freshness timestamps.
2. Latest topic digests run on schedule and are user-personalized.
3. Hallucination-like unsupported claims are reduced via evidence gating.
Status: Complete (Implementation) / Validation Pending
Progress:
1. Added research intelligence core (`ResearchIntelligenceEngine`) with source ingest, dedupe, ranking, citations, and contradiction flags.
2. Added watchlist + digest generation support (topic subscriptions with digest sections).
3. Added API endpoints for research ingest/query/watchlists/digest.
4. Added cadence-aware due-digest runner and watchlist last-digest tracking.
5. Added stricter freshness/trust filtering and citation health scoring in research query responses.
6. Added research source-adapter abstraction (register/list/run adapters) and default DuckDuckGo adapter wiring.
7. Added automation action hooks for research digests/adapter ingestion so rules can trigger research pipelines.

### Phase 9: Production Software Delivery Engine (Sprint 12)
Goal: Build projects from prompt to production safely.

Scope:
1. SDLC automation (design -> code -> test -> secure -> deploy)
2. Stack-aware project templates (frontend/backend/full-stack)
3. CI/CD and rollback automation

Tasks:
1. Add project bootstrap templates for major stacks and cloud targets.
2. Add mandatory lint/test/SAST/dependency audit gates.
3. Add deployment profiles (dev/stage/prod) with release controls.
4. Add post-deploy observability checks and auto-rollback triggers.

Acceptance Criteria:
1. JARVIS can generate and deploy reference services with passing gates.
2. Deployment failures auto-trigger rollback and incident notes.
3. Build-to-release lead time is tracked and improving.

### Phase 10: Personal Ops (Email + File Intelligence) (Sprint 13)
Goal: Make JARVIS useful for daily personal productivity workflows.

Scope:
1. Gmail/Outlook/IMAP operations with scoped permissions
2. File read/summarize and knowledge extraction
3. Image-folder relevance grouping and organization workflows

Tasks:
1. Add provider-grade email connectors (OAuth lifecycle, rate-limit handling).
2. Add email triage modes: draft, classify, prioritize, schedule, follow-up.
3. Add secure file indexing with ACL-aware retrieval and summarization.
4. Add image embedding pipeline for clustering, dedupe, and relevance sorting.
5. Add safe “preview + apply + undo” for file/image organization operations.

Acceptance Criteria:
1. Email workflows are auditable, permission-scoped, and reversible.
2. Large file corpora can be summarized with confidence/freshness metadata.
3. Image organization supports preview and one-click rollback.

### Phase 11: Human-Like Interaction + Proactive Assistance (Sprint 14+)
Goal: Move toward “Jarvis-like” usability, not just task execution.

Scope:
1. Proactive context-aware recommendations
2. Fast multimodal interaction with interruptions and continuity
3. Better preference modeling and personalization

Tasks:
1. Add proactive event engine (calendar, tasks, anomalies, reminders).
2. Improve conversational continuity across text/voice/image contexts.
3. Expand user preference model (tone, cadence, risk tolerance, routines).
4. Add strict safety policy for autonomous proactive actions.

Acceptance Criteria:
1. Proactive suggestions are useful, timely, and non-intrusive.
2. Cross-modal conversations maintain coherent context.
3. Autonomy remains policy-safe with zero approval bypass.

### Cross-Phase Guardrails (Mandatory)
1. Security: least-privilege scopes, secrets management, policy-as-code tests.
2. Reliability: unit/integration/e2e/load suites with SLO alerting live.
3. Governance: full audit lineage, migration notes, rollback playbooks.
4. Cost: local-vs-API budget controls and model routing spend dashboards.

### Release Metric Targets
1. Reliability: workflow completion >= 95%, MTTR < 30 min.
2. Latency: query p95 <= 1500 ms, voice p95 <= 900 ms in target profile.
3. Quality: correction rate down trend; verifier pass rate up trend.
4. Safety: approval bypass count = 0; unauthorized action blocks = 100%.
5. Productivity: measurable time saved on research/coding/email/file ops.
