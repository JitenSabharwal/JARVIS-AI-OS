# Phase 12: Learning Domains Roadmap

## Goal
Expand JARVIS from coding + ops assistant into a broader personal and professional copilot with measurable outcomes.

## Domain Backlog (Priority Order)

### 1) Personal Productivity
- Scope: calendar planning, meeting prep, follow-up drafting, smart reminders, task prioritization.
- Data: meeting transcripts, task datasets, email-style drafting corpora.
- Connectors: calendar, email, notes, tasks.
- Eval: on-time follow-up rate, meeting-summary quality rubric, task-priority agreement score.

### 2) Finance Operations
- Scope: budgeting insights, invoice extraction, spend categorization, anomaly flags.
- Data: financial QA + accounting corpora.
- Connectors: docs/folders, accounting exports, spreadsheet APIs.
- Eval: extraction precision/recall, category F1, anomaly false-positive rate.

### 3) Customer & Support Ops
- Scope: ticket triage, intent tagging, risk detection, response drafting.
- Data: customer support/dialogue and classification datasets.
- Connectors: helpdesk + CRM.
- Eval: triage accuracy, SLA risk recall, response acceptance rate.

### 4) Legal & Compliance Workflows
- Scope: clause extraction, policy-to-control mapping, audit evidence summarization.
- Data: legal corpora already in manifests + policy docs.
- Connectors: document stores, internal policy repository.
- Eval: clause extraction F1, control mapping precision, audit trace completeness.

### 5) Data Analyst Mode
- Scope: spreadsheet QA, dashboard explanation, trend/anomaly narratives.
- Data: table QA + business analytics datasets.
- Connectors: sheets/BI exports.
- Eval: answer correctness on benchmark sheets, insight usefulness rubric.

### 6) Multimodal Design Copilot
- Scope: UI critique, wireframe-to-spec, visual QA, image edit instruction planning.
- Data: WebSight, UI structure datasets, image instruction/edit datasets.
- Connectors: design files, screenshot/image folders.
- Eval: spec completeness score, UI heuristic coverage, edit instruction accuracy.

### 7) Language & Voice Intelligence
- Scope: English/Hindi/German/Sanskrit understanding and translation (text + audio).
- Data: FLORES, FLEURS, CVSS, Sanskrit corpora.
- Connectors: voice pipeline, transcript service.
- Eval: BLEU/COMET proxy, ASR/WER proxy, human adequacy/fluency review.

### 8) Agent Self-Improvement Loop
- Scope: failure replay, tool-use reflection, plan-vs-outcome tuning.
- Data: internal run traces, audit events, task outcomes.
- Connectors: audit log + orchestrator telemetry.
- Eval: task success trend, retries per task, cost per successful completion.

## Execution Model
- Sprint A (Foundations): dataset ingestion + metadata schema + evaluation harness.
- Sprint B (Productivity/Finance/Support): ship connectors and domain prompts.
- Sprint C (Legal/Analyst/Design): enable multimodal and document-heavy workflows.
- Sprint D (Language/Self-improvement): cross-lingual voice + adaptive policy tuning.

## Cross-Phase Acceptance Criteria
- Security: least-privilege connector scopes and auditable actions.
- Reliability: each domain has unit + integration smoke tests.
- Cost: router policy bounds local/API usage per domain.
- Governance: decision logs include dataset source + policy version.

## Immediate Next Tasks
1. Add domain tags in dataset metadata (`coding`, `ops`, `design`, `language`, `legal`, `finance`, `support`).
2. Build `scripts/eval_phase12.py` scaffold with per-domain metric stubs.
3. Add connector capability matrix in `docs/PERSONAL_OPS_RUNBOOK.md`.
4. Extend development plan with a Phase 12 section and sprint milestones.
