# Phase 11 Runbook: Human-Like Interaction + Proactive Assistance

## Scope
- Proactive event/suggestion pipeline.
- Cross-modal continuity (text/voice/image).
- Preference-driven personalization (tone, cadence, risk tolerance, routines).
- Autonomous proactive safety and approval controls.

## API Surface
- `POST /api/v1/proactive/events`
- `POST /api/v1/proactive/preferences`
- `GET /api/v1/proactive/suggestions`
- `GET /api/v1/proactive/profile/{user_id}`
- `POST /api/v1/proactive/suggestions/{suggestion_id}/ack`
- `POST /api/v1/proactive/suggestions/{suggestion_id}/dismiss`
- `POST /api/v1/proactive/suggestions/{suggestion_id}/snooze`
- `POST /api/v1/proactive/actions/execute`

## Operational Notes
- Proactive suggestions are deduped with per-user cooldown windows.
- Dismissed suggestions are hidden from default suggestion listing.
- Snoozed suggestions are hidden until `snoozed_until`.
- Autonomous proactive execution is blocked unless:
  - policy decision allows it, and
  - approval token is provided and validated when required.
- Orchestrator validates approval tokens against task action and rejects invalid/bypass tokens.

## Migration Notes
- No external schema migrations required.
- New in-memory suggestion lifecycle state fields:
  - `status`
  - `snoozed_until`
  - `acknowledged_at`
  - `dismissed_at`
- API consumers can continue using existing proactive endpoints unchanged.
- New endpoints are additive and backward compatible.

## Rollback Playbook
1. Disable proactive autonomous execution by setting user preference:
   - `autonomous_actions_enabled=false`
2. Stop sending requests to `POST /api/v1/proactive/actions/execute`.
3. Revert to recommendation-only mode:
   - use `events/preferences/suggestions/profile` endpoints only.
4. If needed, disable proactive recommendations entirely per user:
   - `proactive_enabled=false`.
5. Monitor:
   - `proactive_event_total` counters
   - API error rates
   - audit events with `event_type=proactive`

## Security Guardrails
- Least privilege approval actions should be capability-scoped (`orchestrator:execute:<capability>`).
- Never reuse broad approval actions for unrelated capabilities.
- Keep approval TTL short for high-risk actions.

## Validation Checklist (Non-test)
- Confirm proactive suggestions can be acknowledged/dismissed/snoozed through API.
- Confirm autonomous proactive action returns `403` without required approval token.
- Confirm same action returns `202` with valid approved token.
- Confirm audit entries are emitted for proactive suggestion/action endpoints.
