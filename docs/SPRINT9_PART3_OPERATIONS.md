# Sprint 9 Part 3 Operations Notes

## Backward Compatibility / Migration Notes

1. New API endpoints were added (non-breaking):
   - `GET /api/v1/connectors/health`
   - `GET /api/v1/connectors/{connector_name}/health`
   - `POST /api/v1/automation/dead-letters/replay`
   - `POST /api/v1/automation/dead-letters/{dead_letter_id}/resolve`
2. Dead-letter entries now include:
   - `dead_letter_id`
   - `replay_count`
   - `last_replay_at`
   - `last_replay_status`
   - `last_replay_result`
3. API mode startup now auto-registers production-profile connectors:
   - `calendar`
   - `mail`
   - `files_notifications`
4. Existing connector invoke and automation event endpoints remain unchanged.

## Connector Failure Containment Runbook

1. Detect degraded connectors:
   - Call `GET /api/v1/connectors/health`
   - Check `unhealthy` list and per-connector `circuit_open` state.
   - Check `GET /api/v1/status` for ingress pressure and automation backlog signals.
2. Isolate impact:
   - Circuit-open connectors reject calls with HTTP `503` and do not block other connectors.
   - Automation actions depending on failing connectors move to dead-letter queue.
3. Recover:
   - Fix root cause (credentials, endpoint, file permissions, provider outage).
   - Re-run failed tasks from dead letters via replay endpoint.
4. Validate:
   - Re-check connector health endpoint.
   - Confirm dead-letter replay success and reduced queue size.
   - Confirm `GET /api/v1/metrics` counters for:
     - `connector_invoke_total:*`
     - `automation_event_total:*`
     - `ingress_reject_total:*` (should stay near zero in normal load)

## Dead-Letter Replay / Reversal Workflow

1. Inspect failures:
   - `GET /api/v1/automation/dead-letters?limit=100`
2. Replay:
   - `POST /api/v1/automation/dead-letters/replay`
   - Body: `{"dead_letter_id":"...", "timeout_seconds":10, "remove_on_success":true}`
3. Resolve manually (reversal or accepted skip):
   - `POST /api/v1/automation/dead-letters/{dead_letter_id}/resolve`
   - Body: `{"reason":"manual_reversal_completed"}`
4. Audit:
   - Check `/api/v1/audit` for replay/resolve records.
   - Check `/api/v1/automation/history` for resolution lineage.

## Rollback Playbook (Feature Toggles)

1. If connector rollout causes regressions:
   - Remove connector registration in `jarvis_main.py` API mode.
   - Restart service and validate `/api/v1/health`.
2. If replay workflow causes instability:
   - Disable replay endpoint exposure in `interfaces/api_interface.py` route setup.
   - Continue operating with read-only dead-letter inspection endpoint.
3. If automation connector actions are unstable:
   - Remove `register_default_automation_actions(...)` call in API startup.
   - Existing non-connector automation actions continue to work.
