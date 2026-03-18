# Delivery Engine Runbook

## Purpose

Operational playbook for `POST /api/v1/delivery/releases/run` with global config from `JARVIS_DELIVERY_*`.

## Pre-Flight Checklist

1. Verify API is healthy: `GET /api/v1/health`
2. Verify delivery runtime config: `GET /api/v1/delivery/capabilities`
3. Confirm required gate inputs or gate commands are configured.
4. Confirm deploy target is allowed in `JARVIS_DELIVERY_ALLOWED_DEPLOY_TARGETS`.

## Global Config Switches

- `JARVIS_DELIVERY_COMMAND_EXECUTION_ENABLED`
- `JARVIS_DELIVERY_COMMAND_TIMEOUT_SECONDS`
- `JARVIS_DELIVERY_MAX_OUTPUT_CHARS`
- `JARVIS_DELIVERY_DEPLOY_MAX_RETRIES`
- `JARVIS_DELIVERY_DEPLOY_RETRY_BACKOFF_SECONDS`
- `JARVIS_DELIVERY_ALLOWED_DEPLOY_TARGETS`
- `JARVIS_DELIVERY_DEFAULT_WORKING_DIR`
- `JARVIS_DELIVERY_LOCAL_DEPLOY_COMMAND`
- `JARVIS_DELIVERY_AWS_DEPLOY_COMMAND`
- `JARVIS_DELIVERY_GCP_DEPLOY_COMMAND`
- `JARVIS_DELIVERY_VERCEL_DEPLOY_COMMAND`

## Command Templates

Use these as starting points and adapt to your stack.

### Local

```bash
JARVIS_DELIVERY_LOCAL_DEPLOY_COMMAND="python3 -c \"print('local deploy ok')\""
```

### AWS (example placeholder)

```bash
JARVIS_DELIVERY_AWS_DEPLOY_COMMAND="python3 -c \"print('aws deploy placeholder')\""
```

### GCP (example placeholder)

```bash
JARVIS_DELIVERY_GCP_DEPLOY_COMMAND="python3 -c \"print('gcp deploy placeholder')\""
```

### Vercel (example placeholder)

```bash
JARVIS_DELIVERY_VERCEL_DEPLOY_COMMAND="python3 -c \"print('vercel deploy placeholder')\""
```

## Release Execution Flows

### 0) Reference Smoke Script

```bash
python scripts/delivery_reference_smoke.py
python scripts/delivery_reference_smoke.py --real-gates
```

### A) Config-Driven Deploy Command

```bash
curl -X POST http://127.0.0.1:8080/api/v1/delivery/releases/run \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "demo-service",
    "profile": "prod",
    "deploy_target": "aws",
    "approved": true,
    "context": {
      "gates": {
        "lint": true,
        "test": true,
        "sast": true,
        "dependency_audit": true
      }
    }
  }'
```

### B) Per-Run Gate and Deploy Commands

```bash
curl -X POST http://127.0.0.1:8080/api/v1/delivery/releases/run \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "demo-service",
    "profile": "prod",
    "deploy_target": "aws",
    "approved": true,
    "context": {
      "gate_commands": {
        "lint": ["python3", "-c", "raise SystemExit(0)"],
        "test": ["python3", "-c", "raise SystemExit(0)"],
        "sast": ["python3", "-c", "raise SystemExit(0)"],
        "dependency_audit": ["python3", "-c", "raise SystemExit(0)"]
      },
      "deploy_commands": {
        "aws": ["python3", "-c", "raise SystemExit(0)"]
      },
      "command_timeout_seconds": 120,
      "max_output_chars": 2000
    }
  }'
```

## Failure Triage

1. If release status is `blocked`:
- Inspect `pipeline.failed_gates`
- Re-run with corrected gate inputs/commands

2. If release status is `waiting_approval`:
- Re-run with `approved=true` after human approval

3. If release status is `rolled_back`:
- Check `release.rollback_reason` and `release.incident_note`
- Validate `deploy.exit_code`, `deploy.stderr`, and command timeout settings

4. If deploy target rejected:
- Update `JARVIS_DELIVERY_ALLOWED_DEPLOY_TARGETS`

## Rollback Playbook

1. Freeze deploy path:
- Set `JARVIS_DELIVERY_COMMAND_EXECUTION_ENABLED=false`
- Restart API service

2. Restrict targets:
- Set `JARVIS_DELIVERY_ALLOWED_DEPLOY_TARGETS=local`
- Restart API service

3. Re-enable gradually:
- Add targets one by one (`aws`, `gcp`, `vercel`)
- Validate with low-risk `profile=dev` first

## Security Notes

1. Prefer fixed command templates in global config for production.
2. Avoid passing secrets in request payloads.
3. Keep output truncation enabled (`JARVIS_DELIVERY_MAX_OUTPUT_CHARS`).
4. Keep strict `allowed_deploy_targets` in production.
