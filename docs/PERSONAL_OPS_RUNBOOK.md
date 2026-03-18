# Personal Ops Runbook (Phase 10)

## Purpose

Operational guide for Personal Ops workflows exposed via typed API endpoints:

- `POST /api/v1/email/{operation}`
- `POST /api/v1/files/intel/{operation}`
- `POST /api/v1/images/intel/{operation}`

All endpoints enforce connector policy scopes and audit logging.

## Required Scopes

### Email Ops (`email_ops`)

- `connector:email:oauth:write`
- `connector:email:write`
- `connector:email:read`
- `connector:email:draft`
- `connector:email:triage`
- `connector:email:schedule`
- `connector:email:send`
- `connector:email:undo`
- `connector:email:audit`

### File Intel (`file_intel`)

- `connector:file_intel:index`
- `connector:file_intel:read`
- `connector:file_intel:write`

### Image Intel (`image_intel`)

- `connector:image_intel:read`
- `connector:image_intel:plan`
- `connector:image_intel:write`
- `connector:image_intel:audit`

## Email Ops Examples

### 1) Connect account (OAuth)

```bash
curl -X POST http://127.0.0.1:8080/api/v1/email/oauth_connect \
  -H "Content-Type: application/json" \
  -H "X-Scopes: connector:email:oauth:write" \
  -d '{
    "account_id": "acc-demo-1",
    "provider": "gmail",
    "access_token": "token",
    "refresh_token": "refresh",
    "expires_in_sec": 3600
  }'
```

### 2) Ingest and triage a message

```bash
curl -X POST http://127.0.0.1:8080/api/v1/email/ingest_inbox \
  -H "Content-Type: application/json" \
  -H "X-Scopes: connector:email:write" \
  -d '{
    "account_id": "acc-demo-1",
    "messages": [{
      "message_id": "m-1",
      "from": "alice@example.com",
      "subject": "Need status",
      "body": "Please send latest update"
    }]
  }'

curl -X POST http://127.0.0.1:8080/api/v1/email/classify \
  -H "Content-Type: application/json" \
  -H "X-Scopes: connector:email:triage" \
  -d '{
    "account_id": "acc-demo-1",
    "message_id": "m-1",
    "label": "important"
  }'
```

### 3) Undo latest reversible action

```bash
curl -X POST http://127.0.0.1:8080/api/v1/email/undo \
  -H "Content-Type: application/json" \
  -H "X-Scopes: connector:email:undo" \
  -d '{
    "account_id": "acc-demo-1"
  }'
```

## File Intel Examples

### 1) Index a file

```bash
curl -X POST http://127.0.0.1:8080/api/v1/files/intel/index_file \
  -H "Content-Type: application/json" \
  -H "X-Scopes: connector:file_intel:index" \
  -d '{
    "path": "project.txt",
    "acl_tags": ["team-a"]
  }'
```

### 2) Summarize indexed doc (ACL aware)

```bash
curl -X POST http://127.0.0.1:8080/api/v1/files/intel/summarize_indexed \
  -H "Content-Type: application/json" \
  -H "X-Scopes: connector:file_intel:read" \
  -d '{
    "doc_id": "doc-id-here",
    "actor_acl_tags": ["team-a"],
    "max_chars": 1200
  }'
```

## Image Intel Examples

### 1) Preview organization plan

```bash
curl -X POST http://127.0.0.1:8080/api/v1/images/intel/preview_organize \
  -H "Content-Type: application/json" \
  -H "X-Scopes: connector:image_intel:plan" \
  -d '{
    "path": "source",
    "target_root": "organized",
    "strategy": "by_relevance",
    "recursive": true
  }'
```

### 2) Apply and rollback

```bash
curl -X POST http://127.0.0.1:8080/api/v1/images/intel/apply_plan \
  -H "Content-Type: application/json" \
  -H "X-Scopes: connector:image_intel:write" \
  -d '{"plan_id": "img-plan-..."}'

curl -X POST http://127.0.0.1:8080/api/v1/images/intel/undo_plan \
  -H "Content-Type: application/json" \
  -H "X-Scopes: connector:image_intel:write" \
  -d '{"plan_id": "img-plan-..."}'
```

## Validation Checklist

1. OAuth account connects and refreshes.
2. Email triage operations write audit trail and allow undo for reversible actions.
3. File summaries return confidence and freshness metadata.
4. ACL tags block unauthorized file summary retrieval.
5. Image organization supports preview, apply, and undo with no data loss.
