# GitHub + Google MCP Integration

This JARVIS build supports MCP-first enrichment for:

1. LinkedIn (`linkedin_mcp`)
2. GitHub (`github_mcp`)
3. Google Search (`google_search_mcp`)

## How it works

For conversation-triggered enrichment jobs:

1. Parse source intent from request text (`linkedin`, `github`, `google`)
2. Try matching MCP connector first
3. On MCP failure/unavailable, fall back to existing web/research enrichment

## Environment variables

Set in `.env`:

```bash
JARVIS_LINKEDIN_MCP_ENDPOINT=
JARVIS_LINKEDIN_MCP_TOOL_ENRICH=user-info
JARVIS_LINKEDIN_MCP_AUTH_TOKEN=

JARVIS_GITHUB_MCP_ENDPOINT=
JARVIS_GITHUB_MCP_TOOL_ENRICH=search_repositories
JARVIS_GITHUB_MCP_AUTH_TOKEN=

JARVIS_GOOGLE_MCP_ENDPOINT=
JARVIS_GOOGLE_MCP_TOOL_ENRICH=google_search
JARVIS_GOOGLE_MCP_AUTH_TOKEN=
```

## Notes

1. Tool names differ by MCP server implementation. Override `*_TOOL_ENRICH` as needed.
2. You can inspect loaded connectors at:
   - `GET /api/v1/connectors`
   - `GET /api/v1/connectors/health`
3. Enrichment status:
   - `GET /api/v1/world/enrichment/jobs/{job_id}`
   - `GET /api/v1/realtime/sessions/{session_id}/notifications`

## Local MCP servers included

You can run local built-in servers from this repo:

1. GitHub MCP:
   - `python3 scripts/github_mcp_server.py --host 127.0.0.1 --port 8766`
2. Google Search MCP:
   - `python3 scripts/google_search_mcp_server.py --host 127.0.0.1 --port 8767`

Then set:

```bash
JARVIS_GITHUB_MCP_ENDPOINT=http://127.0.0.1:8766
JARVIS_GOOGLE_MCP_ENDPOINT=http://127.0.0.1:8767
```
