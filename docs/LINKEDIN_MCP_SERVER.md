# LinkedIn MCP Server (Local)

This project now includes a local LinkedIn MCP server:

- Script: `scripts/linkedin_mcp_server.py`
- Protocol: HTTP JSON-RPC
- Methods:
1. `tools/list`
2. `tools/call`

Supported tools:
1. `user-info`
2. `profile-search`

## Run

```bash
python3 scripts/linkedin_mcp_server.py --host 127.0.0.1 --port 8765
```

Health:

```bash
curl -sS http://127.0.0.1:8765/health
```

## Wire into JARVIS

Set in `.env`:

```bash
JARVIS_LINKEDIN_MCP_ENDPOINT=http://127.0.0.1:8765
JARVIS_LINKEDIN_MCP_TOOL_ENRICH=user-info
JARVIS_LINKEDIN_MCP_AUTH_TOKEN=
```

Restart backend after updating env.

## Notes

1. `user-info` works best with a direct LinkedIn profile URL (`/in/...`).
2. Without a direct URL it returns a LinkedIn people-search URL and guidance.
3. API/site availability and page protections can limit metadata extraction.
