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
python3 scripts/linkedin_mcp_server.py --host 127.0.0.1 --port 8765 \
  --storage-state data/linkedin_storage_state.json
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
LINKEDIN_STORAGE_STATE_PATH=data/linkedin_storage_state.json
```

Restart backend after updating env.

## One-time login bootstrap (Playwright)

```bash
pip install playwright
python -m playwright install chromium
python3 scripts/bootstrap_linkedin_playwright_auth.py --state-path data/linkedin_storage_state.json
```

Then restart LinkedIn MCP server with `--storage-state`.

## Alternative: import cookies from your normal browser session

If Google blocks sign-in in automation, login in your normal browser and export cookies,
then generate storage-state JSON:

```bash
python3 scripts/import_linkedin_cookies.py \
  --cookies-file /path/to/linkedin_cookies.json \
  --state-path data/linkedin_storage_state.json
```

Or paste a raw cookie header:

```bash
python3 scripts/import_linkedin_cookies.py \
  --cookie-header "li_at=...; JSESSIONID=...; bcookie=..." \
  --state-path data/linkedin_storage_state.json
```

## Notes

1. `user-info` works best with a direct LinkedIn profile URL (`/in/...`).
2. Without `--storage-state`, LinkedIn may redirect to sign-in.
3. With valid storage-state, the MCP uses session cookies for authenticated fetches.
4. Session state can expire; rerun bootstrap when needed.
