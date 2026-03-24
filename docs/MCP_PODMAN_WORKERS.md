# MCP Workers On Podman

Run MCP servers in Podman instead of local terminal scripts.

## Modes

- `mcp-multi`: one container per MCP server (`linkedin`, `github`, `google`)
- `mcp-single`: one container running all three MCP servers

## Start (multi-worker recommended)

```bash
podman compose --env-file .env.podman --profile mcp-multi up -d \
  jarvis-mcp-linkedin jarvis-mcp-github jarvis-mcp-google
```

## Start (single worker)

```bash
podman compose --env-file .env.podman --profile mcp-single up -d jarvis-mcp-worker
```

## Stop

```bash
podman compose --env-file .env.podman --profile mcp-multi down
podman compose --env-file .env.podman --profile mcp-single down
```

## API endpoint wiring

If API runs on host macOS, set in `.env`:

```bash
JARVIS_LINKEDIN_MCP_ENDPOINT=http://127.0.0.1:8765
JARVIS_GITHUB_MCP_ENDPOINT=http://127.0.0.1:8766
JARVIS_GOOGLE_MCP_ENDPOINT=http://127.0.0.1:8767
JARVIS_LINKEDIN_STORAGE_STATE_PATH=/workspace/data/linkedin_storage_state.json
```

If API runs inside the same compose project, set:

```bash
JARVIS_LINKEDIN_MCP_ENDPOINT=http://jarvis-mcp-linkedin:8765
JARVIS_GITHUB_MCP_ENDPOINT=http://jarvis-mcp-github:8766
JARVIS_GOOGLE_MCP_ENDPOINT=http://jarvis-mcp-google:8767
JARVIS_LINKEDIN_STORAGE_STATE_PATH=/workspace/data/linkedin_storage_state.json
```

## LinkedIn one-time login bootstrap

Run on host once to create storage-state JSON:

```bash
pip install playwright
python -m playwright install chromium
python3 scripts/bootstrap_linkedin_playwright_auth.py --state-path data/linkedin_storage_state.json
```

If automated login is blocked, import cookies from your normal browser session instead:

```bash
python3 scripts/import_linkedin_cookies.py \
  --cookies-file /path/to/linkedin_cookies.json \
  --state-path data/linkedin_storage_state.json
```
