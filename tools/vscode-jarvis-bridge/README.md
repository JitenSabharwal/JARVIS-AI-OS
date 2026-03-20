# Jarvis Continue Bridge (VS Code)

This extension runs a local HTTP proxy that forwards Continue requests to Jarvis and injects active VS Code context headers:

- `X-Jarvis-Workspace`
- `X-Workspace-Path`
- `X-Jarvis-Active-File`
- `X-Jarvis-Selection`

## Why

Continue sends OpenAI-compatible chat requests, but not always the active workspace path.  
This bridge makes workspace-aware `/code`, `/repo`, and `/workflow` commands work without manually passing `--workspace`.

## Install (local development)

1. Open this folder in VS Code:
   - `tools/vscode-jarvis-bridge`
2. Press `F5` to launch Extension Development Host.
   - If VS Code shows "Select debugger", choose `Run Jarvis Bridge Extension`.
   - This repo includes `.vscode/launch.json` for that option.
3. In your main VS Code window, configure Continue to point to the bridge:

```yaml
models:
  - name: Jarvis via Bridge
    provider: openai
    model: jarvis-default
    apiBase: http://127.0.0.1:8787/v1
    apiKey: YOUR_JARVIS_API_TOKEN
```

## Extension settings

- `jarvisBridge.targetBaseUrl` (default `http://127.0.0.1:8080`)
- `jarvisBridge.listenPort` (default `8787`)
- `jarvisBridge.forceWorkspacePath` (default empty)
- `jarvisBridge.workspacePathMappings` (default `[]`)

### Recommended setup for containerized Jarvis

If Jarvis runs in a container and your code lives on host paths (for example `/Users/...`), map host paths to container paths so `/repo` and `/code` work without `--workspace`.

Example VS Code settings (`settings.json`):

```json
{
  "jarvisBridge.workspacePathMappings": [
    {
      "from": "/Users/sabharwal/Jiten/ai_research/JARVIS-AI-OS",
      "to": "/app"
    },
    {
      "from": "/Volumes/Jiten-2026/AI_SSD/ai-research/projects",
      "to": "/workspace/projects"
    }
  ]
}
```

For single-repo setups, easiest option:

```json
{
  "jarvisBridge.forceWorkspacePath": "/app"
}
```

## Commands

- `Jarvis Bridge: Restart`
- `Jarvis Bridge: Stop`

## Jarvis server requirements

Set allowlisted roots:

```env
JARVIS_CODE_ALLOWED_ROOTS=/Users/sabharwal:/runtime:/workspace
```

Jarvis API now reads workspace headers and can use them as default workspace hints.
