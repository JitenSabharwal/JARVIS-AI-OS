#!/usr/bin/env python3
"""
Local GitHub MCP server (HTTP JSON-RPC) for JARVIS.

Implements:
- tools/list
- tools/call
  - search_repositories
  - user-info
"""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _gh_get(url: str, token: str = "", timeout_s: float = 12.0) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "jarvis-github-mcp/0.1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers, method="GET")
    with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8", errors="ignore")
    return json.loads(raw or "{}")


def _search_repositories(arguments: dict[str, Any], token: str) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        return {"ok": False, "error": "query is required"}
    max_results = max(1, min(15, int(arguments.get("max_results", 6) or 6)))
    url = f"https://api.github.com/search/repositories?q={quote_plus(query)}&per_page={max_results}"
    payload = _gh_get(url, token=token)
    items = []
    for row in list(payload.get("items", []))[:max_results]:
        if not isinstance(row, dict):
            continue
        items.append(
            {
                "name": str(row.get("full_name", "")).strip() or str(row.get("name", "")).strip(),
                "html_url": str(row.get("html_url", "")).strip(),
                "description": str(row.get("description", "")).strip(),
                "language": str(row.get("language", "")).strip(),
                "stargazers_count": int(row.get("stargazers_count", 0) or 0),
                "updated_at": str(row.get("updated_at", "")).strip(),
            }
        )
    return {
        "ok": True,
        "query": query,
        "count": len(items),
        "results": items,
        "captured_at": _now_iso(),
    }


def _user_info(arguments: dict[str, Any], token: str) -> dict[str, Any]:
    username = str(arguments.get("username", "")).strip()
    if not username:
        return {"ok": False, "error": "username is required"}
    url = f"https://api.github.com/users/{quote_plus(username)}"
    row = _gh_get(url, token=token)
    if not isinstance(row, dict) or row.get("message") == "Not Found":
        return {"ok": False, "error": "user not found", "username": username}
    return {
        "ok": True,
        "username": username,
        "name": str(row.get("name", "")).strip(),
        "bio": str(row.get("bio", "")).strip(),
        "company": str(row.get("company", "")).strip(),
        "html_url": str(row.get("html_url", "")).strip(),
        "public_repos": int(row.get("public_repos", 0) or 0),
        "followers": int(row.get("followers", 0) or 0),
        "captured_at": _now_iso(),
    }


def _tool_list() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": "search_repositories",
                "description": "Search GitHub repositories by query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["query"],
                    "additionalProperties": True,
                },
            },
            {
                "name": "user-info",
                "description": "Get a GitHub user's public profile data.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "username": {"type": "string"},
                    },
                    "required": ["username"],
                    "additionalProperties": True,
                },
            },
        ]
    }


def _dispatch_jsonrpc(payload: dict[str, Any], token: str) -> dict[str, Any]:
    rid = payload.get("id")
    method = str(payload.get("method", "")).strip()
    params = payload.get("params", {})
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": _tool_list()}
    if method == "tools/call":
        if not isinstance(params, dict):
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": "invalid params"}}
        name = str(params.get("name", "")).strip()
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        if name == "search_repositories":
            return {"jsonrpc": "2.0", "id": rid, "result": _search_repositories(arguments, token)}
        if name == "user-info":
            return {"jsonrpc": "2.0", "id": rid, "result": _user_info(arguments, token)}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown tool: {name}"}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown method: {method}"}}


class _Handler(BaseHTTPRequestHandler):
    server_version = "jarvis-github-mcp/0.1"
    token = ""

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/health", "/healthz"}:
            self._write_json({"ok": True, "service": "github-mcp", "ts": _now_iso()})
            return
        self._write_json({"ok": False, "error": "use POST json-rpc"}, status=405)

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except Exception:
            length = 0
        if length <= 0:
            self._write_json({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "empty body"}}, status=400)
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._write_json({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "invalid json"}}, status=400)
            return
        if not isinstance(payload, dict):
            self._write_json({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}, status=400)
            return
        reply = _dispatch_jsonrpc(payload, self.token)
        self._write_json(reply, status=200)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local GitHub MCP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""))
    args = parser.parse_args()
    _Handler.token = str(args.token or "").strip()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"github-mcp listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

