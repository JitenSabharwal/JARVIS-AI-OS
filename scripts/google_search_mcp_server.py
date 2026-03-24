#!/usr/bin/env python3
"""
Local Google Search MCP server (HTTP JSON-RPC) for JARVIS.

Implements:
- tools/list
- tools/call
  - google_search

Primary mode:
- Google Custom Search JSON API (if GOOGLE_API_KEY + GOOGLE_CSE_ID are set)

Fallback:
- DuckDuckGo Instant Answer API for lightweight open search.
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


def _http_get_json(url: str, timeout_s: float = 12.0) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": "jarvis-google-mcp/0.1"}, method="GET")
    with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8", errors="ignore")
    return json.loads(raw or "{}")


def _google_cse_search(query: str, max_results: int, api_key: str, cse_id: str) -> list[dict[str, Any]]:
    num = max(1, min(10, int(max_results)))
    url = (
        "https://www.googleapis.com/customsearch/v1"
        f"?key={quote_plus(api_key)}&cx={quote_plus(cse_id)}&q={quote_plus(query)}&num={num}"
    )
    payload = _http_get_json(url)
    out: list[dict[str, Any]] = []
    for row in list(payload.get("items", []))[:num]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "title": str(row.get("title", "")).strip(),
                "url": str(row.get("link", "")).strip(),
                "snippet": str(row.get("snippet", "")).strip(),
            }
        )
    return out


def _duckduckgo_search(query: str, max_results: int) -> list[dict[str, Any]]:
    url = (
        "https://api.duckduckgo.com/"
        f"?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
    )
    payload = _http_get_json(url)
    rows = []
    for item in list(payload.get("RelatedTopics", [])):
        if isinstance(item, dict) and isinstance(item.get("Topics"), list):
            rows.extend(item.get("Topics", []))
        else:
            rows.append(item)
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("Text", "")).strip()
        href = str(row.get("FirstURL", "")).strip()
        if not text and not href:
            continue
        out.append(
            {
                "title": text[:120],
                "url": href,
                "snippet": text[:260],
            }
        )
        if len(out) >= max(1, min(10, int(max_results))):
            break
    return out


def _google_search(arguments: dict[str, Any], api_key: str, cse_id: str) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        return {"ok": False, "error": "query is required"}
    max_results = max(1, min(10, int(arguments.get("max_results", 6) or 6)))
    results: list[dict[str, Any]]
    source = "duckduckgo_fallback"
    if api_key and cse_id:
        try:
            results = _google_cse_search(query, max_results, api_key, cse_id)
            source = "google_custom_search"
        except Exception:
            results = _duckduckgo_search(query, max_results)
            source = "duckduckgo_fallback"
    else:
        results = _duckduckgo_search(query, max_results)
    return {
        "ok": True,
        "query": query,
        "count": len(results),
        "source": source,
        "results": results,
        "captured_at": _now_iso(),
    }


def _tool_list() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": "google_search",
                "description": "Search the web by query and return ranked result snippets.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["query"],
                    "additionalProperties": True,
                },
            }
        ]
    }


def _dispatch_jsonrpc(payload: dict[str, Any], api_key: str, cse_id: str) -> dict[str, Any]:
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
        if name == "google_search":
            return {"jsonrpc": "2.0", "id": rid, "result": _google_search(arguments, api_key, cse_id)}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown tool: {name}"}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown method: {method}"}}


class _Handler(BaseHTTPRequestHandler):
    server_version = "jarvis-google-mcp/0.1"
    google_api_key = ""
    google_cse_id = ""

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
            self._write_json({"ok": True, "service": "google-search-mcp", "ts": _now_iso()})
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
        reply = _dispatch_jsonrpc(payload, self.google_api_key, self.google_cse_id)
        self._write_json(reply, status=200)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Google Search MCP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--google-api-key", default=os.getenv("GOOGLE_API_KEY", ""))
    parser.add_argument("--google-cse-id", default=os.getenv("GOOGLE_CSE_ID", ""))
    args = parser.parse_args()
    _Handler.google_api_key = str(args.google_api_key or "").strip()
    _Handler.google_cse_id = str(args.google_cse_id or "").strip()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"google-search-mcp listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

