#!/usr/bin/env python3
"""
Local LinkedIn MCP server (HTTP JSON-RPC) for JARVIS.

Implements:
- tools/list
- tools/call (user-info, profile-search)

This server is intentionally lightweight and dependency-free.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_linkedin_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    if "linkedin.com" not in str(parsed.netloc or "").lower():
        return ""
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def _extract_profile_slug(url: str) -> str:
    parsed = urlparse(url)
    parts = [p for p in str(parsed.path or "").split("/") if p]
    if len(parts) >= 2 and parts[0].lower() in {"in", "pub"}:
        return parts[1].strip()
    return ""


def _search_url(query: str) -> str:
    q = quote_plus(str(query or "").strip())
    return f"https://www.linkedin.com/search/results/people/?keywords={q}"


def _safe_fetch(url: str, timeout_s: float = 8.0) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        ctype = str(resp.headers.get("Content-Type", "")).lower()
        if "text/html" not in ctype:
            return ""
        body = resp.read().decode("utf-8", errors="ignore")
        return body[:300_000]


def _extract_html_profile_fields(html: str) -> dict[str, str]:
    text = str(html or "")
    if not text:
        return {}
    title = ""
    headline = ""
    m_title = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if m_title:
        title = re.sub(r"\s+", " ", m_title.group(1)).strip()
    m_desc = re.search(
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_desc:
        headline = re.sub(r"\s+", " ", m_desc.group(1)).strip()
    return {
        "name": title[:160],
        "headline": headline[:240],
    }


def _profile_search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        return {"ok": False, "error": "query is required"}
    return {
        "ok": True,
        "query": query,
        "search_url": _search_url(query),
        "captured_at": _now_iso(),
    }


def _user_info(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    profile_url = _normalize_linkedin_url(str(arguments.get("profile_url", "")).strip())
    user_id = str(arguments.get("user_id", "")).strip()
    if not query and not profile_url:
        return {"ok": False, "error": "query or profile_url is required"}
    if not profile_url and query:
        profile_url = _search_url(query)

    out: dict[str, Any] = {
        "ok": True,
        "user_id": user_id,
        "query": query,
        "profile_url": profile_url,
        "source": "linkedin_mcp_local",
        "captured_at": _now_iso(),
    }
    slug = _extract_profile_slug(profile_url)
    if slug:
        out["profile_slug"] = slug

    # Best effort: for direct profile URLs only.
    if "/in/" in profile_url or "/pub/" in profile_url:
        try:
            html = _safe_fetch(profile_url)
            fields = _extract_html_profile_fields(html)
            for k, v in fields.items():
                if v:
                    out[k] = v
        except Exception as exc:  # noqa: BLE001
            out["fetch_error"] = str(exc)
    else:
        out["note"] = "Search URL returned. Provide a direct LinkedIn profile URL for richer extraction."
    return out


def _tool_list() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": "user-info",
                "description": "Get LinkedIn profile metadata using query or direct profile_url.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "profile_url": {"type": "string"},
                        "user_id": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
            {
                "name": "profile-search",
                "description": "Build a LinkedIn people-search URL for a query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": True,
                },
            },
        ]
    }


def _dispatch_jsonrpc(payload: dict[str, Any]) -> dict[str, Any]:
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
        if name == "user-info":
            return {"jsonrpc": "2.0", "id": rid, "result": _user_info(arguments)}
        if name == "profile-search":
            return {"jsonrpc": "2.0", "id": rid, "result": _profile_search(arguments)}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown tool: {name}"}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown method: {method}"}}


class _Handler(BaseHTTPRequestHandler):
    server_version = "jarvis-linkedin-mcp/0.1"

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
            self._write_json({"ok": True, "service": "linkedin-mcp", "ts": _now_iso()})
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
        reply = _dispatch_jsonrpc(payload)
        self._write_json(reply, status=200)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local LinkedIn MCP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"linkedin-mcp listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

