#!/usr/bin/env python3
"""
Local LinkedIn MCP server (HTTP JSON-RPC) for JARVIS.

Implements:
- tools/list
- tools/call (user-info, profile-search)

Authentication model:
- Optional Playwright storage-state JSON can be provided.
- Cookies from that file are reused for LinkedIn requests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
    q = quote_plus(_linkedin_people_keywords(query))
    return f"https://www.linkedin.com/search/results/people/?keywords={q}"


def _linkedin_people_keywords(query: str) -> str:
    raw = str(query or "").strip()
    if not raw:
        return "person"
    low = raw.lower()
    low = re.sub(r"https?://\S+", " ", low)
    low = re.sub(r"[|,;:()\[\]{}]+", " ", low)
    low = re.sub(
        r"\b("
        r"linkedin|linked\s*in|profile|profiles|people|person|search|result|results|source|"
        r"learn|about|enrich|enrichment|fetch|get|find|lookup|crawl|scrape|from|web|website|internet|"
        r"my|me|for|of|on|the|a|an|please|queued|request|conversational|using|run|query"
        r")\b",
        " ",
        low,
    )
    cleaned = re.sub(r"\s+", " ", low).strip()
    if not cleaned:
        return "person"
    tokens = cleaned.split(" ")
    # Keep query concise for LinkedIn people-search behavior.
    return " ".join(tokens[:6]).strip() or "person"


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


def _looks_like_signin_page(html: str, final_url: str) -> bool:
    text = str(html or "").lower()
    furl = str(final_url or "").lower()
    if "linkedin.com/login" in furl or "linkedin.com/checkpoint" in furl:
        return True
    return (
        ("sign in" in text or "signin" in text)
        and ("linkedin" in text)
        and ("email or phone" in text or "password" in text)
    )


def _extract_first_profile_url_from_search(html: str) -> str:
    text = str(html or "")
    if not text:
        return ""
    # Prefer canonical /in/ links.
    for m in re.finditer(r"https://www\.linkedin\.com/in/[A-Za-z0-9_%\-]+/?", text):
        value = _normalize_linkedin_url(m.group(0))
        if value:
            return value
    return ""


@dataclass
class _AuthConfig:
    storage_state_path: str = ""
    cookie_header: str = ""

    def has_auth(self) -> bool:
        return bool(self.cookie_header)


def _load_cookie_header_from_storage_state(path: str) -> str:
    p = Path(str(path or "").strip()).expanduser()
    if not p.exists():
        return ""
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    rows = payload.get("cookies", [])
    if not isinstance(rows, list):
        return ""
    now = time.time()
    pairs: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        domain = str(row.get("domain", "")).lower()
        if "linkedin.com" not in domain:
            continue
        name = str(row.get("name", "")).strip()
        value = str(row.get("value", "")).strip()
        if not name or not value:
            continue
        expires = row.get("expires")
        if isinstance(expires, (int, float)) and float(expires) > 0 and float(expires) < now:
            continue
        pairs.append(f"{name}={value}")
    return "; ".join(pairs)


def _safe_fetch(url: str, timeout_s: float = 10.0, cookie_header: str = "") -> tuple[str, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
        )
    }
    if cookie_header:
        headers["Cookie"] = cookie_header
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        ctype = str(resp.headers.get("Content-Type", "")).lower()
        final_url = str(resp.geturl() or url)
        if "text/html" not in ctype:
            return "", final_url
        body = resp.read().decode("utf-8", errors="ignore")
        return body[:500_000], final_url


def _profile_search(arguments: dict[str, Any], auth: _AuthConfig) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        return {"ok": False, "error": "query is required"}
    normalized_query = _linkedin_people_keywords(query)
    url = _search_url(query)
    out = {
        "ok": True,
        "query": query,
        "normalized_query": normalized_query,
        "search_url": url,
        "authenticated": auth.has_auth(),
        "captured_at": _now_iso(),
    }
    if not auth.has_auth():
        out["note"] = "Set LinkedIn storage-state auth to open search/profile pages without login wall."
    return out


def _user_info(arguments: dict[str, Any], auth: _AuthConfig) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    normalized_query = _linkedin_people_keywords(query) if query else ""
    profile_url = _normalize_linkedin_url(str(arguments.get("profile_url", "")).strip())
    user_id = str(arguments.get("user_id", "")).strip()
    if not query and not profile_url:
        return {"ok": False, "error": "query or profile_url is required"}

    out: dict[str, Any] = {
        "ok": True,
        "user_id": user_id,
        "query": query,
        "normalized_query": normalized_query,
        "profile_url": profile_url,
        "source": "linkedin_mcp_local",
        "authenticated": auth.has_auth(),
        "captured_at": _now_iso(),
    }

    target_url = profile_url or _search_url(query)
    if not profile_url:
        out["search_url"] = target_url

    try:
        html, final_url = _safe_fetch(target_url, cookie_header=auth.cookie_header)
    except Exception as exc:  # noqa: BLE001
        return {**out, "ok": False, "error": f"fetch_failed: {exc}"}

    out["final_url"] = final_url

    if _looks_like_signin_page(html, final_url):
        out["ok"] = False
        out["auth_required"] = True
        out["error"] = "linkedin_login_required"
        out["note"] = (
            "LinkedIn redirected to sign-in. Provide Playwright storage-state cookies "
            "via --storage-state or LINKEDIN_STORAGE_STATE_PATH."
        )
        return out

    resolved_profile = profile_url
    if not resolved_profile and "/search/results/people" in final_url:
        resolved_profile = _extract_first_profile_url_from_search(html)
        if resolved_profile:
            out["profile_url"] = resolved_profile

    slug = _extract_profile_slug(out.get("profile_url", ""))
    if slug:
        out["profile_slug"] = slug

    fields = _extract_html_profile_fields(html)
    for k, v in fields.items():
        if v:
            out[k] = v

    if not out.get("profile_url"):
        out["note"] = "Search page fetched. Provide direct profile_url for richer extraction."
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


def _dispatch_jsonrpc(payload: dict[str, Any], auth: _AuthConfig) -> dict[str, Any]:
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
            return {"jsonrpc": "2.0", "id": rid, "result": _user_info(arguments, auth)}
        if name == "profile-search":
            return {"jsonrpc": "2.0", "id": rid, "result": _profile_search(arguments, auth)}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown tool: {name}"}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown method: {method}"}}


class _Handler(BaseHTTPRequestHandler):
    server_version = "jarvis-linkedin-mcp/0.2"
    auth = _AuthConfig()

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
            self._write_json(
                {
                    "ok": True,
                    "service": "linkedin-mcp",
                    "authenticated": self.auth.has_auth(),
                    "storage_state_path": self.auth.storage_state_path,
                    "ts": _now_iso(),
                }
            )
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
        reply = _dispatch_jsonrpc(payload, self.auth)
        self._write_json(reply, status=200)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local LinkedIn MCP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--storage-state",
        default=os.getenv("LINKEDIN_STORAGE_STATE_PATH", ""),
        help="Path to Playwright storage-state JSON for authenticated LinkedIn access.",
    )
    args = parser.parse_args()

    auth = _AuthConfig(storage_state_path=str(args.storage_state or "").strip())
    auth.cookie_header = _load_cookie_header_from_storage_state(auth.storage_state_path)
    _Handler.auth = auth

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(
        f"linkedin-mcp listening on http://{args.host}:{args.port} "
        f"(authenticated={'yes' if auth.has_auth() else 'no'})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
