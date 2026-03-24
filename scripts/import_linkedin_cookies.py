#!/usr/bin/env python3
"""
Import LinkedIn cookies into Playwright storage-state JSON.

Supports:
1) Raw cookie header string: "li_at=...; JSESSIONID=..."
2) JSON export file:
   - list of cookie objects
   - or {"cookies": [ ... ]}
3) Netscape cookies.txt format
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _parse_cookie_header(value: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    raw = str(value or "").strip()
    if not raw:
        return out
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    for part in parts:
        if "=" not in part:
            continue
        name, val = part.split("=", 1)
        n = name.strip()
        v = val.strip()
        if not n or not v:
            continue
        out.append({"name": n, "value": v})
    return out


def _parse_json_cookies(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = []
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("cookies"), list):
        rows = payload.get("cookies", [])
    else:
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        value = str(row.get("value", "")).strip()
        if not name or not value:
            continue
        out.append(
            {
                "name": name,
                "value": value,
                "domain": str(row.get("domain", "")).strip(),
                "path": str(row.get("path", "/")).strip() or "/",
                "secure": bool(row.get("secure", True)),
                "httpOnly": bool(row.get("httpOnly", False)),
                "sameSite": str(row.get("sameSite", "")).strip(),
                "expires": row.get("expires", row.get("expirationDate", row.get("expiry", None))),
            }
        )
    return out


def _parse_netscape_cookies(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return out
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 7:
            continue
        domain, _include_subdomains, cookie_path, secure_flag, expires, name, value = parts
        name = str(name).strip()
        value = str(value).strip()
        if not name or not value:
            continue
        try:
            exp_val: float | None = float(expires)
        except Exception:
            exp_val = None
        out.append(
            {
                "name": name,
                "value": value,
                "domain": str(domain).strip(),
                "path": str(cookie_path).strip() or "/",
                "secure": str(secure_flag).strip().upper() == "TRUE",
                "httpOnly": False,
                "sameSite": "Lax",
                "expires": exp_val,
            }
        )
    return out


def _normalize_same_site(value: str) -> str:
    low = str(value or "").strip().lower()
    if low == "strict":
        return "Strict"
    if low == "none":
        return "None"
    return "Lax"


def _normalize_cookie_row(row: dict[str, Any], default_expiry_ts: float) -> dict[str, Any]:
    name = str(row.get("name", "")).strip()
    value = str(row.get("value", "")).strip()
    domain = str(row.get("domain", "")).strip() or ".linkedin.com"
    if "linkedin.com" not in domain.lower():
        domain = ".linkedin.com"
    if not domain.startswith("."):
        # LinkedIn cookies generally work best for all subdomains.
        domain = f".{domain}"
    path = str(row.get("path", "/")).strip() or "/"
    secure = bool(row.get("secure", True))
    http_only = bool(row.get("httpOnly", False))
    same_site = _normalize_same_site(str(row.get("sameSite", "")).strip())
    expires_raw = row.get("expires", None)
    expires = default_expiry_ts
    if isinstance(expires_raw, (int, float)) and float(expires_raw) > 0:
        expires = float(expires_raw)
    return {
        "name": name,
        "value": value,
        "domain": domain,
        "path": path,
        "expires": expires,
        "httpOnly": http_only,
        "secure": secure,
        "sameSite": same_site,
    }


def _dedupe_cookies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = f"{row.get('name','')}|{row.get('domain','')}|{row.get('path','/')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _filter_linkedin(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        domain = str(row.get("domain", ".linkedin.com")).lower()
        if "linkedin.com" not in domain:
            continue
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Import LinkedIn cookies into Playwright storage-state JSON")
    parser.add_argument("--cookie-header", default="")
    parser.add_argument("--cookie-header-file", default="")
    parser.add_argument("--cookies-file", default="", help="JSON or Netscape cookies.txt export")
    parser.add_argument("--state-path", default="data/linkedin_storage_state.json")
    parser.add_argument("--expires-days", type=int, default=14)
    args = parser.parse_args()

    raw_rows: list[dict[str, Any]] = []

    if str(args.cookie_header).strip():
        raw_rows.extend(_parse_cookie_header(str(args.cookie_header)))

    if str(args.cookie_header_file).strip():
        p = Path(str(args.cookie_header_file)).expanduser()
        if p.exists():
            raw_rows.extend(_parse_cookie_header(p.read_text(encoding="utf-8", errors="ignore")))

    if str(args.cookies_file).strip():
        p = Path(str(args.cookies_file)).expanduser()
        if p.exists():
            parsed = _parse_json_cookies(p)
            if not parsed:
                parsed = _parse_netscape_cookies(p)
            raw_rows.extend(parsed)

    if not raw_rows:
        print("No cookies found. Provide --cookie-header or --cookies-file.")
        return 2

    expiry_days = max(1, int(args.expires_days or 14))
    default_expiry_ts = time.time() + float(expiry_days * 86400)
    normalized = []
    for row in raw_rows:
        n = str(row.get("name", "")).strip()
        v = str(row.get("value", "")).strip()
        if not n or not v:
            continue
        normalized.append(_normalize_cookie_row(row, default_expiry_ts))

    normalized = _filter_linkedin(normalized)
    normalized = _dedupe_cookies(normalized)
    if not normalized:
        print("No LinkedIn-domain cookies found after filtering.")
        return 3

    target = Path(str(args.state_path)).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cookies": normalized,
        "origins": [{"origin": "https://www.linkedin.com", "localStorage": []}],
    }
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    names = {str(c.get("name", "")) for c in normalized}
    has_li_at = "li_at" in names
    print(f"Saved storage state: {target}")
    print(f"Cookies imported: {len(normalized)} (li_at={'yes' if has_li_at else 'no'})")
    if not has_li_at:
        print("Warning: li_at cookie missing. LinkedIn auth may still fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
