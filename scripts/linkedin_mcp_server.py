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
from html import unescape
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


def _humanize_slug(slug: str) -> str:
    text = str(slug or "").strip().strip("/")
    if not text:
        return ""
    text = re.sub(r"[-_]+", " ", text)
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return " ".join(part.capitalize() for part in text.split(" "))


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
    company = ""
    location = ""
    m_title = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if m_title:
        title = re.sub(r"\s+", " ", m_title.group(1)).strip()
    m_og_title = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_og_title:
        title = re.sub(r"\s+", " ", m_og_title.group(1)).strip() or title
    m_desc = re.search(
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_desc:
        headline = re.sub(r"\s+", " ", m_desc.group(1)).strip()
    # Try JSON-LD person schema blocks when visible.
    for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', text, flags=re.IGNORECASE | re.DOTALL):
        raw = str(m.group(1) or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        rows = obj if isinstance(obj, list) else [obj]
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("@type", "")).lower() not in {"person", "profilepage"}:
                continue
            name_val = str(row.get("name", "")).strip()
            if name_val:
                title = name_val
            works_for = row.get("worksFor")
            if isinstance(works_for, dict):
                company_val = str(works_for.get("name", "")).strip()
                if company_val:
                    company = company_val
            home_loc = row.get("homeLocation")
            if isinstance(home_loc, dict):
                loc_val = str(home_loc.get("name", "")).strip()
                if loc_val:
                    location = loc_val
            if not location:
                addr = row.get("address")
                if isinstance(addr, dict):
                    location = str(addr.get("addressLocality", "")).strip()
    if not company and headline:
        m_company = re.search(r"\bat\s+([A-Z][A-Za-z0-9&.,'() \-]{1,80})", headline)
        if m_company:
            company = m_company.group(1).strip(" .,!?\t\r\n")
    return {
        "name": title[:160],
        "headline": headline[:240],
        "company": company[:120],
        "location": location[:120],
    }


def _html_to_text(html: str) -> str:
    text = str(html or "")
    if not text:
        return ""
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_profile_text(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    # Remove high-frequency LinkedIn chrome/noise terms.
    noise_patterns = [
        r"\b(Home|My Network|Jobs|Messaging|Notifications|For Business|Try Premium)\b",
        r"\b(Open to|Add section|Enhance profile|Resources|Suggested for you|Analytics)\b",
        r"\b(Who your viewers also viewed|People you may know|Connect|View)\b",
        r"\b(500\+ connections|Contact info|Profile language|Public profile & URL)\b",
    ]
    cleaned = raw
    for pat in noise_patterns:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_profile_detail_links(html: str, profile_url: str) -> list[str]:
    text = str(html or "")
    if not text:
        return []
    profile_base = str(profile_url or "").strip().rstrip("/")
    out: list[str] = []
    seen: set[str] = set()
    # Absolute links for details and contact overlay.
    for m in re.finditer(
        r'href=["\'](https://www\.linkedin\.com/in/[^"\']+/(?:details/[^"\']+|overlay/contact-info/?)[^"\']*)["\']',
        text,
        flags=re.IGNORECASE,
    ):
        url = _normalize_linkedin_url(m.group(1))
        if not url:
            continue
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(url)
        if len(out) >= 8:
            return out
    # Relative links for details and contact overlay.
    for m in re.finditer(
        r'href=["\'](/in/[^"\']+/(?:details/[^"\']+|overlay/contact-info/?)[^"\']*)["\']',
        text,
        flags=re.IGNORECASE,
    ):
        rel = str(m.group(1)).split("?", 1)[0].split("#", 1)[0]
        url = _normalize_linkedin_url(f"https://www.linkedin.com{rel}")
        if not url:
            continue
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(url)
        if len(out) >= 8:
            return out
    # Always include canonical detail/overlay pages from profile base.
    if profile_base and "/in/" in profile_base:
        for suffix in [
            "/details/experience/",
            "/details/skills/",
            "/overlay/contact-info/",
            "/details/education/",
        ]:
            url = _normalize_linkedin_url(profile_base + suffix)
            if not url:
                continue
            key = url.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(url)
            if len(out) >= 8:
                break
    return out


def _extract_profile_from_text(page_text: str) -> dict[str, str]:
    text = str(page_text or "")
    out: dict[str, str] = {}
    if not text:
        return out
    # Location pattern commonly shown on profile header.
    m_loc = re.search(
        r"\b([A-Z][A-Za-z.'\- ]{1,60}(?:Metropolitan Area| Area|, [A-Z][A-Za-z.'\- ]{1,60}))\b",
        text,
    )
    if m_loc:
        out["location"] = m_loc.group(1).strip()[:120]
    # Education hints.
    m_edu = re.search(r"\b([A-Z][A-Za-z0-9&.'\- ]{2,80}(?:University|College|School|Institute))\b", text)
    if m_edu:
        out["education"] = m_edu.group(1).strip()[:120]
    # Company from "at X" phrase.
    m_company = re.search(r"\bat\s+([A-Z][A-Za-z0-9&.,'() \-]{1,80})", text)
    if m_company:
        out["company"] = m_company.group(1).strip(" .,!?\t\r\n")[:120]
    # Headline pattern: "<role> at <company>"
    m_headline = re.search(
        r"\b([A-Z][A-Za-z0-9&.,'() \-]{2,90}\s+at\s+[A-Z][A-Za-z0-9&.,'() \-]{2,90})\b",
        text,
    )
    if m_headline:
        out["headline"] = m_headline.group(1).strip()[:240]
    # Skill hints from common technical terms if present on the page.
    skills = re.findall(
        r"\b(Python|Java|Golang|TypeScript|JavaScript|React|Node\.js|Kubernetes|AWS|Azure|GCP|Machine Learning|AI|DevOps)\b",
        text,
        flags=re.IGNORECASE,
    )
    if skills:
        uniq: list[str] = []
        seen: set[str] = set()
        for s in skills:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)
            if len(uniq) >= 12:
                break
        out["skills"] = ", ".join(uniq)
    return out


def _extract_contact_from_text(page_text: str) -> dict[str, Any]:
    text = str(page_text or "")
    out: dict[str, Any] = {}
    if not text:
        return out
    emails = sorted(set(re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text)))
    raw_phones = sorted(
        set(
            re.findall(
                r"(?:\+\d[\d\s\-().]{7,}\d|\b\d{2,4}[\s\-]?\d{3,4}[\s\-]?\d{4,6}\b)",
                text,
            )
        )
    )
    phones: list[str] = []
    seen_phone: set[str] = set()
    for p in raw_phones:
        clean = re.sub(r"[^\d+]+", "", str(p or "").strip())
        digits = re.sub(r"\D+", "", clean)
        # Ignore short numeric ids often found in profile links/assets.
        if len(digits) < 10:
            continue
        key = digits
        if key in seen_phone:
            continue
        seen_phone.add(key)
        phones.append(clean)
        if len(phones) >= 5:
            break

    raw_websites = sorted(
        set(
            re.findall(
                r"(https?://[^\s,;\"')]+|(?:www\.)[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?:/[^\s,;\"')]*)?)",
                text,
                flags=re.IGNORECASE,
            )
        )
    )
    websites: list[str] = []
    seen_site: set[str] = set()
    asset_exts = (".js", ".json", ".css", ".map", ".svg", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".woff", ".woff2", ".ttf")
    for raw in raw_websites:
        token = str(raw or "").strip().strip(".,;:")
        if not token:
            continue
        url = token if token.lower().startswith(("http://", "https://")) else f"https://{token}"
        parsed = urlparse(url)
        host = str(parsed.netloc or "").lower().strip()
        path = str(parsed.path or "").lower().strip()
        if not host or "." not in host:
            continue
        if "linkedin.com" in host or "licdn.com" in host:
            continue
        if any(path.endswith(ext) for ext in asset_exts):
            continue
        if any(x in token.lower() for x in ["emoji-picker", "localized-configs", "ui-core", "aero-v1", "assets", "\\"]):
            continue
        key = f"{host}{path}"
        if key in seen_site:
            continue
        seen_site.add(key)
        websites.append(url)
        if len(websites) >= 8:
            break
    if emails:
        out["contact_email"] = emails[:5]
    if phones:
        out["contact_phone"] = [re.sub(r"\s+", " ", p).strip() for p in phones[:5]]
    if websites:
        out["contact_websites"] = websites
    parts: list[str] = []
    if emails:
        parts.append("emails: " + ", ".join(emails[:3]))
    if phones:
        parts.append("phones: " + ", ".join(out.get("contact_phone", [])[:2]))
    if websites:
        parts.append("websites: " + ", ".join(websites[:3]))
    if parts:
        out["contact_info"] = " | ".join(parts)
    return out


def _extract_skill_items(page_text: str) -> list[str]:
    text = str(page_text or "")
    if not text:
        return []
    known = re.findall(
        r"\b(Python|Java|Golang|TypeScript|JavaScript|React|Node\.js|Kubernetes|AWS|Azure|GCP|Machine Learning|AI|DevOps|SQL|C\+\+|C#|Docker|Terraform)\b",
        text,
        flags=re.IGNORECASE,
    )
    out: list[str] = []
    seen: set[str] = set()
    for s in known:
        norm = re.sub(r"\s+", " ", str(s).strip())
        key = norm.lower()
        if not norm or key in seen:
            continue
        seen.add(key)
        out.append(norm)
        if len(out) >= 20:
            break
    return out


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
        if self.cookie_header:
            return True
        p = Path(str(self.storage_state_path or "").strip()).expanduser()
        return bool(str(self.storage_state_path or "").strip() and p.exists())


def _load_cookie_header_from_storage_state(path: str) -> str:
    p = Path(str(path or "").strip()).expanduser()
    if not p.exists():
        return ""
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    rows: list[Any] = []
    # Accept either Playwright storage-state object ({cookies:[...]})
    # or a plain exported cookie-array JSON.
    if isinstance(payload, dict):
        maybe = payload.get("cookies", [])
        if isinstance(maybe, list):
            rows = maybe
    elif isinstance(payload, list):
        rows = payload
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
        if expires in (None, ""):
            expires = row.get("expirationDate")
        if isinstance(expires, (int, float)) and float(expires) > 0 and float(expires) < now:
            continue
        pairs.append(f"{name}={value}")
    return "; ".join(pairs)


def _load_cookie_rows_from_storage_state(path: str) -> list[dict[str, Any]]:
    p = Path(str(path or "").strip()).expanduser()
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows: list[Any] = []
    if isinstance(payload, dict):
        maybe = payload.get("cookies", [])
        if isinstance(maybe, list):
            rows = maybe
    elif isinstance(payload, list):
        rows = payload
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(dict(row))
    return out


def _normalize_cookie_for_playwright(row: dict[str, Any]) -> dict[str, Any] | None:
    name = str(row.get("name", "")).strip()
    value = str(row.get("value", "")).strip()
    domain = str(row.get("domain", "")).strip()
    path = str(row.get("path", "/") or "/").strip() or "/"
    if not name or value is None or not domain:
        return None
    host_only = bool(row.get("hostOnly", False))
    if host_only:
        domain = domain.lstrip(".")
    cookie: dict[str, Any] = {
        "name": name,
        "value": value,
        "domain": domain,
        "path": path,
        "secure": bool(row.get("secure", False)),
        "httpOnly": bool(row.get("httpOnly", False)),
    }
    same_site_raw = str(row.get("sameSite", "")).strip().lower()
    same_site_map = {
        "lax": "Lax",
        "strict": "Strict",
        "none": "None",
        "no_restriction": "None",
        "unspecified": "Lax",
    }
    if same_site_raw in same_site_map:
        cookie["sameSite"] = same_site_map[same_site_raw]
    expires = row.get("expires")
    if expires in (None, ""):
        expires = row.get("expirationDate")
    if isinstance(expires, (int, float)) and float(expires) > 0:
        cookie["expires"] = float(expires)
    return cookie


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


def _safe_fetch_dynamic(
    url: str,
    *,
    storage_state_path: str,
    timeout_s: float = 25.0,
    max_scroll_steps: int = 16,
    idle_rounds: int = 3,
    capture_snapshot: bool = False,
) -> tuple[str, str, dict[str, Any]]:
    """Fetch page with Playwright, expand lazy content and scroll to bottom."""
    if not str(storage_state_path or "").strip():
        return "", url, {}
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        return "", url, {}

    final_url = url
    html = ""
    diagnostics: dict[str, Any] = {"dynamic_fetch_used": True}
    timeout_ms = max(5_000, int(timeout_s * 1000))
    state_path = str(Path(storage_state_path).expanduser())
    raw_cookie_rows = _load_cookie_rows_from_storage_state(state_path)

    use_persistent_profile = str(os.getenv("LINKEDIN_MCP_USE_PERSISTENT_PROFILE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    chrome_user_data_dir = str(os.getenv("LINKEDIN_MCP_CHROME_USER_DATA_DIR", "")).strip()
    chrome_profile_dir = str(os.getenv("LINKEDIN_MCP_CHROME_PROFILE_DIR", "Default")).strip() or "Default"
    chrome_channel = str(os.getenv("LINKEDIN_MCP_CHROME_CHANNEL", "chrome")).strip() or "chrome"
    headless = str(os.getenv("LINKEDIN_MCP_HEADLESS", "1")).strip().lower() not in {"0", "false", "no", "off"}

    with sync_playwright() as pw:
        browser = None
        context = None
        try:
            if use_persistent_profile and chrome_user_data_dir:
                udir = str(Path(chrome_user_data_dir).expanduser())
                context = pw.chromium.launch_persistent_context(
                    user_data_dir=udir,
                    channel=chrome_channel,
                    headless=headless,
                    args=[
                        f"--profile-directory={chrome_profile_dir}",
                        "--disable-blink-features=AutomationControlled",
                    ],
                )
                diagnostics["auth_source"] = "persistent_chrome_profile"
                diagnostics["chrome_user_data_dir"] = udir
                diagnostics["chrome_profile_dir"] = chrome_profile_dir
                diagnostics["chrome_channel"] = chrome_channel
            else:
                browser = pw.chromium.launch(headless=headless, args=["--disable-blink-features=AutomationControlled"])
                # Prefer native Playwright storage-state when available.
                try:
                    context = browser.new_context(storage_state=state_path)
                    diagnostics["auth_source"] = "playwright_storage_state"
                except Exception:
                    context = browser.new_context()
                    diagnostics["auth_source"] = "manual_cookie_injection"
                    normalized_cookies: list[dict[str, Any]] = []
                    for row in raw_cookie_rows:
                        item = _normalize_cookie_for_playwright(row)
                        if item and "linkedin.com" in str(item.get("domain", "")).lower():
                            normalized_cookies.append(item)
                    if normalized_cookies:
                        try:
                            context.add_cookies(normalized_cookies)
                            diagnostics["cookie_rows_loaded"] = len(normalized_cookies)
                        except Exception as exc:
                            diagnostics["cookie_injection_error"] = str(exc)
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(1200)

            # Expand obvious collapsible sections before scrolling.
            for _ in range(4):
                clicked_any = False
                for text in ["See more", "Show more", "More"]:
                    try:
                        loc = page.locator(f"button:has-text('{text}')")
                        count = min(loc.count(), 8)
                        for idx in range(count):
                            btn = loc.nth(idx)
                            if btn.is_visible():
                                btn.click(timeout=500)
                                clicked_any = True
                                page.wait_for_timeout(120)
                    except Exception:
                        continue
                if not clicked_any:
                    break

            # Scroll until no more growth in document height.
            stagnant = 0
            for _ in range(max(1, max_scroll_steps)):
                try:
                    prev_h = int(page.evaluate("() => document.body ? document.body.scrollHeight : 0") or 0)
                except Exception:
                    prev_h = 0
                try:
                    page.evaluate("() => window.scrollBy(0, Math.max(window.innerHeight, 900))")
                except Exception:
                    try:
                        page.mouse.wheel(0, 1200)
                    except Exception:
                        pass
                page.wait_for_timeout(750)
                # Re-expand newly loaded chunks.
                try:
                    loc = page.locator("button:has-text('See more'), button:has-text('Show more')")
                    count = min(loc.count(), 6)
                    for idx in range(count):
                        btn = loc.nth(idx)
                        if btn.is_visible():
                            try:
                                btn.click(timeout=300)
                            except Exception:
                                pass
                except Exception:
                    pass
                try:
                    new_h = int(page.evaluate("() => document.body ? document.body.scrollHeight : 0") or 0)
                except Exception:
                    new_h = prev_h
                if new_h <= prev_h + 8:
                    stagnant += 1
                else:
                    stagnant = 0
                if stagnant >= max(1, idle_rounds):
                    break

            page.wait_for_timeout(900)
            if capture_snapshot:
                try:
                    shot = page.screenshot(type="jpeg", quality=55, full_page=True)
                    if isinstance(shot, (bytes, bytearray)):
                        import base64

                        diagnostics["page_snapshot_jpeg_base64"] = base64.b64encode(bytes(shot)).decode("ascii")
                except Exception as exc:
                    diagnostics["snapshot_error"] = str(exc)
                    # Fallback snapshot when full-page capture fails.
                    try:
                        shot2 = page.screenshot(type="jpeg", quality=55)
                        if isinstance(shot2, (bytes, bytearray)):
                            import base64

                            diagnostics["page_snapshot_jpeg_base64"] = base64.b64encode(bytes(shot2)).decode("ascii")
                    except Exception as exc2:
                        diagnostics["snapshot_error_fallback"] = str(exc2)
            try:
                elements = page.evaluate(
                    """() => {
                        const kws = {
                          experience: ["experience", "worked", "employment", "position"],
                          skills: ["skills", "endorsement", "technology", "tech stack"],
                          contact: ["contact", "email", "phone", "website", "portfolio"]
                        };
                        function kindFor(t) {
                          const low = String(t || "").toLowerCase();
                          for (const [k, arr] of Object.entries(kws)) {
                            if (arr.some((w) => low.includes(w))) return k;
                          }
                          return "other";
                        }
                        function sel(el) {
                          if (!el || !el.tagName) return "";
                          let s = el.tagName.toLowerCase();
                          if (el.id) s += "#" + el.id;
                          const cls = String(el.className || "").trim().split(/\\s+/).filter(Boolean).slice(0, 2);
                          if (cls.length) s += "." + cls.join(".");
                          return s;
                        }
                        const rows = [];
                        const nodes = Array.from(document.querySelectorAll("section, article, div, li"));
                        for (const el of nodes) {
                          const txt = String((el.innerText || "")).replace(/\\s+/g, " ").trim();
                          if (txt.length < 24 || txt.length > 900) continue;
                          const k = kindFor(txt);
                          const r = el.getBoundingClientRect();
                          if (!r || r.width < 100 || r.height < 20) continue;
                          rows.push({
                            kind: k,
                            selector: sel(el),
                            text: txt.slice(0, 320),
                            x: Math.round(r.x), y: Math.round(r.y),
                            w: Math.round(r.width), h: Math.round(r.height)
                          });
                        }
                        // If keyword-based tagging is sparse, still return top visible blocks.
                        if (rows.length < 12) {
                          const fallbackRows = [];
                          for (const el of nodes) {
                            const txt = String((el.innerText || "")).replace(/\\s+/g, " ").trim();
                            if (txt.length < 40 || txt.length > 420) continue;
                            const r = el.getBoundingClientRect();
                            if (!r || r.width < 140 || r.height < 24) continue;
                            fallbackRows.push({
                              kind: "other",
                              selector: sel(el),
                              text: txt.slice(0, 320),
                              x: Math.round(r.x), y: Math.round(r.y),
                              w: Math.round(r.width), h: Math.round(r.height)
                            });
                          }
                          fallbackRows.sort((a,b) => (a.y - b.y) || (b.w - a.w));
                          for (const row of fallbackRows.slice(0, 40)) rows.push(row);
                        }
                        rows.sort((a,b) => a.y - b.y);
                        return rows.slice(0, 160);
                    }"""
                )
                if isinstance(elements, list):
                    diagnostics["extraction_elements"] = elements
            except Exception as exc:
                diagnostics["elements_error"] = str(exc)
            final_url = str(page.url or url)
            html = page.content() or ""
            diagnostics["html_chars"] = len(html or "")
        finally:
            if context is not None:
                try:
                    context.close()
                except Exception:
                    pass
            if browser is not None:
                try:
                    browser.close()
                except Exception:
                    pass
    return html[:900_000], final_url, diagnostics


def _safe_fetch_linkedin_page(
    url: str,
    auth: _AuthConfig,
    *,
    timeout_s: float = 12.0,
    capture_snapshot: bool = False,
) -> tuple[str, str, dict[str, Any]]:
    """Prefer dynamic Playwright fetch for authenticated LinkedIn pages."""
    allow_dynamic = str(os.getenv("LINKEDIN_MCP_USE_DYNAMIC_FETCH", "1")).strip().lower() not in {"0", "false", "no", "off"}
    if allow_dynamic and auth.has_auth() and str(auth.storage_state_path or "").strip():
        html, final_url, diagnostics = _safe_fetch_dynamic(
            url,
            storage_state_path=auth.storage_state_path,
            timeout_s=max(timeout_s, 20.0),
            max_scroll_steps=int(os.getenv("LINKEDIN_MCP_SCROLL_STEPS", "16") or "16"),
            idle_rounds=int(os.getenv("LINKEDIN_MCP_SCROLL_IDLE_ROUNDS", "3") or "3"),
            capture_snapshot=capture_snapshot,
        )
        if html:
            if isinstance(diagnostics, dict):
                diagnostics["dynamic_attempted"] = True
            return html, final_url, diagnostics if isinstance(diagnostics, dict) else {}
    html, final_url = _safe_fetch(url, timeout_s=timeout_s, cookie_header=auth.cookie_header)
    return html, final_url, {"dynamic_attempted": bool(allow_dynamic), "dynamic_used": False, "html_chars": len(html or "")}


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
        html, final_url, page_diag = _safe_fetch_linkedin_page(target_url, auth, capture_snapshot=True)
    except Exception as exc:  # noqa: BLE001
        return {
            **out,
            "ok": False,
            "error": f"fetch_failed: {exc}",
            "diag_dynamic_attempted": True,
            "diag_dynamic_used": False,
        }

    out["final_url"] = final_url
    if isinstance(page_diag, dict):
        out["diag_dynamic_attempted"] = bool(page_diag.get("dynamic_attempted", False))
        out["diag_dynamic_used"] = bool(page_diag.get("dynamic_fetch_used", False))
        out["diag_html_chars"] = int(page_diag.get("html_chars", 0) or 0)
        if str(page_diag.get("auth_source", "")).strip():
            out["auth_source"] = str(page_diag.get("auth_source", "")).strip()
        if str(page_diag.get("chrome_profile_dir", "")).strip():
            out["chrome_profile_dir"] = str(page_diag.get("chrome_profile_dir", "")).strip()
        if str(page_diag.get("chrome_channel", "")).strip():
            out["chrome_channel"] = str(page_diag.get("chrome_channel", "")).strip()
        if "cookie_rows_loaded" in page_diag:
            try:
                out["cookie_rows_loaded"] = int(page_diag.get("cookie_rows_loaded", 0) or 0)
            except Exception:
                out["cookie_rows_loaded"] = 0
        if str(page_diag.get("cookie_injection_error", "")).strip():
            out["cookie_injection_error"] = str(page_diag.get("cookie_injection_error", "")).strip()[:200]
        if str(page_diag.get("snapshot_error", "")).strip():
            out["diag_snapshot_error"] = str(page_diag.get("snapshot_error", "")).strip()[:200]
        if str(page_diag.get("elements_error", "")).strip():
            out["diag_elements_error"] = str(page_diag.get("elements_error", "")).strip()[:200]

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
            # Fetch resolved profile page for richer extraction instead of keeping search-page metadata.
            try:
                html2, final_url2, page_diag2 = _safe_fetch_linkedin_page(resolved_profile, auth, capture_snapshot=True)
                if html2:
                    html = html2
                    out["final_url"] = final_url2
                    if isinstance(page_diag2, dict) and page_diag2:
                        page_diag = page_diag2
            except Exception:
                pass

    slug = _extract_profile_slug(out.get("profile_url", ""))
    if slug:
        out["profile_slug"] = slug

    fields = _extract_html_profile_fields(html)
    page_text = _clean_profile_text(_html_to_text(html))
    detail_links = _extract_profile_detail_links(html, out.get("profile_url", ""))
    detail_chunks: list[str] = []
    detail_elements: list[dict[str, Any]] = []
    detail_text_by_kind: dict[str, list[str]] = {"experience": [], "skills": [], "contact": [], "education": [], "other": []}
    for link in detail_links[:6]:
        try:
            dhtml, _, ddiag = _safe_fetch_linkedin_page(link, auth)
        except Exception:
            continue
        dtext = _clean_profile_text(_html_to_text(dhtml))
        if dtext:
            detail_chunks.append(dtext[:5000])
            low = str(link).lower()
            if "/details/experience" in low:
                detail_text_by_kind["experience"].append(dtext[:5000])
            elif "/details/skills" in low:
                detail_text_by_kind["skills"].append(dtext[:5000])
            elif "/overlay/contact-info" in low:
                detail_text_by_kind["contact"].append(dtext[:5000])
            elif "/details/education" in low:
                detail_text_by_kind["education"].append(dtext[:5000])
            else:
                detail_text_by_kind["other"].append(dtext[:5000])
        if isinstance(ddiag, dict):
            rows = ddiag.get("extraction_elements", [])
            if isinstance(rows, list):
                for row in rows[:40]:
                    if isinstance(row, dict):
                        detail_elements.append(dict(row))
    combined_text = page_text
    if detail_chunks:
        combined_text = " ".join([page_text] + detail_chunks).strip()
    text_fields = _extract_profile_from_text(combined_text or page_text)
    contact_source = " ".join(detail_text_by_kind.get("contact", [])).strip()
    if not contact_source:
        # Fallback only when contact overlay was not fetched.
        contact_source = combined_text
    contact_fields = _extract_contact_from_text(contact_source)
    skill_items = _extract_skill_items(" ".join(detail_text_by_kind.get("skills", []) + [combined_text]))
    for k, v in fields.items():
        if v:
            out[k] = v
    for k, v in text_fields.items():
        if v and not str(out.get(k, "")).strip():
            out[k] = v
    for k, v in contact_fields.items():
        if v:
            out[k] = v
    if skill_items:
        out["skill_items"] = skill_items
        if not str(out.get("skills", "")).strip():
            out["skills"] = ", ".join(skill_items[:20])
    if detail_text_by_kind.get("experience"):
        out["experience_text"] = " ".join(detail_text_by_kind["experience"])[:15000]
    if detail_text_by_kind.get("skills"):
        out["skills_text"] = " ".join(detail_text_by_kind["skills"])[:15000]
    if detail_text_by_kind.get("contact"):
        out["contact_text"] = " ".join(detail_text_by_kind["contact"])[:12000]
    if isinstance(page_diag, dict):
        shot = str(page_diag.get("page_snapshot_jpeg_base64", "")).strip()
        if shot:
            out["page_snapshot_jpeg_base64"] = shot
        elems = page_diag.get("extraction_elements", [])
        if isinstance(elems, list):
            out["extraction_elements"] = elems[:160]
        # Expose minimal diagnostics for debugging extraction path.
        out["diag_dynamic_attempted"] = bool(page_diag.get("dynamic_attempted", False))
        out["diag_dynamic_used"] = bool(page_diag.get("dynamic_fetch_used", False))
        out["diag_html_chars"] = int(page_diag.get("html_chars", 0) or 0)
        if str(page_diag.get("auth_source", "")).strip():
            out["auth_source"] = str(page_diag.get("auth_source", "")).strip()
        if str(page_diag.get("chrome_profile_dir", "")).strip():
            out["chrome_profile_dir"] = str(page_diag.get("chrome_profile_dir", "")).strip()
        if str(page_diag.get("chrome_channel", "")).strip():
            out["chrome_channel"] = str(page_diag.get("chrome_channel", "")).strip()
        if str(page_diag.get("snapshot_error", "")).strip():
            out["diag_snapshot_error"] = str(page_diag.get("snapshot_error", "")).strip()[:200]
        if str(page_diag.get("elements_error", "")).strip():
            out["diag_elements_error"] = str(page_diag.get("elements_error", "")).strip()[:200]
    if detail_elements:
        out["detail_extraction_elements"] = detail_elements[:200]
    # Fallback when LinkedIn returns generic search-ish title.
    if str(out.get("name", "")).strip().lower() in {"search | linkedin", "linkedin", "sign in"} and slug:
        out["name"] = _humanize_slug(slug)
    if combined_text:
        out["profile_text"] = combined_text[:15000]
        out["profile_text_len"] = len(combined_text)
    if detail_links:
        out["detail_links"] = detail_links[:6]
        out["detail_links_count"] = len(detail_links[:6])

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


def _runtime_diagnostics(auth: _AuthConfig) -> dict[str, Any]:
    playwright_available = False
    try:
        import playwright  # type: ignore  # noqa: F401

        playwright_available = True
    except Exception:
        playwright_available = False
    storage_path = str(auth.storage_state_path or "").strip()
    storage_exists = bool(storage_path and Path(storage_path).expanduser().exists())
    dynamic_enabled = str(os.getenv("LINKEDIN_MCP_USE_DYNAMIC_FETCH", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    persistent_profile_enabled = str(os.getenv("LINKEDIN_MCP_USE_PERSISTENT_PROFILE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return {
        "playwright_available": playwright_available,
        "dynamic_fetch_enabled": dynamic_enabled,
        "persistent_profile_enabled": persistent_profile_enabled,
        "chrome_user_data_dir": str(os.getenv("LINKEDIN_MCP_CHROME_USER_DATA_DIR", "")).strip(),
        "chrome_profile_dir": str(os.getenv("LINKEDIN_MCP_CHROME_PROFILE_DIR", "Default")).strip() or "Default",
        "chrome_channel": str(os.getenv("LINKEDIN_MCP_CHROME_CHANNEL", "chrome")).strip() or "chrome",
        "storage_state_path": storage_path,
        "storage_state_exists": storage_exists,
    }


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
            diag = _runtime_diagnostics(self.auth)
            self._write_json(
                {
                    "ok": True,
                    "service": "linkedin-mcp",
                    "authenticated": self.auth.has_auth(),
                    "storage_state_path": self.auth.storage_state_path,
                    "runtime": diag,
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
