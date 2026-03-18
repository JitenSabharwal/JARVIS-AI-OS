#!/usr/bin/env python3
"""
Import non-Hugging-Face URL manifests into JARVIS research ingestion API.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None  # type: ignore[assignment]
    import urllib.error
    import urllib.request


DEFAULT_MANIFEST = "config/non_hf_sources_multi_domain_manifest.txt"
DEFAULT_STATE_FILE = "data/non_hf_ingest_state.json"


def _domain_tag_from_manifest(path: Path) -> str:
    stem = path.stem
    if stem == "agri_external_sources_manifest":
        return "agri"
    prefix = "non_hf_sources_"
    suffix = "_manifest"
    if stem.startswith(prefix) and stem.endswith(suffix):
        mid = stem[len(prefix) : -len(suffix)].strip().lower()
        return mid or "unknown"
    if stem == "non_hf_sources_multi_domain_manifest":
        return "multi"
    return "unknown"


def _read_manifest(path: Path) -> List[str]:
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(("http://", "https://")):
            continue
        out.append(line)
    return out


def _chunks(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "imported": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "imported": {}}
    imported = payload.get("imported", {}) if isinstance(payload, dict) else {}
    if not isinstance(imported, dict):
        imported = {}
    return {"version": 1, "imported": imported}


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _extract_title_and_text(raw_html: str, max_chars: int) -> Tuple[str, str]:
    text = str(raw_html or "")
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
    if m:
        title = html.unescape(m.group(1)).strip()
    cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return title, cleaned


def _fetch_url(url: str, timeout: int, max_chars: int) -> Dict[str, Any]:
    user_agent = "JARVIS-AI-OS/1.0 (+non_hf_ingest)"
    if requests is not None:
        resp = requests.get(url, timeout=max(2, int(timeout)), headers={"User-Agent": user_agent})
        resp.raise_for_status()
        ctype = str(resp.headers.get("Content-Type", "")).lower()
        body = resp.text
    else:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=max(2, int(timeout))) as res:
                raw = res.read()
                ctype = str(res.headers.get("Content-Type", "")).lower()
            body = raw.decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code}: {detail[:200]}") from exc

    if ctype and ("text" not in ctype and "json" not in ctype and "xml" not in ctype and "html" not in ctype):
        raise RuntimeError(f"unsupported content-type: {ctype}")

    title, content = _extract_title_and_text(body, max_chars=max_chars)
    if not content:
        raise RuntimeError("empty extracted content")
    return {"title": title, "content": content, "content_type": ctype}


def _url_fingerprint(url: str, content: str) -> str:
    h = hashlib.sha256()
    h.update(url.encode("utf-8"))
    h.update(b"|")
    h.update(str(len(content)).encode("utf-8"))
    h.update(b"|")
    h.update(content[:1024].encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _infer_source_type(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if any(k in host for k in ("arxiv.org", "pubmed.", "crossref.org", "openalex.org", "fao.org", "worldbank.org", "imf.org")):
        return "official"
    if any(k in host for k in ("github.com", "kaggle.com")):
        return "blog"
    return "news"


def main() -> None:
    parser = argparse.ArgumentParser(description="Import non-HF source URLs into JARVIS research ingestion API.")
    parser.add_argument(
        "--manifest",
        nargs="+",
        default=[DEFAULT_MANIFEST],
        help=f"One or more non-HF manifest files (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8080", help="API base URL")
    parser.add_argument("--topic", default="external-sources", help="Topic label for imported items")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size per ingest request")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds for source fetch")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max extracted chars per URL")
    parser.add_argument("--auth-token", default="", help="Bearer token if API auth is enabled")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE, help="Path for dedupe state JSON")
    parser.add_argument("--dry-run", action="store_true", help="Only print summary and planned URLs")
    args = parser.parse_args()

    manifests = [Path(p).expanduser().resolve() for p in args.manifest]
    for mf in manifests:
        if not mf.exists():
            raise SystemExit(f"Manifest not found: {mf}")

    urls: List[str] = []
    url_domain_tags: Dict[str, set[str]] = {}
    seen: set[str] = set()
    for mf in manifests:
        domain_tag = _domain_tag_from_manifest(mf)
        for url in _read_manifest(mf):
            if url not in url_domain_tags:
                url_domain_tags[url] = set()
            url_domain_tags[url].add(domain_tag)
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)

    state_path = Path(args.state_file).expanduser().resolve()
    state = _load_state(state_path)
    imported = dict(state.get("imported", {}))

    items: List[Dict[str, Any]] = []
    skipped_unchanged = 0
    fetch_failures: List[Dict[str, str]] = []

    for url in urls:
        try:
            fetched = _fetch_url(url, timeout=int(args.timeout), max_chars=max(1000, int(args.max_chars)))
            fp = _url_fingerprint(url, fetched["content"])
            if imported.get(url) == fp:
                skipped_unchanged += 1
                continue
            parsed = urlparse(url)
            domain_tags = sorted(url_domain_tags.get(url, set()))
            title = (fetched.get("title") or "").strip() or f"{parsed.netloc}{parsed.path or '/'}"
            items.append(
                {
                    "title": title[:240],
                    "url": url,
                    "content": str(fetched.get("content", "")),
                    "topic": str(args.topic),
                    "source_type": _infer_source_type(url),
                    "metadata": {
                        "loader": "import_non_hf_sources.py",
                        "domain_tags": domain_tags,
                        "domain_primary": domain_tags[0] if domain_tags else "",
                        "source_host": parsed.netloc,
                        "source_path": parsed.path or "/",
                        "fingerprint": fp,
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            fetch_failures.append({"url": url, "error": str(exc)})

    preview = {
        "manifests": [str(m) for m in manifests],
        "urls_total": len(urls),
        "items_ready": len(items),
        "skipped_unchanged": skipped_unchanged,
        "fetch_failures": len(fetch_failures),
        "state_file": str(state_path),
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        for row in fetch_failures[:20]:
            print(f"- fail {row['url']} :: {row['error']}")
        return

    endpoint = f"{str(args.api_base).rstrip('/')}/api/v1/research/ingest"
    headers = {"Content-Type": "application/json"}
    if str(args.auth_token).strip():
        headers["Authorization"] = f"Bearer {str(args.auth_token).strip()}"

    inserted_total = 0
    skipped_total = 0
    for batch in _chunks(items, max(1, int(args.batch_size))):
        if requests is not None:
            resp = requests.post(endpoint, headers=headers, json={"items": batch}, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
        else:
            body = json.dumps({"items": batch}).encode("utf-8")
            req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as res:
                    payload = json.loads(res.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"HTTP {exc.code} ingest failure: {detail}") from exc

        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        inserted_total += int(data.get("inserted", 0))
        skipped_total += int(data.get("skipped_duplicates", 0))

    for it in items:
        u = str(it.get("url", ""))
        fp = str(it.get("metadata", {}).get("fingerprint", ""))
        if u and fp:
            imported[u] = fp
    state["imported"] = imported
    _save_state(state_path, state)

    out = {
        **preview,
        "inserted_total": inserted_total,
        "skipped_duplicates_total": skipped_total,
    }
    print(json.dumps(out, indent=2))
    for row in fetch_failures[:20]:
        print(f"- fail {row['url']} :: {row['error']}")


if __name__ == "__main__":
    main()
