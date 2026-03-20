#!/usr/bin/env python3
"""
Import non-Hugging-Face URL manifests into JARVIS research ingestion API.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import ssl
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
DEFAULT_DOWNLOAD_ROOT = "/Volumes/Jiten-2026/AI_SSD/ai-research/datasets/non_hf_sources"
DEFAULT_USER_AGENT = "JARVIS-AI-OS/1.0 (+research-ingest; contact: local-user)"
DEFAULT_FAILED_OUT = "data/non_hf_failed_urls_manifest.txt"


def _log(message: str) -> None:
    print(f"[import_non_hf_sources] {message}", flush=True)


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


def _resolve_auth_token(cli_token: str) -> str:
    token = str(cli_token or "").strip()
    if token:
        return token
    return str(os.environ.get("JARVIS_API_TOKEN", "")).strip()


def _verify_query(
    *,
    api_base: str,
    auth_token: str,
    topic: str,
    query_text: str,
    timeout: int = 20,
) -> Dict[str, Any]:
    endpoint = f"{api_base.rstrip('/')}/api/v1/research/query"
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    payload = {"query": query_text, "topic": topic, "max_results": 1}
    if requests is not None:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=max(2, int(timeout)))
        resp.raise_for_status()
        data = resp.json()
    else:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=max(2, int(timeout))) as res:
            data = json.loads(res.read().decode("utf-8"))
    if not isinstance(data, dict):
        return {"ok": False, "error": "invalid_query_response"}
    if not bool(data.get("success", False)):
        return {"ok": False, "error": str(data.get("error", "query_failed"))}
    inner = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
    return {
        "ok": True,
        "result_count": int(inner.get("result_count", 0)),
        "rag_context_count": int(inner.get("rag_context_count", 0)),
    }


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


def _fallback_urls(url: str) -> List[str]:
    out = [url]
    u = url.strip()
    if u == "https://www.cisa.gov/known-exploited-vulnerabilities-catalog":
        out.append("https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json")
    if u == "https://www.sec.gov/edgar/sec-api-documentation":
        out.append("https://data.sec.gov/submissions/CIK0000320193.json")
    if u == "https://www.imf.org/en/Data":
        out.append("https://data.imf.org/en/Resource-Pages/IMF-API")
    if u == "https://www.gbif.org/":
        out.append("https://api.gbif.org/v1/occurrence/search?limit=1")
    if u == "https://mlhub.earth/":
        out.append("https://www.radiant.earth/mlhub/")
    if u == "https://www.cabdirect.org/":
        out.append("https://www.cabi.org/cabdirect")
    if u == "https://www.ayush.gov.in/":
        out.append("https://main.ayush.gov.in/")
    return out


def _fetch_url(
    url: str,
    timeout: int,
    max_chars: int,
    *,
    user_agent: str,
    insecure_ssl: bool,
) -> Dict[str, Any]:
    headers = {"User-Agent": user_agent, "Accept": "text/html,application/json,*/*;q=0.5"}
    verify_ssl = not bool(insecure_ssl)
    errors: List[str] = []
    for candidate in _fallback_urls(url):
        try:
            if requests is not None:
                resp = requests.get(candidate, timeout=max(2, int(timeout)), headers=headers, verify=verify_ssl)
                resp.raise_for_status()
                ctype = str(resp.headers.get("Content-Type", "")).lower()
                body = resp.text
            else:
                req = urllib.request.Request(candidate, headers=headers, method="GET")
                ctx = None
                if insecure_ssl:
                    ctx = ssl._create_unverified_context()
                try:
                    with urllib.request.urlopen(req, timeout=max(2, int(timeout)), context=ctx) as res:
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
            return {"title": title, "content": content, "content_type": ctype, "resolved_url": candidate}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate} :: {exc}")
            continue
    raise RuntimeError(" | ".join(errors))


def _url_fingerprint(url: str, content: str) -> str:
    h = hashlib.sha256()
    h.update(url.encode("utf-8"))
    h.update(b"|")
    h.update(str(len(content)).encode("utf-8"))
    h.update(b"|")
    h.update(content[:1024].encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _slug(value: str, max_len: int = 80) -> str:
    raw = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip().lower()).strip("-")
    if not raw:
        raw = "item"
    if len(raw) > max_len:
        raw = raw[:max_len].rstrip("-")
    return raw or "item"


def _infer_source_type(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if any(k in host for k in ("arxiv.org", "pubmed.", "crossref.org", "openalex.org", "fao.org", "worldbank.org", "imf.org")):
        return "official"
    if any(k in host for k in ("github.com", "kaggle.com")):
        return "blog"
    return "news"


def _check_api_reachable(api_base: str, auth_token: str, timeout: int) -> tuple[bool, str]:
    health_url = f"{api_base.rstrip('/')}/api/v1/health"
    headers: Dict[str, str] = {}
    if auth_token.strip():
        headers["Authorization"] = f"Bearer {auth_token.strip()}"
    try:
        if requests is not None:
            resp = requests.get(health_url, headers=headers, timeout=max(2, int(timeout)))
            if resp.status_code >= 400:
                return False, f"Health endpoint returned HTTP {resp.status_code} at {health_url}"
            return True, health_url

        req = urllib.request.Request(health_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=max(2, int(timeout))):
            pass
        return True, health_url
    except Exception as exc:  # noqa: BLE001
        return False, f"Cannot reach API at {health_url}: {exc}"


def _save_download_bundle(
    *,
    download_root: Path,
    item: Dict[str, Any],
) -> str:
    meta = item.get("metadata", {}) if isinstance(item, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    url = str(item.get("url", ""))
    host = str(meta.get("source_host", "")).strip() or (urlparse(url).netloc or "unknown-host")
    domain_primary = str(meta.get("domain_primary", "")).strip() or "unknown"
    title = str(item.get("title", "")).strip() or "untitled"
    fp = str(meta.get("fingerprint", "")).strip() or _url_fingerprint(url, str(item.get("content", "")))
    folder = download_root / domain_primary / _slug(host, max_len=60)
    folder.mkdir(parents=True, exist_ok=True)
    base = f"{_slug(title, max_len=64)}-{fp[:12]}"
    txt_path = folder / f"{base}.txt"
    json_path = folder / f"{base}.json"
    txt_path.write_text(str(item.get("content", "")), encoding="utf-8")
    json_path.write_text(json.dumps(item, indent=2, sort_keys=True), encoding="utf-8")
    return str(txt_path)


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
    parser.add_argument(
        "--verify-after-ingest",
        action="store_true",
        help="Run a small /api/v1/research/query check after ingest",
    )
    parser.add_argument(
        "--verify-query",
        default="retrieval test",
        help="Query text used with --verify-after-ingest (default: 'retrieval test')",
    )
    parser.add_argument("--api-timeout", type=int, default=20, help="API timeout seconds for verify check")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE, help="Path for dedupe state JSON")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent for source fetching")
    parser.add_argument("--insecure-ssl", action="store_true", help="Disable SSL certificate verification for fetches")
    parser.add_argument(
        "--failed-out",
        default=DEFAULT_FAILED_OUT,
        help=f"Output file path containing failed URLs (default: {DEFAULT_FAILED_OUT})",
    )
    parser.add_argument("--download-only", action="store_true", help="Download and save extracted content locally, skip API ingest")
    parser.add_argument(
        "--download-root",
        default=DEFAULT_DOWNLOAD_ROOT,
        help=f"Download output root for --download-only (default: {DEFAULT_DOWNLOAD_ROOT})",
    )
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
    _log(f"fetch_started urls={len(urls)}")

    for idx, url in enumerate(urls, start=1):
        try:
            fetched = _fetch_url(
                url,
                timeout=int(args.timeout),
                max_chars=max(1000, int(args.max_chars)),
                user_agent=str(args.user_agent),
                insecure_ssl=bool(args.insecure_ssl),
            )
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
                        "resolved_url": str(fetched.get("resolved_url", url)),
                        "fingerprint": fp,
                        "dataset_origin": "non_hf",
                        "dataset_verified": False,
                        "dataset_confidence": 0.72,
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            fetch_failures.append({"url": url, "error": str(exc)})
        if idx % 100 == 0:
            _log(
                "fetch_in_progress "
                f"processed={idx}/{len(urls)} "
                f"ready={len(items)} "
                f"skipped_already_imported={skipped_unchanged} "
                f"failures={len(fetch_failures)}"
            )

    _log(
        "fetch_completed "
        f"processed={len(urls)} ready={len(items)} "
        f"skipped_already_imported={skipped_unchanged} "
        f"failures={len(fetch_failures)}"
    )

    preview = {
        "manifests": [str(m) for m in manifests],
        "urls_total": len(urls),
        "items_ready": len(items),
        "skipped_unchanged": skipped_unchanged,
        "fetch_failures": len(fetch_failures),
        "state_file": str(state_path),
    }
    failed_out = Path(args.failed_out).expanduser().resolve()
    failed_out.parent.mkdir(parents=True, exist_ok=True)
    failed_out.write_text(
        "\n".join(sorted({str(row.get("url", "")).strip() for row in fetch_failures if str(row.get("url", "")).strip()}))
        + ("\n" if fetch_failures else ""),
        encoding="utf-8",
    )
    preview["failed_out"] = str(failed_out)
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        for row in fetch_failures[:20]:
            print(f"- fail {row['url']} :: {row['error']}")
        return

    if bool(args.download_only):
        download_root = Path(args.download_root).expanduser().resolve()
        saved_files: List[str] = []
        for it in items:
            saved_files.append(_save_download_bundle(download_root=download_root, item=it))
        for it in items:
            u = str(it.get("url", ""))
            fp = str(it.get("metadata", {}).get("fingerprint", ""))
            if u and fp:
                imported[u] = fp
        state["imported"] = imported
        _save_state(state_path, state)
        out = {
            **preview,
            "download_only": True,
            "download_root": str(download_root),
            "saved_files": len(saved_files),
        }
        print(json.dumps(out, indent=2))
        for row in fetch_failures[:20]:
            print(f"- fail {row['url']} :: {row['error']}")
        return

    resolved_token = _resolve_auth_token(args.auth_token)
    ok, msg = _check_api_reachable(str(args.api_base), resolved_token, int(args.timeout))
    if not ok:
        raise SystemExit(
            f"{msg}\n"
            "Start the API first (e.g., `python jarvis_main.py --mode api` or Podman compose), "
            "or pass --api-base with the correct host:port."
        )

    endpoint = f"{str(args.api_base).rstrip('/')}/api/v1/research/ingest"
    headers = {"Content-Type": "application/json"}
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token}"

    inserted_total = 0
    skipped_total = 0
    batch_size = max(1, int(args.batch_size))
    batch_count = max(1, (len(items) + batch_size - 1) // batch_size)
    for batch_idx, batch in enumerate(_chunks(items, batch_size), start=1):
        _log(f"ingest_batch_in_progress batch={batch_idx}/{batch_count} items={len(batch)}")
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
        inserted = int(data.get("inserted", 0))
        skipped = int(data.get("skipped_duplicates", 0))
        inserted_total += inserted
        skipped_total += skipped
        _log(
            "ingest_batch_completed "
            f"batch={batch_idx}/{batch_count} "
            f"inserted={inserted} skipped_duplicates={skipped} "
            f"inserted_total={inserted_total} skipped_total={skipped_total}"
        )

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
    if bool(args.verify_after_ingest):
        try:
            out["verify"] = _verify_query(
                api_base=str(args.api_base),
                auth_token=resolved_token,
                topic=str(args.topic),
                query_text=str(args.verify_query),
                timeout=int(args.api_timeout),
            )
        except Exception as exc:  # noqa: BLE001
            out["verify"] = {"ok": False, "error": str(exc)}
    print(json.dumps(out, indent=2))
    for row in fetch_failures[:20]:
        print(f"- fail {row['url']} :: {row['error']}")


if __name__ == "__main__":
    main()
