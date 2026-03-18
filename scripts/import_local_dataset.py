#!/usr/bin/env python3
"""
Import local dataset folders into JARVIS research ingestion endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None  # type: ignore[assignment]
    import urllib.error
    import urllib.request


DEFAULT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".json",
    ".jsonl",
    ".csv",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".py",
    ".yaml",
    ".yml",
}
DEFAULT_DATASET_ROOT = "/Volumes/Jiten-2026/AI_SSD/ai-research/datasets"
DEFAULT_DOMAIN_INDEX_FILE = ".jarvis_dataset_domains.json"


def _iter_files(root: Path, recursive: bool = True) -> Iterable[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    for p in iterator:
        if p.is_file():
            yield p


def _read_text(path: Path, max_chars: int) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    data = data.strip()
    if len(data) > max_chars:
        return data[:max_chars]
    return data


def _build_item(path: Path, root: Path, topic: str, source_type: str, max_chars: int) -> Dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    content = _read_text(path, max_chars=max_chars)
    return {
        "title": rel,
        "url": f"file://{path}",
        "content": content,
        "topic": topic,
        "source_type": source_type,
        "metadata": {
            "dataset_root": str(root),
            "relative_path": rel,
            "extension": path.suffix.lower(),
            "loader": "import_local_dataset.py",
        },
    }


def _fingerprint(path: Path, rel: str, topic: str, source_type: str, content: str) -> str:
    st = path.stat()
    h = hashlib.sha256()
    h.update(rel.encode("utf-8"))
    h.update(b"|")
    h.update(str(st.st_size).encode("utf-8"))
    h.update(b"|")
    h.update(str(int(st.st_mtime)).encode("utf-8"))
    h.update(b"|")
    h.update(topic.encode("utf-8"))
    h.update(b"|")
    h.update(source_type.encode("utf-8"))
    h.update(b"|")
    h.update(content[:512].encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "imported": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "imported": {}}
    if not isinstance(data, dict):
        return {"version": 1, "imported": {}}
    imported = data.get("imported", {})
    if not isinstance(imported, dict):
        imported = {}
    return {"version": 1, "imported": imported}


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _chunks(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _load_domain_index(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    mapping = payload.get("mapping", {})
    if not isinstance(mapping, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not isinstance(value, list):
            continue
        tags = [str(v).strip().lower() for v in value if str(v).strip()]
        if tags:
            out[key] = sorted(set(tags))
    return out


def _dataset_id_for_file(path: Path, root: Path) -> str:
    rel_parts = path.relative_to(root).parts
    if len(rel_parts) < 4:
        return ""
    if rel_parts[0] != "huggingface":
        return ""
    return f"{rel_parts[1]}/{rel_parts[2]}"


def import_folder(
    *,
    root: Path,
    api_base: str,
    topic: str,
    source_type: str,
    recursive: bool,
    max_chars: int,
    extensions: set[str],
    batch_size: int,
    auth_token: str,
    state_file: Path,
    domain_index_file: Path,
) -> Dict[str, Any]:
    files = [p for p in _iter_files(root, recursive=recursive) if p.suffix.lower() in extensions]
    state = _load_state(state_file)
    imported_index = dict(state.get("imported", {}))
    domain_index = _load_domain_index(domain_index_file)
    items: List[Dict[str, Any]] = []
    skipped_local_duplicates = 0
    seen_fingerprints: set[str] = set()

    for p in files:
        rel = p.relative_to(root).as_posix()
        content = _read_text(p, max_chars=max_chars)
        if not content:
            continue
        fp = _fingerprint(p, rel=rel, topic=topic, source_type=source_type, content=content)
        if fp in seen_fingerprints:
            skipped_local_duplicates += 1
            continue
        seen_fingerprints.add(fp)
        if imported_index.get(rel) == fp:
            skipped_local_duplicates += 1
            continue
        dataset_id = _dataset_id_for_file(p, root=root)
        domain_tags = domain_index.get(dataset_id, [])
        item = {
            "title": rel,
            "url": f"file://{p}",
            "content": content,
            "topic": topic,
            "source_type": source_type,
            "metadata": {
                "dataset_root": str(root),
                "relative_path": rel,
                "extension": p.suffix.lower(),
                "loader": "import_local_dataset.py",
                "fingerprint": fp,
                "dataset_id": dataset_id,
                "domain_tags": domain_tags,
                "domain_primary": domain_tags[0] if domain_tags else "",
            },
        }
        items.append(item)

    inserted_total = 0
    skipped_total = 0
    endpoint = f"{api_base.rstrip('/')}/api/v1/research/ingest"
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    for batch in _chunks(items, max(1, int(batch_size))):
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

    # Mark all sent items as imported snapshots for dedupe across runs.
    for item in items:
        rel = str(item.get("metadata", {}).get("relative_path", ""))
        fp = str(item.get("metadata", {}).get("fingerprint", ""))
        if rel and fp:
            imported_index[rel] = fp
    state["imported"] = imported_index
    _save_state(state_file, state)

    return {
        "files_scanned": len(files),
        "items_sent": len(items),
        "inserted_total": inserted_total,
        "skipped_duplicates_total": skipped_total,
        "skipped_local_duplicates": skipped_local_duplicates,
        "state_file": str(state_file),
        "domain_index_file": str(domain_index_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import local files into JARVIS research ingestion API.")
    parser.add_argument(
        "folder",
        nargs="?",
        default=DEFAULT_DATASET_ROOT,
        help=f"Folder to import (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8080", help="API base URL")
    parser.add_argument("--topic", default="local-dataset", help="Topic label for imported items")
    parser.add_argument("--source-type", default="blog", help="Source type for imported items")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max chars per file content")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size per ingest request")
    parser.add_argument("--extensions", default="", help="Comma-separated extensions (e.g. .js,.tsx,.md)")
    parser.add_argument("--auth-token", default="", help="Bearer token if API auth is enabled")
    parser.add_argument(
        "--state-file",
        default="",
        help="Importer dedupe state file path (default: <folder>/.jarvis_ingest_state.json)",
    )
    parser.add_argument(
        "--domain-index-file",
        default="",
        help=(
            "Dataset domain index file path "
            f"(default: <folder>/{DEFAULT_DOMAIN_INDEX_FILE})"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print import summary, do not call API")
    args = parser.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")

    if args.extensions.strip():
        exts = {e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}" for e in args.extensions.split(",") if e.strip()}
    else:
        exts = set(DEFAULT_EXTENSIONS)

    domain_index_file = (
        Path(args.domain_index_file).expanduser().resolve()
        if str(args.domain_index_file).strip()
        else (root / DEFAULT_DOMAIN_INDEX_FILE)
    )

    files = [p for p in _iter_files(root, recursive=bool(args.recursive)) if p.suffix.lower() in exts]
    preview = {
        "folder": str(root),
        "files_matching": len(files),
        "extensions": sorted(exts),
        "topic": args.topic,
        "source_type": args.source_type,
        "domain_index_file": str(domain_index_file),
        "state_file": str(
            Path(args.state_file).expanduser().resolve()
            if str(args.state_file).strip()
            else (root / ".jarvis_ingest_state.json")
        ),
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return

    result = import_folder(
        root=root,
        api_base=args.api_base,
        topic=args.topic,
        source_type=args.source_type,
        recursive=bool(args.recursive),
        max_chars=max(1000, int(args.max_chars)),
        extensions=exts,
        batch_size=max(1, int(args.batch_size)),
        auth_token=args.auth_token.strip(),
        state_file=(
            Path(args.state_file).expanduser().resolve()
            if str(args.state_file).strip()
            else (root / ".jarvis_ingest_state.json")
        ),
        domain_index_file=domain_index_file,
    )
    print(json.dumps({**preview, **result}, indent=2))


if __name__ == "__main__":
    main()
