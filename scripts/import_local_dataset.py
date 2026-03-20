#!/usr/bin/env python3
"""
Import local dataset folders into JARVIS research ingestion endpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
    ".parquet",
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


def _log(message: str) -> None:
    print(f"[import_local_dataset] {message}", flush=True)


def _iter_files(root: Path, recursive: bool = True) -> Iterable[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    for p in iterator:
        if p.is_file():
            yield p


def _is_lfs_pointer_text(text: str) -> bool:
    t = str(text or "")
    return t.startswith("version https://git-lfs.github.com/spec/v1")


def _read_text(path: Path, max_chars: int) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if _is_lfs_pointer_text(data):
        return ""
    data = data.strip()
    if len(data) > max_chars:
        return data[:max_chars]
    return data


def _structured_records_to_text(records: List[Dict[str, Any]], max_chars: int) -> str:
    chunks: List[str] = []
    total = 0
    for rec in records:
        line = json.dumps(rec, ensure_ascii=False)
        if total + len(line) + 1 > max_chars:
            break
        chunks.append(line)
        total += len(line) + 1
    return "\n".join(chunks).strip()


def _read_structured(path: Path, max_chars: int) -> str:
    ext = path.suffix.lower()
    if ext == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return ""
        if isinstance(payload, list):
            rows = [x for x in payload if isinstance(x, dict)][:200]
            if rows:
                return _structured_records_to_text(rows, max_chars=max_chars)
        if isinstance(payload, dict):
            return _structured_records_to_text([payload], max_chars=max_chars)
        return ""

    if ext == ".jsonl":
        rows: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f):
                    if idx >= 500:
                        break
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        rows.append(obj)
        except Exception:
            return ""
        return _structured_records_to_text(rows, max_chars=max_chars)

    if ext == ".csv":
        rows: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    if idx >= 500:
                        break
                    rows.append({str(k): str(v) for k, v in row.items()})
        except Exception:
            return ""
        return _structured_records_to_text(rows, max_chars=max_chars)

    if ext == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception:
            return ""
        try:
            table = pq.read_table(path)
            rows = table.to_pylist()[:200]
            dict_rows = [r for r in rows if isinstance(r, dict)]
            return _structured_records_to_text(dict_rows, max_chars=max_chars)
        except Exception:
            return ""

    return ""


def _read_content(path: Path, max_chars: int) -> str:
    ext = path.suffix.lower()
    if ext in {".json", ".jsonl", ".csv", ".parquet"}:
        structured = _read_structured(path, max_chars=max_chars)
        if structured:
            return structured
    return _read_text(path, max_chars=max_chars)


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
    _log(f"scan_started files={len(files)} root={root}")
    state = _load_state(state_file)
    imported_index = dict(state.get("imported", {}))
    domain_index = _load_domain_index(domain_index_file)
    items: List[Dict[str, Any]] = []
    skipped_local_duplicates = 0
    skipped_already_imported = 0
    skipped_empty_content = 0
    seen_fingerprints: set[str] = set()
    scanned = 0

    for p in files:
        scanned += 1
        rel = p.relative_to(root).as_posix()
        content = _read_content(p, max_chars=max_chars)
        if not content:
            skipped_empty_content += 1
            continue
        fp = _fingerprint(p, rel=rel, topic=topic, source_type=source_type, content=content)
        if fp in seen_fingerprints:
            skipped_local_duplicates += 1
            continue
        seen_fingerprints.add(fp)
        if imported_index.get(rel) == fp:
            skipped_already_imported += 1
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
                "dataset_origin": "hf" if dataset_id else "local",
                "dataset_verified": bool(dataset_id),
                "dataset_confidence": 1.0 if dataset_id else 0.75,
            },
        }
        items.append(item)
        if scanned % 500 == 0:
            _log(
                "scan_in_progress "
                f"scanned={scanned}/{len(files)} "
                f"ready={len(items)} "
                f"skipped_already_imported={skipped_already_imported} "
                f"skipped_local_duplicates={skipped_local_duplicates} "
                f"skipped_empty={skipped_empty_content}"
            )

    _log(
        "scan_completed "
        f"scanned={len(files)} ready={len(items)} "
        f"skipped_already_imported={skipped_already_imported} "
        f"skipped_local_duplicates={skipped_local_duplicates} "
        f"skipped_empty={skipped_empty_content}"
    )

    inserted_total = 0
    skipped_total = 0
    endpoint = f"{api_base.rstrip('/')}/api/v1/research/ingest"
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    batch_count = max(1, (len(items) + max(1, int(batch_size)) - 1) // max(1, int(batch_size)))
    for batch_idx, batch in enumerate(_chunks(items, max(1, int(batch_size))), start=1):
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
        "skipped_already_imported": skipped_already_imported,
        "skipped_empty_content": skipped_empty_content,
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
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size per ingest request")
    parser.add_argument("--extensions", default="", help="Comma-separated extensions (e.g. .js,.tsx,.md)")
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
        auth_token=_resolve_auth_token(args.auth_token),
        state_file=(
            Path(args.state_file).expanduser().resolve()
            if str(args.state_file).strip()
            else (root / ".jarvis_ingest_state.json")
        ),
        domain_index_file=domain_index_file,
    )
    out: Dict[str, Any] = {**preview, **result}
    if bool(args.verify_after_ingest):
        try:
            out["verify"] = _verify_query(
                api_base=str(args.api_base),
                auth_token=_resolve_auth_token(args.auth_token),
                topic=str(args.topic),
                query_text=str(args.verify_query),
                timeout=int(args.api_timeout),
            )
        except Exception as exc:  # noqa: BLE001
            out["verify"] = {"ok": False, "error": str(exc)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
