#!/usr/bin/env python3
"""
Sync Hugging Face model repos listed in manifest(s) into local storage.

Uses huggingface_hub.snapshot_download for reliable, resumable downloads.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

DEFAULT_MANIFEST = "config/hf_mlx_models_manifest.txt"
DEFAULT_MODELS_ROOT = "/Volumes/Jiten-2026/AI_SSD/huggingface/hub"


def _default_models_root() -> str:
    hf_home = str(os.environ.get("HF_HOME", "")).strip()
    if hf_home:
        return str((Path(hf_home).expanduser().resolve() / "hub"))
    return DEFAULT_MODELS_ROOT


def _normalize_repo_id(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if text.startswith("https://huggingface.co/"):
        parts = [p for p in text.split("/") if p]
        if "huggingface.co" in parts:
            idx = parts.index("huggingface.co")
            if idx + 2 < len(parts):
                maybe_kind = parts[idx + 1]
                if maybe_kind == "models":
                    return "/".join(parts[idx + 2 : idx + 4])
                return "/".join(parts[idx + 1 : idx + 3])
    if text.startswith("models--"):
        # Cache folder name -> repo id
        return text.replace("models--", "", 1).replace("--", "/", 1)
    return text


def _read_manifest(path: Path) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            repo_raw, category = line.split("|", 1)
        else:
            repo_raw, category = line, "uncategorized"
        repo_id = _normalize_repo_id(repo_raw)
        if repo_id:
            items.append((repo_id, category.strip() or "uncategorized"))
    return items


def _dest_path(root: Path, repo_id: str) -> Path:
    return root / ("models--" + repo_id.replace("/", "--"))


def _download_one(
    repo_id: str,
    *,
    root: Path,
    dry_run: bool,
    skip_existing: bool,
) -> Dict[str, object]:
    dest = _dest_path(root, repo_id)
    already_exists = dest.exists() and any(dest.iterdir())
    if skip_existing and already_exists:
        return {
            "repo_id": repo_id,
            "dest": str(dest),
            "action": "skip-existing",
            "ok": True,
        }
    if dry_run:
        return {
            "repo_id": repo_id,
            "dest": str(dest),
            "action": "dry-run-existing" if already_exists else "dry-run-missing",
            "ok": True,
        }

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {
            "repo_id": repo_id,
            "dest": str(dest),
            "action": "download",
            "ok": False,
            "error": (
                f"huggingface_hub not available: {exc}. "
                "Install with: pip install huggingface_hub"
            ),
        }

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return {
        "repo_id": repo_id,
        "dest": str(dest),
        "action": "update" if already_exists else "download",
        "ok": True,
    }


def _iter_unique(items: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen: Set[str] = set()
    out: List[Tuple[str, str]] = []
    for repo_id, category in items:
        key = repo_id.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((repo_id, category))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/update HF model snapshots from manifest.")
    parser.add_argument(
        "--manifest",
        nargs="+",
        default=[DEFAULT_MANIFEST],
        help=f"One or more manifest files (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--models-root",
        default=_default_models_root(),
        help=(
            "Models root. Defaults to $HF_HOME/hub when HF_HOME is set; "
            f"otherwise {DEFAULT_MODELS_ROOT}"
        ),
    )
    parser.add_argument(
        "--include-category",
        nargs="*",
        default=[],
        help="Only download categories listed here (for example: embedding reranker).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip model repos that already exist in models-root.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions only.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args()

    manifests = [Path(p).expanduser().resolve() for p in args.manifest]
    for p in manifests:
        if not p.exists():
            raise SystemExit(f"Manifest not found: {p}")

    root = Path(args.models_root).expanduser().resolve()
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    all_items: List[Tuple[str, str]] = []
    for m in manifests:
        all_items.extend(_read_manifest(m))
    items = _iter_unique(all_items)

    categories = {c.strip().lower() for c in args.include_category if c.strip()}
    if categories:
        items = [(rid, cat) for rid, cat in items if cat.strip().lower() in categories]

    results: List[Dict[str, object]] = []
    for repo_id, category in items:
        res = _download_one(
            repo_id,
            root=root,
            dry_run=args.dry_run,
            skip_existing=bool(args.skip_existing),
        )
        res["category"] = category
        results.append(res)

    ok = sum(1 for r in results if bool(r.get("ok")))
    fail = len(results) - ok
    payload = {
        "manifests": [str(m) for m in manifests],
        "models_root": str(root),
        "items_total": len(results),
        "ok": ok,
        "failed": fail,
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"models={len(results)} ok={ok} failed={fail}")
    print(f"models_root={root}")
    for r in results:
        prefix = "ok" if bool(r.get("ok")) else "fail"
        print(f"- [{prefix}] {r.get('action','')} {r.get('repo_id','')} ({r.get('category','')})")
        if not bool(r.get("ok")) and r.get("error"):
            print(f"  error: {r.get('error')}")


if __name__ == "__main__":
    main()
