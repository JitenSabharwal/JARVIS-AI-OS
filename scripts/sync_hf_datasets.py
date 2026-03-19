#!/usr/bin/env python3
"""
Sync Hugging Face datasets listed in a manifest into local dataset root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


DEFAULT_MANIFEST = "config/hf_datasets_manifest.txt"
DEFAULT_DATASET_ROOT = "/Volumes/Jiten-2026/AI_SSD/ai-research/datasets"
DEFAULT_DOMAIN_INDEX_FILE = ".jarvis_dataset_domains.json"


def _read_manifest(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _domain_tag_from_manifest(path: Path) -> str:
    stem = path.stem
    if stem == "hf_datasets_manifest":
        return "all"
    prefix = "hf_datasets_"
    suffix = "_manifest"
    if stem.startswith(prefix) and stem.endswith(suffix):
        mid = stem[len(prefix) : -len(suffix)].strip().lower()
        return mid or "unknown"
    return "unknown"


def _dataset_id_from_url(url: str) -> str:
    parts = [p for p in url.strip().split("/") if p]
    if "datasets" not in parts:
        raise ValueError(f"Not a datasets URL: {url}")
    idx = parts.index("datasets")
    if idx + 2 >= len(parts):
        raise ValueError(f"Invalid dataset URL: {url}")
    owner = parts[idx + 1]
    name = parts[idx + 2]
    return f"{owner}/{name}"


def _dest_path(root: Path, dataset_id: str) -> Path:
    owner, name = dataset_id.split("/", 1)
    return root / "huggingface" / owner / name


def _run(cmd: List[str], cwd: Path | None = None) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    merged = "\n".join(x for x in [out, err] if x)
    return proc.returncode, merged


def _maybe_lfs_pull(dest: Path) -> Tuple[bool, str]:
    code, out = _run(["git", "lfs", "version"])
    if code != 0:
        return False, "git-lfs not available; skipping git lfs pull"
    code, out = _run(["git", "lfs", "pull"], cwd=dest)
    return code == 0, out[:2000]


def _sync_one(url: str, root: Path, update: bool, dry_run: bool, with_lfs: bool) -> dict:
    dataset_id = _dataset_id_from_url(url)
    dest = _dest_path(root, dataset_id)
    repo_url = f"https://huggingface.co/datasets/{dataset_id}"

    if dry_run:
        return {"dataset": dataset_id, "dest": str(dest), "action": "dry-run"}

    dest.parent.mkdir(parents=True, exist_ok=True)

    if not dest.exists():
        code, out = _run(["git", "clone", repo_url, str(dest)])
        result = {
            "dataset": dataset_id,
            "dest": str(dest),
            "action": "clone",
            "ok": code == 0,
            "output": out[:2000],
        }
        if code == 0 and with_lfs:
            lfs_ok, lfs_out = _maybe_lfs_pull(dest)
            result["lfs_ok"] = lfs_ok
            if lfs_out:
                result["lfs_output"] = lfs_out
        return result

    if not update:
        return {"dataset": dataset_id, "dest": str(dest), "action": "skip-existing", "ok": True}

    if not (dest / ".git").exists():
        return {
            "dataset": dataset_id,
            "dest": str(dest),
            "action": "skip-non-git-folder",
            "ok": False,
            "output": "Destination exists but is not a git repo.",
        }

    code, out = _run(["git", "pull", "--ff-only"], cwd=dest)
    result = {
        "dataset": dataset_id,
        "dest": str(dest),
        "action": "update",
        "ok": code == 0,
        "output": out[:2000],
    }
    if code == 0 and with_lfs:
        lfs_ok, lfs_out = _maybe_lfs_pull(dest)
        result["lfs_ok"] = lfs_ok
        if lfs_out:
            result["lfs_output"] = lfs_out
    return result


def _iter_sync(urls: Iterable[str], root: Path, update: bool, dry_run: bool, with_lfs: bool) -> List[dict]:
    results: List[dict] = []
    for url in urls:
        try:
            results.append(_sync_one(url, root=root, update=update, dry_run=dry_run, with_lfs=with_lfs))
        except Exception as exc:  # noqa: BLE001
            results.append({"url": url, "action": "error", "ok": False, "error": str(exc)})
    return results


def _write_domain_index(out_path: Path, url_domain_tags: Dict[str, Set[str]]) -> None:
    dataset_map: Dict[str, List[str]] = {}
    for url, tags in sorted(url_domain_tags.items()):
        dataset_id = _dataset_id_from_url(url)
        dataset_map[dataset_id] = sorted(tags)
    payload = {
        "version": 1,
        "mapping": dataset_map,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone/update Hugging Face datasets from manifest.")
    parser.add_argument(
        "--manifest",
        nargs="+",
        default=[DEFAULT_MANIFEST],
        help=f"One or more manifest files (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--domain-index-out",
        default="",
        help=(
            "Optional output path for dataset domain index JSON "
            f"(default: <dataset-root>/{DEFAULT_DOMAIN_INDEX_FILE})"
        ),
    )
    parser.add_argument("--update", action="store_true", help="Update already cloned repos with git pull")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Clone missing datasets only; skip updates for existing repos",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without cloning/updating")
    parser.add_argument(
        "--no-lfs",
        action="store_true",
        help="Skip git lfs pull after clone/update",
    )
    args = parser.parse_args()

    manifest_paths = [Path(p).expanduser().resolve() for p in args.manifest]
    root = Path(args.dataset_root).expanduser().resolve()
    for manifest in manifest_paths:
        if not manifest.exists():
            raise SystemExit(f"Manifest not found: {manifest}")
    if not bool(args.dry_run):
        root.mkdir(parents=True, exist_ok=True)
    urls: List[str] = []
    url_domain_tags: Dict[str, Set[str]] = {}
    seen: set[str] = set()
    for manifest in manifest_paths:
        domain_tag = _domain_tag_from_manifest(manifest)
        for url in _read_manifest(manifest):
            if url not in url_domain_tags:
                url_domain_tags[url] = set()
            url_domain_tags[url].add(domain_tag)
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
    update_enabled = bool(args.update) and not bool(args.download_only)
    results = _iter_sync(
        urls,
        root=root,
        update=update_enabled,
        dry_run=bool(args.dry_run),
        with_lfs=not bool(args.no_lfs),
    )

    domain_index_out = (
        Path(args.domain_index_out).expanduser().resolve()
        if str(args.domain_index_out).strip()
        else (root / DEFAULT_DOMAIN_INDEX_FILE)
    )
    _write_domain_index(domain_index_out, url_domain_tags)

    ok_count = sum(1 for r in results if r.get("ok", True))
    fail_count = len(results) - ok_count
    print(f"datasets={len(results)} ok={ok_count} failed={fail_count}")
    print(f"domain_index={domain_index_out}")
    for r in results:
        action = r.get("action", "unknown")
        dataset = r.get("dataset") or r.get("url", "")
        ok = r.get("ok", True)
        print(f"- [{ 'ok' if ok else 'fail' }] {action} {dataset}")
        output = str(r.get("output", "")).strip()
        if output and not ok:
            print(f"  output: {output}")
        if not ok and "error" in r:
            print(f"  error: {r.get('error')}")


if __name__ == "__main__":
    main()
