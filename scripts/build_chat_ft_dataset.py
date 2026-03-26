#!/usr/bin/env python3
"""
Build fine-tune ready chat datasets from ConversationManager chat event logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = str(line or "").strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _as_messages(
    row: dict[str, Any],
    *,
    system_prompt: str,
    max_chars: int,
    include_metadata: bool,
) -> dict[str, Any] | None:
    outcome = str(row.get("outcome", "")).strip().lower()
    if outcome != "ok":
        return None
    user_text = str(row.get("user_input", "")).strip()
    assistant_text = str(row.get("assistant_response", "")).strip()
    if not user_text or not assistant_text:
        return None
    if "couldn't generate" in assistant_text.lower():
        return None
    item: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt[:max_chars]},
            {"role": "user", "content": user_text[:max_chars]},
            {"role": "assistant", "content": assistant_text[:max_chars]},
        ]
    }
    if include_metadata:
        item["metadata"] = {
            "source": "chat_training_events",
            "timestamp": row.get("timestamp"),
            "intent": str(row.get("intent", "")).strip(),
            "task_type": str(row.get("task_type", "")).strip(),
            "modality": str(row.get("modality", "")).strip(),
            "session_id": str(row.get("session_id", "")).strip(),
            "user_id": str(row.get("user_id", "")).strip(),
        }
    return item


def _stable_key(item: dict[str, Any]) -> str:
    msgs = item.get("messages", [])
    user = ""
    assistant = ""
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "")).strip().lower()
            content = str(m.get("content", "")).strip()
            if role == "user":
                user = content
            elif role == "assistant":
                assistant = content
    digest = hashlib.sha256(f"{user}\n{assistant}".encode("utf-8")).hexdigest()
    return digest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val JSONL from chat event logs.")
    parser.add_argument("--input", default="data/chat_training_events.jsonl")
    parser.add_argument("--out-train", default="data/ft/chat_train.jsonl")
    parser.add_argument("--out-val", default="data/ft/chat_val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument(
        "--system-prompt",
        default="You are JARVIS. Be natural, clear, and helpful. Give final answers only.",
    )
    parser.add_argument("--include-metadata", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    rows = _iter_jsonl(in_path)
    examples: list[dict[str, Any]] = []
    for row in rows:
        ex = _as_messages(
            row,
            system_prompt=str(args.system_prompt),
            max_chars=max(128, int(args.max_chars)),
            include_metadata=bool(args.include_metadata),
        )
        if ex:
            examples.append(ex)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ex in examples:
        key = _stable_key(ex)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)

    rnd = random.Random(int(args.seed))
    rnd.shuffle(deduped)
    if int(args.max_samples) > 0:
        deduped = deduped[: int(args.max_samples)]
    val_ratio = max(0.0, min(0.5, float(args.val_ratio)))
    val_count = int(round(len(deduped) * val_ratio))
    val_count = max(0, min(len(deduped), val_count))
    val_rows = deduped[:val_count]
    train_rows = deduped[val_count:]

    train_path = Path(args.out_train).expanduser().resolve()
    val_path = Path(args.out_val).expanduser().resolve()
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    summary = {
        "version": 1,
        "input": str(in_path),
        "rows_in": len(rows),
        "examples_kept": len(examples),
        "examples_deduped": len(deduped),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "out_train": str(train_path),
        "out_val": str(val_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
