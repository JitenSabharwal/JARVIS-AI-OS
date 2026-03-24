"""
Persistent multi-person identity registry for realtime vision recognition.

This module stores compact image-embedding profiles per enrolled person and
matches incoming person crops against the registry.
"""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infrastructure.logger import get_logger
from infrastructure.multimodal_embedding import MultiModalEmbeddingEngine

logger = get_logger("person_identity_registry")


def _normalize(vec: list[float]) -> list[float]:
    total = sum(v * v for v in vec)
    if total <= 0.0:
        return list(vec)
    norm = total ** 0.5
    return [v / norm for v in vec]


def _mean_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for vec in vectors:
        if len(vec) != dim:
            continue
        for i, value in enumerate(vec):
            out[i] += float(value)
    count = float(max(1, len(vectors)))
    return _normalize([v / count for v in out])


def _decode_b64_image(raw: str) -> bytes:
    value = str(raw or "").strip()
    if not value:
        return b""
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1].strip()
    try:
        return base64.b64decode(value, validate=False)
    except Exception:  # noqa: BLE001
        return b""


@dataclass
class IdentityRecord:
    person_id: str
    display_name: str
    embedding: list[float]
    sample_count: int
    samples: list[dict[str, Any]]
    created_at: float
    updated_at: float
    metadata: dict[str, Any]

    def to_public(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "display_name": self.display_name,
            "sample_count": int(self.sample_count),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_public()
        payload["embedding"] = [float(v) for v in self.embedding]
        payload["samples"] = [
            {
                "sample_id": str(s.get("sample_id") or ""),
                "image_b64": str(s.get("image_b64") or ""),
                "embedding": [float(v) for v in list(s.get("embedding") or [])],
                "created_at": float(s.get("created_at") or time.time()),
            }
            for s in list(self.samples or [])
            if isinstance(s, dict)
        ]
        payload["metadata"] = dict(self.metadata or {})
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IdentityRecord":
        return cls(
            person_id=str(payload.get("person_id") or ""),
            display_name=str(payload.get("display_name") or "").strip(),
            embedding=[float(v) for v in list(payload.get("embedding") or [])],
            sample_count=int(payload.get("sample_count") or 0),
            samples=[
                {
                    "sample_id": str(s.get("sample_id") or ""),
                    "image_b64": str(s.get("image_b64") or ""),
                    "embedding": [float(v) for v in list(s.get("embedding") or [])],
                    "created_at": float(s.get("created_at") or time.time()),
                }
                for s in list(payload.get("samples") or [])
                if isinstance(s, dict)
            ],
            created_at=float(payload.get("created_at") or time.time()),
            updated_at=float(payload.get("updated_at") or time.time()),
            metadata=dict(payload.get("metadata") or {}),
        )


class PersonIdentityRegistry:
    """Simple local identity registry with persistent embeddings."""

    def __init__(
        self,
        *,
        state_path: str = "data/vision_identities.json",
        embedding_backend: str = "local_deterministic",
        embedding_dim: int = 128,
        match_threshold: float = 0.82,
        soft_threshold: float = 0.68,
        margin_threshold: float = 0.02,
    ) -> None:
        self._state_path = Path(state_path).expanduser()
        self._embedder = MultiModalEmbeddingEngine(backend=embedding_backend, dim=embedding_dim)
        self._match_threshold = float(match_threshold)
        self._soft_threshold = float(soft_threshold)
        self._margin_threshold = float(margin_threshold)
        self._records: dict[str, IdentityRecord] = {}
        self._load()

    @classmethod
    def from_env(cls) -> "PersonIdentityRegistry":
        path = str(os.getenv("JARVIS_VISION_IDENTITIES_PATH", "data/vision_identities.json")).strip()
        backend = str(os.getenv("JARVIS_VISION_IDENTITIES_BACKEND", "local_deterministic")).strip()
        try:
            dim = int(os.getenv("JARVIS_VISION_IDENTITIES_DIM", "128") or 128)
        except Exception:
            dim = 128
        try:
            threshold = float(os.getenv("JARVIS_VISION_IDENTITY_THRESHOLD", "0.82") or 0.82)
        except Exception:
            threshold = 0.82
        try:
            soft_threshold = float(os.getenv("JARVIS_VISION_IDENTITY_SOFT_THRESHOLD", "0.68") or 0.68)
        except Exception:
            soft_threshold = 0.68
        try:
            margin = float(os.getenv("JARVIS_VISION_IDENTITY_MARGIN", "0.02") or 0.02)
        except Exception:
            margin = 0.02
        return cls(
            state_path=path,
            embedding_backend=backend,
            embedding_dim=dim,
            match_threshold=threshold,
            soft_threshold=soft_threshold,
            margin_threshold=margin,
        )

    def list_identities(self) -> list[dict[str, Any]]:
        rows = [rec.to_public() for rec in self._records.values()]
        rows.sort(key=lambda r: str(r.get("display_name", "")).lower())
        return rows

    def count(self) -> int:
        return len(self._records)

    def enroll(
        self,
        *,
        display_name: str,
        sample_images_b64: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = str(display_name or "").strip()
        if not name:
            raise ValueError("display_name is required")
        embeddings: list[list[float]] = []
        samples: list[dict[str, Any]] = []
        for raw in sample_images_b64:
            image_bytes = _decode_b64_image(raw)
            if len(image_bytes) < 64:
                continue
            emb = self._embedder.embed_image(image_bytes)
            embeddings.append(emb)
            samples.append(
                {
                    "sample_id": f"smp_{uuid.uuid4().hex[:12]}",
                    "image_b64": str(raw or "").strip(),
                    "embedding": [float(v) for v in emb],
                    "created_at": time.time(),
                }
            )
        if not embeddings:
            raise ValueError("No valid samples were provided")
        now = time.time()
        person_id = f"person_{uuid.uuid4().hex[:12]}"
        record = IdentityRecord(
            person_id=person_id,
            display_name=name,
            embedding=_mean_vectors(embeddings),
            sample_count=len(embeddings),
            samples=samples[:60],
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        self._records[person_id] = record
        self._save()
        logger.info("Enrolled identity %s (%s samples)", person_id, record.sample_count)
        return record.to_public()

    def delete(self, person_id: str) -> bool:
        pid = str(person_id or "").strip()
        if not pid or pid not in self._records:
            return False
        del self._records[pid]
        self._save()
        return True

    def list_samples(self, person_id: str) -> list[dict[str, Any]]:
        pid = str(person_id or "").strip()
        rec = self._records.get(pid)
        if rec is None:
            raise KeyError("identity not found")
        rows = []
        for s in list(rec.samples or []):
            if not isinstance(s, dict):
                continue
            rows.append(
                {
                    "sample_id": str(s.get("sample_id") or ""),
                    "image_b64": str(s.get("image_b64") or ""),
                    "created_at": float(s.get("created_at") or rec.created_at),
                }
            )
        rows.sort(key=lambda r: float(r.get("created_at", 0.0)), reverse=True)
        return rows

    def delete_sample(self, person_id: str, sample_id: str) -> dict[str, Any]:
        pid = str(person_id or "").strip()
        sid = str(sample_id or "").strip()
        rec = self._records.get(pid)
        if rec is None:
            raise KeyError("identity not found")
        if not sid:
            raise ValueError("sample_id is required")
        kept: list[dict[str, Any]] = []
        removed = False
        for s in list(rec.samples or []):
            if not isinstance(s, dict):
                continue
            if str(s.get("sample_id") or "").strip() == sid:
                removed = True
                continue
            kept.append(s)
        if not removed:
            raise KeyError("sample not found")
        if len(kept) < 1:
            raise ValueError("cannot delete last sample; delete identity instead")
        embeddings = [
            [float(v) for v in list(s.get("embedding") or []) if isinstance(v, (int, float))]
            for s in kept
            if isinstance(s, dict)
        ]
        embeddings = [v for v in embeddings if v]
        if not embeddings:
            raise ValueError("remaining samples are invalid")
        rec.samples = kept[:60]
        rec.embedding = _mean_vectors(embeddings)
        rec.sample_count = len(rec.samples)
        rec.updated_at = time.time()
        self._save()
        return rec.to_public()

    def recognize_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not samples:
            return out
        identities = list(self._records.values())
        for item in samples:
            sample_id = item.get("sample_id")
            detection_index = item.get("detection_index")
            image_b64 = str(item.get("image_b64") or "").strip()
            image_bytes = _decode_b64_image(image_b64)
            if len(image_bytes) < 64:
                out.append(
                    {
                        "sample_id": sample_id,
                        "detection_index": detection_index,
                        "unknown": True,
                        "reason": "invalid_sample",
                        "score": 0.0,
                    }
                )
                continue
            emb = self._embedder.embed_image(image_bytes)
            best: IdentityRecord | None = None
            best_score = -1.0
            second = -1.0
            for rec in identities:
                score = self._embedder.cosine_similarity(emb, rec.embedding)
                if score > best_score:
                    second = best_score
                    best_score = score
                    best = rec
                elif score > second:
                    second = score
            margin = float(best_score - second) if second >= 0.0 else float(best_score)
            single_identity_mode = len(identities) <= 1
            strong_accept = best is not None and best_score >= self._match_threshold and (
                single_identity_mode or margin >= self._margin_threshold
            )
            soft_accept = best is not None and best_score >= self._soft_threshold and (
                single_identity_mode or margin >= max(self._margin_threshold, 0.01)
            )
            accepted = strong_accept or soft_accept
            payload: dict[str, Any] = {
                "sample_id": sample_id,
                "detection_index": detection_index,
                "unknown": not accepted,
                "score": round(float(best_score if best_score > 0 else 0.0), 4),
                "margin": round(float(margin), 4),
            }
            if best is not None:
                payload.update(
                    {
                        "candidate_person_id": best.person_id,
                        "candidate_display_name": best.display_name,
                    }
                )
            if accepted and best is not None:
                payload.update(best.to_public())
            out.append(payload)
        return out

    def _load(self) -> None:
        self._records = {}
        try:
            if not self._state_path.exists():
                return
            payload = self._state_path.read_text(encoding="utf-8")
            data = json.loads(payload)
            rows = list(data.get("records") or [])
            for row in rows:
                if not isinstance(row, dict):
                    continue
                rec = IdentityRecord.from_dict(row)
                if rec.person_id and rec.display_name and rec.embedding:
                    self._records[rec.person_id] = rec
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load identity registry %s: %s", self._state_path, exc)

    def _save(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "updated_at": time.time(),
                "records": [rec.to_dict() for rec in self._records.values()],
            }
            self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save identity registry %s: %s", self._state_path, exc)


__all__ = ["PersonIdentityRegistry"]
