"""
Lightweight multimodal embedding utilities (text + image) for local retrieval.
"""

from __future__ import annotations

import base64
import hashlib
import math
import re
from pathlib import Path
from typing import Dict, List, Optional


class MultiModalEmbeddingEngine:
    """Deterministic local embeddings for text and image bytes."""

    def __init__(self, *, backend: str = "local_deterministic", dim: int = 64) -> None:
        requested = str(backend or "local_deterministic").strip().lower()
        self.backend_requested = requested
        self.backend = requested
        self._mlx_available = False
        if requested == "mlx_clip":
            # Future-ready hook: use MLX CLIP when available, else fallback.
            try:
                import mlx  # type: ignore  # noqa: F401

                self._mlx_available = True
            except Exception:  # noqa: BLE001
                self.backend = "local_deterministic"
        self.dim = max(16, int(dim))

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        clean = str(text or "").strip().lower()
        if not clean:
            return vec
        tokens = re.findall(r"[a-z0-9_]+", clean)
        if not tokens:
            tokens = clean.split()
        for tok in tokens:
            digest = hashlib.blake2b(tok.encode("utf-8", errors="ignore"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little", signed=False)
            idx = raw % self.dim
            sign = -1.0 if ((raw >> 8) & 1) else 1.0
            weight = 1.0 + ((raw >> 16) & 0x0F) / 32.0
            vec[idx] += sign * weight
        return self._normalize(vec)

    def embed_image(self, image_bytes: bytes) -> List[float]:
        vec = [0.0] * self.dim
        if not image_bytes:
            return vec
        limit = min(len(image_bytes), 1_000_000)
        data = image_bytes[:limit]
        stride = max(1, len(data) // 25_000)
        pos = 0
        for b in data[::stride]:
            idx = (pos * 131 + int(b)) % self.dim
            vec[idx] += (float(b) / 255.0) + 0.01
            pos += 1
        return self._normalize(vec)

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _normalize(vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0.0:
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def image_bytes_from_metadata(metadata: Dict[str, object]) -> Optional[bytes]:
        b64_keys = ("image_b64", "image_base64", "image_bytes_b64")
        for key in b64_keys:
            raw = metadata.get(key)
            if isinstance(raw, str) and raw.strip():
                try:
                    return base64.b64decode(raw.strip(), validate=False)
                except Exception:  # noqa: BLE001
                    pass
        raw_path = metadata.get("image_path")
        if isinstance(raw_path, str) and raw_path.strip():
            p = Path(raw_path).expanduser()
            try:
                if p.exists() and p.is_file():
                    return p.read_bytes()
            except Exception:  # noqa: BLE001
                return None
        return None


__all__ = ["MultiModalEmbeddingEngine"]
