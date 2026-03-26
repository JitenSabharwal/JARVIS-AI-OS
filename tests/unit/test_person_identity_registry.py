from __future__ import annotations

import base64
import time

from infrastructure.person_identity_registry import IdentityRecord, PersonIdentityRegistry


class _StubEmbedder:
    def __init__(self, scores_by_embedding: dict[tuple[float, ...], float]) -> None:
        self._scores_by_embedding = dict(scores_by_embedding)

    def embed_image(self, _image_bytes: bytes) -> list[float]:
        return [42.0]

    def cosine_similarity(self, _a: list[float], b: list[float]) -> float:
        return float(self._scores_by_embedding.get(tuple(float(x) for x in b), 0.0))


def _sample_b64() -> str:
    raw = b"\x89JPEG" * 40
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def _identity(person_id: str, name: str, emb: list[float]) -> IdentityRecord:
    now = time.time()
    return IdentityRecord(
        person_id=person_id,
        display_name=name,
        embedding=list(emb),
        sample_count=1,
        samples=[],
        created_at=now,
        updated_at=now,
        metadata={},
    )


def test_single_identity_requires_strong_score() -> None:
    reg = PersonIdentityRegistry(
        state_path="/tmp/test_vision_identities_single.json",
        match_threshold=0.82,
        soft_threshold=0.68,
        margin_threshold=0.02,
        single_identity_match_threshold=0.9,
    )
    reg._records = {"person_jiten": _identity("person_jiten", "Jiten", [1.0, 0.0])}
    reg._embedder = _StubEmbedder({(1.0, 0.0): 0.75})

    out = reg.recognize_samples(
        [{"sample_id": "det-0", "detection_index": 0, "image_b64": _sample_b64()}]
    )
    assert len(out) == 1
    assert out[0]["unknown"] is True
    assert out[0]["candidate_display_name"] == "Jiten"
    assert out[0]["score"] == 0.75


def test_multi_identity_soft_accept_with_margin_still_allowed() -> None:
    reg = PersonIdentityRegistry(
        state_path="/tmp/test_vision_identities_multi.json",
        match_threshold=0.82,
        soft_threshold=0.68,
        margin_threshold=0.02,
        single_identity_match_threshold=0.9,
    )
    reg._records = {
        "person_alex": _identity("person_alex", "Alex", [1.0, 0.0]),
        "person_sam": _identity("person_sam", "Sam", [0.0, 1.0]),
    }
    reg._embedder = _StubEmbedder({(1.0, 0.0): 0.70, (0.0, 1.0): 0.65})

    out = reg.recognize_samples(
        [{"sample_id": "det-1", "detection_index": 1, "image_b64": _sample_b64()}]
    )
    assert len(out) == 1
    assert out[0]["unknown"] is False
    assert out[0]["display_name"] == "Alex"
    assert out[0]["person_id"] == "person_alex"
