from infrastructure.social_scene_orchestrator import SocialSceneOrchestrator


def test_social_scene_orchestrator_emits_identity_event_and_prompt() -> None:
    orch = SocialSceneOrchestrator()
    out = orch.ingest_detections(
        session_id="s1",
        detections=[
            {"label": "person", "score": 0.91, "bbox": [10, 10, 100, 140], "trackId": "trk-1"},
        ],
        now=1000.0,
    )
    assert isinstance(out.get("events"), list)
    out2 = orch.ingest_detections(
        session_id="s1",
        detections=[
            {
                "label": "person",
                "score": 0.93,
                "bbox": [12, 12, 100, 140],
                "trackId": "trk-1",
                "personId": "pid-jiten",
                "identity": "Jiten",
                "identityScore": 0.88,
            },
        ],
        now=1001.0,
    )
    event_types = [str(e.get("type", "")) for e in out2.get("events", [])]
    assert "person_identified" in event_types
    assert "greet" in str(out2.get("prompt_hint", "")).lower()
    assert out2.get("coverage", {}).get("matched", 0) >= 1


def test_social_scene_orchestrator_cooldown_prevents_repeat_identified_spam() -> None:
    orch = SocialSceneOrchestrator()
    base = {
        "label": "person",
        "score": 0.9,
        "bbox": [10, 10, 80, 120],
        "trackId": "trk-1",
        "personId": "pid-1",
        "identity": "Alex",
        "identityScore": 0.85,
    }
    first = orch.ingest_detections(session_id="s1", detections=[base], now=2000.0)
    second = orch.ingest_detections(session_id="s1", detections=[base], now=2001.0)
    first_ident = [e for e in first.get("events", []) if str(e.get("type")) == "person_identified"]
    second_ident = [e for e in second.get("events", []) if str(e.get("type")) == "person_identified"]
    assert len(first_ident) >= 1
    assert len(second_ident) == 0


def test_social_scene_orchestrator_person_left_event_after_grace() -> None:
    orch = SocialSceneOrchestrator()
    orch.ingest_detections(
        session_id="s1",
        detections=[{"label": "person", "score": 0.86, "bbox": [1, 1, 40, 80], "trackId": "trk-9"}],
        now=3000.0,
    )
    out = orch.ingest_detections(session_id="s1", detections=[], now=3003.0)
    event_types = [str(e.get("type", "")) for e in out.get("events", [])]
    assert "person_left" in event_types


def test_social_scene_timeline_and_explain() -> None:
    orch = SocialSceneOrchestrator()
    orch.ingest_detections(
        session_id="s2",
        detections=[{"label": "person", "score": 0.9, "bbox": [0, 0, 60, 120], "trackId": "trk-a"}],
        now=4000.0,
    )
    timeline = orch.get_timeline("s2", limit=10)
    assert timeline["session_id"] == "s2"
    assert timeline["count"] >= 1
    event_id = str(timeline["items"][0]["event_id"])
    explain = orch.explain_event("s2", event_id=event_id)
    assert explain["found"] is True
    assert explain["event"]["event_id"] == event_id
