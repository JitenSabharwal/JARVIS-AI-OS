from __future__ import annotations

from infrastructure.slo_metrics import (
    SLOMetrics,
    evaluate_slo_snapshot,
    get_slo_metrics,
    reset_slo_metrics,
)


def test_slo_metrics_counter_and_latency_snapshot() -> None:
    metrics = SLOMetrics()
    metrics.inc("api_requests_total", label="GET /api/v1/health")
    metrics.observe_latency("api_request_latency_ms", 10.0, label="GET /api/v1/health")
    metrics.observe_latency("api_request_latency_ms", 30.0, label="GET /api/v1/health")

    snap = metrics.snapshot()
    assert snap["counters"]["api_requests_total:GET /api/v1/health"] == 1
    lat = snap["latency"]["api_request_latency_ms:GET /api/v1/health"]
    assert lat["count"] == 2
    assert lat["p95_ms"] >= lat["p50_ms"]


def test_global_slo_metrics_singleton_reset() -> None:
    reset_slo_metrics()
    m1 = get_slo_metrics()
    m1.inc("x", label="y")
    snap1 = m1.snapshot()
    assert snap1["counters"]["x:y"] == 1

    reset_slo_metrics()
    m2 = get_slo_metrics()
    snap2 = m2.snapshot()
    assert "x:y" not in snap2["counters"]


def test_evaluate_slo_snapshot_detects_violation() -> None:
    metrics = SLOMetrics()
    for _ in range(30):
        metrics.observe_latency("api_request_latency_ms", 2500.0, label="GET /api/v1/status")
        metrics.inc("api_responses_total", label="GET /api/v1/status 500")
        metrics.inc("api_errors_total", label="5xx")

    snap = metrics.snapshot()
    slo = evaluate_slo_snapshot(snap)
    assert slo["healthy"] is False
    assert any(v["metric"] == "api_request_p95_ms" for v in slo["violations"])
    assert any(v["metric"] == "api_error_rate_pct" for v in slo["violations"])


def test_evaluate_slo_snapshot_healthy_when_under_thresholds() -> None:
    metrics = SLOMetrics()
    for _ in range(30):
        metrics.observe_latency("api_request_latency_ms", 100.0, label="GET /api/v1/health")
        metrics.inc("api_responses_total", label="GET /api/v1/health 200")

    snap = metrics.snapshot()
    slo = evaluate_slo_snapshot(snap)
    assert slo["healthy"] is True
    assert slo["violations"] == []
