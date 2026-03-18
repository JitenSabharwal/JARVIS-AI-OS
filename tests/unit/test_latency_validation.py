from infrastructure.latency_validation import percentile, validate_latency_budget


def test_percentile_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 0.50) == 30.0
    assert percentile(values, 0.95) == 50.0


def test_validate_latency_budget_pass_fail() -> None:
    pass_result = validate_latency_budget([100.0, 120.0, 110.0, 125.0, 130.0], target_p95_ms=150.0)
    assert pass_result.sample_count == 5
    assert pass_result.passed is True

    fail_result = validate_latency_budget([200.0, 220.0, 230.0, 240.0, 260.0], target_p95_ms=210.0)
    assert fail_result.sample_count == 5
    assert fail_result.passed is False
