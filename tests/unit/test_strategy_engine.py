from __future__ import annotations

from core.strategy_engine import StrategyEngine


def test_strategy_engine_adaptive_changes_caps() -> None:
    engine = StrategyEngine(adaptive_enabled=True)
    decision = engine.select(
        lane_caps={"developer_lane": 1, "analyst_lane": 1},
        lane_pressure={"developer_lane": 5, "analyst_lane": 1},
    )
    assert decision.strategy_id == "adaptive"
    assert decision.lane_caps["developer_lane"] >= 2
