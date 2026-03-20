from __future__ import annotations

from infrastructure.policy_cost_engine import PolicyContext, PolicyCostEngine


def test_policy_cost_engine_budget_prefers_local() -> None:
    engine = PolicyCostEngine(enabled=True)
    decision = engine.decide(
        PolicyContext(
            route="chat",
            task_type="general_query",
            user_id="u1",
            budget_usd=0.001,
        )
    )
    assert decision.prefer_local is True
    assert "budget" in decision.reason
