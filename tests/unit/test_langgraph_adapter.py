from infrastructure.langgraph_adapter import LangGraphWorkflowAdapter


def test_langgraph_adapter_build_execution_waves_fallback() -> None:
    adapter = LangGraphWorkflowAdapter(enabled=True)
    waves = adapter.build_execution_waves(
        [
            {"name": "a", "depends_on": []},
            {"name": "b", "depends_on": ["a"]},
            {"name": "c", "depends_on": ["a"]},
            {"name": "d", "depends_on": ["b", "c"]},
        ]
    )
    assert waves[0] == ["a"]
    assert set(waves[1]) == {"b", "c"}
    assert waves[2] == ["d"]


def test_langgraph_adapter_respects_max_wave_size() -> None:
    adapter = LangGraphWorkflowAdapter(enabled=False, max_wave_size=1)
    waves = adapter.build_execution_waves(
        [
            {"name": "a", "depends_on": []},
            {"name": "b", "depends_on": ["a"]},
            {"name": "c", "depends_on": ["a"]},
            {"name": "d", "depends_on": ["a"]},
        ]
    )
    assert waves == [["a"], ["b"], ["c"], ["d"]]


def test_langgraph_adapter_uses_langgraph_path_when_available(monkeypatch) -> None:
    adapter = LangGraphWorkflowAdapter(enabled=True)
    monkeypatch.setattr(adapter, "_langgraph_available", True)

    def fake_langgraph(_steps):
        return [["a"], ["b"]]

    monkeypatch.setattr(adapter, "_build_execution_waves_with_langgraph", fake_langgraph)
    waves = adapter.build_execution_waves(
        [
            {"name": "a", "depends_on": []},
            {"name": "b", "depends_on": ["a"]},
        ]
    )
    assert waves == [["a"], ["b"]]
    assert adapter.last_plan_meta.get("engine") == "langgraph"


def test_langgraph_adapter_multi_agent_flow_hints() -> None:
    adapter = LangGraphWorkflowAdapter(enabled=False)
    flow = adapter.build_multi_agent_flow(
        [
            {"name": "plan", "capability": "plan_workflow", "depends_on": []},
            {"name": "code", "capability": "update_codebase", "depends_on": ["plan"]},
            {"name": "verify", "capability": "verify_policy", "depends_on": ["code"]},
        ]
    )
    waves = flow.get("waves", [])
    hints = flow.get("step_hints", {})
    assert waves and waves[0] == ["plan"]
    assert isinstance(hints, dict)
    assert hints["plan"]["lane"] == "manager_lane"
    assert hints["code"]["lane"] == "developer_lane"
    assert hints["verify"]["lane"] == "verifier_lane"


def test_langgraph_adapter_execution_state_transition_retry() -> None:
    adapter = LangGraphWorkflowAdapter(enabled=False)
    state = adapter.init_execution_state(
        [
            {"name": "a", "capability": "plan_workflow", "depends_on": []},
            {"name": "b", "capability": "update_codebase", "depends_on": ["a"]},
        ],
        max_retries=1,
    )
    assert state["ready"] == ["a"]
    state = adapter.transition_execution_state(state, [{"step": "a", "status": "completed"}])
    assert "b" in state["ready"]
    state = adapter.transition_execution_state(state, [{"step": "b", "status": "failed"}])
    # first failure remains pending due to retry budget
    assert state["steps"]["b"]["status"] == "pending"
