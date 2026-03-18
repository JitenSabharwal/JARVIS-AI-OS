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
