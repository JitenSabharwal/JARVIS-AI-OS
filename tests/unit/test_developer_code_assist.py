from __future__ import annotations

import json

import pytest

from agents.specialized_agents import DeveloperAgent
from infrastructure.model_router import CallableModelProvider, ModelRouter


@pytest.mark.asyncio
async def test_update_codebase_applies_replace_edit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_CODE_ALLOWED_ROOTS", str(tmp_path))
    (tmp_path / "package.json").write_text('{"name":"demo","scripts":{}}', encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    target = src / "App.jsx"
    target.write_text("export default function App(){ return <div>Hello</div>; }", encoding="utf-8")

    async def planner_handler(_request):
        return json.dumps(
            {
                "summary": "Update greeting",
                "edits": [
                    {
                        "path": "src/App.jsx",
                        "operation": "replace",
                        "find": "Hello",
                        "replace": "Hello Jarvis",
                    }
                ],
            }
        )

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=planner_handler,
            supported_modalities={"text"},
        )
    )
    agent = DeveloperAgent()
    agent.set_model_router(router)

    result = await agent.handle_update_codebase(
        {
            "workspace_path": str(tmp_path),
            "instruction": "Update the App greeting",
            "dry_run": False,
            "run_checks": False,
            "max_files": 5,
        }
    )
    assert result["applied_count"] == 1
    assert "src/App.jsx" in result["files_touched"]
    assert "Hello Jarvis" in target.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_update_codebase_dry_run_does_not_write(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_CODE_ALLOWED_ROOTS", str(tmp_path))
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    target = tmp_path / "main.py"
    target.write_text("print('old')\n", encoding="utf-8")

    async def planner_handler(_request):
        return json.dumps(
            {
                "summary": "Change output",
                "edits": [
                    {
                        "path": "main.py",
                        "operation": "replace",
                        "find": "old",
                        "replace": "new",
                    }
                ],
            }
        )

    router = ModelRouter(
        local_provider=CallableModelProvider(
            name="local",
            provider_type="local",
            handler=planner_handler,
            supported_modalities={"text"},
        )
    )
    agent = DeveloperAgent()
    agent.set_model_router(router)

    result = await agent.handle_update_codebase(
        {
            "workspace_path": str(tmp_path),
            "instruction": "Change print to new",
            "dry_run": True,
            "run_checks": False,
            "max_files": 5,
        }
    )
    assert result["applied_count"] == 1
    assert target.read_text(encoding="utf-8") == "print('old')\n"


def test_detect_analysis_contradictions() -> None:
    analysis = {
        "summary": "There are no API routes in this codebase.",
        "architecture": "No route handlers present.",
        "entry_points": ["jarvis_main.py"],
    }
    facts = {"route_files": ["interfaces/api_interface.py"]}
    contradictions = DeveloperAgent._detect_analysis_contradictions(analysis=analysis, facts=facts)
    assert contradictions


def test_apply_evidence_policy_filters_unknown_sources() -> None:
    evidence = [
        {"claim": "Known", "sources": ["interfaces/api_interface.py"]},
        {"claim": "Unknown", "sources": ["missing/path.py"]},
    ]
    file_samples = [{"path": "interfaces/api_interface.py", "snippet": "x"}]
    repo_index = {"entries": [{"path": "core/config.py"}]}
    kept, dropped = DeveloperAgent._apply_evidence_policy(
        evidence=evidence,
        file_samples=file_samples,
        repo_index=repo_index,
    )
    assert len(kept) == 1
    assert kept[0]["claim"] == "Known"
    assert dropped == ["Unknown"]


@pytest.mark.asyncio
async def test_high_depth_pipeline_includes_reviewer_stage(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("JARVIS_CODE_ALLOWED_ROOTS", str(tmp_path))
    agent = DeveloperAgent()

    async def fake_analyze(**_kwargs):
        return {
            "summary": "s",
            "architecture": "a",
            "entry_points": ["main.py"],
            "key_modules": ["interfaces"],
            "important_flows": ["request -> response"],
            "risks": ["r"],
            "next_steps": ["n"],
            "evidence": [{"claim": "c", "sources": ["main.py"]}],
        }

    async def fake_route(*, prompt, task_type, privacy_level):  # noqa: ARG001
        if "You are a reviewer agent" in prompt:
            return json.dumps(
                {
                    "summary": "reviewed",
                    "architecture": "reviewed-arch",
                    "entry_points": ["main.py"],
                    "key_modules": ["interfaces"],
                    "important_flows": ["request -> response"],
                    "risks": ["r"],
                    "next_steps": ["n"],
                    "evidence": [{"claim": "c", "sources": ["main.py"]}],
                }
            )
        if "You are a verifier agent" in prompt:
            return json.dumps(
                {
                    "summary": "verified",
                    "architecture": "verified-arch",
                    "entry_points": ["main.py"],
                    "key_modules": ["interfaces"],
                    "important_flows": ["request -> response"],
                    "risks": ["r"],
                    "next_steps": ["n"],
                    "evidence": [{"claim": "c", "sources": ["main.py"]}],
                }
            )
        return "{}"

    monkeypatch.setattr(agent, "_analyze_codebase", fake_analyze)
    monkeypatch.setattr(agent, "_route_text_generation", fake_route)

    result = await agent._run_high_depth_analysis_pipeline(  # type: ignore[attr-defined]
        workspace=tmp_path,
        stack="python",
        question="explain",
        file_samples=[{"path": "main.py", "snippet": "print('x')"}],
        repo_tree="main.py",
        facts={},
        analysis_plan=["one", "two"],
        max_rounds=2,
    )
    pipeline = result.get("_depth_pipeline", {})
    assert isinstance(pipeline, dict)
    assert pipeline.get("stages") == ["planner", "analyst", "reviewer", "verifier"]
    assert int(pipeline.get("rounds", 0)) == 2
