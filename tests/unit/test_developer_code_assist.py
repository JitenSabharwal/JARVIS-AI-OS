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
