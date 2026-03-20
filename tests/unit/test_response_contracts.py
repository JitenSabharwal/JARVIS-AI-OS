from __future__ import annotations

from core.response_contracts import (
    CodeAssistRequest,
    CodeWorkflowRequest,
    RepoUnderstandRequest,
    RequestEnvelope,
    VerifiedResponse,
)


def test_verified_response_from_repo_result_and_text() -> None:
    result = {
        "summary": "Repository uses layered architecture.",
        "architecture": "API routes call orchestrator and specialized agents.",
        "entry_points": ["jarvis_main.py", "main.py"],
        "key_modules": ["interfaces", "agents", "core"],
        "important_flows": ["request->router->agent->response"],
        "risks": ["Missing tests on edge paths."],
        "next_steps": ["Add integration tests."],
        "evidence": [{"claim": "Entry points detected.", "sources": ["jarvis_main.py", "main.py"]}],
        "confidence": 0.82,
        "coverage_pct": 42.5,
        "open_questions": ["Validate production config paths."],
        "signals": {"top_directories": ["interfaces", "agents"], "dependency_files": ["requirements.txt"]},
        "analysis_plan": ["Identify entry points", "Trace runtime flow"],
    }
    verified = VerifiedResponse.from_repo_result(result)
    text = verified.to_user_text()
    assert "Repository uses layered architecture." in text
    assert "Architecture:" in text
    assert "Entry points:" in text
    assert "Evidence:" in text
    assert "Confidence:" in text


def test_request_envelope_defaults_and_normalization() -> None:
    env = RequestEnvelope.from_any(
        {
            "request_id": "r1",
            "user_id": "",
            "session_id": "s1",
            "model": "",
            "workspace_path": "/tmp/repo",
            "route": "chat",
        }
    )
    assert env.request_id == "r1"
    assert env.user_id == "continue_user"
    assert env.model == "jarvis-default"
    assert env.workspace_path == "/tmp/repo"


def test_request_contract_parsers() -> None:
    code_req, code_err = CodeAssistRequest.from_any(
        {"workspace_path": "/tmp/repo", "instruction": "fix bug", "max_files": 50}
    )
    assert code_err == ""
    assert code_req is not None
    assert code_req.max_files == 50

    repo_req, repo_err = RepoUnderstandRequest.from_any(
        {"workspace_path": "/tmp/repo", "question": "explain", "depth": "high"}
    )
    assert repo_err == ""
    assert repo_req is not None
    assert repo_req.depth == "high"

    wf_req, wf_err = CodeWorkflowRequest.from_any(
        {"workspace_path": "/tmp/repo", "goal": "refactor", "max_workers": 9}
    )
    assert wf_err == ""
    assert wf_req is not None
    assert wf_req.max_workers == 8
