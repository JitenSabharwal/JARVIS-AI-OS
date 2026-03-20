from __future__ import annotations

from interfaces.api_interface import APIInterface


def test_parse_code_assist_chat_request_slash_command() -> None:
    parsed = APIInterface._parse_code_assist_chat_request(
        last_user="/code --workspace /tmp/proj --dry-run --no-checks update api client",
        body={},
    )
    assert parsed is not None
    assert parsed["workspace_path"] == "/tmp/proj"
    assert parsed["instruction"] == "update api client"
    assert parsed["dry_run"] is True
    assert parsed["run_checks"] is False


def test_parse_code_assist_chat_request_mode_payload() -> None:
    parsed = APIInterface._parse_code_assist_chat_request(
        last_user="ignored",
        body={
            "jarvis_mode": "code_assist",
            "workspace_path": "/tmp/project",
            "instruction": "fix lint errors",
            "dry_run": False,
            "run_checks": True,
        },
    )
    assert parsed is not None
    assert parsed["workspace_path"] == "/tmp/project"
    assert parsed["instruction"] == "fix lint errors"


def test_parse_chat_tool_request_repo_command() -> None:
    parsed = APIInterface._parse_chat_tool_request(
        last_user="/repo --workspace /tmp/proj --max-files 25 --depth high --no-tree explain architecture",
        body={},
    )
    assert parsed is not None
    assert parsed["type"] == "repo_understand"
    assert parsed["workspace_path"] == "/tmp/proj"
    assert parsed["question"] == "explain architecture"
    assert parsed["max_files"] == 25
    assert parsed["depth"] == "high"
    assert parsed["include_tree"] is False


def test_parse_chat_tool_request_repo_mode_payload() -> None:
    parsed = APIInterface._parse_chat_tool_request(
        last_user="unused",
        body={
            "jarvis_mode": "repo_understand",
            "workspace_path": "/tmp/repo",
            "question": "how does auth flow work",
            "max_files": 40,
            "include_tree": True,
        },
    )
    assert parsed is not None
    assert parsed["type"] == "repo_understand"
    assert parsed["workspace_path"] == "/tmp/repo"
    assert parsed["question"] == "how does auth flow work"


def test_parse_chat_tool_request_workflow_command() -> None:
    parsed = APIInterface._parse_chat_tool_request(
        last_user="/workflow --workspace /tmp/proj --max-workers 4 --dry-run --no-checks split auth, api, ui refactor",
        body={},
    )
    assert parsed is not None
    assert parsed["type"] == "code_workflow"
    assert parsed["workspace_path"] == "/tmp/proj"
    assert parsed["max_workers"] == 4
    assert parsed["dry_run"] is True
    assert parsed["run_checks"] is False


def test_parse_chat_tool_request_uses_workspace_hint_when_flag_missing() -> None:
    parsed = APIInterface._parse_chat_tool_request(
        last_user="/repo explain architecture",
        body={},
        workspace_hint="/tmp/hinted-repo",
    )
    assert parsed is not None
    assert parsed["type"] == "repo_understand"
    assert parsed["workspace_path"] == "/tmp/hinted-repo"


def test_parse_chat_tool_request_reports_missing_workspace_for_slash_command() -> None:
    parsed = APIInterface._parse_chat_tool_request(
        last_user="/repo explain architecture",
        body={},
        workspace_hint="",
    )
    assert parsed is not None
    assert parsed["type"] == "command_error"
    assert "workspace path" in str(parsed.get("error", "")).lower()


def test_parse_chat_tool_request_reports_invalid_quote_syntax() -> None:
    parsed = APIInterface._parse_chat_tool_request(
        last_user='/code --workspace /tmp/proj "unterminated',
        body={},
        workspace_hint="",
    )
    assert parsed is not None
    assert parsed["type"] == "command_error"
    assert "couldn't parse" in str(parsed.get("error", "")).lower()
