from __future__ import annotations

import pytest

from infrastructure.approval import ApprovalManager
from skills.system_skills import RunCommandSkill


pytestmark = pytest.mark.asyncio


async def test_run_command_allows_safe_command_without_approval() -> None:
    skill = RunCommandSkill()
    result = await skill.safe_execute({"command": "echo", "args": ["hello"]})
    assert result.success is True


async def test_run_command_requires_approval_for_sensitive_command() -> None:
    ApprovalManager.reset_instance()
    skill = RunCommandSkill()
    result = await skill.safe_execute({"command": "ps"})
    assert result.success is False
    assert "invalid or not approved" in (result.error or "")


async def test_run_command_sensitive_command_with_approval_succeeds() -> None:
    ApprovalManager.reset_instance()
    approval_manager = ApprovalManager.get_instance()
    approval = approval_manager.create_request(
        action="run_command:ps",
        requested_by="tester",
        reason="health diagnostics",
    )
    approval_manager.approve(approval.approval_id, approver="lead")

    skill = RunCommandSkill()
    result = await skill.safe_execute(
        {
            "command": "ps",
            "approval_token": approval.approval_token,
            "justification": "health diagnostics",
        }
    )
    # Execution outcome may vary by platform, but policy validation should pass.
    assert "invalid or not approved" not in (result.error or "")
