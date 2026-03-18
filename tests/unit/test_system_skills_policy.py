from __future__ import annotations

import pytest

from infrastructure.approval import ApprovalManager
from infrastructure.slo_metrics import get_slo_metrics, reset_slo_metrics
from skills.system_skills import RunCommandSkill


pytestmark = pytest.mark.asyncio


async def test_run_command_allows_safe_command_without_approval() -> None:
    reset_slo_metrics()
    skill = RunCommandSkill()
    result = await skill.safe_execute({"command": "echo", "args": ["hello"]})
    assert result.success is True
    assert result.metadata.get("phase") == "verify"


async def test_run_command_requires_approval_for_sensitive_command() -> None:
    reset_slo_metrics()
    ApprovalManager.reset_instance()
    skill = RunCommandSkill()
    result = await skill.safe_execute({"command": "ps"})
    assert result.success is False
    assert "invalid or not approved" in (result.error or "")
    assert result.metadata.get("phase") == "pre_check"


async def test_run_command_sensitive_command_with_approval_succeeds() -> None:
    reset_slo_metrics()
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


async def test_run_command_verify_phase_enforces_expected_exit_code() -> None:
    reset_slo_metrics()
    skill = RunCommandSkill()
    result = await skill.safe_execute(
        {"command": "echo", "args": ["hello"], "expected_exit_codes": [1]}
    )
    assert result.success is False
    assert result.metadata.get("phase") == "verify"
    metrics = get_slo_metrics().snapshot()
    assert any("run_command_verify_failed_total" in key for key in metrics["counters"])
