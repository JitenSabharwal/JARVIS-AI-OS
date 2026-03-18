from __future__ import annotations

import pytest

from infrastructure.approval import ApprovalManager
from skills.base_skill import BaseSkill, SkillResult
from skills.tools_registry import ToolsRegistry


class _ContractSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "contract_skill"

    @property
    def description(self) -> str:
        return "skill for contract tests"

    @property
    def category(self) -> str:
        return "custom"

    @property
    def version(self) -> str:
        return "1.0.0"

    async def execute(self, params: dict) -> SkillResult:
        return SkillResult.ok({"ok": True, "params": params})

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
                "required": ["message"],
            },
        }

    def validate_params(self, params: dict) -> None:
        return None


def test_approval_manager_token_validation() -> None:
    ApprovalManager.reset_instance()
    manager = ApprovalManager.get_instance()

    req = manager.create_request(
        action="run_command:ps",
        requested_by="tester",
        reason="debug",
    )
    assert manager.validate_token(req.approval_token, expected_action="run_command:ps") is False

    manager.approve(req.approval_id, approver="security")
    assert manager.validate_token(req.approval_token, expected_action="run_command:ps") is True
    assert manager.validate_token(req.approval_token, expected_action="run_command:ifconfig") is False


@pytest.mark.asyncio
async def test_tools_registry_rejects_unknown_parameters() -> None:
    ToolsRegistry.reset()
    registry = ToolsRegistry.get_instance()
    registry.register_skill(_ContractSkill(), overwrite=True)

    result = await registry.execute_skill(
        "contract_skill",
        {"message": "hello", "extra": "not-allowed"},
    )
    assert result.success is False
    assert "unknown parameter" in (result.error or "")

