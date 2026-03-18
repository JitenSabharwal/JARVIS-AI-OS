from core.config import get_config, reset_config


def test_delivery_config_defaults() -> None:
    reset_config()
    cfg = get_config(reload=True)
    assert cfg.delivery.command_execution_enabled is True
    assert cfg.delivery.command_timeout_seconds >= 1.0
    assert "aws" in cfg.delivery.allowed_deploy_targets


def test_delivery_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_DELIVERY_COMMAND_EXECUTION_ENABLED", "false")
    monkeypatch.setenv("JARVIS_DELIVERY_COMMAND_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("JARVIS_DELIVERY_ALLOWED_DEPLOY_TARGETS", "local,aws")
    monkeypatch.setenv("JARVIS_DELIVERY_AWS_DEPLOY_COMMAND", "python3 -c \"raise SystemExit(0)\"")
    reset_config()
    cfg = get_config(reload=True)
    assert cfg.delivery.command_execution_enabled is False
    assert cfg.delivery.command_timeout_seconds == 45.0
    assert cfg.delivery.allowed_deploy_targets == ["local", "aws"]
    assert "SystemExit(0)" in cfg.delivery.aws_deploy_command
