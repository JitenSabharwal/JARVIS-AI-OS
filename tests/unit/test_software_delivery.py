from infrastructure.software_delivery import SoftwareDeliveryEngine


def test_bootstrap_templates_and_profiles() -> None:
    engine = SoftwareDeliveryEngine()
    templates = engine.list_templates()
    profiles = engine.list_profiles()
    assert len(templates) >= 3
    assert any(t["template_id"] == "backend_fastapi" for t in templates)
    assert any(p["name"] == "prod" for p in profiles)

    bootstrap = engine.bootstrap_project(
        template_id="backend_fastapi",
        project_name="demo_service",
        cloud_target="aws",
        include_ci=True,
    )
    assert bootstrap["project_name"] == "demo_service"
    assert bootstrap["file_count"] >= 2
    paths = {f["path"] for f in bootstrap["files"]}
    assert ".github/workflows/ci.yml" in paths


def test_pipeline_requires_mandatory_gates() -> None:
    engine = SoftwareDeliveryEngine()

    failed = engine.run_pipeline(
        project_name="demo_service",
        gate_inputs={"lint": True, "test": True},
    )
    assert failed["all_passed"] is False
    assert "sast" in failed["failed_gates"]
    assert "dependency_audit" in failed["failed_gates"]

    passed = engine.run_pipeline(
        project_name="demo_service",
        gate_inputs={
            "lint": True,
            "test": {"passed": True, "duration_ms": 1200},
            "sast": True,
            "dependency_audit": True,
        },
    )
    assert passed["all_passed"] is True
    assert passed["failed_gates"] == []


def test_release_profile_controls_and_auto_rollback() -> None:
    engine = SoftwareDeliveryEngine()
    pipeline_ok = engine.run_pipeline(
        project_name="demo_service",
        gate_inputs={
            "lint": True,
            "test": True,
            "sast": True,
            "dependency_audit": True,
        },
    )

    waiting = engine.create_release(
        project_name="demo_service",
        profile="prod",
        pipeline_result=pipeline_ok,
        approved=False,
    )
    assert waiting["status"] == "waiting_approval"

    deployed = engine.create_release(
        project_name="demo_service",
        profile="prod",
        pipeline_result=pipeline_ok,
        approved=True,
        post_deploy={"error_rate_pct": 0.3, "p95_latency_ms": 900.0, "availability_pct": 99.9},
    )
    assert deployed["status"] == "deployed"

    rolled_back = engine.create_release(
        project_name="demo_service",
        profile="prod",
        pipeline_result=pipeline_ok,
        approved=True,
        post_deploy={"error_rate_pct": 7.0, "p95_latency_ms": 1900.0, "availability_pct": 98.0},
    )
    assert rolled_back["status"] == "rolled_back"
    assert "error_rate" in (rolled_back.get("rollback_reason") or "")


def test_release_lead_time_summary() -> None:
    engine = SoftwareDeliveryEngine()
    pipeline_ok = engine.run_pipeline(
        project_name="demo_service",
        gate_inputs={
            "lint": True,
            "test": True,
            "sast": True,
            "dependency_audit": True,
        },
    )
    engine.create_release(
        project_name="demo_service",
        profile="dev",
        pipeline_result=pipeline_ok,
        approved=True,
        build_started_at=1.0,
    )
    engine.create_release(
        project_name="demo_service",
        profile="dev",
        pipeline_result=pipeline_ok,
        approved=True,
        build_started_at=2.0,
    )
    metrics = engine.get_lead_time_summary()
    assert metrics["release_count"] >= 2
    assert metrics["average_lead_time_seconds"] >= 0.0


def test_runner_pipeline_and_deploy_success_path() -> None:
    engine = SoftwareDeliveryEngine()
    result = engine.run_release_pipeline(
        project_name="demo_service",
        profile="prod",
        deploy_target="aws",
        approved=True,
        context={
            "gates": {
                "lint": True,
                "test": True,
                "sast": True,
                "dependency_audit": True,
            },
            "deploy": {"success": True},
        },
        post_deploy={"error_rate_pct": 0.1, "p95_latency_ms": 700.0, "availability_pct": 99.95},
    )
    assert result["pipeline"]["all_passed"] is True
    assert result["release"]["status"] == "deployed"
    assert result["deploy"]["success"] is True


def test_runner_pipeline_deploy_failure_triggers_rollback() -> None:
    engine = SoftwareDeliveryEngine()
    result = engine.run_release_pipeline(
        project_name="demo_service",
        profile="prod",
        deploy_target="aws",
        approved=True,
        context={
            "gates": {
                "lint": True,
                "test": True,
                "sast": True,
                "dependency_audit": True,
            },
            "deploy": {"success": False},
        },
    )
    assert result["pipeline"]["all_passed"] is True
    assert result["release"]["status"] == "rolled_back"
    assert result["release"]["rollback_reason"] == "deploy_failed"


def test_command_backed_gate_runner() -> None:
    engine = SoftwareDeliveryEngine()
    result = engine.run_pipeline_with_runners(
        project_name="demo_service",
        context={
            "gate_commands": {
                "lint": ["python3", "-c", "raise SystemExit(0)"],
                "test": ["python3", "-c", "raise SystemExit(0)"],
                "sast": ["python3", "-c", "raise SystemExit(0)"],
                "dependency_audit": ["python3", "-c", "raise SystemExit(0)"],
            }
        },
    )
    assert result["all_passed"] is True
    assert result["gate_results"]["lint"]["details"]["runner"] == "subprocess"


def test_command_backed_deploy_adapter_failure() -> None:
    engine = SoftwareDeliveryEngine()
    result = engine.run_release_pipeline(
        project_name="demo_service",
        profile="prod",
        deploy_target="aws",
        approved=True,
        context={
            "gates": {
                "lint": True,
                "test": True,
                "sast": True,
                "dependency_audit": True,
            },
            "deploy_commands": {
                "aws": ["python3", "-c", "raise SystemExit(9)"],
            },
        },
    )
    assert result["deploy"]["success"] is False
    assert result["release"]["status"] == "rolled_back"
