"""
JARVIS AI OS — Application entry point.

Run with::

    python jarvis_main.py [--mode cli|api|voice] [--debug]

or, after ``pip install -e .``::

    jarvis [--mode cli|api|voice] [--debug]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER = r"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
  AI Operating System  —  v1.0.0
"""


# ---------------------------------------------------------------------------
# Shutdown helpers
# ---------------------------------------------------------------------------

_stop_event: asyncio.Event | None = None


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Register SIGINT / SIGTERM handlers for graceful shutdown."""
    global _stop_event
    _stop_event = asyncio.Event()

    def _signal_cb() -> None:
        print("\nShutdown signal received …")
        if _stop_event:
            _stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_cb)
        except (NotImplementedError, RuntimeError):
            # Windows / non-main-thread fallback
            signal.signal(sig, lambda *_: _signal_cb())


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------

async def main(mode: str = "cli", debug: bool = False) -> None:
    """Bootstrap and run JARVIS AI OS.

    Args:
        mode: Execution mode — ``"cli"``, ``"api"``, or ``"voice"``.
        debug: Enable debug-level logging when ``True``.
    """
    from infrastructure.logger import setup_logger

    log_level = logging.DEBUG if debug else logging.INFO
    setup_logger(level=log_level)
    logger = logging.getLogger("jarvis.main")

    print(_BANNER)
    logger.info("Starting JARVIS AI OS in '%s' mode …", mode)

    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)

    # ── Core infrastructure ───────────────────────────────────────────────
    try:
        from core.config import get_config
        from infrastructure.message_bus import MessageBus
        from infrastructure.monitoring import Monitor

        config = get_config()
        model_router = None
        try:
            from infrastructure.model_provider_factory import build_model_router_from_config

            model_router = build_model_router_from_config(config)
            if model_router:
                logger.info("Hybrid model router enabled from config.")
            else:
                logger.info("Hybrid model router disabled (no active providers).")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialise model router: %s", exc)

        bus = MessageBus()
        monitor = Monitor()
        await bus.start()
        logger.info("Message bus started.")

        # ── Agent registry + orchestrator ─────────────────────────────────
        from core.orchestrator import MasterOrchestrator
        from agents.agent_registry import AgentRegistry
        from agents.specialized_agents import AnalystAgent, DeveloperAgent, ManagerAgent
        from agents.coordinator_agent import CoordinatorAgent
        from infrastructure.langgraph_adapter import LangGraphWorkflowAdapter

        orchestrator = MasterOrchestrator()
        orchestrator.set_langgraph_adapter(
            LangGraphWorkflowAdapter(enabled=bool(config.research.langgraph_enabled))
        )
        registry = AgentRegistry.get_instance()

        for AgentCls in (CoordinatorAgent, AnalystAgent, DeveloperAgent, ManagerAgent):
            agent = AgentCls()
            if model_router and hasattr(agent, "set_model_router"):
                agent.set_model_router(model_router)
            await agent.start()
            await orchestrator.register_agent(agent)
            registry.register(agent)
            logger.info("Registered agent: %s", agent.name)

        await orchestrator.start()
        logger.info("Orchestrator started.")

        # ── Skills registry ───────────────────────────────────────────────
        from skills.tools_registry import ToolsRegistry
        skills_registry = ToolsRegistry.get_instance()
        skills_registry.load_builtin_skills()
        logger.info("Skills registry loaded (%d skills).", len(skills_registry.list_active_skills()))

        # ── Memory ────────────────────────────────────────────────────────
        from interfaces.conversation_manager import ConversationManager
        conv_manager = ConversationManager()
        if model_router:
            conv_manager.set_model_router(model_router)

        # ── Mode-specific startup ─────────────────────────────────────────
        api_interface: Any = None
        voice_interface: Any = None

        if mode == "api":
            from interfaces.api_interface import APIInterface
            from infrastructure.automation_actions import register_default_automation_actions
            from infrastructure.builtin_connectors import build_default_connector_registry
            from infrastructure.neo4j_graph_store import Neo4jGraphStore
            from infrastructure.research_adapters import DuckDuckGoAdapter
            from infrastructure.software_delivery import SoftwareDeliveryEngine

            api_interface = APIInterface(
                host=config.api.host,
                port=config.api.port,
                auth_token=config.api.token,
                slo_thresholds={
                    "api_request_p95_ms": config.infrastructure.slo_api_p95_ms,
                    "api_error_rate_pct": config.infrastructure.slo_api_error_rate_pct,
                    "run_command_p95_ms": config.infrastructure.slo_run_command_p95_ms,
                    "connectors_unhealthy_count": config.infrastructure.slo_connectors_unhealthy_count,
                    "automation_dead_letters_backlog": config.infrastructure.slo_automation_dead_letters_backlog,
                    "min_samples": float(config.infrastructure.slo_min_samples),
                },
            )
            api_interface.set_orchestrator(orchestrator)
            api_interface.set_skills_registry(skills_registry)
            api_interface.set_conversation_manager(conv_manager)
            api_interface.set_monitor(monitor)
            api_interface.set_software_delivery_engine(
                SoftwareDeliveryEngine(
                    delivery_config={
                        "command_execution_enabled": config.delivery.command_execution_enabled,
                        "command_timeout_seconds": config.delivery.command_timeout_seconds,
                        "max_output_chars": config.delivery.max_output_chars,
                        "deploy_max_retries": config.delivery.deploy_max_retries,
                        "deploy_retry_backoff_seconds": config.delivery.deploy_retry_backoff_seconds,
                        "allowed_deploy_targets": config.delivery.allowed_deploy_targets,
                        "default_working_dir": config.delivery.default_working_dir,
                        "local_deploy_command": config.delivery.local_deploy_command,
                        "aws_deploy_command": config.delivery.aws_deploy_command,
                        "gcp_deploy_command": config.delivery.gcp_deploy_command,
                        "vercel_deploy_command": config.delivery.vercel_deploy_command,
                    }
                )
            )
            connector_registry = build_default_connector_registry(config.data_dir)
            api_interface.set_connector_registry(connector_registry)
            register_default_automation_actions(api_interface.automation_engine, connector_registry)
            api_interface.research_engine.set_hierarchical_rag_enabled(
                bool(config.research.hierarchical_rag_enabled)
            )
            graph_store = Neo4jGraphStore(
                enabled=bool(config.research.neo4j_enabled),
                uri=str(config.research.neo4j_uri),
                username=str(config.research.neo4j_username),
                password=str(config.research.neo4j_password),
                database=str(config.research.neo4j_database),
            )
            api_interface.research_engine.set_graph_store(graph_store)
            try:
                api_interface.research_engine.register_adapter(DuckDuckGoAdapter())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to register default research adapter: %s", exc)
            await api_interface.start()
            logger.info("API server started on http://%s:%d", config.api.host, config.api.port)

        elif mode == "voice":
            from interfaces.voice_interface import VoiceInterface
            voice_interface = VoiceInterface()
            if not voice_interface.is_available():
                logger.warning("Voice libraries not available; falling back to CLI mode.")
                mode = "cli"
            else:
                async def _voice_cb(payload: dict[str, Any]) -> str:
                    session_id = conv_manager.get_or_create_session("voice_user")
                    return await conv_manager.process_input(
                        session_id,
                        payload.get("text", ""),
                        modality=str(payload.get("modality", "voice")),
                        media=payload.get("media") or {},
                        context=payload.get("context") or {},
                    )

                voice_interface.set_multimodal_callback(_voice_cb)
                voice_interface.start()

        # ── Interactive loop ──────────────────────────────────────────────
        if mode == "cli":
            print("JARVIS is ready. Type your query (or 'exit' to quit).\n")
            while True:
                if _stop_event and _stop_event.is_set():
                    break
                try:
                    query = await asyncio.get_event_loop().run_in_executor(None, input, "You: ")
                except EOFError:
                    break
                if query.strip().lower() in {"exit", "quit", "bye"}:
                    break
                session_id = conv_manager.get_or_create_session("cli_user")
                response = await conv_manager.process_input(session_id, query)
                print(f"JARVIS: {response}\n")

        elif mode == "voice" and voice_interface:
            await voice_interface.listen_and_respond()

        else:
            # API mode — wait for shutdown signal
            assert _stop_event is not None
            await _stop_event.wait()

    except Exception as exc:
        logging.getLogger("jarvis.main").error("Fatal startup error: %s", exc, exc_info=True)
        sys.exit(1)

    finally:
        logger.info("Shutting down JARVIS …")
        if api_interface:
            try:
                api_interface.research_engine.close()
            except Exception:  # noqa: BLE001
                pass
            await api_interface.stop()
        try:
            await orchestrator.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            await bus.stop()
        except Exception:  # noqa: BLE001
            pass
        logger.info("Shutdown complete. Goodbye.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli_entry() -> None:
    """Console-script entry point registered by setup.py / pyproject.toml."""
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="JARVIS AI Operating System",
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "api", "voice"],
        default="cli",
        help="Run mode (default: cli)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()
    asyncio.run(main(mode=args.mode, debug=args.debug))


if __name__ == "__main__":
    cli_entry()
