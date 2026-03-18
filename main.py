"""
JARVIS AI OS CLI entrypoint.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
from typing import Optional

from interfaces.api_interface import APIInterface
from infrastructure.logger import get_logger

logger = get_logger("jarvis.main")


async def _run_api(host: str, port: int, auth_token: str | None) -> None:
    api = APIInterface(host=host, port=port, auth_token=auth_token)
    await api.start()

    stop_event = asyncio.Event()

    def _shutdown_handler(*_: object) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown_handler)
        except NotImplementedError:
            # Some platforms (e.g., Windows) don't support add_signal_handler.
            pass

    logger.info("JARVIS API is running. Press Ctrl+C to stop.")
    await stop_event.wait()
    await api.stop()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JARVIS AI OS CLI")
    parser.add_argument(
        "command",
        nargs="?",
        default="api",
        choices=["api"],
        help="Command to run (default: api).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API bind host.")
    parser.add_argument("--port", type=int, default=8080, help="API bind port.")
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Optional bearer token for API auth.",
    )
    return parser


def cli_entry(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "api":
        try:
            asyncio.run(_run_api(args.host, args.port, args.auth_token))
        except KeyboardInterrupt:
            pass
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(cli_entry())

