#!/usr/bin/env python3
"""
Bootstrap LinkedIn authenticated session for MCP using Playwright storage state.

Usage:
  python3 scripts/bootstrap_linkedin_playwright_auth.py \
    --state-path data/linkedin_storage_state.json
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="One-time LinkedIn login bootstrap using Playwright")
    parser.add_argument("--state-path", default="data/linkedin_storage_state.json")
    parser.add_argument("--login-url", default="https://www.linkedin.com/login")
    args = parser.parse_args()

    target = Path(str(args.state_path)).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("Playwright is not installed.")
        print("Install with:")
        print("  pip install playwright")
        print("  python -m playwright install chromium")
        return 2

    print("Opening browser for LinkedIn login...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(str(args.login_url), wait_until="domcontentloaded")
        input(
            "Complete LinkedIn login in the opened browser, then press Enter here "
            "to save session state..."
        )
        context.storage_state(path=str(target))
        browser.close()

    print(f"Saved storage state to: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
