"""
Policy + cost awareness decisions before routing/execution.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(slots=True)
class PolicyContext:
    route: str
    task_type: str
    user_id: str
    sla_tier: str = "standard"
    latency_sensitive: bool = False
    max_latency_ms: int | None = None
    budget_usd: float | None = None
    privacy_level: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PolicyDecision:
    allow: bool = True
    reason: str = "default_allow"
    prefer_local: bool | None = None
    max_latency_ms: int | None = None
    budget_usd: float | None = None
    allowed_providers: list[str] = field(default_factory=lambda: ["local", "api"])
    tags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow": self.allow,
            "reason": self.reason,
            "prefer_local": self.prefer_local,
            "max_latency_ms": self.max_latency_ms,
            "budget_usd": self.budget_usd,
            "allowed_providers": list(self.allowed_providers),
            "tags": dict(self.tags),
        }


class PolicyCostEngine:
    def __init__(self, *, enabled: bool = False, ledger_path: str = "data/policy_cost_ledger.json") -> None:
        self.enabled = bool(enabled)
        self.ledger_path = str(ledger_path)
        self._ledger: Dict[str, Dict[str, float]] = self._load_ledger()

    @classmethod
    def from_env(cls) -> "PolicyCostEngine":
        enabled = str(os.getenv("JARVIS_POLICY_COST_ENABLED", "false")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        ledger_path = str(os.getenv("JARVIS_POLICY_COST_LEDGER_PATH", "data/policy_cost_ledger.json")).strip()
        return cls(enabled=enabled, ledger_path=ledger_path)

    def decide(self, ctx: PolicyContext) -> PolicyDecision:
        if not self.enabled:
            return PolicyDecision(
                allow=True,
                reason="policy_disabled",
                prefer_local=None,
                max_latency_ms=ctx.max_latency_ms,
                budget_usd=ctx.budget_usd,
            )

        tier = str(ctx.sla_tier or "standard").strip().lower()
        budget = ctx.budget_usd
        latency = ctx.max_latency_ms
        prefer_local: bool | None = None
        reason = "policy_default"
        allowed = ["local", "api"]

        user_key = str(ctx.user_id or "anonymous")
        spent = self._ledger.get(user_key, {}).get("spent_usd", 0.0)
        if budget is not None and spent >= float(budget):
            return PolicyDecision(
                allow=False,
                reason="budget_exhausted",
                prefer_local=True,
                max_latency_ms=ctx.max_latency_ms,
                budget_usd=budget,
                allowed_providers=["local"],
                tags={"spent_usd": spent, "budget_usd": budget, "sla_tier": tier},
            )
        if str(ctx.privacy_level).strip().lower() == "high":
            prefer_local = True
            allowed = ["local"]
            reason = "high_privacy_local_only"
        elif budget is not None and float(budget) <= 0.002:
            prefer_local = True
            reason = "low_budget_prefers_local"
        elif tier in {"gold", "premium"} and bool(ctx.latency_sensitive):
            latency = min(int(latency or 1200), 900)
            prefer_local = False
            reason = "premium_latency_prefers_api"
        elif tier in {"bronze", "free"}:
            prefer_local = True
            latency = int(latency or 2500)
            reason = "low_tier_cost_prefers_local"

        return PolicyDecision(
            allow=True,
            reason=reason,
            prefer_local=prefer_local,
            max_latency_ms=latency,
            budget_usd=budget,
            allowed_providers=allowed,
            tags={
                "sla_tier": tier,
                "latency_sensitive": bool(ctx.latency_sensitive),
                "task_type": ctx.task_type,
                "route": ctx.route,
                "spent_usd": spent,
            },
        )

    def record_usage(self, *, user_id: str, cost_usd: float, tokens_total: int = 0) -> Dict[str, float]:
        key = str(user_id or "anonymous")
        entry = self._ledger.get(key, {"spent_usd": 0.0, "tokens_total": 0.0})
        entry["spent_usd"] = float(entry.get("spent_usd", 0.0) or 0.0) + max(0.0, float(cost_usd or 0.0))
        entry["tokens_total"] = float(entry.get("tokens_total", 0.0) or 0.0) + max(0.0, float(tokens_total or 0))
        self._ledger[key] = entry
        self._persist_ledger()
        return {"spent_usd": float(entry["spent_usd"]), "tokens_total": float(entry["tokens_total"])}

    def get_ledger_entry(self, user_id: str) -> Dict[str, float]:
        key = str(user_id or "anonymous")
        e = self._ledger.get(key, {})
        return {"spent_usd": float(e.get("spent_usd", 0.0) or 0.0), "tokens_total": float(e.get("tokens_total", 0.0) or 0.0)}

    def _load_ledger(self) -> Dict[str, Dict[str, float]]:
        file = Path(self.ledger_path)
        if not file.exists():
            return {}
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {}
            out: Dict[str, Dict[str, float]] = {}
            for k, v in payload.items():
                if not isinstance(v, dict):
                    continue
                out[str(k)] = {
                    "spent_usd": float(v.get("spent_usd", 0.0) or 0.0),
                    "tokens_total": float(v.get("tokens_total", 0.0) or 0.0),
                }
            return out
        except Exception:
            return {}

    def _persist_ledger(self) -> None:
        file = Path(self.ledger_path)
        try:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(self._ledger, indent=2), encoding="utf-8")
        except Exception:
            return
