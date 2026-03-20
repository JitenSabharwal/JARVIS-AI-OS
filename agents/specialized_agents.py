"""
Domain-specific specialized agents for JARVIS AI OS.

Provides three ready-to-use agent implementations:

- :class:`AnalystAgent` – data analysis, research, and report generation
- :class:`DeveloperAgent` – code generation, review, and debugging
- :class:`ManagerAgent` – project planning, task breakdown, and resource allocation

Each agent inherits from :class:`~agents.base_agent.ConcreteAgent` and
implements concrete ``handle_<capability>`` methods with realistic logic that
integrates with the rest of the system.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base_agent import ConcreteAgent
from core.agent_framework import AgentCapability
from infrastructure.logger import get_logger
from infrastructure.model_router import PrivacyLevel
from utils.exceptions import AgentError, SkillExecutionError
from utils.helpers import generate_id, timestamp_now


# ---------------------------------------------------------------------------
# AnalystAgent
# ---------------------------------------------------------------------------


class AnalystAgent(ConcreteAgent):
    """Agent specialising in data analysis, research, and report generation.

    Capabilities:
    - ``analyze_data`` – summarise and extract insights from structured data
    - ``research_topic`` – gather and synthesise information about a subject
    - ``generate_report`` – compose a structured report from provided data

    Args:
        name: Agent name (default ``"analyst"``).
        agent_id: Optional pre-assigned ID.
        config_overrides: Optional config value overrides.
    """

    _ANALYST_CAPABILITIES: List[AgentCapability] = [
        AgentCapability(
            name="analyze_data",
            description=(
                "Perform statistical or qualitative analysis on a dataset and "
                "return key insights, trends, and anomalies."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "data": {
                        "description": "List of records or a dict of named series.",
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["statistical", "qualitative", "trend", "anomaly"],
                        "default": "statistical",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Subset of fields to focus on.",
                    },
                },
                "required": ["data"],
            },
        ),
        AgentCapability(
            name="research_topic",
            description=(
                "Conduct research on a given topic and return a structured "
                "summary with key findings and references."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "depth": {
                        "type": "string",
                        "enum": ["brief", "standard", "deep"],
                        "default": "standard",
                    },
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["topic"],
            },
        ),
        AgentCapability(
            name="generate_report",
            description=(
                "Compose a structured report from raw data or analysis results, "
                "with configurable sections and formatting."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "data": {},
                    "sections": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "plain", "json"],
                        "default": "markdown",
                    },
                },
                "required": ["title", "data"],
            },
        ),
    ]

    def __init__(
        self,
        name: str = "analyst",
        agent_id: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, agent_id=agent_id, config_overrides=config_overrides)
        self._logger = get_logger(f"agent.{name}")

    def get_capabilities(self) -> List[AgentCapability]:
        return super().get_capabilities() + list(self._ANALYST_CAPABILITIES)

    # ------------------------------------------------------------------
    # Capability handlers
    # ------------------------------------------------------------------

    async def handle_analyze_data(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyse a dataset and return structured insights.

        Supports ``statistical``, ``qualitative``, ``trend``, and ``anomaly``
        analysis modes.

        Args:
            parameters: Must contain ``data``.  Optional ``analysis_type``,
                ``fields``.

        Returns:
            Dict with ``analysis_type``, ``insights``, ``summary``, and metadata.
        """
        data = parameters["data"]
        analysis_type: str = parameters.get("analysis_type", "statistical")
        focus_fields: List[str] = parameters.get("fields", [])

        self._logger.info(
            "AnalystAgent '%s' running %s analysis", self.name, analysis_type
        )

        # Normalise data to a list of records for uniform processing
        records: List[Any] = data if isinstance(data, list) else (
            [data] if isinstance(data, dict) else list(data)
        )
        record_count = len(records)

        insights: List[str] = []
        metrics: Dict[str, Any] = {}

        if analysis_type == "statistical":
            numeric_fields = self._extract_numeric_fields(records, focus_fields)
            for field_name, values in numeric_fields.items():
                if values:
                    field_min = min(values)
                    field_max = max(values)
                    field_avg = sum(values) / len(values)
                    field_range = field_max - field_min
                    metrics[field_name] = {
                        "min": field_min,
                        "max": field_max,
                        "avg": round(field_avg, 4),
                        "range": field_range,
                        "count": len(values),
                    }
                    insights.append(
                        f"Field '{field_name}': avg={field_avg:.2f}, "
                        f"range=[{field_min}, {field_max}]"
                    )

        elif analysis_type == "trend":
            if record_count >= 2:
                insights.append(
                    f"Dataset contains {record_count} records; "
                    "trend analysis requires time-series context."
                )
            else:
                insights.append("Insufficient data for trend analysis (need ≥ 2 records).")

        elif analysis_type == "anomaly":
            numeric_fields = self._extract_numeric_fields(records, focus_fields)
            for field_name, values in numeric_fields.items():
                if len(values) >= 3:
                    avg = sum(values) / len(values)
                    variance = sum((v - avg) ** 2 for v in values) / len(values)
                    std = variance ** 0.5
                    anomalies = [v for v in values if abs(v - avg) > 2 * std]
                    if anomalies:
                        noun = "anomaly" if len(anomalies) == 1 else "anomalies"
                        insights.append(
                            f"Field '{field_name}': {len(anomalies)} potential "
                            f"{noun} detected (>2σ from mean={avg:.2f})."
                        )
                    else:
                        insights.append(
                            f"Field '{field_name}': no anomalies detected."
                        )

        elif analysis_type == "qualitative":
            text_fields = self._extract_text_fields(records, focus_fields)
            for field_name, texts in text_fields.items():
                unique_values = set(texts)
                insights.append(
                    f"Field '{field_name}': {len(unique_values)} distinct value(s) "
                    f"across {len(texts)} record(s)."
                )

        if not insights:
            insights.append(f"Processed {record_count} record(s). No specific insights derived.")

        return {
            "analysis_type": analysis_type,
            "record_count": record_count,
            "focus_fields": focus_fields,
            "insights": insights,
            "metrics": metrics,
            "summary": f"{analysis_type.capitalize()} analysis of {record_count} records completed.",
            "analyst_id": self.agent_id,
            "timestamp": timestamp_now(),
        }

    async def handle_research_topic(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Research a topic and return a structured summary.

        Args:
            parameters: Must contain ``topic`` (str).  Optional ``depth``
                (``"brief"`` | ``"standard"`` | ``"deep"``) and
                ``focus_areas`` (list of str).

        Returns:
            Dict with ``topic``, ``summary``, ``key_findings``, ``depth``.
        """
        topic: str = parameters["topic"]
        depth: str = parameters.get("depth", "standard")
        focus_areas: List[str] = parameters.get("focus_areas", [])

        self._logger.info(
            "AnalystAgent '%s' researching topic='%s' depth=%s",
            self.name,
            topic,
            depth,
        )

        # Simulate depth-based research with proportional output
        finding_count = {"brief": 3, "standard": 6, "deep": 10}.get(depth, 6)

        key_findings = [
            f"Finding {i + 1}: Analysis of '{topic}' reveals aspect {i + 1}."
            for i in range(finding_count)
        ]
        if focus_areas:
            for area in focus_areas[:3]:
                key_findings.append(
                    f"Focus area '{area}': Relevant considerations identified for '{topic}'."
                )

        summary = (
            f"Research on '{topic}' at depth '{depth}' yielded {len(key_findings)} "
            f"key findings. "
            + (
                f"Focus areas examined: {', '.join(focus_areas)}."
                if focus_areas
                else "No specific focus areas requested."
            )
        )

        return {
            "topic": topic,
            "depth": depth,
            "focus_areas": focus_areas,
            "summary": summary,
            "key_findings": key_findings,
            "finding_count": len(key_findings),
            "analyst_id": self.agent_id,
            "timestamp": timestamp_now(),
        }

    async def handle_generate_report(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compose a structured report from data or analysis results.

        Args:
            parameters: Must contain ``title`` (str) and ``data`` (any).
                Optional ``sections`` (list) and ``format``
                (``"markdown"`` | ``"plain"`` | ``"json"``).

        Returns:
            Dict with ``title``, ``content``, ``format``, ``section_count``.
        """
        title: str = parameters["title"]
        data: Any = parameters["data"]
        sections: List[str] = parameters.get("sections", ["Summary", "Analysis", "Conclusions"])
        fmt: str = parameters.get("format", "markdown")

        self._logger.info(
            "AnalystAgent '%s' generating '%s' report (format=%s)",
            self.name,
            title,
            fmt,
        )

        section_bodies: Dict[str, str] = {}
        for section in sections:
            if section == "Summary":
                body = f"This report covers: {title}. Data provided contains {self._data_size(data)} item(s)."
            elif section == "Analysis":
                body = f"Detailed analysis of '{title}' data indicates patterns and trends relevant to the subject matter."
            elif section == "Conclusions":
                body = f"Based on the analysis, key conclusions for '{title}' have been drawn and are presented above."
            elif section == "Recommendations":
                body = f"The following recommendations are proposed based on findings related to '{title}'."
            else:
                body = f"Content for section '{section}' of report '{title}'."
            section_bodies[section] = body

        if fmt == "markdown":
            content_parts = [f"# {title}\n", f"*Generated: {timestamp_now()}*\n"]
            for section, body in section_bodies.items():
                content_parts.append(f"\n## {section}\n\n{body}\n")
            content = "\n".join(content_parts)
        elif fmt == "plain":
            content_parts = [f"{title.upper()}", "=" * len(title), ""]
            for section, body in section_bodies.items():
                content_parts.extend([f"{section}:", f"  {body}", ""])
            content = "\n".join(content_parts)
        else:  # json
            content = str({
                "title": title,
                "sections": section_bodies,
                "generated_at": timestamp_now(),
            })

        return {
            "title": title,
            "content": content,
            "format": fmt,
            "section_count": len(sections),
            "sections": list(section_bodies.keys()),
            "analyst_id": self.agent_id,
            "timestamp": timestamp_now(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_numeric_fields(
        records: List[Any], focus: List[str]
    ) -> Dict[str, List[float]]:
        """Extract numeric field values from a list of dict records."""
        result: Dict[str, List[float]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            items = (
                {k: v for k, v in record.items() if k in focus}.items()
                if focus
                else record.items()
            )
            for key, value in items:
                if isinstance(value, (int, float)):
                    result.setdefault(key, []).append(float(value))
        return result

    @staticmethod
    def _extract_text_fields(
        records: List[Any], focus: List[str]
    ) -> Dict[str, List[str]]:
        """Extract text field values from a list of dict records."""
        result: Dict[str, List[str]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            items = (
                {k: v for k, v in record.items() if k in focus}.items()
                if focus
                else record.items()
            )
            for key, value in items:
                if isinstance(value, str):
                    result.setdefault(key, []).append(value)
        return result

    @staticmethod
    def _data_size(data: Any) -> int:
        if isinstance(data, (list, tuple)):
            return len(data)
        if isinstance(data, dict):
            return len(data)
        return 1


# ---------------------------------------------------------------------------
# DeveloperAgent
# ---------------------------------------------------------------------------


class DeveloperAgent(ConcreteAgent):
    """Agent specialising in software engineering tasks.

    Capabilities:
    - ``generate_code`` – produce code from a natural-language specification
    - ``review_code`` – analyse code for quality, style, and correctness
    - ``debug_code`` – identify bugs and propose fixes in provided code

    Args:
        name: Agent name (default ``"developer"``).
        agent_id: Optional pre-assigned ID.
        config_overrides: Optional config value overrides.
    """

    _DEVELOPER_CAPABILITIES: List[AgentCapability] = [
        AgentCapability(
            name="generate_code",
            description=(
                "Generate source code in the specified language from a "
                "natural-language specification."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "specification": {"type": "string"},
                    "language": {"type": "string", "default": "python"},
                    "style_guide": {"type": "string"},
                    "include_tests": {"type": "boolean", "default": False},
                },
                "required": ["specification"],
            },
        ),
        AgentCapability(
            name="review_code",
            description=(
                "Review code for correctness, style, security, and performance, "
                "returning structured feedback."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "language": {"type": "string"},
                    "focus": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "e.g. ['security', 'performance', 'readability']",
                    },
                },
                "required": ["code"],
            },
        ),
        AgentCapability(
            name="debug_code",
            description=(
                "Analyse code for bugs, explain root causes, and propose fixes."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "error_message": {"type": "string"},
                    "language": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["code"],
            },
        ),
        AgentCapability(
            name="update_codebase",
            description=(
                "Apply targeted code updates to a project workspace from a natural-language instruction. "
                "Supports React/Next.js/Node.js/Python repositories."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "workspace_path": {"type": "string"},
                    "instruction": {"type": "string"},
                    "dry_run": {"type": "boolean", "default": False},
                    "run_checks": {"type": "boolean", "default": True},
                    "max_files": {"type": "integer", "default": 10},
                },
                "required": ["workspace_path", "instruction"],
            },
        ),
        AgentCapability(
            name="understand_codebase",
            description=(
                "Analyze a repository and return architecture, modules, entry points, and practical guidance."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "workspace_path": {"type": "string"},
                    "question": {"type": "string"},
                    "max_files": {"type": "integer", "default": 30},
                    "include_tree": {"type": "boolean", "default": True},
                    "depth": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "default": "medium",
                    },
                },
                "required": ["workspace_path"],
            },
        ),
    ]

    def __init__(
        self,
        name: str = "developer",
        agent_id: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, agent_id=agent_id, config_overrides=config_overrides)
        self._logger = get_logger(f"agent.{name}")

    def get_capabilities(self) -> List[AgentCapability]:
        return super().get_capabilities() + list(self._DEVELOPER_CAPABILITIES)

    # ------------------------------------------------------------------
    # Capability handlers
    # ------------------------------------------------------------------

    async def handle_generate_code(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate source code from a specification.

        Args:
            parameters: Must contain ``specification`` (str).  Optional
                ``language`` (default ``"python"``), ``style_guide``,
                ``include_tests`` (bool).

        Returns:
            Dict with ``code``, ``language``, ``lines``, ``test_code``.
        """
        specification: str = parameters["specification"]
        language: str = parameters.get("language", "python").lower()
        style_guide: str = parameters.get("style_guide", "")
        include_tests: bool = parameters.get("include_tests", False)

        self._logger.info(
            "DeveloperAgent '%s' generating %s code for: %s",
            self.name,
            language,
            specification[:80],
        )

        # Prefer routed model generation when configured; fallback to deterministic skeleton.
        func_name = self._spec_to_identifier(specification)
        prompt = (
            f"Generate {language} code for: {specification}\n"
            f"Style guide: {style_guide or 'default'}\n"
            "Return code only."
        )
        routed_code = await self._route_text_generation(
            prompt=prompt,
            task_type="coding",
            privacy_level=PrivacyLevel.MEDIUM,
        )
        code = routed_code or self._build_code_skeleton(func_name, specification, language, style_guide)
        test_code: Optional[str] = None
        if include_tests:
            test_code = self._build_test_skeleton(func_name, language)

        lines = code.count("\n") + 1
        return {
            "code": code,
            "test_code": test_code,
            "language": language,
            "function_name": func_name,
            "lines": lines,
            "specification": specification,
            "developer_id": self.agent_id,
            "timestamp": timestamp_now(),
        }

    async def handle_review_code(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review code and return structured feedback.

        Args:
            parameters: Must contain ``code`` (str).  Optional ``language``,
                ``focus`` (list of focus areas).

        Returns:
            Dict with ``issues``, ``suggestions``, ``score``, ``summary``.
        """
        code: str = parameters["code"]
        language: str = parameters.get("language", "python")
        focus: List[str] = parameters.get("focus", ["correctness", "readability", "security"])

        self._logger.info(
            "DeveloperAgent '%s' reviewing %s code (%d chars, focus=%s)",
            self.name,
            language,
            len(code),
            focus,
        )

        issues: List[Dict[str, Any]] = []
        suggestions: List[str] = []

        # Heuristic static-analysis checks
        lines = code.splitlines()
        line_count = len(lines)

        # Long lines
        long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines and "readability" in focus:
            issues.append({
                "severity": "minor",
                "category": "readability",
                "message": f"Lines exceeding 120 characters: {long_lines[:5]}",
                "lines": long_lines[:5],
            })

        # TODO / FIXME markers
        todo_lines = [
            i + 1
            for i, line in enumerate(lines)
            if re.search(r"\b(TODO|FIXME|HACK|XXX)\b", line, re.IGNORECASE)
        ]
        if todo_lines:
            issues.append({
                "severity": "minor",
                "category": "completeness",
                "message": f"Unresolved TODO/FIXME markers on lines: {todo_lines}",
                "lines": todo_lines,
            })
            suggestions.append("Resolve or track TODO/FIXME items in your issue tracker.")

        # Bare except
        if language == "python" and "except:" in code:
            issues.append({
                "severity": "major",
                "category": "correctness",
                "message": "Bare 'except:' clause catches all exceptions including SystemExit.",
                "recommendation": "Use 'except Exception:' or a specific exception type.",
            })
            suggestions.append("Replace bare 'except:' with specific exception types.")

        # Hardcoded secrets heuristic
        if "security" in focus:
            secret_patterns = [r'password\s*=\s*["\'][^"\']+["\']', r'api_key\s*=\s*["\'][^"\']+["\']']
            for pattern in secret_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    issues.append({
                        "severity": "critical",
                        "category": "security",
                        "message": "Possible hardcoded credential detected.",
                        "recommendation": "Use environment variables or a secrets manager.",
                    })
                    suggestions.append("Never hardcode secrets; use environment variables.")
                    break

        if not suggestions:
            suggestions.append("Code looks clean. Consider adding docstrings if missing.")

        # Composite score
        severity_weights = {"critical": 30, "major": 15, "minor": 5}
        deductions = sum(
            severity_weights.get(issue["severity"], 5) for issue in issues
        )
        score = max(0, 100 - deductions)

        return {
            "language": language,
            "focus": focus,
            "line_count": line_count,
            "issue_count": len(issues),
            "issues": issues,
            "suggestions": suggestions,
            "score": score,
            "summary": (
                f"Reviewed {line_count} lines of {language} code. "
                f"Found {len(issues)} issue(s). Quality score: {score}/100."
            ),
            "developer_id": self.agent_id,
            "timestamp": timestamp_now(),
        }

    async def handle_debug_code(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify bugs in code and propose fixes.

        Args:
            parameters: Must contain ``code`` (str).  Optional
                ``error_message`` (str), ``language`` (str), ``context`` (str).

        Returns:
            Dict with ``bugs``, ``fixes``, ``root_cause``, ``fixed_code``.
        """
        code: str = parameters["code"]
        error_message: str = parameters.get("error_message", "")
        language: str = parameters.get("language", "python")
        context: str = parameters.get("context", "")

        self._logger.info(
            "DeveloperAgent '%s' debugging %s code (error=%s)",
            self.name,
            language,
            (error_message[:60] + "...") if len(error_message) > 60 else error_message,
        )

        bugs: List[Dict[str, Any]] = []
        fixes: List[str] = []
        root_cause = "Undetermined"
        fixed_code = code

        # Pattern-based bug detection
        if language == "python":
            # Mutable default argument
            if re.search(r"def\s+\w+\(.*=\s*[\[\{]", code):
                bugs.append({
                    "type": "mutable_default_argument",
                    "severity": "major",
                    "description": "Mutable default argument detected. Default values are shared across calls.",
                    "recommendation": "Use 'None' as default and initialise inside the function.",
                })
                fixes.append("Replace mutable defaults (list/dict) with None and initialise in the function body.")
                root_cause = "Mutable default argument"

            # Division without zero-guard
            if re.search(r"\/\s*\w+", code) and "ZeroDivisionError" in error_message:
                bugs.append({
                    "type": "zero_division",
                    "severity": "major",
                    "description": "Division operation without zero-guard detected.",
                    "recommendation": "Add a check: 'if denominator != 0' before dividing.",
                })
                fixes.append("Wrap divisions in zero-checks or use try/except ZeroDivisionError.")
                root_cause = "Division by zero"
                fixed_code = re.sub(
                    r"(\w+)\s*/\s*(\w+)",
                    r"(\1 / \2 if \2 != 0 else None)",
                    fixed_code,
                    count=1,
                )

            # Index out of range
            if "IndexError" in error_message:
                bugs.append({
                    "type": "index_error",
                    "severity": "major",
                    "description": "Possible list index out of range.",
                    "recommendation": "Check list length before accessing by index.",
                })
                fixes.append("Guard index access with 'if index < len(collection):'.")
                root_cause = "Index out of range"

        if not bugs:
            bugs.append({
                "type": "unknown",
                "severity": "unknown",
                "description": (
                    f"No common bug patterns detected. "
                    + (f"Error reported: {error_message}" if error_message else "Manual inspection recommended.")
                ),
            })
            root_cause = "No pattern match; manual review required"

        return {
            "language": language,
            "error_message": error_message,
            "bug_count": len(bugs),
            "bugs": bugs,
            "fixes": fixes,
            "root_cause": root_cause,
            "fixed_code": fixed_code,
            "context_provided": bool(context),
            "developer_id": self.agent_id,
            "timestamp": timestamp_now(),
        }

    async def handle_update_codebase(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply model-planned updates to a local codebase workspace."""
        workspace_path = str(parameters.get("workspace_path", "")).strip()
        instruction = str(parameters.get("instruction", "")).strip()
        dry_run = bool(parameters.get("dry_run", False))
        run_checks = bool(parameters.get("run_checks", True))
        max_files = int(parameters.get("max_files", 10) or 10)

        if not workspace_path:
            raise AgentError("Missing required parameter: workspace_path")
        if not instruction:
            raise AgentError("Missing required parameter: instruction")

        workspace = Path(workspace_path).expanduser().resolve()
        if not workspace.exists() or not workspace.is_dir():
            raise AgentError(f"Workspace does not exist or is not a directory: {workspace}")
        if not self._is_workspace_allowed(workspace):
            raise AgentError(
                f"Workspace path is outside allowed roots: {workspace}. "
                "Set JARVIS_CODE_ALLOWED_ROOTS to permit this path."
            )

        stack = self._detect_project_stack(workspace)
        file_samples = self._collect_file_samples(workspace, stack=stack, max_files=max_files)
        plan = await self._plan_codebase_edits(
            workspace=workspace,
            stack=stack,
            instruction=instruction,
            file_samples=file_samples,
        )
        edits = plan.get("edits", [])
        if not isinstance(edits, list):
            edits = []

        apply_result = self._apply_code_edits(
            workspace=workspace,
            edits=edits,
            dry_run=dry_run,
        )
        checks = []
        if run_checks and not dry_run and apply_result.get("applied_count", 0) > 0:
            checks = await self._run_project_checks(workspace, stack=stack)

        return {
            "workspace_path": str(workspace),
            "stack": stack,
            "instruction": instruction,
            "dry_run": dry_run,
            "plan_summary": str(plan.get("summary", "")).strip(),
            "planned_edit_count": len(edits),
            "applied_count": int(apply_result.get("applied_count", 0)),
            "skipped_count": int(apply_result.get("skipped_count", 0)),
            "files_touched": apply_result.get("files_touched", []),
            "applied_edits": apply_result.get("applied_edits", []),
            "skipped_edits": apply_result.get("skipped_edits", []),
            "checks": checks,
            "developer_id": self.agent_id,
            "timestamp": timestamp_now(),
        }

    async def handle_understand_codebase(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a repository and return a practical architecture summary."""
        workspace_path = str(parameters.get("workspace_path", "")).strip()
        question = str(parameters.get("question", "")).strip() or "Explain this repository."
        depth = self._normalize_analysis_depth(str(parameters.get("depth", "medium") or "medium"))
        max_files = self._resolve_max_files_for_depth(parameters.get("max_files"), depth=depth)
        include_tree = bool(parameters.get("include_tree", True))

        if not workspace_path:
            raise AgentError("Missing required parameter: workspace_path")
        workspace = Path(workspace_path).expanduser().resolve()
        if not workspace.exists() or not workspace.is_dir():
            raise AgentError(f"Workspace does not exist or is not a directory: {workspace}")
        if not self._is_workspace_allowed(workspace):
            raise AgentError(
                f"Workspace path is outside allowed roots: {workspace}. "
                "Set JARVIS_CODE_ALLOWED_ROOTS to permit this path."
            )

        stack = self._detect_project_stack(workspace)
        repo_index = self._build_or_load_repo_index(workspace, stack=stack, depth=depth)
        analysis_plan = self._build_depth_subquestions(question=question, depth=depth)
        file_samples = self._collect_file_samples_with_index(
            workspace=workspace,
            stack=stack,
            max_files=max_files,
            question=question,
            repo_index=repo_index,
            depth=depth,
        )
        repo_tree = self._build_repo_tree(workspace, max_entries=140) if include_tree else ""
        facts = self._extract_repo_facts(
            stack=stack,
            question=question,
            file_samples=file_samples,
            repo_index=repo_index,
        )
        facts["analysis_plan"] = analysis_plan
        if depth == "high":
            analysis = await self._run_high_depth_analysis_pipeline(
                workspace=workspace,
                stack=stack,
                question=question,
                file_samples=file_samples,
                repo_tree=repo_tree,
                facts=facts,
                analysis_plan=analysis_plan,
                max_rounds=3,
            )
        else:
            analysis = await self._analyze_codebase(
                workspace=workspace,
                stack=stack,
                question=question,
                file_samples=file_samples,
                repo_tree=repo_tree,
                depth=depth,
                facts=facts,
            )
        if not isinstance(analysis, dict):
            analysis = {}
        depth_pipeline = analysis.get("_depth_pipeline", {})
        if not isinstance(depth_pipeline, dict):
            depth_pipeline = {}
        inferred = self._infer_repo_overview(
            stack=stack,
            file_samples=file_samples,
            question=question,
        )
        signals = self._collect_repo_signals(
            stack=stack,
            file_samples=file_samples,
            facts=facts,
        )
        consistency = self._run_analysis_consistency_check(
            analysis=analysis,
            file_samples=file_samples,
            repo_index=repo_index,
            facts=facts,
        )
        summary = str(analysis.get("summary", "")).strip()
        if not summary:
            summary = inferred["summary"]
        architecture = str(analysis.get("architecture", "")).strip() or inferred["architecture"]
        entry_points = analysis.get("entry_points", [])
        if not isinstance(entry_points, list) or not entry_points:
            entry_points = inferred["entry_points"]
        key_modules = analysis.get("key_modules", [])
        if not isinstance(key_modules, list) or not key_modules:
            key_modules = inferred["key_modules"]
        important_flows = analysis.get("important_flows", [])
        if not isinstance(important_flows, list) or not important_flows:
            important_flows = inferred["important_flows"]
        risks = analysis.get("risks", [])
        if not isinstance(risks, list) or not risks:
            risks = inferred["risks"]
        next_steps = analysis.get("next_steps", [])
        if not isinstance(next_steps, list) or not next_steps:
            next_steps = inferred["next_steps"]
        evidence = analysis.get("evidence", [])
        if not isinstance(evidence, list) or not evidence:
            evidence = inferred["evidence"]
        evidence, dropped_claims = self._apply_evidence_policy(
            evidence=evidence,
            file_samples=file_samples,
            repo_index=repo_index,
        )
        if dropped_claims:
            risks = list(risks) + ["Some claims were removed due to missing source evidence."]
        confidence, coverage_pct, open_questions = self._estimate_analysis_quality(
            analysis=analysis,
            repo_index=repo_index,
            file_samples=file_samples,
            consistency=consistency,
        )
        if consistency.get("unsupported_claims"):
            open_questions.extend(
                [f"Validate claim: {c}" for c in consistency.get("unsupported_claims", [])[:3]]
            )
        if consistency.get("contradictions"):
            open_questions.extend(
                [f"Resolve contradiction: {c}" for c in consistency.get("contradictions", [])[:3]]
            )
        if depth == "high" and consistency.get("contradictions"):
            summary = (
                "High-depth analysis found contradictory signals in the available code evidence. "
                "Returning a cautious result pending manual validation."
            )
            architecture = ""
            important_flows = []
            confidence = min(confidence, 0.35)
            risks = list(risks) + ["High-depth fail-close triggered due to contradiction detection."]

        return {
            "workspace_path": str(workspace),
            "stack": stack,
            "question": question,
            "depth": depth,
            "analysis_plan": analysis_plan,
            "depth_pipeline": depth_pipeline,
            "summary": summary,
            "architecture": architecture,
            "entry_points": entry_points,
            "key_modules": key_modules,
            "important_flows": important_flows,
            "risks": risks,
            "next_steps": next_steps,
            "evidence": evidence,
            "signals": signals,
            "facts": facts,
            "consistency": consistency,
            "confidence": confidence,
            "coverage_pct": coverage_pct,
            "open_questions": open_questions,
            "sampled_file_count": len(file_samples),
            "sampled_files": [f.get("path", "") for f in file_samples],
            "repo_tree": repo_tree,
            "developer_id": self.agent_id,
            "timestamp": timestamp_now(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spec_to_identifier(specification: str) -> str:
        """Derive a snake_case function name from a specification string."""
        words = re.findall(r"[a-zA-Z]+", specification)[:5]
        return "_".join(w.lower() for w in words) or "generated_function"

    @staticmethod
    def _build_code_skeleton(
        func_name: str, specification: str, language: str, style_guide: str
    ) -> str:
        """Build a representative code skeleton."""
        if language == "python":
            return textwrap.dedent(f"""\
                \"\"\"
                {specification}
                \"\"\"
                from typing import Any


                def {func_name}(*args: Any, **kwargs: Any) -> Any:
                    \"\"\"
                    {specification}

                    Args:
                        *args: Positional arguments.
                        **kwargs: Keyword arguments.

                    Returns:
                        Result of the operation.
                    \"\"\"
                    # TODO: Implement {func_name}
                    raise NotImplementedError("{func_name} is not yet implemented")
            """)
        if language in ("javascript", "typescript"):
            ts_type = ": any" if language == "typescript" else ""
            return textwrap.dedent(f"""\
                /**
                 * {specification}
                 * @returns {{any}} Result of the operation.
                 */
                function {func_name}(...args{ts_type}){ts_type} {{
                    // TODO: Implement {func_name}
                    throw new Error("{func_name} is not yet implemented");
                }}

                module.exports = {{ {func_name} }};
            """)
        # Generic fallback
        return f"# {specification}\n# Language: {language}\n# TODO: implement {func_name}\n"

    @staticmethod
    def _build_test_skeleton(func_name: str, language: str) -> str:
        """Build a basic test skeleton for *func_name*."""
        if language == "python":
            return textwrap.dedent(f"""\
                import pytest
                from . import {func_name}


                def test_{func_name}_basic():
                    \"\"\"Basic smoke test for {func_name}.\"\"\"
                    # Arrange
                    # Act
                    # Assert
                    pass


                def test_{func_name}_edge_cases():
                    \"\"\"Edge case tests for {func_name}.\"\"\"
                    pass
            """)
        return f"// Tests for {func_name}\n// TODO: add test cases\n"

    @staticmethod
    def _is_workspace_allowed(workspace: Path) -> bool:
        raw = os.getenv("JARVIS_CODE_ALLOWED_ROOTS", "").strip()
        roots: list[Path] = []
        if raw:
            for token in raw.split(os.pathsep):
                token = token.strip()
                if token:
                    roots.append(Path(token).expanduser().resolve())
        if not roots:
            roots = [Path.cwd().resolve(), Path("/runtime").resolve()]
        for root in roots:
            try:
                workspace.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _detect_project_stack(workspace: Path) -> str:
        has_py = (workspace / "pyproject.toml").exists() or (workspace / "requirements.txt").exists()
        has_pkg = (workspace / "package.json").exists()
        is_next = (workspace / "next.config.js").exists() or (workspace / "next.config.mjs").exists()
        if is_next:
            return "nextjs"
        if has_pkg and any((workspace / p).exists() for p in ("src/App.jsx", "src/App.tsx", "vite.config.ts", "vite.config.js")):
            return "react"
        if has_pkg:
            return "nodejs"
        if has_py:
            return "python"
        return "generic"

    @staticmethod
    def _normalize_analysis_depth(depth: str) -> str:
        d = str(depth or "medium").strip().lower()
        if d in {"low", "medium", "high"}:
            return d
        return "medium"

    @staticmethod
    def _build_depth_subquestions(*, question: str, depth: str) -> list[str]:
        base = [
            "What are the primary entry points and startup path?",
            "How is the codebase organized into modules and responsibilities?",
            "What are the key runtime/data flows?",
            "What are the main risks or unclear areas from the current evidence?",
        ]
        q = question.lower()
        if "performance" in q:
            base.append("Which modules are likely latency or throughput bottlenecks?")
        if "security" in q or "auth" in q:
            base.append("Where are authentication/authorization boundaries and risks?")
        if "data" in q:
            base.append("How does data move from input to persistence/output?")
        if depth == "high":
            base.extend(
                [
                    "Which modules are most central by import/reference patterns?",
                    "What testing coverage signals exist around critical flows?",
                    "Which claims need validation with broader repository coverage?",
                ]
            )
        elif depth == "low":
            return base[:3]
        return base[:8]

    @staticmethod
    def _resolve_max_files_for_depth(max_files_value: Any, *, depth: str) -> int:
        if max_files_value is not None:
            try:
                return max(8, min(400, int(max_files_value)))
            except Exception:
                pass
        defaults = {"low": 40, "medium": 100, "high": 220}
        return defaults.get(depth, 100)

    @staticmethod
    def _repo_index_cache_path(workspace: Path) -> Path:
        return workspace / ".jarvis" / "repo_index.json"

    def _build_or_load_repo_index(self, workspace: Path, *, stack: str, depth: str) -> dict[str, Any]:
        cache_path = self._repo_index_cache_path(workspace)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        def _fingerprint() -> str:
            sig_parts = [f"{workspace}:{stack}"]
            count = 0
            for p in workspace.rglob("*"):
                if not p.is_file():
                    continue
                if any(part in {".git", "node_modules", ".next", ".venv", "venv", "__pycache__", "dist", "build", "runtime"} for part in p.parts):
                    continue
                try:
                    st = p.stat()
                except Exception:
                    continue
                sig_parts.append(f"{p}:{st.st_size}:{int(st.st_mtime)}")
                count += 1
                if count >= 5000:
                    break
            return str(abs(hash("|".join(sig_parts))))

        fp = _fingerprint()
        try:
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and str(cached.get("fingerprint", "")) == fp:
                    return cached
        except Exception:
            pass

        entries: list[dict[str, Any]] = []
        path_limit = 12000 if depth == "high" else 6000 if depth == "medium" else 2500
        for p in workspace.rglob("*"):
            if not p.is_file():
                continue
            if any(part in {".git", "node_modules", ".next", ".venv", "venv", "__pycache__", "dist", "build", "runtime"} for part in p.parts):
                continue
            try:
                st = p.stat()
            except Exception:
                continue
            rel = str(p.relative_to(workspace)).replace("\\", "/")
            ext = p.suffix.lower()
            entry = {
                "path": rel,
                "ext": ext,
                "size": int(st.st_size),
                "top_dir": rel.split("/", 1)[0] if "/" in rel else ".",
                "is_entrypoint": rel.lower() in {"main.py", "jarvis_main.py", "app.py", "server.py", "index.js", "src/main.ts", "src/index.ts"},
                "is_dependency_file": rel.lower() in {"pyproject.toml", "requirements.txt", "package.json"},
                "is_config": rel.lower().startswith("config/") or rel.lower().endswith(".env"),
            }
            entries.append(entry)
            if len(entries) >= path_limit:
                break

        index = {
            "version": 1,
            "generated_at": timestamp_now(),
            "workspace": str(workspace),
            "stack": stack,
            "depth": depth,
            "fingerprint": fp,
            "total_files": len(entries),
            "entries": entries,
        }
        try:
            cache_path.write_text(json.dumps(index), encoding="utf-8")
        except Exception:
            pass
        return index

    def _collect_file_samples_with_index(
        self,
        *,
        workspace: Path,
        stack: str,
        max_files: int,
        question: str,
        repo_index: dict[str, Any],
        depth: str,
    ) -> list[dict[str, str]]:
        entries = repo_index.get("entries", []) if isinstance(repo_index.get("entries", []), list) else []
        if not entries:
            return self._collect_file_samples(workspace, stack=stack, max_files=max_files)

        q = question.lower()
        keywords = [w for w in re.findall(r"[a-zA-Z_]{3,}", q) if w not in {"what", "this", "that", "with", "from", "about"}]
        focus_dirs = {"interfaces", "agents", "core", "infrastructure", "memory", "config", "scripts"}

        ranked: list[tuple[int, str]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path", ""))
            path_l = path.lower()
            score = 0
            if bool(entry.get("is_entrypoint", False)):
                score += 90
            if bool(entry.get("is_dependency_file", False)):
                score += 70
            if bool(entry.get("is_config", False)):
                score += 40
            top_dir = str(entry.get("top_dir", "")).strip()
            if top_dir in focus_dirs:
                score += 35
            if "architecture" in q or "data flow" in q:
                if top_dir in {"interfaces", "agents", "core", "infrastructure"}:
                    score += 20
            for kw in keywords[:20]:
                if kw in path_l:
                    score += 6
            if path_l.endswith(".py"):
                score += 4
            if path_l.endswith(".md"):
                score += 2
            ranked.append((score, path))

        ranked.sort(key=lambda t: t[0], reverse=True)
        selected_paths: list[str] = []
        seen: set[str] = set()
        for _, path in ranked:
            if path in seen:
                continue
            seen.add(path)
            selected_paths.append(path)
            if len(selected_paths) >= max_files:
                break

        samples: list[dict[str, str]] = []
        for rel in selected_paths:
            file_path = (workspace / rel).resolve()
            try:
                file_path.relative_to(workspace)
            except ValueError:
                continue
            if not file_path.exists() or not file_path.is_file():
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            snippet_len = 3200 if depth == "high" else 2500
            snippet = content if len(content) <= snippet_len else (content[:2200] + "\n...\n" + content[-700:])
            samples.append({"path": rel, "snippet": snippet})
        return samples

    def _extract_repo_facts(
        self,
        *,
        stack: str,
        question: str,
        file_samples: list[dict[str, str]],
        repo_index: dict[str, Any],
    ) -> dict[str, Any]:
        paths = [str(f.get("path", "")).strip() for f in file_samples if str(f.get("path", "")).strip()]
        snippets = [str(f.get("snippet", "")) for f in file_samples]
        route_files: list[str] = []
        import_edges: list[dict[str, str]] = []
        symbols: list[dict[str, str]] = []
        for path, snip in zip(paths, snippets):
            if ".router.add_" in snip or "@app." in snip or "FastAPI(" in snip:
                route_files.append(path)
            for m in re.finditer(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", snip, re.MULTILINE):
                import_edges.append({"file": path, "import": m.group(1)})
                if len(import_edges) >= 180:
                    break
            for m in re.finditer(r"^\s*(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", snip, re.MULTILINE):
                symbols.append({"file": path, "symbol": m.group(1)})
                if len(symbols) >= 220:
                    break
        entries = repo_index.get("entries", []) if isinstance(repo_index.get("entries", []), list) else []
        total_files = int(repo_index.get("total_files", len(entries)) or 0)
        indexed_paths = [
            str(e.get("path", "")).strip()
            for e in entries
            if isinstance(e, dict) and str(e.get("path", "")).strip()
        ]
        import_graph = self._compute_import_graph_insights(import_edges)
        path_traces = self._derive_path_traces(
            stack=stack,
            sampled_paths=paths,
            route_files=route_files,
            import_graph=import_graph,
        )
        evidence_candidates = self._build_evidence_candidates(
            sampled_paths=paths,
            indexed_paths=indexed_paths,
            route_files=route_files,
        )
        return {
            "stack": stack,
            "question": question,
            "sampled_file_count": len(paths),
            "indexed_file_count": total_files,
            "sampled_paths": paths[:80],
            "indexed_paths": indexed_paths[:500],
            "route_files": route_files[:20],
            "import_edges": import_edges[:140],
            "symbols": symbols[:180],
            "import_graph_top_modules": import_graph.get("top_modules", []),
            "path_traces": path_traces[:20],
            "evidence_candidates": evidence_candidates[:30],
        }

    @staticmethod
    def _compute_import_graph_insights(import_edges: list[dict[str, str]]) -> dict[str, Any]:
        out_degree: Counter[str] = Counter()
        in_degree: Counter[str] = Counter()
        for edge in import_edges:
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("file", "")).strip()
            imp = str(edge.get("import", "")).strip()
            if not src or not imp:
                continue
            src_mod = src.split("/", 1)[0] if "/" in src else src
            imp_mod = imp.split(".", 1)[0]
            out_degree[src_mod] += 1
            in_degree[imp_mod] += 1
        mods = set(out_degree.keys()) | set(in_degree.keys())
        ranked: list[dict[str, Any]] = []
        for mod in mods:
            in_d = int(in_degree.get(mod, 0))
            out_d = int(out_degree.get(mod, 0))
            ranked.append(
                {
                    "module": mod,
                    "in_degree": in_d,
                    "out_degree": out_d,
                    "score": float((2 * in_d) + out_d),
                }
            )
        ranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return {"top_modules": ranked[:12]}

    @staticmethod
    def _derive_path_traces(
        *,
        stack: str,
        sampled_paths: list[str],
        route_files: list[str],
        import_graph: dict[str, Any],
    ) -> list[str]:
        entry_candidates = [p for p in sampled_paths if p.lower() in {"main.py", "jarvis_main.py", "app.py", "server.py"}]
        if not entry_candidates:
            entry_candidates = [p for p in sampled_paths if p.lower().endswith("main.py")][:2]
        top_modules = import_graph.get("top_modules", []) if isinstance(import_graph.get("top_modules", []), list) else []
        module_names = [str(m.get("module", "")).strip() for m in top_modules if isinstance(m, dict) and str(m.get("module", "")).strip()]
        traces: list[str] = []
        for entry in entry_candidates[:2]:
            if route_files:
                traces.append(f"{entry} -> {route_files[0]} -> {' -> '.join(module_names[:3])}" if module_names else f"{entry} -> {route_files[0]}")
            else:
                traces.append(f"{entry} -> {' -> '.join(module_names[:3])}" if module_names else entry)
        if stack == "python" and not traces and sampled_paths:
            traces.append(f"{sampled_paths[0]} -> module-layer traversal inferred from imports")
        return traces

    @staticmethod
    def _build_evidence_candidates(
        *,
        sampled_paths: list[str],
        indexed_paths: list[str],
        route_files: list[str],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        if route_files:
            candidates.append(
                {
                    "claim": "Routing/files for request handling were observed.",
                    "sources": route_files[:6],
                }
            )
        dep_sources = [p for p in sampled_paths if p.lower() in {"requirements.txt", "pyproject.toml", "package.json"}]
        if dep_sources:
            candidates.append(
                {
                    "claim": "Dependency manifests were present in sampled files.",
                    "sources": dep_sources[:4],
                }
            )
        cfg_sources = [p for p in indexed_paths if p.lower().startswith("config/")][:6]
        if cfg_sources:
            candidates.append(
                {
                    "claim": "Configuration hierarchy exists and was indexed.",
                    "sources": cfg_sources,
                }
            )
        return candidates

    @staticmethod
    def _run_analysis_consistency_check(
        *,
        analysis: dict[str, Any],
        file_samples: list[dict[str, str]],
        repo_index: dict[str, Any],
        facts: dict[str, Any],
    ) -> dict[str, Any]:
        sampled_paths = {str(f.get("path", "")).strip() for f in file_samples if str(f.get("path", "")).strip()}
        indexed_paths = {
            str(e.get("path", "")).strip()
            for e in (repo_index.get("entries", []) if isinstance(repo_index.get("entries", []), list) else [])
            if isinstance(e, dict) and str(e.get("path", "")).strip()
        }
        evidence = analysis.get("evidence", []) if isinstance(analysis.get("evidence", []), list) else []
        unsupported_claims: list[str] = []
        supported_claims = 0
        for item in evidence[:40]:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            sources = item.get("sources", []) if isinstance(item.get("sources", []), list) else []
            if any((str(src).strip() in sampled_paths) or (str(src).strip() in indexed_paths) for src in sources):
                supported_claims += 1
            elif claim:
                unsupported_claims.append(claim)
        contradictions = DeveloperAgent._detect_analysis_contradictions(analysis=analysis, facts=facts)
        return {
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims[:8],
            "evidence_items": len(evidence),
            "contradictions": contradictions[:8],
        }

    @staticmethod
    def _detect_analysis_contradictions(*, analysis: dict[str, Any], facts: dict[str, Any]) -> list[str]:
        contradictions: list[str] = []
        route_files = facts.get("route_files", []) if isinstance(facts.get("route_files", []), list) else []
        summary = str(analysis.get("summary", "")).lower()
        architecture = str(analysis.get("architecture", "")).lower()
        merged = f"{summary}\n{architecture}"
        if route_files and ("no api" in merged or "no route" in merged):
            contradictions.append("Analysis claims no API/routes but route-like files were detected.")
        entry_points = analysis.get("entry_points", []) if isinstance(analysis.get("entry_points", []), list) else []
        if entry_points and ("no entry" in merged or "missing entry point" in merged):
            contradictions.append("Analysis claims missing entry points but entry points were listed.")
        return contradictions

    @staticmethod
    def _apply_evidence_policy(
        *,
        evidence: list[Any],
        file_samples: list[dict[str, str]],
        repo_index: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        sampled_paths = {str(f.get("path", "")).strip() for f in file_samples if str(f.get("path", "")).strip()}
        indexed_paths = {
            str(e.get("path", "")).strip()
            for e in (repo_index.get("entries", []) if isinstance(repo_index.get("entries", []), list) else [])
            if isinstance(e, dict) and str(e.get("path", "")).strip()
        }
        kept: list[dict[str, Any]] = []
        dropped_claims: list[str] = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            raw_sources = item.get("sources", []) if isinstance(item.get("sources", []), list) else []
            sources = [str(s).strip() for s in raw_sources if str(s).strip()]
            valid_sources = [s for s in sources if (s in sampled_paths) or (s in indexed_paths)]
            if valid_sources:
                kept.append({"claim": claim, "sources": valid_sources})
            elif claim:
                dropped_claims.append(claim)
        return kept, dropped_claims

    @staticmethod
    def _estimate_analysis_quality(
        *,
        analysis: dict[str, Any],
        repo_index: dict[str, Any],
        file_samples: list[dict[str, str]],
        consistency: dict[str, Any],
    ) -> tuple[float, float, list[str]]:
        total_files = int(repo_index.get("total_files", 0) or 0)
        sampled = len(file_samples)
        coverage_pct = (100.0 * sampled / total_files) if total_files > 0 else 0.0
        supported = int(consistency.get("supported_claims", 0) or 0)
        evidence_items = int(consistency.get("evidence_items", 0) or 0)
        evidence_ratio = (supported / evidence_items) if evidence_items > 0 else 0.0
        base = 0.35 + min(0.35, coverage_pct / 200.0)
        confidence = max(0.0, min(0.98, base + (0.28 * evidence_ratio)))
        open_questions: list[str] = []
        if coverage_pct < 15.0:
            open_questions.append("Coverage is low; increase max_files or depth for stronger confidence.")
        if evidence_items == 0:
            open_questions.append("No explicit evidence mapping returned by model output.")
        if consistency.get("unsupported_claims"):
            open_questions.append("Some claims were not supported by sampled sources.")
        if consistency.get("contradictions"):
            open_questions.append("Conflicting statements were detected between analysis claims and extracted facts.")
            confidence = max(0.0, confidence - 0.18)
        architecture = str(analysis.get("architecture", "")).strip()
        if not architecture:
            open_questions.append("Architecture summary was inferred heuristically, not model-derived.")
        return round(confidence, 3), round(coverage_pct, 1), open_questions[:6]

    @staticmethod
    def _collect_file_samples(workspace: Path, *, stack: str, max_files: int) -> list[dict[str, str]]:
        allow_ext = {
            "python": {".py", ".toml", ".md"},
            "react": {".js", ".jsx", ".ts", ".tsx", ".css", ".json", ".md"},
            "nextjs": {".js", ".jsx", ".ts", ".tsx", ".css", ".json", ".md"},
            "nodejs": {".js", ".ts", ".json", ".md"},
            "generic": {".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md"},
        }.get(stack, {".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md"})
        skip_dirs = {".git", "node_modules", ".next", ".venv", "venv", "__pycache__", "dist", "build", "runtime"}
        all_files: list[Path] = []
        for path in workspace.rglob("*"):
            if not path.is_file():
                continue
            if any(part in skip_dirs for part in path.parts):
                continue
            if path.suffix.lower() not in allow_ext:
                continue
            if path.stat().st_size > 160_000:
                continue
            all_files.append(path)
        priority_exact = {
            "readme.md",
            "pyproject.toml",
            "requirements.txt",
            "package.json",
            "dockerfile",
            "docker-compose.yml",
            "main.py",
            "jarvis_main.py",
            "app.py",
            "server.py",
        }
        priority_dirs = ("interfaces/", "agents/", "core/", "infrastructure/", "memory/", "scripts/")
        prioritized: list[Path] = []
        fallback: list[Path] = []
        for path in all_files:
            rel = str(path.relative_to(workspace)).replace("\\", "/")
            rel_l = rel.lower()
            if rel_l in priority_exact or any(rel_l.startswith(d) for d in priority_dirs):
                prioritized.append(path)
            else:
                fallback.append(path)
        files = (prioritized + fallback)[: max(1, max_files)]
        samples: list[dict[str, str]] = []
        for path in files:
            rel = str(path.relative_to(workspace))
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            snippet = content if len(content) <= 2500 else (content[:1800] + "\n...\n" + content[-500:])
            samples.append({"path": rel, "snippet": snippet})
        return samples

    async def _plan_codebase_edits(
        self,
        *,
        workspace: Path,
        stack: str,
        instruction: str,
        file_samples: list[dict[str, str]],
    ) -> dict[str, Any]:
        file_block = "\n\n".join(
            f"FILE: {f.get('path','')}\n{f.get('snippet','')}"
            for f in file_samples
        )
        prompt = (
            "You are a senior software engineer. Plan concrete repository edits.\n"
            "Return JSON only with schema:\n"
            "{"
            "\"summary\":\"...\","
            "\"edits\":["
            "{\"path\":\"relative/path\",\"operation\":\"replace|append|create\",\"find\":\"...\",\"replace\":\"...\",\"content\":\"...\"}"
            "]}\n"
            "Rules:\n"
            "- Use only provided file paths unless operation=create.\n"
            "- Keep edits minimal and precise.\n"
            "- For operation=replace include exact 'find' and 'replace'.\n"
            "- For operation=append include 'content'.\n"
            "- For operation=create include full file 'content'.\n"
            f"Workspace: {workspace}\n"
            f"Project stack: {stack}\n"
            f"Instruction: {instruction}\n"
            f"Candidate files:\n{file_block}\n"
            "JSON:"
        )
        routed = await self._route_text_generation(
            prompt=prompt,
            task_type="coding",
            privacy_level=PrivacyLevel.MEDIUM,
        )
        if not routed:
            return {"summary": "No model route available for planning.", "edits": []}
        payload = self._extract_json_object(routed)
        if not isinstance(payload, dict):
            return {"summary": "Planner returned invalid JSON.", "edits": []}
        if "edits" not in payload or not isinstance(payload.get("edits"), list):
            payload["edits"] = []
        return payload

    async def _analyze_codebase(
        self,
        *,
        workspace: Path,
        stack: str,
        question: str,
        file_samples: list[dict[str, str]],
        repo_tree: str,
        depth: str,
        facts: dict[str, Any],
    ) -> dict[str, Any]:
        file_block = "\n\n".join(
            f"FILE: {f.get('path','')}\n{f.get('snippet','')}"
            for f in file_samples
        )
        facts_block = json.dumps(facts, ensure_ascii=True)
        prompt = (
            "You are a senior software architect analyzing a code repository.\n"
            "Return JSON only with schema:\n"
            "{"
            "\"summary\":\"...\","
            "\"architecture\":\"...\","
            "\"entry_points\":[\"...\"],"
            "\"key_modules\":[\"...\"],"
            "\"important_flows\":[\"...\"],"
            "\"risks\":[\"...\"],"
            "\"next_steps\":[\"...\"],"
            "\"evidence\":[{\"claim\":\"...\",\"sources\":[\"relative/path\"]}]"
            "}\n"
            "Rules:\n"
            "- Every major claim must map to source files in evidence.\n"
            "- Prefer concrete module/file names over generic wording.\n"
            "- If unknown, state uncertainty in risks/open questions style.\n"
            f"Workspace: {workspace}\n"
            f"Project stack: {stack}\n"
            f"Depth mode: {depth}\n"
            f"Question: {question}\n"
            f"Facts JSON: {facts_block}\n"
            f"Repository tree:\n{repo_tree}\n"
            f"File samples:\n{file_block}\n"
            "JSON:"
        )
        routed = await self._route_text_generation(
            prompt=prompt,
            task_type="analysis",
            privacy_level=PrivacyLevel.MEDIUM,
        )
        if not routed:
            return {}
        payload = self._extract_json_object(routed)
        if not isinstance(payload, dict):
            return {}
        if depth in {"medium", "high"}:
            check_prompt = (
                "You are validating an architecture analysis for evidence quality.\n"
                "Return JSON only with schema:\n"
                "{"
                "\"summary\":\"...\","
                "\"architecture\":\"...\","
                "\"entry_points\":[\"...\"],"
                "\"key_modules\":[\"...\"],"
                "\"important_flows\":[\"...\"],"
                "\"risks\":[\"...\"],"
                "\"next_steps\":[\"...\"],"
                "\"evidence\":[{\"claim\":\"...\",\"sources\":[\"relative/path\"]}]"
                "}\n"
                "Strengthen weak or unsupported claims. Keep concise.\n"
                f"Facts JSON: {facts_block}\n"
                f"Candidate JSON: {json.dumps(payload, ensure_ascii=True)}\n"
                "JSON:"
            )
            routed_check = await self._route_text_generation(
                prompt=check_prompt,
                task_type="analysis",
                privacy_level=PrivacyLevel.MEDIUM,
            )
            checked = self._extract_json_object(routed_check) if routed_check else None
            if isinstance(checked, dict):
                return checked
        return payload

    async def _run_high_depth_analysis_pipeline(
        self,
        *,
        workspace: Path,
        stack: str,
        question: str,
        file_samples: list[dict[str, str]],
        repo_tree: str,
        facts: dict[str, Any],
        analysis_plan: list[str],
        max_rounds: int,
    ) -> dict[str, Any]:
        """
        Multi-pass high-depth analysis pipeline with bounded rounds.

        Roles:
        - planner: derive sub-questions (from analysis_plan)
        - analyst: run focused analysis per sub-question
        - reviewer: tighten each partial and remove weak claims
        - verifier: merge + refine with consistency emphasis
        """
        rounds = max(1, min(int(max_rounds or 3), 4))
        planner_steps = [s for s in analysis_plan if str(s).strip()][: rounds + 1]
        merged: dict[str, Any] = {}
        collected_evidence: list[dict[str, Any]] = []
        stage_artifacts: list[dict[str, Any]] = []

        for idx, step in enumerate(planner_steps[:rounds]):
            focused_question = f"{question}\nFocus area: {step}"
            partial = await self._analyze_codebase(
                workspace=workspace,
                stack=stack,
                question=focused_question,
                file_samples=file_samples,
                repo_tree=repo_tree,
                depth="medium",
                facts=facts,
            )
            if not isinstance(partial, dict):
                continue
            review_prompt = (
                "You are a reviewer agent. Tighten this partial repository analysis JSON.\n"
                "Return JSON only with fields: summary, architecture, entry_points, key_modules, important_flows, risks, next_steps, evidence.\n"
                "Remove generic claims, keep evidence-backed specifics.\n"
                f"Facts JSON: {json.dumps(facts, ensure_ascii=True)}\n"
                f"Focus step: {step}\n"
                f"Partial JSON: {json.dumps(partial, ensure_ascii=True)}\n"
                "JSON:"
            )
            reviewed = partial
            routed_review = await self._route_text_generation(
                prompt=review_prompt,
                task_type="analysis",
                privacy_level=PrivacyLevel.MEDIUM,
            )
            reviewed_json = self._extract_json_object(routed_review) if routed_review else None
            if isinstance(reviewed_json, dict):
                reviewed = reviewed_json
            stage_artifacts.append(
                {
                    "round": idx + 1,
                    "focus": step,
                    "analyst_keys": sorted([str(k) for k in partial.keys()][:16]),
                    "reviewer_keys": sorted([str(k) for k in reviewed.keys()][:16]),
                }
            )
            if not merged:
                merged = dict(reviewed)
            else:
                for key in ("summary", "architecture"):
                    if not str(merged.get(key, "")).strip() and str(reviewed.get(key, "")).strip():
                        merged[key] = reviewed.get(key)
                for key in ("entry_points", "key_modules", "important_flows", "risks", "next_steps"):
                    cur = merged.get(key, [])
                    nxt = reviewed.get(key, [])
                    if not isinstance(cur, list):
                        cur = []
                    if not isinstance(nxt, list):
                        nxt = []
                    merged[key] = list(dict.fromkeys([str(x).strip() for x in (cur + nxt) if str(x).strip()]))[:20]
            ev = reviewed.get("evidence", [])
            if isinstance(ev, list):
                for item in ev:
                    if isinstance(item, dict):
                        collected_evidence.append(item)

        verifier_prompt = (
            "You are a verifier agent. Merge candidate analysis into one consistent, evidence-backed JSON.\n"
            "Return JSON only with fields: summary, architecture, entry_points, key_modules, important_flows, risks, next_steps, evidence.\n"
            "Drop claims that cannot be tied to evidence sources.\n"
            f"Facts JSON: {json.dumps(facts, ensure_ascii=True)}\n"
            f"Candidate JSON: {json.dumps(merged, ensure_ascii=True)}\n"
            f"Collected evidence: {json.dumps(collected_evidence[:60], ensure_ascii=True)}\n"
            "JSON:"
        )
        routed_verify = await self._route_text_generation(
            prompt=verifier_prompt,
            task_type="analysis",
            privacy_level=PrivacyLevel.MEDIUM,
        )
        verified = self._extract_json_object(routed_verify) if routed_verify else None
        if isinstance(verified, dict):
            if "evidence" not in verified and collected_evidence:
                verified["evidence"] = collected_evidence[:40]
            verified["_depth_pipeline"] = {
                "mode": "high",
                "rounds": len(stage_artifacts),
                "max_rounds": rounds,
                "stages": ["planner", "analyst", "reviewer", "verifier"],
                "artifacts": stage_artifacts,
            }
            return verified
        if merged:
            if "evidence" not in merged and collected_evidence:
                merged["evidence"] = collected_evidence[:40]
            merged["_depth_pipeline"] = {
                "mode": "high",
                "rounds": len(stage_artifacts),
                "max_rounds": rounds,
                "stages": ["planner", "analyst", "reviewer", "verifier"],
                "artifacts": stage_artifacts,
            }
            return merged
        return {}

    @staticmethod
    def _extract_json_object(text: str) -> Any:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            pass
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    @staticmethod
    def _infer_repo_overview(
        *,
        stack: str,
        file_samples: list[dict[str, str]],
        question: str,
    ) -> dict[str, Any]:
        paths = [str(f.get("path", "")).strip() for f in file_samples if str(f.get("path", "")).strip()]
        lower_paths = [p.lower() for p in paths]
        question_l = question.lower()

        entry_candidates = [
            "main.py",
            "jarvis_main.py",
            "app.py",
            "server.py",
            "manage.py",
            "index.js",
            "server.js",
            "src/main.ts",
            "src/index.ts",
        ]
        entry_points = [p for p in paths if any(p.lower().endswith(c) for c in entry_candidates)][:8]
        if not entry_points:
            entry_points = paths[:4]

        module_hints = ["agents/", "interfaces/", "infrastructure/", "core/", "memory/", "scripts/", "config/"]
        key_modules: list[str] = []
        for hint in module_hints:
            if any(lp.startswith(hint) for lp in lower_paths):
                key_modules.append(hint.rstrip("/"))
        if not key_modules:
            key_modules = sorted({p.split("/", 1)[0] for p in paths if "/" in p})[:8]

        flow = (
            "Request enters API interface, routes through conversation/model layers, "
            "and invokes specialized agents for task execution."
            if any(lp.startswith("interfaces/") for lp in lower_paths)
            and any(lp.startswith("agents/") for lp in lower_paths)
            else "Primary flow appears to be file-driven execution from discovered entry points."
        )
        data_flow = (
            "Configuration and runtime state are loaded from env/config, with persistence handled by memory/research backends."
            if any(lp.startswith("config/") for lp in lower_paths)
            or any(lp.startswith("memory/") for lp in lower_paths)
            else "Data flow likely centers around in-process execution with local file inputs."
        )
        important_flows = [flow, data_flow]

        risks = [
            "Entry points and module boundaries should be validated against full repository tree, not only sampled files.",
            "Environment-specific paths or tokens may cause runtime divergence across local vs container execution.",
        ]
        if "architecture" in question_l or "data flow" in question_l:
            next_steps = [
                "Run repo analysis with higher max_files to widen coverage.",
                "Trace top-level startup path from entry points into service/agent modules.",
                "Document module contracts and external dependencies.",
            ]
        else:
            next_steps = [
                "Ask a focused question on one subsystem for deeper analysis.",
                "Increase max_files for broader repository coverage.",
            ]

        sampled = ", ".join(paths[:12])
        summary = (
            f"This appears to be a {stack} project. I scanned {len(paths)} file(s) and identified likely "
            f"entry points ({', '.join(entry_points[:4])}) and core modules ({', '.join(key_modules[:5])})."
        )
        architecture = (
            f"The codebase is organized around {stack} runtime components with modules: {', '.join(key_modules[:6])}. "
            f"Sampled paths: {sampled}."
            if sampled
            else f"The codebase appears to use a {stack} structure."
        )
        evidence: list[dict[str, Any]] = []
        if entry_points:
            evidence.append(
                {"claim": "Likely entry points were identified from sampled files.", "sources": entry_points[:4]}
            )
        if key_modules:
            module_sources = [p for p in paths if "/" in p][:8]
            evidence.append(
                {"claim": "Key modules were inferred from top-level directory structure.", "sources": module_sources}
            )

        return {
            "summary": summary,
            "architecture": architecture,
            "entry_points": entry_points,
            "key_modules": key_modules,
            "important_flows": important_flows,
            "risks": risks,
            "next_steps": next_steps,
            "evidence": evidence,
        }

    @staticmethod
    def _collect_repo_signals(
        *,
        stack: str,
        file_samples: list[dict[str, str]],
        facts: dict[str, Any],
    ) -> dict[str, Any]:
        paths = [str(f.get("path", "")).strip() for f in file_samples if str(f.get("path", "")).strip()]
        snippets = [str(f.get("snippet", "")) for f in file_samples]
        ext_counter: Counter[str] = Counter()
        dir_counter: Counter[str] = Counter()
        for p in paths:
            parts = p.split("/")
            if len(parts) > 1:
                dir_counter[parts[0]] += 1
            suffix = Path(p).suffix.lower() or "<none>"
            ext_counter[suffix] += 1

        route_hits: list[str] = []
        dep_files = {"pyproject.toml", "requirements.txt", "package.json"}
        dependencies_seen = [p for p in paths if Path(p).name.lower() in dep_files][:5]
        for p, snip in zip(paths, snippets):
            if len(route_hits) >= 8:
                break
            if ".router.add_" in snip or "@app." in snip or "FastAPI(" in snip:
                route_hits.append(p)

        top_dirs = [name for name, _ in dir_counter.most_common(8)]
        top_ext = [name for name, _ in ext_counter.most_common(8)]
        module_distribution = [{ "module": name, "count": count } for name, count in dir_counter.most_common(12)]
        graph_top_modules = facts.get("import_graph_top_modules", []) if isinstance(facts.get("import_graph_top_modules", []), list) else []
        path_traces = facts.get("path_traces", []) if isinstance(facts.get("path_traces", []), list) else []
        return {
            "stack": stack,
            "top_directories": top_dirs,
            "module_distribution": module_distribution,
            "top_extensions": top_ext,
            "dependency_files": dependencies_seen,
            "route_related_files": route_hits,
            "graph_top_modules": graph_top_modules[:8],
            "path_traces": path_traces[:8],
            "sampled_file_count": len(paths),
        }

    @staticmethod
    def _build_repo_tree(workspace: Path, *, max_entries: int = 120) -> str:
        skip_dirs = {".git", "node_modules", ".next", ".venv", "venv", "__pycache__", "dist", "build", "runtime"}
        paths: list[str] = []
        for p in workspace.rglob("*"):
            try:
                rel = str(p.relative_to(workspace))
            except Exception:
                continue
            if not rel:
                continue
            if any(part in skip_dirs for part in p.parts):
                continue
            if p.is_dir():
                continue
            paths.append(rel)
            if len(paths) >= max(1, max_entries):
                break
        paths.sort()
        return "\n".join(paths)

    @staticmethod
    def _apply_code_edits(
        *,
        workspace: Path,
        edits: list[Any],
        dry_run: bool,
    ) -> dict[str, Any]:
        applied_edits: list[dict[str, Any]] = []
        skipped_edits: list[dict[str, Any]] = []
        touched: set[str] = set()

        for idx, raw_edit in enumerate(edits):
            if not isinstance(raw_edit, dict):
                skipped_edits.append({"index": idx, "reason": "invalid_edit_payload"})
                continue
            path_raw = str(raw_edit.get("path", "")).strip()
            op = str(raw_edit.get("operation", "")).strip().lower()
            if not path_raw or not op:
                skipped_edits.append({"index": idx, "path": path_raw, "reason": "missing_path_or_operation"})
                continue
            if path_raw.startswith("/") or ".." in Path(path_raw).parts:
                skipped_edits.append({"index": idx, "path": path_raw, "reason": "unsafe_path"})
                continue
            file_path = (workspace / path_raw).resolve()
            try:
                file_path.relative_to(workspace)
            except ValueError:
                skipped_edits.append({"index": idx, "path": path_raw, "reason": "outside_workspace"})
                continue

            try:
                if op == "create":
                    content = str(raw_edit.get("content", ""))
                    if not content.strip():
                        skipped_edits.append({"index": idx, "path": path_raw, "reason": "empty_create_content"})
                        continue
                    if not dry_run:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(content, encoding="utf-8")
                    touched.add(path_raw)
                    applied_edits.append({"index": idx, "path": path_raw, "operation": op})
                    continue

                if not file_path.exists():
                    skipped_edits.append({"index": idx, "path": path_raw, "reason": "file_not_found"})
                    continue
                original = file_path.read_text(encoding="utf-8")
                updated = original

                if op == "replace":
                    find = str(raw_edit.get("find", ""))
                    repl = str(raw_edit.get("replace", ""))
                    if not find:
                        skipped_edits.append({"index": idx, "path": path_raw, "reason": "missing_find"})
                        continue
                    if find not in original:
                        skipped_edits.append({"index": idx, "path": path_raw, "reason": "find_not_found"})
                        continue
                    updated = original.replace(find, repl, 1)
                elif op == "append":
                    content = str(raw_edit.get("content", ""))
                    if not content:
                        skipped_edits.append({"index": idx, "path": path_raw, "reason": "empty_append_content"})
                        continue
                    joiner = "" if original.endswith("\n") or not original else "\n"
                    updated = f"{original}{joiner}{content}"
                else:
                    skipped_edits.append({"index": idx, "path": path_raw, "reason": f"unsupported_operation:{op}"})
                    continue

                if updated == original:
                    skipped_edits.append({"index": idx, "path": path_raw, "reason": "no_change"})
                    continue
                if not dry_run:
                    file_path.write_text(updated, encoding="utf-8")
                touched.add(path_raw)
                applied_edits.append({"index": idx, "path": path_raw, "operation": op})
            except Exception as exc:
                skipped_edits.append({"index": idx, "path": path_raw, "reason": f"apply_error:{exc}"})

        return {
            "applied_count": len(applied_edits),
            "skipped_count": len(skipped_edits),
            "files_touched": sorted(touched),
            "applied_edits": applied_edits,
            "skipped_edits": skipped_edits,
        }

    async def _run_project_checks(self, workspace: Path, *, stack: str) -> list[dict[str, Any]]:
        commands: list[str] = []
        if stack == "python":
            commands.append("python3 -m compileall -q .")
        elif stack in {"nodejs", "react", "nextjs"}:
            if (workspace / "package.json").exists():
                commands.extend([
                    "npm run -s lint --if-present",
                    "npm run -s test --if-present",
                ])
        results: list[dict[str, Any]] = []
        for cmd in commands:
            started = asyncio.get_running_loop().time()
            proc = await asyncio.to_thread(
                subprocess.run,
                cmd,
                shell=True,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=180,
            )
            elapsed_ms = round((asyncio.get_running_loop().time() - started) * 1000.0, 2)
            output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            results.append(
                {
                    "command": cmd,
                    "return_code": int(proc.returncode),
                    "ok": proc.returncode == 0,
                    "latency_ms": elapsed_ms,
                    "output_tail": output[-1200:],
                }
            )
        return results


# ---------------------------------------------------------------------------
# ManagerAgent
# ---------------------------------------------------------------------------


class ManagerAgent(ConcreteAgent):
    """Agent specialising in project management and resource coordination.

    Capabilities:
    - ``plan_project`` – break down a goal into a structured project plan
    - ``breakdown_task`` – decompose a complex task into sub-tasks
    - ``allocate_resources`` – assign resources to tasks based on constraints

    Args:
        name: Agent name (default ``"manager"``).
        agent_id: Optional pre-assigned ID.
        config_overrides: Optional config value overrides.
    """

    _MANAGER_CAPABILITIES: List[AgentCapability] = [
        AgentCapability(
            name="plan_project",
            description=(
                "Create a structured project plan with milestones, tasks, "
                "timelines, and resource requirements."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "deadline": {"type": "string"},
                    "team_size": {"type": "integer", "minimum": 1},
                    "methodology": {
                        "type": "string",
                        "enum": ["agile", "waterfall", "kanban", "hybrid"],
                        "default": "agile",
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["goal"],
            },
        ),
        AgentCapability(
            name="breakdown_task",
            description=(
                "Decompose a complex task or goal into actionable sub-tasks "
                "with priority, effort, and dependency information."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "max_subtasks": {"type": "integer", "minimum": 1, "default": 10},
                    "granularity": {
                        "type": "string",
                        "enum": ["coarse", "medium", "fine"],
                        "default": "medium",
                    },
                },
                "required": ["task"],
            },
        ),
        AgentCapability(
            name="allocate_resources",
            description=(
                "Assign available resources (agents, tools, time) to a set of "
                "tasks, optimising for throughput and deadlines."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                    "resources": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["priority_first", "round_robin", "deadline_driven"],
                        "default": "priority_first",
                    },
                },
                "required": ["tasks", "resources"],
            },
        ),
    ]

    def __init__(
        self,
        name: str = "manager",
        agent_id: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, agent_id=agent_id, config_overrides=config_overrides)
        self._logger = get_logger(f"agent.{name}")

    def get_capabilities(self) -> List[AgentCapability]:
        return super().get_capabilities() + list(self._MANAGER_CAPABILITIES)

    # ------------------------------------------------------------------
    # Capability handlers
    # ------------------------------------------------------------------

    async def handle_plan_project(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a structured project plan.

        Args:
            parameters: Must contain ``goal`` (str).  Optional ``deadline``,
                ``team_size``, ``methodology``, ``constraints``.

        Returns:
            Dict with ``plan_id``, ``milestones``, ``tasks``, ``risks``.
        """
        goal: str = parameters["goal"]
        deadline: str = parameters.get("deadline", "TBD")
        team_size: int = parameters.get("team_size", 3)
        methodology: str = parameters.get("methodology", "agile")
        constraints: List[str] = parameters.get("constraints", [])

        self._logger.info(
            "ManagerAgent '%s' planning project: %s (methodology=%s)",
            self.name,
            goal[:80],
            methodology,
        )

        plan_id = generate_id("plan")

        routed_summary = await self._route_text_generation(
            prompt=(
                f"Create a concise project planning rationale for goal: {goal}\n"
                f"Methodology: {methodology}\nDeadline: {deadline}\nTeam size: {team_size}"
            ),
            task_type="analysis",
            privacy_level=PrivacyLevel.MEDIUM,
        )

        # Generate milestone structure based on methodology
        if methodology == "agile":
            milestones = [
                {"name": "Sprint 1 - Discovery", "goal": "Requirements & design", "duration_weeks": 2},
                {"name": "Sprint 2 - Core", "goal": "Core feature development", "duration_weeks": 2},
                {"name": "Sprint 3 - Integration", "goal": "Integration & testing", "duration_weeks": 2},
                {"name": "Sprint 4 - Release", "goal": "Hardening & release", "duration_weeks": 2},
            ]
        elif methodology == "waterfall":
            milestones = [
                {"name": "Requirements", "goal": "Complete requirements specification", "duration_weeks": 3},
                {"name": "Design", "goal": "Architecture and detailed design", "duration_weeks": 3},
                {"name": "Implementation", "goal": "Full feature development", "duration_weeks": 6},
                {"name": "Testing", "goal": "QA and acceptance testing", "duration_weeks": 2},
                {"name": "Deployment", "goal": "Production deployment", "duration_weeks": 1},
            ]
        else:  # kanban / hybrid
            milestones = [
                {"name": "Backlog Grooming", "goal": "Prioritise and refine work items", "duration_weeks": 1},
                {"name": "Active Development", "goal": "Continuous delivery", "duration_weeks": 8},
                {"name": "Review & Retrospective", "goal": "Quality check and lessons learned", "duration_weeks": 1},
            ]

        # Generate per-milestone tasks
        tasks = []
        for i, milestone in enumerate(milestones):
            for j in range(min(team_size, 3)):
                tasks.append({
                    "id": f"task-{i + 1}-{j + 1}",
                    "title": f"{milestone['name']} – Work Item {j + 1}",
                    "milestone": milestone["name"],
                    "effort_days": 2 + j,
                    "priority": "high" if j == 0 else "medium",
                    "assignee": f"team_member_{j + 1}",
                    "status": "pending",
                })

        # Identify risks from constraints
        risks = [
            {"risk": f"Constraint '{c}' may delay timeline", "mitigation": "Monitor and adjust sprint goals."}
            for c in constraints
        ]
        if team_size < 2:
            risks.append({
                "risk": "Single-person team creates bus-factor risk",
                "mitigation": "Document all decisions; consider part-time support.",
            })

        return {
            "plan_id": plan_id,
            "goal": goal,
            "methodology": methodology,
            "deadline": deadline,
            "team_size": team_size,
            "milestone_count": len(milestones),
            "task_count": len(tasks),
            "milestones": milestones,
            "tasks": tasks,
            "risks": risks,
            "planning_rationale": routed_summary or "",
            "manager_id": self.agent_id,
            "timestamp": timestamp_now(),
        }

    async def handle_breakdown_task(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decompose a complex task into sub-tasks.

        Args:
            parameters: Must contain ``task`` (str).  Optional ``max_subtasks``
                and ``granularity`` (``"coarse"`` | ``"medium"`` | ``"fine"``).

        Returns:
            Dict with ``subtasks`` list, each having ``id``, ``title``,
            ``effort``, ``priority``, ``dependencies``.
        """
        task: str = parameters["task"]
        max_subtasks: int = parameters.get("max_subtasks", 10)
        granularity: str = parameters.get("granularity", "medium")

        self._logger.info(
            "ManagerAgent '%s' breaking down task: %s (granularity=%s)",
            self.name,
            task[:80],
            granularity,
        )

        # Granularity maps to number of sub-tasks and effort estimates
        subtask_count = {"coarse": 3, "medium": 6, "fine": 10}.get(granularity, 6)
        subtask_count = min(subtask_count, max_subtasks)

        effort_map = {"coarse": "L", "medium": "M", "fine": "S"}
        effort = effort_map.get(granularity, "M")

        phases = [
            "Analysis & planning",
            "Design",
            "Implementation",
            "Testing",
            "Integration",
            "Documentation",
            "Review",
            "Deployment",
            "Monitoring",
            "Retrospective",
        ]

        subtasks = []
        for i in range(subtask_count):
            phase = phases[i % len(phases)]
            dep = [f"subtask-{i}"] if i > 0 else []
            subtasks.append({
                "id": f"subtask-{i + 1}",
                "title": f"{phase}: {task[:50]}",
                "description": f"Complete the {phase.lower()} phase for: {task}",
                "effort": effort,
                "priority": "high" if i < 2 else ("medium" if i < subtask_count - 2 else "low"),
                "estimated_hours": {"S": 2, "M": 4, "L": 8}.get(effort, 4) * (1 + i % 3),
                "dependencies": dep,
                "status": "pending",
            })

        total_hours = sum(s["estimated_hours"] for s in subtasks)

        return {
            "task": task,
            "granularity": granularity,
            "subtask_count": len(subtasks),
            "subtasks": subtasks,
            "total_estimated_hours": total_hours,
            "critical_path": [s["id"] for s in subtasks if s["priority"] == "high"],
            "manager_id": self.agent_id,
            "timestamp": timestamp_now(),
        }

    async def handle_allocate_resources(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assign resources to tasks.

        Args:
            parameters: Must contain ``tasks`` (list) and ``resources``
                (list).  Optional ``strategy``.

        Returns:
            Dict with ``assignments`` list mapping task→resource, plus
            utilisation statistics.
        """
        tasks: List[Dict[str, Any]] = parameters["tasks"]
        resources: List[Dict[str, Any]] = parameters["resources"]
        strategy: str = parameters.get("strategy", "priority_first")

        self._logger.info(
            "ManagerAgent '%s' allocating %d resource(s) to %d task(s) (strategy=%s)",
            self.name,
            len(resources),
            len(tasks),
            strategy,
        )

        if not resources:
            return {
                "strategy": strategy,
                "assignments": [],
                "unassigned_tasks": [t.get("id", str(i)) for i, t in enumerate(tasks)],
                "utilisation": {},
                "warning": "No resources provided; all tasks unassigned.",
                "manager_id": self.agent_id,
                "timestamp": timestamp_now(),
            }

        # Sort tasks by strategy
        sorted_tasks = list(tasks)
        if strategy == "priority_first":
            priority_order = {"high": 0, "critical": 0, "medium": 1, "low": 2}
            sorted_tasks.sort(
                key=lambda t: priority_order.get(str(t.get("priority", "medium")).lower(), 1)
            )
        elif strategy == "deadline_driven":
            sorted_tasks.sort(key=lambda t: t.get("deadline", "9999-12-31"))

        # Simple round-robin or priority assignment
        assignments: List[Dict[str, Any]] = []
        resource_load: Dict[str, int] = {
            r.get("id", str(i)): 0 for i, r in enumerate(resources)
        }
        resource_list = list(resources)

        for task_idx, task in enumerate(sorted_tasks):
            task_id = task.get("id", f"task-{task_idx + 1}")
            if strategy == "round_robin":
                resource = resource_list[task_idx % len(resource_list)]
            else:
                # Assign to least-loaded resource
                resource = min(
                    resource_list,
                    key=lambda r: resource_load.get(r.get("id", str(r)), 0),
                )

            resource_id = resource.get("id", str(resource_list.index(resource)))
            resource_load[resource_id] = resource_load.get(resource_id, 0) + 1

            assignments.append({
                "task_id": task_id,
                "task_title": task.get("title", task_id),
                "resource_id": resource_id,
                "resource_name": resource.get("name", resource_id),
                "assigned_at": timestamp_now(),
            })

        utilisation = {
            rid: {"task_count": count, "utilisation_pct": round(count / len(tasks) * 100, 1)}
            for rid, count in resource_load.items()
        }

        return {
            "strategy": strategy,
            "task_count": len(tasks),
            "resource_count": len(resources),
            "assignments": assignments,
            "unassigned_tasks": [],
            "utilisation": utilisation,
            "manager_id": self.agent_id,
            "timestamp": timestamp_now(),
        }


__all__ = ["AnalystAgent", "DeveloperAgent", "ManagerAgent"]
