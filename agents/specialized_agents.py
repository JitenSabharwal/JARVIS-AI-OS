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
import re
import textwrap
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
