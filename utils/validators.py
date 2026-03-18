"""
Input validation utilities for the JARVIS AI OS system.

All validators raise :class:`~utils.exceptions.ValidationError` on failure so
callers can use a single ``except`` clause for any validation problem.

The :class:`InputValidator` class provides schema-based validation backed by
*jsonschema* when it is installed, and falls back to a lightweight built-in
checker otherwise.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# jsonschema – optional dependency
# ---------------------------------------------------------------------------

try:
    import jsonschema  # type: ignore

    _JSONSCHEMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JSONSCHEMA_AVAILABLE = False
    logger.debug(
        "jsonschema not installed; InputValidator will use built-in validation"
    )

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_API_KEY_RE = re.compile(r"^[A-Za-z0-9_\-]{8,512}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]{0,127}$")

# ---------------------------------------------------------------------------
# Task validation
# ---------------------------------------------------------------------------

_TASK_REQUIRED_FIELDS: dict[str, type] = {
    "id": str,
    "type": str,
    "description": str,
}
_TASK_OPTIONAL_FIELDS: dict[str, type] = {
    "priority": int,
    "timeout": (int, float),  # type: ignore[dict-item]
    "dependencies": list,
    "metadata": dict,
    "agent_id": str,
    "skill": str,
    "parameters": dict,
}


def validate_task_definition(task: Any) -> None:
    """Validate a task definition dictionary.

    Expected keys: ``id``, ``type``, ``description`` (required);
    ``priority``, ``timeout``, ``dependencies``, ``metadata``,
    ``agent_id``, ``skill``, ``parameters`` (optional).

    Args:
        task: The object to validate (must be a :class:`dict`).

    Raises:
        ValidationError: If *task* is not a dict or contains invalid fields.
    """
    if not isinstance(task, dict):
        raise ValidationError(field="task", reason="must be a dict")

    for field, expected_type in _TASK_REQUIRED_FIELDS.items():
        if field not in task:
            raise ValidationError(field=field, reason="required field is missing")
        if not isinstance(task[field], expected_type):
            raise ValidationError(
                field=field,
                reason=f"expected {expected_type.__name__}, got {type(task[field]).__name__}",
            )
        if not task[field]:
            raise ValidationError(field=field, reason="must not be empty")

    if not _IDENTIFIER_RE.match(task["id"]):
        raise ValidationError(
            field="id",
            reason="must match pattern [A-Za-z_][A-Za-z0-9_-]{0,127}",
        )

    for field, expected_type in _TASK_OPTIONAL_FIELDS.items():
        if field in task and not isinstance(task[field], expected_type):
            raise ValidationError(
                field=field,
                reason=f"expected {expected_type}, got {type(task[field]).__name__}",
            )

    if "priority" in task and not (0 <= task["priority"] <= 10):
        raise ValidationError(field="priority", reason="must be between 0 and 10")

    if "timeout" in task and task["timeout"] <= 0:
        raise ValidationError(field="timeout", reason="must be a positive number")

    if "dependencies" in task:
        for i, dep in enumerate(task["dependencies"]):
            if not isinstance(dep, str):
                raise ValidationError(
                    field=f"dependencies[{i}]",
                    reason="each dependency must be a string task ID",
                )


# ---------------------------------------------------------------------------
# Agent config validation
# ---------------------------------------------------------------------------

_AGENT_REQUIRED: dict[str, type] = {
    "agent_id": str,
    "name": str,
    "type": str,
}
_VALID_AGENT_TYPES = {"reactive", "deliberative", "hybrid", "specialized", "tool"}


def validate_agent_config(config: Any) -> None:
    """Validate an agent configuration dictionary.

    Required keys: ``agent_id``, ``name``, ``type``.
    Optional keys: ``capabilities`` (list), ``max_concurrent_tasks`` (int),
    ``timeout`` (int/float), ``model`` (str), ``metadata`` (dict).

    Args:
        config: The object to validate.

    Raises:
        ValidationError: If *config* is invalid.
    """
    if not isinstance(config, dict):
        raise ValidationError(field="agent_config", reason="must be a dict")

    for field, expected_type in _AGENT_REQUIRED.items():
        if field not in config:
            raise ValidationError(field=field, reason="required field is missing")
        if not isinstance(config[field], expected_type):
            raise ValidationError(
                field=field,
                reason=f"expected {expected_type.__name__}",
            )
        if not config[field]:
            raise ValidationError(field=field, reason="must not be empty")

    if not _IDENTIFIER_RE.match(config["agent_id"]):
        raise ValidationError(
            field="agent_id",
            reason="must match pattern [A-Za-z_][A-Za-z0-9_-]{0,127}",
        )

    if config["type"] not in _VALID_AGENT_TYPES:
        raise ValidationError(
            field="type",
            reason=f"must be one of {sorted(_VALID_AGENT_TYPES)}",
        )

    if "capabilities" in config:
        if not isinstance(config["capabilities"], list):
            raise ValidationError(field="capabilities", reason="must be a list")
        for cap in config["capabilities"]:
            if not isinstance(cap, str):
                raise ValidationError(
                    field="capabilities",
                    reason="each capability must be a string",
                )

    if "max_concurrent_tasks" in config:
        val = config["max_concurrent_tasks"]
        if not isinstance(val, int) or val < 1:
            raise ValidationError(
                field="max_concurrent_tasks",
                reason="must be a positive integer",
            )


# ---------------------------------------------------------------------------
# Skill schema validation
# ---------------------------------------------------------------------------

_VALID_PARAM_TYPES = {"string", "integer", "number", "boolean", "array", "object", "null"}


def validate_skill_schema(schema: Any) -> None:
    """Validate a skill parameter schema.

    The schema should follow a simplified JSON-Schema-like structure::

        {
            "name": "my_skill",
            "description": "...",
            "parameters": {
                "param_name": {
                    "type": "string",
                    "description": "...",
                    "required": True
                }
            }
        }

    Args:
        schema: The schema dict to validate.

    Raises:
        ValidationError: If the schema is structurally invalid.
    """
    if not isinstance(schema, dict):
        raise ValidationError(field="skill_schema", reason="must be a dict")

    for field in ("name", "description"):
        if field not in schema:
            raise ValidationError(field=field, reason="required field is missing")
        if not isinstance(schema[field], str) or not schema[field]:
            raise ValidationError(field=field, reason="must be a non-empty string")

    if not _IDENTIFIER_RE.match(schema["name"]):
        raise ValidationError(
            field="name",
            reason="must match pattern [A-Za-z_][A-Za-z0-9_-]{0,127}",
        )

    if "parameters" in schema:
        params = schema["parameters"]
        if not isinstance(params, dict):
            raise ValidationError(field="parameters", reason="must be a dict")

        for param_name, param_def in params.items():
            if not isinstance(param_def, dict):
                raise ValidationError(
                    field=f"parameters.{param_name}",
                    reason="parameter definition must be a dict",
                )
            if "type" not in param_def:
                raise ValidationError(
                    field=f"parameters.{param_name}.type",
                    reason="required field is missing",
                )
            ptype = param_def["type"]
            if isinstance(ptype, list):
                for t in ptype:
                    if t not in _VALID_PARAM_TYPES:
                        raise ValidationError(
                            field=f"parameters.{param_name}.type",
                            reason=f"'{t}' is not a valid type",
                        )
            elif ptype not in _VALID_PARAM_TYPES:
                raise ValidationError(
                    field=f"parameters.{param_name}.type",
                    reason=f"'{ptype}' is not a valid type",
                )


# ---------------------------------------------------------------------------
# Workflow validation
# ---------------------------------------------------------------------------


def validate_workflow_definition(workflow: Any) -> None:
    """Validate a workflow definition dictionary.

    Required keys: ``workflow_id``, ``name``, ``steps`` (list).
    Each step must be a dict with at least ``step_id`` and ``task_type``.

    Args:
        workflow: The object to validate.

    Raises:
        ValidationError: If *workflow* is structurally invalid.
    """
    if not isinstance(workflow, dict):
        raise ValidationError(field="workflow", reason="must be a dict")

    for field in ("workflow_id", "name"):
        if field not in workflow:
            raise ValidationError(field=field, reason="required field is missing")
        if not isinstance(workflow[field], str) or not workflow[field]:
            raise ValidationError(field=field, reason="must be a non-empty string")

    if "steps" not in workflow:
        raise ValidationError(field="steps", reason="required field is missing")
    if not isinstance(workflow["steps"], list) or not workflow["steps"]:
        raise ValidationError(field="steps", reason="must be a non-empty list")

    seen_step_ids: set[str] = set()
    for idx, step in enumerate(workflow["steps"]):
        prefix = f"steps[{idx}]"
        if not isinstance(step, dict):
            raise ValidationError(field=prefix, reason="each step must be a dict")
        for field in ("step_id", "task_type"):
            if field not in step:
                raise ValidationError(
                    field=f"{prefix}.{field}",
                    reason="required field is missing",
                )
            if not isinstance(step[field], str) or not step[field]:
                raise ValidationError(
                    field=f"{prefix}.{field}",
                    reason="must be a non-empty string",
                )
        if step["step_id"] in seen_step_ids:
            raise ValidationError(
                field=f"{prefix}.step_id",
                reason=f"duplicate step_id '{step['step_id']}'",
            )
        seen_step_ids.add(step["step_id"])

        if "depends_on" in step:
            if not isinstance(step["depends_on"], list):
                raise ValidationError(
                    field=f"{prefix}.depends_on",
                    reason="must be a list of step IDs",
                )
            for dep in step["depends_on"]:
                if not isinstance(dep, str):
                    raise ValidationError(
                        field=f"{prefix}.depends_on",
                        reason="each dependency must be a string step ID",
                    )


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------


def validate_api_key(key: Any, *, provider: str = "") -> None:
    """Check that *key* looks like a plausible API key.

    Validates format only; does **not** make a network call to verify the key.

    Args:
        key: The value to check.
        provider: Optional provider name used in the error message.

    Raises:
        ValidationError: If *key* is not a non-empty string that matches
            ``^[A-Za-z0-9_-]{8,512}$``.
    """
    field = f"api_key({provider})" if provider else "api_key"
    if not isinstance(key, str):
        raise ValidationError(field=field, reason="must be a string")
    if not key:
        raise ValidationError(field=field, reason="must not be empty")
    if not _API_KEY_RE.match(key):
        raise ValidationError(
            field=field,
            reason="must be 8–512 characters (A-Z, a-z, 0-9, _, -)",
        )


# ---------------------------------------------------------------------------
# File path validation
# ---------------------------------------------------------------------------

_UNSAFE_PATH_RE = re.compile(r"\.\.[/\\]|[\x00]")


def validate_file_path(
    path: Any,
    *,
    base_dir: Optional[str] = None,
    must_exist: bool = False,
    allow_absolute: bool = False,
) -> Path:
    """Validate and resolve a file path, preventing directory traversal.

    Args:
        path: The path string to validate.
        base_dir: If provided, the path must resolve to a location inside
            this directory (prevents ``../`` escapes).
        must_exist: When ``True``, raise if the resolved path does not exist
            on the filesystem.
        allow_absolute: When ``False`` (default) and *base_dir* is set,
            reject absolute *path* values.

    Returns:
        The resolved :class:`pathlib.Path`.

    Raises:
        ValidationError: On any check failure.
    """
    if not isinstance(path, (str, Path)):
        raise ValidationError(field="path", reason="must be a string or Path")

    path_str = str(path)

    if _UNSAFE_PATH_RE.search(path_str):
        raise ValidationError(field="path", reason="path contains unsafe sequences")

    resolved = Path(path_str)

    if base_dir is not None:
        base = Path(base_dir).resolve()

        if not allow_absolute and resolved.is_absolute():
            raise ValidationError(
                field="path",
                reason="absolute paths are not permitted when base_dir is set",
            )

        try:
            resolved = (base / resolved).resolve()
        except Exception as exc:
            raise ValidationError(
                field="path", reason=f"could not resolve path: {exc}"
            ) from exc

        # Ensure the resolved path is inside base_dir
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise ValidationError(
                field="path",
                reason=f"path resolves outside of base directory '{base}'",
            ) from exc

    if must_exist and not resolved.exists():
        raise ValidationError(field="path", reason=f"path does not exist: {resolved}")

    return resolved


# ---------------------------------------------------------------------------
# Schema-based validator class
# ---------------------------------------------------------------------------


class InputValidator:
    """Schema-based input validator.

    Uses *jsonschema* when available; otherwise performs basic type/required
    field checks via a built-in fallback.

    Args:
        schema: A JSON-Schema-compatible dict describing the expected shape of
            the input.

    Example::

        validator = InputValidator({
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
        })
        validator.validate({"name": "Alice", "age": 30})  # OK
        validator.validate({"name": "Bob"})               # raises ValidationError
    """

    def __init__(self, schema: dict[str, Any]) -> None:
        if not isinstance(schema, dict):
            raise ValueError("schema must be a dict")
        self._schema = schema

        if _JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.Draft7Validator.check_schema(schema)
                self._validator = jsonschema.Draft7Validator(schema)
            except jsonschema.SchemaError as exc:
                raise ValueError(f"Invalid JSON schema: {exc.message}") from exc
        else:
            self._validator = None  # type: ignore[assignment]

    def validate(self, data: Any) -> None:
        """Validate *data* against the schema.

        Args:
            data: The Python object to validate.

        Raises:
            ValidationError: If *data* does not conform to the schema.
        """
        if _JSONSCHEMA_AVAILABLE and self._validator is not None:
            errors = list(self._validator.iter_errors(data))
            if errors:
                # Report all errors, most important first
                errors.sort(key=lambda e: e.path)
                first = errors[0]
                field = ".".join(str(p) for p in first.absolute_path) or "root"
                raise ValidationError(field=field, reason=first.message)
        else:
            self._fallback_validate(data, self._schema)

    def _fallback_validate(self, data: Any, schema: dict[str, Any]) -> None:
        """Lightweight schema validation used when jsonschema is not available."""
        schema_type = schema.get("type")
        if schema_type:
            self._check_type(data, schema_type)

        if isinstance(data, dict):
            for required_field in schema.get("required", []):
                if required_field not in data:
                    raise ValidationError(
                        field=required_field,
                        reason="required field is missing",
                    )
            props: dict[str, Any] = schema.get("properties", {})
            for prop_name, prop_schema in props.items():
                if prop_name in data:
                    self._fallback_validate(data[prop_name], prop_schema)

        if isinstance(data, (int, float)) and not isinstance(data, bool):
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            if minimum is not None and data < minimum:
                raise ValidationError(
                    field="value",
                    reason=f"value {data} is less than minimum {minimum}",
                )
            if maximum is not None and data > maximum:
                raise ValidationError(
                    field="value",
                    reason=f"value {data} exceeds maximum {maximum}",
                )

        if isinstance(data, str):
            min_len = schema.get("minLength")
            max_len = schema.get("maxLength")
            pattern = schema.get("pattern")
            if min_len is not None and len(data) < min_len:
                raise ValidationError(
                    field="value",
                    reason=f"string length {len(data)} < minLength {min_len}",
                )
            if max_len is not None and len(data) > max_len:
                raise ValidationError(
                    field="value",
                    reason=f"string length {len(data)} > maxLength {max_len}",
                )
            if pattern and not re.search(pattern, data):
                raise ValidationError(
                    field="value",
                    reason=f"string does not match pattern '{pattern}'",
                )

        enum_values = schema.get("enum")
        if enum_values is not None and data not in enum_values:
            raise ValidationError(
                field="value",
                reason=f"value {data!r} is not one of {enum_values}",
            )

    @staticmethod
    def _check_type(data: Any, schema_type: str | list[str]) -> None:
        """Raise ValidationError if *data* does not match *schema_type*."""
        _TYPE_MAP: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        types = schema_type if isinstance(schema_type, list) else [schema_type]
        for t in types:
            expected = _TYPE_MAP.get(t)
            if expected is None:
                continue
            # bool is subclass of int in Python; treat carefully
            if t == "integer" and isinstance(data, bool):
                continue
            if t == "boolean" and not isinstance(data, bool):
                continue
            if isinstance(data, expected):
                return
        raise ValidationError(
            field="value",
            reason=f"expected type {schema_type!r}, got {type(data).__name__}",
        )


__all__ = [
    "validate_task_definition",
    "validate_agent_config",
    "validate_skill_schema",
    "validate_workflow_definition",
    "validate_api_key",
    "validate_file_path",
    "InputValidator",
]
