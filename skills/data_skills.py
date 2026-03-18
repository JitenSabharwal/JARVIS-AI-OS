"""
Data analysis and processing skills for JARVIS AI OS.

All statistics are computed with pure Python (no numpy/pandas required).

Skills provided:
- :class:`ParseJSONSkill`      – Parse, validate, and pretty-print JSON
- :class:`ParseCSVSkill`       – Parse CSV data into structured records
- :class:`DataSummarySkill`    – Compute descriptive statistics
- :class:`FormatDataSkill`     – Convert between JSON, CSV, and plain text
- :class:`FilterDataSkill`     – Filter lists/dicts by criteria
- :class:`TransformDataSkill`  – Map/reduce/sort/group operations on data
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from typing import Any, Callable, Dict, List, Optional, Union

from infrastructure.logger import get_logger
from skills.base_skill import BaseSkill, SkillParameter, SkillResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure-Python statistics helpers
# ---------------------------------------------------------------------------


def _numeric_values(data: List[Any]) -> List[float]:
    """Extract numeric values from a mixed list."""
    result = []
    for item in data:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            pass
    return result


def _compute_stats(values: List[float]) -> Dict[str, Any]:
    """Compute descriptive statistics for *values* without numpy."""
    if not values:
        return {"count": 0}

    n = len(values)
    sorted_vals = sorted(values)
    total = sum(values)
    mean = total / n

    # Median
    mid = n // 2
    if n % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    else:
        median = sorted_vals[mid]

    # Standard deviation (population if n==1, sample otherwise)
    if n >= 2:
        stdev = statistics.stdev(values)
        variance = statistics.variance(values)
    else:
        stdev = 0.0
        variance = 0.0

    # Percentiles
    def percentile(p: float) -> float:
        idx = (n - 1) * p / 100
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        fraction = idx - lower
        return sorted_vals[lower] + fraction * (sorted_vals[upper] - sorted_vals[lower])

    return {
        "count": n,
        "sum": total,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": round(mean, 6),
        "median": round(median, 6),
        "stdev": round(stdev, 6),
        "variance": round(variance, 6),
        "p25": round(percentile(25), 6),
        "p75": round(percentile(75), 6),
        "p95": round(percentile(95), 6),
        "range": sorted_vals[-1] - sorted_vals[0],
    }


# ---------------------------------------------------------------------------
# ParseJSONSkill
# ---------------------------------------------------------------------------


class ParseJSONSkill(BaseSkill):
    """Parse a JSON string and return the structured object with optional formatting."""

    @property
    def name(self) -> str:
        return "parse_json"

    @property
    def description(self) -> str:
        return (
            "Parse a JSON string into a structured Python object. "
            "Optionally pretty-print, validate schema, or extract a sub-path."
        )

    @property
    def category(self) -> str:
        return "data"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter("json_string", "string", "The JSON-encoded string to parse.", required=True),
            SkillParameter(
                "pretty_print",
                "boolean",
                "Return a formatted, indented JSON string instead of the raw object.",
                required=False,
                default=False,
            ),
            SkillParameter(
                "path",
                "string",
                (
                    "Dot-notation path to extract a sub-value "
                    "(e.g. 'data.users.0.name')."
                ),
                required=False,
                default="",
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not isinstance(params.get("json_string"), str):
            raise ValueError("'json_string' must be a string.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        json_string: str = params["json_string"]
        pretty_print = bool(params.get("pretty_print", False))
        path_str: str = params.get("path", "").strip()

        try:
            parsed = json.loads(json_string)
        except json.JSONDecodeError as exc:
            return SkillResult.failure(
                error=f"Invalid JSON: {exc}",
                metadata={"position": exc.pos, "line": exc.lineno, "col": exc.colno},
            )

        # Navigate dot-notation path
        if path_str:
            current = parsed
            for segment in path_str.split("."):
                try:
                    if isinstance(current, list):
                        current = current[int(segment)]
                    elif isinstance(current, dict):
                        current = current[segment]
                    else:
                        return SkillResult.failure(
                            error=f"Cannot navigate into {type(current).__name__} at segment '{segment}'."
                        )
                except (KeyError, IndexError, ValueError) as exc:
                    return SkillResult.failure(error=f"Path '{path_str}' not found: {exc}")
            parsed = current

        if pretty_print:
            output = json.dumps(parsed, indent=2, default=str)
            return SkillResult.ok(
                data={"formatted": output, "type": type(parsed).__name__},
                metadata={"pretty_printed": True},
            )

        return SkillResult.ok(
            data={"parsed": parsed, "type": type(parsed).__name__},
            metadata={"pretty_printed": False},
        )


# ---------------------------------------------------------------------------
# ParseCSVSkill
# ---------------------------------------------------------------------------


class ParseCSVSkill(BaseSkill):
    """Parse CSV text into a list of row dictionaries."""

    @property
    def name(self) -> str:
        return "parse_csv"

    @property
    def description(self) -> str:
        return (
            "Parse a CSV-formatted string into a list of records. "
            "Supports custom delimiters and optional header row."
        )

    @property
    def category(self) -> str:
        return "data"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter("csv_string", "string", "The CSV-formatted text to parse.", required=True),
            SkillParameter(
                "delimiter",
                "string",
                "Field delimiter character (default: ',').",
                required=False,
                default=",",
            ),
            SkillParameter(
                "has_header",
                "boolean",
                "Whether the first row is a header row.",
                required=False,
                default=True,
            ),
            SkillParameter(
                "max_rows",
                "integer",
                "Maximum number of data rows to return (0 = all).",
                required=False,
                default=0,
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not isinstance(params.get("csv_string"), str):
            raise ValueError("'csv_string' must be a string.")
        delimiter = params.get("delimiter", ",")
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise ValueError("'delimiter' must be a single character.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        csv_string: str = params["csv_string"]
        delimiter: str = params.get("delimiter", ",")
        has_header = bool(params.get("has_header", True))
        max_rows = int(params.get("max_rows", 0))

        try:
            reader = csv.reader(io.StringIO(csv_string), delimiter=delimiter)
            rows = list(reader)
        except csv.Error as exc:
            return SkillResult.failure(error=f"CSV parse error: {exc}")

        if not rows:
            return SkillResult.ok(data={"records": [], "columns": [], "row_count": 0})

        if has_header:
            headers = rows[0]
            data_rows = rows[1:]
        else:
            headers = [f"col_{i}" for i in range(len(rows[0]))]
            data_rows = rows

        if max_rows:
            data_rows = data_rows[:max_rows]

        records = []
        for row in data_rows:
            record: Dict[str, Any] = {}
            for i, col in enumerate(headers):
                record[col] = row[i] if i < len(row) else None
            records.append(record)

        return SkillResult.ok(
            data={"records": records, "columns": headers, "row_count": len(records)},
            metadata={"has_header": has_header, "delimiter": delimiter, "truncated": bool(max_rows)},
        )


# ---------------------------------------------------------------------------
# DataSummarySkill
# ---------------------------------------------------------------------------


class DataSummarySkill(BaseSkill):
    """Compute descriptive statistics for a list of numbers or table column."""

    @property
    def name(self) -> str:
        return "data_summary"

    @property
    def description(self) -> str:
        return (
            "Compute descriptive statistics (min, max, mean, median, stdev, etc.) "
            "for a list of numeric values or a named column in a list of records."
        )

    @property
    def category(self) -> str:
        return "data"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter(
                "data",
                "array",
                "A list of numbers, or a list of dicts if 'column' is specified.",
                required=True,
            ),
            SkillParameter(
                "column",
                "string",
                "Key to extract from each dict when 'data' is a list of records.",
                required=False,
                default="",
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        if "data" not in params or not isinstance(params["data"], list):
            raise ValueError("'data' must be a list.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        raw_data: List[Any] = params["data"]
        column: str = params.get("column", "").strip()

        if column:
            values_raw = [row.get(column) for row in raw_data if isinstance(row, dict)]
        else:
            values_raw = raw_data

        values = _numeric_values(values_raw)
        non_numeric_count = len(values_raw) - len(values)

        if not values:
            return SkillResult.failure(
                error="No numeric values found in the provided data.",
                metadata={"total_items": len(raw_data), "non_numeric": non_numeric_count},
            )

        stats = _compute_stats(values)

        return SkillResult.ok(
            data=stats,
            metadata={
                "column": column or None,
                "non_numeric_count": non_numeric_count,
                "total_input_rows": len(raw_data),
            },
        )


# ---------------------------------------------------------------------------
# FormatDataSkill
# ---------------------------------------------------------------------------


class FormatDataSkill(BaseSkill):
    """Convert structured data between JSON, CSV, and plain-text formats."""

    @property
    def name(self) -> str:
        return "format_data"

    @property
    def description(self) -> str:
        return (
            "Convert data between formats: json, csv, or text. "
            "Input must be a JSON string representing a list of dicts or a flat list."
        )

    @property
    def category(self) -> str:
        return "data"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter(
                "data",
                "string",
                "JSON-encoded data (list of dicts or flat list).",
                required=True,
            ),
            SkillParameter(
                "output_format",
                "string",
                "Target format: 'json', 'csv', or 'text'.",
                required=True,
            ),
            SkillParameter(
                "indent",
                "integer",
                "JSON indentation level (2 or 4). Applies to output_format='json'.",
                required=False,
                default=2,
            ),
            SkillParameter(
                "delimiter",
                "string",
                "CSV delimiter character. Applies to output_format='csv'.",
                required=False,
                default=",",
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not isinstance(params.get("data"), str):
            raise ValueError("'data' must be a JSON string.")
        fmt = params.get("output_format", "")
        if fmt not in ("json", "csv", "text"):
            raise ValueError("'output_format' must be 'json', 'csv', or 'text'.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        data_str: str = params["data"]
        output_format: str = params["output_format"]
        indent = int(params.get("indent", 2))
        delimiter: str = params.get("delimiter", ",")

        try:
            parsed = json.loads(data_str)
        except json.JSONDecodeError as exc:
            return SkillResult.failure(error=f"Invalid input JSON: {exc}")

        if output_format == "json":
            output = json.dumps(parsed, indent=indent, default=str, ensure_ascii=False)
            return SkillResult.ok(data={"output": output, "format": "json"})

        if output_format == "csv":
            if not isinstance(parsed, list):
                return SkillResult.failure(error="CSV output requires a list as input.")
            buf = io.StringIO()
            if parsed and isinstance(parsed[0], dict):
                fieldnames = list(parsed[0].keys())
                writer = csv.DictWriter(buf, fieldnames=fieldnames, delimiter=delimiter)
                writer.writeheader()
                for row in parsed:
                    # Fill missing keys with empty string
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
            else:
                writer_flat = csv.writer(buf, delimiter=delimiter)
                for item in parsed:
                    writer_flat.writerow([item] if not isinstance(item, (list, tuple)) else item)
            return SkillResult.ok(data={"output": buf.getvalue(), "format": "csv"})

        # output_format == "text"
        if isinstance(parsed, list):
            lines = [str(item) for item in parsed]
        elif isinstance(parsed, dict):
            lines = [f"{k}: {v}" for k, v in parsed.items()]
        else:
            lines = [str(parsed)]
        return SkillResult.ok(data={"output": "\n".join(lines), "format": "text"})


# ---------------------------------------------------------------------------
# FilterDataSkill
# ---------------------------------------------------------------------------


class FilterDataSkill(BaseSkill):
    """Filter a list of items by field values or expression predicates."""

    @property
    def name(self) -> str:
        return "filter_data"

    @property
    def description(self) -> str:
        return (
            "Filter a JSON-encoded list of dicts or primitives by specifying "
            "field criteria. Supports equality, comparison, and contains operators."
        )

    @property
    def category(self) -> str:
        return "data"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter(
                "data",
                "string",
                "JSON-encoded list to filter.",
                required=True,
            ),
            SkillParameter(
                "filters",
                "array",
                (
                    "List of filter objects with keys: "
                    "field (str), operator (eq|ne|gt|lt|gte|lte|contains|startswith|endswith), value."
                ),
                required=True,
            ),
            SkillParameter(
                "logic",
                "string",
                "Combine filters with 'and' (all must match) or 'or' (any must match).",
                required=False,
                default="and",
            ),
            SkillParameter(
                "limit",
                "integer",
                "Maximum number of results to return (0 = all).",
                required=False,
                default=0,
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not isinstance(params.get("data"), str):
            raise ValueError("'data' must be a JSON string.")
        filters = params.get("filters", [])
        if not isinstance(filters, list) or not filters:
            raise ValueError("'filters' must be a non-empty list.")
        valid_ops = {"eq", "ne", "gt", "lt", "gte", "lte", "contains", "startswith", "endswith"}
        for f in filters:
            if not isinstance(f, dict):
                raise ValueError("Each filter must be a dict.")
            if "field" not in f or "operator" not in f or "value" not in f:
                raise ValueError("Each filter must have 'field', 'operator', and 'value' keys.")
            if f["operator"] not in valid_ops:
                raise ValueError(f"Operator must be one of: {sorted(valid_ops)}")
        logic = params.get("logic", "and")
        if logic not in ("and", "or"):
            raise ValueError("'logic' must be 'and' or 'or'.")

    def _matches(self, item: Any, filter_def: Dict[str, Any]) -> bool:
        """Test whether *item* satisfies a single filter."""
        field = filter_def["field"]
        op = filter_def["operator"]
        expected = filter_def["value"]

        if isinstance(item, dict):
            actual = item.get(field)
        else:
            actual = item

        try:
            if op == "eq":
                return actual == expected
            if op == "ne":
                return actual != expected
            if op == "gt":
                return float(actual) > float(expected)
            if op == "lt":
                return float(actual) < float(expected)
            if op == "gte":
                return float(actual) >= float(expected)
            if op == "lte":
                return float(actual) <= float(expected)
            if op == "contains":
                return str(expected).lower() in str(actual).lower()
            if op == "startswith":
                return str(actual).lower().startswith(str(expected).lower())
            if op == "endswith":
                return str(actual).lower().endswith(str(expected).lower())
        except (TypeError, ValueError):
            return False
        return False

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        data_str: str = params["data"]
        filters: List[Dict[str, Any]] = params["filters"]
        logic: str = params.get("logic", "and")
        limit = int(params.get("limit", 0))

        try:
            items: List[Any] = json.loads(data_str)
            if not isinstance(items, list):
                return SkillResult.failure(error="Input 'data' must be a JSON array.")
        except json.JSONDecodeError as exc:
            return SkillResult.failure(error=f"Invalid input JSON: {exc}")

        results = []
        for item in items:
            if logic == "and":
                passes = all(self._matches(item, f) for f in filters)
            else:
                passes = any(self._matches(item, f) for f in filters)
            if passes:
                results.append(item)
            if limit and len(results) >= limit:
                break

        return SkillResult.ok(
            data={
                "results": results,
                "count": len(results),
                "total_input": len(items),
            },
            metadata={"logic": logic, "filter_count": len(filters), "truncated": bool(limit)},
        )


# ---------------------------------------------------------------------------
# TransformDataSkill
# ---------------------------------------------------------------------------


class TransformDataSkill(BaseSkill):
    """Apply map, reduce, sort, or group-by operations on a JSON list."""

    @property
    def name(self) -> str:
        return "transform_data"

    @property
    def description(self) -> str:
        return (
            "Transform a JSON-encoded list using operations: "
            "sort, reverse, unique, pluck (extract a field), group_by, flatten, or count."
        )

    @property
    def category(self) -> str:
        return "data"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter("data", "string", "JSON-encoded list to transform.", required=True),
            SkillParameter(
                "operation",
                "string",
                (
                    "Operation to apply: "
                    "sort | reverse | unique | pluck | group_by | flatten | count | head | tail."
                ),
                required=True,
            ),
            SkillParameter(
                "field",
                "string",
                "Field name used by sort, pluck, and group_by operations.",
                required=False,
                default="",
            ),
            SkillParameter(
                "ascending",
                "boolean",
                "Sort order for 'sort' operation (default: ascending).",
                required=False,
                default=True,
            ),
            SkillParameter(
                "n",
                "integer",
                "Number of items for 'head' or 'tail' operations.",
                required=False,
                default=10,
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not isinstance(params.get("data"), str):
            raise ValueError("'data' must be a JSON string.")
        valid_ops = {"sort", "reverse", "unique", "pluck", "group_by", "flatten", "count", "head", "tail"}
        op = params.get("operation", "")
        if op not in valid_ops:
            raise ValueError(f"'operation' must be one of: {sorted(valid_ops)}")
        # pluck and group_by always need a field; sort only needs one for dict lists
        if op in ("pluck", "group_by") and not params.get("field", "").strip():
            raise ValueError(f"'field' is required for operation '{op}'.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        data_str: str = params["data"]
        operation: str = params["operation"]
        field: str = params.get("field", "").strip()
        ascending = bool(params.get("ascending", True))
        n = int(params.get("n", 10))

        try:
            items: Any = json.loads(data_str)
            if not isinstance(items, list):
                return SkillResult.failure(error="Input 'data' must be a JSON array.")
        except json.JSONDecodeError as exc:
            return SkillResult.failure(error=f"Invalid input JSON: {exc}")

        try:
            if operation == "count":
                return SkillResult.ok(data={"count": len(items)})

            if operation == "reverse":
                return SkillResult.ok(data={"result": list(reversed(items)), "count": len(items)})

            if operation == "head":
                return SkillResult.ok(data={"result": items[:n], "count": min(n, len(items))})

            if operation == "tail":
                return SkillResult.ok(data={"result": items[-n:], "count": min(n, len(items))})

            if operation == "flatten":
                def _flatten(lst: List[Any]) -> List[Any]:
                    out = []
                    for item in lst:
                        if isinstance(item, list):
                            out.extend(_flatten(item))
                        else:
                            out.append(item)
                    return out

                result = _flatten(items)
                return SkillResult.ok(data={"result": result, "count": len(result)})

            if operation == "unique":
                seen = []
                seen_set: list = []
                for item in items:
                    key = json.dumps(item, sort_keys=True, default=str)
                    if key not in seen_set:
                        seen_set.append(key)
                        seen.append(item)
                return SkillResult.ok(data={"result": seen, "count": len(seen), "removed": len(items) - len(seen)})

            if operation == "pluck":
                result = [
                    row.get(field)
                    for row in items
                    if isinstance(row, dict)
                ]
                return SkillResult.ok(data={"result": result, "field": field, "count": len(result)})

            if operation == "sort":
                def sort_key(item: Any) -> Any:
                    if isinstance(item, dict):
                        val = item.get(field)
                    else:
                        val = item
                    if val is None:
                        return ("", 0)
                    try:
                        return (0, float(val))
                    except (TypeError, ValueError):
                        return (1, str(val))

                sorted_items = sorted(items, key=sort_key, reverse=not ascending)
                return SkillResult.ok(
                    data={"result": sorted_items, "count": len(sorted_items)},
                    metadata={"field": field, "ascending": ascending},
                )

            if operation == "group_by":
                groups: Dict[str, List[Any]] = {}
                for item in items:
                    key = str(item.get(field, "__missing__")) if isinstance(item, dict) else str(item)
                    groups.setdefault(key, []).append(item)
                result = [
                    {"group": k, "items": v, "count": len(v)}
                    for k, v in sorted(groups.items())
                ]
                return SkillResult.ok(
                    data={"result": result, "group_count": len(result)},
                    metadata={"field": field},
                )

        except Exception as exc:  # noqa: BLE001
            return SkillResult.failure(
                error=f"Transform operation '{operation}' failed: {exc}",
                metadata={"operation": operation, "exception_type": type(exc).__name__},
            )

        return SkillResult.failure(error=f"Unknown operation: '{operation}'")


__all__ = [
    "ParseJSONSkill",
    "ParseCSVSkill",
    "DataSummarySkill",
    "FormatDataSkill",
    "FilterDataSkill",
    "TransformDataSkill",
]
