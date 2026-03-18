"""
File operation skills for JARVIS AI OS.

All path inputs are validated against directory traversal and absolute-path
injection.  By default, operations are restricted to the current working
directory tree; pass ``allowed_root`` to a :class:`BaseSkill` subclass
init to extend this.

Skills provided:
- :class:`ReadFileSkill`       – Read file contents with encoding detection
- :class:`WriteFileSkill`      – Write or append to a file
- :class:`ListFilesSkill`      – List directory contents with filtering
- :class:`DeleteFileSkill`     – Delete a file (requires confirmation flag)
- :class:`CopyMoveFileSkill`   – Copy or move files/directories
- :class:`SearchInFilesSkill`  – Grep-like search across files
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from infrastructure.logger import get_logger
from skills.base_skill import BaseSkill, SkillParameter, SkillResult

logger = get_logger(__name__)

# Default maximum file size to read in one call (10 MB).
_MAX_READ_BYTES = 10 * 1024 * 1024
_SAFE_ROOT = Path.cwd()


# ---------------------------------------------------------------------------
# Path-safety helper
# ---------------------------------------------------------------------------


def _resolve_safe(path_str: str, root: Optional[Path] = None) -> Path:
    """Resolve *path_str* relative to *root* and ensure it stays within *root*.

    Args:
        path_str: User-supplied path string.
        root: Allowed filesystem root.  Defaults to :func:`Path.cwd`.

    Returns:
        A resolved :class:`Path` that is a child of *root*.

    Raises:
        ValueError: If the resolved path escapes *root* (directory traversal).
    """
    allowed = (root or _SAFE_ROOT).resolve()
    candidate = (allowed / path_str).resolve()
    if not str(candidate).startswith(str(allowed)):
        raise ValueError(
            f"Path '{path_str}' resolves outside the allowed directory '{allowed}'."
        )
    return candidate


def _detect_encoding(raw: bytes) -> str:
    """Heuristic encoding detection without chardet.

    Returns UTF-8 for BOM/valid sequences, falls back to latin-1.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


# ---------------------------------------------------------------------------
# ReadFileSkill
# ---------------------------------------------------------------------------


class ReadFileSkill(BaseSkill):
    """Read and return the text contents of a file."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file and return it as text. "
            "Automatically detects encoding.  Supports optional line range."
        )

    @property
    def category(self) -> str:
        return "file"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter("path", "string", "Relative path to the file to read.", required=True),
            SkillParameter(
                "start_line",
                "integer",
                "First line to return (1-based, inclusive). 0 = beginning.",
                required=False,
                default=0,
            ),
            SkillParameter(
                "end_line",
                "integer",
                "Last line to return (inclusive). 0 = end of file.",
                required=False,
                default=0,
            ),
            SkillParameter(
                "encoding",
                "string",
                "Force a specific encoding (e.g. 'utf-8'). Auto-detected when empty.",
                required=False,
                default="",
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not params.get("path", "").strip():
            raise ValueError("'path' must be a non-empty string.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        path_str = params["path"].strip()
        start_line = int(params.get("start_line", 0))
        end_line = int(params.get("end_line", 0))
        forced_encoding: str = params.get("encoding", "").strip()

        try:
            file_path = _resolve_safe(path_str)
        except ValueError as exc:
            return SkillResult.failure(error=str(exc))

        if not file_path.exists():
            return SkillResult.failure(error=f"File not found: '{path_str}'")
        if not file_path.is_file():
            return SkillResult.failure(error=f"Path is not a regular file: '{path_str}'")

        file_size = file_path.stat().st_size
        if file_size > _MAX_READ_BYTES:
            return SkillResult.failure(
                error=(
                    f"File size {file_size} bytes exceeds limit {_MAX_READ_BYTES} bytes. "
                    "Use start_line/end_line to read portions."
                ),
                metadata={"file_size": file_size},
            )

        try:
            raw = file_path.read_bytes()
            encoding = forced_encoding or _detect_encoding(raw)
            content = raw.decode(encoding, errors="replace")
        except OSError as exc:
            return SkillResult.failure(error=f"Failed to read file: {exc}")

        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        if start_line or end_line:
            s = max(0, start_line - 1) if start_line else 0
            e = end_line if end_line else total_lines
            lines = lines[s:e]
            content = "".join(lines)

        return SkillResult.ok(
            data={"path": str(file_path), "content": content, "lines": len(lines)},
            metadata={
                "total_lines": total_lines,
                "file_size": file_size,
                "encoding": encoding,
            },
        )


# ---------------------------------------------------------------------------
# WriteFileSkill
# ---------------------------------------------------------------------------


class WriteFileSkill(BaseSkill):
    """Write or append text content to a file, creating it if necessary."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write text content to a file. Can overwrite or append. "
            "Creates the file (and missing parent directories) automatically."
        )

    @property
    def category(self) -> str:
        return "file"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter("path", "string", "Relative path for the destination file.", required=True),
            SkillParameter("content", "string", "Text content to write.", required=True),
            SkillParameter(
                "mode",
                "string",
                "Write mode: 'overwrite' (default) or 'append'.",
                required=False,
                default="overwrite",
            ),
            SkillParameter(
                "encoding",
                "string",
                "File encoding (default: 'utf-8').",
                required=False,
                default="utf-8",
            ),
            SkillParameter(
                "create_dirs",
                "boolean",
                "Automatically create missing parent directories.",
                required=False,
                default=True,
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not params.get("path", "").strip():
            raise ValueError("'path' must be a non-empty string.")
        if "content" not in params:
            raise ValueError("'content' is required.")
        mode = params.get("mode", "overwrite")
        if mode not in ("overwrite", "append"):
            raise ValueError("'mode' must be 'overwrite' or 'append'.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        path_str = params["path"].strip()
        content: str = params["content"]
        mode = params.get("mode", "overwrite")
        encoding = params.get("encoding", "utf-8")
        create_dirs = bool(params.get("create_dirs", True))

        try:
            file_path = _resolve_safe(path_str)
        except ValueError as exc:
            return SkillResult.failure(error=str(exc))

        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        elif not file_path.parent.exists():
            return SkillResult.failure(
                error=f"Parent directory '{file_path.parent}' does not exist."
            )

        open_mode = "w" if mode == "overwrite" else "a"
        try:
            with file_path.open(open_mode, encoding=encoding) as fh:
                fh.write(content)
        except OSError as exc:
            return SkillResult.failure(error=f"Failed to write file: {exc}")

        return SkillResult.ok(
            data={"path": str(file_path), "bytes_written": len(content.encode(encoding))},
            metadata={"mode": mode, "encoding": encoding},
        )


# ---------------------------------------------------------------------------
# ListFilesSkill
# ---------------------------------------------------------------------------


class ListFilesSkill(BaseSkill):
    """List files and directories within a given path."""

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return (
            "List files and directories at the given path. "
            "Supports glob-style pattern filtering and recursive listing."
        )

    @property
    def category(self) -> str:
        return "file"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter(
                "path",
                "string",
                "Directory path to list. Defaults to current directory.",
                required=False,
                default=".",
            ),
            SkillParameter(
                "pattern",
                "string",
                "Glob pattern to filter entries (e.g. '*.py', '*.txt').",
                required=False,
                default="*",
            ),
            SkillParameter(
                "recursive",
                "boolean",
                "Recursively list subdirectories.",
                required=False,
                default=False,
            ),
            SkillParameter(
                "include_hidden",
                "boolean",
                "Include hidden files/directories (starting with '.').",
                required=False,
                default=False,
            ),
            SkillParameter(
                "max_entries",
                "integer",
                "Maximum number of entries to return (1–1000).",
                required=False,
                default=100,
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        max_entries = params.get("max_entries", 100)
        if not isinstance(max_entries, int) or not (1 <= max_entries <= 1000):
            raise ValueError("'max_entries' must be between 1 and 1000.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        path_str = params.get("path", ".").strip() or "."
        pattern = params.get("pattern", "*") or "*"
        recursive = bool(params.get("recursive", False))
        include_hidden = bool(params.get("include_hidden", False))
        max_entries = int(params.get("max_entries", 100))

        try:
            dir_path = _resolve_safe(path_str)
        except ValueError as exc:
            return SkillResult.failure(error=str(exc))

        if not dir_path.exists():
            return SkillResult.failure(error=f"Directory not found: '{path_str}'")
        if not dir_path.is_dir():
            return SkillResult.failure(error=f"Path is not a directory: '{path_str}'")

        glob_fn = dir_path.rglob if recursive else dir_path.glob
        entries = []
        for entry in sorted(glob_fn(pattern)):
            if not include_hidden and entry.name.startswith("."):
                continue
            try:
                stat = entry.stat()
                entries.append({
                    "name": entry.name,
                    "path": str(entry.relative_to(_SAFE_ROOT)),
                    "type": "directory" if entry.is_dir() else "file",
                    "size": stat.st_size if entry.is_file() else None,
                    "modified": stat.st_mtime,
                })
            except OSError:
                continue

            if len(entries) >= max_entries:
                break

        return SkillResult.ok(
            data={"path": str(dir_path), "entries": entries, "count": len(entries)},
            metadata={"pattern": pattern, "recursive": recursive, "truncated": len(entries) >= max_entries},
        )


# ---------------------------------------------------------------------------
# DeleteFileSkill
# ---------------------------------------------------------------------------


class DeleteFileSkill(BaseSkill):
    """Delete a file or empty directory (requires explicit confirmation)."""

    @property
    def name(self) -> str:
        return "delete_file"

    @property
    def description(self) -> str:
        return (
            "Permanently delete a file or empty directory. "
            "Requires 'confirm=true' to execute as a safety guard."
        )

    @property
    def category(self) -> str:
        return "file"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter("path", "string", "Path to the file or empty directory to delete.", required=True),
            SkillParameter(
                "confirm",
                "boolean",
                "Must be true to confirm deletion. Prevents accidental deletes.",
                required=True,
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not params.get("path", "").strip():
            raise ValueError("'path' must be a non-empty string.")
        if not params.get("confirm"):
            raise ValueError("'confirm' must be true to proceed with deletion.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        path_str = params["path"].strip()

        try:
            target = _resolve_safe(path_str)
        except ValueError as exc:
            return SkillResult.failure(error=str(exc))

        if not target.exists():
            return SkillResult.failure(error=f"Path not found: '{path_str}'")

        try:
            if target.is_dir():
                target.rmdir()
                deleted_type = "directory"
            else:
                target.unlink()
                deleted_type = "file"
        except OSError as exc:
            return SkillResult.failure(error=f"Failed to delete: {exc}")

        logger.info("DeleteFileSkill: deleted %s '%s'", deleted_type, target)
        return SkillResult.ok(
            data={"deleted": str(target), "type": deleted_type},
            metadata={"path": path_str},
        )


# ---------------------------------------------------------------------------
# CopyMoveFileSkill
# ---------------------------------------------------------------------------


class CopyMoveFileSkill(BaseSkill):
    """Copy or move a file or directory to a new location."""

    @property
    def name(self) -> str:
        return "copy_move_file"

    @property
    def description(self) -> str:
        return (
            "Copy or move a file or directory from a source path to a destination path. "
            "Use 'operation=copy' or 'operation=move'."
        )

    @property
    def category(self) -> str:
        return "file"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter("source", "string", "Source file or directory path.", required=True),
            SkillParameter("destination", "string", "Destination path.", required=True),
            SkillParameter(
                "operation",
                "string",
                "Operation to perform: 'copy' or 'move'.",
                required=True,
            ),
            SkillParameter(
                "overwrite",
                "boolean",
                "Overwrite destination if it already exists.",
                required=False,
                default=False,
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not params.get("source", "").strip():
            raise ValueError("'source' must be a non-empty string.")
        if not params.get("destination", "").strip():
            raise ValueError("'destination' must be a non-empty string.")
        if params.get("operation") not in ("copy", "move"):
            raise ValueError("'operation' must be 'copy' or 'move'.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        src_str = params["source"].strip()
        dst_str = params["destination"].strip()
        operation = params["operation"]
        overwrite = bool(params.get("overwrite", False))

        try:
            src = _resolve_safe(src_str)
            dst = _resolve_safe(dst_str)
        except ValueError as exc:
            return SkillResult.failure(error=str(exc))

        if not src.exists():
            return SkillResult.failure(error=f"Source not found: '{src_str}'")
        if dst.exists() and not overwrite:
            return SkillResult.failure(
                error=f"Destination '{dst_str}' already exists. Use overwrite=true."
            )

        try:
            if operation == "copy":
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=overwrite)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            else:  # move
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
        except (OSError, shutil.Error) as exc:
            return SkillResult.failure(error=f"Operation failed: {exc}")

        logger.info("CopyMoveFileSkill: %s '%s' -> '%s'", operation, src, dst)
        return SkillResult.ok(
            data={"operation": operation, "source": str(src), "destination": str(dst)},
        )


# ---------------------------------------------------------------------------
# SearchInFilesSkill
# ---------------------------------------------------------------------------


class SearchInFilesSkill(BaseSkill):
    """Search for a text pattern across multiple files (grep-like)."""

    @property
    def name(self) -> str:
        return "search_in_files"

    @property
    def description(self) -> str:
        return (
            "Search for a text pattern in files within a directory. "
            "Supports plain-text and regular-expression patterns."
        )

    @property
    def category(self) -> str:
        return "file"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        params = [
            SkillParameter("pattern", "string", "Text or regex pattern to search for.", required=True),
            SkillParameter(
                "path",
                "string",
                "Directory to search in. Defaults to current directory.",
                required=False,
                default=".",
            ),
            SkillParameter(
                "file_pattern",
                "string",
                "Glob pattern for file names to include (e.g. '*.py').",
                required=False,
                default="*",
            ),
            SkillParameter(
                "use_regex",
                "boolean",
                "Treat 'pattern' as a regular expression.",
                required=False,
                default=False,
            ),
            SkillParameter(
                "case_sensitive",
                "boolean",
                "Perform case-sensitive matching.",
                required=False,
                default=False,
            ),
            SkillParameter(
                "max_matches",
                "integer",
                "Maximum total matches to return (1–500).",
                required=False,
                default=50,
            ),
            SkillParameter(
                "context_lines",
                "integer",
                "Number of context lines to include around each match (0–5).",
                required=False,
                default=0,
            ),
        ]
        return self._build_schema(params)

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not params.get("pattern", "").strip():
            raise ValueError("'pattern' must be a non-empty string.")
        max_matches = params.get("max_matches", 50)
        if not isinstance(max_matches, int) or not (1 <= max_matches <= 500):
            raise ValueError("'max_matches' must be between 1 and 500.")
        context_lines = params.get("context_lines", 0)
        if not isinstance(context_lines, int) or not (0 <= context_lines <= 5):
            raise ValueError("'context_lines' must be between 0 and 5.")

    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        pattern_str = params["pattern"].strip()
        path_str = params.get("path", ".").strip() or "."
        file_pattern = params.get("file_pattern", "*") or "*"
        use_regex = bool(params.get("use_regex", False))
        case_sensitive = bool(params.get("case_sensitive", False))
        max_matches = int(params.get("max_matches", 50))
        context_lines = int(params.get("context_lines", 0))

        try:
            search_dir = _resolve_safe(path_str)
        except ValueError as exc:
            return SkillResult.failure(error=str(exc))

        if not search_dir.is_dir():
            return SkillResult.failure(error=f"Not a directory: '{path_str}'")

        # Compile regex
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            if use_regex:
                compiled = re.compile(pattern_str, flags)
            else:
                compiled = re.compile(re.escape(pattern_str), flags)
        except re.error as exc:
            return SkillResult.failure(error=f"Invalid regex pattern: {exc}")

        matches: List[Dict[str, Any]] = []
        files_searched = 0

        for file_path in sorted(search_dir.rglob(file_pattern)):
            if not file_path.is_file():
                continue
            if file_path.stat().st_size > _MAX_READ_BYTES:
                continue
            try:
                raw = file_path.read_bytes()
                encoding = _detect_encoding(raw)
                lines = raw.decode(encoding, errors="replace").splitlines()
            except OSError:
                continue

            files_searched += 1
            for line_no, line in enumerate(lines, start=1):
                if compiled.search(line):
                    context_before = lines[max(0, line_no - 1 - context_lines): line_no - 1]
                    context_after = lines[line_no: min(len(lines), line_no + context_lines)]
                    matches.append({
                        "file": str(file_path.relative_to(_SAFE_ROOT)),
                        "line_number": line_no,
                        "line": line,
                        "context_before": context_before,
                        "context_after": context_after,
                    })
                    if len(matches) >= max_matches:
                        break

            if len(matches) >= max_matches:
                break

        return SkillResult.ok(
            data={
                "pattern": pattern_str,
                "matches": matches,
                "total_matches": len(matches),
                "files_searched": files_searched,
            },
            metadata={
                "truncated": len(matches) >= max_matches,
                "search_dir": str(search_dir),
            },
        )


__all__ = [
    "ReadFileSkill",
    "WriteFileSkill",
    "ListFilesSkill",
    "DeleteFileSkill",
    "CopyMoveFileSkill",
    "SearchInFilesSkill",
]
