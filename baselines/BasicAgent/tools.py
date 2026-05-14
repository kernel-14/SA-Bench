from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _trim_output(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n...[output truncated]...\n{tail}"


def _resolve_workspace_path(workspace_root: Path, raw_path: str) -> Path:
    if raw_path.startswith("/home/paper/"):
        return workspace_root / "paper" / raw_path.removeprefix("/home/paper/")
    if raw_path == "/home/paper":
        return workspace_root / "paper"
    if raw_path.startswith("/home/submission/"):
        return workspace_root / "repo" / raw_path.removeprefix("/home/submission/")
    if raw_path == "/home/submission":
        return workspace_root / "repo"
    if raw_path == "/home/blacklist.txt":
        return workspace_root.parent / "blacklist.txt"
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return path_obj
    return workspace_root / raw_path


def _rewrite_home_paths_in_command(workspace_root: Path, command: str) -> str:
    replacements = {
        "/home/paper": str((workspace_root / "paper").resolve()),
        "/home/submission": str((workspace_root / "repo").resolve()),
        "/home/blacklist.txt": str((workspace_root.parent / "blacklist.txt").resolve()),
    }
    rewritten = command
    for source, target in replacements.items():
        rewritten = rewritten.replace(source, target)
    return rewritten


@dataclass(frozen=True)
class ToolResult:
    content: str
    submitted: bool = False
    submission_note: str = ""


class BaseTool:
    def name(self) -> str:
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        raise NotImplementedError

    def execute(self, workspace_root: Path, **kwargs: Any) -> ToolResult:
        raise NotImplementedError


class BashTool(BaseTool):
    def __init__(self, timeout_seconds: int, max_output_chars: int) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def name(self) -> str:
        return "bash"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name(),
                "description": "Run one shell command in the current run workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, workspace_root: Path, **kwargs: Any) -> ToolResult:
        original_command = kwargs["command"]
        command = _rewrite_home_paths_in_command(workspace_root, original_command)
        completed = subprocess.run(
            command,
            cwd=workspace_root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
        )
        output = (
            f"$ {original_command}\n"
            f"[rewritten] {command}\n"
            f"[exit_code] {completed.returncode}\n"
            f"[stdout]\n{completed.stdout}\n"
            f"[stderr]\n{completed.stderr}"
        )
        return ToolResult(content=_trim_output(output, self.max_output_chars))


class ReadFileChunkTool(BaseTool):
    def __init__(self, default_num_lines: int = 120) -> None:
        self.default_num_lines = default_num_lines

    def name(self) -> str:
        return "read_file_chunk"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name(),
                "description": "Read a file in a paginated chunk with line numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "num_lines": {"type": "integer"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, workspace_root: Path, **kwargs: Any) -> ToolResult:
        target = _resolve_workspace_path(workspace_root, kwargs["path"]).resolve()
        if not target.exists():
            return ToolResult(content=f"File not found: {target}")
        display_path = str(target.relative_to(workspace_root)) if target.is_relative_to(workspace_root) else str(target)
        if target.is_dir():
            entries = sorted(str(p.relative_to(target)) for p in target.iterdir())
            return ToolResult(content=f"[directory] {display_path}\n" + "\n".join(entries))
        start_line = int(kwargs.get("start_line", 1))
        num_lines = int(kwargs.get("num_lines", self.default_num_lines))
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        begin = max(start_line - 1, 0)
        end = min(begin + num_lines, len(lines))
        chunk_lines = [f"{index + 1}: {lines[index]}" for index in range(begin, end)]
        return ToolResult(
            content=f"[file] {display_path}\n" + "\n".join(chunk_lines)
        )


class SearchFileTool(BaseTool):
    def __init__(self, max_output_chars: int) -> None:
        self.max_output_chars = max_output_chars

    def name(self) -> str:
        return "search_file"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name(),
                "description": "Search for a pattern in a file or directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["pattern", "path"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, workspace_root: Path, **kwargs: Any) -> ToolResult:
        pattern = kwargs["pattern"]
        resolved_path = _resolve_workspace_path(workspace_root, kwargs["path"])
        rg_path = shutil.which("rg")
        if rg_path:
            command = [rg_path, "-n", pattern, str(resolved_path)]
        else:
            command = ["grep", "-RIn", pattern, str(resolved_path)]
        completed = subprocess.run(
            command,
            cwd=workspace_root,
            text=True,
            capture_output=True,
            timeout=15,
        )
        output = completed.stdout or completed.stderr or "[no matches]"
        return ToolResult(content=_trim_output(output, self.max_output_chars))


class SubmitTool(BaseTool):
    def name(self) -> str:
        return "submit"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name(),
                "description": "Finish the run and provide a short submission note.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, workspace_root: Path, **kwargs: Any) -> ToolResult:
        del workspace_root
        summary = kwargs["summary"]
        payload = json.dumps({"submitted": True, "summary": summary}, ensure_ascii=False)
        return ToolResult(content=payload, submitted=True, submission_note=summary)
