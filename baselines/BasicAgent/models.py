from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunPaths:
    root: Path
    workspace: Path
    repo: Path
    inputs: Path
    monitoring_blacklist: Path
    agent_log: Path
    messages_jsonl: Path
    meta_json: Path
    submission_json: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "RunPaths":
        root_path = Path(root)
        workspace = root_path / "workspace"
        return cls(
            root=root_path,
            workspace=workspace,
            repo=workspace / "repo",
            inputs=workspace / "paper",
            monitoring_blacklist=root_path / "blacklist.txt",
            agent_log=root_path / "agent.log",
            messages_jsonl=root_path / "messages.jsonl",
            meta_json=root_path / "meta.json",
            submission_json=root_path / "submission.json",
        )


@dataclass
class AgentResult:
    submitted: bool
    submission_note: str
    steps: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    elapsed_seconds: float = 0.0
    step_timings: list[dict[str, Any]] = field(default_factory=list)
