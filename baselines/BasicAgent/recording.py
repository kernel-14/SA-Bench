from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from baselines.BasicAgent.models import RunPaths


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_run_id(case_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{case_id}-{stamp}"


def ensure_run_dirs(run_paths: RunPaths) -> None:
    run_paths.root.mkdir(parents=True, exist_ok=True)
    run_paths.workspace.mkdir(parents=True, exist_ok=True)
    run_paths.repo.mkdir(parents=True, exist_ok=True)
    run_paths.inputs.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
