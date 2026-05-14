from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from baselines.BasicAgent.cases import CaseFiles
from baselines.BasicAgent.config import AppConfig


def build_spec_payload(case_id: str, paper_dir: Path, model_name: str) -> dict[str, Any]:
    return {
        "paper": {
            "case_id": case_id,
            "paper_dir": str(paper_dir),
        },
        "runtime": {
            "model_name": model_name,
            "mode": "static-dev",
        },
    }


def build_case_spec(case: CaseFiles, config: AppConfig) -> dict[str, Any]:
    payload = build_spec_payload(case.case_id, case.paper_dir, config.model_name)
    payload["paper"].update(
        {
            "title": case.title,
            "paper_md": str(case.paper_md.relative_to(config.repo_root)),
            "config_yaml": str(case.config_yaml.relative_to(config.repo_root)),
            "addendum_md": (
                str(case.addendum_md.relative_to(config.repo_root))
                if case.addendum_md
                else None
            ),
            "blacklist_txt": (
                str(case.blacklist_txt.relative_to(config.repo_root))
                if case.blacklist_txt
                else None
            ),
            "rubric_json": (
                str(case.rubric_json.relative_to(config.repo_root))
                if case.rubric_json
                else None
            ),
        }
    )
    payload["runtime"].update(
        {
            "time_limit_seconds": config.time_limit_seconds,
            "max_steps": config.max_steps,
            "command_timeout_seconds": config.command_timeout_seconds,
            "expected_artifacts": ["README.md", "main.py", "requirements.txt"],
        }
    )
    payload["notes"] = [
        "PaperBench-lite static run.",
        "This run mirrors BasicAgent multi-turn tool use but skips reproduction execution.",
        "Judging target is static dev-style repo generation only.",
    ]
    return payload


def write_spec_file(spec_path: Path, payload: dict[str, Any]) -> None:
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
