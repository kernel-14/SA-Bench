from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    repo_root: Path
    data_dir: Path
    experiments_dir: Path
    specs_dir: Path
    runs_dir: Path
    logs_dir: Path
    records_dir: Path
    experiment_list_path: Path
    model_name: str
    api_key: str | None
    api_base_url: str | None
    max_steps: int
    time_limit_seconds: int
    command_timeout_seconds: int
    max_output_chars: int

    @classmethod
    def from_env(
        cls,
        repo_root: str | Path | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        api_base_url: str | None = None,
        max_steps: int = 80,
        time_limit_seconds: int = 900,
        command_timeout_seconds: int = 30,
        max_output_chars: int = 12000,
    ) -> "AppConfig":
        root = Path(repo_root or Path.cwd()).resolve()
        experiments_dir = root / "experiments"
        runs_root = os.environ.get("PAPERBENCH_RUNS_ROOT")
        runs_dir = Path(runs_root).resolve() if runs_root else (experiments_dir / "runs")
        return cls(
            repo_root=root,
            data_dir=root / "data" / "papers",
            experiments_dir=experiments_dir,
            specs_dir=experiments_dir / "specs",
            runs_dir=runs_dir,
            logs_dir=experiments_dir / "logs",
            records_dir=experiments_dir / "records",
            experiment_list_path=experiments_dir / "experiment_list.md",
            model_name=model_name or os.environ.get("OPENAI_MODEL", "gpt-4o"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            api_base_url=api_base_url or os.environ.get("OPENAI_BASE_URL"),
            max_steps=max_steps,
            time_limit_seconds=time_limit_seconds,
            command_timeout_seconds=command_timeout_seconds,
            max_output_chars=max_output_chars,
        )
