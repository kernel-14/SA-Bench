from __future__ import annotations

from pathlib import Path

from baselines.BasicAgent.recording import utc_now_text


def format_experiment_entry(case_id: str, model_name: str, run_id: str) -> str:
    return f"| {utc_now_text()} | {case_id} | {model_name} | {run_id} | static-dev | created |"


def ensure_experiment_list(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Experiment List\n\n"
        "| timestamp_utc | case_id | model | run_id | mode | note |\n"
        "| --- | --- | --- | --- | --- | --- |\n",
        encoding="utf-8",
    )


def append_experiment_entry(path: Path, case_id: str, model_name: str, run_id: str) -> None:
    ensure_experiment_list(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(format_experiment_entry(case_id, model_name, run_id) + "\n")
