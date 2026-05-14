from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from baselines.BasicAgent.agent import BasicAgent
from baselines.BasicAgent.cases import hydrate_workspace, load_case_files
from baselines.BasicAgent.config import AppConfig
from baselines.BasicAgent.experiment_list import append_experiment_entry
from baselines.BasicAgent.models import RunPaths
from baselines.BasicAgent.prompts import build_system_prompt, build_user_prompt
from baselines.BasicAgent.recording import append_text, build_run_id, ensure_run_dirs, write_json
from baselines.BasicAgent.specs import build_case_spec, write_spec_file
from baselines.BasicAgent.tools import BashTool, ReadFileChunkTool, SearchFileTool, SubmitTool


def _default_status_payload() -> dict[str, Any]:
    return {"cases": {}}


def _update_status(config: AppConfig, case_id: str, run_id: str, status: str, run_dir: Path) -> None:
    import fcntl
    import json

    status_path = config.records_dir / "paper_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            content = fh.read()
            if content.strip():
                payload = json.loads(content)
            else:
                payload = _default_status_payload()
            payload["cases"][case_id] = {
                "status": status,
                "run_id": run_id,
                "run_dir": str(run_dir.relative_to(config.repo_root)),
            }
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def run_case(
    case_id: str,
    config: AppConfig,
    dry_run: bool = False,
) -> Path:
    case = load_case_files(config.data_dir, case_id)
    run_id = build_run_id(case_id)
    run_paths = RunPaths.from_root(config.runs_dir / case_id / run_id)
    ensure_run_dirs(run_paths)
    hydrate_workspace(case, run_paths)

    spec_payload = build_case_spec(case, config)
    write_spec_file(config.specs_dir / f"{case_id}.json", spec_payload)
    append_experiment_entry(config.experiment_list_path, case_id, config.model_name, run_id)
    _update_status(config, case_id, run_id, "initialized", run_paths.root)
    case_log_path = config.logs_dir / f"{case_id}.log"
    append_text(case_log_path, f"[run-start] {run_id} dry_run={dry_run}\n")

    meta_payload = {
        "case_id": case_id,
        "run_id": run_id,
        "model_name": config.model_name,
        "api_base_url": config.api_base_url,
        "mode": "static-dev",
        "dry_run": dry_run,
    }
    write_json(run_paths.meta_json, meta_payload)

    if dry_run:
        write_json(
            run_paths.submission_json,
            {"submitted": False, "summary": "dry run only", "repo_dir": str(run_paths.repo)},
        )
        _update_status(config, case_id, run_id, "dry-run", run_paths.root)
        append_text(case_log_path, f"[run-end] {run_id} status=dry-run\n")
        return run_paths.root

    if not config.api_key:
        raise ValueError("OPENAI_API_KEY is required for non-dry runs")

    client = OpenAI(api_key=config.api_key, base_url=config.api_base_url)
    prompt_messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(case, run_paths)},
    ]
    agent = BasicAgent(
        client=client,
        model_name=config.model_name,
        tools=[
            BashTool(
                timeout_seconds=config.command_timeout_seconds,
                max_output_chars=config.max_output_chars,
            ),
            ReadFileChunkTool(),
            SearchFileTool(max_output_chars=config.max_output_chars),
            SubmitTool(),
        ],
        max_steps=config.max_steps,
        time_limit_seconds=config.time_limit_seconds,
        messages_path=run_paths.messages_jsonl,
        agent_log_path=run_paths.agent_log,
    )
    wall_start = time.time()
    result = agent.run(run_paths.workspace, prompt_messages)
    wall_elapsed = round(time.time() - wall_start, 1)

    # claude-sonnet-4-6 pricing: $3/M input, $15/M output
    cost_usd = (result.total_input_tokens / 1e6) * 3.0 + (result.total_output_tokens / 1e6) * 15.0

    write_json(
        run_paths.submission_json,
        {
            "submitted": result.submitted,
            "summary": result.submission_note,
            "steps": result.steps,
            "repo_dir": str(run_paths.repo),
        },
    )
    # update meta with usage stats
    meta_payload["usage"] = {
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
        "total_tokens": result.total_input_tokens + result.total_output_tokens,
        "estimated_cost_usd": round(cost_usd, 4),
        "elapsed_seconds": wall_elapsed,
        "steps": result.steps,
        "step_timings": result.step_timings,
    }
    write_json(run_paths.meta_json, meta_payload)

    final_status = "submitted" if result.submitted else "stopped"
    _update_status(config, case_id, run_id, final_status, run_paths.root)
    append_text(
        case_log_path,
        f"[run-end] {run_id} status={final_status} steps={result.steps} "
        f"elapsed={wall_elapsed}s tokens={result.total_input_tokens}+{result.total_output_tokens} "
        f"cost=${cost_usd:.4f}\n",
    )
    print(
        f"[{case_id}] done: {result.steps} steps, {wall_elapsed}s, "
        f"{result.total_input_tokens:,}+{result.total_output_tokens:,} tokens, "
        f"~${cost_usd:.3f}"
    )
    return run_paths.root
