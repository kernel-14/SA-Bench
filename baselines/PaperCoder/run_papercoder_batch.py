"""Batch runner for PaperCoder — runs the 3-stage pipeline on all papers.

Usage:
  PAPERCODER_RUNS_ROOT=experiments/runs/deepseek_papercoder \\
    conda run -n sa-bench python baselines/PaperCoder/run_papercoder_batch.py \\
    --model deepseek-v4-pro \\
    --api-base https://api.deepseek.com \\
    --api-key-env DEEPSEEK_API_KEY \\
    --concurrency 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

CODES_DIR = Path(__file__).resolve().parent / "codes"
S2ORC_DIR = PROJECT_ROOT / "data" / "s2orc_jsons"


def paper_already_done(runs_root: Path, paper_id: str) -> bool:
    repo_dir = runs_root / paper_id / f"{paper_id}_repo"
    if not repo_dir.exists():
        return False
    py_files = list(repo_dir.glob("**/*.py"))
    return len(py_files) > 0


def run_paper(paper_id: str, runs_root: Path, model: str, api_base: str, api_key: str) -> dict:
    """Run the full PaperCoder pipeline for a single paper."""
    t0 = time.time()

    paper_dir = runs_root / paper_id
    output_dir = paper_dir
    output_repo_dir = paper_dir / f"{paper_id}_repo"
    s2orc_json = S2ORC_DIR / f"{paper_id}.json"
    cleaned_json = paper_dir / f"{paper_id}_cleaned.json"

    if not s2orc_json.exists():
        return {"paper_id": paper_id, "status": "error", "error": f"S2ORC JSON not found: {s2orc_json}"}

    paper_dir.mkdir(parents=True, exist_ok=True)
    output_repo_dir.mkdir(parents=True, exist_ok=True)

    # Log file
    log_path = paper_dir / "run.log"

    def log(msg: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as lf:
            lf.write(f"[{timestamp}] {msg}\n")

    log(f"Starting PaperCoder for {paper_id}")

    try:
        # Step 0: Preprocess (clean S2ORC JSON)
        log("Step 0: Preprocessing S2ORC JSON")
        subprocess.run(
            [sys.executable, str(CODES_DIR / "0_pdf_process.py"),
             "--input_json_path", str(s2orc_json),
             "--output_json_path", str(cleaned_json)],
            check=True, capture_output=True, text=True, timeout=60
        )

        # Step 1: Planning
        log("Step 1: Planning")
        subprocess.run(
            [sys.executable, str(CODES_DIR / "1_planning.py"),
             "--paper_name", paper_id,
             "--gpt_version", model,
             "--pdf_json_path", str(cleaned_json),
             "--output_dir", str(output_dir),
             "--api_base", api_base,
             "--api_key", api_key],
            check=True, capture_output=True, text=True, timeout=1800
        )

        # Step 1.1: Extract config
        log("Step 1.1: Extract config")
        subprocess.run(
            [sys.executable, str(CODES_DIR / "1.1_extract_config.py"),
             "--paper_name", paper_id,
             "--output_dir", str(output_dir)],
            check=True, capture_output=True, text=True, timeout=30
        )

        config_yaml = output_dir / "planning_config.yaml"
        if config_yaml.exists():
            import shutil
            shutil.copy(str(config_yaml), str(output_repo_dir / "config.yaml"))

        # Step 2: Analysis
        log("Step 2: Analysis")
        subprocess.run(
            [sys.executable, str(CODES_DIR / "2_analyzing.py"),
             "--paper_name", paper_id,
             "--gpt_version", model,
             "--pdf_json_path", str(cleaned_json),
             "--output_dir", str(output_dir),
             "--api_base", api_base,
             "--api_key", api_key],
            check=True, capture_output=True, text=True, timeout=3600
        )

        # Step 3: Coding
        log("Step 3: Coding")
        subprocess.run(
            [sys.executable, str(CODES_DIR / "3_coding.py"),
             "--paper_name", paper_id,
             "--gpt_version", model,
             "--pdf_json_path", str(cleaned_json),
             "--output_dir", str(output_dir),
             "--output_repo_dir", str(output_repo_dir),
             "--api_base", api_base,
             "--api_key", api_key],
            check=True, capture_output=True, text=True, timeout=3600
        )

        elapsed = time.time() - t0
        py_files = list(output_repo_dir.glob("**/*.py"))
        log(f"Completed in {elapsed:.0f}s, {len(py_files)} .py files generated")
        return {"paper_id": paper_id, "status": "success", "elapsed": elapsed, "py_files": len(py_files)}

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        error_msg = e.stderr[-500:] if e.stderr else str(e)
        log(f"ERROR after {elapsed:.0f}s: {error_msg}")
        return {"paper_id": paper_id, "status": "error", "elapsed": elapsed, "error": error_msg}
    except Exception as e:
        elapsed = time.time() - t0
        log(f"EXCEPTION after {elapsed:.0f}s: {e}")
        return {"paper_id": paper_id, "status": "error", "elapsed": elapsed, "error": str(e)}


def list_all_papers() -> list[str]:
    return sorted([p.stem for p in S2ORC_DIR.glob("*.json")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY",
                        help="Env var name for the API key")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--runs-root", type=str, default=None)
    parser.add_argument("--papers", type=str, default=None,
                        help="Comma-separated paper IDs")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run even if output exists")
    args = parser.parse_args()

    api_base = args.api_base or os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(f"ERROR: API key not found in env var {args.api_key_env}")
        sys.exit(1)

    runs_root = Path(args.runs_root) if args.runs_root else (
        PROJECT_ROOT / "experiments" / "runs" / f"{args.model}_papercoder")
    runs_root.mkdir(parents=True, exist_ok=True)

    if args.papers:
        paper_ids = [p.strip() for p in args.papers.split(",")]
    else:
        paper_ids = list_all_papers()

    to_run = []
    skipped = 0
    for pid in paper_ids:
        if paper_already_done(runs_root, pid) and not args.no_skip:
            skipped += 1
            continue
        to_run.append(pid)

    print(f"Model: {args.model}")
    print(f"API base: {api_base}")
    print(f"Papers: {len(paper_ids)} total, {len(to_run)} to run, {skipped} skipped")
    print(f"Concurrency: {args.concurrency}")
    print(f"Runs root: {runs_root}")
    print(f"{'='*60}")

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_paper, pid, runs_root, args.model, api_base, api_key): pid
            for pid in to_run
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status_icon = "OK" if result["status"] == "success" else "FAIL"
            print(f"  [{status_icon}] {result['paper_id']} "
                  f"({result.get('elapsed', 0):.0f}s)"
                  f"{' — ' + str(result.get('py_files', '')) + ' py files' if result['status'] == 'success' else ''}")

    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] != "success")
    print(f"\n{'='*60}")
    print(f"Done. {succeeded} succeeded, {failed} failed, {skipped} skipped.")


if __name__ == "__main__":
    main()
