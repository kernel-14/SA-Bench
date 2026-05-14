from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from baselines.BasicAgent.models import RunPaths


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    paper_dir: Path
    paper_md: Path
    config_yaml: Path
    addendum_md: Path | None
    blacklist_txt: Path | None
    rubric_json: Path | None
    title: str


def _parse_simple_yaml_title(config_yaml: Path) -> str:
    if not config_yaml.exists():
        return config_yaml.parent.name
    for line in config_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("title:"):
            return line.split(":", 1)[1].strip().strip("'").strip('"')
    return config_yaml.parent.name


def load_case_files(data_dir: Path, case_id: str) -> CaseFiles:
    paper_dir = data_dir / case_id
    if not paper_dir.exists():
        raise FileNotFoundError(f"Unknown case: {case_id}")
    paper_md = paper_dir / "paper.md"
    config_yaml = paper_dir / "config.yaml"
    if not paper_md.exists():
        raise FileNotFoundError(f"Missing paper markdown: {paper_md}")
    if not config_yaml.exists():
        raise FileNotFoundError(f"Missing config: {config_yaml}")
    addendum_md = paper_dir / "addendum.md"
    blacklist_txt = paper_dir / "blacklist.txt"
    rubric_json = paper_dir / "rubric.json"
    return CaseFiles(
        case_id=case_id,
        paper_dir=paper_dir,
        paper_md=paper_md,
        config_yaml=config_yaml,
        addendum_md=addendum_md if addendum_md.exists() else None,
        blacklist_txt=blacklist_txt if blacklist_txt.exists() else None,
        rubric_json=rubric_json if rubric_json.exists() else None,
        title=_parse_simple_yaml_title(config_yaml),
    )


def hydrate_workspace(case: CaseFiles, run_paths: RunPaths) -> None:
    run_paths.repo.mkdir(parents=True, exist_ok=True)
    run_paths.inputs.mkdir(parents=True, exist_ok=True)
    files_to_copy = [
        case.paper_md,
        case.config_yaml,
        case.addendum_md,
    ]
    for source in files_to_copy:
        if source is None:
            continue
        shutil.copy2(source, run_paths.inputs / source.name)
    if case.blacklist_txt is not None:
        shutil.copy2(case.blacklist_txt, run_paths.monitoring_blacklist)
