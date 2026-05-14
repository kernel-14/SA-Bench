from __future__ import annotations

from pathlib import Path

from baselines.BasicAgent.cases import CaseFiles
from baselines.BasicAgent.models import RunPaths


DEFAULT_CONTINUE_MESSAGE = (
    "Please proceed to the next step using your best judgement. If you believe you are "
    "finished, double check your work to continue to refine and improve your submission."
)


def build_system_prompt() -> str:
    return (
        "You are a helpful agent attempting to solve a task.\n"
        "- You have several functions available to help with solving the task. Each message may "
        "perform one function call. You will see the result of the function right after sending "
        "the message. If you need to perform multiple actions, you can always send more messages "
        "with subsequent function calls.\n"
        "- Do some reasoning before your actions, describing what function calls you are going to "
        "use and how they fit into your plan.\n"
        "- Use as much of the available time as possible to refine and optimize your solution: "
        "Your goal is to achieve the best possible score, not just to submit your first working solution. "
        "Keep working until the time limit expires.\n"
        "- The file read tool is paginated, so you will need to call it multiple times to make sure "
        "that you have read all the relevant parts of the paper.\n"
        "- This is a multi-step task. DO THINGS BIT BY BIT by slowly building up your solution with "
        "multiple tool calls, don't try to do everything in one go. Just reason about what is the "
        "best next step and execute it. Then take the next step after you receive the output of the previous step.\n"
        "- You should try and create a clean codebase. When it makes sense you should edit or overwrite "
        "files rather than creating new ones.\n"
        "- When writing code, do not write newlines as the string \\n, instead use the newline character.\n"
        "- This benchmark variant is static-only: NEVER execute Python scripts, pip install, or run "
        "any code. Your deliverable is the codebase itself, not execution results. Writing correct code is the goal.\n"
        "- Finish only by calling submit.\n\n"
        "Remember, you are tasked with replicating a whole paper, not just a single part of it or a minimal example. "
        "The file read tool is PAGINATED, so you will need to CALL IT MULTIPLE TIMES to make sure that "
        "you have read all the relevant parts of the paper."
    )


def build_user_prompt(case: CaseFiles, run_paths: RunPaths) -> str:
    paper_dir = Path("/home/paper")
    submission_dir = Path("/home/submission")
    blacklist_path = Path("/home/blacklist.txt")
    addendum_line = (
        "There is an addendum file at `/home/paper/addendum.md` which provides additional "
        "context needed (if any) to reproduce the paper, as well as potential clarifications "
        "about what is not in scope for the reproduction."
        if case.addendum_md
        else "No addendum file is provided for this paper."
    )
    return f"""You are tasked with reproducing a research paper.

You have been provided with a research paper located at `{paper_dir}`. The directory contains the paper in markdown format.

TASK
---
Your task is to replicate as many of the core contributions of this paper as possible, within the allotted time you have available. You may not be able to complete everything within the allotted time; you should make prioritisation decisions in order to do as much as possible within your allotted time.

Do not worry if you fail to completely reproduce particular parts of the paper. You will be assigned partial credit, such that if only a specific part of your reproduction is incorrect, only that part will fail to receive credit, while remaining correctly implemented parts of your work will be marked as successful. Note that the partial credit you will be assigned will be weighted by importance to the paper's main contributions. Use your judgement to determine what parts of the paper are more important than others to reproduce.

Any experiments that are only introduced in the Appendix are not considered core contributions of the paper, and so are out of scope. This is the case even if the experiment is referenced in the main text. HOWEVER, if an experiment is described in the main body of the text, but some of the details used have been put in the appendix, then that experiment is still in scope.

RESOURCES
---
{addendum_line}

You are allowed to browse and use resources you can find online, however you must not look at or use the paper's codebase or any other resources listed in `{blacklist_path}`, as this is considered cheating.

SUBMISSION
---
The output of your work must be a directory containing all the necessary components (code, configs, README) to reproduce the paper. Your code should be located at `{submission_dir}`. Do NOT run git init or git add — this wastes time and steps.

For this static benchmark variant, do NOT run any code, pip install, or execute Python scripts. Your deliverable is the source code itself — write the best paper-grounded codebase you can from the text alone. Your submitted repository should still contain a README.md file that describes what you were able to achieve in your reproduction attempt, explains how your codebase relates to various parts of the reproduction, and documents any assumptions or unresolved missing details from the paper.

STRATEGY
---
When possible do not use specific versions of Python packages; you don't want to get stuck solving a dependency conflict.

You should start by reading the paper and the entirety of the addendum; the addendum sometimes contains important information about how to replicate the paper.

LOCAL WORKSPACE MAPPING
---
- `/home/paper` corresponds to `paper/` in the current workspace.
- `{submission_dir}` corresponds to `repo/` in the current workspace.
- `{blacklist_path}` corresponds to `../blacklist.txt` relative to `paper/`.

CASE INFO
---
- Paper ID: `{case.case_id}`
- Title: `{case.title}`
- Run directory: `{run_paths.root}`
"""
