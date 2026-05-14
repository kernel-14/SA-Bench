"""
Data download script for MA-RLHF experiments.

Downloads and preprocesses the datasets used in the paper:
- OpenAI TL;DR (summarize_from_feedback)
- Anthropic HH-RLHF
- OpenAI WebGPT Comparisons
- APPS (code generation)

Usage:
    python download_data.py --output_dir data/ --datasets tldr hh_rlhf webgpt apps
"""

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_tldr(output_dir: str):
    """
    Download OpenAI TL;DR summarization dataset.
    HuggingFace: openai/summarize_from_feedback
    
    The dataset contains 93k human-annotated preference pairs for training
    and 86k pairs for validation (Section B.1).
    """
    from datasets import load_dataset

    logger.info("Downloading TL;DR dataset...")
    dataset = load_dataset("openai/summarize_from_feedback", "comparisons")

    output_path = os.path.join(output_dir, "tldr_comparisons.jsonl")
    with open(output_path, "w") as f:
        for split in ["train", "validation"]:
            if split in dataset:
                for item in dataset[split]:
                    # Normalize format
                    record = {
                        "info": item.get("info", {}),
                        "chosen": item.get("choice", 0),
                        "summaries": item.get("summaries", []),
                    }
                    # Extract chosen/rejected
                    summaries = item.get("summaries", [])
                    choice = item.get("choice", 0)
                    if len(summaries) >= 2:
                        record["chosen"] = summaries[choice].get("text", "")
                        record["rejected"] = summaries[1 - choice].get("text", "")
                    f.write(json.dumps(record) + "\n")

    logger.info(f"TL;DR saved to {output_path}")


def download_hh_rlhf(output_dir: str):
    """
    Download Anthropic HH-RLHF dataset.
    HuggingFace: Anthropic/hh-rlhf
    
    Contains 112k preference-labeled instances for training,
    12.5k for validation (Section B.1).
    """
    from datasets import load_dataset

    logger.info("Downloading HH-RLHF dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf")

    output_path = os.path.join(output_dir, "hh_rlhf.jsonl")
    with open(output_path, "w") as f:
        for split in ["train", "test"]:
            if split in dataset:
                for item in dataset[split]:
                    record = {
                        "chosen": item.get("chosen", ""),
                        "rejected": item.get("rejected", ""),
                    }
                    f.write(json.dumps(record) + "\n")

    logger.info(f"HH-RLHF saved to {output_path}")


def download_webgpt(output_dir: str):
    """
    Download OpenAI WebGPT Comparisons dataset.
    HuggingFace: openai/webgpt_comparisons
    
    Contains 19.6k instances for training (Section B.1).
    We split 5% for validation.
    """
    from datasets import load_dataset

    logger.info("Downloading WebGPT Comparisons dataset...")
    dataset = load_dataset("openai/webgpt_comparisons")

    output_path = os.path.join(output_dir, "webgpt_comparisons.jsonl")
    with open(output_path, "w") as f:
        for split in ["train"]:
            if split in dataset:
                for item in dataset[split]:
                    record = {
                        "question": item.get("question", {}),
                        "answer_0": item.get("answer_0", ""),
                        "answer_1": item.get("answer_1", ""),
                        "score_0": item.get("score_0", 0),
                        "score_1": item.get("score_1", 0),
                    }
                    f.write(json.dumps(record) + "\n")

    logger.info(f"WebGPT saved to {output_path}")


def download_apps(output_dir: str):
    """
    Download APPS (Automated Programming Progress Standard) dataset.
    HuggingFace: codeparrot/apps
    
    Contains 5k training and 5k test instances (Section B.1).
    """
    from datasets import load_dataset

    logger.info("Downloading APPS dataset...")
    dataset = load_dataset("codeparrot/apps")

    for split, filename in [("train", "apps_train.jsonl"), ("test", "apps_test.jsonl")]:
        if split in dataset:
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w") as f:
                for item in dataset[split]:
                    record = {
                        "question": item.get("question", ""),
                        "solutions": item.get("solutions", "[]"),
                        "input_output": item.get("input_output", "{}"),
                        "difficulty": item.get("difficulty", "interview"),
                        "url": item.get("url", ""),
                    }
                    f.write(json.dumps(record) + "\n")
            logger.info(f"APPS {split} saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/", help="Output directory")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["tldr", "hh_rlhf", "webgpt", "apps"],
        choices=["tldr", "hh_rlhf", "webgpt", "apps"],
        help="Datasets to download",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    download_fns = {
        "tldr": download_tldr,
        "hh_rlhf": download_hh_rlhf,
        "webgpt": download_webgpt,
        "apps": download_apps,
    }

    for dataset_name in args.datasets:
        try:
            download_fns[dataset_name](args.output_dir)
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")

    logger.info("Download complete!")


if __name__ == "__main__":
    main()
