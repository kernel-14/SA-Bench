"""
Script for collecting model responses for the de-anonymization experiment.

This script queries 22 models with 8 categories of prompts (200 prompts per category,
50 responses per model per prompt) as described in Section 2.3 of the paper.

Usage:
    python collect_responses.py \
        --config configs/models.yaml \
        --prompts_dir data/prompts \
        --output_dir data/responses \
        --category english
"""

import os
import sys
import json
import argparse
import logging
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_query import collect_responses, get_querier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_prompts(prompts_dir: str, category: str, n_prompts: int = 200) -> list:
    """
    Load prompts for a given category.

    Args:
        prompts_dir: Directory containing prompt files
        category: Prompt category name
        n_prompts: Maximum number of prompts to load

    Returns:
        List of prompt strings
    """
    prompts_path = Path(prompts_dir) / f"{category}.txt"

    if not prompts_path.exists():
        logger.warning(f"Prompts file not found: {prompts_path}")
        return []

    with open(prompts_path) as f:
        prompts = [line.strip() for line in f if line.strip()]

    return prompts[:n_prompts]


def save_responses(
    responses: dict,
    output_dir: str,
    category: str,
    model_name: str,
):
    """
    Save model responses to disk.

    Directory structure:
        output_dir/
            {category}/
                {prompt_id}/
                    {model_name}.json

    Args:
        responses: Dictionary mapping prompt -> list of responses
        output_dir: Output directory
        category: Prompt category
        model_name: Model name
    """
    output_path = Path(output_dir) / category

    for prompt_idx, (prompt, prompt_responses) in enumerate(responses.items()):
        prompt_dir = output_path / f"prompt_{prompt_idx:04d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        # Save prompt text
        prompt_file = prompt_dir / "prompt.txt"
        if not prompt_file.exists():
            prompt_file.write_text(prompt)

        # Save responses
        response_file = prompt_dir / f"{model_name}.json"
        with open(response_file, "w") as f:
            json.dump(prompt_responses, f, indent=2)

    logger.info(
        f"Saved {len(responses)} prompts for model {model_name} "
        f"in category {category}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect model responses for de-anonymization experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default="data/prompts",
        help="Directory containing prompt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/responses",
        help="Directory to save responses",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="english",
        choices=[
            "english", "chinese", "spanish", "indonesian", "persian",
            "coding", "math", "safety_violating"
        ],
        help="Prompt category to collect",
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=200,
        help="Number of prompts per category",
    )
    parser.add_argument(
        "--n_responses",
        type=int,
        default=50,
        help="Number of responses per model per prompt",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens per response",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to query (default: all models in config)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    models = config["models"]
    if args.models:
        models = [m for m in models if m["name"] in args.models]

    # Load prompts
    prompts = load_prompts(args.prompts_dir, args.category, args.n_prompts)
    if not prompts:
        logger.error(
            f"No prompts found for category {args.category}. "
            f"Please add prompts to {args.prompts_dir}/{args.category}.txt"
        )
        return

    logger.info(
        f"Collecting responses for {len(models)} models, "
        f"{len(prompts)} prompts, {args.n_responses} responses each"
    )

    # Collect responses for each model
    for model_config in models:
        model_name = model_config["name"]
        query_method = model_config["query_method"]

        logger.info(f"Collecting responses for model: {model_name}")

        try:
            responses = collect_responses(
                model_name=model_name,
                query_method=query_method,
                prompts=prompts,
                n_responses_per_prompt=args.n_responses,
                max_tokens=args.max_tokens,
            )

            save_responses(
                responses=responses,
                output_dir=args.output_dir,
                category=args.category,
                model_name=model_name,
            )

        except Exception as e:
            logger.error(f"Failed to collect responses for {model_name}: {e}")
            continue

    logger.info("Response collection complete!")


if __name__ == "__main__":
    main()
