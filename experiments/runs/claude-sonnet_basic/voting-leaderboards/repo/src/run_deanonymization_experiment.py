"""
Main script for running the de-anonymization experiment (Section 2).

This script:
1. Collects responses from 22 models using 8 prompt categories
2. Trains training-based detectors using BoW, TF-IDF, and length features
3. Evaluates identity-probing detectors
4. Reports detection accuracy for each model and prompt category

Usage:
    python run_deanonymization_experiment.py --config configs/models.yaml \
        --data_dir data/responses --output_dir results/deanonymization
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training_based_detector import (
    ModelDetector,
    evaluate_detector_across_prompts,
    FEATURE_TYPES,
)
from identity_probing_detector import (
    run_identity_probing_experiment,
    IDENTITY_PROBING_PROMPTS,
    MODEL_KEYWORDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_responses(data_dir: str) -> dict:
    """
    Load pre-collected model responses from disk.

    Expected directory structure:
        data_dir/
            {category}/
                {prompt_id}/
                    {model_name}.json  # List of responses

    Args:
        data_dir: Path to the data directory

    Returns:
        Dictionary: {category: {prompt: {model_name: [responses]}}}
    """
    data = {}
    data_path = Path(data_dir)

    for category_dir in sorted(data_path.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        data[category] = {}

        for prompt_dir in sorted(category_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue

            # Load prompt text
            prompt_file = prompt_dir / "prompt.txt"
            if not prompt_file.exists():
                continue
            prompt = prompt_file.read_text().strip()

            data[category][prompt] = {}

            # Load responses for each model
            for response_file in sorted(prompt_dir.glob("*.json")):
                model_name = response_file.stem
                with open(response_file) as f:
                    responses = json.load(f)
                data[category][prompt][model_name] = responses

    return data


def run_training_based_experiment(
    all_responses: dict,
    target_models: list,
    feature_types: list = None,
    output_dir: str = None,
) -> dict:
    """
    Run the training-based detector experiment.

    Evaluates detectors for each target model across all prompt categories
    and feature types.

    Args:
        all_responses: Dictionary of responses by category/prompt/model
        target_models: List of target model names to evaluate
        feature_types: List of feature types to evaluate
        output_dir: Directory to save results

    Returns:
        Dictionary with results for each model and feature type
    """
    if feature_types is None:
        feature_types = FEATURE_TYPES

    results = {}

    for target_model in target_models:
        results[target_model] = {}
        logger.info(f"Evaluating target model: {target_model}")

        for feature_type in feature_types:
            logger.info(f"  Feature type: {feature_type}")
            category_results = {}

            for category, category_data in all_responses.items():
                # Flatten to prompt -> {model: responses} format
                prompt_responses = {}
                for prompt, model_responses in category_data.items():
                    prompt_responses[prompt] = model_responses

                eval_results = evaluate_detector_across_prompts(
                    target_model=target_model,
                    all_responses=prompt_responses,
                    feature_type=feature_type,
                )
                category_results[category] = eval_results

            results[target_model][feature_type] = category_results

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "training_based_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return results


def run_identity_probing_experiment_all(
    model_responses: dict,
    target_models: list,
    output_dir: str = None,
) -> dict:
    """
    Run the identity-probing experiment for all target models.

    Args:
        model_responses: Dictionary mapping prompt -> list of responses per model
        target_models: List of target model names
        output_dir: Directory to save results

    Returns:
        Dictionary with results for each model
    """
    results = {}

    for target_model in target_models:
        if target_model not in model_responses:
            logger.warning(f"No responses found for model {target_model}")
            continue

        logger.info(f"Running identity-probing for: {target_model}")
        model_results = run_identity_probing_experiment(
            model_responses=model_responses[target_model],
            target_model=target_model,
        )
        results[target_model] = model_results

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "identity_probing_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return results


def print_results_table(results: dict, feature_type: str = "bow"):
    """
    Print results in a format similar to Table 3 in the paper.

    Args:
        results: Results from run_training_based_experiment
        feature_type: Feature type to display
    """
    print(f"\n{'='*80}")
    print(f"Training-based Detector Results (Feature: {feature_type})")
    print(f"{'='*80}")
    print(f"{'Model':<40} {'English':>10} {'Chinese':>10} {'Math':>10} {'Coding':>10}")
    print(f"{'-'*80}")

    for model_name, model_results in results.items():
        if feature_type not in model_results:
            continue

        ft_results = model_results[feature_type]
        english_acc = ft_results.get("english", {}).get("average_accuracy", "N/A")
        chinese_acc = ft_results.get("chinese", {}).get("average_accuracy", "N/A")
        math_acc = ft_results.get("math", {}).get("average_accuracy", "N/A")
        coding_acc = ft_results.get("coding", {}).get("average_accuracy", "N/A")

        def fmt(v):
            return f"{v:.1f}" if isinstance(v, float) else str(v)

        print(
            f"{model_name:<40} {fmt(english_acc):>10} {fmt(chinese_acc):>10} "
            f"{fmt(math_acc):>10} {fmt(coding_acc):>10}"
        )


def print_identity_probing_table(results: dict):
    """
    Print identity-probing results in a format similar to Table 2 in the paper.

    Args:
        results: Results from run_identity_probing_experiment_all
    """
    print(f"\n{'='*100}")
    print("Identity-Probing Detector Results")
    print(f"{'='*100}")

    prompts = IDENTITY_PROBING_PROMPTS
    header = f"{'Model':<40}"
    for p in prompts:
        header += f" {p[:20]:>22}"
    print(header)
    print(f"{'-'*100}")

    for model_name, model_results in results.items():
        row = f"{model_name:<40}"
        for prompt in prompts:
            if prompt in model_results:
                acc = model_results[prompt]["accuracy"]
                row += f" {acc:>22.1f}"
            else:
                row += f" {'N/A':>22}"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Run de-anonymization experiment"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/responses",
        help="Directory containing model responses",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/deanonymization",
        help="Directory to save results",
    )
    parser.add_argument(
        "--target_models",
        nargs="+",
        default=[
            "claude-3-5-sonnet-20240620",
            "gemini-1.5-pro",
            "gpt-4o-mini-2024-07-18",
            "gemma-2-27b-it",
            "llama-3.1-70b-instruct",
            "mixtral-8x7b-instruct-v0.1",
            "qwen2-72b-instruct",
        ],
        help="Target models to evaluate",
    )
    parser.add_argument(
        "--feature_types",
        nargs="+",
        default=FEATURE_TYPES,
        help="Feature types to evaluate",
    )
    args = parser.parse_args()

    # Load responses
    logger.info(f"Loading responses from {args.data_dir}")
    if os.path.exists(args.data_dir):
        all_responses = load_responses(args.data_dir)
    else:
        logger.warning(
            f"Data directory {args.data_dir} not found. "
            "Please collect model responses first using collect_responses.py"
        )
        return

    # Run training-based experiment
    logger.info("Running training-based detector experiment...")
    training_results = run_training_based_experiment(
        all_responses=all_responses,
        target_models=args.target_models,
        feature_types=args.feature_types,
        output_dir=args.output_dir,
    )

    # Print results
    for feature_type in args.feature_types:
        print_results_table(training_results, feature_type)

    logger.info("De-anonymization experiment complete!")


if __name__ == "__main__":
    main()
