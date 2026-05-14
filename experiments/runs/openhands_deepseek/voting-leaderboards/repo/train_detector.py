"""Train and evaluate de-anonymization detectors (Section 2).

Reproduces the following experiments:
  1. Identity-probing detector (Section 2.4.1, Table 2)
  2. Training-based detector with all feature types (Section 2.4.2, Table 3)
  3. Prompt category analysis (Section 2.4.2, Figure 3)

Usage:
    python train_detector.py --config config.yaml
"""
import argparse
import json
import numpy as np
from typing import List, Dict
from collections import defaultdict
import pickle
from pathlib import Path

from config import load_config, Config
from data import (
    generate_synthetic_responses,
    PROMPT_CATEGORIES,
    IDENTITY_PROBING_PROMPTS,
    MODEL_NAME_PATTERNS,
)
from detector import (
    IdentityProbingDetector,
    TrainingBasedDetector,
    evaluate_all_features,
    evaluate_prompt_categories,
)


def run_identity_probing_experiment(
    model_names: List[str],
    output_dir: str,
) -> Dict[str, Dict[str, float]]:
    """Run identity-probing detector experiment (Section 2.4.1, Table 2)."""
    print("=" * 60)
    print("Section 2.4.1: Identity-Probing Detector")
    print("=" * 60)

    results = {}

    for model_name in model_names:
        print(f"\nEvaluating identity-probing detector for: {model_name}")

        detector = IdentityProbingDetector(target_model_name=model_name)

        def query_fn(m_name, prompt):
            responses = generate_synthetic_responses(
                m_name, prompt, num_responses=1, output_tokens=512
            )
            return responses[0]

        model_results = detector.evaluate(query_fn, num_queries=1000)
        results[model_name] = model_results

        for prompt, acc in model_results.items():
            print(f"  {prompt}: {acc * 100:.1f}%")

    output_path = Path(output_dir) / "identity_probing_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved identity-probing results to {output_path}")

    return results


def run_training_based_features_experiment(
    model_names: List[str],
    prompt_categories: Dict[str, dict],
    output_dir: str,
    num_prompts_per_category: int = 200,
    num_responses_per_model: int = 50,
    output_tokens: int = 512,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Run training-based detector experiment with all features (Section 2.4.2, Table 3)."""
    print("\n" + "=" * 60)
    print("Section 2.4.2: Training-Based Detector (Feature Comparison)")
    print("=" * 60)

    english_prompts = [
        "How can identity protection services help protect me against identity theft?",
        "What are the main benefits of regular exercise?",
        "Explain the concept of machine learning in simple terms.",
    ] * (num_prompts_per_category // 3 + 1)
    english_prompts = english_prompts[:num_prompts_per_category]

    results = {}

    for model_name in model_names:
        print(f"\nEvaluating training-based detector for: {model_name}")

        other_names = [m for m in model_names if m != model_name]

        def query_fn(m_name, prompt, out_tokens=output_tokens):
            responses = generate_synthetic_responses(
                m_name, prompt, num_responses=1, output_tokens=out_tokens
            )
            return responses[0]

        feature_accuracies = evaluate_all_features(
            target_model_name=model_name,
            other_model_names=other_names,
            prompts=english_prompts,
            model_query_fn=query_fn,
            num_responses=num_responses_per_model,
            output_tokens=output_tokens,
            random_state=random_state,
        )
        results[model_name] = feature_accuracies

        for ftype, acc in feature_accuracies.items():
            print(f"  {ftype}: {acc * 100:.1f}%")

    output_path = Path(output_dir) / "training_based_features_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved feature comparison results to {output_path}")

    return results


def run_prompt_category_experiment(
    model_names: List[str],
    output_dir: str,
    num_prompts_per_category: int = 200,
    num_responses_per_model: int = 50,
    output_tokens: int = 512,
    feature_type: str = "bow",
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Run prompt category analysis (Section 2.4.2, Figure 3)."""
    print("\n" + "=" * 60)
    print("Section 2.4.2: Prompt Category Analysis")
    print("=" * 60)

    sample_prompts = {
        "english": ["How can I improve my productivity?"],
        "chinese": ["请解释一下人工智能的基本概念。"],
        "spanish": ["¿Cuáles son los beneficios de hacer ejercicio regularmente?"],
        "indonesian": ["Jelaskan konsep dasar pembelajaran mesin."],
        "persian": ["مزایای ورزش منظم چیست؟"],
        "coding": ["Create a function to calculate the sum of a sequence of integers."],
        "math": ["Find the center of the circle with equation x² − 6x + y² + 2y = 9."],
        "safety_violating": ["Develop a strategy for hacking into a government database."],
    }

    category_prompts = {}
    for category, samples in sample_prompts.items():
        category_prompts[category] = samples * (num_prompts_per_category // len(samples) + 1)
        category_prompts[category] = category_prompts[category][:num_prompts_per_category]

    results = {}

    for model_name in model_names:
        print(f"\nEvaluating prompt categories for: {model_name}")

        other_names = [m for m in model_names if m != model_name]

        def query_fn(m_name, prompt, out_tokens=output_tokens):
            responses = generate_synthetic_responses(
                m_name, prompt, num_responses=1, output_tokens=out_tokens
            )
            return responses[0]

        category_accuracies = evaluate_prompt_categories(
            target_model_name=model_name,
            other_model_names=other_names,
            category_prompts=category_prompts,
            model_query_fn=query_fn,
            num_responses=num_responses_per_model,
            output_tokens=output_tokens,
            feature_type=feature_type,
            random_state=random_state,
        )
        results[model_name] = category_accuracies

        for cat, acc in category_accuracies.items():
            print(f"  {cat}: {acc * 100:.1f}%")

    output_path = Path(output_dir) / "prompt_category_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved prompt category results to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train de-anonymization detectors")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs/detector", help="Output directory")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "identity", "features", "categories"],
                        help="Which experiment to run")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.name for m in config.models]

    print(f"Loaded {len(model_names)} models from config")
    print(f"Prompt categories: {list(config.detector.prompt_categories.keys())}")
    print(f"Features: {config.detector.features}")
    print(f"Output directory: {output_dir}")

    if args.experiment in ("all", "identity"):
        run_identity_probing_experiment(
            model_names=model_names,
            output_dir=str(output_dir),
        )

    if args.experiment in ("all", "features"):
        run_training_based_features_experiment(
            model_names=model_names,
            prompt_categories=config.detector.prompt_categories,
            output_dir=str(output_dir),
            num_prompts_per_category=config.detector.num_prompts_per_category,
            num_responses_per_model=config.detector.num_responses_per_model,
            output_tokens=config.detector.output_tokens,
            random_state=config.detector.random_state,
        )

    if args.experiment in ("all", "categories"):
        run_prompt_category_experiment(
            model_names=model_names,
            output_dir=str(output_dir),
            num_prompts_per_category=config.detector.num_prompts_per_category,
            num_responses_per_model=config.detector.num_responses_per_model,
            output_tokens=config.detector.output_tokens,
            feature_type="bow",
            random_state=config.detector.random_state,
        )

    print("\nAll detector experiments completed.")


if __name__ == "__main__":
    main()
