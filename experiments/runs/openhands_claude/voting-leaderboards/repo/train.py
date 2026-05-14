"""
Training pipeline for the adversarial leaderboard manipulation paper.

Orchestrates:
  1. Training-based detector experiments (Section 2.3, Table 3, Figure 3)
  2. Adversarial voting simulations (Section 3, Tables 4, 5, 8, 9)
  3. Mitigation experiments (Section 4.3, Figures 4, 5, 6)

All experiments use synthetic data by default (no API keys required).
To use real model responses, provide a responses JSON file via --responses_file.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from bradley_terry import (
    fit_bradley_terry,
    get_rankings,
    strengths_to_elo,
)
from config import (
    ALL_MODELS,
    HIGH_RANKED_MODELS,
    LOW_RANKED_MODELS,
    MAIN_EVAL_MODELS,
    PROMPT_CATEGORIES,
    BradleyTerryConfig,
    CostModelConfig,
    DetectorConfig,
    MitigationConfig,
    SimulationConfig,
)
from data import (
    SAMPLE_PROMPTS,
    VoteRecord,
    build_detector_dataset,
    dataframe_to_votes,
    generate_synthetic_arena_votes,
    generate_synthetic_dataset,
    load_arena_votes,
    load_model_responses,
    votes_to_dataframe,
)
from detector import (
    evaluate_detector_across_models,
    evaluate_detector_by_category,
    evaluate_identity_probing_detector,
)
from features import FeatureType
from mitigation import (
    compute_attack_cost,
    compute_detector_training_cost,
    evaluate_noise_scales,
    simulate_detection_scenario1,
)
from simulation import (
    simulate_high_ranked_models,
    simulate_varying_detector_accuracy,
    simulate_varying_non_target_strategy,
)


# ---------------------------------------------------------------------------
# Experiment 1: Training-based detector (Section 2.3)
# ---------------------------------------------------------------------------

def run_detector_experiment(
    responses_file: Optional[str] = None,
    output_dir: str = "results",
    feature_types: Optional[List[FeatureType]] = None,
    models: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    num_prompts: int = 200,
    num_responses: int = 50,
    random_seed: int = 42,
) -> Dict:
    """
    Train and evaluate the training-based detector across models and categories.

    Reproduces Table 3 and Figure 3.
    """
    if feature_types is None:
        feature_types = ["length_word", "length_char", "bow", "tfidf"]
    if models is None:
        models = MAIN_EVAL_MODELS
    if categories is None:
        categories = PROMPT_CATEGORIES

    os.makedirs(output_dir, exist_ok=True)

    if responses_file and os.path.exists(responses_file):
        print(f"Loading responses from {responses_file}")
        all_responses = load_model_responses(responses_file)
        # Expected format: {category: {prompt: {model: [responses]}}}
        responses_by_category = all_responses
    else:
        print("Generating synthetic responses for detector experiment...")
        responses_by_category = {}
        for category in categories:
            prompts = SAMPLE_PROMPTS.get(category, SAMPLE_PROMPTS["english"])[:num_prompts]
            responses_by_category[category] = generate_synthetic_dataset(
                models=models,
                prompts=prompts,
                category=category,
                num_responses_per_model=num_responses,
                random_seed=random_seed,
            )

    results = {}

    # Table 3: Feature comparison on English prompts
    print("\n=== Table 3: Feature comparison (English prompts) ===")
    english_responses = responses_by_category.get("english", {})
    table3_results = {}

    for feature_type in feature_types:
        print(f"  Feature: {feature_type}")
        model_accuracies = evaluate_detector_across_models(
            responses_by_prompt_model=english_responses,
            target_models=models,
            all_models=models,
            feature_type=feature_type,
            num_positive=num_responses,
            num_negative=num_responses,
            random_state=random_seed,
        )
        table3_results[feature_type] = model_accuracies
        for model, acc in model_accuracies.items():
            print(f"    {model}: {acc * 100:.1f}%")

    results["table3"] = table3_results

    # Figure 3: Accuracy by category (BoW features)
    print("\n=== Figure 3: Accuracy by category (BoW features) ===")
    figure3_results = evaluate_detector_by_category(
        responses_by_category=responses_by_category,
        target_models=models,
        feature_type="bow",
        num_positive=num_responses,
        num_negative=num_responses,
        random_state=random_seed,
    )

    for category, model_accs in figure3_results.items():
        print(f"  Category: {category}")
        for model, acc in model_accs.items():
            print(f"    {model}: {acc * 100:.1f}%")

    results["figure3"] = figure3_results

    # Save results
    output_path = os.path.join(output_dir, "detector_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetector results saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Adversarial voting simulation (Section 3)
# ---------------------------------------------------------------------------

def run_simulation_experiment(
    votes_file: Optional[str] = None,
    output_dir: str = "results",
    sim_config: Optional[SimulationConfig] = None,
    bt_config: Optional[BradleyTerryConfig] = None,
    run_ablations: bool = True,
) -> Dict:
    """
    Run adversarial voting simulations to estimate attack cost.

    Reproduces Tables 4, 5, 8, and 9.
    """
    if sim_config is None:
        sim_config = SimulationConfig()
    if bt_config is None:
        bt_config = BradleyTerryConfig()

    os.makedirs(output_dir, exist_ok=True)

    # Load or generate voting data
    if votes_file and os.path.exists(votes_file):
        print(f"Loading votes from {votes_file}")
        initial_votes = load_arena_votes(votes_file)
        models = list(set(
            [v.model_a for v in initial_votes] + [v.model_b for v in initial_votes]
        ))
    else:
        print("Generating synthetic arena votes for simulation...")
        # Use high-ranked + low-ranked models for simulation
        sim_models = list(HIGH_RANKED_MODELS.keys()) + list(LOW_RANKED_MODELS.keys())
        # Add some middle-ranked models
        sim_models += [f"model_{i}" for i in range(1, 121)]
        models = sim_models

        # Generate initial votes proportional to the paper's dataset scale
        # (1.67M votes, but we use a smaller scale for tractability)
        initial_votes = generate_synthetic_arena_votes(
            models=models,
            num_votes=50000,
            random_seed=42,
        )

    print(f"Loaded {len(initial_votes)} initial votes for {len(models)} models")

    results = {}

    # Table 4: High-ranked models
    print("\n=== Table 4: High-ranked model attack simulation ===")
    high_ranked_subset = {
        m: info for m, info in HIGH_RANKED_MODELS.items()
        if m in models
    }

    if high_ranked_subset:
        target_ranks_high = [1, 2, 3, 4, 5]
        table4_results = simulate_high_ranked_models(
            initial_votes=initial_votes,
            models=models,
            high_ranked_models=high_ranked_subset,
            target_ranks=target_ranks_high,
            bt_config=bt_config,
            sim_config=sim_config,
        )

        for model, rank_results in table4_results.items():
            print(f"  {model}:")
            for target_rank, (votes, interactions) in rank_results.items():
                print(f"    -> rank {target_rank}: {votes:.0f} votes, {interactions:.0f} interactions")

        results["table4"] = {
            m: {str(r): {"votes": v, "interactions": i} for r, (v, i) in rr.items()}
            for m, rr in table4_results.items()
        }

    # Table 5: Low-ranked models
    print("\n=== Table 5: Low-ranked model attack simulation ===")
    low_ranked_subset = {
        m: info for m, info in LOW_RANKED_MODELS.items()
        if m in models
    }

    if low_ranked_subset:
        target_ranks_low = [125, 126, 127, 128, 129]
        table5_results = simulate_high_ranked_models(
            initial_votes=initial_votes,
            models=models,
            high_ranked_models=low_ranked_subset,
            target_ranks=target_ranks_low,
            bt_config=bt_config,
            sim_config=sim_config,
        )

        for model, rank_results in table5_results.items():
            print(f"  {model}:")
            for target_rank, (votes, interactions) in rank_results.items():
                print(f"    -> rank {target_rank}: {votes:.0f} votes, {interactions:.0f} interactions")

        results["table5"] = {
            m: {str(r): {"votes": v, "interactions": i} for r, (v, i) in rr.items()}
            for m, rr in table5_results.items()
        }

    if run_ablations:
        # Table 8: Ablation over detector accuracy
        print("\n=== Table 8: Detector accuracy ablation ===")
        target_model_ablation = "llama-13b" if "llama-13b" in models else models[-1]
        target_ranks_ablation = [79, 109, 119, 124, 127, 128]
        detector_accuracies = [1.0, 0.95, 0.9]

        table8_results = simulate_varying_detector_accuracy(
            target_model=target_model_ablation,
            initial_votes=initial_votes,
            models=models,
            target_ranks=[r for r in target_ranks_ablation if r < len(models)],
            detector_accuracies=detector_accuracies,
            bt_config=bt_config,
            sim_config=sim_config,
        )

        for acc, rank_results in table8_results.items():
            print(f"  Accuracy {acc}:")
            for target_rank, (votes, interactions) in rank_results.items():
                print(f"    -> rank {target_rank}: {votes:.0f} votes, {interactions:.0f} interactions")

        results["table8"] = {
            str(acc): {str(r): {"votes": v, "interactions": i} for r, (v, i) in rr.items()}
            for acc, rr in table8_results.items()
        }

        # Table 9: Ablation over non-target strategies
        print("\n=== Table 9: Non-target strategy ablation ===")
        strategies = ["nothing", "random_upvote", "tie", "both_bad"]
        target_ranks_strategy = [79, 109, 119, 124, 127, 128]

        table9_results = simulate_varying_non_target_strategy(
            target_model=target_model_ablation,
            initial_votes=initial_votes,
            models=models,
            target_ranks=[r for r in target_ranks_strategy if r < len(models)],
            strategies=strategies,
            bt_config=bt_config,
            sim_config=sim_config,
        )

        for strategy, rank_results in table9_results.items():
            print(f"  Strategy '{strategy}':")
            for target_rank, interactions in rank_results.items():
                print(f"    -> rank {target_rank}: {interactions:.0f} interactions")

        results["table9"] = {
            s: {str(r): i for r, i in rr.items()}
            for s, rr in table9_results.items()
        }

    # Save results
    output_path = os.path.join(output_dir, "simulation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSimulation results saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Mitigation experiments (Section 4.3)
# ---------------------------------------------------------------------------

def run_mitigation_experiment(
    votes_file: Optional[str] = None,
    output_dir: str = "results",
    mitigation_config: Optional[MitigationConfig] = None,
    bt_config: Optional[BradleyTerryConfig] = None,
) -> Dict:
    """
    Run mitigation experiments.

    Reproduces Figures 4, 5, and 6.
    """
    if mitigation_config is None:
        mitigation_config = MitigationConfig()
    if bt_config is None:
        bt_config = BradleyTerryConfig()

    os.makedirs(output_dir, exist_ok=True)

    # Load or generate voting data
    if votes_file and os.path.exists(votes_file):
        initial_votes = load_arena_votes(votes_file)
        models = list(set(
            [v.model_a for v in initial_votes] + [v.model_b for v in initial_votes]
        ))
    else:
        print("Generating synthetic arena votes for mitigation experiment...")
        sim_models = list(HIGH_RANKED_MODELS.keys()) + list(LOW_RANKED_MODELS.keys())
        sim_models += [f"model_{i}" for i in range(1, 121)]
        models = sim_models
        initial_votes = generate_synthetic_arena_votes(
            models=models,
            num_votes=50000,
            random_seed=42,
        )

    # Compute true BT strengths
    true_strengths = fit_bradley_terry(models, initial_votes, bt_config)
    target_model = models[0]  # Use top-ranked model as target

    results = {}

    # Figure 4: Scenario 1 detection
    print("\n=== Figure 4: Scenario 1 - Known benign distribution ===")
    votes_per_user_range = [10, 20, 50, 100, 200]
    figure4_results = {}

    for num_votes in votes_per_user_range:
        # Naive adversary
        fpr_naive, tpr_naive = simulate_detection_scenario1(
            target_model=target_model,
            models=models,
            true_strengths=true_strengths,
            num_votes_per_user=num_votes,
            num_benign_users=200,
            num_malicious_users=100,
            alpha=mitigation_config.alpha,
            num_simulations=mitigation_config.num_simulations // 10,
            attacker_uses_public_ranking=False,
            random_seed=42,
        )

        # Sophisticated adversary (uses public ranking)
        fpr_smart, tpr_smart = simulate_detection_scenario1(
            target_model=target_model,
            models=models,
            true_strengths=true_strengths,
            num_votes_per_user=num_votes,
            num_benign_users=200,
            num_malicious_users=100,
            alpha=mitigation_config.alpha,
            num_simulations=mitigation_config.num_simulations // 10,
            attacker_uses_public_ranking=True,
            random_seed=42,
        )

        figure4_results[num_votes] = {
            "naive": {"fpr": fpr_naive, "tpr": tpr_naive},
            "smart": {"fpr": fpr_smart, "tpr": tpr_smart},
        }
        print(f"  {num_votes} votes/user: naive TPR={tpr_naive:.2f}, smart TPR={tpr_smart:.2f}")

    results["figure4"] = figure4_results

    # Figures 5 and 6: Scenario 2 - Perturbed leaderboard
    print("\n=== Figures 5 & 6: Scenario 2 - Perturbed leaderboard ===")
    noise_results = evaluate_noise_scales(
        target_model=target_model,
        models=models,
        true_strengths=true_strengths,
        noise_scales=mitigation_config.noise_scales,
        num_votes_per_user=50,
        num_benign_users=200,
        num_malicious_users=100,
        random_seed=42,
    )

    for noise_scale, metrics in noise_results.items():
        print(
            f"  Noise={noise_scale:.1f}: "
            f"TPR={metrics['tpr']:.2f}, "
            f"FPR={metrics['fpr']:.2f}, "
            f"Rank change={metrics['mean_rank_change']:.1f}"
        )

    results["figures5_6"] = {str(k): v for k, v in noise_results.items()}

    # Attack cost model (Section 4.1)
    print("\n=== Section 4.1: Attack cost model ===")
    detector_cost = compute_detector_training_cost()
    print(f"  Detector training cost: ${detector_cost:.2f}")

    # Without mitigations (single account, minimal action cost)
    cost_no_mitigation = compute_attack_cost(
        num_actions=1000,
        max_actions_per_account=1_000_000,
        cost_per_account=0.0,
        cost_per_action=0.0,
        detector_training_cost=detector_cost,
    )
    print(f"  Cost without mitigations (1000 actions): ${cost_no_mitigation.total_cost:.2f}")

    # With rate limiting (100 actions per account, $1 per account)
    cost_rate_limited = compute_attack_cost(
        num_actions=1000,
        max_actions_per_account=100,
        cost_per_account=1.0,
        cost_per_action=0.0,
        detector_training_cost=detector_cost,
    )
    print(f"  Cost with rate limiting (100/account, $1/account): ${cost_rate_limited.total_cost:.2f}")

    # With CAPTCHA
    cost_captcha = compute_attack_cost(
        num_actions=1000,
        max_actions_per_account=1_000_000,
        cost_per_account=0.0,
        cost_per_action=0.001,
        detector_training_cost=detector_cost,
    )
    print(f"  Cost with CAPTCHA ($0.001/action): ${cost_captcha.total_cost:.2f}")

    results["cost_model"] = {
        "detector_training_cost": detector_cost,
        "no_mitigation": {
            "total": cost_no_mitigation.total_cost,
            "accounts": cost_no_mitigation.num_accounts_needed,
        },
        "rate_limited": {
            "total": cost_rate_limited.total_cost,
            "accounts": cost_rate_limited.num_accounts_needed,
        },
        "captcha": {
            "total": cost_captcha.total_cost,
            "accounts": cost_captcha.num_accounts_needed,
        },
    }

    # Save results
    output_path = os.path.join(output_dir, "mitigation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMitigation results saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from 'Exploring and Mitigating Adversarial "
                    "Manipulation of Voting-Based Leaderboards'"
    )
    parser.add_argument(
        "--experiment",
        choices=["detector", "simulation", "mitigation", "all"],
        default="all",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--responses_file",
        type=str,
        default=None,
        help="Path to pre-collected model responses JSON file.",
    )
    parser.add_argument(
        "--votes_file",
        type=str,
        default=None,
        help="Path to Chatbot Arena voting data file (JSON/CSV).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=200,
        help="Number of prompts per category for detector training.",
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        default=50,
        help="Number of responses per model per prompt.",
    )
    parser.add_argument(
        "--detection_accuracy",
        type=float,
        default=0.95,
        help="Assumed detector accuracy for simulation.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of simulation runs to average.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no_ablations",
        action="store_true",
        help="Skip ablation experiments (Tables 8, 9).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sim_config = SimulationConfig(
        detection_accuracy=args.detection_accuracy,
        false_positive_rate=1.0 - args.detection_accuracy,
        false_negative_rate=1.0 - args.detection_accuracy,
        random_seed=args.random_seed,
        num_runs=args.num_runs,
    )

    if args.experiment in ("detector", "all"):
        print("=" * 60)
        print("EXPERIMENT 1: Training-based detector (Section 2.3)")
        print("=" * 60)
        run_detector_experiment(
            responses_file=args.responses_file,
            output_dir=args.output_dir,
            num_prompts=args.num_prompts,
            num_responses=args.num_responses,
            random_seed=args.random_seed,
        )

    if args.experiment in ("simulation", "all"):
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Adversarial voting simulation (Section 3)")
        print("=" * 60)
        run_simulation_experiment(
            votes_file=args.votes_file,
            output_dir=args.output_dir,
            sim_config=sim_config,
            run_ablations=not args.no_ablations,
        )

    if args.experiment in ("mitigation", "all"):
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: Mitigation experiments (Section 4.3)")
        print("=" * 60)
        run_mitigation_experiment(
            votes_file=args.votes_file,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
