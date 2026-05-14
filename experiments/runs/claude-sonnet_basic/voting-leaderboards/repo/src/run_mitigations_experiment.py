"""
Main script for running the mitigations experiment (Section 4).

This script evaluates the effectiveness of various mitigations against
adversarial manipulation of voting-based leaderboards.

Replicates Figures 4, 5, and 6 from the paper:
- Figure 4: Scenario 1 - Likelihood test for malicious user detection
- Figure 5: Scenario 2 - Neyman-Pearson detector with perturbed leaderboard
- Figure 6: Utility impact of noise on leaderboard rankings

Usage:
    python run_mitigations_experiment.py \
        --data_path data/chatbot_arena_votes.csv \
        --output_dir results/mitigations
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bradley_terry import (
    compute_bradley_terry_coefficients,
    get_rankings,
    compute_benign_vote_distribution,
)
from adversarial_simulation import (
    LeaderboardState,
    load_historical_data,
    create_synthetic_leaderboard,
)
from mitigations import (
    MaliciousUserDetector,
    NeymanPearsonDetector,
    generate_adversarial_vote_sequence,
    run_mitigation_experiment,
    compute_attack_cost,
    estimate_detector_training_cost,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_scenario1_experiment(
    leaderboard: LeaderboardState,
    target_model: str,
    votes_per_user_range: list = None,
    n_users: int = 200,
    output_dir: str = None,
) -> dict:
    """
    Run Scenario 1 experiment: Known benign distribution.

    Evaluates the likelihood test for detecting malicious users.
    Replicates Figure 4 from the paper.

    Args:
        leaderboard: Leaderboard state
        target_model: Target model name
        votes_per_user_range: Range of votes per user to evaluate
        n_users: Number of users to simulate
        output_dir: Directory to save results

    Returns:
        Dictionary with detection results
    """
    if votes_per_user_range is None:
        votes_per_user_range = [10, 20, 30, 50, 100, 200]

    target_idx = leaderboard.get_model_index(target_model)
    n_models = len(leaderboard.model_names)
    benign_dist = compute_benign_vote_distribution(leaderboard.ratings)

    rng = np.random.RandomState(42)
    results = {}

    for n_votes in votes_per_user_range:
        logger.info(f"Evaluating with {n_votes} votes per user...")

        # Generate benign sequences
        benign_sequences = []
        for _ in range(n_users // 2):
            seq = list(rng.choice(n_models, size=n_votes, p=benign_dist))
            benign_sequences.append(seq)

        # Generate naive adversarial sequences
        naive_sequences = []
        for _ in range(n_users // 2):
            seq = generate_adversarial_vote_sequence(
                target_model_idx=target_idx,
                n_votes=n_votes,
                n_models=n_models,
                benign_dist=benign_dist,
                attack_direction="up",
                use_public_rankings=False,
                rng=rng,
            )
            naive_sequences.append(seq)

        # Generate sophisticated adversarial sequences (using public rankings)
        sophisticated_sequences = []
        for _ in range(n_users // 2):
            seq = generate_adversarial_vote_sequence(
                target_model_idx=target_idx,
                n_votes=n_votes,
                n_models=n_models,
                benign_dist=benign_dist,
                attack_direction="up",
                use_public_rankings=True,
                rng=rng,
            )
            sophisticated_sequences.append(seq)

        # Evaluate detector
        detector = MaliciousUserDetector(
            benign_vote_distribution=benign_dist,
            model_names=leaderboard.model_names,
            significance_level=0.01,
            n_simulations=1000,  # Reduced for speed
            random_seed=42,
        )

        naive_results = detector.evaluate_detection(benign_sequences, naive_sequences)
        sophisticated_results = detector.evaluate_detection(
            benign_sequences, sophisticated_sequences
        )

        results[n_votes] = {
            "naive_adversary": naive_results,
            "sophisticated_adversary": sophisticated_results,
        }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "scenario1_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return results


def run_scenario2_experiment(
    leaderboard: LeaderboardState,
    target_model: str,
    noise_scales: list = None,
    votes_per_user: int = 50,
    n_users: int = 200,
    output_dir: str = None,
) -> dict:
    """
    Run Scenario 2 experiment: Known benign and malicious distributions.

    Evaluates the Neyman-Pearson detector with perturbed leaderboard.
    Replicates Figures 5 and 6 from the paper.

    Args:
        leaderboard: Leaderboard state
        target_model: Target model name
        noise_scales: List of noise scales to evaluate
        votes_per_user: Number of votes per user
        n_users: Number of users to simulate
        output_dir: Directory to save results

    Returns:
        Dictionary with detection and utility results
    """
    if noise_scales is None:
        noise_scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

    target_idx = leaderboard.get_model_index(target_model)
    n_models = len(leaderboard.model_names)
    benign_dist = compute_benign_vote_distribution(leaderboard.ratings)

    rng = np.random.RandomState(42)
    results = {}

    # Generate benign sequences (same for all noise scales)
    benign_sequences = []
    for _ in range(n_users // 2):
        seq = list(rng.choice(n_models, size=votes_per_user, p=benign_dist))
        benign_sequences.append(seq)

    for noise_scale in noise_scales:
        logger.info(f"Evaluating with noise scale: {noise_scale}")

        detector = NeymanPearsonDetector(
            true_ratings=leaderboard.ratings,
            model_names=leaderboard.model_names,
            noise_scale=noise_scale,
            significance_level=0.01,
            random_seed=42,
        )

        # Generate adversarial sequences using perturbed rankings
        adversarial_sequences = []
        for _ in range(n_users // 2):
            seq = generate_adversarial_vote_sequence(
                target_model_idx=target_idx,
                n_votes=votes_per_user,
                n_models=n_models,
                benign_dist=detector.adversarial_dist,
                attack_direction="up",
                use_public_rankings=True,
                rng=rng,
            )
            adversarial_sequences.append(seq)

        detection_results = detector.evaluate_detection(
            benign_sequences, adversarial_sequences
        )
        utility_impact = detector.compute_utility_impact()

        results[noise_scale] = {
            "detection": detection_results,
            "utility_impact": utility_impact,
        }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "scenario2_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return results


def plot_scenario1_results(results: dict, output_dir: str = None):
    """
    Plot Scenario 1 results (Figure 4 in the paper).

    Args:
        results: Results from run_scenario1_experiment
        output_dir: Directory to save plots
    """
    votes_per_user = sorted(results.keys())

    naive_tpr = [results[n]["naive_adversary"]["true_positive_rate"] for n in votes_per_user]
    naive_fpr = [results[n]["naive_adversary"]["false_positive_rate"] for n in votes_per_user]
    sophisticated_tpr = [
        results[n]["sophisticated_adversary"]["true_positive_rate"] for n in votes_per_user
    ]
    sophisticated_fpr = [
        results[n]["sophisticated_adversary"]["false_positive_rate"] for n in votes_per_user
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # True positive rate
    axes[0].plot(votes_per_user, naive_tpr, "b-o", label="Naive adversary")
    axes[0].plot(votes_per_user, sophisticated_tpr, "r-s", label="Sophisticated adversary")
    axes[0].set_xlabel("Votes per user")
    axes[0].set_ylabel("True positive rate (detection rate)")
    axes[0].set_title("Scenario 1: Malicious User Detection Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # False positive rate
    axes[1].plot(votes_per_user, naive_fpr, "b-o", label="Naive adversary")
    axes[1].plot(votes_per_user, sophisticated_fpr, "r-s", label="Sophisticated adversary")
    axes[1].axhline(y=0.01, color="g", linestyle="--", label="Alpha = 0.01")
    axes[1].set_xlabel("Votes per user")
    axes[1].set_ylabel("False positive rate")
    axes[1].set_title("Scenario 1: False Positive Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 0.1])

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "figure4_scenario1.png"), dpi=150)
        logger.info(f"Figure 4 saved to {output_dir}/figure4_scenario1.png")

    plt.close()


def plot_scenario2_results(results: dict, output_dir: str = None):
    """
    Plot Scenario 2 results (Figures 5 and 6 in the paper).

    Args:
        results: Results from run_scenario2_experiment
        output_dir: Directory to save plots
    """
    noise_scales = sorted(results.keys())

    tpr = [results[n]["detection"]["true_positive_rate"] for n in noise_scales]
    fpr = [results[n]["detection"]["false_positive_rate"] for n in noise_scales]
    utility = [results[n]["utility_impact"] for n in noise_scales]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Detection rate vs noise scale (Figure 5)
    axes[0].plot(noise_scales, tpr, "b-o", label="Detection rate (TPR)")
    axes[0].plot(noise_scales, fpr, "r-s", label="False positive rate (FPR)")
    axes[0].axhline(y=0.01, color="g", linestyle="--", label="Alpha = 0.01")
    axes[0].set_xlabel("Noise scale")
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Scenario 2: Detection vs Noise Scale (Figure 5)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Utility impact vs noise scale (Figure 6)
    axes[1].plot(noise_scales, utility, "g-^", label="Avg rank change")
    axes[1].set_xlabel("Noise scale")
    axes[1].set_ylabel("Average absolute rank change")
    axes[1].set_title("Utility Impact of Noise (Figure 6)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "figure5_6_scenario2.png"), dpi=150)
        logger.info(f"Figures 5 & 6 saved to {output_dir}/figure5_6_scenario2.png")

    plt.close()


def print_cost_analysis():
    """
    Print the attack cost analysis from Section 4.1.
    """
    print("\n" + "=" * 60)
    print("Attack Cost Analysis (Section 4.1)")
    print("=" * 60)

    # Estimate detector training cost
    detector_cost = estimate_detector_training_cost(
        n_prompts=200,
        n_proprietary_models=10,
        n_opensource_models=20,
        responses_per_model=50,
        max_output_tokens=512,
        proprietary_cost_per_million=5.00,
        opensource_cost_per_million=1.80,
    )
    print(f"\nDetector training cost: ${detector_cost:.2f}")
    print("(Based on 200 prompts, 10 proprietary + 20 open-source models)")

    # Cost without mitigations
    print("\n--- Without Mitigations ---")
    cost_no_mitigation = compute_attack_cost(
        n_actions=1000,  # ~1000 votes needed
        max_actions_per_account=float("inf"),  # No limit
        cost_per_account=0.0,
        cost_per_action=0.0,  # Minimal cost
        detector_cost=detector_cost,
    )
    print(f"Total cost (1000 votes): ${cost_no_mitigation:.2f}")
    print("(Dominated by detector training cost)")

    # Cost with authentication
    print("\n--- With Authentication (Section 4.2.1) ---")
    for cost_per_account in [0.0, 1.0, 5.0, 10.0]:
        cost_auth = compute_attack_cost(
            n_actions=1000,
            max_actions_per_account=100,  # Rate limit: 100 actions per account
            cost_per_account=cost_per_account,
            cost_per_action=0.0,
            detector_cost=detector_cost,
        )
        print(
            f"  c_account=${cost_per_account:.2f}, m=100: "
            f"Total cost = ${cost_auth:.2f}"
        )

    # Cost with CAPTCHA
    print("\n--- With CAPTCHA (Section 4.2.4) ---")
    for captcha_cost in [0.001, 0.01, 0.1]:
        cost_captcha = compute_attack_cost(
            n_actions=1000,
            max_actions_per_account=float("inf"),
            cost_per_account=0.0,
            cost_per_action=captcha_cost,
            detector_cost=detector_cost,
        )
        print(
            f"  c_CAPTCHA=${captcha_cost:.3f}: "
            f"Total cost = ${cost_captcha:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run mitigations experiment"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to Chatbot Arena voting data CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/mitigations",
        help="Directory to save results",
    )
    parser.add_argument(
        "--n_models",
        type=int,
        default=50,
        help="Number of models for synthetic leaderboard",
    )
    args = parser.parse_args()

    # Load or create leaderboard
    if args.data_path and os.path.exists(args.data_path):
        logger.info(f"Loading historical data from {args.data_path}")
        model_names, wins_matrix, vote_counts = load_historical_data(args.data_path)
        ratings = compute_bradley_terry_coefficients(wins_matrix)
        rankings = get_rankings(ratings, model_names)
        leaderboard = LeaderboardState(
            model_names=model_names,
            wins_matrix=wins_matrix,
            ratings=ratings,
            rankings=rankings,
        )
    else:
        logger.info(
            f"Creating synthetic leaderboard with {args.n_models} models."
        )
        leaderboard = create_synthetic_leaderboard(n_models=args.n_models)

    # Select target model (middle-ranked)
    n_models = len(leaderboard.model_names)
    target_model = leaderboard.rankings[n_models // 2][1]
    logger.info(f"Target model: {target_model}")

    # Print cost analysis
    print_cost_analysis()

    # Run Scenario 1 experiment
    logger.info("Running Scenario 1 experiment...")
    scenario1_results = run_scenario1_experiment(
        leaderboard=leaderboard,
        target_model=target_model,
        votes_per_user_range=[10, 20, 30, 50, 100],
        n_users=100,
        output_dir=args.output_dir,
    )
    plot_scenario1_results(scenario1_results, args.output_dir)

    # Run Scenario 2 experiment
    logger.info("Running Scenario 2 experiment...")
    scenario2_results = run_scenario2_experiment(
        leaderboard=leaderboard,
        target_model=target_model,
        noise_scales=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
        votes_per_user=50,
        n_users=100,
        output_dir=args.output_dir,
    )
    plot_scenario2_results(scenario2_results, args.output_dir)

    logger.info("Mitigations experiment complete!")


if __name__ == "__main__":
    main()
