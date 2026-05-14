"""
Main script for running the adversarial voting simulation (Section 3).

This script simulates adversarial attacks on the Chatbot Arena leaderboard
to estimate the number of votes and interactions needed to manipulate rankings.

Replicates Tables 4 and 5 from the paper:
- Table 4: High-ranked models (top 5)
- Table 5: Low-ranked models (bottom 5)

Usage:
    python run_simulation_experiment.py \
        --data_path data/chatbot_arena_votes.csv \
        --output_dir results/simulation
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

from bradley_terry import (
    compute_bradley_terry_coefficients,
    get_rankings,
)
from adversarial_simulation import (
    AdversarialSimulator,
    SimulationConfig,
    LeaderboardState,
    load_historical_data,
    create_synthetic_leaderboard,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_high_ranked_experiment(
    leaderboard: LeaderboardState,
    config: SimulationConfig,
    output_dir: str = None,
) -> dict:
    """
    Run the adversarial voting experiment for high-ranked models (Table 4).

    Targets the top 5 models and tries to move them to ranks 1-5.

    Args:
        leaderboard: Initial leaderboard state
        config: Simulation configuration
        output_dir: Directory to save results

    Returns:
        Dictionary with results
    """
    # Get top 5 models
    top_5 = [(rank, name, rating) for rank, name, rating in leaderboard.rankings[:5]]
    target_models = [name for _, name, _ in top_5]
    target_ranks = [1, 2, 3, 4, 5]

    logger.info("Running high-ranked model experiment...")
    logger.info(f"Target models: {target_models}")

    simulator = AdversarialSimulator(leaderboard, config)
    results = simulator.run_full_experiment(target_models, target_ranks)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "high_ranked_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")

    return results


def run_low_ranked_experiment(
    leaderboard: LeaderboardState,
    config: SimulationConfig,
    output_dir: str = None,
) -> dict:
    """
    Run the adversarial voting experiment for low-ranked models (Table 5).

    Targets the bottom 5 models and tries to move them to nearby ranks.

    Args:
        leaderboard: Initial leaderboard state
        config: Simulation configuration
        output_dir: Directory to save results

    Returns:
        Dictionary with results
    """
    n_models = len(leaderboard.model_names)

    # Get bottom 5 models
    bottom_5 = [
        (rank, name, rating)
        for rank, name, rating in leaderboard.rankings[n_models - 5:]
    ]
    target_models = [name for _, name, _ in bottom_5]

    # Target ranks: nearby positions for each model
    bottom_ranks = [rank for rank, _, _ in bottom_5]
    min_rank = min(bottom_ranks)
    target_ranks = list(range(min_rank - 4, min_rank + 5))

    logger.info("Running low-ranked model experiment...")
    logger.info(f"Target models: {target_models}")

    simulator = AdversarialSimulator(leaderboard, config)
    results = simulator.run_full_experiment(target_models, target_ranks)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "low_ranked_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")

    return results


def run_detector_accuracy_ablation(
    leaderboard: LeaderboardState,
    target_model: str,
    target_ranks: list,
    detector_accuracies: list = None,
    output_dir: str = None,
) -> dict:
    """
    Run ablation study on detector accuracy (Table 8 in Appendix B.2).

    Args:
        leaderboard: Initial leaderboard state
        target_model: Target model name
        target_ranks: List of target ranks to achieve
        detector_accuracies: List of detector accuracies to evaluate
        output_dir: Directory to save results

    Returns:
        Dictionary with results for each accuracy level
    """
    if detector_accuracies is None:
        detector_accuracies = [1.0, 0.95, 0.9]

    results = {}

    for accuracy in detector_accuracies:
        logger.info(f"Running ablation with detector accuracy: {accuracy}")

        config = SimulationConfig(
            detection_accuracy=accuracy,
            false_positive_rate=1.0 - accuracy,
            false_negative_rate=1.0 - accuracy,
        )

        simulator = AdversarialSimulator(leaderboard, config)
        accuracy_results = {}

        for target_rank in target_ranks:
            result = simulator.simulate_attack(target_model, target_rank)
            accuracy_results[target_rank] = {
                "adversarial_votes": result["adversarial_votes"]
                if result["achieved"]
                else ">max",
                "total_interactions": result["total_interactions"]
                if result["achieved"]
                else ">max",
            }

        results[accuracy] = accuracy_results

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "detector_accuracy_ablation.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")

    return results


def run_non_detection_strategy_ablation(
    leaderboard: LeaderboardState,
    target_model: str,
    target_ranks: list,
    strategies: list = None,
    output_dir: str = None,
) -> dict:
    """
    Run ablation study on non-detection strategies (Table 9 in Appendix B.2).

    Args:
        leaderboard: Initial leaderboard state
        target_model: Target model name
        target_ranks: List of target ranks to achieve
        strategies: List of non-detection strategies to evaluate
        output_dir: Directory to save results

    Returns:
        Dictionary with results for each strategy
    """
    if strategies is None:
        strategies = ["do_nothing", "random_upvote", "vote_tie", "vote_tie_both_bad"]

    results = {}

    for strategy in strategies:
        logger.info(f"Running ablation with non-detection strategy: {strategy}")

        config = SimulationConfig(
            detection_accuracy=0.95,
            non_detection_strategy=strategy,
        )

        simulator = AdversarialSimulator(leaderboard, config)
        strategy_results = {}

        for target_rank in target_ranks:
            result = simulator.simulate_attack(target_model, target_rank)
            strategy_results[target_rank] = {
                "total_interactions": result["total_interactions"]
                if result["achieved"]
                else ">max",
            }

        results[strategy] = strategy_results

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "non_detection_strategy_ablation.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")

    return results


def print_simulation_table(results: dict, title: str = "Simulation Results"):
    """
    Print simulation results in a format similar to Tables 4 and 5 in the paper.

    Args:
        results: Results from run_full_experiment
        title: Table title
    """
    print(f"\n{'='*100}")
    print(title)
    print(f"{'='*100}")

    # Get all target ranks
    all_ranks = set()
    for model_results in results.values():
        all_ranks.update(model_results.keys())
    all_ranks = sorted(all_ranks)

    # Print header
    header = f"{'Model':<40} {'Current Rank':>15}"
    for rank in all_ranks:
        header += f" {'Rank ' + str(rank):>12}"
    print(header)
    print(f"{'-'*100}")

    # Print rows
    for model_name, model_results in results.items():
        current_rank = "N/A"
        row = f"{model_name:<40} {current_rank:>15}"
        for rank in all_ranks:
            if rank in model_results:
                votes = model_results[rank].get("adversarial_votes", "N/A")
                row += f" {str(votes):>12}"
            else:
                row += f" {'N/A':>12}"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Run adversarial voting simulation"
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
        default="results/simulation",
        help="Directory to save results",
    )
    parser.add_argument(
        "--detection_accuracy",
        type=float,
        default=0.95,
        help="Detector accuracy (default: 0.95 as in paper)",
    )
    parser.add_argument(
        "--n_models",
        type=int,
        default=130,
        help="Number of models for synthetic leaderboard (if no data provided)",
    )
    parser.add_argument(
        "--run_ablations",
        action="store_true",
        help="Run ablation studies",
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
            f"No data path provided or file not found. "
            f"Creating synthetic leaderboard with {args.n_models} models."
        )
        leaderboard = create_synthetic_leaderboard(n_models=args.n_models)

    logger.info(f"Leaderboard has {len(leaderboard.model_names)} models")
    logger.info("Top 5 models:")
    for rank, name, rating in leaderboard.rankings[:5]:
        logger.info(f"  #{rank}: {name} (rating: {rating:.4f})")

    # Default simulation config
    config = SimulationConfig(
        detection_accuracy=args.detection_accuracy,
        false_positive_rate=1.0 - args.detection_accuracy,
        false_negative_rate=1.0 - args.detection_accuracy,
    )

    # Run high-ranked experiment
    high_results = run_high_ranked_experiment(leaderboard, config, args.output_dir)
    print_simulation_table(high_results, "High-Ranked Models (Table 4)")

    # Run low-ranked experiment
    low_results = run_low_ranked_experiment(leaderboard, config, args.output_dir)
    print_simulation_table(low_results, "Low-Ranked Models (Table 5)")

    # Run ablations if requested
    if args.run_ablations:
        n_models = len(leaderboard.model_names)
        bottom_model = leaderboard.rankings[n_models - 1][1]
        bottom_rank = leaderboard.rankings[n_models - 1][0]

        target_ranks = [
            bottom_rank - 50,
            bottom_rank - 20,
            bottom_rank - 10,
            bottom_rank - 5,
            bottom_rank - 2,
            bottom_rank - 1,
        ]
        target_ranks = [r for r in target_ranks if r > 0]

        # Detector accuracy ablation
        logger.info("Running detector accuracy ablation...")
        acc_results = run_detector_accuracy_ablation(
            leaderboard=leaderboard,
            target_model=bottom_model,
            target_ranks=target_ranks,
            output_dir=args.output_dir,
        )

        # Non-detection strategy ablation
        logger.info("Running non-detection strategy ablation...")
        strategy_results = run_non_detection_strategy_ablation(
            leaderboard=leaderboard,
            target_model=bottom_model,
            target_ranks=target_ranks,
            output_dir=args.output_dir,
        )

    logger.info("Simulation experiment complete!")


if __name__ == "__main__":
    main()
