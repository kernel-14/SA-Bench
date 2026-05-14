#!/usr/bin/env python3
"""
Main entry point for reproducing experiments from:
"Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards"

Usage:
    python run_experiments.py [--n_models N] [--output_dir DIR] [--synthetic]

This script runs all three core experiments:
1. De-anonymization of model responses (Section 2)
2. Adversarial vote estimation via simulation (Section 3)  
3. Mitigation evaluation (Section 4)
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments import run_all_experiments
from src.models import get_all_model_names, PROMPT_EXAMPLES
from src.deanonymization import (
    IdentityProbingDetector, TrainingBasedDetector,
    compute_pca_visualization, IDENTITY_PROBING_PROMPTS
)
from src.simulation import (
    BradleyTerryModel, LeaderboardSimulation, AttackerConfig,
    estimate_votes_for_rank_change,
)
from src.mitigations import (
    AttackCost, MaliciousUserDetector, PerturbedLeaderboard,
    estimate_detector_training_cost, compute_vote_distribution_from_ratings,
    evaluate_malicious_detection_with_noise,
)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from 'Exploring and Mitigating "
                    "Adversarial Manipulation of Voting-Based Leaderboards'"
    )
    parser.add_argument(
        "--n_models", type=int, default=22,
        help="Number of models to use in experiments (default: 22)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save results (default: results/)"
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="Use synthetic data for testing (no API calls)"
    )
    parser.add_argument(
        "--skip_mitigations", action="store_true",
        help="Skip mitigations experiment (Section 4)"
    )
    parser.add_argument(
        "--only_deanonymization", action="store_true",
        help="Run only de-anonymization experiment (Section 2)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("REPRODUCING: Exploring and Mitigating Adversarial Manipulation")
    print("            of Voting-Based Leaderboards")
    print("=" * 70)
    print(f"Models: {args.n_models}")
    print(f"Output: {args.output_dir}")
    print(f"Mode: {'Synthetic data' if args.synthetic else 'Real API queries'}")
    print()
    
    # Run experiments
    run_all_experiments(
        output_dir=args.output_dir,
        n_models=args.n_models,
        use_synthetic=args.synthetic,
    )
    
    print("\n" + "=" * 70)
    print("Experiments complete! Results saved to:", args.output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
