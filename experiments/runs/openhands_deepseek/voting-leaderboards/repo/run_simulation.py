"""Run adversarial voting simulations (Section 3).

Reproduces Tables 4, 5, 8, and 9:
  - Table 4: High-ranked models (votes/interactions to change ranking)
  - Table 5: Low-ranked models (votes/interactions to change ranking)
  - Table 8: Ablation over detector accuracy
  - Table 9: Ablation over non-detection strategies

Usage:
    python run_simulation.py --config config.yaml
"""
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List

from config import load_config, Config
from simulation import (
    run_vote_simulation,
    ablation_detector_accuracy,
    ablation_non_detection_strategies,
    AttackOutcome,
)


def save_outcomes(outcomes: List[AttackOutcome], path: str):
    """Save attack outcomes to JSON."""
    data = []
    for o in outcomes:
        data.append({
            "target_model": o.target_model,
            "current_rank": o.current_rank,
            "total_votes": o.total_votes,
            "votes_to_ranks": {str(k): v for k, v in o.votes_to_ranks.items()},
            "interactions_to_ranks": {str(k): v for k, v in o.interactions_to_ranks.items()},
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def print_outcome_table(outcomes: List[AttackOutcome], label: str = "Models"):
    """Print results in a format similar to Tables 4 and 5."""
    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")

    # Collect all target ranks
    all_target_ranks = set()
    for o in outcomes:
        all_target_ranks.update(o.votes_to_ranks.keys())
    all_target_ranks = sorted(all_target_ranks)

    # Header
    header = f"{'Target Model':<35} {'Rank':>5} {'Votes':>7} "
    for tr in all_target_ranks:
        header += f" Rank {tr:>3} "
    print(header)
    print("-" * len(header))

    # Rows
    for o in outcomes:
        row = f"{o.target_model:<35} {o.current_rank:>5} {o.total_votes:>7} "
        for tr in all_target_ranks:
            votes = o.votes_to_ranks.get(tr, "N/A")
            if isinstance(votes, int):
                row += f" {votes:>8} "
            else:
                row += f" {'N/A':>8} "
        print(row)

    # Interactions
    print(f"\n  Interactions required:")
    header2 = f"{'Target Model':<35} {'Rank':>5} "
    for tr in all_target_ranks:
        header2 += f" Rank {tr:>3} "
    print(header2)
    print("-" * len(header2))
    for o in outcomes:
        row = f"{o.target_model:<35} {o.current_rank:>5} "
        for tr in all_target_ranks:
            interactions = o.interactions_to_ranks.get(tr, "N/A")
            if isinstance(interactions, int):
                row += f" {interactions:>8} "
            else:
                row += f" {'N/A':>8} "
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Run adversarial voting simulations")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs/simulation", help="Output directory")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "tables", "ablation_accuracy", "ablation_strategies"],
                        help="Which experiment to run")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.name for m in config.models]

    high_ranked_models = [
        "chatgpt-4o-latest",
        "gemini-1.5-pro-exp-0801",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "claude-3-5-sonnet-20240620",
    ]
    high_ranked_models = [m for m in high_ranked_models if m in model_names]
    if not high_ranked_models:
        high_ranked_models = model_names[:5]

    low_ranked_models = model_names[-5:] if len(model_names) >= 5 else model_names

    if args.experiment in ("all", "tables"):
        print("\n" + "=" * 80)
        print("  TABLE 4: High-ranked models")
        print("=" * 80)

        # UP attack
        print("\n--- Up(M, x): Rise x positions ---")
        outcomes_up = run_vote_simulation(
            model_names=model_names,
            target_models=high_ranked_models,
            attack_type="up",
            target_deltas=[1, 2, 3, 4, 5],
            detector_accuracy=config.simulation.detector_accuracy,
            false_positive_rate=config.simulation.false_positive_rate,
            false_negative_rate=config.simulation.false_negative_rate,
            non_detection_strategy="do_nothing",
            bradley_terry_scale=config.simulation.bradley_terry_scale,
            steps_per_check=config.simulation.steps_per_check,
        )
        print_outcome_table(outcomes_up, "HIGH-RANKED MODELS - UP ATTACK")
        save_outcomes(outcomes_up, str(output_dir / "table4_up.json"))

        # DOWN attack
        print("\n--- Down(M, x): Fall x positions ---")
        outcomes_down = run_vote_simulation(
            model_names=model_names,
            target_models=high_ranked_models,
            attack_type="down",
            target_deltas=[1, 2, 3, 4, 5],
            detector_accuracy=config.simulation.detector_accuracy,
            false_positive_rate=config.simulation.false_positive_rate,
            false_negative_rate=config.simulation.false_negative_rate,
            non_detection_strategy="do_nothing",
            bradley_terry_scale=config.simulation.bradley_terry_scale,
            steps_per_check=config.simulation.steps_per_check,
        )
        print_outcome_table(outcomes_down, "HIGH-RANKED MODELS - DOWN ATTACK")
        save_outcomes(outcomes_down, str(output_dir / "table4_down.json"))

        print("\n" + "=" * 80)
        print("  TABLE 5: Low-ranked models")
        print("=" * 80)

        outcomes_low = run_vote_simulation(
            model_names=model_names,
            target_models=low_ranked_models,
            attack_type="up",
            target_deltas=[1, 2, 5, 10, 20, 50],
            detector_accuracy=config.simulation.detector_accuracy,
            false_positive_rate=config.simulation.false_positive_rate,
            false_negative_rate=config.simulation.false_negative_rate,
            non_detection_strategy="do_nothing",
            bradley_terry_scale=config.simulation.bradley_terry_scale,
            steps_per_check=config.simulation.steps_per_check,
        )
        print_outcome_table(outcomes_low, "LOW-RANKED MODELS - UP ATTACK")
        save_outcomes(outcomes_low, str(output_dir / "table5.json"))

    if args.experiment in ("all", "ablation_accuracy"):
        print("\n" + "=" * 80)
        print("  TABLE 8: Ablation - Detector Accuracy")
        print("=" * 80)

        target = low_ranked_models[-1] if low_ranked_models else model_names[-1]
        for acc in [1.0, 0.95, 0.9]:
            fnr = fpr = 1.0 - acc
            outcomes = run_vote_simulation(
                model_names=model_names,
                target_models=[target],
                attack_type="up",
                target_deltas=[1, 2, 5, 10, 20, 50],
                detector_accuracy=acc,
                false_positive_rate=fpr,
                false_negative_rate=fnr,
                non_detection_strategy="do_nothing",
                bradley_terry_scale=config.simulation.bradley_terry_scale,
                steps_per_check=config.simulation.steps_per_check,
            )
            print(f"\nDetector accuracy = {acc}")
            print_outcome_table(outcomes, f"ABLATION ACC={acc}")
            save_outcomes(outcomes, str(output_dir / f"table8_acc_{acc}.json"))

    if args.experiment in ("all", "ablation_strategies"):
        print("\n" + "=" * 80)
        print("  TABLE 9: Ablation - Non-Detection Strategies")
        print("=" * 80)

        for strategy in config.simulation.non_detection_strategies:
            # High-ranked
            target_high = high_ranked_models[-1] if high_ranked_models else model_names[0]
            outcomes = run_vote_simulation(
                model_names=model_names,
                target_models=[target_high],
                attack_type="up",
                target_deltas=[1, 2, 3, 4],
                detector_accuracy=config.simulation.detector_accuracy,
                false_positive_rate=config.simulation.false_positive_rate,
                false_negative_rate=config.simulation.false_negative_rate,
                non_detection_strategy=strategy,
                bradley_terry_scale=config.simulation.bradley_terry_scale,
                steps_per_check=config.simulation.steps_per_check,
            )
            print(f"\nStrategy: {strategy} (High-ranked: {target_high})")
            print_outcome_table(outcomes, f"NON-DETECT STRATEGY={strategy} (HIGH)")
            save_outcomes(outcomes, str(output_dir / f"table9_high_{strategy}.json"))

            # Low-ranked
            target_low = low_ranked_models[-1] if low_ranked_models else model_names[-1]
            outcomes = run_vote_simulation(
                model_names=model_names,
                target_models=[target_low],
                attack_type="up",
                target_deltas=[1, 2, 5, 10, 20, 50],
                detector_accuracy=config.simulation.detector_accuracy,
                false_positive_rate=config.simulation.false_positive_rate,
                false_negative_rate=config.simulation.false_negative_rate,
                non_detection_strategy=strategy,
                bradley_terry_scale=config.simulation.bradley_terry_scale,
                steps_per_check=config.simulation.steps_per_check,
            )
            print(f"\nStrategy: {strategy} (Low-ranked: {target_low})")
            print_outcome_table(outcomes, f"NON-DETECT STRATEGY={strategy} (LOW)")
            save_outcomes(outcomes, str(output_dir / f"table9_low_{strategy}.json"))

    print("\nAll simulation experiments completed.")


if __name__ == "__main__":
    main()
