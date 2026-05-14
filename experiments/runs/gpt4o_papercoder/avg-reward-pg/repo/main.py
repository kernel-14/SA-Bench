"""
main.py

Entry point for the experiment pipeline. This file orchestrates the generation of MDPs,
training of policies, and evaluation of metrics as outlined in the paper and experimental design.
It integrates all modules into a cohesive pipeline.
"""

import yaml
import os
from typing import Dict, Any

from mdp_generator import MDPGenerator
from policy_gradient import PolicyGradient
from trainer import Trainer
from evaluation import Evaluation
from utils.constants import validate_configurations, get_hyperparameters


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads and parses the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' is not found.")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def log_config_details(config: Dict[str, Any]) -> None:
    """
    Logs the configuration details.

    Args:
        config (dict): Configuration dictionary.
    """
    print("Experiment Configuration:")
    print("-------------------------")
    for section, settings in config.items():
        print(f"{section}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    print("-------------------------")


def run_experiment(config: Dict[str, Any]) -> None:
    """
    Executes the training and evaluation pipeline based on the provided configuration.

    Args:
        config (Dict[str, Any]): Experiment configuration dictionary.
    """
    # Load hyperparameters from config.yaml
    hyperparams = get_hyperparameters()

    # Initialize MDPGenerator
    mdp_generator = MDPGenerator(config["mdp"])

    # Loop through state space, action space, and reward variance levels
    for state_size in config["mdp"]["state_space_sizes"]:
        for action_size in config["mdp"]["action_space_sizes"]:
            for reward_variance in config["mdp"]["reward_variance_levels"]:
                print(f"Running experiment for state_size={state_size}, action_size={action_size}, reward_variance={reward_variance}")

                # Step 1: Generate MDP
                mdp = mdp_generator.generate_mdp(state_size=state_size, action_size=action_size, reward_variance=reward_variance)

                # Step 2: Initialize PolicyGradient
                policy_gradient = PolicyGradient(learning_rate=hyperparams["learning_rate"])

                # Step 3: Initialize Trainer
                trainer = Trainer(policy_gradient=policy_gradient, mdp=mdp, num_iterations=hyperparams["num_iterations"])

                # Step 4: Train the policy
                results = trainer.train()

                # Step 5: Evaluate the final policy
                evaluator = Evaluation(mdp)
                evaluation_metrics = evaluator.evaluate(policy=results["final_policy"])

                # Step 6: Plot convergence trends
                metrics = {
                    "average_rewards": results["rewards"],
                    "suboptimality_gaps": results["suboptimality_gaps"]
                }
                evaluator.plot_convergence(metrics)

                print(f"Experiment completed for state_size={state_size}, action_size={action_size}, reward_variance={reward_variance}")
                print(f"Final Average Reward: {evaluation_metrics['average_reward']:.4f}")
                print(f"Suboptimality Gap: {evaluation_metrics['average_reward']:.4f}\n")


def main():
    """
    Main entry point for the experiment pipeline.
    """
    # Load configuration
    config = load_config("config.yaml")

    # Validate configuration before starting experiments
    validate_configurations()

    # Log configuration details for reproducibility
    log_config_details(config)

    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()
