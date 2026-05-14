# main.py
import os
import yaml
import torch
from typing import Dict

from dataset_loader import DatasetLoader
from rwm_model import RWMModel
from trainer import Trainer
from policy_optimizer import PolicyOptimizer
from evaluation import Evaluation


def load_config(config_path: str) -> Dict:
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the config.yaml file.

    Returns:
        Dict: Parsed configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def run_pipeline(config_path: str) -> None:
    """
    Main function to execute the pipeline: data loading, training, policy optimization, and evaluation.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)

    # Initialize DatasetLoader
    print("Initializing dataset...")
    dataset_loader = DatasetLoader(config=config, dataset_path="datasets/")
    dataset = dataset_loader.load_data()

    # Initialize Robotic World Model (RWM)
    print("Initializing the Robotic World Model (RWM)...")
    rwm_model = RWMModel(
        history_horizon=config["training"]["history_horizon"],
        forecast_horizon=config["training"]["forecast_horizon"],
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        output_dim=config["model"]["output_dim"],
        privileged_dim=config["model"]["output_dim"],  # Assuming equal to the output_dim for privileged
        decay_factor=config["training"]["forecast_decay"],
    )

    # Train the Robotic World Model
    print("Starting RWM training...")
    trainer = Trainer(
        model=rwm_model,
        data=dataset,
        config=config,
    )
    trainer.train()

    # Optimize policy using MBPO-PPO
    print("Initializing policy optimizer...")
    policy_optimizer = PolicyOptimizer(
        model=rwm_model,
        config=config,
        buffer=[]
    )
    print("Starting policy optimization...")
    policy_optimizer.train_policy()

    # Evaluate the model and policy
    print("Starting evaluation...")
    evaluation = Evaluation(
        model=rwm_model,
        trained_policy=policy_optimizer.policy,
        config=config,
    )

    # Evaluate autoregressive prediction accuracy
    prediction_metrics = evaluation.evaluate_prediction(test_loader=dataset["test"])
    print(f"Prediction Metrics: {prediction_metrics}")

    # Evaluate the optimized policy
    policy_metrics = evaluation.evaluate_policy(buffer=policy_optimizer.buffer)
    print(f"Policy Metrics: {policy_metrics}")

    # Visualize results
    print("Visualizing results...")
    evaluation.visualize_rollouts(test_loader=dataset["test"])

    print("Pipeline execution completed!")


if __name__ == "__main__":
    # Default configuration file path
    CONFIG_PATH = "config.yaml"

    # Run the pipeline
    run_pipeline(config_path=CONFIG_PATH)
