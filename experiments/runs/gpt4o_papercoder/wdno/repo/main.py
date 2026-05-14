# main.py
"""
Main entry-point module for Wavelet Diffusion Neural Operator (WDNO).

Handles configuration loading, data preparation, model initialization, training, evaluation,
and super-resolution tasks based on the workflow and mode selected.
"""

import argparse
import sys
from typing import Optional
import torch

from config import Config
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluator import Evaluator


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for specifying mode of operation and configuration file path.

    Returns:
        argparse.Namespace: Parsed arguments containing mode and configuration file path.
    """
    parser = argparse.ArgumentParser(description="Wavelet Diffusion Neural Operator (WDNO)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate_simulation",
                                                                    "evaluate_control", "super_resolution"],
                        help="Execution mode: train, evaluate_simulation, evaluate_control, super_resolution.")
    parser.add_argument("--output", type=str, default=None, help="Optional path for saving results (evaluation mode).")
    return parser.parse_args()


def print_metrics(metrics: dict, title: str = "Evaluation Results") -> None:
    """
    Prints metrics in a formatted output.

    Args:
        metrics (dict): Dictionary containing computed metrics.
        title (str): Title to display before the metrics.
    """
    print(f"\n=== {title} ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


def save_results_to_file(metrics: dict, output_path: str) -> None:
    """
    Saves evaluation results to a file.

    Args:
        metrics (dict): Dictionary containing computed metrics.
        output_path (str): File path to save the results.
    """
    try:
        with open(output_path, "w") as f:
            f.write("=== Evaluation Results ===\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.6f}\n")
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")


def main() -> None:
    """
    Main program workflow for WDNO, integrating configuration, data loading, model handling, training, and evaluation.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)

    # Load datasets
    try:
        dataset_loader = DatasetLoader(config.get("data"))
    except Exception as e:
        print(f"Error initializing dataset loader: {e}")
        sys.exit(1)

    # Initialize model
    try:
        model = Model(config.get("training"))
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

    # Mode-specific operations
    if args.mode == "train":
        # Training mode
        try:
            data = dataset_loader.load_data()
            trainer = Trainer(model, data, config.get("training"))
            trainer.train()
        except Exception as e:
            print(f"Error during training: {e}")
            sys.exit(1)

    elif args.mode in ["evaluate_simulation", "evaluate_control"]:
        # Evaluation mode
        try:
            data = dataset_loader.load_data()
            evaluator = Evaluator(model, data, config.get("evaluation"))

            if args.mode == "evaluate_simulation":
                metrics = evaluator.evaluate_simulation()
            else:
                metrics = evaluator.evaluate_control()

            print_metrics(metrics, title="Evaluation Metrics")
            if args.output:
                save_results_to_file(metrics, args.output)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            sys.exit(1)

    elif args.mode == "super_resolution":
        # Super-resolution mode
        try:
            super_res_data = dataset_loader.prepare_super_resolution_data(dataset_loader.load_data())
            evaluator = Evaluator(model, super_res_data, config.get("evaluation"))
            metrics = evaluator.evaluate_simulation()  # Assumes evaluation on generated resolutions

            print_metrics(metrics, title="Super-Resolution Metrics")
            if args.output:
                save_results_to_file(metrics, args.output)
        except Exception as e:
            print(f"Error during super-resolution: {e}")
            sys.exit(1)

    else:
        print(f"Invalid mode selected: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
