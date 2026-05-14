"""
main.py

Entry point for the SAM 2 pipeline, orchestrating dataset preparation, model initialization,
training, and evaluation routines. This module ties together components and ensures the
reproducibility of experiments as described in the SAM 2 paper.
"""

import os
import argparse
import torch
import yaml
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation


class Main:
    """
    Orchestrates the SAM 2 pipeline: loading configuration, preparing datasets, initializing the model,
    training, and evaluation.
    """

    def __init__(self) -> None:
        """
        Initialize the Main class.
        """
        self.config = None

    def load_config(self, config_path: str) -> dict:
        """
        Load the configuration from the provided YAML file.

        Args:
            config_path (str): Path to the configuration YAML file.

        Returns:
            dict: Parsed configuration dictionary.
        """
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        return self.config

    def validate_config(self) -> None:
        """
        Validate that the necessary configurations are present.

        Raises:
            KeyError: If a required field is missing.
        """
        required_keys = ['training', 'data', 'model', 'evaluation', 'device', 'logging']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required configuration section: '{key}'")

    def run_experiment(self, config_path: str) -> None:
        """
        Main runner function for the SAM 2 pipeline.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        # Step 1: Load and validate the configuration
        print("[INFO] Loading configuration...")
        config = self.load_config(config_path)
        self.validate_config()

        # Step 2: Prepare datasets
        print("[INFO] Initializing datasets...")
        dataset_loader = DatasetLoader(config)
        train_loader, val_loader, test_loader = dataset_loader.load_data()

        # Step 3: Initialize the model
        print("[INFO] Initializing the model...")
        model = Model(
            backbone=config['model']['backbone'],
            memory_size=config['model']['memory_size'],
            prompt_types=config['model']['prompt_types']
        )

        # Load pre-trained weights if available
        pretrained_weights = config['model'].get('pretrained_weights', None)
        if pretrained_weights and os.path.exists(pretrained_weights):
            print(f"[INFO] Loading pre-trained weights from {pretrained_weights}...")
            model.load_state_dict(torch.load(pretrained_weights))
        else:
            print("[WARNING] No pre-trained weights were provided or found. Starting from scratch.")

        # Step 4: Train the model
        print("[INFO] Starting training...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        trainer.train()

        # Step 5: Evaluate the model
        print("[INFO] Starting evaluation...")
        evaluator = Evaluation(
            model=model,
            test_loader=test_loader,
            config=config
        )
        print("[INFO] Running zero-shot evaluation...")
        zero_shot_results = evaluator.evaluate_zero_shot()
        print("[INFO] Zero-shot results:", zero_shot_results)

        print("[INFO] Running interactive evaluation...")
        interactive_results = evaluator.evaluate_interactive()
        print("[INFO] Interactive evaluation results:", interactive_results)

        # Finalize outputs
        print("[INFO] Experiment completed successfully.")


if __name__ == "__main__":
    # Argument parser for command-line configuration path input
    parser = argparse.ArgumentParser(description="Run SAM 2: Segment Anything in Images and Videos")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    # Run the SAM 2 pipeline
    main = Main()
    main.run_experiment(config_path=args.config)
