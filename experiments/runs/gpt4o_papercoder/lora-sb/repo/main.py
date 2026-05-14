"""
main.py
The central entry point for the LoRA-SB implementation. It orchestrates dataset loading,
model initialization, training, and evaluation using the design previously specified.
"""

import os
import yaml
import logging
from typing import Any, Dict, Tuple

import torch
from dataset_loader import DatasetLoader
from model import LoRA_SB_Model
from trainer import Trainer
from evaluation import Evaluator
from utils import set_random_seed, set_device


class Main:
    """
    Main Class: Coordinates the execution of the LoRA-SB workflow, including dataset handling,
    model initialization, training, and evaluation.
    """

    def __init__(self, config_file: str = "config.yaml") -> None:
        """
        Initialize the Main class, reading configuration and setting up logging.

        Args:
            config_file (str): Path to the configuration file (default: "config.yaml").
        """
        self.config = self._load_config(config_file)
        self._setup_logging()
        set_random_seed(42)  # Ensure reproducibility
        self.device = set_device(self.config["hardware"]["use_gpu"])

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load YAML configuration file.

        Args:
            config_file (str): Path to the YAML config file.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            logging.info(f"Configuration loaded from {config_file}")
            return config

    def _setup_logging(self) -> None:
        """
        Set up logging with INFO level and timestamp formatting.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [Main] %(message)s",
            handlers=[
                logging.StreamHandler()
            ]
        )
        logging.info("Logging is successfully configured.")

    def run(self) -> None:
        """
        The main execution flow: loads data, initializes the model, trains it, and evaluates performance.
        """
        logging.info("Starting execution of the LoRA-SB pipeline.")

        # Step 1: Load datasets
        dataset_name = self.config["dataset"]["name"]
        logging.info(f"Loading dataset: {dataset_name}")
        dataset_loader = DatasetLoader(dataset_name, self.config)
        train_loader, val_loader, test_loader = dataset_loader.load_data()

        # Step 2: Initialize the LoRA-SB model
        model_config = self.config["model"]
        base_model_name = model_config["base_name"]
        rank = model_config["rank"]
        scaling_factor = model_config["scaling_factor"]
        logging.info(f"Initializing LoRA-SB model with base model: {base_model_name}, rank: {rank}, "
                     f"scaling factor: {scaling_factor}.")
        model = LoRA_SB_Model(base_model_name=base_model_name, rank=rank[0], scaling_factor=scaling_factor)
        
        # Step 3: Compute initial gradients and update LoRA-SB matrices
        init_samples = self.config["dataset"]["num_init_samples"]
        logging.info(f"Running gradient computation for initialization with {init_samples} samples.")
        trainer = Trainer(model, (train_loader, val_loader), self.config)
        B_init, A_init, R_init = trainer.compute_gradients(init_samples)
        model.initialize_with_svd(torch.stack([B_init, A_init, R_init]), num_samples=init_samples)

        # Step 4: Fine-tune model
        logging.info("Starting training of LoRA-SB model.")
        trainer.train()

        # Step 5: Evaluate model
        evaluation_metrics = self.config["evaluation"]["metrics"]
        logging.info(f"Evaluating model using metrics: {evaluation_metrics}")
        evaluator = Evaluator(model, test_loader, evaluation_metrics)
        evaluation_results = evaluator.evaluate()

        # Step 6: Log evaluation results
        logging.info("Evaluation results:")
        for metric, result in evaluation_results.items():
            logging.info(f"  {metric}: {result:.4f}")

        logging.info("Execution of the LoRA-SB pipeline completed successfully.")


if __name__ == "__main__":
    # Instantiate and execute the main pipeline
    main_pipeline = Main(config_file="config.yaml")
    main_pipeline.run()
