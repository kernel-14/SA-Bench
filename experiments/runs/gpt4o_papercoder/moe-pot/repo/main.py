## main.py

import sys
from utilities import load_yaml_config, log_metrics, set_random_seeds
from dataset_loader import DatasetLoader
from moe_pot_model import MoEPOTModel
from trainer import Trainer
from evaluation import Evaluation
import argparse
import os


class Main:
    """
    Main orchestrator for the MoE-POT experimental lifecycle. It handles
    configuration parsing, dataset initialization, model construction,
    training, and evaluation.
    """

    def __init__(self, config_path: str):
        """
        Initialize the Main class with the configuration path.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config_path = config_path
        self.config = self.load_config()
        set_random_seeds(self.config.get("random_seed", 42))

    def load_config(self) -> dict:
        """
        Load the YAML configuration file.

        Returns:
            dict: Parsed configuration settings.
        """
        try:
            config = load_yaml_config(self.config_path)
            print(f"Configuration loaded successfully from {self.config_path}")
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            sys.exit(1)
        return config

    def run_experiment(self) -> None:
        """
        Execute the entire lifecycle of the MoE-POT experiment, including:
        1. Pretraining.
        2. Fine-tuning.
        3. Evaluation.
        """
        # Step 1: Initialize the DatasetLoader
        dataset_loader = DatasetLoader(self.config)

        # Step 2: Load pretraining data
        print("Loading pretraining datasets...")
        train_data, val_data = dataset_loader.load_pretraining_data()

        # Step 3: Initialize the MoE-POT model
        print("Initializing the MoE-POT model...")
        model = MoEPOTModel(params=self.config["architecture"])
        print("MoE-POT model initialized.")

        # Step 4: Pretraining the model
        print("Starting pretraining...")
        trainer = Trainer(model=model, train_data=train_data, val_data=val_data, config=self.config)
        trainer.train()
        print("Pretraining completed.")

        # Step 5: Fine-tuning the model
        print("Loading fine-tuning datasets...")
        fine_tune_train, fine_tune_val = dataset_loader.load_finetune_data()
        print("Fine-tuning the pre-trained model...")
        fine_tune_trainer = Trainer(model=model, train_data=fine_tune_train, val_data=fine_tune_val, config=self.config)
        fine_tune_trainer.train()
        print("Fine-tuning completed.")

        # Step 6: Evaluate the model on downstream tasks
        print("Loading downstream datasets for evaluation...")
        test_data, eval_data = dataset_loader.load_downstream_data()
        evaluation = Evaluation(model=model, test_data=test_data, config=self.config)
        
        print("Evaluating L2 Relative Error (L2RE)...")
        l2re_metrics = evaluation.evaluate_l2re()
        
        print("Computing rollout error for error accumulation analysis...")
        rollout_metrics = evaluation.compute_rollout_error(initial_inputs=test_data, ground_truth=eval_data)
        
        print("Analyzing router-gating network behavior for interpretability...")
        interpretability_metrics = evaluation.analyze_router_gate_behavior()

        # Step 7: Log evaluation metrics
        print("Logging evaluation metrics...")
        metrics_log_path = os.path.join(self.config.get("log_dir", "./logs"), "final_metrics.json")
        log_metrics({**l2re_metrics, **rollout_metrics, **interpretability_metrics}, metrics_log_path)
        print(f"Metrics logged at {metrics_log_path}")

        print("Experiment completed successfully.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MoE-POT experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the experiment configuration YAML file.",
    )
    args = parser.parse_args()

    # Run the main experiment
    main = Main(config_path=args.config)
    main.run_experiment()
