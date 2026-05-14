# main.py
import os
import yaml
import json
from dataset_loader import DatasetLoader
from model import TransformerModel
from trainer import Trainer
from evaluation import Evaluation

class Main:
    """
    Orchestrates the entire experiment pipeline:
    - Loads configuration from YAML file.
    - Initializes dataset loader, model, trainer, and evaluator.
    - Conducts training and evaluation.
    - Logs and saves results.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initializes the Main class and loads the experiment configuration.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)

        # Parse configuration for modular access across components
        self.training_config = self.config["training"]
        self.model_config = self.config["model"]
        self.dataset_config = self.config["dataset"]
        self.evaluation_config = self.config["evaluation"]
        self.optimization_config = self.config["optimization"]

    def _load_config(self, config_path: str):
        """
        Loads the YAML configuration file.

        Args:
            config_path (str): Path to YAML configuration file.

        Returns:
            dict: Parsed configuration dictionary.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def run_experiment(self):
        """
        Executes the full experiment pipeline:
        - Loads datasets.
        - Initializes and trains the model.
        - Evaluates the model on specified tasks.
        - Saves the evaluation results.
        """
        # Step 1: Load datasets
        print("Loading datasets...")
        dataset_loader = DatasetLoader(self.dataset_config)
        training_data = dataset_loader.load_training_data()
        evaluation_data = dataset_loader.load_evaluation_data()

        # Step 2: Initialize the model
        print("Initializing the model...")
        transformer_model = TransformerModel(
            base_model=self.model_config["type"],
            num_heads=self.model_config["num_heads"],
            gating_config=self.model_config["gating_config"]
        )

        # Step 3: Train the model
        print("Starting training...")
        trainer = Trainer(
            model=transformer_model,
            data=training_data,
            config=self.training_config
        )
        trainer.train()

        # Save the trained model
        print("Saving the trained model...")
        os.makedirs("saved_models", exist_ok=True)
        trainer.save_model(output_path=os.path.join("saved_models", "final_model.pt"))

        # Step 4: Evaluate the model
        print("Starting evaluation...")
        evaluator = Evaluation(
            model=transformer_model,
            data=evaluation_data,
            metrics=self.evaluation_config["metrics"],
            config=self.config
        )
        evaluation_results = evaluator.evaluate()

        # Step 5: Save evaluation results
        print("Saving evaluation results...")
        os.makedirs("results", exist_ok=True)
        results_path = os.path.join("results", "evaluation_results.json")
        with open(results_path, "w") as result_file:
            json.dump(evaluation_results, result_file, indent=4)
        print(f"Evaluation results saved to {results_path}")

if __name__ == "__main__":
    # Main entry point for the experiment
    main = Main(config_path="configs/config.yaml")
    main.run_experiment()
