## main.py

import os
import yaml
from typing import Dict
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation

class Main:
    """
    Main orchestrator for implementing, training, and evaluating the SC-FNO framework.
    This class handles the pipeline flow, integrates components, and ensures reproducibility.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the experimental pipeline using the configuration file.

        Args:
        - config_path (str): Path to the configuration YAML file.
        """
        self.config = self._load_config(config_path)

        # Ensure output directories are ready
        self.output_dir = self.config.get("experiment", {}).get("output_dir", "results/")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize dataset loader
        self.dataset_loader = DatasetLoader(config_path=config_path)

        # Initialize placeholders for training components
        self.model = None
        self.trainer = None
        self.evaluation_handler = None

    def run_experiment(self):
        """
        Execute the full experiment pipeline:
        Dataset preparation, model training, and validation.
        """
        try:
            print("==== Starting Experiment ====")
            # Step 1: Dataset Preparation
            print("Loading and preparing datasets...")
            equation_name = self.config["experiment"]["equations"]["ode1"]["name"]  # Default first experiment
            print(f"Target equation: {equation_name}")
            
            # Load data and gradients
            inputs, solutions, gradients = self.dataset_loader.load_data(equation_name)
            if gradients is None:
                print(f"Precomputing gradients using method: {self.dataset_loader.gradient_method}")
                gradients = self.dataset_loader.precompute_gradients()

            # Split data into train, validation, and test sets
            train_loader, val_loader, test_loader = self.dataset_loader.split_data()

            # Step 2: Model Initialization
            print("Initializing model...")
            input_dims = (inputs.shape[1],)
            output_dims = (solutions.shape[1],)
            model_params = self.config["model"]
            self.model = Model(
                input_dims=input_dims,
                output_dims=output_dims,
                fourier_modes=model_params.get("fourier_modes", 8),
                width=model_params.get("width", 20)
            )

            # Step 3: Trainer Setup
            print("Setting up the training pipeline...")
            self.trainer = Trainer(
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=self.config
            )

            # Step 4: Training Execution
            print("Training the model...")
            self.trainer.train()

            print("Training complete. Validating results...")
        except Exception as e:
            print(f"Exception during run_experiment: {e}")
            raise

    def evaluate_experiment(self):
        """
        Evaluate the trained model on the test dataset and compute metrics.
        """
        try:
            print("==== Starting Evaluation ====")
            # Ensure the model and test data are prepared
            if not self.model:
                raise RuntimeError("Model not initialized. Ensure you run the experiment first.")
            if not self.dataset_loader:
                raise RuntimeError("Dataset not initialized. Ensure dataset is loaded during the experiment.")

            # Initialize evaluation handler if not already done
            if not self.evaluation_handler:
                print("Initializing evaluation handler...")
                _, _, test_loader = self.dataset_loader.split_data()
                self.evaluation_handler = Evaluation(
                    model=self.model, test_loader=test_loader, config_path="config/config.yaml"
                )

            # Evaluate metrics for solutions and sensitivities
            print("Computing evaluation metrics...")
            metrics = self.evaluation_handler.evaluate_metrics()
            print("Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.6f}")

            # Generalization evaluation
            print("Assessing generalization ability...")
            generalization_metrics = self.evaluation_handler.evaluate_generalization()
            print("Generalization Metrics:")
            for key, value in generalization_metrics.items():
                print(f"  {key}: {value:.6f}")

            # Visualizations
            print("Generating evaluation visualizations...")
            self.evaluation_handler.generate_visualizations(metrics)
            print(f"Results saved in: {self.output_dir}")

        except Exception as e:
            print(f"Exception during evaluate_experiment: {e}")
            raise

    def _load_config(self, config_path: str) -> Dict:
        """
        Load and validate the configuration file.

        Args:
        - config_path (str): Path to the configuration YAML file.

        Returns:
        - Dict: Parsed and validated configuration dictionary.
        """
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

            with open(config_path, "r") as file:
                config = yaml.safe_load(file)

            # Validate required fields
            required_fields = ["training", "model", "loss_weights", "data", "experiment"]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required configuration field: '{field}'")

            print(f"Configuration loaded successfully from {config_path}")
            return config
        except Exception as e:
            print(f"Error loading config file: {e}")
            raise

if __name__ == "__main__":
    # Instantiate Main with default config
    main = Main()

    # Run experiment (training)
    main.run_experiment()

    # Evaluate experiment (testing)
    main.evaluate_experiment()
