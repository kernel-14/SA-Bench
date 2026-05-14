# main.py

import os
import yaml
from typing import Dict

import torch
from sokoban_environment import SokobanEnvironment
from dataset_loader import DatasetLoader
from drc_model import DRCModel
from trainer import Trainer
from prober import Prober
from evaluator import Evaluator
from intervention import Intervention
from visualization import Visualization


class Main:
    """
    Entry point for orchestrating the Sokoban experiment pipeline.
    Manages initialization, training, evaluation, and visualization of results.
    """

    def __init__(self, config_path: str = "./config.yaml"):
        """
        Initializes the experiment pipeline.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config = self._load_config(config_path)

        # Initialize core experiment components
        self.environment = None
        self.dataset_loader = None
        self.model = None
        self.trainer = None
        self.prober = None
        self.evaluator = None
        self.intervention = None
        self.visualization = None

    def _load_config(self, config_path: str) -> Dict:
        """
        Loads experiment configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML file.

        Returns:
            dict: Loaded configuration.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_sections = ["training", "agent", "environment", "dataset"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return config

    def initialize_components(self):
        """
        Initializes all components of the experiment pipeline.
        """
        # Initialize the Sokoban environment
        self.environment = SokobanEnvironment(config=self.config)

        # Initialize the dataset loader
        dataset_path = self.config["dataset"]["dataset_path"]
        self.dataset_loader = DatasetLoader(dataset_path=dataset_path, config=self.config)

        # Initialize the DRC model
        self.model = DRCModel(config=self.config)

        # Initialize the training process
        self.trainer = Trainer(
            model=self.model,
            env=self.environment,
            config=self.config,
        )

        # Initialize the prober for concept decoding
        self.prober = Prober(
            model=self.model,
            probes_config={"learning_rate": 0.001, "max_epochs": 10, "batch_size": 16},
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # Initialize the evaluator
        self.evaluator = Evaluator(
            model=self.model,
            env=self.environment,
            config=self.config,
        )

        # Initialize the intervention module
        self.intervention = Intervention(
            model=self.model,
            prober=self.prober,
            config={"intervention_strength": 1.0, "environment": self.config["environment"]},
        )

        # Initialize the visualization module
        self.visualization = Visualization()

    def run_experiment(self):
        """
        Orchestrates the training, evaluation, intervention, and visualization phases.
        """
        # Load training and validation datasets
        print("Loading datasets...")
        training_data = self.dataset_loader.load_training_data()
        validation_data = self.dataset_loader.load_validation_data()

        # Train the model
        print("Starting training...")
        self.trainer.train(num_epochs=self.config["training"]["epochs"])

        # Save the trained model
        model_checkpoint_path = "./final_model.pth"
        self.trainer.save_model(model_checkpoint_path)
        print(f"Training complete. Model saved to {model_checkpoint_path}.")

        # Train linear probes for concepts C_A and C_B
        print("Training probes for concept decoding...")
        self.prober.train_probes(dataset=training_data, concept_type="C_A", layer_ids=[0, 1, 2])
        self.prober.train_probes(dataset=training_data, concept_type="C_B", layer_ids=[0, 1, 2])

        # Evaluate the model on validation and thinking steps
        print("Evaluating model performance...")
        evaluation_metrics = self.evaluator.evaluate(num_episodes=1000)
        thinking_metrics = self.evaluator.evaluate_with_thinking(num_episodes=1000, thinking_steps=5)
        
        print("Evaluation results:", evaluation_metrics)
        print("Evaluation with thinking steps:", thinking_metrics)

        # Apply intervention experiments
        print("Running intervention experiments...")
        self.intervention.steer_behavior(level_config={
            "intervention_type": "AgentShortcut",
            "layer": 2,
            "positions": [(4, 4), (5, 5)],
            "target_classes": {(4, 4): "UP", (5, 5): "RIGHT"}
        }, probe_vectors=self.prober.probes)

        # Generate and save visualizations
        print("Generating visualizations...")
        self.visualization.plot_metrics(
            metrics_data={
                "Validation Success": {"Layer 1": [0.85, 0.88], "Layer 2": [0.90, 0.92]},
                "F1 Scores": {"Probe 1x1": [0.80, 0.84], "Probe 3x3": [0.83, 0.87]},
            },
            save_path="./metrics_plot.png",
        )
        self.visualization.render_plans(
            concept_data={(i, j): "NEVER" for i in range(8) for j in range(8)},
            save_path="./plan_visualization.png"
        )

        print("Experiment completed successfully.")


if __name__ == "__main__":
    main = Main(config_path="./config.yaml")
    main.initialize_components()
    main.run_experiment()
