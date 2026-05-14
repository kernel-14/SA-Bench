## main.py

import os
import yaml
from typing import Dict
from dataset_loader import DatasetLoader
from fr_vae import FRVAE
from transformer import TransformerGenerator
from trainer import Trainer
from evaluation import Evaluation


class Main:
    """Primary entry point for running the NFIG framework pipeline."""

    def __init__(self, config_file: str = "config.yaml"):
        """
        Initializes the main pipeline with a given configuration file.

        Args:
            config_file (str): Path to the YAML configuration file. Default: "config.yaml".
        """
        # Parse configuration
        with open(config_file, "r") as file:
            self.config: Dict = yaml.safe_load(file)

        # Validate configuration keys
        required_sections = ["training", "dataset", "model", "evaluation"]
        for section in required_sections:
            if section not in self.config:
                raise KeyError(f"Missing required section '{section}' in config.yaml")

        # Extract configuration values
        self.train_split = self.config["dataset"]["train_split"]
        self.val_split = self.config["dataset"]["val_split"]
        self.test_split = self.config["dataset"]["test_split"]
        self.resolution = self.config["dataset"]["resolution"]
        self.epochs = self.config["training"]["epochs"]
        self.batch_size = self.config["training"]["batch_size"]

        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize pipeline components
        print("Initializing DatasetLoader...")
        self.dataset_loader = DatasetLoader(self.config)

        print("Initializing FRVAE...")
        encoder_params = self.config["model"]["fr_vae"]["encoder_params"]
        decoder_params = self.config["model"]["fr_vae"]["decoder_params"]
        codebook_size = self.config["model"]["fr_vae"]["codebook_size"]
        self.fr_vae = FRVAE(encoder_params, decoder_params, codebook_size)

        print("Initializing TransformerGenerator...")
        vocab_size = self.config["model"]["transformer_generator"]["vocab_size"]
        depth = self.config["model"]["transformer_generator"]["depth"]
        head_dim = self.config["model"]["transformer_generator"]["head_dim"]
        self.generator = TransformerGenerator(vocab_size, depth, head_dim)

        print("Initializing Trainer...")
        self.trainer = Trainer(
            fr_vae=self.fr_vae,
            generator=self.generator,
            dataset_loader=self.dataset_loader,
            config=self.config,
        )

        print("Initializing Evaluation...")
        metrics = self.config["evaluation"]["metrics"]
        self.evaluator = Evaluation(generator=self.generator, dataset_loader=self.dataset_loader, metrics=metrics)

    def run_experiment(self) -> None:
        """
        Executes the full pipeline: training, evaluation, and visualization.

        Steps include:
        1. Training FRVAE and Transformer models.
        2. Evaluating the trained models on specified metrics.
        3. Visualizing results.
        """
        print("Starting the NFIG experiment pipeline...")

        # Step 1: Train the model
        print("\n=== Training ===")
        self.trainer.train()

        # Step 2: Evaluate the trained model
        print("\n=== Evaluating ===")
        metrics = self.evaluator.evaluate()
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Step 3: Visualize results
        print("\n=== Visualizing Results ===")
        visualization_dir = os.path.join(self.output_dir, "visualizations")
        self.evaluator.visualize_results(output_dir=visualization_dir)
        print(f"Visualizations saved in: {visualization_dir}")

        print("\n=== Experiment Completed ===")


# Entry point for running the system
if __name__ == "__main__":
    main = Main(config_file="config.yaml")
    main.run_experiment()
