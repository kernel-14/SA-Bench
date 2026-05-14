## main.py
import torch
import argparse
from utilities import Utilities
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation


class Main:
    """
    Entry point of the NGPT experimental framework. Parses configuration, orchestrates dataset loading,
    model initialization, training, evaluation, and checkpointing.
    """

    def __init__(self, config_path: str):
        """
        Initializes the Main class by loading the configuration file and setting up components.

        Args:
            config_path (str): File path to the configuration file (config.yaml).
        """
        # Load configuration
        self.config = Utilities.get_config(config_path)

        # Initialize dataset loader
        self.dataset_loader = DatasetLoader(self.config)

        # Initialize the model
        self.model = Model(self.config["model"])

        # Prepare optimizer and training parameters
        self.optimizer = None

        # Placeholder for training and validation data
        self.train_data = None
        self.validation_data = None

    def run_experiment(self):
        """
        High-level execution of the NGPT experimental workflow.
        """
        # 1. Load and preprocess dataset
        self._load_and_preprocess_dataset()

        # 2. Initialize optimizer and training components
        trainer = self._setup_training()

        # 3. Train the model
        trainer.train(epochs=self.config["training"]["epochs"])

        # 4. Evaluate the model
        self._evaluate_model()

    def _load_and_preprocess_dataset(self):
        """
        Loads, tokenizes, and preprocesses the dataset using the DatasetLoader.
        """
        print("Loading and preprocessing dataset...")
        raw_dataset = self.dataset_loader.load_data()

        # Preprocessing train and validation datasets
        self.train_data = self.dataset_loader.process_data(raw_dataset["train"])
        self.validation_data = self.dataset_loader.process_data(raw_dataset["validation"])
        print("Dataset loading and preprocessing completed.")

    def _setup_training(self) -> Trainer:
        """
        Sets up the optimizer and returns the Trainer object.

        Returns:
            Trainer: Configured Trainer instance.
        """
        print("Setting up training...")

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )

        # Create Trainer instance
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            data=(self.train_data, self.validation_data),
            config=self.config
        )
        trainer.setup_optimizer()
        print("Training setup completed.")
        return trainer

    def _evaluate_model(self):
        """
        Performs model evaluation, generating performance metrics.
        """
        print("Evaluating the model...")
        evaluation = Evaluation(
            model=self.model,
            data=self.validation_data,
            config=self.config
        )
        results = evaluation.evaluate()

        # Pretty-print results
        print("\nEvaluation Results:")
        for metric_name, metric_value in results.items():
            print(f"{metric_name}: {metric_value:.4f}")

        print("Model evaluation completed.")

    def _checkpoint_model(self, save_path: str):
        """
        Saves the model state, optimizer state, and scaling parameters to a checkpoint.

        Args:
            save_path (str): Path to save the checkpoint.
        """
        print(f"Saving checkpoint to {save_path}...")
        Utilities.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scaling_factors={
                "alpha_a": self.model.alpha_a.detach().cpu(),
                "alpha_m": self.model.alpha_m.detach().cpu(),
                "s_z": self.model.s_z.detach().cpu()
            },
            save_path=save_path
        )
        print("Checkpoint saved.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run NGPT experiment.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Instantiate and run the pipeline
    main = Main(args.config)
    main.run_experiment()
