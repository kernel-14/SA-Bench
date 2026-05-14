## main.py

import os
import torch
from config_loader import ConfigLoader
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation
from utils import set_random_seed, log_metrics


class Main:
    """
    Orchestrates pretraining, adaptation, and evaluation workflows for the Mixture-of-Experts model.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """
        Initialize the Main orchestrator with the specified configuration file path.
        :param config_path: Path to the configuration file.
        """
        self.config_path = config_path
        self.config = None
        self.dataset_loader = None
        self.model = None
        self.trainer = None
        self.evaluator = None

    def run_experiment(self) -> None:
        """
        Run the full experimental pipeline, including pretraining, adaptation, and evaluation.
        """
        try:
            print("\n[INFO]: Loading configuration...")
            self._load_configuration()

            print("\n[INFO]: Setting random seed...")
            self._set_random_seed()

            print("\n[INFO]: Loading pretraining dataset...")
            pretraining_data = self._load_pretraining_dataset()

            print("\n[INFO]: Initializing model...")
            self._initialize_model()

            print("\n[INFO]: Starting pretraining...")
            self._run_pretraining(pretraining_data)

            print("\n[INFO]: Loading adaptation datasets...")
            adaptation_data = self._load_adaptation_datasets()

            print("\n[INFO]: Starting adaptation (SFT and DPO)...")
            self._run_adaptation(adaptation_data)

            print("\n[INFO]: Loading evaluation dataset...")
            evaluation_data = self._load_evaluation_datasets()

            print("\n[INFO]: Starting evaluation...")
            self._run_evaluation(evaluation_data)

            print("\n[INFO]: Experiment completed successfully.")

        except Exception as e:
            print(f"[ERROR]: Failed to complete experiment. Error: {e}")

    def _load_configuration(self) -> None:
        """
        Load and parse the configuration settings.
        """
        config_loader = ConfigLoader(self.config_path)
        self.config = config_loader.load_config()

    def _set_random_seed(self) -> None:
        """
        Set the random seed for reproducibility across the workflow.
        """
        seed = self.config.get("global_seed", 42)
        set_random_seed(seed)

    def _load_pretraining_dataset(self) -> torch.utils.data.Dataset:
        """
        Load and preprocess the pretraining dataset.
        :return: Pretraining dataset object ready for training.
        """
        self.dataset_loader = DatasetLoader(self.config)
        return self.dataset_loader.load_pretraining_data()

    def _initialize_model(self) -> None:
        """
        Initialize the Mixture-of-Experts model architecture with the configuration settings.
        """
        model_architecture = self.config["model"]["architecture"]
        model_params = self.config["model"]
        self.model = Model(model_architecture, model_params)

    def _run_pretraining(self, pretraining_data: torch.utils.data.Dataset) -> None:
        """
        Run the pretraining workflow, including checkpointing and metrics logging.
        :param pretraining_data: Preprocessed pretraining dataset.
        """
        self.trainer = Trainer(self.model, pretraining_data, self.config)
        self.trainer.train_pretraining()

    def _load_adaptation_datasets(self) -> Dict[str, torch.utils.data.Dataset]:
        """
        Load and preprocess adaptation datasets for supervised fine-tuning and preference alignment.
        :return: Dictionary containing SFT and DPO datasets.
        """
        return self.dataset_loader.load_adaptation_data()

    def _run_adaptation(self, adaptation_data: Dict[str, torch.utils.data.Dataset]) -> None:
        """
        Run the adaptation workflow for both SFT and DPO.
        :param adaptation_data: Dictionary containing SFT and DPO datasets.
        """
        # Fine-tune the model with Supervised Fine-Tuning (SFT)
        print("[INFO]: Performing instruction tuning (SFT)...")
        self.trainer.train_data = adaptation_data["sft"]
        self.trainer.train_adaptation(adaptation_type="sft")

        # Perform preference alignment with Direct Preference Optimization (DPO)
        print("[INFO]: Performing preference tuning (DPO)...")
        self.trainer.train_data = adaptation_data["dpo"]
        self.trainer.train_adaptation(adaptation_type="dpo")

    def _load_evaluation_datasets(self) -> Dict[str, torch.utils.data.Dataset]:
        """
        Load and preprocess the evaluation dataset for testing benchmarks.
        :return: Evaluation dataset object.
        """
        return self.dataset_loader.load_adaptation_data()

    def _run_evaluation(self, evaluation_data: Dict[str, torch.utils.data.Dataset]) -> None:
        """
        Evaluate the pretrained and adapted model on various benchmarks.
        :param evaluation_data: Dataset object for evaluation.
        """
        self.evaluator = Evaluation(self.model, evaluation_data, self.config)
        results = self.evaluator.evaluate()

        print("[INFO]: Evaluation results:")
        for benchmark, metrics in results.items():
            print(f"Benchmark: {benchmark}, Metrics: {metrics}")

        # Log final evaluation metrics
        log_metrics(results, step=0, log_to_wandb=True)


if __name__ == "__main__":
    # Entry point for running the OLMoE experiment pipeline
    main_obj = Main(config_path="config.yaml")
    main_obj.run_experiment()
