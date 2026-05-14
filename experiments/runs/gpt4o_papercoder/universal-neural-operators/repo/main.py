## main.py

"""
Entry point for orchestrating the pretraining, fine-tuning, and evaluation workflows
for neural operator models as part of the reproducibility effort for the paper.

This script integrates configuration management, dataset handling, model initialization,
training processes, and performance evaluation into a single cohesive pipeline.
"""

# Required Imports
import os
from config import Config
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation
from utils import set_seed

def run_experiment(config_path: str = "config.yaml") -> None:
    """
    Main function to execute the pretraining, fine-tuning, and evaluation pipeline.

    Args:
        config_path (str): Path to the configuration file (default: "config.yaml").

    Returns:
        None
    """
    # Load configuration
    print("Loading configuration...")
    config = Config(file_path=config_path).config

    # Set random seed for reproducibility
    print("Setting random seed...")
    set_seed(config["random_seed"])

    # Initialize the DatasetLoader
    print("Initializing datasets...")
    dataset_loader = DatasetLoader(config=config)
    train_data, val_data, test_data = dataset_loader.load_data()

    # Initialize the Model
    print(f"Initializing model: {config['model']['type']}...")
    model_params = {
        "in_channels": config["model"].get("in_channels", 2),  # Example: 2D PDE input
        "out_channels": config["model"].get("out_channels", 1),  # Example: PDE solution
        "modes": config["model"].get("hidden_modes", 16),
        "layers": config["model"].get("layers", 4),
        "kernel_size": config["model"].get("kernel_size", 5),  # Optional for Mamba-SSM
        "latent_size": config["model"].get("latent_size", 128),  # Optional for Perceiver IO
        "attention_layers": config["model"].get("attention_layers", 2)  # Optional for transformers
    }
    model_type = config["model"]["type"]
    model = Model(model_type=model_type, params=model_params)

    # Pretraining phase
    print("Starting pretraining...")
    trainer = Trainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=config
    )
    trainer.pretrain()

    # Save pretraining checkpoint
    print("Saving pretraining checkpoint...")
    pretrain_checkpoint_path = os.path.join(config["logging"]["checkpoint_dir"], "pretraining_checkpoint.pth")
    trainer.save_checkpoint(pretrain_checkpoint_path)

    # Fine-tuning phase
    print("Freezing core layers for fine-tuning...")
    model.freeze_layers(except_layers=["lifting", "projection"])

    print("Starting fine-tuning...")
    trainer = Trainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=config
    )
    trainer.finetune()

    # Save fine-tuning checkpoint
    print("Saving fine-tuning checkpoint...")
    finetune_checkpoint_path = os.path.join(config["logging"]["checkpoint_dir"], "fine_tuning_checkpoint.pth")
    trainer.save_checkpoint(finetune_checkpoint_path)

    # Evaluation phase
    print("Starting evaluation...")
    evaluation = Evaluation(
        model=model,
        test_data=test_data,
        config=config
    )
    metrics = evaluation.evaluate()

    # Log evaluation results
    print("Logging metrics and visualizations...")
    evaluation.visualize_results(model_predictions=None, ground_truth=None)
    evaluation.log_metrics(metrics)

    print("Experiment completed successfully!")


if __name__ == "__main__":
    run_experiment()
