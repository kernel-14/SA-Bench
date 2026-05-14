## main.py

import os
import yaml
from typing import Dict, Any
import numpy as np
from dataset_loader import DatasetLoader
from model import FourierNeuralOperator
from trainer import Trainer
from linearization import Linearization
from evaluation import Evaluation

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def run_low_data_experiment(config: Dict[str, Any]) -> None:
    """
    Execute the low-data regime experiment pipeline.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    # Step 1: Dataset Preparation
    print("Loading low-data regime dataset...")
    data_loader = DatasetLoader(config)
    low_data = data_loader.load_low_data()
    train_data = low_data["train"]
    test_data = low_data["test"]

    # Step 2: Model Initialization
    print("Initializing Fourier Neural Operator for low-data regime...")
    model = FourierNeuralOperator(
        modes=config["model"]["fno"]["modes"],
        hidden_dimensions=config["model"]["fno"]["hidden_dimensions"],
        num_blocks=config["model"]["fno"]["num_blocks"]
    )

    # Step 3: Training the Model
    print(f"Training model on {len(train_data['inputs'])} trajectories...")
    trainer = Trainer(model=model, optimizer=None, config=config)
    training_summary = trainer.train(train_data, epochs=config["training"]["epochs_low_data"])
    print(f"Training completed in {training_summary['runtime']:.2f}s.")

    # Step 4: Linearization and Uncertainty Quantification
    print("Applying linearization for uncertainty quantification...")
    linearization = Linearization(model=model)
    jacobian = linearization.compute_jacobian(inputs=test_data["inputs"])
    mean = test_data["inputs"]  # Example: Use inputs as mean for simplistic demonstration
    covariance = np.eye(jacobian.shape[-1]) * 1e-2  # Example covariance (must be replaced by learned Σ)
    gp = linearization.construct_gp(mean=mean, covariance=covariance)

    # Step 5: Evaluation
    print("Evaluating test data...")
    evaluation = Evaluation(model=model, metrics=config["evaluation"]["metrics"])
    metrics = evaluation.evaluate({
        "inputs": test_data["inputs"],
        "truth": test_data["outputs"],
        "predictions": {
            "mean": gp["mean"],
            "std": np.sqrt(gp["variance"])  # Convert variance to std
        }
    })
    evaluation.visualize_uncertainty(predictions=gp, truth=test_data["outputs"])

def run_ood_experiment(config: Dict[str, Any]) -> None:
    """
    Execute the out-of-distribution (OOD) experiment pipeline.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    # Step 1: Dataset Preparation
    print("Loading OOD regime dataset...")
    data_loader = DatasetLoader(config)
    ood_data = data_loader.load_ood_data()
    train_data = ood_data["Base"]
    ood_datasets = {name: data for name, data in ood_data.items() if name != "Base"}

    # Step 2: Model Initialization
    print("Initializing Fourier Neural Operator for OOD regime...")
    model = FourierNeuralOperator(
        modes=config["model"]["fno"]["modes"],
        hidden_dimensions=config["model"]["fno"]["hidden_dimensions"],
        num_blocks=config["model"]["fno"]["num_blocks"]
    )

    # Step 3: Training the Model
    print(f"Training model on {len(train_data['inputs'])} Base trajectories...")
    trainer = Trainer(model=model, optimizer=None, config=config)
    training_summary = trainer.train(train_data, epochs=config["training"]["epochs_ood"])
    print(f"Training completed in {training_summary['runtime']:.2f}s.")

    # Step 4: Evaluation Across OOD Scenarios
    evaluation = Evaluation(model=model, metrics=config["evaluation"]["metrics"])
    for name, data in ood_datasets.items():
        print(f"Evaluating on OOD dataset '{name}'...")
        predictions = model.apply({"params": training_summary["weights"]}, data["inputs"])
        metrics = evaluation.evaluate({
            "inputs": data["inputs"],
            "truth": data["outputs"],
            "predictions": {
                "mean": predictions,
                "std": np.zeros_like(predictions)  # Assume no uncertainty for demo
            }
        })
        evaluation.visualize_uncertainty(predictions=metrics, truth=data["outputs"], save_filename=f"{name}_uncertainty.png")

def main() -> None:
    """
    Main entry point to orchestrate the LUNO pipeline.
    """
    # Load configuration
    config = load_config()

    # Run Low-Data Experiment
    print("Starting Low-Data Experiment...")
    run_low_data_experiment(config)

    # Run OOD Experiment
    print("\nStarting Out-of-Distribution (OOD) Experiment...")
    run_ood_experiment(config)

if __name__ == "__main__":
    main()
