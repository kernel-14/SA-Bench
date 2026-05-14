```python
# main.py
import os
from utils import load_config, set_random_seed, save_results, plot_metrics
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation

def run_experiment():
    """
    Main function to orchestrate the entire experimental workflow:
    - Load configurations
    - Prepare datasets
    - Initialize models with PEFT modules
    - Train models
    - Evaluate and analyze results
    """
    # Step 1: Load configuration and set random seeds for reproducibility
    config_path = "config.yaml"
    config = load_config(config_path)
    set_random_seed(config.get("training", {}).get("seed", 42))
    print("Configuration loaded successfully.")

    # Step 2: Prepare datasets
    loader = DatasetLoader(config)
    print("Loading datasets...")

    # Load datasets for experiments
    datasets = {
        "low_shot": lambda task_name: loader.load_vtab_dataset(task_name),
        "many_shot": lambda dataset_name: loader.load_many_shot_dataset(dataset_name),
        "distribution_shift": loader.load_imagenet_distribution_shifts
    }

    # Experiment results placeholder
    results = {}

    # Step 3: Iterate over tasks and methods for experiments
    for experiment_type, dataset_loader_fn in datasets.items():
        print(f"Starting experiments for {experiment_type} tasks...")

        dataset_names = []
        if experiment_type == "low_shot":
            dataset_names = config["datasets"]["low_shot"]["source"].split(",")
        elif experiment_type == "many_shot":
 datasets defined explicitly.
# Required APIs