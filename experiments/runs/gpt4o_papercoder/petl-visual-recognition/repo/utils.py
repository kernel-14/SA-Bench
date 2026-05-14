# utils.py
import os
import json
import yaml
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict

def load_config(file_path: str) -> dict:
    """Load a YAML configuration file and return it as a dictionary."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing the YAML configuration file: {e}")
    
    # Validate required keys
    required_keys = ["training", "datasets", "backbone", "peft_methods", "evaluation"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key '{key}' in configuration file.")
    
    return config

def set_random_seed(seed: int) -> None:
    """Set the random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For all available GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_results(results: Dict[str, Any], save_path: str) -> None:
    """Save results in a structured JSON file."""
    if not isinstance(results, dict):
        raise ValueError("Results must be a dictionary.")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save result as JSON
    try:
        with open(save_path, 'w') as file:
            json.dump(results, file, indent=4)
    except IOError as e:
        raise IOError(f"Error writing to file {save_path}: {e}")

def plot_metrics(metrics: Dict[str, Any], save_path: str) -> None:
    """Generate and save visualizations based on metrics."""
    if not metrics:
        raise ValueError("Metrics dictionary is empty.")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Basic line plot for accuracy and/or loss trends
    if "Accuracy" in metrics and "Loss" in metrics:
        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(metrics["Accuracy"]) + 1))
        plt.plot(epochs, metrics["Accuracy"], label="Accuracy", marker='o', linestyle="--")
        plt.plot(epochs, metrics["Loss"], label="Loss", marker='x', linestyle=":")
        plt.title("Model Performance Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    
    # If metrics dict contains prediction similarity matrix, generate heatmap
    if "Prediction Similarity" in metrics:
        similarity_matrix = metrics["Prediction Similarity"]
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Prediction Similarity Matrix")
        plt.xlabel("PEFT Method")
        plt.ylabel("PEFT Method")
        heatmap_save_path = save_path.replace(".png", "_similarity_matrix.png")
        plt.savefig(heatmap_save_path)
        plt.close()
    
    # If ensemble performance is provided, generate bar plot
    if "Ensemble Performance" in metrics:
        ensemble_data = metrics["Ensemble Performance"]
        methods = list(ensemble_data.keys())
        scores = list(ensemble_data.values())
        plt.figure(figsize=(10, 6))
        plt.bar(methods, scores, color='skyblue')
        plt.title("Ensemble Performance Across PEFT Methods")
        plt.xlabel("Method")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        bar_plot_save_path = save_path.replace(".png", "_ensemble_performance.png")
        plt.savefig(bar_plot_save_path)
        plt.close()
