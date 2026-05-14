## utils.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List
from model import DiffusionModel


class Utils:
    """
    Utilities for checkpoint handling and visualization. Provides reusable methods for saving/loading model states
    and generating plots for experimental results.
    """

    def __init__(self):
        """Initialize the Utils class for modular reusable functions."""
        pass

    def save_checkpoint(self, model: DiffusionModel, epoch: int, output_directory: str = "results/") -> None:
        """
        Save the current checkpoint of the model along with metadata.

        Args:
            model (DiffusionModel): The trained PyTorch model instance to be saved.
            epoch (int): The current epoch number for recordkeeping.
            output_directory (str): Directory path where the checkpoint will be saved.
        """
        os.makedirs(output_directory, exist_ok=True)
        checkpoint_path = os.path.join(output_directory, f"checkpoint_epoch_{epoch}.pth")

        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "config": model.params  # Save configuration for reproducibility
            }, checkpoint_path)
            print(f"[INFO] Checkpoint successfully saved at: {checkpoint_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> DiffusionModel:
        """
        Load model state from checkpoint and return the model and epoch metadata.

        Args:
            checkpoint_path (str): Path to the checkpoint file (.pth).

        Returns:
            DiffusionModel: Restored model instance with loaded weights.
            int: Restored epoch number.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path)
            model = DiffusionModel(params=checkpoint.get("config"))
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint["epoch"]
            print(f"[INFO] Checkpoint successfully loaded from: {checkpoint_path}")
            return model, epoch
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to load checkpoint: {e}")

    def plot_metrics(self, evaluation_results: Dict[str, List[float]], output_directory: str = "results/") -> None:
        """
        Plot evaluation metrics such as TV distance and KL divergence.

        Args:
            evaluation_results (Dict[str, List[float]]): Dictionary containing metrics to visualize.
                Example: {"total_variation": [0.12, 0.11, 0.08], "kl_divergence": [0.5, 0.3, 0.15]}.
            output_directory (str): Directory to save plot images.
        """
        os.makedirs(output_directory, exist_ok=True)

        for metric_name, metric_values in evaluation_results.items():
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(metric_values)), metric_values, marker="o", label=metric_name)
            plt.xlabel("Iterations")
            plt.ylabel(f"{metric_name.capitalize()} Value")
            plt.title(f"{metric_name.capitalize()} Convergence Over Iterations")
            plt.legend()
            plt.grid()

            plot_path = os.path.join(output_directory, f"{metric_name}_convergence.png")
            try:
                plt.savefig(plot_path)
                print(f"[INFO] {metric_name.capitalize()} plot saved at: {plot_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save plot for {metric_name}: {e}")
            finally:
                plt.close()  # Prevent overlapping plots

    def plot_sampling_trajectory(self, y_true: torch.Tensor, y_sampled: torch.Tensor, output_path: str) -> None:
        """
        Plot sampling trajectory, comparing sampled data and ground truth.

        Args:
            y_true (torch.Tensor): Ground truth data samples from the target distribution.
            y_sampled (torch.Tensor): Sampled data from the reverse process.
            output_path (str): Path to save the scatter plot.
        """
        try:
            plt.figure(figsize=(8, 8))
            y_true_np = y_true.cpu().numpy()
            y_sampled_np = y_sampled.cpu().numpy()

            if y_true_np.shape[1] == 1:  # 1D case
                bins = 100
                plt.hist(y_true_np.flatten(), bins=bins, alpha=0.5, label="Ground truth", density=True)
                plt.hist(y_sampled_np.flatten(), bins=bins, alpha=0.5, label="Sampled", density=True)
            elif y_true_np.shape[1] == 2:  # 2D case
                plt.scatter(y_true_np[:, 0], y_true_np[:, 1], alpha=0.5, label="Ground truth", color="blue")
                plt.scatter(y_sampled_np[:, 0], y_sampled_np[:, 1], alpha=0.5, label="Sampled", color="red")
            else:
                raise ValueError("[ERROR] Cannot visualize data beyond 2D!")

            plt.title("Sampling Trajectory")
            plt.legend()
            plt.grid(True)
            plt.savefig(output_path)
            print(f"[INFO] Sampling trajectory plot saved at: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to plot sampling trajectory: {e}")
        finally:
            plt.close()
