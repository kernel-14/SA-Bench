# visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional

class Visualization:
    """
    Visualization class to generate qualitative and quantitative analyses for the experiments.
    Includes plotting metrics, rendering internal plans, visualizing agent trajectories,
    and representing intervention effects.
    """

    def __init__(self):
        """
        Initializes the Visualization class for rendering and saving experimental visualizations.
        """
        sns.set_theme(style="whitegrid")

    def render_plans(self, concept_data: Dict[Tuple[int, int], Union[str, Tuple[float, float]]], save_path: str) -> None:
        """
        Renders the agent's internal plans decoded from probes, such as C_A and C_B.

        Args:
            concept_data (Dict[Tuple[int, int], Union[str, Tuple[float, float]]]):
                A dictionary with grid square coordinates as keys, and values as either:
                - "NEVER" for squares the agent does not plan to step onto/push off of.
                - A directional vector (dx, dy) for grid arrows.
            save_path (str): File path to save the visualization.
        """
        grid_size = 8
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Render the grid
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="--", linewidth=0.5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Mark squares
        for (x, y), value in concept_data.items():
            if value == "NEVER":
                ax.add_patch(plt.Rectangle((y - 0.35, x - 0.35), 0.7, 0.7, color="lightgray", alpha=0.5))
            elif isinstance(value, tuple):  # Direction vector
                dx, dy = value
                plt.arrow(
                    y, x, dx * 0.3, -dy * 0.3,
                    head_width=0.2, head_length=0.2, fc="blue", ec="blue"
                )

        plt.title("Internal Plan Visualization")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def plot_metrics(self, metrics_data: Dict[str, Dict[str, List[float]]], save_path: str) -> None:
        """
        Plots quantitative metrics, such as Macro-F1 scores or success rates.

        Args:
            metrics_data (Dict[str, Dict[str, List[float]]]):
                A dictionary where:
                - Each key is a metric's name (e.g., "Macro-F1").
                - Values are another dictionary mapping layers or epochs to values.
            save_path (str): Path to save the resulting plot.
        """
        plt.figure(figsize=(10, 6))

        for metric_name, metric_values in metrics_data.items():
            for label, values in metric_values.items():
                plt.plot(values, label=f"{metric_name} - {label}")

        plt.xlabel("Epochs / Layers")
        plt.ylabel("Metric Values")
        plt.title("Metric Visualization")
        plt.legend()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def visualize_trajectory(self, trajectory_data: Dict[str, List], save_path: str) -> None:
        """
        Visualizes agent trajectories overlayed with Sokoban states.

        Args:
            trajectory_data (Dict[str, List]): Contains:
                - "states": List of 8x8 Sokoban boards (current states).
                - "predictions": List of dictionaries containing C_A & C_B predictions.
                - "actions": List of agent's chosen actions per step.
                - "rewards": List of rewards received at each step.
            save_path (str): Path to save the trajectory visualization.
        """
        states = trajectory_data["states"]
        predictions = trajectory_data["predictions"]
        actions = trajectory_data["actions"]
        rewards = trajectory_data["rewards"]
        
        total_steps = len(states)
        for step in range(total_steps):
            board = states[step]
            prediction = predictions[step]

            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(
                board[:, :, 0], cbar=False, linewidths=0.1, linecolor="gray",
                annot=False, cmap="YlGnBu", square=True, ax=ax
            )

            # Overlay predictions
            for (x, y), value in prediction.items():
                if value == "NEVER":
                    ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color="gray", alpha=0.6))
                elif isinstance(value, tuple):
                    dx, dy = value
                    plt.arrow(
                        y, x, dx * 0.2, -dy * 0.2,
                        head_width=0.2, head_length=0.2, fc="blue", ec="blue"
                    )

            plt.title(f"Sokoban Trajectory - Step {step + 1}\nAction: {actions[step]}, Reward: {rewards[step]}")
            plt.savefig(f"{save_path}_step_{step + 1}.png", bbox_inches="tight")
            plt.close(fig)

    def generate_intervention_effects(self, success_rates: Dict[str, List[float]], layer_info: List[int], save_path: str) -> None:
        """
        Visualizes intervention success rates across layers for trained vs. random probes.

        Args:
            success_rates (Dict[str, List[float]]): Dictionary with keys like "Trained" and "Random"
                where values are success rates across layers.
            layer_info (List[int]): The list of ConvLSTM layers' indices.
            save_path (str): Path to save intervention bar plots.
        """
        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        x_indices = np.arange(len(layer_info))

        # Plot Trained vs Random
        plt.bar(x_indices, success_rates["Trained"], bar_width, label="Trained Probes")
        plt.bar(x_indices + bar_width, success_rates["Random"], bar_width, label="Random Probes")

        plt.xticks(x_indices + bar_width / 2, [f"Layer {i}" for i in layer_info])
        plt.xlabel("Layers")
        plt.ylabel("Success Rates")
        plt.title("Intervention Effectiveness by Layer")
        plt.legend()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def compare_probe_effectiveness(
        self, layer_results: Dict[str, List[float]], save_path: str
    ) -> None:
        """
        Compares probe effectiveness (e.g., 1x1 vs 3x3) across layers.

        Args:
            layer_results (Dict[str, List[float]]): Mapping of probe sizes ("1x1", "3x3") to Macro-F1 scores per layer.
            save_path (str): Path to save the comparison plot.
        """
        plt.figure(figsize=(10, 6))

        for probe_size, results in layer_results.items():
            plt.plot(results, label=f"{probe_size} Probes")

        plt.xlabel("Layer")
        plt.ylabel("Macro-F1 Score")
        plt.title("Probe Effectiveness by Layer")
        plt.legend()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
