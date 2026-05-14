"""
evaluation.py
Provides the Evaluator class to calculate performance metrics and generate visualizations.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
import os
from absl import logging

# Local application imports
from config import Config
from utils import compute_rmse, compute_nll, compute_chi_squared


class Evaluator:
    """
    Evaluator class to calculate performance metrics (RMSE, NLL, Chi-squared)
    and generate visualizations from model predictions.
    """

    def __init__(self, test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], config: Config):
        """
        Initializes the Evaluator.

        Args:
            test_data: A tuple (test_inputs, test_targets_full_trajectories, test_initial_conditions).
                       - test_inputs: (num_test_traj, initial_time_steps, *spatial_dims, total_input_channels)
                       - test_targets_full_trajectories: (num_test_traj, full_temporal_res, *spatial_dims, output_channels)
                       - test_initial_conditions: (num_test_traj, *spatial_dims, output_channels) - first frame, usually.
            config: The global Config object with experimental settings.
        """
        self.config = config

        # Store full ground truth trajectories for potential autoregressive evaluation
        # Shape: (num_test_traj, full_temporal_res, spatial_x, spatial_y, output_channels)
        _, self.y_true_full_trajectories, _ = test_data
        
        self.num_test_traj: int = self.y_true_full_trajectories.shape[0]
        self.full_temporal_res: int = self.y_true_full_trajectories.shape[1]
        
        # Determine spatial dimensions. For 1D, spatial_res_y will be 1 from config.
        # test_targets_full_trajectories.shape will be (N, T, X, Y, C)
        self.spatial_res_x: int = self.y_true_full_trajectories.shape[2]
        self.spatial_res_y: int = self.y_true_full_trajectories.shape[3]
        self.output_channels: int = self.y_true_full_trajectories.shape[4]
        self.initial_time_steps: int = config.initial_time_steps

        # Flag for 1D vs 2D plotting logic
        self.is_1d_pde: bool = config.dimensions == "1D"

        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Evaluator initialized. Results will be saved to {self.results_dir}")

    def calculate_metrics(
        self,
        predictions_mean: jnp.ndarray, # (num_test_traj, spatial_x, spatial_y, output_channels)
        predictions_std: jnp.ndarray,  # (num_test_traj, spatial_x, spatial_y, output_channels)
        eval_time_step_idx: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Calculates RMSE, NLL, and Chi-squared metrics.

        Args:
            predictions_mean: Predicted mean values.
                              Shape: (num_test_traj, spatial_x, spatial_y, output_channels).
            predictions_std: Predicted standard deviation values. Same shape as predictions_mean.
            eval_time_step_idx: Optional. The index of the time step in the full ground truth
                                trajectory that these predictions correspond to.
                                If None, defaults to `config.initial_time_steps` (the first predicted step).

        Returns:
            A dictionary containing the computed 'rmse', 'nll', and 'chi_squared' mean values.
        """
        if eval_time_step_idx is None:
            # For next-step prediction, target is at initial_time_steps index
            target_idx = self.initial_time_steps
        else:
            target_idx = eval_time_step_idx

        # Retrieve ground truth for the specific time step
        # Shape: (num_test_traj, spatial_x, spatial_y, output_channels)
        y_true_batch = self.y_true_full_trajectories[:, target_idx, ...]

        # Ensure shapes match before metric calculation
        if y_true_batch.shape != predictions_mean.shape:
            logging.error(f"Shape mismatch: y_true_batch {y_true_batch.shape} "
                          f"vs predictions_mean {predictions_mean.shape}")
            raise ValueError("Ground truth and prediction shapes do not match.")

        # Compute metrics using utility functions
        rmse_val = compute_rmse(y_true_batch, predictions_mean)
        nll_val = compute_nll(y_true_batch, predictions_mean, predictions_std)
        chi_squared_val = compute_chi_squared(y_true_batch, predictions_mean, predictions_std)

        metrics = {
            "rmse": float(rmse_val),
            "nll": float(nll_val),
            "chi_squared": float(chi_squared_val),
        }
        return metrics

    def plot_predictions(
        self,
        method_name: str,
        plot_identifier: str,
        predictions_mean_sample: jnp.ndarray, # (spatial_x, spatial_y, output_channels)
        predictions_std_sample: jnp.ndarray,  # (spatial_x, spatial_y, output_channels)
        ground_truth_sample: jnp.ndarray,     # (spatial_x, spatial_y, output_channels)
        samples: Optional[jnp.ndarray] = None, # (num_ensemble_samples, spatial_x, spatial_y, output_channels)
        time_step_to_plot: Optional[int] = None, # For AR rollouts, what time step this prediction represents
    ) -> None:
        """
        Generates and saves a plot comparing ground truth, predictions, and uncertainty.

        Args:
            method_name: Name of the UQ method (e.g., "LUNO-LA").
            plot_identifier: A descriptive string for the plot (e.g., "single_step_example_0", "rollout_t_15").
            predictions_mean_sample: A single mean prediction (unbatched).
                                     Shape: (*spatial_dims, output_channels).
            predictions_std_sample: A single standard deviation prediction (unbatched).
                                    Shape: (*spatial_dims, output_channels).
            ground_truth_sample: A single corresponding ground truth (unbatched).
                                 Shape: (*spatial_dims, output_channels).
            samples: Optional. JAX array of samples from the predictive distribution for this single input.
                     Shape: (num_ensemble_samples, *spatial_dims, output_channels).
            time_step_to_plot: Optional. The actual time step index this plot corresponds to (for title).
        """
        # Squeeze out singleton spatial_y dimension for 1D plotting
        if self.is_1d_pde and self.spatial_res_y == 1:
            ground_truth_sample = ground_truth_sample.squeeze(axis=1) # (X, C)
            predictions_mean_sample = predictions_mean_sample.squeeze(axis=1) # (X, C)
            predictions_std_sample = predictions_std_sample.squeeze(axis=1) # (X, C)
            if samples is not None:
                samples = samples.squeeze(axis=2) # (num_samples, X, C)

        # Plot only the first output channel for simplicity if multiple exist
        gt_plot = ground_truth_sample[..., 0]
        mean_plot = predictions_mean_sample[..., 0]
        std_plot = predictions_std_sample[..., 0]
        if samples is not None:
            samples_plot = samples[..., 0] # (num_samples, spatial_x or spatial_x, spatial_y)

        # Define the plot filename
        filename = self.results_dir / f"{method_name}_{plot_identifier}_prediction.png"
        
        # Decide on number of samples to plot from 'samples' if available
        num_plot_samples = 4 # As per Figure 2 description
        if samples is not None:
            # Take a few samples (max 4, or fewer if less available)
            samples_to_plot = samples_plot[:min(num_plot_samples, samples_plot.shape[0])]

        if self.is_1d_pde:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            x_coords = np.arange(self.spatial_res_x)

            # Top row: target, mean, 1.96 std, samples
            axes[0].plot(x_coords, gt_plot, label="Target (Ground Truth)", color='black', linestyle='-')
            axes[0].plot(x_coords, mean_plot, label="Mean Prediction", color='red', linestyle='--')
            
            # 1.96 standard deviations
            upper_bound = mean_plot + 1.96 * std_plot
            lower_bound = mean_plot - 1.96 * std_plot
            axes[0].fill_between(x_coords, lower_bound, upper_bound, color='red', alpha=0.2, label="1.96 Std Dev")

            if samples is not None:
                for i, s_val in enumerate(samples_to_plot):
                    axes[0].plot(x_coords, s_val, color='blue', alpha=0.5, linewidth=0.8, label=f"Sample {i+1}" if i == 0 else "")
            
            axes[0].set_title(f"Prediction for {method_name} at t={time_step_to_plot if time_step_to_plot is not None else self.initial_time_steps}")
            axes[0].set_ylabel("Value")
            axes[0].legend()
            axes[0].grid(True)

            # Bottom row: spread/std of the predictive distribution
            axes[1].plot(x_coords, std_plot, label="Predictive Standard Deviation", color='green')
            axes[1].set_xlabel("Spatial Coordinate (x)")
            axes[1].set_ylabel("Std Dev")
            axes[1].legend()
            axes[1].grid(True)

        else: # 2D PDE plotting (heatmaps)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12)) # Two rows, three columns

            # Vmin/Vmax for consistent color scaling
            vmin_data = min(gt_plot.min(), mean_plot.min())
            vmax_data = max(gt_plot.max(), mean_plot.max())

            # Row 1: Target, Mean, Residual
            im00 = axes[0, 0].imshow(gt_plot, cmap='viridis', origin='lower', vmin=vmin_data, vmax=vmax_data)
            axes[0, 0].set_title("Target")
            fig.colorbar(im00, ax=axes[0, 0])

            im01 = axes[0, 1].imshow(mean_plot, cmap='viridis', origin='lower', vmin=vmin_data, vmax=vmax_data)
            axes[0, 1].set_title("Mean Prediction")
            fig.colorbar(im01, ax=axes[0, 1])

            residual = jnp.abs(gt_plot - mean_plot)
            im02 = axes[0, 2].imshow(residual, cmap='hot', origin='lower')
            axes[0, 2].set_title("Absolute Residual")
            fig.colorbar(im02, ax=axes[0, 2])
            
            # Row 2: Predictive Std Dev, One Sample, Normed Residual (Q values per point)
            im10 = axes[1, 0].imshow(std_plot, cmap='magma', origin='lower')
            axes[1, 0].set_title("Predictive Std Dev")
            fig.colorbar(im10, ax=axes[1, 0])

            if samples is not None and samples_to_plot.shape[0] > 0:
                im11 = axes[1, 1].imshow(samples_to_plot[0], cmap='viridis', origin='lower', vmin=vmin_data, vmax=vmax_data)
                axes[1, 1].set_title("One Sample")
                fig.colorbar(im11, ax=axes[1, 1])
            else:
                axes[1, 1].set_title("No Samples Available")
                axes[1, 1].axis('off') # Turn off axis if no data

            # Normalized residual: (y - mean)^2 / std^2
            normed_residual = jnp.square(gt_plot - mean_plot) / (jnp.square(std_plot) + 1e-6)
            im12 = axes[1, 2].imshow(normed_residual, cmap='coolwarm', origin='lower', vmin=0, vmax=5) # Clamp for better visualization
            axes[1, 2].set_title("Normalized Residual ((y-mean)^2/std^2)")
            fig.colorbar(im12, ax=axes[1, 2])

            fig.suptitle(f"2D Prediction for {method_name} at t={time_step_to_plot if time_step_to_plot is not None else self.initial_time_steps}", fontsize=16)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        logging.info(f"Plot saved to {filename}")

