"""
evaluation.py

Provides the Evaluation class for the RWM paper reproduction.

It implements:
- Autoregressive prediction accuracy scoring (with optional noise injection),
  returning the normalised mean squared error over a forecasting horizon.
- Policy evaluation in the original simulator (deterministic rollouts),
  returning mean and standard deviation of cumulative rewards.
- Plotting utilities to visualise prediction trajectories, error vs forecast step,
  and model comparison bar charts.

All configuration is read from the `config` dictionary provided at initialisation.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Normal

from env_utils import IsaacEnvWrapper, ROBOT_CONFIGS, compute_reward
from dataset import TrajectoryBuffer
from world_model import WorldModel, RWM
from ppo_agent import PPOAgent


# ------------------------------------------------------------------------------
# Helper: sliding window extraction from a TrajectoryBuffer
# ------------------------------------------------------------------------------
def extract_test_windows(
    buffer: TrajectoryBuffer,
    history_len: int,
    num_steps: int,
    max_windows: int = 100,
    shuffle: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    """
    Extract contiguous windows of length `history_len + num_steps` from the buffer.

    Args:
        buffer: TrajectoryBuffer containing full episodes.
        history_len: M, the context horizon.
        num_steps: Number of future steps to predict (rollout length).
        max_windows: Maximum number of windows to return (for efficiency).
        shuffle: Whether to shuffle the windows.

    Returns:
        A list of dictionaries, each containing:
            'obs_seq' : (L, obs_dim)
            'act_seq' : (L, act_dim)
            'priv_seq': (L, priv_dim)   # may be zero-length if priv_dim=0
        where L = history_len + num_steps.
    """
    windows = []
    L = history_len + num_steps

    for ep in buffer.episodes:
        T = ep["obs"].shape[0]
        if T < L:
            continue
        # Slide over the episode with stride 1 (or bigger if too many)
        stride = max(1, (T - L) // min(max_windows, T - L + 1) if T > L else 1)
        for start in range(0, T - L + 1, stride):
            sl = slice(start, start + L)
            windows.append({
                "obs_seq": ep["obs"][sl],
                "act_seq": ep["act"][sl],
                "priv_seq": ep["priv"][sl] if "priv" in ep else torch.zeros(L, 0),
            })
            if len(windows) >= max_windows:
                break
        if len(windows) >= max_windows:
            break

    if shuffle:
        np.random.shuffle(windows)

    # Trim to exactly max_windows if needed
    return windows[:max_windows]


# ------------------------------------------------------------------------------
# Evaluation class
# ------------------------------------------------------------------------------
class Evaluation:
    """
    Encapsulates evaluation routines for world models and policies.

    Args:
        env: IsaacEnvWrapper instance for policy evaluation (ground truth).
        world_model: A trained WorldModel (RWM or baseline) instance.
        ppo_agent: A trained PPOAgent instance.
        config: Configuration dictionary (from config.yaml).
    """

    def __init__(
        self,
        env: IsaacEnvWrapper,
        world_model: WorldModel,
        ppo_agent: PPOAgent,
        config: Dict,
    ):
        self.env = env
        self.wm = world_model
        self.ppo = ppo_agent
        self.config = config

        # Extract hyper‑parameters
        self.history_len = config["world_model"]["history_len"]
        self.forecast_len = config["world_model"]["forecast_len"]
        self.obs_dim = self.wm.obs_dim
        self.act_dim = self.wm.act_dim
        self.priv_dim = self.wm.priv_dim
        self.device = next(self.wm.parameters()).device if len(list(self.wm.parameters())) > 0 else torch.device("cpu")

        # Normalisation statistics (must be present in the world model)
        self.obs_mean = self.wm.obs_mean
        self.obs_std = self.wm.obs_std
        self.act_mean = self.wm.act_mean
        self.act_std = self.wm.act_std

        # Ensure statistics are not all zeros (i.e., have been set)
        if torch.allclose(self.obs_std, torch.ones_like(self.obs_std)):
            print("Warning: world model normalisation statistics may not have been set. "
                  "Prediction errors will be unnormalised.")

    # ------------------------------------------------------------------
    # Autoregressive prediction accuracy
    # ------------------------------------------------------------------
    def evaluate_prediction(
        self,
        buffer: TrajectoryBuffer,
        num_steps: int = 100,
        noise_std: float = 0.0,
        max_windows: int = 100,
        batch_size: int = 64,          # process multiple windows in parallel for speed
        per_dimension: bool = False,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Evaluate the world model's autoregressive prediction accuracy.

        Args:
            buffer: TrajectoryBuffer containing test episodes.
            num_steps: Number of future steps to predict (rollout horizon).
            noise_std: Standard deviation of Gaussian noise added to the observations
                       and actions during the autoregressive rollout (0 = none).
            max_windows: Maximum number of independent test windows to evaluate.
            batch_size: Number of windows to process in a single forward pass.
            per_dimension: If True, return per‑dimension error vectors.

        Returns:
            Dictionary with:
                'mean_error': scalar average normalised MSE over all steps and windows.
                'step_errors': np.ndarray of shape (num_steps,) with average error per forecast step.
                'per_dim_error': (optional) np.ndarray of shape (obs_dim,) with average error per dimension.
        """
        windows = extract_test_windows(buffer, self.history_len, num_steps, max_windows)
        if not windows:
            raise ValueError("No test windows of sufficient length found in the buffer.")

        # We'll collect step‑wise errors for all windows
        step_errors = np.zeros(num_steps)
        window_count = 0
        per_dim_error_accum = np.zeros(self.obs_dim) if per_dimension else None

        # Process windows in mini‑batches
        num_windows = len(windows)
        for b_start in range(0, num_windows, batch_size):
            b_end = min(b_start + batch_size, num_windows)
            batch_windows = windows[b_start:b_end]
            B = len(batch_windows)

            # Prepare tensors of shape (B, L, ...)
            obs_seq = torch.stack([w["obs_seq"] for w in batch_windows], dim=0).to(self.device)
            act_seq = torch.stack([w["act_seq"] for w in batch_windows], dim=0).to(self.device)
            # priv is ignored for prediction error

            L = obs_seq.shape[1]  # = history_len + num_steps
            M = self.history_len
            N = num_steps

            # Normalise observations and actions for consistency with world model's training
            # Note: world model's forward normally normalises internally, but we are doing
            # step‑by‑step with GRU, so we need to pre‑normalise.
            obs_norm = self._normalize(obs_seq, self.obs_mean, self.obs_std)
            act_norm = self._normalize(act_seq, self.act_mean, self.act_std)

            # Initialise GRU hidden state
            h = torch.zeros(
                self.wm.gru.num_layers, B, self.wm.gru.hidden_size,
                device=self.device
            )

            # Process history (M steps) with ground‑truth
            for t in range(M):
                inp = torch.cat([obs_norm[:, t:t+1, :], act_norm[:, t:t+1, :]], dim=-1)
                _, h = self.wm.gru(inp, h)

            # Autoregressive forecast
            # We'll keep track of the last predicted observation, which will be normalised
            # because it's a sample from the Gaussian head.
            cur_obs_norm = obs_norm[:, M-1]   # (B, obs_dim)  last history obs
            for k in range(N):
                # The "current" hidden state h corresponds to after having seen obs_seq[:, M-1+k]? No,
                # we processed history up to M-1. For k=0, we need to predict step M.
                # At state h after seeing last history input (obs_norm[M-1], act_norm[M-1]),
                # we are ready to predict next observation for step M.
                # So we directly apply the head.
                last_hidden = h[-1]   # (B, hidden)
                # Predict distribution
                obs_params = self.wm.obs_head(last_hidden)   # (B, obs_dim*2)
                mu = obs_params[..., :self.obs_dim]
                log_std = obs_params[..., self.obs_dim:]
                log_std = torch.clamp(log_std, -20.0, 2.0)
                std = torch.exp(log_std) + self.wm.std_min

                # For deterministic evaluation, use mean
                pred_norm = mu

                # Compute normalised error against ground truth (at step M+k)
                true_norm = obs_norm[:, M + k]   # (B, obs_dim)
                # Normalised squared error per dimension using obs_std (which is already applied as scaling? No, our true_norm is already normalised, so the error is already scale‑invariant. But the paper's "relative prediction error" uses original scale and normalises by std? Actually they use relative error. We follow the formula given in the design document: e = (1/K) sum_k (|| pred_obs - true_obs ||^2 / (|| true_obs ||^2 + epsilon) ) but that is for original scale. However, the logic analysis says to use `(pred_obs - true_obs) / std` squared and average. That yields a sort of normalised MSE. Which is correct? The paper's exact metric is not specified, but the relative error e they show in figures is likely normalised by the data variance. Using `(pred - true)/std` squared gives something akin to a z‑score error, which is common. Since we have `obs_std`, we'll use that for normalisation. 
                # In the original implementation, the error might be computed as MSE between denormalised predictions and true observations, divided by the variance of the true observations. But using the world model's std is a reasonable proxy. 
                # We'll stick with the normalised MSE with respect to the standard deviation, which gives a dimensionless error. 
                # We'll compute it as: mean over dims of ((pred_norm - true_norm) ** 2), because obs_norm are already standardised to have unit variance if std was correctly estimated? Actually, after normalisation with mean/std, each dimension should have std ~1. So squared error is already normalised. So we don't need to divide again. So step_error = ((pred_norm - true_norm) ** 2).mean().
                squared_errors = (pred_norm - true_norm) ** 2   # (B, obs_dim)
                step_error = squared_errors.mean(dim=-1)        # (B,)
                step_errors[k] += step_error.sum().item()
                window_count += B
                if per_dimension:
                    per_dim_error_accum += squared_errors.sum(dim=0).cpu().numpy()

                # Prepare next input: pre‑normalised sampled observation + action for step M+k
                # The action is ground truth (with optional noise)
                act_next = act_norm[:, M + k]   # (B, act_dim)
                if noise_std > 0:
                    act_next = act_next + torch.randn_like(act_next) * noise_std

                # For the next observation, we need to feed the world model with a sample.
                # Use a sample from the predicted distribution (reparameterized) if noise_std>0 or even if zero? The paper's autoregressive training uses reparameterization, so for consistency we should sample. For deterministic evaluation we can use mean, but then we might get slightly different behavior because the model was trained with sampling. Usually, using mean is fine for evaluation. However, when noise_std>0 they added noise to the fed-back observation, which is why they added noise. We'll follow the logic analysis: feed pred_obs_noisy = pred_obs + noise, where noise is injected. If noise_std=0, pred_obs_noisy = pred_obs (i.e., mean). So we'll just add external noise.
                if noise_std > 0:
                    # Add noise to the observation before feeding back
                    obs_next_feed = pred_norm + torch.randn_like(pred_norm) * noise_std
                else:
                    obs_next_feed = pred_norm

                # Prepare input for GRU: (B, 1, in_dim)
                next_inp = torch.cat([obs_next_feed.unsqueeze(1), act_next.unsqueeze(1)], dim=-1)
                _, h = self.wm.gru(next_inp, h)

        # After all windows
        step_errors /= window_count   # average per step
        mean_error = step_errors.mean()

        results = {
            "mean_error": mean_error,
            "step_errors": step_errors,
        }
        if per_dimension and per_dim_error_accum is not None:
            per_dim_error = per_dim_error_accum / window_count
            results["per_dim_error"] = per_dim_error

        return results

    # ------------------------------------------------------------------
    # Policy evaluation in the original simulator
    # ------------------------------------------------------------------
    def evaluate_policy(
        self,
        n_episodes: int = 20,
        deterministic: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run the PPO policy in the high‑fidelity simulator and return
        the average undiscounted return.

        Args:
            n_episodes: Number of evaluation episodes.
            deterministic: If True, use the mean action instead of sampling.
            seed: Random seed for the environment (optional).

        Returns:
            Dictionary with 'mean_reward' and 'std_reward'.
        """
        returns = []
        for ep_idx in range(n_episodes):
            # Reset environment (the wrapper returns policy obs directly)
            obs, _ = self.env.reset(seed=seed + ep_idx if seed is not None else None)
            done = False
            ep_return = 0.0
            while not done:
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                if deterministic:
                    # Use the mean of the distribution
                    with torch.no_grad():
                        action_mean = self.ppo.actor(obs_tensor)
                        action = action_mean
                else:
                    action, _ = self.ppo.act(obs_tensor)
                action_np = action.squeeze(0).detach().cpu().numpy()

                # Step environment
                obs, reward, term, trunc, info = self.env.step(action_np)
                done = term or trunc
                ep_return += reward

            returns.append(ep_return)

        mean_reward = np.mean(returns)
        std_reward = np.std(returns)
        return {"mean_reward": mean_reward, "std_reward": std_reward}

    # ------------------------------------------------------------------
    # Plotting utilities
    # ------------------------------------------------------------------
    def plot_trajectory_overlay(
        self,
        buffer: TrajectoryBuffer,
        episode_idx: int = 0,
        num_steps: int = 100,
        var_indices: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualise ground‑truth vs autoregressive prediction for a single episode.
        Produces a plot similar to Fig. 3a.

        Args:
            buffer: TrajectoryBuffer containing the episode.
            episode_idx: Index of the episode to visualise.
            num_steps: Number of forecast steps to show.
            var_indices: List of variable indices to plot (default: first 6).
            save_path: If given, save the figure to this path.
        """
        if episode_idx >= len(buffer.episodes):
            raise IndexError(f"Episode {episode_idx} out of range ({len(buffer.episodes)}).")

        ep = buffer.episodes[episode_idx]
        T = ep["obs"].shape[0]
        if T < self.history_len + num_steps:
            raise ValueError(f"Episode too short (len={T}), need at least {self.history_len + num_steps}.")

        # Extract ground truth sequence
        true_obs_seq = ep["obs"][:self.history_len + num_steps].clone()
        true_act_seq = ep["act"][:self.history_len + num_steps].clone()

        # Normalise inputs (as same as in world model)
        true_obs_norm = self._normalize(true_obs_seq.unsqueeze(0), self.obs_mean, self.obs_std).squeeze(0)  # (L, obs_dim)
        true_act_norm = self._normalize(true_act_seq.unsqueeze(0), self.act_mean, self.act_std).squeeze(0)

        # Run autoregressive prediction
        with torch.no_grad():
            M = self.history_len
            h = torch.zeros(self.wm.gru.num_layers, 1, self.wm.gru.hidden_size, device=self.device)
            # History embedding
            for t in range(M):
                inp = torch.cat([true_obs_norm[t:t+1].unsqueeze(0), true_act_norm[t:t+1].unsqueeze(0)], dim=-1)
                _, h = self.wm.gru(inp, h)

            pred_obs_norm_list = []
            # Use the last history observation as initial input for next hidden? No, we need to feed the predicted observation.
            cur_obs_norm = true_obs_norm[M-1:M]  # (1, obs_dim)
            for k in range(num_steps):
                last_hidden = h[-1]
                obs_params = self.wm.obs_head(last_hidden)
                mu = obs_params[:, :self.obs_dim]
                # Use deterministic mean for plotting
                pred_norm = mu  # (1, obs_dim)
                pred_obs_norm_list.append(pred_norm)

                # Next input: use the predicted observation and ground truth action (no noise)
                act_next = true_act_norm[M + k: M + k + 1]
                next_inp = torch.cat([pred_norm.unsqueeze(1), act_next.unsqueeze(1)], dim=-1)
                _, h = self.wm.gru(next_inp, h)

            # Denormalise predictions
            pred_obs_seq = self._denormalize(torch.cat(pred_obs_norm_list, dim=0), self.obs_mean, self.obs_std)  # (N, obs_dim)

        # Choose variables to display
        if var_indices is None:
            var_indices = list(range(min(6, self.obs_dim)))

        # Create plot
        time_steps = np.arange(num_steps) + M  # start from step M
        fig, axes = plt.subplots(len(var_indices), 1, figsize=(8, 2 * len(var_indices)), sharex=True)
        if len(var_indices) == 1:
            axes = [axes]

        for i, dim in enumerate(var_indices):
            ax = axes[i]
            true_vals = true_obs_seq[M:, dim].cpu().numpy()
            pred_vals = pred_obs_seq[:, dim].cpu().numpy()
            ax.plot(time_steps, true_vals, 'k-', label='Ground Truth')
            ax.plot(time_steps, pred_vals, 'r--', label='RWM Prediction')
            ax.set_ylabel(f'Var {dim}')
            ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel('Step')
        fig.suptitle(f'Autoregressive trajectory overlay (Episode {episode_idx})')
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_error_vs_step(
        self,
        buffer: TrajectoryBuffer,
        noise_levels: List[float],
        num_steps: int = 50,
        max_windows: int = 100,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot prediction error as a function of forecast step for multiple noise levels.
        Reproduces Fig. 3b.

        Args:
            buffer: TrajectoryBuffer with test sequences.
            noise_levels: List of noise standard deviations (e.g., [0.0, 0.05, 0.1]).
            num_steps: Number of forecast steps to evaluate.
            max_windows: Number of test windows to use.
            save_path: Optional file path for saving the figure.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        for sig in noise_levels:
            res = self.evaluate_prediction(
                buffer, num_steps=num_steps, noise_std=sig, max_windows=max_windows
            )
            step_errors = res["step_errors"]   # length num_steps
            ax.plot(range(1, num_steps + 1), step_errors,
                    label=f'$\\sigma$={sig:.3f}' if sig > 0 else 'No noise')

        ax.set_xlabel('Forecast step')
        ax.set_ylabel('Normalised prediction error')
        ax.legend()
        ax.set_title('Prediction error under input noise')
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "mean_error",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a bar chart comparing different models on a given metric.
        Reproduces the style of Fig. 4.

        Args:
            results: Dictionary mapping model name -> dict of metrics (must contain `metric`).
            metric: Which metric to compare (e.g., 'mean_error').
            save_path: Optional file path.
        """
        names = list(results.keys())
        values = [results[name][metric] for name in names]

        fig, ax = plt.subplots(figsize=(0.8 * len(names), 5))
        bars = ax.bar(names, values, color='skyblue', edgecolor='black')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Model comparison: {metric}')
        # Add text labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (x - mean) / (std + 1e-8)

    def _denormalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

