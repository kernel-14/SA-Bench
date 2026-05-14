"""Evaluation metrics for world model and policy training.

Includes:
- Relative autoregressive prediction error (as in Fig 3, Fig 4)
- Rollout evaluation comparing predicted vs ground truth trajectories
"""

from typing import Dict
import numpy as np
import torch


def compute_relative_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    per_step: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute relative prediction error: ||pred - target|| / ||target||.

    Args:
        predictions: (..., horizon, dim) — predicted values
        targets: (..., horizon, dim) — ground truth values
        per_step: If True, return error per forecast step. Otherwise mean.

    Returns:
        dict with 'relative_error': array of shape (horizon,) or scalar
    """
    with torch.no_grad():
        error = torch.norm(predictions - targets, dim=-1)  # (..., horizon)
        norm_target = torch.norm(targets, dim=-1)  # (..., horizon)
        rel_error = error / (norm_target + 1e-8)

    if per_step:
        result = rel_error.mean(dim=0).cpu().numpy()  # (horizon,)
    else:
        result = rel_error.mean().cpu().numpy()

    return {"relative_error": result}


def evaluate_rollout(
    model,
    observations: torch.Tensor,
    actions: torch.Tensor,
    history_horizon: int,
    forecast_horizon: int,
) -> Dict[str, np.ndarray]:
    """Perform a full autoregressive rollout evaluation.

    Args:
        model: World model with forecast_autoregressive method
        observations: (B, total_length, obs_dim)
        actions: (B, total_length-1, action_dim)
        history_horizon: M
        forecast_horizon: number of steps to predict

    Returns:
        dict with predictions and ground truth arrays
    """
    model.eval()
    device = next(model.parameters()).device

    observations = observations.to(device)
    actions = actions.to(device)

    batch_size = observations.shape[0]
    M = history_horizon

    obs_history = observations[:, :M]
    effective_horizon = min(forecast_horizon, observations.shape[1] - M)

    with torch.no_grad():
        result = model.forecast_autoregressive(
            observations=obs_history,
            actions=actions[:, :M - 1 + effective_horizon],
            forecast_horizon=effective_horizon,
        )

    pred_obs = result["predicted_obs"].cpu().numpy()
    true_obs = observations[:, M:M + effective_horizon].cpu().numpy()

    metrics = compute_relative_error(
        torch.as_tensor(pred_obs),
        torch.as_tensor(true_obs),
        per_step=True,
    )

    return {
        "predicted_observations": pred_obs,
        "true_observations": true_obs,
        "relative_error": metrics["relative_error"],
    }
