"""
Evaluation utilities for RWM.

Includes:
- Autoregressive rollout evaluation
- Noise robustness testing
- Model comparison utilities
- Metrics computation (relative prediction error)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from .world_model import RoboticWorldModel
from .baselines import MLPWorldModel, RSSM, TransformerWorldModel


def compute_relative_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    per_step: bool = True,
) -> torch.Tensor:
    """
    Compute relative prediction error e as defined in the paper.
    
    e = MSE(predicted, target) / Var(target)
    
    Args:
        predicted: (batch, N, dim) or (N, dim)
        target: same shape as predicted
        per_step: if True, return per-step errors
        
    Returns:
        Relative prediction error(s)
    """
    if per_step:
        # Compute per-step relative error
        N = predicted.shape[1]
        errors = []
        for k in range(N):
            pred_k = predicted[:, k, :]
            targ_k = target[:, k, :]
            mse = ((pred_k - targ_k) ** 2).mean()
            var = targ_k.var()
            errors.append((mse / (var + 1e-8)).item())
        return torch.tensor(errors)
    else:
        mse = ((predicted - target) ** 2).mean()
        var = target.var()
        return mse / (var + 1e-8)


def autoregressive_rollout(
    model: nn.Module,
    obs_history: torch.Tensor,
    act_history: torch.Tensor,
    act_future: torch.Tensor,
    use_sampling: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Perform autoregressive rollout for evaluation.
    
    Args:
        model: RWM or baseline model
        obs_history: (batch, M, obs_dim)
        act_history: (batch, M, act_dim)
        act_future: (batch, N, act_dim)
        use_sampling: if True, sample from distribution
    """
    if isinstance(model, RoboticWorldModel):
        if use_sampling:
            return model.autoregressive_forward_with_sampling(
                obs_history, act_history, act_future,
                use_reparameterization=False,
            )
        else:
            return model.autoregressive_forward(
                obs_history, act_history, act_future
            )
    elif isinstance(model, MLPWorldModel):
        return model.autoregressive_forward(
            obs_history, act_history, act_future
        )
    else:
        # For RSSM and Transformer, do step-by-step autoregression
        batch = obs_history.shape[0]
        N = act_future.shape[1]
        device = obs_history.device
        
        obs_means_list = []
        obs_stds_list = []
        
        obs_seq = obs_history.clone()
        act_seq = act_history.clone()
        h = None
        
        for k in range(N):
            if isinstance(model, RSSM):
                # RSSM needs full sequence but we feed step by step
                o_t = obs_seq[:, -1:, :]
                a_t = act_future[:, k:k+1, :]
                pred = model.forward(o_t, a_t, h)
            else:
                # For transformer, use recent context
                ctx_len = min(32, obs_seq.shape[1])
                pred = model.forward(obs_seq[:, -ctx_len:, :], act_seq[:, -ctx_len:, :])
                pred_obs_mean = pred['obs_means'][:, -1, :]
                pred_obs_std = pred['obs_stds'][:, -1, :]
                obs_means_list.append(pred_obs_mean)
                obs_stds_list.append(pred_obs_std)
                obs_seq = torch.cat([obs_seq, pred_obs_mean.unsqueeze(1)], dim=1)
                act_seq = torch.cat([act_seq, act_future[:, k:k+1, :]], dim=1)
                continue
            
            obs_mean = pred['obs_means'][:, -1, :]
            obs_std = pred['obs_stds'][:, -1, :]
            obs_means_list.append(obs_mean)
            obs_stds_list.append(obs_std)
            obs_seq = torch.cat([obs_seq, obs_mean.unsqueeze(1)], dim=1)
            act_seq = torch.cat([act_seq, act_future[:, k:k+1, :]], dim=1)
        
        return {
            'obs_means': torch.stack(obs_means_list, dim=1),
            'obs_stds': torch.stack(obs_stds_list, dim=1),
        }


def evaluate_noise_robustness(
    model: nn.Module,
    obs_history: torch.Tensor,
    act_history: torch.Tensor,
    act_future: torch.Tensor,
    noise_levels: List[float] = [0.0, 0.01, 0.05, 0.1, 0.2],
    noise_seed: int = 42,
) -> Dict[float, np.ndarray]:
    """
    Evaluate model robustness under varying noise levels (Section 4.2).
    
    Adds Gaussian noise to observations and actions, then measures
    prediction error over autoregressive rollouts.
    
    Args:
        model: World model
        obs_history: Clean observation history
        act_history: Clean action history
        act_future: Clean future actions
        noise_levels: List of noise standard deviations
        noise_seed: Random seed for reproducibility
        
    Returns:
        Dict mapping noise level to per-step relative errors
    """
    torch.manual_seed(noise_seed)
    results = {}
    
    for noise_std in noise_levels:
        # Add noise to observations
        noisy_obs = obs_history + noise_std * torch.randn_like(obs_history)
        noisy_act = act_history + noise_std * torch.randn_like(act_history)
        noisy_act_future = act_future + noise_std * torch.randn_like(act_future)
        
        # Perform rollout
        predictions = autoregressive_rollout(
            model, noisy_obs, noisy_act, noisy_act_future
        )
        
        # We need ground truth for comparison - use original predictions
        # without noise as reference? Or use clean rollout?
        # Paper compares predictions from noisy inputs to clean predictions
        clean_predictions = autoregressive_rollout(
            model, obs_history, act_history, act_future
        )
        
        # Compute error between noisy and clean predictions
        N = predictions['obs_means'].shape[1]
        errors = []
        for k in range(N):
            diff = (predictions['obs_means'][:, k, :] - clean_predictions['obs_means'][:, k, :]) ** 2
            errors.append(diff.mean().item())
        
        results[noise_std] = np.array(errors)
    
    return results


def compare_models(
    models: Dict[str, nn.Module],
    test_trajectories: List[Dict[str, np.ndarray]],
    history_horizon: int = 32,
    forecast_horizon: int = 8,
    batch_size: int = 256,
    device: torch.device = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple world model architectures on test trajectories.
    
    Computes the relative autoregressive prediction error for each model.
    Used for reproducing Fig. 4 results.
    
    Args:
        models: Dict mapping model name to model instance
        test_trajectories: List of test trajectories
        history_horizon: M
        forecast_horizon: N
        batch_size: Batch size for evaluation
        device: torch device
        
    Returns:
        Dict mapping model name to metrics dict
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare test windows
    from .training import TrajectoryDataset
    dataset = TrajectoryDataset(test_trajectories, history_horizon, forecast_horizon)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    results = {}
    
    for name, model in models.items():
        model.to(device)
        model.eval()
        
        all_per_step_errors = []
        
        with torch.no_grad():
            for batch in dataloader:
                obs = batch['obs'].to(device)
                act = batch['act'].to(device)
                
                M = history_horizon
                N = min(forecast_horizon, obs.shape[1] - M - 1)
                
                obs_history = obs[:, :M, :]
                act_history = act[:, :M, :]
                act_future = act[:, M-1:M+N-1, :]  # N actions for N predictions
                
                predictions = autoregressive_rollout(
                    model, obs_history, act_history, act_future
                )
                
                obs_targets = obs[:, M:M+N, :]
                per_step_error = compute_relative_error(
                    predictions['obs_means'][:, :N, :],
                    obs_targets,
                    per_step=True,
                )
                all_per_step_errors.append(per_step_error)
        
        # Average over batches
        mean_per_step = torch.stack(all_per_step_errors).mean(dim=0)
        
        results[name] = {
            'per_step_errors': mean_per_step.numpy(),
            'mean_error': mean_per_step.mean().item(),
            'final_step_error': mean_per_step[-1].item(),
        }
    
    return results
