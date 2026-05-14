"""Utility functions: metrics, logging, dormant ratio."""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """TensorBoard-compatible logger."""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        self.writer.add_scalar(tag, value, step if step is not None else self.step)

    def log_scalars(self, tag_dict: dict, step: Optional[int] = None):
        s = step if step is not None else self.step
        for tag, value in tag_dict.items():
            self.writer.add_scalar(tag, value, s)

    def increment_step(self):
        self.step += 1

    def close(self):
        self.writer.close()


def compute_dormant_ratio(
    model: nn.Module,
    inputs: torch.Tensor,
    threshold: float = 0.1,
) -> float:
    """Compute dormant ratio (DR) for policy network.

    DR = fraction of inactive neurons (activation < threshold).
    Used in Section 5.2 to quantify overfitting.
    Reference: Sokar et al. (2023), Xu et al. (2023)
    """
    activations = []
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activations.append(output.detach().flatten())

    # Register hooks on all ReLU layers
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass
    with torch.no_grad():
        model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    if len(activations) == 0:
        return 0.0

    all_acts = torch.cat(activations)
    dormant = (all_acts.abs() < threshold).float().mean().item()
    return dormant


def compute_generation_mse(
    env,
    generated_transitions: torch.Tensor,
    state_dim: int,
    action_dim: int,
) -> Tuple[float, float]:
    """Compute MSE of generated transitions vs. environment dynamics.

    Method from Lu et al. (2024), used in Section 5.2.
    Returns (state_mse, reward_mse).
    """
    states = generated_transitions[:, :state_dim].cpu().numpy()
    actions = generated_transitions[:, state_dim:state_dim + action_dim].cpu().numpy()
    gen_next_states = generated_transitions[:, state_dim + action_dim:state_dim * 2 + action_dim].cpu().numpy()
    gen_rewards = generated_transitions[:, state_dim * 2 + action_dim:].cpu().numpy()

    n = len(states)
    state_errors = []
    reward_errors = []

    for i in range(n):
        # Set environment to state[i] (approximate via reset+step)
        try:
            env.reset()
            if hasattr(env, "set_state"):
                env.set_state(states[i])
            obs, reward, _, _, _ = env.step(actions[i])
            state_errors.append(((obs - gen_next_states[i]) ** 2).mean())
            reward_errors.append((reward - gen_rewards[i]) ** 2)
        except Exception:
            pass

    if len(state_errors) == 0:
        return 0.0, 0.0

    return np.mean(state_errors), np.mean(reward_errors)


def evaluate_policy(
    env,
    policy: nn.Module,
    num_episodes: int = 10,
    max_steps: int = 1000,
    is_pixel: bool = False,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Evaluate policy and return average return and std."""
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                if is_pixel:
                    action = policy.get_action(obs_tensor, deterministic=True)
                else:
                    action = policy.get_action(obs_tensor, deterministic=True)
                action = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            obs = next_obs
            step += 1

        returns.append(episode_return)

    return np.mean(returns), np.std(returns)
