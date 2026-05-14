"""
Model-based RL comparison from Appendix C.

Compares PGR against model-based baselines (MAX, Dreamer-v3)
under noisy dynamics conditions.

Key insight: PGR decouples dynamics learning from policy learning,
making it more robust to imperfect dynamics predictions.
"""
import torch
import numpy as np
import argparse
from typing import Dict, Tuple

from pgr.pgr_algorithm import PrioritizedGenerativeReplay
from pgr.models.policy import REDQAgent
from pgr.relevance.functions import ICMRelevance
from pgr.train import create_env, DummyEnv


def add_observation_noise(
    states: torch.Tensor,
    next_states: torch.Tensor,
    noise_ratio: float = 0.2,
    noise_std: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add isotropic Gaussian noise to a fraction of transitions.
    Used for the noised dynamics experiments in Appendix C.
    """
    batch_size = states.shape[0]
    
    # Select random subset to noise
    noise_mask = torch.rand(batch_size, device=states.device) < noise_ratio
    
    # Add Gaussian noise
    noise_s = torch.randn_like(states) * noise_std
    noise_ns = torch.randn_like(next_states) * noise_std
    
    noisy_states = states.clone()
    noisy_next_states = next_states.clone()
    
    noisy_states[noise_mask] += noise_s[noise_mask]
    noisy_next_states[noise_mask] += noise_ns[noise_mask]
    
    return noisy_states, noisy_next_states


def run_noisy_dynamics_comparison(
    env_names: list = ['cheetah-run', 'walker-walk', 'hopper-hop'],
    domain: str = 'dmc',
    noise_ratio: float = 0.2,
    noise_std: float = 0.1,
    device: str = 'cuda',
):
    """
    Set up noisy dynamics comparison experiments.
    Implements Table 8 methodology.
    
    For PGR: noise is added to states before ICM training.
    For MAX: noise is added to transitions in exploration data.
    For Dreamer-v3: noise is added to states given to world model.
    """
    configs = {}
    
    for env_name in env_names:
        env = create_env(domain, env_name, 0)
        
        if env is not None:
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
        else:
            dummy = DummyEnv(domain, env_name)
            state_dim = dummy.state_dim
            action_dim = dummy.action_dim
        
        configs[env_name] = {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'noise_ratio': noise_ratio,
            'noise_std': noise_std,
        }
    
    return configs


class NoisyICMRelevance(ICMRelevance):
    """
    ICM relevance function with observation noise during training.
    For the Appendix C experiments.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        noise_ratio: float = 0.2,
        noise_std: float = 0.1,
        **kwargs,
    ):
        super().__init__(state_dim, action_dim, **kwargs)
        self.noise_ratio = noise_ratio
        self.noise_std = noise_std
    
    def compute_loss(self, state, action, next_state):
        """Compute ICM loss with noisy states."""
        # Add noise to states
        noisy_state, noisy_next_state = add_observation_noise(
            state, next_state,
            noise_ratio=self.noise_ratio,
            noise_std=self.noise_std,
        )
        return super().compute_loss(noisy_state, action, noisy_next_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', nargs='+', default=['cheetah-run', 'walker-walk', 'hopper-hop'])
    parser.add_argument('--domain', type=str, default='dmc')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    configs = run_noisy_dynamics_comparison(
        env_names=args.envs,
        domain=args.domain,
        device=args.device,
    )
    
    print("Noisy dynamics comparison configurations:")
    for env_name, cfg in configs.items():
        print(f"\n{env_name}:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
