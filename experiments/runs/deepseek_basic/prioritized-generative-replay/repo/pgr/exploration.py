"""
Exploration bonus experiments from Section 5.1 and Appendix B.

Compares PGR against:
- Explicit exploration bonuses (ICM curiosity, RND)
- Implicit exploration bonuses (NoisyNets, Bootstrapped DQN)
- Combined PGR + exploration bonuses
"""
import torch
import torch.nn as nn
import numpy as np
import argparse
from typing import Dict, Tuple

from pgr.pgr_algorithm import PrioritizedGenerativeReplay
from pgr.models.policy import (
    REDQAgent, 
    MLPActor, 
    REDQQNetwork,
    NoisyLinear,
    BootstrappedQNetwork,
)
from pgr.relevance.functions import ICMRelevance, RNDRelevance
from pgr.train import create_env, DummyEnv


def create_noisy_redq(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 2,
    device: str = 'cuda',
) -> REDQAgent:
    """
    Create REDQ agent with noisy networks (Fortunato et al., 2018).
    Replaces linear layers with NoisyLinear.
    """
    # Note: this modifies the Q-network architecture
    agent = REDQAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
    )
    
    # Replace Q-network layers with noisy versions
    for q_net in agent.critic.q_networks:
        # Replace linear layers
        modules = list(q_net.children())
        new_modules = []
        for module in modules:
            if isinstance(module, nn.Linear):
                noisy = NoisyLinear(module.in_features, module.out_features)
                new_modules.append(noisy)
            else:
                new_modules.append(module)
        # Rebuild sequential
        q_net._modules = {str(i): m for i, m in enumerate(new_modules)}
    
    # Same for target
    for q_net in agent.critic_target.q_networks:
        modules = list(q_net.children())
        new_modules = []
        for module in modules:
            if isinstance(module, nn.Linear):
                noisy = NoisyLinear(module.in_features, module.out_features)
                new_modules.append(noisy)
            else:
                new_modules.append(module)
        q_net._modules = {str(i): m for i, m in enumerate(new_modules)}
    
    return agent


def create_bootstrapped_redq(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_heads: int = 10,
    device: str = 'cuda',
):
    """
    Create REDQ agent with bootstrapped Q-values (Osband et al., 2016).
    """
    agent = REDQAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        ensemble_size=num_heads,
        device=device,
    )
    
    # The existing REDQ ensemble already provides bootstrapping
    # Add mask sampling for data subsets
    agent.use_bootstrapping = True
    agent.num_heads = num_heads
    
    return agent


def add_exploration_bonus_to_agent(
    agent: REDQAgent,
    state_dim: int,
    action_dim: int,
    bonus_type: str = 'curiosity',
    bonus_weight: float = 0.1,
    device: str = 'cuda',
):
    """
    Add an intrinsic exploration bonus (reward) to a base RL agent.
    Used for the "REDQ + Curiosity" and "SYNTHER + Curiosity" baselines.
    """
    if bonus_type == 'curiosity':
        bonus_module = ICMRelevance(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)
        bonus_optimizer = torch.optim.Adam(bonus_module.parameters(), lr=1e-3)
    elif bonus_type == 'rnd':
        bonus_module = RNDRelevance(
            state_dim=state_dim,
        ).to(device)
        bonus_optimizer = torch.optim.Adam(
            [p for p in bonus_module.parameters() if p.requires_grad], 
            lr=1e-3
        )
    else:
        bonus_module = None
        bonus_optimizer = None
    
    return {
        'module': bonus_module,
        'optimizer': bonus_optimizer,
        'weight': bonus_weight,
        'type': bonus_type,
    }


def run_exploration_comparison(
    env_name: str = 'quadruped-walk',
    domain: str = 'dmc',
    device: str = 'cuda',
):
    """
    Set up and compare various exploration methods.
    Implements Tables 6 and 7 comparisons.
    
    Methods:
    - REDQ (baseline)
    - REDQ + Curiosity (explicit bonus)
    - PGR (NoisyNets) (implicit bonus combined with PGR)
    - PGR (BootDQN) (implicit bonus combined with PGR)
    - PGR (Curiosity) (standard PGR)
    """
    env = create_env(domain, env_name, 0)
    
    if env is not None:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        dummy = DummyEnv(domain, env_name)
        state_dim = dummy.state_dim
        action_dim = dummy.action_dim
    
    configs = {}
    
    # 1. REDQ baseline
    configs['redq'] = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': 256,
        'num_layers': 2,
    }
    
    # 2. REDQ + Curiosity (explicit bonus)
    configs['redq_curiosity'] = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': 256,
        'num_layers': 2,
        'bonus_type': 'curiosity',
        'bonus_weight': 0.1,
    }
    
    # 3. SYNTHER baseline (PGR with no condition = unconditional)
    configs['synther'] = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': 256,
        'num_layers': 2,
        'use_generation': True,
        'conditional': False,
    }
    
    # 4. SYNTHER + Curiosity
    configs['synther_curiosity'] = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': 256,
        'num_layers': 2,
        'bonus_type': 'curiosity',
        'bonus_weight': 0.1,
        'use_generation': True,
        'conditional': False,
    }
    
    # 5. PGR (NoisyNets)
    configs['pgr_noisy'] = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': 256,
        'num_layers': 2,
        'relevance': 'curiosity',
        'use_noisy': True,
    }
    
    # 6. PGR (BootDQN)
    configs['pgr_bootdqn'] = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': 256,
        'num_layers': 2,
        'relevance': 'curiosity',
        'use_bootstrap': True,
    }
    
    # 7. PGR (Curiosity) - standard
    configs['pgr_curiosity'] = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': 256,
        'num_layers': 2,
        'relevance': 'curiosity',
    }
    
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='quadruped-walk')
    parser.add_argument('--domain', type=str, default='dmc')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    configs = run_exploration_comparison(
        env_name=args.env,
        domain=args.domain,
        device=args.device,
    )
    
    print("Exploration comparison configurations:")
    for name, cfg in configs.items():
        print(f"\n{name}:")
        for k, v in cfg.items():
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
