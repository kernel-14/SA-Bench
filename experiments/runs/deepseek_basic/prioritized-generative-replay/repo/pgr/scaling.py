"""
Scaling experiments from Section 5.3 of the paper.

Three experiments:
1. Network size scaling (Fig. 7a)
2. Synthetic data ratio scaling (Fig. 7b)
3. Combined scaling with higher UTD (Fig. 7c)
"""
import torch
import numpy as np
import argparse
from typing import Dict, List, Tuple
import json

from pgr.pgr_algorithm import PrioritizedGenerativeReplay
from pgr.models.policy import REDQAgent
from pgr.train import create_env, DummyEnv, evaluate, run_episode


def run_scaling_experiment(
    env_name: str = 'quadruped-walk',
    domain: str = 'dmc',
    total_steps: int = 100_000,
    seed: int = 0,
    device: str = 'cuda',
):
    """
    Run all three scaling experiments for PGR and SYNTHER comparison.
    """
    env = create_env(domain, env_name, seed)
    eval_env = create_env(domain, env_name, seed + 1000)
    
    if env is not None:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        dummy = DummyEnv(domain, env_name)
        state_dim = dummy.state_dim
        action_dim = dummy.action_dim
    
    results = {}
    
    # === Experiment 1: Network Size Scaling ===
    print("=" * 60)
    print("Experiment 1: Network Size Scaling")
    print("=" * 60)
    
    configs_small = {
        'hidden_dim': 256,
        'num_layers': 2,
        'batch_size': 256,
        'synthetic_ratio': 0.5,
        'utd_ratio': 20,
    }
    
    configs_large = {
        'hidden_dim': 512,
        'num_layers': 3,
        'batch_size': 1024,
        'synthetic_ratio': 0.5,
        'utd_ratio': 20,
    }
    
    # PGR Small
    print("\nPGR Small Network...")
    pgr_small = PrioritizedGenerativeReplay(
        state_dim=state_dim,
        action_dim=action_dim,
        relevance_type='curiosity',
        synthetic_data_ratio=configs_small['synthetic_ratio'],
        batch_size=configs_small['batch_size'],
        utd_ratio=configs_small['utd_ratio'],
        device=device,
    )
    agent_small = REDQAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=configs_small['hidden_dim'],
        num_layers=configs_small['num_layers'],
        utd_ratio=configs_small['utd_ratio'],
        device=device,
    )
    pgr_small.set_policy(agent_small.actor, agent_small.critic, agent_small.critic_target)
    
    # PGR Large
    print("\nPGR Large Network...")
    pgr_large = PrioritizedGenerativeReplay(
        state_dim=state_dim,
        action_dim=action_dim,
        relevance_type='curiosity',
        synthetic_data_ratio=configs_large['synthetic_ratio'],
        batch_size=configs_large['batch_size'],
        utd_ratio=configs_large['utd_ratio'],
        device=device,
    )
    agent_large = REDQAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=configs_large['hidden_dim'],
        num_layers=configs_large['num_layers'],
        utd_ratio=configs_large['utd_ratio'],
        device=device,
    )
    pgr_large.set_policy(agent_large.actor, agent_large.critic, agent_large.critic_target)
    
    results['network_size'] = {
        'pgr_small_config': configs_small,
        'pgr_large_config': configs_large,
    }
    
    # === Experiment 2: Synthetic Data Ratio Scaling ===
    print("\n" + "=" * 60)
    print("Experiment 2: Synthetic Data Ratio Scaling")
    print("=" * 60)
    
    ratio_configs = [
        {'batch_size': 256, 'synthetic_ratio': 0.5, 'label': 'r=0.5'},
        {'batch_size': 512, 'synthetic_ratio': 0.75, 'label': 'r=0.75'},
        {'batch_size': 1024, 'synthetic_ratio': 0.875, 'label': 'r=0.875'},
    ]
    
    results['synthetic_ratio'] = []
    
    for cfg in ratio_configs:
        print(f"\nPGR {cfg['label']}...")
        pgr = PrioritizedGenerativeReplay(
            state_dim=state_dim,
            action_dim=action_dim,
            relevance_type='curiosity',
            synthetic_data_ratio=cfg['synthetic_ratio'],
            batch_size=cfg['batch_size'],
            utd_ratio=20,
            device=device,
        )
        agent = REDQAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            num_layers=2,
            utd_ratio=20,
            device=device,
        )
        pgr.set_policy(agent.actor, agent.critic, agent.critic_target)
        results['synthetic_ratio'].append(cfg)
    
    # === Experiment 3: Combined Scaling with UTD ===
    print("\n" + "=" * 60)
    print("Experiment 3: Combined Scaling")
    print("=" * 60)
    
    combined_configs = [
        {
            'hidden_dim': 256, 'num_layers': 2,
            'batch_size': 256, 'synthetic_ratio': 0.5, 'utd_ratio': 20,
            'syn_buffer_size': 1_000_000,
            'label': 'Base (Small, r=0.5, UTD=20)'
        },
        {
            'hidden_dim': 512, 'num_layers': 3,
            'batch_size': 1024, 'synthetic_ratio': 0.75, 'utd_ratio': 20,
            'syn_buffer_size': 1_000_000,
            'label': 'Large + r=0.75, UTD=20'
        },
        {
            'hidden_dim': 512, 'num_layers': 3,
            'batch_size': 1024, 'synthetic_ratio': 0.75, 'utd_ratio': 40,
            'syn_buffer_size': 2_000_000,
            'label': 'Large + r=0.75, UTD=40'
        },
    ]
    
    results['combined'] = []
    
    for cfg in combined_configs:
        print(f"\nPGR {cfg['label']}...")
        pgr = PrioritizedGenerativeReplay(
            state_dim=state_dim,
            action_dim=action_dim,
            relevance_type='curiosity',
            synthetic_data_ratio=cfg['synthetic_ratio'],
            batch_size=cfg['batch_size'],
            utd_ratio=cfg['utd_ratio'],
            buffer_size=cfg['syn_buffer_size'],
            device=device,
        )
        agent = REDQAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            utd_ratio=cfg['utd_ratio'],
            device=device,
        )
        pgr.set_policy(agent.actor, agent.critic, agent.critic_target)
        results['combined'].append(cfg)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='quadruped-walk')
    parser.add_argument('--domain', type=str, default='dmc')
    parser.add_argument('--output', type=str, default='scaling_results.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    results = run_scaling_experiment(
        env_name=args.env,
        domain=args.domain,
        device=args.device,
    )
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
