"""
Script to run intervention experiments (Section 6.1, Table 1).

From the paper:
- Agent-Shortcut (AS) interventions: steer agent to take longer path
- Box-Shortcut (BS) interventions: steer agent to push box longer route
- Test at each of the 3 ConvLSTM layers
- Compare trained probes vs randomly initialized probes
- Report success rates averaged over 5 probe seeds

Expected results (Table 1):
                Layer 1         Layer 2         Layer 3
AS  Trained:   94.6 (±0.5)    90.1 (±1.9)    98.8 (±0.0)
AS  Random:    33.7 (±32.7)   29.8 (±36.8)   27.8 (±37.9)
BS  Trained:   56.2 (±1.4)    72.7 (±1.1)    80.6 (±2.4)
BS  Random:    31.5 (±13.9)   30.9 (±25.8)    4.1 (±5.4)

Usage:
    python scripts/run_interventions.py 
        --agent_path /path/to/agent.pt 
        --probe_dir /path/to/probes
        --levels_dir /path/to/intervention_levels
"""

import argparse
import os
import sys
import torch
import numpy as np
import json
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent.drc_agent import DRCAgent
from environment.sokoban import SokobanEnv
from probing.linear_probe import LinearProbe
from interventions.interventions import (
    AgentShortcutIntervention, BoxShortcutIntervention, run_intervention_episode
)
from probing.concepts import ConceptClass


def load_intervention_levels(levels_dir: str, intervention_type: str) -> List[Dict]:
    """
    Load intervention levels and their configurations.
    
    Args:
        levels_dir: Directory containing level files
        intervention_type: 'agent_shortcut' or 'box_shortcut'
        
    Returns:
        List of dicts with level and intervention config
    """
    import json
    
    config_path = os.path.join(levels_dir, f'{intervention_type}_configs.json')
    
    if not os.path.exists(config_path):
        print(f"Warning: No config file found at {config_path}")
        print("Using example levels for demonstration...")
        return create_example_levels(intervention_type)
    
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    return configs


def create_example_levels(intervention_type: str) -> List[Dict]:
    """
    Create example intervention levels for demonstration.
    
    These are simplified versions of the handcrafted levels described in the paper.
    The paper uses 25 handcrafted levels × 8 augmentations = 200 levels.
    """
    levels = []
    
    if intervention_type == 'agent_shortcut':
        # Example Agent-Shortcut level:
        # Agent can go directly right to boxes/targets (short path)
        # or go down and around (long path)
        level = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 1, 1, 1, 1, 1, 0],  # agent at (1,1)
            [0, 0, 0, 0, 1, 1, 1, 0],  # wall blocking short path
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 2, 1, 1, 0],  # box at (4,4)
            [0, 1, 1, 1, 6, 1, 1, 0],  # target at (5,4)
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int32)
        
        config = {
            'level': level.tolist(),
            'short_route': [(1, 2), (1, 3)],  # squares on short path
            'long_route_dirs': [((2, 1), ConceptClass.DOWN)],  # first square of long path
            'desired_outcome': 'long_path',
        }
        levels.append(config)
    
    else:  # box_shortcut
        # Example Box-Shortcut level:
        # Three boxes adjacent to targets, one box can go short or long route
        level = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 4, 4, 4, 1, 1, 1, 0],  # 3 boxes on targets
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 3, 1, 1, 1, 1, 0],  # agent at (3,2)
            [0, 1, 2, 1, 1, 1, 1, 0],  # box at (4,2)
            [0, 1, 6, 1, 6, 1, 1, 0],  # targets at (5,2) and (5,4)
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int32)
        
        config = {
            'level': level.tolist(),
            'short_route': [(4, 2)],  # short route: push box directly down
            'long_route_dirs': [((4, 2), ConceptClass.RIGHT)],  # push box right first
            'box_initial_pos': (4, 2),
            'desired_outcome': 'long_route',
        }
        levels.append(config)
    
    return levels


def run_intervention_experiment(
    agent: DRCAgent,
    probe: LinearProbe,
    levels_configs: List[Dict],
    intervention_type: str,
    layer: int,
    alpha: float,
    device: torch.device,
    use_random_probe: bool = False,
    num_seeds: int = 5,
) -> Dict:
    """
    Run intervention experiment.
    
    Args:
        agent: DRC agent
        probe: Trained probe
        levels_configs: List of level configurations
        intervention_type: 'agent_shortcut' or 'box_shortcut'
        layer: Layer to intervene on (0-indexed)
        alpha: Intervention strength
        device: Device
        use_random_probe: Whether to use random probe
        num_seeds: Number of seeds
        
    Returns:
        Dict with success rates
    """
    env = SokobanEnv()
    success_rates = []
    
    for seed in range(num_seeds):
        if use_random_probe:
            torch.manual_seed(seed)
            test_probe = LinearProbe(hidden_channels=32, num_classes=5, probe_size=1).to(device)
            # Scale to match trained probe norm
            with torch.no_grad():
                trained_norm = probe.get_class_vectors().norm()
                random_norm = test_probe.get_class_vectors().norm()
                if random_norm > 0:
                    test_probe.conv.weight.data *= (trained_norm / random_norm)
        else:
            test_probe = probe
        
        successes = 0
        total = 0
        
        for config in levels_configs:
            level = np.array(config['level'], dtype=np.int32)
            
            if intervention_type == 'agent_shortcut':
                intervention = AgentShortcutIntervention(
                    short_route_squares=[tuple(s) for s in config['short_route']],
                    long_route_squares_dirs=[
                        (tuple(s), d) for s, d in config['long_route_dirs']
                    ],
                    probe_ca=test_probe,
                    layer=layer,
                    alpha=alpha,
                    p=1,
                )
            else:
                intervention = BoxShortcutIntervention(
                    short_route_squares=[tuple(s) for s in config['short_route']],
                    long_route_squares_dirs=[
                        (tuple(s), d) for s, d in config['long_route_dirs']
                    ],
                    box_initial_pos=tuple(config['box_initial_pos']),
                    probe_cb=test_probe,
                    layer=layer,
                    alpha=alpha,
                    p=1,
                )
            
            result = run_intervention_episode(
                agent, env, level, intervention, device
            )
            
            if result['solved']:
                successes += 1
            total += 1
        
        if total > 0:
            success_rates.append(successes / total * 100)
    
    return {
        'mean': np.mean(success_rates),
        'std': np.std(success_rates),
        'per_seed': success_rates,
    }


def main():
    parser = argparse.ArgumentParser(description='Run intervention experiments (Table 1)')
    parser.add_argument('--agent_path', type=str, required=True, help='Path to trained agent')
    parser.add_argument('--probe_dir', type=str, required=True, help='Path to trained probes')
    parser.add_argument('--levels_dir', type=str, default=None, help='Path to intervention levels')
    parser.add_argument('--output_dir', type=str, default='intervention_results', help='Output directory')
    parser.add_argument('--alpha', type=float, default=1.0, help='Intervention strength')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Load agent
    checkpoint = torch.load(args.agent_path, map_location=device)
    D = checkpoint.get('D', 3)
    N = checkpoint.get('N', 3)
    hidden_channels = checkpoint.get('hidden_channels', 32)
    
    agent = DRCAgent(D=D, N=N, hidden_channels=hidden_channels).to(device)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()
    
    print(f"Loaded DRC({D},{N}) agent")
    
    # Load levels
    if args.levels_dir:
        as_configs = load_intervention_levels(args.levels_dir, 'agent_shortcut')
        bs_configs = load_intervention_levels(args.levels_dir, 'box_shortcut')
    else:
        print("No levels directory provided, using example levels...")
        as_configs = create_example_levels('agent_shortcut')
        bs_configs = create_example_levels('box_shortcut')
    
    print(f"Loaded {len(as_configs)} Agent-Shortcut levels, {len(bs_configs)} Box-Shortcut levels")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments for each layer
    results = {}
    
    print("\n=== Intervention Results (Table 1) ===")
    print(f"{'':5} {'Layer 1 Trained':>18} {'Layer 1 Random':>16} "
          f"{'Layer 2 Trained':>18} {'Layer 2 Random':>16} "
          f"{'Layer 3 Trained':>18} {'Layer 3 Random':>16}")
    
    for intervention_type, configs, concept in [
        ('agent_shortcut', as_configs, 'ca'),
        ('box_shortcut', bs_configs, 'cb'),
    ]:
        row_results = {}
        
        for layer in range(D):
            # Load probe
            probe_path = os.path.join(
                args.probe_dir, 
                f'probe_layer{layer+1}_{concept}_1x1_best.pt'
            )
            
            if not os.path.exists(probe_path):
                print(f"Warning: Probe not found at {probe_path}")
                continue
            
            probe = LinearProbe(hidden_channels=hidden_channels, num_classes=5, probe_size=1).to(device)
            probe.load_state_dict(torch.load(probe_path, map_location=device))
            probe.eval()
            
            # Trained probe
            trained_results = run_intervention_experiment(
                agent, probe, configs, intervention_type, layer, args.alpha, device,
                use_random_probe=False, num_seeds=args.num_seeds
            )
            
            # Random probe
            random_results = run_intervention_experiment(
                agent, probe, configs, intervention_type, layer, args.alpha, device,
                use_random_probe=True, num_seeds=args.num_seeds
            )
            
            row_results[f'layer_{layer+1}'] = {
                'trained': trained_results,
                'random': random_results,
            }
        
        results[intervention_type] = row_results
        
        # Print row
        label = 'AS' if intervention_type == 'agent_shortcut' else 'BS'
        row_str = f"{label:5}"
        for layer in range(D):
            if f'layer_{layer+1}' in row_results:
                t = row_results[f'layer_{layer+1}']['trained']
                r = row_results[f'layer_{layer+1}']['random']
                row_str += f" {t['mean']:>6.1f} (±{t['std']:.1f})"
                row_str += f" {r['mean']:>6.1f} (±{r['std']:.1f})"
        print(row_str)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'intervention_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
