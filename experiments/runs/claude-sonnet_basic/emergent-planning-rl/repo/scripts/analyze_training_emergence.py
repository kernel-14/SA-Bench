"""
Script to analyze the emergence of planning during training.

From the paper (Section 6.2, Figure 9):
- Collect checkpoints every 1M transitions for first 50M transitions
- For each checkpoint, measure:
  1. Macro F1 of probes for C_A and C_B
  2. Percentage of extra medium levels solved with 5 thinking steps
- Show strong correlation between these two quantities

Usage:
    python scripts/analyze_training_emergence.py 
        --checkpoint_dir /path/to/checkpoints 
        --data_dir /path/to/boxoban
        --output_dir /path/to/results
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
from environment.boxoban_loader import BoxobanLoader
from probing.linear_probe import LinearProbe, compute_macro_f1
from probing.probe_trainer import ConceptDataset, train_probe, evaluate_probe
from utils.data_collection import collect_dataset


def evaluate_checkpoint(
    checkpoint_path: str,
    train_levels: List[np.ndarray],
    val_levels: List[np.ndarray],
    medium_levels: List[np.ndarray],
    device: torch.device,
    num_train_episodes: int = 1000,
    num_val_episodes: int = 500,
    num_medium_episodes: int = 1000,
    num_thinking_steps: int = 5,
    num_probe_seeds: int = 5,
) -> Dict:
    """
    Evaluate a single checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        train_levels: Training levels for probe training
        val_levels: Validation levels for probe evaluation
        medium_levels: Medium difficulty levels for thinking steps evaluation
        device: Device
        num_train_episodes: Episodes for probe training
        num_val_episodes: Episodes for probe validation
        num_medium_episodes: Episodes for thinking steps evaluation
        num_thinking_steps: Number of thinking steps
        num_probe_seeds: Number of probe seeds
        
    Returns:
        Dict with evaluation results
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    D = checkpoint.get('D', 3)
    N = checkpoint.get('N', 3)
    hidden_channels = checkpoint.get('hidden_channels', 32)
    
    agent = DRCAgent(D=D, N=N, hidden_channels=hidden_channels).to(device)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()
    
    # Collect datasets for probing
    train_data = collect_dataset(agent, train_levels, num_train_episodes, device)
    val_data = collect_dataset(agent, val_levels, num_val_episodes, device)
    
    # Train and evaluate probes for each layer
    probe_results = {}
    
    for layer in range(D):
        probe_results[f'layer_{layer+1}'] = {}
        
        for concept in ['ca', 'cb']:
            cell_states_train = train_data['cell_states'][layer]
            cell_states_val = val_data['cell_states'][layer]
            
            if concept == 'ca':
                labels_train = train_data['ca_labels']
                labels_val = val_data['ca_labels']
            else:
                labels_train = train_data['cb_labels']
                labels_val = val_data['cb_labels']
            
            seed_f1s = []
            
            for seed in range(num_probe_seeds):
                torch.manual_seed(seed)
                
                probe = LinearProbe(
                    hidden_channels=hidden_channels,
                    num_classes=5,
                    probe_size=1,
                ).to(device)
                
                train_dataset = ConceptDataset(cell_states_train, labels_train)
                val_dataset = ConceptDataset(cell_states_val, labels_val)
                
                train_probe(probe, train_dataset, device=device)
                eval_results = evaluate_probe(probe, val_dataset, device=device)
                seed_f1s.append(eval_results['macro_f1'])
            
            probe_results[f'layer_{layer+1}'][concept] = {
                'mean_f1': np.mean(seed_f1s),
                'std_f1': np.std(seed_f1s),
            }
    
    # Evaluate thinking steps benefit
    env = SokobanEnv()
    
    # Base solve rate
    base_solved = set()
    for i in range(min(num_medium_episodes, len(medium_levels))):
        level = medium_levels[i]
        obs = env.reset(level)
        hidden_states = agent.init_hidden(batch_size=1, device=device)
        
        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _, hidden_states, _ = agent.forward(obs_tensor, hidden_states)
            action = logits.argmax(dim=-1).item()
            obs, _, done, info = env.step(action)
        
        if info.get('solved', False):
            base_solved.add(i)
    
    # Solve rate with thinking steps
    thinking_solved = set()
    for i in range(min(num_medium_episodes, len(medium_levels))):
        level = medium_levels[i]
        obs = env.reset(level)
        hidden_states = agent.init_hidden(batch_size=1, device=device)
        
        for _ in range(num_thinking_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                _, _, hidden_states, _ = agent.forward(obs_tensor, hidden_states)
            obs, _, done, _ = env.step(0)
            if done:
                break
        
        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _, hidden_states, _ = agent.forward(obs_tensor, hidden_states)
            action = logits.argmax(dim=-1).item()
            obs, _, done, info = env.step(action)
        
        if info.get('solved', False):
            thinking_solved.add(i)
    
    extra_solved = thinking_solved - base_solved
    extra_pct = len(extra_solved) / min(num_medium_episodes, len(medium_levels)) * 100
    
    return {
        'probe_results': probe_results,
        'base_solve_rate': len(base_solved) / min(num_medium_episodes, len(medium_levels)),
        'thinking_solve_rate': len(thinking_solved) / min(num_medium_episodes, len(medium_levels)),
        'extra_solved_pct': extra_pct,
        'total_transitions': checkpoint.get('total_transitions', 0),
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze emergence of planning during training')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory with checkpoints')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Boxoban dataset')
    parser.add_argument('--output_dir', type=str, default='emergence_results', help='Output directory')
    parser.add_argument('--num_train_episodes', type=int, default=1000, help='Training episodes for probes')
    parser.add_argument('--num_val_episodes', type=int, default=500, help='Validation episodes for probes')
    parser.add_argument('--num_medium_episodes', type=int, default=1000, help='Medium level episodes')
    parser.add_argument('--num_thinking_steps', type=int, default=5, help='Number of thinking steps')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Load levels
    print("Loading Boxoban levels...")
    loader = BoxobanLoader(args.data_dir)
    train_levels = loader.load_levels('train')
    val_levels = loader.load_levels('valid')
    medium_levels = loader.load_levels('medium', max_levels=args.num_medium_episodes)
    
    print(f"Loaded {len(train_levels)} train, {len(val_levels)} val, {len(medium_levels)} medium levels")
    
    # Find all checkpoints
    checkpoint_files = sorted([
        f for f in os.listdir(args.checkpoint_dir) 
        if f.endswith('.pt') and 'checkpoint' in f
    ])
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate each checkpoint
    all_results = []
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)
        print(f"\nEvaluating {checkpoint_file}...")
        
        try:
            results = evaluate_checkpoint(
                checkpoint_path,
                train_levels, val_levels, medium_levels,
                device,
                num_train_episodes=args.num_train_episodes,
                num_val_episodes=args.num_val_episodes,
                num_medium_episodes=args.num_medium_episodes,
                num_thinking_steps=args.num_thinking_steps,
            )
            
            results['checkpoint'] = checkpoint_file
            all_results.append(results)
            
            print(f"  Extra solved: {results['extra_solved_pct']:.1f}%")
            for layer_key, layer_results in results['probe_results'].items():
                ca_f1 = layer_results.get('ca', {}).get('mean_f1', 0)
                cb_f1 = layer_results.get('cb', {}).get('mean_f1', 0)
                print(f"  {layer_key}: C_A F1={ca_f1:.4f}, C_B F1={cb_f1:.4f}")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'emergence_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Print correlation summary (like Figure 9)
    print("\n=== Correlation: Probe F1 vs Extra Levels Solved ===")
    print(f"{'Checkpoint':>20} {'Extra Solved%':>15} {'C_A F1 (L3)':>12} {'C_B F1 (L3)':>12}")
    
    for result in all_results:
        extra_pct = result['extra_solved_pct']
        ca_f1 = result['probe_results'].get('layer_3', {}).get('ca', {}).get('mean_f1', 0)
        cb_f1 = result['probe_results'].get('layer_3', {}).get('cb', {}).get('mean_f1', 0)
        print(f"{result['checkpoint']:>20} {extra_pct:>15.1f} {ca_f1:>12.4f} {cb_f1:>12.4f}")


if __name__ == '__main__':
    main()
