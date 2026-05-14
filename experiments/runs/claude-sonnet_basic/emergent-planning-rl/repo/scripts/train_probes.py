"""
Script to train linear probes for concept-based interpretability.

From the paper (Section 4):
- Train 1x1 and 3x3 probes to predict C_A and C_B
- Train probes at each of the 3 ConvLSTM layers
- Train baseline probes using raw observations
- Use 5 random seeds per probe
- Evaluate using macro F1 score

Usage:
    python scripts/train_probes.py --agent_path /path/to/agent.pt --data_dir /path/to/boxoban
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
from probing.linear_probe import LinearProbe, BaselineProbe, compute_macro_f1
from probing.probe_trainer import (
    ConceptDataset, ObsDataset, train_probe, evaluate_probe
)
from utils.data_collection import collect_dataset


def main():
    parser = argparse.ArgumentParser(description='Train linear probes for concept interpretability')
    parser.add_argument('--agent_path', type=str, required=True, help='Path to trained agent')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Boxoban dataset')
    parser.add_argument('--output_dir', type=str, default='probe_results', help='Output directory')
    parser.add_argument('--num_train_episodes', type=int, default=3000, help='Training episodes')
    parser.add_argument('--num_test_episodes', type=int, default=1000, help='Test episodes')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--probe_sizes', type=int, nargs='+', default=[1, 3], help='Probe sizes')
    parser.add_argument('--num_epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load agent
    print(f"Loading agent from {args.agent_path}...")
    checkpoint = torch.load(args.agent_path, map_location=device)
    
    D = checkpoint.get('D', 3)
    N = checkpoint.get('N', 3)
    hidden_channels = checkpoint.get('hidden_channels', 32)
    
    agent = DRCAgent(D=D, N=N, hidden_channels=hidden_channels).to(device)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()
    
    print(f"Loaded DRC({D},{N}) agent")
    
    # Load levels
    print("Loading Boxoban levels...")
    loader = BoxobanLoader(args.data_dir)
    train_levels = loader.load_levels('train')
    val_levels = loader.load_levels('valid')
    
    print(f"Loaded {len(train_levels)} training levels, {len(val_levels)} validation levels")
    
    # Collect datasets
    print(f"Collecting training dataset ({args.num_train_episodes} episodes)...")
    train_data = collect_dataset(
        agent, train_levels, args.num_train_episodes, device, verbose=True
    )
    
    print(f"Collecting test dataset ({args.num_test_episodes} episodes)...")
    test_data = collect_dataset(
        agent, val_levels, args.num_test_episodes, device, verbose=True
    )
    
    print(f"Training transitions: {len(train_data['ca_labels'])}")
    print(f"Test transitions: {len(test_data['ca_labels'])}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train probes for each layer and concept
    results = {}
    
    for layer in range(D):
        results[f'layer_{layer+1}'] = {}
        
        for concept in ['ca', 'cb']:
            results[f'layer_{layer+1}'][concept] = {}
            
            # Get cell states and labels
            cell_states_train = train_data['cell_states'][layer]
            cell_states_test = test_data['cell_states'][layer]
            
            if concept == 'ca':
                labels_train = train_data['ca_labels']
                labels_test = test_data['ca_labels']
            else:
                labels_train = train_data['cb_labels']
                labels_test = test_data['cb_labels']
            
            for probe_size in args.probe_sizes:
                seed_f1s = []
                trained_probes = []
                
                for seed in range(args.num_seeds):
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    
                    probe = LinearProbe(
                        hidden_channels=hidden_channels,
                        num_classes=5,
                        probe_size=probe_size,
                    ).to(device)
                    
                    train_dataset = ConceptDataset(cell_states_train, labels_train)
                    test_dataset = ConceptDataset(cell_states_test, labels_test)
                    
                    train_probe(
                        probe, train_dataset,
                        num_epochs=args.num_epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.lr,
                        weight_decay=args.weight_decay,
                        device=device,
                    )
                    
                    eval_results = evaluate_probe(probe, test_dataset, device=device)
                    seed_f1s.append(eval_results['macro_f1'])
                    trained_probes.append(probe)
                    
                    print(f"Layer {layer+1}, {concept.upper()}, {probe_size}x{probe_size}, "
                          f"Seed {seed}: Macro F1 = {eval_results['macro_f1']:.4f}")
                
                mean_f1 = np.mean(seed_f1s)
                std_f1 = np.std(seed_f1s)
                
                results[f'layer_{layer+1}'][concept][f'{probe_size}x{probe_size}'] = {
                    'mean_f1': mean_f1,
                    'std_f1': std_f1,
                    'f1_per_seed': seed_f1s,
                }
                
                print(f"\nLayer {layer+1}, {concept.upper()}, {probe_size}x{probe_size}: "
                      f"Macro F1 = {mean_f1:.4f} ± {std_f1:.4f}")
                
                # Save best probe
                best_seed = np.argmax(seed_f1s)
                probe_path = os.path.join(
                    args.output_dir,
                    f'probe_layer{layer+1}_{concept}_{probe_size}x{probe_size}_best.pt'
                )
                torch.save(trained_probes[best_seed].state_dict(), probe_path)
    
    # Train baseline probes
    print("\nTraining baseline probes...")
    results['baseline'] = {}
    
    for concept in ['ca', 'cb']:
        results['baseline'][concept] = {}
        
        if concept == 'ca':
            labels_train = train_data['ca_labels']
            labels_test = test_data['ca_labels']
        else:
            labels_train = train_data['cb_labels']
            labels_test = test_data['cb_labels']
        
        for probe_size in args.probe_sizes:
            seed_f1s = []
            
            for seed in range(args.num_seeds):
                torch.manual_seed(seed)
                
                probe = BaselineProbe(obs_channels=7, num_classes=5, probe_size=probe_size).to(device)
                
                train_dataset = ObsDataset(train_data['observations'], labels_train)
                test_dataset = ObsDataset(test_data['observations'], labels_test)
                
                train_probe(
                    probe, train_dataset,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    weight_decay=args.weight_decay,
                    device=device,
                )
                
                eval_results = evaluate_probe(probe, test_dataset, device=device)
                seed_f1s.append(eval_results['macro_f1'])
            
            mean_f1 = np.mean(seed_f1s)
            std_f1 = np.std(seed_f1s)
            
            results['baseline'][concept][f'{probe_size}x{probe_size}'] = {
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'f1_per_seed': seed_f1s,
            }
            
            print(f"Baseline, {concept.upper()}, {probe_size}x{probe_size}: "
                  f"Macro F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'probe_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Print summary table (like Figure 4 in paper)
    print("\n=== Summary (Macro F1 Scores) ===")
    print(f"{'':20} {'C_A 1x1':>10} {'C_A 3x3':>10} {'C_B 1x1':>10} {'C_B 3x3':>10}")
    
    for layer in range(D):
        layer_key = f'layer_{layer+1}'
        ca_1x1 = results[layer_key]['ca'].get('1x1', {}).get('mean_f1', 0)
        ca_3x3 = results[layer_key]['ca'].get('3x3', {}).get('mean_f1', 0)
        cb_1x1 = results[layer_key]['cb'].get('1x1', {}).get('mean_f1', 0)
        cb_3x3 = results[layer_key]['cb'].get('3x3', {}).get('mean_f1', 0)
        print(f"Layer {layer+1:15} {ca_1x1:>10.4f} {ca_3x3:>10.4f} {cb_1x1:>10.4f} {cb_3x3:>10.4f}")
    
    baseline_ca_1x1 = results['baseline']['ca'].get('1x1', {}).get('mean_f1', 0)
    baseline_ca_3x3 = results['baseline']['ca'].get('3x3', {}).get('mean_f1', 0)
    baseline_cb_1x1 = results['baseline']['cb'].get('1x1', {}).get('mean_f1', 0)
    baseline_cb_3x3 = results['baseline']['cb'].get('3x3', {}).get('mean_f1', 0)
    print(f"{'Baseline':20} {baseline_ca_1x1:>10.4f} {baseline_ca_3x3:>10.4f} "
          f"{baseline_cb_1x1:>10.4f} {baseline_cb_3x3:>10.4f}")


if __name__ == '__main__':
    main()
