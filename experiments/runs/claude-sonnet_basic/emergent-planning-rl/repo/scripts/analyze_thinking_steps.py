"""
Script to analyze plan refinement during thinking steps.

From the paper (Section 5, Figure 6):
- Force agent to remain stationary for 5 'thinking steps'
- This gives 15 internal ticks of extra test-time compute
- Measure macro F1 of probes at each tick
- Shows that plans iteratively improve with more compute

Also measures:
- Percentage of extra levels solved with thinking steps (Figure 9)
- Co-emergence of concept representations and planning-like behavior

Usage:
    python scripts/analyze_thinking_steps.py --agent_path /path/to/agent.pt 
        --probe_dir /path/to/probes --data_dir /path/to/boxoban
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
from probing.probe_trainer import ConceptDataset, evaluate_probe
from utils.data_collection import collect_thinking_steps_data, get_box_positions
from probing.concepts import extract_concepts_from_episode


def evaluate_thinking_steps_benefit(
    agent: DRCAgent,
    levels: List[np.ndarray],
    device: torch.device,
    num_episodes: int = 1000,
    max_thinking_steps: int = 5,
) -> Dict[int, float]:
    """
    Measure how many additional levels the agent solves with thinking steps.
    
    Args:
        agent: DRC agent
        levels: List of levels to evaluate on
        device: Device
        num_episodes: Number of episodes
        max_thinking_steps: Maximum thinking steps to test
        
    Returns:
        Dict mapping thinking_steps -> solve_rate
    """
    env = SokobanEnv()
    
    # First, find which levels the agent can solve without thinking steps
    base_solved = set()
    
    for i in range(min(num_episodes, len(levels))):
        level = levels[i]
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
    
    print(f"Base solve rate: {len(base_solved)/min(num_episodes, len(levels)):.3f}")
    
    # Now test with different numbers of thinking steps
    results = {0: len(base_solved) / min(num_episodes, len(levels))}
    
    for thinking_steps in range(1, max_thinking_steps + 1):
        solved_with_thinking = set()
        
        for i in range(min(num_episodes, len(levels))):
            level = levels[i]
            obs = env.reset(level)
            hidden_states = agent.init_hidden(batch_size=1, device=device)
            
            # Thinking steps
            for _ in range(thinking_steps):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, _, hidden_states, _ = agent.forward(obs_tensor, hidden_states)
                obs, _, done, _ = env.step(0)  # noop
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
                solved_with_thinking.add(i)
        
        # Extra levels solved (not solved without thinking)
        extra_solved = solved_with_thinking - base_solved
        extra_pct = len(extra_solved) / min(num_episodes, len(levels)) * 100
        
        results[thinking_steps] = len(solved_with_thinking) / min(num_episodes, len(levels))
        print(f"Thinking steps {thinking_steps}: Solve rate = {results[thinking_steps]:.3f}, "
              f"Extra solved = {extra_pct:.1f}%")
    
    return results


def analyze_plan_refinement_during_thinking(
    agent: DRCAgent,
    probe_ca: LinearProbe,
    probe_cb: LinearProbe,
    levels: List[np.ndarray],
    device: torch.device,
    num_episodes: int = 1000,
    num_thinking_steps: int = 5,
    layer: int = 2,  # 0-indexed, so layer 3 = index 2
) -> Dict:
    """
    Analyze how probe F1 changes during thinking steps.
    
    From Figure 6 in the paper.
    
    Args:
        agent: DRC agent
        probe_ca: Trained probe for C_A
        probe_cb: Trained probe for C_B
        levels: List of levels
        device: Device
        num_episodes: Number of episodes
        num_thinking_steps: Number of thinking steps
        layer: Layer to analyze (0-indexed)
        
    Returns:
        Dict with F1 scores at each tick
    """
    env = SokobanEnv()
    num_ticks = num_thinking_steps * agent.N
    
    # Collect cell states at each tick and concept labels
    tick_ca_f1s = {tick: [] for tick in range(num_ticks)}
    tick_cb_f1s = {tick: [] for tick in range(num_ticks)}
    
    episodes_collected = 0
    level_idx = 0
    
    while episodes_collected < num_episodes and level_idx < len(levels):
        level = levels[level_idx % len(levels)]
        level_idx += 1
        
        obs = env.reset(level)
        hidden_states = agent.init_hidden(batch_size=1, device=device)
        
        # Collect cell states at each tick during thinking steps
        tick_cell_states = []
        
        for step in range(num_thinking_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Manual tick-by-tick forward pass
            x = obs_tensor.permute(0, 3, 1, 2).float()
            i_t = agent.encoder(x)
            
            current_states = list(hidden_states)
            top_down = torch.zeros_like(i_t)
            
            with torch.no_grad():
                for n in range(agent.N):
                    new_states = []
                    for d, cell in enumerate(agent.convlstm_cells):
                        h, c = current_states[d]
                        if d == 0:
                            cell_input = torch.cat([i_t, top_down], dim=1)
                        else:
                            cell_input = i_t
                        new_h, new_c = cell(cell_input, h, c)
                        new_states.append((new_h, new_c))
                    
                    current_states = new_states
                    top_down = current_states[-1][0]
                    
                    # Store cell state at this tick
                    cell_state = current_states[layer][1].squeeze(0).cpu().numpy()
                    tick_cell_states.append(cell_state)
            
            hidden_states = current_states
            obs, _, done, _ = env.step(0)  # noop
            if done:
                break
        
        # Play episode to get actual future behavior
        agent_positions = [env.get_agent_pos()]
        box_positions_per_step = [get_box_positions(env.get_grid())]
        
        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _, hidden_states, _ = agent.forward(obs_tensor, hidden_states)
            action = logits.argmax(dim=-1).item()
            obs, _, done, _ = env.step(action)
            agent_positions.append(env.get_agent_pos())
            box_positions_per_step.append(get_box_positions(env.get_grid()))
        
        # Compute concept labels
        ca_labels, cb_labels = extract_concepts_from_episode(
            agent_positions, box_positions_per_step
        )
        
        if not ca_labels:
            continue
        
        # Evaluate probes at each tick
        true_ca = ca_labels[0]  # Labels at start of episode
        true_cb = cb_labels[0]
        
        for tick_idx, cell_state in enumerate(tick_cell_states[:num_ticks]):
            cell_tensor = torch.FloatTensor(cell_state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                ca_pred = probe_ca.predict(cell_tensor).squeeze(0).cpu().numpy()
                cb_pred = probe_cb.predict(cell_tensor).squeeze(0).cpu().numpy()
            
            ca_f1 = compute_macro_f1(ca_pred.flatten(), true_ca.flatten())
            cb_f1 = compute_macro_f1(cb_pred.flatten(), true_cb.flatten())
            
            tick_ca_f1s[tick_idx].append(ca_f1)
            tick_cb_f1s[tick_idx].append(cb_f1)
        
        episodes_collected += 1
    
    # Average F1 across episodes
    avg_ca_f1 = {tick: np.mean(f1s) for tick, f1s in tick_ca_f1s.items() if f1s}
    avg_cb_f1 = {tick: np.mean(f1s) for tick, f1s in tick_cb_f1s.items() if f1s}
    
    return {
        'ca_f1_per_tick': avg_ca_f1,
        'cb_f1_per_tick': avg_cb_f1,
        'num_episodes': episodes_collected,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze thinking steps and plan refinement')
    parser.add_argument('--agent_path', type=str, required=True, help='Path to trained agent')
    parser.add_argument('--probe_dir', type=str, required=True, help='Path to trained probes')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Boxoban dataset')
    parser.add_argument('--output_dir', type=str, default='thinking_results', help='Output directory')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--num_thinking_steps', type=int, default=5, help='Number of thinking steps')
    parser.add_argument('--layer', type=int, default=3, help='Layer to analyze (1-indexed)')
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
    
    # Load probes
    layer_idx = args.layer - 1  # Convert to 0-indexed
    
    probe_ca = LinearProbe(hidden_channels=hidden_channels, num_classes=5, probe_size=1).to(device)
    probe_ca_path = os.path.join(args.probe_dir, f'probe_layer{args.layer}_ca_1x1_best.pt')
    probe_ca.load_state_dict(torch.load(probe_ca_path, map_location=device))
    probe_ca.eval()
    
    probe_cb = LinearProbe(hidden_channels=hidden_channels, num_classes=5, probe_size=1).to(device)
    probe_cb_path = os.path.join(args.probe_dir, f'probe_layer{args.layer}_cb_1x1_best.pt')
    probe_cb.load_state_dict(torch.load(probe_cb_path, map_location=device))
    probe_cb.eval()
    
    # Load levels
    loader = BoxobanLoader(args.data_dir)
    medium_levels = loader.load_levels('medium', max_levels=args.num_episodes)
    
    print(f"Loaded {len(medium_levels)} medium levels")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze thinking steps benefit
    print("\nAnalyzing thinking steps benefit...")
    thinking_results = evaluate_thinking_steps_benefit(
        agent, medium_levels, device, 
        num_episodes=args.num_episodes,
        max_thinking_steps=args.num_thinking_steps,
    )
    
    # Analyze plan refinement during thinking
    print("\nAnalyzing plan refinement during thinking steps...")
    refinement_results = analyze_plan_refinement_during_thinking(
        agent, probe_ca, probe_cb, medium_levels, device,
        num_episodes=args.num_episodes,
        num_thinking_steps=args.num_thinking_steps,
        layer=layer_idx,
    )
    
    # Save results
    results = {
        'thinking_steps_benefit': thinking_results,
        'plan_refinement': {
            'ca_f1_per_tick': refinement_results['ca_f1_per_tick'],
            'cb_f1_per_tick': refinement_results['cb_f1_per_tick'],
        },
    }
    
    results_path = os.path.join(args.output_dir, 'thinking_steps_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\n=== Plan Refinement During Thinking Steps ===")
    print(f"{'Tick':>6} {'C_A F1':>10} {'C_B F1':>10}")
    for tick in range(args.num_thinking_steps * N):
        ca_f1 = refinement_results['ca_f1_per_tick'].get(tick, 0)
        cb_f1 = refinement_results['cb_f1_per_tick'].get(tick, 0)
        print(f"{tick+1:>6} {ca_f1:>10.4f} {cb_f1:>10.4f}")


if __name__ == '__main__':
    main()
