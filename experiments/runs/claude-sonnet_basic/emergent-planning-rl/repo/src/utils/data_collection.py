"""
Data collection utilities for generating probe training/test datasets.

From the paper:
- Training dataset: 3000 episodes from Boxoban unfiltered training set
- Test dataset: 1000 episodes from Boxoban unfiltered validation set
- For checkpoint experiments: 1000 and 500 episodes respectively

Each episode generates multiple transitions, each with:
- Cell state activations at each layer (after final tick)
- Concept labels (C_A and C_B) for each grid square
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.sokoban import SokobanEnv, WALL, EMPTY, BOX, AGENT, BOX_ON_TARGET, AGENT_ON_TARGET, TARGET
from probing.concepts import (
    ConceptClass, extract_concepts_from_episode,
    compute_agent_approach_direction, compute_box_push_direction
)


GRID_SIZE = 8


def get_box_positions(grid: np.ndarray) -> List[Tuple[int, int]]:
    """Get positions of all boxes in the grid."""
    positions = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j] in (BOX, BOX_ON_TARGET):
                positions.append((i, j))
    return sorted(positions)  # Sort for consistent ordering


def collect_episode_data(
    agent,
    env: SokobanEnv,
    level: np.ndarray,
    device: torch.device,
    thinking_steps: int = 0,
) -> Optional[Dict]:
    """
    Collect data from a single episode.
    
    Args:
        agent: DRC agent
        env: Sokoban environment
        level: Level to play
        device: Device for agent
        thinking_steps: Number of forced stationary steps at start
        
    Returns:
        Dict with episode data, or None if episode failed
    """
    obs = env.reset(level)
    hidden_states = agent.init_hidden(batch_size=1, device=device)
    
    # Track trajectories
    agent_positions = []
    box_positions_per_step = []
    cell_states_per_step = []  # List of [layer1_cell, layer2_cell, layer3_cell]
    observations = []
    actions = []
    
    # Record initial state
    grid = env.get_grid()
    agent_pos = env.get_agent_pos()
    box_positions = get_box_positions(grid)
    
    agent_positions.append(agent_pos)
    box_positions_per_step.append(box_positions)
    observations.append(obs.copy())
    
    # Perform thinking steps (forced stationary)
    for _ in range(thinking_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, hidden_states, cell_states_tick = agent.forward(
                obs_tensor, hidden_states, return_cell_states=True
            )
        # Store cell states from final tick
        if cell_states_tick:
            final_tick_cells = cell_states_tick[-1]  # Last tick
            cell_states_per_step.append([c.squeeze(0).cpu().numpy() for c in final_tick_cells])
        
        # Noop action during thinking
        obs, _, done, _ = env.step(0)  # noop
        if done:
            break
    
    # Play episode
    done = False
    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, _, hidden_states, cell_states_tick = agent.forward(
                obs_tensor, hidden_states, return_cell_states=True
            )
        
        # Store cell states from final tick
        if cell_states_tick:
            final_tick_cells = cell_states_tick[-1]  # Last tick
            cell_states_per_step.append([c.squeeze(0).cpu().numpy() for c in final_tick_cells])
        
        # Select action (greedy)
        action = logits.argmax(dim=-1).item()
        actions.append(action)
        
        obs, reward, done, info = env.step(action)
        
        # Record new state
        grid = env.get_grid()
        agent_pos = env.get_agent_pos()
        box_positions = get_box_positions(grid)
        
        agent_positions.append(agent_pos)
        box_positions_per_step.append(box_positions)
        observations.append(obs.copy())
    
    # Compute concept labels for each step
    ca_labels, cb_labels = extract_concepts_from_episode(
        agent_positions, box_positions_per_step
    )
    
    return {
        'agent_positions': agent_positions,
        'box_positions_per_step': box_positions_per_step,
        'cell_states_per_step': cell_states_per_step,
        'observations': observations,
        'actions': actions,
        'ca_labels': ca_labels,
        'cb_labels': cb_labels,
        'solved': info.get('solved', False),
    }


def collect_dataset(
    agent,
    levels: List[np.ndarray],
    num_episodes: int,
    device: torch.device,
    thinking_steps: int = 0,
    verbose: bool = False,
) -> Dict:
    """
    Collect a dataset of episodes for probe training/evaluation.
    
    Args:
        agent: DRC agent
        levels: List of Sokoban levels
        num_episodes: Number of episodes to collect
        device: Device for agent
        thinking_steps: Number of thinking steps
        verbose: Whether to print progress
        
    Returns:
        Dict with collected data
    """
    env = SokobanEnv()
    
    all_cell_states = {layer: [] for layer in range(agent.D)}
    all_ca_labels = []
    all_cb_labels = []
    all_observations = []
    
    episodes_collected = 0
    level_idx = 0
    
    while episodes_collected < num_episodes and level_idx < len(levels):
        level = levels[level_idx % len(levels)]
        level_idx += 1
        
        episode_data = collect_episode_data(
            agent, env, level, device, thinking_steps=thinking_steps
        )
        
        if episode_data is None:
            continue
        
        # Extract cell states and labels for each step
        cell_states = episode_data['cell_states_per_step']
        ca_labels = episode_data['ca_labels']
        cb_labels = episode_data['cb_labels']
        observations = episode_data['observations']
        
        # Align: cell states are collected after each step
        # We want cell states at step t to predict concepts at step t
        min_len = min(len(cell_states), len(ca_labels), len(cb_labels))
        
        for t in range(min_len):
            if t < len(cell_states):
                for layer in range(agent.D):
                    if layer < len(cell_states[t]):
                        all_cell_states[layer].append(cell_states[t][layer])
                
                if t < len(ca_labels):
                    all_ca_labels.append(ca_labels[t])
                if t < len(cb_labels):
                    all_cb_labels.append(cb_labels[t])
                if t < len(observations):
                    all_observations.append(observations[t])
        
        episodes_collected += 1
        
        if verbose and episodes_collected % 100 == 0:
            print(f"Collected {episodes_collected}/{num_episodes} episodes")
    
    return {
        'cell_states': all_cell_states,
        'ca_labels': all_ca_labels,
        'cb_labels': all_cb_labels,
        'observations': all_observations,
        'num_episodes': episodes_collected,
    }


def collect_thinking_steps_data(
    agent,
    levels: List[np.ndarray],
    num_episodes: int,
    num_thinking_steps: int,
    device: torch.device,
    verbose: bool = False,
) -> Dict:
    """
    Collect data during thinking steps to analyze plan refinement.
    
    For each episode, forces the agent to remain stationary for
    num_thinking_steps steps and records cell states at each tick.
    
    Args:
        agent: DRC agent
        levels: List of Sokoban levels
        num_episodes: Number of episodes
        num_thinking_steps: Number of thinking steps
        device: Device for agent
        verbose: Whether to print progress
        
    Returns:
        Dict with cell states at each tick during thinking steps
    """
    env = SokobanEnv()
    
    # cell_states_per_tick[tick][layer] = list of cell states
    num_ticks = num_thinking_steps * agent.N
    cell_states_per_tick = {
        tick: {layer: [] for layer in range(agent.D)}
        for tick in range(num_ticks)
    }
    
    # Also collect concept labels (based on actual future behavior)
    ca_labels_per_episode = []
    cb_labels_per_episode = []
    
    episodes_collected = 0
    level_idx = 0
    
    while episodes_collected < num_episodes and level_idx < len(levels):
        level = levels[level_idx % len(levels)]
        level_idx += 1
        
        obs = env.reset(level)
        hidden_states = agent.init_hidden(batch_size=1, device=device)
        
        # Collect cell states during thinking steps
        tick_idx = 0
        for step in range(num_thinking_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Get cell states at each tick
            B = 1
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
                    
                    # Store cell states at this tick
                    for layer in range(agent.D):
                        cell_state = current_states[layer][1].squeeze(0).cpu().numpy()
                        cell_states_per_tick[tick_idx][layer].append(cell_state)
                    
                    tick_idx += 1
            
            hidden_states = current_states
            
            # Noop action during thinking
            obs, _, done, _ = env.step(0)
            if done:
                break
        
        # Now play the episode to get actual future behavior
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
        
        # Compute concept labels based on actual behavior
        ca_labels, cb_labels = extract_concepts_from_episode(
            agent_positions, box_positions_per_step
        )
        
        if ca_labels:
            ca_labels_per_episode.append(ca_labels[0])  # Labels at start of episode
        if cb_labels:
            cb_labels_per_episode.append(cb_labels[0])
        
        episodes_collected += 1
        
        if verbose and episodes_collected % 100 == 0:
            print(f"Collected {episodes_collected}/{num_episodes} episodes")
    
    return {
        'cell_states_per_tick': cell_states_per_tick,
        'ca_labels': ca_labels_per_episode,
        'cb_labels': cb_labels_per_episode,
        'num_episodes': episodes_collected,
        'num_ticks': num_ticks,
    }
