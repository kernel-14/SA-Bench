import torch
"""
Concept Labeling for Sokoban Episodes.

Generates labels for the planning-relevant concepts C_A and C_B
by processing episode trajectories.

C_A (Agent Approach Direction):
    For each square (x,y), at each time t, what is the direction from which
    the agent will NEXT step onto that square (if ever)?
    Classes: NEVER, UP, DOWN, LEFT, RIGHT

C_B (Box Push Direction):
    For each square (x,y), at each time t, what is the direction in which
    the NEXT box pushed off that square will be pushed (if ever)?
    Classes: NEVER, UP, DOWN, LEFT, RIGHT

Additional concepts (Appendix D.4):
- Agent Approach (binary: NEVER, AGAIN)
- Box Push (binary: NEVER, AGAIN)
- Agent Exit Direction (direction agent leaves a square)
- Box Approach Direction (direction box is pushed onto a square)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from src.sokoban import SquareType, ACTION_DELTAS, SokobanEnv


def compute_concept_labels_ca(
    actions: List[int],
    agent_positions: List[Tuple[int, int]],
    board_size: int = 8,
) -> np.ndarray:
    """
    Compute C_A (Agent Approach Direction) labels for each time step.
    
    For each square and time t, the label is the direction from which the agent
    will NEXT step onto that square. If the agent never steps onto it again, label is NEVER.
    
    Args:
        actions: List of actions taken (1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 0=NOOP)
        agent_positions: List of (y, x) agent positions at each step
        board_size: Size of the board (default 8)
    
    Returns:
        labels: (T, board_size, board_size) integer array of concept classes
    """
    T = len(actions)
    labels = np.full((T, 8, 8), 0, dtype=np.int32)  # Default: NEVER = 0
    
    # For each square, find when the agent steps onto it
    for y in range(board_size):
        for x in range(board_size):
            # Find all times the agent steps onto (y, x)
            entry_times = []
            entry_dirs = []
            
            for t in range(T):
                if t > 0:
                    prev_pos = agent_positions[t - 1]
                    curr_pos = agent_positions[t]
                    if curr_pos == (y, x) and prev_pos != (y, x):
                        # Agent stepped onto (y, x) at time t
                        # The approach direction is the direction FROM prev_pos TO curr_pos
                        dy = curr_pos[0] - prev_pos[0]
                        dx = curr_pos[1] - prev_pos[1]
                        if dy == -1 and dx == 0:
                            d = 1  # UP (agent came from below, moved UP)
                        elif dy == 1 and dx == 0:
                            d = 2  # DOWN
                        elif dy == 0 and dx == -1:
                            d = 3  # LEFT
                        elif dy == 0 and dx == 1:
                            d = 4  # RIGHT
                        else:
                            d = 0
                        entry_times.append(t)
                        entry_dirs.append(d)
            
            if entry_times:
                # For each time t, find the NEXT entry time and its direction
                for t in range(T):
                    # Find first entry_time >= t
                    next_entry = None
                    next_dir = None
                    for et, ed in zip(entry_times, entry_dirs):
                        if et >= t:
                            next_entry = et
                            next_dir = ed
                            break
                    
                    if next_entry is not None:
                        labels[t, y, x] = next_dir
                    else:
                        labels[t, y, x] = 0  # NEVER
            else:
                labels[:, y, x] = 0  # NEVER
    
    return labels


def compute_concept_labels_cb(
    actions: List[int],
    agent_positions: List[Tuple[int, int]],
    box_movements: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]],
    board_size: int = 8,
) -> np.ndarray:
    """
    Compute C_B (Box Push Direction) labels for each time step.
    
    For each square and time t, the label is the direction in which the NEXT box
    pushed off that square will be pushed. If no box is ever pushed off it again, NEVER.
    
    Args:
        actions: List of actions taken
        agent_positions: List of (y, x) agent positions at each step
        box_movements: For each step, list of ((from_y, from_x), (to_y, to_x)) box movements
        board_size: Size of the board (default 8)
    
    Returns:
        labels: (T, board_size, board_size) integer array of concept classes
    """
    T = len(actions)
    labels = np.full((T, 8, 8), 0, dtype=np.int32)  # Default: NEVER
    
    # Collect all box push events
    push_times = []    # time t when push occurs
    push_from = []     # (y, x) square box is pushed from
    push_dir = []      # direction of push
    
    for t, movements in enumerate(box_movements):
        for (fy, fx), (ty, tx) in movements:
            dy = ty - fy
            dx = tx - fx
            if dy == -1 and dx == 0:
                d = 1  # UP
            elif dy == 1 and dx == 0:
                d = 2  # DOWN
            elif dy == 0 and dx == -1:
                d = 3  # LEFT
            elif dy == 0 and dx == 1:
                d = 4  # RIGHT
            else:
                d = 0
            
            push_times.append(t)
            push_from.append((fy, fx))
            push_dir.append(d)
    
    # For each square, find the next push
    for y in range(board_size):
        for x in range(board_size):
            # Filter pushes from this square
            square_pushes = []
            for pt, pf, pd in zip(push_times, push_from, push_dir):
                if pf == (y, x):
                    square_pushes.append((pt, pd))
            
            if square_pushes:
                for t in range(T):
                    # Find first push from (y,x) at time >= t
                    next_pd = None
                    for pt, pd in square_pushes:
                        if pt >= t:
                            next_pd = pd
                            break
                    
                    if next_pd is not None:
                        labels[t, y, x] = next_pd
                    else:
                        labels[t, y, x] = 0  # NEVER
            else:
                labels[:, y, x] = 0  # NEVER
    
    return labels


def compute_concept_labels_ca_binary(
    actions: List[int],
    agent_positions: List[Tuple[int, int]],
    board_size: int = 8,
) -> np.ndarray:
    """
    Compute binary 'Agent Approach' labels.
    Maps squares to {NEVER (0), AGAIN (1)}.
    """
    ca_labels = compute_concept_labels_ca(actions, agent_positions, board_size)
    # Convert: 0 stays NEVER, any direction becomes AGAIN (1)
    binary = (ca_labels > 0).astype(np.int32)
    return binary


def compute_concept_labels_cb_binary(
    actions: List[int],
    agent_positions: List[Tuple[int, int]],
    box_movements: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]],
    board_size: int = 8,
) -> np.ndarray:
    """
    Compute binary 'Box Push' labels.
    Maps squares to {NEVER (0), AGAIN (1)}.
    """
    cb_labels = compute_concept_labels_cb(actions, agent_positions, box_movements, board_size)
    binary = (cb_labels > 0).astype(np.int32)
    return binary


def record_episode_labels(
    env: SokobanEnv,
    agent,
    level: np.ndarray,
    thinking_steps: int = 0,
    greedy: bool = True,
    record_levels: Optional[List[int]] = None,
) -> Dict:
    """
    Run an episode and record all concept labels along with cell state activations.
    
    This is the core data collection function for probe training.
    
    Args:
        env: SokobanEnv instance
        agent: DRCAgent instance
        level: Level array
        thinking_steps: Number of thinking steps at start
        greedy: Whether to act greedily
        record_levels: Which layers to record cell states from (e.g., [0, 1, 2])
    
    Returns:
        dict with:
        - cell_states: dict mapping layer -> list of (H, W, C) arrays
        - ca_labels: (T, H, W) C_A labels
        - cb_labels: (T, H, W) C_B labels
        - actions: list of actions
        - observations: list of observations
    """
    if record_levels is None:
        record_levels = [0, 1, 2]
    
    obs = env.reset(level)
    agent.reset_state(batch_size=1, device=next(agent.parameters()).device)
    
    # Run thinking steps
    obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)
    obs_tensor = obs_tensor.to(next(agent.parameters()).device)
    
    for _ in range(thinking_steps):
        with torch.no_grad():
            agent.forward(obs_tensor)
    
    # Record full episode
    cell_states_by_layer = {d: [] for d in record_levels}
    actions = []
    rewards = []
    agent_positions = []
    box_movements_per_step = []  # list of [(from_pos, to_pos)]
    observations = [obs]
    
    # Initial positions
    ay, ax = env.state.agent_pos
    agent_positions.append((ay, ax))
    box_movements_per_step.append([])
    
    # Record initial cell states
    for d in record_levels:
        cs = agent.c_t[d].clone().cpu().squeeze(0).permute(1, 2, 0).numpy()
        cell_states_by_layer[d].append(cs)
    
    for step in range(120):
        if greedy:
            with torch.no_grad():
                logits, _ = agent.forward(obs_tensor)
            action = logits.argmax(dim=-1).item()
        else:
            action = agent.act_sample(obs_tensor)
        
        # Record box positions before step
        boxes_before = env.state.boxes.copy()
        
        obs, reward, done, _ = env.step(action)
        
        # Record box movements
        boxes_after = env.state.boxes
        box_moves = []
        for y in range(8):
            for x in range(8):
                if boxes_before[y, x] and not boxes_after[y, x]:
                    # Box left this square - find where it went
                    # It must have moved in the direction of the action
                    if action >= 1 and action <= 4:
                        dy, dx = ACTION_DELTAS[action]
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < 8 and 0 <= nx < 8 and boxes_after[ny, nx]:
                            box_moves.append(((y, x), (ny, nx)))
        
        actions.append(action)
        rewards.append(reward)
        observations.append(obs)
        ay, ax = env.state.agent_pos
        agent_positions.append((ay, ax))
        box_movements_per_step.append(box_moves)
        
        # Record cell states
        for d in record_levels:
            cs = agent.c_t[d].clone().cpu().squeeze(0).permute(1, 2, 0).numpy()
            cell_states_by_layer[d].append(cs)
        
        if done:
            break
        
        obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)
        obs_tensor = obs_tensor.to(next(agent.parameters()).device)
    
    # Compute concept labels
    # For each time step including initial state
    T = len(actions)
    
    ca_labels = compute_concept_labels_ca(actions, agent_positions)
    cb_labels = compute_concept_labels_cb(actions, agent_positions, box_movements_per_step)
    
    # Trim cell states to match labels
    T_cs = len(cell_states_by_layer[record_levels[0]])
    for d in record_levels:
        cell_states_by_layer[d] = cell_states_by_layer[d][:T]
    
    return {
        'cell_states': cell_states_by_layer,
        'ca_labels': ca_labels,
        'cb_labels': cb_labels,
        'actions': actions,
        'rewards': rewards,
        'agent_positions': agent_positions,
        'observations': observations,
        'solved': env.is_solved(),
    }
