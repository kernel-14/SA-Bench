"""
Intervention experiments for verifying causal role of concept representations.

From the paper (Section 6.1):
Interventions involve adding concept vectors learned by probes to the agent's
activations to force it to represent concepts in specific ways.

For a 1x1 probe with class vector w_k in R^32:
  g'_{x,y} <- g_{x,y} + alpha * w_k

Two types of intervention levels:
1. Agent-Shortcut levels: Agent can follow short or long path to boxes/targets
   - Intervene using C_A vectors to steer agent to take long path
   
2. Box-Shortcut levels: Box can be pushed short or long route to target
   - Intervene using C_B vectors to steer agent to push box long route

Each intervention consists of:
1. Short-route intervention: Add NEVER vector to positions on short route
2. Directional intervention: Add directional vector to first square(s) of long route

Success criterion: Agent solves level in the desired suboptimal way.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import copy

from probing.concepts import ConceptClass


GRID_SIZE = 8
NUM_CLASSES = 5


class AgentShortcutIntervention:
    """
    Intervention for Agent-Shortcut levels.
    
    In these levels, all boxes and targets are in one region, and the agent
    can follow either a long or short path to this region.
    
    Intervention:
    1. Add NEVER vector for C_A to positions on the short path
    2. Add directional vector for C_A to first p squares of long path
    
    From Algorithm 1 in the paper.
    """
    
    def __init__(
        self,
        short_route_squares: List[Tuple[int, int]],
        long_route_squares_dirs: List[Tuple[Tuple[int, int], int]],
        probe_ca,  # Trained 1x1 probe for C_A
        layer: int = 2,  # Layer to intervene on (0-indexed)
        alpha: float = 1.0,  # Intervention strength
        p: int = 1,  # Number of long route squares to intervene on
    ):
        """
        Args:
            short_route_squares: Positions on the short route
            long_route_squares_dirs: List of (position, direction_class) for long route
            probe_ca: Trained 1x1 probe for C_A
            layer: ConvLSTM layer to intervene on
            alpha: Intervention strength
            p: Number of long route squares to intervene on
        """
        self.short_route_squares = short_route_squares
        self.long_route_squares_dirs = long_route_squares_dirs[:p]
        self.probe_ca = probe_ca
        self.layer = layer
        self.alpha = alpha
        
        # Get class vectors from probe
        with torch.no_grad():
            class_vectors = probe_ca.get_class_vectors()  # (num_classes, hidden_channels)
            self.never_vector = class_vectors[ConceptClass.NEVER].cpu()
            self.dir_vectors = {
                ConceptClass.UP: class_vectors[ConceptClass.UP].cpu(),
                ConceptClass.DOWN: class_vectors[ConceptClass.DOWN].cpu(),
                ConceptClass.LEFT: class_vectors[ConceptClass.LEFT].cpu(),
                ConceptClass.RIGHT: class_vectors[ConceptClass.RIGHT].cpu(),
            }
        
        # Track whether agent has moved onto first long route square
        self.first_long_square = long_route_squares_dirs[0][0] if long_route_squares_dirs else None
        self.agent_moved_to_first = False
    
    def reset(self):
        """Reset intervention state for new episode."""
        self.agent_moved_to_first = False
    
    def apply(
        self,
        cell_states: List[Tuple[torch.Tensor, torch.Tensor]],
        agent_pos: Tuple[int, int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply intervention to cell states.
        
        Args:
            cell_states: List of (h, c) tuples for each layer
            agent_pos: Current agent position
            
        Returns:
            Modified cell states
        """
        # Check if agent has moved to first long route square
        if self.first_long_square and agent_pos == self.first_long_square:
            self.agent_moved_to_first = True
        
        # Clone cell states to avoid modifying originals
        new_cell_states = [(h.clone(), c.clone()) for h, c in cell_states]
        
        h, c = new_cell_states[self.layer]
        
        # Short-route intervention: add NEVER vector to short route positions
        for (r, col) in self.short_route_squares:
            c[:, :, r, col] = c[:, :, r, col] + self.alpha * self.never_vector.to(c.device).unsqueeze(0)
        
        # Directional intervention: add directional vector to long route positions
        if not self.agent_moved_to_first:
            for (r, col), dir_class in self.long_route_squares_dirs:
                if dir_class in self.dir_vectors:
                    c[:, :, r, col] = c[:, :, r, col] + self.alpha * self.dir_vectors[dir_class].to(c.device).unsqueeze(0)
        
        new_cell_states[self.layer] = (h, c)
        return new_cell_states


class BoxShortcutIntervention:
    """
    Intervention for Box-Shortcut levels.
    
    In these levels, three boxes are adjacent to targets and a fourth box
    can be pushed a long or short route to a target.
    
    Intervention:
    1. Add NEVER vector for C_B to positions on the short route
    2. Add directional vector for C_B to box's initial position (and p-1 more squares)
    
    From Algorithm 2 in the paper.
    """
    
    def __init__(
        self,
        short_route_squares: List[Tuple[int, int]],
        long_route_squares_dirs: List[Tuple[Tuple[int, int], int]],
        box_initial_pos: Tuple[int, int],
        probe_cb,  # Trained 1x1 probe for C_B
        layer: int = 2,  # Layer to intervene on (0-indexed)
        alpha: float = 1.0,  # Intervention strength
        p: int = 1,  # Number of long route squares to intervene on
    ):
        """
        Args:
            short_route_squares: Positions on the short route
            long_route_squares_dirs: List of (position, direction_class) for long route
            box_initial_pos: Initial position of the box to be pushed
            probe_cb: Trained 1x1 probe for C_B
            layer: ConvLSTM layer to intervene on
            alpha: Intervention strength
            p: Number of long route squares to intervene on
        """
        self.short_route_squares = short_route_squares
        self.long_route_squares_dirs = long_route_squares_dirs[:p]
        self.box_initial_pos = box_initial_pos
        self.probe_cb = probe_cb
        self.layer = layer
        self.alpha = alpha
        
        # Get class vectors from probe
        with torch.no_grad():
            class_vectors = probe_cb.get_class_vectors()  # (num_classes, hidden_channels)
            self.never_vector = class_vectors[ConceptClass.NEVER].cpu()
            self.dir_vectors = {
                ConceptClass.UP: class_vectors[ConceptClass.UP].cpu(),
                ConceptClass.DOWN: class_vectors[ConceptClass.DOWN].cpu(),
                ConceptClass.LEFT: class_vectors[ConceptClass.LEFT].cpu(),
                ConceptClass.RIGHT: class_vectors[ConceptClass.RIGHT].cpu(),
            }
        
        # Track whether box has been pushed off initial position
        self.box_pushed = False
    
    def reset(self):
        """Reset intervention state for new episode."""
        self.box_pushed = False
    
    def apply(
        self,
        cell_states: List[Tuple[torch.Tensor, torch.Tensor]],
        box_positions: List[Tuple[int, int]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply intervention to cell states.
        
        Args:
            cell_states: List of (h, c) tuples for each layer
            box_positions: Current box positions
            
        Returns:
            Modified cell states
        """
        # Check if box has been pushed off initial position
        if self.box_initial_pos not in box_positions:
            self.box_pushed = True
        
        # Clone cell states
        new_cell_states = [(h.clone(), c.clone()) for h, c in cell_states]
        
        h, c = new_cell_states[self.layer]
        
        # Short-route intervention: add NEVER vector to short route positions
        for (r, col) in self.short_route_squares:
            c[:, :, r, col] = c[:, :, r, col] + self.alpha * self.never_vector.to(c.device).unsqueeze(0)
        
        # Directional intervention: add directional vector to long route positions
        if not self.box_pushed:
            for (r, col), dir_class in self.long_route_squares_dirs:
                if dir_class in self.dir_vectors:
                    c[:, :, r, col] = c[:, :, r, col] + self.alpha * self.dir_vectors[dir_class].to(c.device).unsqueeze(0)
        
        new_cell_states[self.layer] = (h, c)
        return new_cell_states


def run_intervention_episode(
    agent,
    env,
    level: np.ndarray,
    intervention,
    device: torch.device,
    max_steps: int = 120,
    desired_outcome_fn=None,
) -> Dict:
    """
    Run an episode with an intervention applied.
    
    Args:
        agent: DRC agent
        env: Sokoban environment
        level: Level to play
        intervention: Intervention object (AgentShortcut or BoxShortcut)
        device: Device for agent
        max_steps: Maximum episode steps
        desired_outcome_fn: Function that checks if desired outcome was achieved
        
    Returns:
        Dict with episode results
    """
    obs = env.reset(level)
    hidden_states = agent.init_hidden(batch_size=1, device=device)
    intervention.reset()
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Forward pass
            logits, value, hidden_states, _ = agent.forward(obs_tensor, hidden_states)
        
        # Apply intervention to cell states
        grid = env.get_grid()
        agent_pos = env.get_agent_pos()
        
        from utils.data_collection import get_box_positions
        box_positions = get_box_positions(grid)
        
        if isinstance(intervention, AgentShortcutIntervention):
            hidden_states = intervention.apply(hidden_states, agent_pos)
        elif isinstance(intervention, BoxShortcutIntervention):
            hidden_states = intervention.apply(hidden_states, box_positions)
        
        # Select action
        action = logits.argmax(dim=-1).item()
        obs, reward, done, info = env.step(action)
        step += 1
    
    success = False
    if desired_outcome_fn is not None:
        success = desired_outcome_fn(env, info)
    elif info.get('solved', False):
        success = True
    
    return {
        'solved': info.get('solved', False),
        'success': success,
        'steps': step,
    }


def evaluate_interventions(
    agent,
    env,
    levels: List[np.ndarray],
    probe,
    intervention_type: str,  # 'agent_shortcut' or 'box_shortcut'
    intervention_configs: List[Dict],
    layer: int,
    alpha: float = 1.0,
    p: int = 1,
    device: Optional[torch.device] = None,
    num_seeds: int = 5,
    use_random_probe: bool = False,
) -> Dict:
    """
    Evaluate intervention success rates.
    
    Args:
        agent: DRC agent
        env: Sokoban environment
        levels: List of levels to test on
        probe: Trained probe (or None for random)
        intervention_type: Type of intervention
        intervention_configs: List of intervention configurations
        layer: Layer to intervene on
        alpha: Intervention strength
        p: Number of long route squares
        device: Device
        num_seeds: Number of probe seeds
        use_random_probe: Whether to use randomly initialized probe
        
    Returns:
        Dict with success rates
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    success_rates = []
    
    for seed in range(num_seeds):
        if use_random_probe:
            # Use randomly initialized probe
            from probing.linear_probe import LinearProbe
            torch.manual_seed(seed)
            test_probe = LinearProbe(hidden_channels=32, num_classes=5, probe_size=1)
            # Scale to match trained probe norm
            if probe is not None:
                with torch.no_grad():
                    trained_norm = probe.get_class_vectors().norm()
                    random_norm = test_probe.get_class_vectors().norm()
                    if random_norm > 0:
                        scale = trained_norm / random_norm
                        test_probe.conv.weight.data *= scale
        else:
            test_probe = probe
        
        successes = 0
        total = 0
        
        for level, config in zip(levels, intervention_configs):
            if intervention_type == 'agent_shortcut':
                intervention = AgentShortcutIntervention(
                    short_route_squares=config['short_route'],
                    long_route_squares_dirs=config['long_route_dirs'],
                    probe_ca=test_probe,
                    layer=layer,
                    alpha=alpha,
                    p=p,
                )
                desired_fn = config.get('desired_outcome_fn')
            else:  # box_shortcut
                intervention = BoxShortcutIntervention(
                    short_route_squares=config['short_route'],
                    long_route_squares_dirs=config['long_route_dirs'],
                    box_initial_pos=config['box_initial_pos'],
                    probe_cb=test_probe,
                    layer=layer,
                    alpha=alpha,
                    p=p,
                )
                desired_fn = config.get('desired_outcome_fn')
            
            result = run_intervention_episode(
                agent, env, level, intervention, device,
                desired_outcome_fn=desired_fn,
            )
            
            if result['success']:
                successes += 1
            total += 1
        
        if total > 0:
            success_rates.append(successes / total * 100)
    
    return {
        'mean_success_rate': np.mean(success_rates),
        'std_success_rate': np.std(success_rates),
        'success_rates_per_seed': success_rates,
    }
