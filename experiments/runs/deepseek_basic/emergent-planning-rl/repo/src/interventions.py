"""
Intervention Module for Steering Agent Behavior.

Implements the intervention procedures described in Section 6 and Appendix B:
- Adding concept vectors w_k to cell state positions
- Agent-Shortcut interventions (Algorithms 1)
- Box-Shortcut interventions (Algorithm 2)
- Cutoff level interventions (Appendix B.3)

Key insight (Section 6.1): g'_{x,y} = g_{x,y} + w_k
where w_k is the probe vector for class k of concept C, added at position (x,y).
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from src.probes import ConceptClasses, get_probe_vectors


class InterventionManager:
    """
    Manages interventions on DRC agent cell states.
    
    Stores probe vectors for different concepts and applies interventions
    by adding them to specified positions in the cell state.
    
    Args:
        ca_vectors: Dict mapping class_idx -> vector for C_A (Agent Approach Direction)
        cb_vectors: Dict mapping class_idx -> vector for C_B (Box Push Direction)
        device: Torch device
    """
    def __init__(
        self,
        ca_vectors: Dict[int, np.ndarray],
        cb_vectors: Dict[int, np.ndarray],
        device: str = 'cpu',
    ):
        self.ca_vectors = {k: torch.from_numpy(v).to(device) for k, v in ca_vectors.items()}
        self.cb_vectors = {k: torch.from_numpy(v).to(device) for k, v in cb_vectors.items()}
        self.device = device
        self.vector_dim = len(ca_vectors[0])
    
    def add_concept_vector(
        self,
        cell_state: torch.Tensor,
        position: Tuple[int, int],
        concept_type: str,  # 'CA' or 'CB'
        class_idx: int,
        alpha: float = 1.0,
    ):
        """
        Add a concept vector to a specific position in the cell state.
        
        Args:
            cell_state: (B, C, H, W) cell state tensor
            position: (y, x) position to intervene on
            concept_type: 'CA' or 'CB'
            class_idx: class index (0=NEVER, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT)
            alpha: intervention strength scaling factor
        """
        vectors = self.ca_vectors if concept_type == 'CA' else self.cb_vectors
        w = vectors[class_idx]  # (C,)
        y, x = position
        cell_state[:, :, y, x] += alpha * w.unsqueeze(0)
    
    def add_concept_vectors_batch(
        self,
        cell_state: torch.Tensor,
        positions: List[Tuple[int, int]],
        concept_type: str,
        class_idx: int,
        alpha: float = 1.0,
    ):
        """Add concept vector to multiple positions."""
        for pos in positions:
            self.add_concept_vector(cell_state, pos, concept_type, class_idx, alpha)


class AgentShortcutIntervention:
    """
    Agent-Shortcut intervention as described in Section 6.1 and Algorithm 1.
    
    In Agent-Shortcut levels, the agent can follow either a short or long path
    to reach a region with boxes and targets. The intervention:
    1. Adds NEVER vector for C_A to squares on the short route (repeated every step)
    2. Adds directional vectors along the long route (repeated until agent steps onto first square)
    
    Args:
        short_route_squares: List of (y, x) positions on the short route
        long_route_squares_dirs: List of ((y, x), DIR) for first p squares of long route
        first_long_square: (y, x) of the first square on the long route
        alpha: intervention strength
        p: number of squares to intervene on for directional intervention
    """
    def __init__(
        self,
        short_route_squares: List[Tuple[int, int]],
        long_route_squares_dirs: List[Tuple[Tuple[int, int], int]],
        first_long_square: Tuple[int, int],
        alpha: float = 1.0,
        p: int = 1,
    ):
        self.short_route_squares = short_route_squares
        self.long_route_squares_dirs = long_route_squares_dirs[:p]
        self.first_long_square = first_long_square
        self.alpha = alpha
        self.directional_active = True  # active until agent steps onto first_long_square
    
    def apply(
        self,
        manager: InterventionManager,
        cell_state: torch.Tensor,
        agent_pos: Tuple[int, int],
    ):
        """
        Apply intervention to cell state.
        
        Args:
            manager: InterventionManager with probe vectors
            cell_state: Current cell state tensor (1, C, H, W)
            agent_pos: Current agent position (y, x)
        """
        # Short-route intervention: add NEVER to short route squares
        manager.add_concept_vectors_batch(
            cell_state, self.short_route_squares, 'CA', 
            ConceptClasses.NEVER, self.alpha
        )
        
        # Check if directional intervention should still be active
        if self.directional_active:
            if agent_pos == self.first_long_square:
                self.directional_active = False
            else:
                # Directional intervention: add directional vectors along long route
                for (y, x), direction in self.long_route_squares_dirs:
                    manager.add_concept_vector(
                        cell_state, (y, x), 'CA', direction, self.alpha
                    )


class BoxShortcutIntervention:
    """
    Box-Shortcut intervention as described in Section 6.1 and Algorithm 2.
    
    In Box-Shortcut levels, one box can be pushed either a short or long route.
    The intervention:
    1. Adds NEVER vector for C_B to squares on the short route (repeated every step)
    2. Adds directional vectors for C_B along the long route (repeated until box is pushed off initial square)
    
    Args:
        short_route_squares: List of (y, x) positions on the short push route
        long_route_squares_dirs: List of ((y, x), DIR) for first p squares of long push route
        box_initial_position: (y, x) of the box to intervene on
        alpha: intervention strength
        p: number of squares to intervene on for directional intervention
    """
    def __init__(
        self,
        short_route_squares: List[Tuple[int, int]],
        long_route_squares_dirs: List[Tuple[Tuple[int, int], int]],
        box_initial_position: Tuple[int, int],
        alpha: float = 1.0,
        p: int = 1,
    ):
        self.short_route_squares = short_route_squares
        self.long_route_squares_dirs = long_route_squares_dirs[:p]
        self.box_initial_position = box_initial_position
        self.alpha = alpha
        self.directional_active = True
    
    def apply(
        self,
        manager: InterventionManager,
        cell_state: torch.Tensor,
        box_positions: Set[Tuple[int, int]],
    ):
        """
        Apply intervention to cell state.
        
        Args:
            manager: InterventionManager with probe vectors
            cell_state: Current cell state tensor (1, C, H, W)
            box_positions: Set of current box positions (y, x)
        """
        # Short-route intervention: add NEVER to short route squares
        manager.add_concept_vectors_batch(
            cell_state, self.short_route_squares, 'CB',
            ConceptClasses.NEVER, self.alpha
        )
        
        # Check if directional intervention should still be active
        if self.directional_active:
            if self.box_initial_position not in box_positions:
                self.directional_active = False
            else:
                # Directional intervention
                for (y, x), direction in self.long_route_squares_dirs:
                    manager.add_concept_vector(
                        cell_state, (y, x), 'CB', direction, self.alpha
                    )


class CutoffIntervention:
    """
    Cutoff level interventions as described in Appendix B.3.
    
    Cutoff levels have a corridor with a target at the entrance and a box+target at the end.
    The agent must NOT myopically push the box at the entrance onto the target.
    
    Three types:
    - Agent-Only: Add directional C_A to target at corridor entrance
    - Box-Only: Add directional C_B to box at corridor entrance
    - Agent-and-Box: Both of the above
    
    Args:
        intervention_type: 'agent_only', 'box_only', or 'agent_and_box'
        target_position: (y, x) of the target at corridor entrance
        target_direction: class index for direction agent should step onto target from
        box_position: (y, x) of the box at corridor entrance
        box_direction: class index for direction box should be pushed
        alpha: intervention strength
    """
    def __init__(
        self,
        intervention_type: str,
        target_position: Tuple[int, int],
        target_direction: int,
        box_position: Tuple[int, int],
        box_direction: int,
        alpha: float = 1.0,
    ):
        self.intervention_type = intervention_type
        self.target_position = target_position
        self.target_direction = target_direction
        self.box_position = box_position
        self.box_direction = box_direction
        self.alpha = alpha
        self.active = True
    
    def apply(
        self,
        manager: InterventionManager,
        cell_state: torch.Tensor,
        box_positions: Set[Tuple[int, int]],
    ):
        """
        Apply intervention. Active until box at entrance is moved.
        """
        if not self.active:
            return
        
        if self.intervention_type in ('agent_only', 'agent_and_box'):
            manager.add_concept_vector(
                cell_state, self.target_position, 'CA',
                self.target_direction, self.alpha
            )
        
        if self.intervention_type in ('box_only', 'agent_and_box'):
            manager.add_concept_vector(
                cell_state, self.box_position, 'CB',
                self.box_direction, self.alpha
            )
        
        # Deactivate when box is moved
        if self.box_position not in box_positions:
            self.active = False


def run_intervened_episode(
    agent,
    env,
    level: np.ndarray,
    intervention_obj,
    intervention_manager: InterventionManager,
    target_layer: int = 2,
    thinking_steps: int = 0,
    greedy: bool = True,
    max_steps: int = 120,
) -> Dict:
    """
    Run an episode with interventions applied at each step.
    
    Args:
        agent: DRCAgent instance
        env: SokobanEnv instance
        level: Level array
        intervention_obj: One of AgentShortcutIntervention, BoxShortcutIntervention, CutoffIntervention
        intervention_manager: InterventionManager
        target_layer: Which ConvLSTM layer to intervene on (0-indexed)
        thinking_steps: Number of thinking steps
        greedy: Whether to act greedily
        max_steps: Maximum steps per episode
    
    Returns:
        dict with episode results
    """
    import torch
    
    obs = env.reset(level)
    agent.reset_state(batch_size=1, device=next(agent.parameters()).device)
    
    obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)
    obs_tensor = obs_tensor.to(next(agent.parameters()).device)
    
    actions = []
    rewards = []
    solved = False
    
    # Thinking steps: apply interventions but don't move
    for _ in range(thinking_steps):
        with torch.no_grad():
            # Forward pass through all layers
            logits, value = agent.forward(obs_tensor)
        
        # Apply intervention to target layer
        intervention_obj.apply(
            intervention_manager,
            agent.c_t[target_layer],
            agent_pos=env.state.agent_pos,
            box_positions=set(zip(*np.where(env.state.boxes))),
        )
    
    for step in range(max_steps):
        # Get box positions before step
        box_positions = set(zip(*np.where(env.state.boxes)))
        agent_pos = env.state.agent_pos
        
        with torch.no_grad():
            logits, value = agent.forward(obs_tensor)
        
        # Apply intervention after forward pass
        if hasattr(intervention_obj, 'apply'):
            intervention_obj.apply(
                intervention_manager,
                agent.c_t[target_layer],
                agent_pos=agent_pos,
                box_positions=box_positions,
            )
        
        if greedy:
            action = logits.argmax(dim=-1).item()
        else:
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        
        obs, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        
        if done:
            solved = env.is_solved()
            break
        
        obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)
        obs_tensor = obs_tensor.to(next(agent.parameters()).device)
    
    return {
        'solved': solved,
        'steps': len(actions),
        'actions': actions,
        'rewards': rewards,
    }
