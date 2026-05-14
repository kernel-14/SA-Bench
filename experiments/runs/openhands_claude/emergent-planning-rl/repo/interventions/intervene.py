import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass

from config import CONCEPT_CLASSES, DIRECTION_TO_DELTA, CELL_TYPES
from model.drc import DRCAgent
from probing.probes import LinearProbe1x1
from environment.sokoban import SokobanEnv


NEVER = CONCEPT_CLASSES["NEVER"]
UP = CONCEPT_CLASSES["UP"]
DOWN = CONCEPT_CLASSES["DOWN"]
LEFT = CONCEPT_CLASSES["LEFT"]
RIGHT = CONCEPT_CLASSES["RIGHT"]

BOX = CELL_TYPES["BOX"]
BOX_ON_TARGET = CELL_TYPES["BOX_ON_TARGET"]
AGENT = CELL_TYPES["AGENT"]
AGENT_ON_TARGET = CELL_TYPES["AGENT_ON_TARGET"]


@dataclass
class InterventionSpec:
    """Specifies a single intervention on the agent's cell state."""
    layer_idx: int
    positions: List[Tuple[int, int]]
    class_idx: int
    alpha: float = 1.0
    concept: str = "CA"


def apply_intervention(
    cell_states: List[torch.Tensor],
    probe: LinearProbe1x1,
    layer_idx: int,
    positions: List[Tuple[int, int]],
    class_idx: int,
    alpha: float = 1.0,
) -> List[torch.Tensor]:
    """
    Apply a concept vector intervention to the agent's cell state.
    
    Implements: g_{x,y} <- g_{x,y} + alpha * w_k
    where w_k is the weight vector for class k from the 1x1 probe.
    
    Args:
        cell_states: list of D tensors (B, C, H, W)
        probe: trained 1x1 linear probe
        layer_idx: which ConvLSTM layer to intervene on
        positions: list of (row, col) positions to intervene at
        class_idx: concept class index (e.g., NEVER=0, UP=1, ...)
        alpha: intervention strength scaling factor
    Returns:
        modified cell_states
    """
    class_vectors = probe.get_class_vectors()
    w_k = class_vectors[class_idx]

    new_cell_states = [c.clone() for c in cell_states]
    target = new_cell_states[layer_idx]

    for r, c in positions:
        target[:, :, r, c] = target[:, :, r, c] + alpha * w_k.unsqueeze(0)

    new_cell_states[layer_idx] = target
    return new_cell_states


def run_agent_shortcut_intervention(
    agent: DRCAgent,
    env: SokobanEnv,
    level: np.ndarray,
    probe_ca: LinearProbe1x1,
    layer_idx: int,
    short_route_positions: List[Tuple[int, int]],
    long_route_positions_dirs: List[Tuple[Tuple[int, int], int]],
    alpha: float = 1.0,
    max_steps: int = 120,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Agent-Shortcut intervention (Algorithm 1 from Appendix B.2).
    
    Intervenes on C_A representations to steer the agent to take the longer path.
    
    Short-route intervention: add NEVER vector to positions on short path (every step)
    Directional intervention: add directional vector to first p squares of long route
                              (until agent moves onto first square of long route)
    
    Args:
        short_route_positions: positions on the short path
        long_route_positions_dirs: list of ((row, col), direction_class) for long route
        alpha: intervention strength
    Returns:
        dict with success, trajectory info
    """
    obs = env.reset(level)
    h, c = agent.init_hidden(1, device)

    first_long_pos = long_route_positions_dirs[0][0] if long_route_positions_dirs else None
    agent_moved_to_first_long = False

    trajectory = []
    done = False
    step = 0

    while not done and step < max_steps:
        c = apply_intervention(
            c, probe_ca, layer_idx,
            short_route_positions, NEVER, alpha
        )

        if not agent_moved_to_first_long:
            for (pos, dir_class) in long_route_positions_dirs:
                c = apply_intervention(
                    c, probe_ca, layer_idx,
                    [pos], dir_class, alpha
                )

        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = agent.forward(obs_tensor, h, c)

        h = out["hidden_states"]
        c = out["cell_states"]

        action = out["policy_logits"].argmax(dim=-1).item()
        obs, reward, done, info = env.step(action)
        trajectory.append({"action": action, "reward": reward})
        step += 1

        if first_long_pos is not None:
            agent_pos = _get_agent_pos(obs)
            if agent_pos == first_long_pos:
                agent_moved_to_first_long = True

    return {
        "solved": info.get("solved", False),
        "step_count": step,
        "trajectory": trajectory,
    }


def run_box_shortcut_intervention(
    agent: DRCAgent,
    env: SokobanEnv,
    level: np.ndarray,
    probe_cb: LinearProbe1x1,
    layer_idx: int,
    short_route_positions: List[Tuple[int, int]],
    box_initial_pos: Tuple[int, int],
    long_route_positions_dirs: List[Tuple[Tuple[int, int], int]],
    alpha: float = 1.0,
    max_steps: int = 120,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Box-Shortcut intervention (Algorithm 2 from Appendix B.2).
    
    Intervenes on C_B representations to steer the agent to push the box
    along the longer route.
    
    Short-route intervention: add NEVER vector to positions on short route (every step)
    Directional intervention: add directional vector to box's initial position
                              (until agent pushes box off initial position)
    """
    obs = env.reset(level)
    h, c = agent.init_hidden(1, device)

    box_pushed_from_initial = False
    trajectory = []
    done = False
    step = 0

    while not done and step < max_steps:
        c = apply_intervention(
            c, probe_cb, layer_idx,
            short_route_positions, NEVER, alpha
        )

        if not box_pushed_from_initial:
            for (pos, dir_class) in long_route_positions_dirs:
                c = apply_intervention(
                    c, probe_cb, layer_idx,
                    [pos], dir_class, alpha
                )

        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = agent.forward(obs_tensor, h, c)

        h = out["hidden_states"]
        c = out["cell_states"]

        action = out["policy_logits"].argmax(dim=-1).item()
        prev_obs = obs.copy()
        obs, reward, done, info = env.step(action)
        trajectory.append({"action": action, "reward": reward})
        step += 1

        if not box_pushed_from_initial:
            if _box_pushed_from(prev_obs, obs, box_initial_pos):
                box_pushed_from_initial = True

    return {
        "solved": info.get("solved", False),
        "step_count": step,
        "trajectory": trajectory,
    }


def run_cutoff_intervention(
    agent: DRCAgent,
    env: SokobanEnv,
    level: np.ndarray,
    probe_ca: Optional[LinearProbe1x1],
    probe_cb: Optional[LinearProbe1x1],
    layer_idx: int,
    corridor_entrance_target: Tuple[int, int],
    corridor_direction: int,
    box_at_entrance: Tuple[int, int],
    push_direction: int,
    intervention_type: str = "agent_and_box",
    alpha: float = 1.0,
    max_steps: int = 120,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Cutoff level intervention (Appendix B.3).
    
    Three types:
      - 'agent_only': add directional C_A vector to corridor entrance target
      - 'box_only': add directional C_B vector to box at entrance
      - 'agent_and_box': both
    
    Intervention repeated until agent moves the box at the corridor entrance.
    """
    obs = env.reset(level)
    h, c = agent.init_hidden(1, device)

    box_moved = False
    trajectory = []
    done = False
    step = 0

    while not done and step < max_steps:
        if not box_moved:
            if intervention_type in ("agent_only", "agent_and_box") and probe_ca is not None:
                c = apply_intervention(
                    c, probe_ca, layer_idx,
                    [corridor_entrance_target], corridor_direction, alpha
                )

            if intervention_type in ("box_only", "agent_and_box") and probe_cb is not None:
                c = apply_intervention(
                    c, probe_cb, layer_idx,
                    [box_at_entrance], push_direction, alpha
                )

        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = agent.forward(obs_tensor, h, c)

        h = out["hidden_states"]
        c = out["cell_states"]

        action = out["policy_logits"].argmax(dim=-1).item()
        prev_obs = obs.copy()
        obs, reward, done, info = env.step(action)
        trajectory.append({"action": action, "reward": reward})
        step += 1

        if not box_moved and _box_pushed_from(prev_obs, obs, box_at_entrance):
            box_moved = True

    return {
        "solved": info.get("solved", False),
        "step_count": step,
        "trajectory": trajectory,
    }


def evaluate_interventions(
    agent: DRCAgent,
    levels: List[np.ndarray],
    intervention_fn,
    probe: LinearProbe1x1,
    layer_idx: int,
    config,
    device: torch.device,
    num_seeds: int = 5,
) -> Dict[str, float]:
    """
    Evaluate intervention success rate over multiple levels and probe seeds.
    
    Success rate is averaged over 5 independently trained/initialized probes.
    """
    from probing.probes import LinearProbe1x1 as LP1x1
    import random

    success_rates = []

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        successes = 0

        for level in levels:
            env = SokobanEnv(grid_size=config.env.grid_size)
            result = intervention_fn(
                agent=agent,
                env=env,
                level=level,
                probe=probe,
                layer_idx=layer_idx,
                alpha=config.intervention.alpha,
                device=device,
            )
            if result.get("solved_desired_way", False):
                successes += 1

        success_rates.append(successes / len(levels) * 100)

    return {
        "mean_success_rate": np.mean(success_rates),
        "std_success_rate": np.std(success_rates),
        "all_success_rates": success_rates,
    }


def create_random_probe(
    hidden_channels: int = 32,
    num_classes: int = 5,
    target_norm: float = 1.0,
) -> LinearProbe1x1:
    """
    Create a randomly initialized probe with similar norm to trained probes.
    Used as baseline for intervention experiments.
    """
    probe = LinearProbe1x1(hidden_channels, num_classes)
    nn.init.normal_(probe.conv.weight)
    with torch.no_grad():
        norms = probe.conv.weight.norm(dim=1, keepdim=True)
        probe.conv.weight.data = probe.conv.weight.data / norms * target_norm
    return probe


def _get_agent_pos(obs: np.ndarray) -> Optional[Tuple[int, int]]:
    """Get agent position from symbolic observation."""
    grid = obs.argmax(axis=-1)
    positions = list(zip(*np.where((grid == AGENT) | (grid == AGENT_ON_TARGET))))
    return positions[0] if positions else None


def _box_pushed_from(
    prev_obs: np.ndarray,
    curr_obs: np.ndarray,
    position: Tuple[int, int],
) -> bool:
    """Check if a box was pushed off a given position."""
    prev_grid = prev_obs.argmax(axis=-1)
    curr_grid = curr_obs.argmax(axis=-1)
    r, c = position
    was_box = prev_grid[r, c] in (BOX, BOX_ON_TARGET)
    is_box = curr_grid[r, c] in (BOX, BOX_ON_TARGET)
    return was_box and not is_box


def generate_agent_shortcut_levels(num_base_levels: int = 25) -> List[Dict]:
    """
    Generate Agent-Shortcut levels programmatically.
    
    These levels have all boxes/targets in one region, and the agent can
    reach them via a short or long path. The agent by default takes the
    short path; interventions steer it to take the long path.
    
    Returns list of level specs with grid, short_route, long_route info.
    """
    levels = []
    grid_size = 8

    for i in range(num_base_levels):
        grid = np.zeros((grid_size, grid_size), dtype=np.int64)

        for r in range(grid_size):
            grid[r, 0] = CELL_TYPES["WALL"]
            grid[r, grid_size - 1] = CELL_TYPES["WALL"]
        for c in range(grid_size):
            grid[0, c] = CELL_TYPES["WALL"]
            grid[grid_size - 1, c] = CELL_TYPES["WALL"]

        grid[1, 1] = AGENT
        grid[1, 6] = BOX
        grid[2, 6] = BOX_ON_TARGET
        grid[3, 6] = BOX
        grid[4, 6] = BOX_ON_TARGET

        short_route = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]
        long_route = [(2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)]

        levels.append({
            "grid": grid,
            "short_route": short_route,
            "long_route": long_route,
        })

    return levels


def generate_box_shortcut_levels(num_base_levels: int = 25) -> List[Dict]:
    """
    Generate Box-Shortcut levels programmatically.
    
    These levels have 3 boxes adjacent to targets and 1 box that can be
    pushed via a short or long route to a target.
    """
    levels = []
    grid_size = 8

    for i in range(num_base_levels):
        grid = np.zeros((grid_size, grid_size), dtype=np.int64)

        for r in range(grid_size):
            grid[r, 0] = CELL_TYPES["WALL"]
            grid[r, grid_size - 1] = CELL_TYPES["WALL"]
        for c in range(grid_size):
            grid[0, c] = CELL_TYPES["WALL"]
            grid[grid_size - 1, c] = CELL_TYPES["WALL"]

        grid[1, 1] = AGENT
        grid[2, 2] = BOX_ON_TARGET
        grid[3, 2] = BOX_ON_TARGET
        grid[4, 2] = BOX_ON_TARGET
        grid[2, 5] = BOX
        grid[2, 6] = CELL_TYPES["TARGET"]

        short_route = [(2, 5)]
        long_route = [(2, 5), (3, 5), (4, 5), (4, 6)]

        levels.append({
            "grid": grid,
            "short_route": short_route,
            "long_route": long_route,
            "box_initial_pos": (2, 5),
        })

    return levels


def augment_levels_with_symmetries(levels: List[Dict]) -> List[Dict]:
    """
    Augment levels by applying vertical reflection and 90/180/270 rotations.
    Creates 8 versions of each level (as described in Section 6.1).
    """
    augmented = []
    for level_spec in levels:
        grid = level_spec["grid"]

        for k in range(4):
            rotated_grid = np.rot90(grid, k=k)
            augmented.append({**level_spec, "grid": rotated_grid.copy()})

        flipped_grid = np.flipud(grid)
        for k in range(4):
            rotated_grid = np.rot90(flipped_grid, k=k)
            augmented.append({**level_spec, "grid": rotated_grid.copy()})

    return augmented
