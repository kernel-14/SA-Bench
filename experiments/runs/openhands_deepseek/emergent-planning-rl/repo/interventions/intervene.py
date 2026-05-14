"""
Intervention experiments for demonstrating causal influence of concept
representations on agent behavior.

Based on Section 6 and Appendix B of the paper.

Interventions add learned concept direction vectors to the agent's cell state
to steer it towards forming and executing specific plans.

Formula: g'_{x,y} = g_{x,y} + alpha * w_k
where w_k is the class direction vector learned by a linear probe.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from ..environment.sokoban import (
    SokobanEnv, parse_boxoban_level, grid_to_symbolic,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_NOOP,
    E_AGENT, E_AGENT_ON_TARGET, E_BOX, E_BOX_ON_TARGET, E_TARGET,
)
from ..models.drc import DRCNet
from ..probing.linear_probe import LinearProbe, CLASS_NAMES

# Map from class name to index
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def get_class_vectors(probe: LinearProbe) -> Dict[int, torch.Tensor]:
    """
    Extract class direction vectors from a trained linear probe.

    For a 1x1 probe (Conv2d with kernel_size=1), the weight has shape
    (num_classes, in_channels, 1, 1). The class vector w_k is the
    weight for class k, flattened to (in_channels,).
    """
    weight = probe.conv.weight  # (num_classes, C, 1, 1)
    bias = probe.conv.bias    # (num_classes,)

    class_vectors = {}
    for k in range(probe.num_classes):
        w_k = weight[k, :, 0, 0]  # (C,)
        class_vectors[k] = w_k.detach().clone()
    return class_vectors


class InterventionEngine:
    """
    Engine for performing interventions on DRC agent cell states.

    Supports:
    - Agent-Shortcut interventions (using C_A concept vectors)
    - Box-Shortcut interventions (using C_B concept vectors)
    - Cutoff interventions
    """

    def __init__(
        self,
        model: DRCNet,
        probe: LinearProbe,
        device: torch.device,
        concept_type: str = "agent_approach",
    ):
        self.model = model.to(device)
        self.probe = probe.to(device)
        self.device = device
        self.concept_type = concept_type

        # Extract class vectors from probe
        self.class_vectors = get_class_vectors(probe)

    def intervene(
        self,
        cell_state: torch.Tensor,
        positions: List[Tuple[int, int]],
        class_name: str,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Add class direction vector to specified positions of cell state.

        Args:
            cell_state: (B, C, H, W) cell state tensor
            positions: list of (r, c) positions to intervene on
            class_name: name of class to encourage (e.g., "NEVER", "UP")
            alpha: intervention strength multiplier

        Returns:
            modified cell state
        """
        if class_name not in CLASS_TO_IDX:
            return cell_state

        cls_idx = CLASS_TO_IDX[class_name]
        w = self.class_vectors[cls_idx].to(self.device)  # (C,)

        B, C, H, W = cell_state.shape
        modified = cell_state.clone()

        for r, c in positions:
            if 0 <= r < H and 0 <= c < W:
                modified[0, :, r, c] = modified[0, :, r, c] + alpha * w

        return modified

    def run_episode_with_intervention(
        self,
        grid: np.ndarray,
        intervention_fn,
        max_steps: int = 120,
    ) -> Tuple[bool, List[dict], List[np.ndarray]]:
        """
        Run an episode with dynamic interventions applied at each step.

        Args:
            grid: initial grid
            intervention_fn: function (model_states, step, prev_actions, env) ->
                modified model_states to use for this step
            max_steps: maximum episode length

        Returns:
            solved: whether the level was solved
            trajectory: list of trajectory dicts
            cell_states_history: list of cell states at each step
        """
        env = SokobanEnv(max_steps=max_steps)
        env.load_level(grid)
        obs = env.reset()

        model_states = None
        trajectory = []
        cell_states_history = []

        done = False
        step = 0

        while not done and step < max_steps:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits, value, model_states = self.model(obs_tensor, model_states)

                # Apply intervention before action selection
                model_states = intervention_fn(
                    model_states, step, trajectory, env
                )

                # Record cell state
                if model_states is not None:
                    for d in range(self.model.num_layers):
                        if model_states[d] is not None:
                            cell_states_history.append(
                                model_states[d][1].cpu().numpy()[0]
                            )
                            break

            # Select action
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()

            trajectory.append({
                "step": step,
                "action": action,
                "agent_pos": env.get_agent_position(),
                "logits": logits.cpu().numpy()[0],
            })

            obs, reward, done, info = env.step(action)
            step += 1

        solved = info.get("solved", env._is_solved())
        return solved, trajectory, cell_states_history

    def run_episode_no_intervention(
        self,
        grid: np.ndarray,
        max_steps: int = 120,
    ) -> Tuple[bool, List[dict], List[np.ndarray]]:
        """Run episode without any intervention (baseline)."""
        return self.run_episode_with_intervention(
            grid,
            intervention_fn=lambda states, step, traj, env: states,
            max_steps=max_steps,
        )


class AgentShortcutIntervention:
    """
    Agent-Shortcut intervention (Section 6.1, Algorithm 1).

    Forces the agent to follow the long path by:
    1. Adding NEVER vector to short route positions (every step)
    2. Adding directional vector to long route start (until agent steps there)
    """

    def __init__(
        self,
        engine: InterventionEngine,
        short_route_squares: List[Tuple[int, int]],
        long_route_squares_dirs: List[Tuple[Tuple[int, int], str]],
        alpha: float = 1.0,
    ):
        self.engine = engine
        self.short_route_squares = short_route_squares
        self.long_route_squares_dirs = long_route_squares_dirs
        self.alpha = alpha
        self._directional_applied = False

    def intervene(
        self,
        model_states: List,
        step: int,
        trajectory: List[dict],
        env: SokobanEnv,
        layer: int = -1,
    ) -> List:
        """
        Apply intervention to model states.
        layer=-1 means last layer.
        """
        if layer < 0:
            layer = self.engine.model.num_layers + layer

        if model_states[layer] is None:
            return model_states

        h, c = model_states[layer]
        B, C, H, W = c.shape

        # Short-route intervention: add NEVER
        c = self.engine.intervene(
            c.unsqueeze(0),  # add batch dim temporarily
            self.short_route_squares,
            "NEVER",
            alpha=self.alpha,
        ).squeeze(0)  # remove batch dim

        # Directional intervention (until agent reaches long route start)
        if not self._directional_applied:
            agent_pos = env.get_agent_position()
            for (lr, lc), direction in self.long_route_squares_dirs:
                if agent_pos == (lr, lc):
                    self._directional_applied = True
                    break
                c = self.engine.intervene(
                    c.unsqueeze(0),
                    [(lr, lc)],
                    direction,
                    alpha=self.alpha,
                ).squeeze(0)

        model_states[layer] = (h, c)
        return model_states


class BoxShortcutIntervention:
    """
    Box-Shortcut intervention (Section 6.1, Algorithm 2).

    Forces a box to be pushed the long route by:
    1. Adding NEVER vector to short route positions (every step)
    2. Adding directional vector to box initial position (until box pushed off)
    """

    def __init__(
        self,
        engine: InterventionEngine,
        short_route_squares: List[Tuple[int, int]],
        box_pos: Tuple[int, int],
        long_direction: str,
        alpha: float = 1.0,
    ):
        self.engine = engine
        self.short_route_squares = short_route_squares
        self.box_pos = box_pos
        self.long_direction = long_direction
        self.alpha = alpha
        self._box_moved = False

    def intervene(
        self,
        model_states: List,
        step: int,
        trajectory: List[dict],
        env: SokobanEnv,
        layer: int = -1,
    ) -> List:
        if layer < 0:
            layer = self.engine.model.num_layers + layer

        if model_states[layer] is None:
            return model_states

        h, c = model_states[layer]
        B, C, H, W = c.shape

        # Short-route intervention: add NEVER
        c = self.engine.intervene(
            c.unsqueeze(0),
            self.short_route_squares,
            "NEVER",
            alpha=self.alpha,
        ).squeeze(0)

        # Directional intervention on box position
        br, bc = self.box_pos
        grid = env.get_grid()
        if not self._box_moved:
            if not (grid[br, bc] == E_BOX or grid[br, bc] == E_BOX_ON_TARGET):
                self._box_moved = True
            else:
                c = self.engine.intervene(
                    c.unsqueeze(0),
                    [(br, bc)],
                    self.long_direction,
                    alpha=self.alpha,
                ).squeeze(0)

        model_states[layer] = (h, c)
        return model_states


class CutoffIntervention:
    """
    Cutoff level intervention (Appendix B.3).

    Three variants:
    - Agent-Only: add directional C_A vector to target at corridor entrance
    - Box-Only: add directional C_B vector to box at corridor entrance
    - Agent-and-Box: both interventions together
    """

    def __init__(
        self,
        engine: InterventionEngine,
        agent_intervention_pos: Optional[Tuple[int, int]] = None,
        agent_intervention_dir: Optional[str] = None,
        box_intervention_pos: Optional[Tuple[int, int]] = None,
        box_intervention_dir: Optional[str] = None,
        alpha: float = 1.0,
    ):
        self.engine = engine
        self.agent_pos = agent_intervention_pos
        self.agent_dir = agent_intervention_dir
        self.box_pos = box_intervention_pos
        self.box_dir = box_intervention_dir
        self.alpha = alpha
        self._intervention_applied = False

    def intervene(
        self,
        model_states: List,
        step: int,
        trajectory: List[dict],
        env: SokobanEnv,
        agent_layer: int = -1,
        box_layer: int = -1,
    ) -> List:
        """
        Apply interventions to model states.

        Uses agent_layer for C_A interventions and box_layer for C_B interventions.
        """
        if self._intervention_applied:
            return model_states

        if agent_layer < 0:
            agent_layer = self.engine.model.num_layers + agent_layer
        if box_layer < 0:
            box_layer = self.engine.model.num_layers + box_layer

        # Agent-Only intervention
        if self.agent_pos is not None and self.agent_dir is not None:
            if model_states[agent_layer] is not None:
                h, c = model_states[agent_layer]
                B, C, H, W = c.shape
                c = self.engine.intervene(
                    c.unsqueeze(0),
                    [self.agent_pos],
                    self.agent_dir,
                    alpha=self.alpha,
                ).squeeze(0)
                model_states[agent_layer] = (h, c)

        # Box-Only intervention
        if self.box_pos is not None and self.box_dir is not None:
            if model_states[box_layer] is not None:
                h, c = model_states[box_layer]
                B, C, H, W = c.shape
                c = self.engine.intervene(
                    c.unsqueeze(0),
                    [self.box_pos],
                    self.box_dir,
                    alpha=self.alpha,
                ).squeeze(0)
                model_states[box_layer] = (h, c)

        # Check if box was moved (stop after)
        grid = env.get_grid()
        if self.box_pos is not None:
            br, bc = self.box_pos
            if not (grid[br, bc] == E_BOX or grid[br, bc] == E_BOX_ON_TARGET):
                self._intervention_applied = True

        return model_states


def evaluate_intervention_success(
    engine: InterventionEngine,
    levels_info: List[Tuple],
    intervention_type: str = "agent_shortcut",
    layer: int = -1,
    alpha: float = 1.0,
    num_repeats: int = 5,
) -> Dict[str, float]:
    """
    Evaluate intervention success rate over a set of levels.

    Args:
        engine: InterventionEngine
        levels_info: list of (grid, short_route_squares, box_pos, long_dirs)
        intervention_type: "agent_shortcut", "box_shortcut", or "cutoff"
        layer: which layer to intervene on
        alpha: intervention strength
        num_repeats: number of times to repeat with different seeds

    Returns:
        metrics: dict with success_rate and other statistics
    """
    successes = 0
    total = 0

    for grid, short_route, pos_or_start, long_dirs in levels_info:
        for _ in range(num_repeats):
            if intervention_type == "agent_shortcut":
                intervention = AgentShortcutIntervention(
                    engine, short_route, long_dirs, alpha=alpha
                )
            elif intervention_type == "box_shortcut":
                box_pos = pos_or_start
                long_dir = long_dirs[0][1] if long_dirs else "RIGHT"
                intervention = BoxShortcutIntervention(
                    engine, short_route, box_pos, long_dir, alpha=alpha
                )
            else:
                continue

            solved, trajectory, cell_states = engine.run_episode_with_intervention(
                grid,
                intervention_fn=lambda states, step, traj, env:
                    intervention.intervene(states, step, traj, env, layer=layer),
            )

            if solved:
                successes += 1
            total += 1

    success_rate = successes / max(1, total)
    return {
        "success_rate": success_rate,
        "num_successes": successes,
        "num_total": total,
    }
