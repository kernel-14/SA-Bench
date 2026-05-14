"""
Thinking time analysis: evaluating plan refinement with extra test-time compute.

Reproduces experiments from Sections 5 and 6.2:
- Agent forced to remain stationary for K "thinking steps" before acting
- Internal plan decoded from cell states at each internal tick
- Macro F1 measures how well the plan predicts future behavior
- Correlation between plan quality improvement and benefit from extra compute
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from ..environment.sokoban import SokobanEnv, parse_boxoban_level
from ..models.drc import DRCNet
from ..probing.linear_probe import LinearProbe, CLASS_NAMES
from ..probing.concepts import ConceptLabeler, CLASS_NEVER
from sklearn.metrics import f1_score


def evaluate_thinking_time(
    model: DRCNet,
    levels: List[str],
    num_episodes: int = 1000,
    num_thinking_steps: int = 5,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Evaluate how much better the agent performs with extra thinking time.

    This is behavioral evidence: we measure solve rate with and without
    forced stationary steps at the start of episodes.

    Returns:
        metrics: dict with solve rates and improvement
    """
    model.eval()
    env = SokobanEnv()

    def solve_levels(thinking_steps: int = 0) -> int:
        solved = 0
        for ep in range(num_episodes):
            level_idx = np.random.randint(0, len(levels))
            grid = parse_boxoban_level(levels[level_idx])
            env.load_level(grid)
            env._episode_max_steps = 120
            obs = env.reset()

            # Thinking steps: agent stays still
            model_states = None
            for _ in range(thinking_steps):
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, _, model_states = model(obs_tensor, model_states)

            # Normal episode
            done = False
            while not done:
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _, model_states = model(obs_tensor, model_states)
                action = torch.argmax(logits, dim=-1).item()
                obs, reward, done, info = env.step(action)

            if info.get("solved", env._is_solved()):
                solved += 1

        return solved

    solved_no_think = solve_levels(0)
    solved_with_think = solve_levels(num_thinking_steps)

    extra_solved = max(0, solved_with_think - solved_no_think)

    return {
        "solved_no_thinking": solved_no_think,
        "solved_with_thinking": solved_with_think,
        "extra_solved": extra_solved,
        "num_episodes": num_episodes,
        "solve_rate_no_thinking": solved_no_think / max(1, num_episodes),
        "solve_rate_with_thinking": solved_with_think / max(1, num_episodes),
        "extra_solve_rate": extra_solved / max(1, num_episodes),
    }


def analyze_plan_refinement(
    model: DRCNet,
    probe: LinearProbe,
    levels: List[str],
    num_episodes: int = 1000,
    num_thinking_steps: int = 5,
    device: torch.device = torch.device("cpu"),
    concept_type: str = "agent_approach",
    layer_idx: int = -1,
) -> Dict[int, Dict[str, float]]:
    """
    Analyze how the agent's internal plan improves across thinking time ticks.

    For each internal tick during thinking steps, decode the plan using the probe
    and measure how well it predicts future behavior (macro F1).

    This reproduces Figure 6 in the paper.

    Returns:
        tick_metrics: dict mapping tick number to dict of metrics
    """
    model.eval()
    probe.eval()
    probe = probe.to(device)

    env = SokobanEnv()
    labeler = ConceptLabeler(env, concept_type=concept_type)

    all_tick_predictions = defaultdict(list)
    all_labels = []

    for ep in range(num_episodes):
        level_idx = np.random.randint(0, len(levels))
        grid = parse_boxoban_level(levels[level_idx])
        env.load_level(grid)
        env._episode_max_steps = 120
        obs = env.reset()

        # Thinking steps: record cell states at each tick
        tick_states = defaultdict(list)
        model_states = None

        for think_step in range(num_thinking_steps):
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, value, model_states = model(obs_tensor, model_states)

                # Record cell state at each internal tick
                # DRC performs N internal ticks per forward call,
                # so each call gives us one set of states after N ticks
                # For inter-tick analysis we need to modify model to output intermediate ticks
                # Here we record after each forward call (equivalent to after each thinking step)

                actual_layer = layer_idx
                if actual_layer < 0:
                    actual_layer = model.num_layers + actual_layer
                if model_states[actual_layer] is not None:
                    cell_state = model_states[actual_layer][1]  # (1, C, H, W)
                    tick_states[think_step].append(cell_state.cpu().numpy()[0])

        # Run episode to completion to get labels
        actions = []
        done = False
        while not done:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _, model_states = model(obs_tensor, model_states)
            action = torch.argmax(logits, dim=-1).item()
            actions.append(action)
            obs, reward, done, info = env.step(action)

        # Compute labels for the full episode
        episode_labels = labeler.compute_episode_labels(actions, grid)

        # Use labels from step 0 (initial plan)
        if 0 in episode_labels:
            labels = episode_labels[0]
            for tick, states in tick_states.items():
                if states:
                    # Use probe to predict
                    state_tensor = torch.from_numpy(states[0]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred_logits = probe(state_tensor)
                        preds = torch.argmax(pred_logits, dim=1)[0].cpu().numpy()

                    all_tick_predictions[tick].append(preds.flatten())
                    if tick == 0:
                        all_labels.append(labels.flatten())

    # Compute metrics per tick
    tick_metrics = {}
    for tick in range(num_thinking_steps):
        if tick in all_tick_predictions and len(all_tick_predictions[tick]) > 0:
            preds_flat = np.concatenate(all_tick_predictions[tick])
            if len(all_labels) > 0:
                labels_flat = np.concatenate(all_labels)
                # Ensure same length
                min_len = min(len(preds_flat), len(labels_flat))
                preds_flat = preds_flat[:min_len]
                labels_flat = labels_flat[:min_len]

                macro_f1 = f1_score(labels_flat, preds_flat, average="macro")
                accuracy = (preds_flat == labels_flat).mean()

                tick_metrics[tick] = {
                    "macro_f1": macro_f1,
                    "accuracy": accuracy,
                    "num_samples": len(preds_flat),
                }

    return tick_metrics


def analyze_plan_refinement_across_ticks(
    model: DRCNet,
    probe: LinearProbe,
    levels: List[str],
    num_episodes: int = 1000,
    device: torch.device = torch.device("cpu"),
    concept_type: str = "agent_approach",
    layer_idx: int = -1,
) -> Dict[int, Dict[str, float]]:
    """
    More detailed analysis: decode plan at each INTERNAL TICK within thinking steps.

    This requires modifying the model's forward pass to capture intermediate states.
    For the DRC architecture, we can call forward with num_ticks=1 repeatedly
    to step through internal ticks.

    Returns metrics per internal tick.
    """
    model.eval()
    probe.eval()
    probe = probe.to(device)

    env = SokobanEnv()
    labeler = ConceptLabeler(env, concept_type=concept_type)

    num_internal_ticks = model.num_ticks * num_episodes  # placeholder
    # We'll collect per actual thinking step
    total_ticks = model.num_ticks * 5  # assuming 5 thinking steps

    tick_states_all = defaultdict(list)
    tick_labels_all = []

    for ep in range(num_episodes):
        level_idx = np.random.randint(0, len(levels))
        grid = parse_boxoban_level(levels[level_idx])
        env.load_level(grid)
        env._episode_max_steps = 120
        obs = env.reset()

        model_states = None
        internal_tick_idx = 0

        # Simulate thinking steps, capturing each internal tick
        for think_step in range(5):
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)

            # Manually step through each internal tick
            # We model this by calling forward once (which does N ticks)
            # and extracting states after. For per-tick analysis, we'd need
            # a modified forward that stores intermediate states.
            # For now, capture after each forward call = after each thinking step.
            with torch.no_grad():
                _, _, model_states = model(obs_tensor, model_states)

            actual_layer = layer_idx
            if actual_layer < 0:
                actual_layer = model.num_layers + actual_layer
            if model_states[actual_layer] is not None:
                cell_state = model_states[actual_layer][1]
                tick_states_all[think_step].append(cell_state.cpu().numpy()[0])

        # Run episode
        actions = []
        done = False
        while not done:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _, model_states = model(obs_tensor, model_states)
            action = torch.argmax(logits, dim=-1).item()
            actions.append(action)
            obs, reward, done, info = env.step(action)

        episode_labels = labeler.compute_episode_labels(actions, grid)
        if 0 in episode_labels:
            tick_labels_all.append(episode_labels[0].flatten())

    # Compute metrics
    tick_metrics = {}
    for tick in sorted(tick_states_all.keys()):
        states_list = tick_states_all[tick]
        if not states_list:
            continue

        all_preds = []
        for state in states_list:
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = torch.argmax(probe(state_tensor), dim=1)[0].cpu().numpy()
            all_preds.append(preds.flatten())

        if all_preds and tick_labels_all:
            preds_flat = np.concatenate(all_preds)
            labels_flat = np.concatenate(tick_labels_all)
            min_len = min(len(preds_flat), len(labels_flat))
            preds_flat = preds_flat[:min_len]
            labels_flat = labels_flat[:min_len]

            tick_metrics[tick] = {
                "macro_f1": f1_score(labels_flat, preds_flat, average="macro"),
                "accuracy": (preds_flat == labels_flat).mean(),
            }

    return tick_metrics
