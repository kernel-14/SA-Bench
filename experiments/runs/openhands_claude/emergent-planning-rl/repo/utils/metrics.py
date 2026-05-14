import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> float:
    """Compute macro-averaged F1 score."""
    return f1_score(
        y_true, y_pred,
        average="macro",
        labels=list(range(num_classes)),
        zero_division=0,
    )


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> Dict[str, List[float]]:
    """Compute per-class precision, recall, and F1."""
    labels = list(range(num_classes))
    return {
        "f1": f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        "precision": precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        "recall": recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
    }


def evaluate_agent(
    agent,
    levels: List[np.ndarray],
    env_config,
    device: torch.device,
    thinking_steps: int = 0,
    greedy: bool = True,
) -> Dict[str, float]:
    """
    Evaluate agent solve rate on a set of levels.
    
    Args:
        thinking_steps: number of forced stationary steps before acting
    Returns:
        dict with solve_rate and mean_reward
    """
    from environment.sokoban import SokobanEnv

    env = SokobanEnv(
        grid_size=env_config.grid_size,
        min_steps=env_config.min_episode_steps,
        max_steps=env_config.max_episode_steps,
    )
    agent.eval()

    solved_count = 0
    total_rewards = []

    for level in levels:
        obs = env.reset(level)
        h, c = agent.init_hidden(1, device)
        total_reward = 0.0
        done = False

        for _ in range(thinking_steps):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                out = agent.forward(obs_tensor, h, c)
            h = out["hidden_states"]
            c = out["cell_states"]

        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                out = agent.forward(obs_tensor, h, c)
            h = out["hidden_states"]
            c = out["cell_states"]

            if greedy:
                action = out["policy_logits"].argmax(dim=-1).item()
            else:
                action = torch.distributions.Categorical(
                    logits=out["policy_logits"]
                ).sample().item()

            obs, reward, done, info = env.step(action)
            total_reward += reward

        if info.get("solved", False):
            solved_count += 1
        total_rewards.append(total_reward)

    return {
        "solve_rate": solved_count / len(levels),
        "mean_reward": np.mean(total_rewards),
        "num_levels": len(levels),
        "num_solved": solved_count,
    }


def compute_thinking_steps_benefit(
    agent,
    levels: List[np.ndarray],
    env_config,
    device: torch.device,
    max_thinking_steps: int = 5,
) -> Dict[int, float]:
    """
    Compute the number of additional levels solved with each number of thinking steps.
    Used for Figure 9 / emergence analysis.
    
    Returns dict mapping thinking_steps -> solve_rate
    """
    results = {}
    for n_think in range(max_thinking_steps + 1):
        metrics = evaluate_agent(agent, levels, env_config, device, thinking_steps=n_think)
        results[n_think] = metrics["solve_rate"]
    return results


def compute_extra_levels_solved(
    agent,
    levels: List[np.ndarray],
    env_config,
    device: torch.device,
    thinking_steps: int = 5,
) -> int:
    """
    Count levels solved with thinking_steps but not without.
    Used for Figure 9 correlation analysis.
    """
    base_metrics = evaluate_agent(agent, levels, env_config, device, thinking_steps=0)
    think_metrics = evaluate_agent(agent, levels, env_config, device, thinking_steps=thinking_steps)

    base_solved = base_metrics["num_solved"]
    think_solved = think_metrics["num_solved"]
    return max(0, think_solved - base_solved)


def compute_probe_f1_over_ticks(
    probe,
    tick_data: Dict[int, Tuple[List, List]],
    device: torch.device,
    num_classes: int = 5,
) -> Dict[int, float]:
    """
    Compute macro F1 at each tick during thinking steps.
    Used for Figure 6 / test-time plan refinement.
    """
    import torch
    from torch.utils.data import DataLoader
    from probing.evaluate import ProbeDataset

    results = {}
    probe.eval()

    for tick_idx, (cell_states, labels) in tick_data.items():
        if not cell_states:
            continue
        ds = ProbeDataset(cell_states, labels)
        loader = DataLoader(ds, batch_size=64, shuffle=False)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for cs, lb in loader:
                cs = cs.to(device)
                logits = probe(cs)
                if logits.dim() == 4:
                    preds = logits.argmax(dim=1).cpu().numpy().reshape(-1)
                    all_preds.append(preds)
                    all_labels.append(lb.numpy().reshape(-1))
                else:
                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(lb.numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        results[tick_idx] = macro_f1(all_labels, all_preds, num_classes)

    return results
