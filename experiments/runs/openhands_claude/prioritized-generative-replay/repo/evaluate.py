"""Evaluation metrics and analysis tools for PGR.

Implements:
- Dormant ratio (DR): fraction of inactive neurons in policy network (Sokar et al., 2023)
- Generation MSE: faithfulness of generated transitions to environment dynamics
- Curiosity value distribution: F(τ) values over training
- tSNE projection of generated transitions (Fig. 2 in paper)
- Policy evaluation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from models.diffusion import ConditionalDiffusion, TransitionNormalizer, unpack_transition_tensor
from models.relevance import BaseRelevance


def compute_dormant_ratio(
    model: nn.Module,
    inputs: torch.Tensor,
    threshold: float = 0.025,
) -> float:
    """Compute the dormant ratio (DR) of a neural network.

    DR is the fraction of neurons with activation magnitude below threshold.
    Higher DR correlates with overfitting in value-based RL (Sokar et al., 2023;
    Xu et al., 2023). Used in Fig. 6a of the paper.

    Args:
        model: Policy or Q-network to evaluate.
        inputs: Input tensor to forward through the network.
        threshold: Activation threshold below which a neuron is considered dormant.

    Returns:
        Dormant ratio in [0, 1].
    """
    activations = []

    def hook_fn(module, input, output):
        if isinstance(module, (nn.ReLU, nn.Mish, nn.ELU, nn.Tanh)):
            activations.append(output.detach().abs().mean(dim=0))

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.Mish, nn.ELU, nn.Tanh)):
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        _ = model(inputs)

    for hook in hooks:
        hook.remove()

    if not activations:
        return 0.0

    all_activations = torch.cat([a.flatten() for a in activations])
    dormant_ratio = (all_activations < threshold).float().mean().item()
    return dormant_ratio


def compute_dormant_ratio_redq(
    agent,
    states: torch.Tensor,
    threshold: float = 0.025,
) -> float:
    """Compute dormant ratio for REDQ agent's actor network.

    The paper (Fig. 6a) measures DR on the policy network. We evaluate
    the actor trunk since it is the primary policy component.
    """
    return compute_dormant_ratio(agent.actor.trunk, states, threshold)


def compute_generation_mse(
    diffusion: ConditionalDiffusion,
    normalizer: TransitionNormalizer,
    env,
    real_buffer,
    state_dim: int,
    action_dim: int,
    n_eval: int = 10_000,
    top_k_ratio: float = 0.1,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure faithfulness of generated transitions to environment dynamics.

    Methodology from Lu et al. (2024) / Section 5.2 of the paper:
    1. Generate transition (s, a, s', r)
    2. Roll out action a from state s in the environment simulator
    3. Compute MSE between generated (s', r) and ground truth

    Returns:
        Dict with 'state_mse', 'reward_mse', and 'total_mse'.
    """
    n_top_k = max(1, int(real_buffer.size * top_k_ratio))
    top_k_data = real_buffer.get_top_k_relevance(n_top_k)
    top_k_relevance = top_k_data["relevance"]

    batch_size = min(n_eval, 512)
    all_state_mse = []
    all_reward_mse = []

    for start in range(0, n_eval, batch_size):
        n_batch = min(batch_size, n_eval - start)
        cond_idx = np.random.randint(0, len(top_k_relevance), size=n_batch)
        conditions = torch.FloatTensor(top_k_relevance[cond_idx]).to(device)

        with torch.no_grad():
            gen_norm = diffusion.sample(n_batch, conditions, use_guidance=True)
        gen = normalizer.denormalize_tensor(gen_norm).cpu().numpy()

        s_gen = gen[:, :state_dim]
        a_gen = gen[:, state_dim: state_dim + action_dim]
        sp_gen = gen[:, state_dim + action_dim: 2 * state_dim + action_dim]
        r_gen = gen[:, 2 * state_dim + action_dim:]

        for i in range(n_batch):
            try:
                env.reset()
                env.unwrapped.set_state_from_obs(s_gen[i])
                sp_true, r_true, _, _, _ = env.step(a_gen[i])
                state_mse = np.mean((sp_gen[i] - sp_true) ** 2)
                reward_mse = (r_gen[i, 0] - r_true) ** 2
                all_state_mse.append(state_mse)
                all_reward_mse.append(reward_mse)
            except Exception:
                pass

    if not all_state_mse:
        return {"state_mse": float("nan"), "reward_mse": float("nan"), "total_mse": float("nan")}

    return {
        "state_mse": float(np.mean(all_state_mse)),
        "reward_mse": float(np.mean(all_reward_mse)),
        "total_mse": float(np.mean(all_state_mse) + np.mean(all_reward_mse)),
        "state_mse_hist": all_state_mse,
        "reward_mse_hist": all_reward_mse,
    }


def compute_curiosity_distribution(
    relevance_fn: BaseRelevance,
    real_buffer,
    n_eval: int = 10_000,
    batch_size: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """Compute distribution of curiosity F-values over real transitions.

    Used in Fig. 6b of the paper to show how curiosity values evolve over training.
    Evaluated every 10K timesteps (inner loop frequency).
    """
    n_eval = min(n_eval, real_buffer.size)
    idx = np.random.choice(real_buffer.size, size=n_eval, replace=False)
    all_curiosity = []

    for start in range(0, n_eval, batch_size):
        end = min(start + batch_size, n_eval)
        batch_idx = idx[start:end]
        batch = {
            "states": torch.FloatTensor(real_buffer.states[batch_idx]).to(device),
            "actions": torch.FloatTensor(real_buffer.actions[batch_idx]).to(device),
            "next_states": torch.FloatTensor(real_buffer.next_states[batch_idx]).to(device),
            "rewards": torch.FloatTensor(real_buffer.rewards[batch_idx]).to(device),
        }
        with torch.no_grad():
            curiosity = relevance_fn.compute(batch).cpu().numpy().flatten()
        all_curiosity.extend(curiosity.tolist())

    return np.array(all_curiosity)


def compute_tsne_projections(
    diffusion_pgr: ConditionalDiffusion,
    diffusion_synther: ConditionalDiffusion,
    normalizer_pgr: TransitionNormalizer,
    normalizer_synther: TransitionNormalizer,
    real_buffer,
    n_samples: int = 10_000,
    top_k_ratio: float = 0.1,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Project PGR and SYNTHER generations to tSNE space for visualization.

    Reproduces Fig. 2 of the paper showing how PGR densifies a distinct
    subspace of transitions compared to SYNTHER.
    """
    from sklearn.manifold import TSNE

    n_top_k = max(1, int(real_buffer.size * top_k_ratio))
    top_k_data = real_buffer.get_top_k_relevance(n_top_k)
    top_k_relevance = top_k_data["relevance"]

    cond_idx = np.random.randint(0, len(top_k_relevance), size=n_samples)
    conditions_pgr = torch.FloatTensor(top_k_relevance[cond_idx]).to(device)
    conditions_synther = torch.zeros(n_samples, 1, device=device)

    with torch.no_grad():
        gen_pgr_norm = diffusion_pgr.sample(n_samples, conditions_pgr, use_guidance=True)
        gen_synther_norm = diffusion_synther.sample(n_samples, conditions_synther, use_guidance=False)

    gen_pgr = normalizer_pgr.denormalize_tensor(gen_pgr_norm).cpu().numpy()
    gen_synther = normalizer_synther.denormalize_tensor(gen_synther_norm).cpu().numpy()

    all_gen = np.concatenate([gen_pgr, gen_synther], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings = tsne.fit_transform(all_gen)

    return {
        "pgr_embeddings": embeddings[:n_samples],
        "synther_embeddings": embeddings[n_samples:],
    }


def evaluate_policy_return(
    agent,
    env,
    n_episodes: int = 10,
) -> Tuple[float, float]:
    """Evaluate policy and return mean and std of episode returns."""
    returns = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_return = 0.0
        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
    return float(np.mean(returns)), float(np.std(returns))


def run_evaluation_suite(
    agent,
    diffusion: Optional[ConditionalDiffusion],
    relevance_fn: Optional[BaseRelevance],
    real_buffer,
    env,
    eval_env,
    normalizer: Optional[TransitionNormalizer],
    state_dim: int,
    action_dim: int,
    step: int,
    n_eval_episodes: int = 10,
    n_curiosity_samples: int = 10_000,
    device: str = "cuda",
    compute_dr: bool = True,
    compute_gen_mse: bool = False,
) -> Dict[str, float]:
    """Run the full evaluation suite used in the paper's analysis sections."""
    results = {}

    mean_return, std_return = evaluate_policy_return(agent, eval_env, n_eval_episodes)
    results["eval/mean_return"] = mean_return
    results["eval/std_return"] = std_return

    if compute_dr:
        sample_states = torch.FloatTensor(
            real_buffer.states[np.random.randint(0, real_buffer.size, 256)]
        ).to(device)
        try:
            dr = compute_dormant_ratio(agent.actor.trunk, sample_states)
            results["eval/dormant_ratio"] = dr
        except Exception:
            pass

    if relevance_fn is not None and real_buffer.size > 0:
        curiosity_vals = compute_curiosity_distribution(
            relevance_fn, real_buffer, n_curiosity_samples, device=device
        )
        results["eval/curiosity_mean"] = float(np.mean(curiosity_vals))
        results["eval/curiosity_std"] = float(np.std(curiosity_vals))
        results["eval/curiosity_p90"] = float(np.percentile(curiosity_vals, 90))

    if compute_gen_mse and diffusion is not None and normalizer is not None:
        mse_results = compute_generation_mse(
            diffusion, normalizer, env, real_buffer,
            state_dim, action_dim, n_eval=1000, device=device
        )
        results.update({f"eval/{k}": v for k, v in mse_results.items() if isinstance(v, float)})

    return results
