import argparse
import os
import time
import numpy as np
import torch
from typing import Dict, Optional

from config import (
    PGRConfig, REDQConfig, DiffusionConfig, ICMConfig, RNDConfig,
    CTSConfig, ECOConfig, ReplayBufferConfig, TrainingConfig,
    SPARSE_REWARD_ENVS, FINGER_TURN_HARD_STEPS,
)
from replay_buffer import ReplayBuffer, LatentReplayBuffer, PixelReplayBuffer, MixedReplayBuffer
from models.diffusion import (
    ConditionalDiffusion,
    TransitionNormalizer,
    build_transition_tensor,
    unpack_transition_tensor,
)
from models.rl_agents import REDQ, SAC, DRQv2
from models.relevance import build_relevance_fn
from utils import (
    make_env,
    set_seed,
    Logger,
    evaluate_policy,
    get_transition_dim,
)


def train_diffusion_model(
    diffusion: ConditionalDiffusion,
    real_buffer: ReplayBuffer,
    normalizer: TransitionNormalizer,
    state_dim: int,
    action_dim: int,
    top_k_ratio: float,
    n_steps: int,
    batch_size: int,
    device: str,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train the conditional diffusion model on real transitions.

    Uses the prompting strategy from Peebles et al. (2022):
    select top-k% transitions by relevance, sample their F values as conditions.
    """
    diffusion.train()
    total_loss = 0.0
    n_top_k = max(1, int(real_buffer.size * top_k_ratio))

    for _ in range(n_steps):
        top_k_data = real_buffer.get_top_k_relevance(n_top_k)
        top_k_relevance = top_k_data["relevance"]

        idx = np.random.randint(0, real_buffer.size, size=batch_size)
        states = torch.FloatTensor(real_buffer.states[idx]).to(device)
        actions = torch.FloatTensor(real_buffer.actions[idx]).to(device)
        next_states = torch.FloatTensor(real_buffer.next_states[idx]).to(device)
        rewards = torch.FloatTensor(real_buffer.rewards[idx]).to(device)

        transitions = build_transition_tensor(states, actions, next_states, rewards)
        transitions_norm = normalizer.normalize_tensor(transitions)

        cond_idx = np.random.randint(0, len(top_k_relevance), size=batch_size)
        conditions = torch.FloatTensor(top_k_relevance[cond_idx]).to(device)

        loss = diffusion.loss(transitions_norm, conditions)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    diffusion.eval()
    return total_loss / n_steps


def generate_synthetic_transitions(
    diffusion: ConditionalDiffusion,
    real_buffer: ReplayBuffer,
    syn_buffer: ReplayBuffer,
    normalizer: TransitionNormalizer,
    state_dim: int,
    action_dim: int,
    n_generate: int,
    top_k_ratio: float,
    guidance_scale: float,
    device: str,
):
    """Generate synthetic transitions and add to synthetic replay buffer.

    Conditions on relevance values sampled from the top-k% of real transitions.
    """
    diffusion.eval()
    n_top_k = max(1, int(real_buffer.size * top_k_ratio))
    top_k_data = real_buffer.get_top_k_relevance(n_top_k)
    top_k_relevance = top_k_data["relevance"]

    batch_size = min(n_generate, 1024)
    generated_all = []

    for start in range(0, n_generate, batch_size):
        n_batch = min(batch_size, n_generate - start)
        cond_idx = np.random.randint(0, len(top_k_relevance), size=n_batch)
        conditions = torch.FloatTensor(top_k_relevance[cond_idx]).to(device)

        with torch.no_grad():
            gen_norm = diffusion.sample(n_batch, conditions, use_guidance=True)

        gen = normalizer.denormalize_tensor(gen_norm)
        generated_all.append(gen.cpu().numpy())

    generated = np.concatenate(generated_all, axis=0)
    s, a, sp, r = (
        generated[:, :state_dim],
        generated[:, state_dim: state_dim + action_dim],
        generated[:, state_dim + action_dim: 2 * state_dim + action_dim],
        generated[:, 2 * state_dim + action_dim:],
    )
    syn_buffer.add_batch(s, a, sp, r, np.zeros((len(s), 1)))


def run_pgr(
    env_name: str,
    relevance_type: str = "curiosity",
    seed: int = 0,
    total_env_steps: int = 100_000,
    hidden_dim: int = 256,
    n_hidden: int = 2,
    utd_ratio: int = 20,
    batch_size: int = 256,
    synthetic_ratio: float = 0.5,
    real_buffer_size: int = 1_000_000,
    syn_buffer_size: int = 1_000_000,
    inner_loop_freq: int = 10_000,
    diffusion_train_steps: int = 50_000,
    diffusion_batch_size: int = 256,
    top_k_ratio: float = 0.1,
    guidance_scale: float = 1.5,
    p_uncond: float = 0.25,
    n_diffusion_steps: int = 1000,
    seed_steps: int = 5_000,
    eval_freq: int = 5_000,
    eval_episodes: int = 10,
    device: str = "cuda",
    log_dir: str = "logs",
    use_wandb: bool = False,
    pixel_obs: bool = False,
    icm_update_freq: float = 0.05,
):
    """Main PGR training loop implementing Algorithm 1 from the paper.

    Outer loop:
        1. Collect transitions with π, add to D_real
        2. Update relevance function F using D_real
    Inner loop (every inner_loop_freq steps):
        3. Train conditional diffusion model G on D_real
        4. Generate synthetic transitions from G, add to D_syn
        5. Train π on samples from D_real ∪ D_syn with ratio r
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if env_name in SPARSE_REWARD_ENVS:
        total_env_steps = FINGER_TURN_HARD_STEPS

    env = make_env(env_name, seed=seed, pixel_obs=pixel_obs)
    eval_env = make_env(env_name, seed=seed + 100, pixel_obs=pixel_obs)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    transition_dim = 2 * state_dim + action_dim + 1

    logger = Logger(
        log_dir=os.path.join(log_dir, f"pgr_{relevance_type}_{env_name}_seed{seed}"),
        use_wandb=use_wandb,
        config={
            "env": env_name,
            "relevance": relevance_type,
            "seed": seed,
            "utd_ratio": utd_ratio,
            "synthetic_ratio": synthetic_ratio,
            "guidance_scale": guidance_scale,
        },
    )

    real_buffer = ReplayBuffer(state_dim, action_dim, real_buffer_size, str(device))
    syn_buffer = ReplayBuffer(state_dim, action_dim, syn_buffer_size, str(device))
    mixed_buffer = MixedReplayBuffer(real_buffer, syn_buffer, synthetic_ratio)

    agent = REDQ(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        utd_ratio=utd_ratio,
        device=str(device),
    )

    relevance_fn = build_relevance_fn(
        relevance_type,
        state_dim=state_dim,
        action_dim=action_dim,
        agent=agent,
        device=str(device),
    )

    diffusion = ConditionalDiffusion(
        transition_dim=transition_dim,
        hidden_dim=256,
        n_hidden_layers=4,
        time_embed_dim=128,
        cond_embed_dim=128,
        n_diffusion_steps=n_diffusion_steps,
        p_uncond=p_uncond,
        guidance_scale=guidance_scale,
    ).to(device)

    diffusion_optimizer = torch.optim.Adam(diffusion.parameters(), lr=3e-4)
    normalizer = TransitionNormalizer()

    state, _ = env.reset()
    episode_return = 0.0
    episode_steps = 0
    n_policy_updates = 0
    diffusion_trained = False

    for step in range(total_env_steps):
        if step < seed_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward
        episode_steps += 1

        real_buffer.add(state, action, next_state, reward, float(terminated))
        state = next_state

        if done:
            logger.log({"train/episode_return": episode_return, "train/episode_steps": episode_steps}, step)
            state, _ = env.reset()
            episode_return = 0.0
            episode_steps = 0

        if step < seed_steps:
            continue

        rel_batch = {
            "states": torch.FloatTensor(real_buffer.states[real_buffer.ptr - 1: real_buffer.ptr]).to(device),
            "actions": torch.FloatTensor(real_buffer.actions[real_buffer.ptr - 1: real_buffer.ptr]).to(device),
            "next_states": torch.FloatTensor(real_buffer.next_states[real_buffer.ptr - 1: real_buffer.ptr]).to(device),
            "rewards": torch.FloatTensor(real_buffer.rewards[real_buffer.ptr - 1: real_buffer.ptr]).to(device),
            "dones": torch.FloatTensor(real_buffer.dones[real_buffer.ptr - 1: real_buffer.ptr]).to(device),
        }

        if hasattr(relevance_fn, "update") and np.random.random() < icm_update_freq:
            if real_buffer.size >= batch_size:
                rel_update_batch = real_buffer.sample(batch_size)
                rel_logs = relevance_fn.update(rel_update_batch)
                logger.log(rel_logs, step)

        if real_buffer.size >= batch_size:
            with torch.no_grad():
                rel_val = relevance_fn.compute(rel_batch).cpu().numpy()
            idx = (real_buffer.ptr - 1) % real_buffer.max_size
            real_buffer.update_relevance(np.array([idx]), rel_val)

        if (step + 1) % inner_loop_freq == 0 and real_buffer.size >= batch_size:
            real_buffer.update_all_relevance(relevance_fn.compute, batch_size=512)

            transitions_arr = real_buffer.as_transitions_array()
            normalizer.fit(transitions_arr)

            diffusion_loss = train_diffusion_model(
                diffusion=diffusion,
                real_buffer=real_buffer,
                normalizer=normalizer,
                state_dim=state_dim,
                action_dim=action_dim,
                top_k_ratio=top_k_ratio,
                n_steps=diffusion_train_steps,
                batch_size=diffusion_batch_size,
                device=str(device),
                optimizer=diffusion_optimizer,
            )
            logger.log({"diffusion/train_loss": diffusion_loss}, step)

            n_generate = syn_buffer_size
            syn_buffer.ptr = 0
            syn_buffer.size = 0
            generate_synthetic_transitions(
                diffusion=diffusion,
                real_buffer=real_buffer,
                syn_buffer=syn_buffer,
                normalizer=normalizer,
                state_dim=state_dim,
                action_dim=action_dim,
                n_generate=n_generate,
                top_k_ratio=top_k_ratio,
                guidance_scale=guidance_scale,
                device=str(device),
            )
            diffusion_trained = True
            logger.log({"diffusion/syn_buffer_size": syn_buffer.size}, step)

        if diffusion_trained and real_buffer.size >= batch_size:
            for _ in range(utd_ratio):
                batch = mixed_buffer.sample(batch_size)
                agent_logs = agent.update(batch)
                n_policy_updates += 1
            logger.log(agent_logs, step)
        elif real_buffer.size >= batch_size:
            for _ in range(1):
                batch = real_buffer.sample(batch_size)
                agent_logs = agent.update(batch)
                n_policy_updates += 1

        if (step + 1) % eval_freq == 0:
            eval_return = evaluate_policy(agent, eval_env, eval_episodes)
            logger.log({"eval/episode_return": eval_return}, step)
            print(f"Step {step + 1}/{total_env_steps} | Eval return: {eval_return:.2f}")

    env.close()
    eval_env.close()
    logger.close()
    return agent, diffusion


def run_pgr_pixel(
    env_name: str,
    relevance_type: str = "curiosity",
    seed: int = 0,
    total_env_steps: int = 100_000,
    feature_dim: int = 50,
    hidden_dim: int = 1024,
    n_hidden: int = 2,
    utd_ratio: int = 1,
    batch_size: int = 256,
    synthetic_ratio: float = 0.5,
    real_buffer_size: int = 1_000_000,
    syn_buffer_size: int = 1_000_000,
    inner_loop_freq: int = 10_000,
    diffusion_train_steps: int = 50_000,
    diffusion_batch_size: int = 256,
    top_k_ratio: float = 0.1,
    guidance_scale: float = 1.5,
    p_uncond: float = 0.25,
    n_diffusion_steps: int = 1000,
    seed_steps: int = 5_000,
    eval_freq: int = 5_000,
    eval_episodes: int = 10,
    image_size: int = 84,
    frame_stack: int = 3,
    device: str = "cuda",
    log_dir: str = "logs",
    use_wandb: bool = False,
):
    """PGR training for pixel-based observations.

    Generates data in the latent space of the CNN visual encoder, following
    Lu et al. (2024) and Esser et al. (2021). Given encoder f_θ and transition
    (s, a, s', r), learns to generate (f_θ(s), a, f_θ(s'), r).
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    env = make_env(env_name, seed=seed, pixel_obs=True, image_size=image_size, frame_stack=frame_stack)
    eval_env = make_env(env_name, seed=seed + 100, pixel_obs=True, image_size=image_size, frame_stack=frame_stack)

    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    latent_dim = feature_dim
    transition_dim = 2 * latent_dim + action_dim + 1

    logger = Logger(
        log_dir=os.path.join(log_dir, f"pgr_pixel_{relevance_type}_{env_name}_seed{seed}"),
        use_wandb=use_wandb,
        config={"env": env_name, "relevance": relevance_type, "seed": seed, "pixel": True},
    )

    pixel_buffer = PixelReplayBuffer(obs_shape, action_dim, real_buffer_size, str(device))
    latent_real_buffer = LatentReplayBuffer(latent_dim, action_dim, real_buffer_size, str(device))
    latent_syn_buffer = LatentReplayBuffer(latent_dim, action_dim, syn_buffer_size, str(device))
    mixed_buffer = MixedReplayBuffer(latent_real_buffer, latent_syn_buffer, synthetic_ratio)

    agent = DRQv2(
        obs_shape=obs_shape,
        action_dim=action_dim,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        device=str(device),
    )

    relevance_fn = build_relevance_fn(
        relevance_type,
        state_dim=latent_dim,
        action_dim=action_dim,
        agent=agent,
        device=str(device),
        use_cnn=False,
    )

    diffusion = ConditionalDiffusion(
        transition_dim=transition_dim,
        hidden_dim=256,
        n_hidden_layers=4,
        time_embed_dim=128,
        cond_embed_dim=128,
        n_diffusion_steps=n_diffusion_steps,
        p_uncond=p_uncond,
        guidance_scale=guidance_scale,
    ).to(device)

    diffusion_optimizer = torch.optim.Adam(diffusion.parameters(), lr=3e-4)
    normalizer = TransitionNormalizer()

    obs, _ = env.reset()
    episode_return = 0.0
    diffusion_trained = False

    for step in range(total_env_steps):
        if step < seed_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += reward

        pixel_buffer.add(obs, action, next_obs, reward, float(terminated))

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device) / 255.0
        next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            latent_s = agent.encoder(obs_t).cpu().numpy()[0]
            latent_sp = agent.encoder(next_obs_t).cpu().numpy()[0]

        latent_real_buffer.add(latent_s, action, latent_sp, reward, float(terminated))

        obs = next_obs
        if done:
            logger.log({"train/episode_return": episode_return}, step)
            obs, _ = env.reset()
            episode_return = 0.0

        if step < seed_steps:
            continue

        if latent_real_buffer.size >= batch_size:
            rel_idx = latent_real_buffer.ptr - 1
            rel_batch = {
                "states": torch.FloatTensor(latent_real_buffer.states[rel_idx: rel_idx + 1]).to(device),
                "actions": torch.FloatTensor(latent_real_buffer.actions[rel_idx: rel_idx + 1]).to(device),
                "next_states": torch.FloatTensor(latent_real_buffer.next_states[rel_idx: rel_idx + 1]).to(device),
                "rewards": torch.FloatTensor(latent_real_buffer.rewards[rel_idx: rel_idx + 1]).to(device),
                "dones": torch.FloatTensor(latent_real_buffer.dones[rel_idx: rel_idx + 1]).to(device),
            }
            with torch.no_grad():
                rel_val = relevance_fn.compute(rel_batch).cpu().numpy()
            latent_real_buffer.update_relevance(np.array([rel_idx % latent_real_buffer.max_size]), rel_val)

        if (step + 1) % inner_loop_freq == 0 and latent_real_buffer.size >= batch_size:
            latent_real_buffer.update_all_relevance(relevance_fn.compute, batch_size=512)

            transitions_arr = latent_real_buffer.as_transitions_array()
            normalizer.fit(transitions_arr)

            diffusion_loss = train_diffusion_model(
                diffusion=diffusion,
                real_buffer=latent_real_buffer,
                normalizer=normalizer,
                state_dim=latent_dim,
                action_dim=action_dim,
                top_k_ratio=top_k_ratio,
                n_steps=diffusion_train_steps,
                batch_size=diffusion_batch_size,
                device=str(device),
                optimizer=diffusion_optimizer,
            )
            logger.log({"diffusion/train_loss": diffusion_loss}, step)

            latent_syn_buffer.ptr = 0
            latent_syn_buffer.size = 0
            generate_synthetic_transitions(
                diffusion=diffusion,
                real_buffer=latent_real_buffer,
                syn_buffer=latent_syn_buffer,
                normalizer=normalizer,
                state_dim=latent_dim,
                action_dim=action_dim,
                n_generate=syn_buffer_size,
                top_k_ratio=top_k_ratio,
                guidance_scale=guidance_scale,
                device=str(device),
            )
            diffusion_trained = True

        if diffusion_trained and latent_real_buffer.size >= batch_size:
            for _ in range(utd_ratio):
                batch = mixed_buffer.sample(batch_size)
                logs = agent.update_from_latent(batch)
            logger.log(logs, step)
        elif latent_real_buffer.size >= batch_size:
            batch = pixel_buffer.sample(batch_size)
            logs = agent.update(batch)

        if (step + 1) % eval_freq == 0:
            eval_return = evaluate_policy(agent, eval_env, eval_episodes)
            logger.log({"eval/episode_return": eval_return}, step)
            print(f"[PGR-Pixel] Step {step + 1}/{total_env_steps} | Eval: {eval_return:.2f}")

    env.close()
    eval_env.close()
    logger.close()
    return agent, diffusion


def main():
    parser = argparse.ArgumentParser(description="Prioritized Generative Replay (PGR)")
    parser.add_argument("--env", type=str, default="quadruped-walk")
    parser.add_argument("--relevance", type=str, default="curiosity",
                        choices=["curiosity", "td_error", "return", "reward", "rnd", "cts", "eco"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_env_steps", type=int, default=100_000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--utd_ratio", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--synthetic_ratio", type=float, default=0.5)
    parser.add_argument("--real_buffer_size", type=int, default=1_000_000)
    parser.add_argument("--syn_buffer_size", type=int, default=1_000_000)
    parser.add_argument("--inner_loop_freq", type=int, default=10_000)
    parser.add_argument("--diffusion_train_steps", type=int, default=50_000)
    parser.add_argument("--diffusion_batch_size", type=int, default=256)
    parser.add_argument("--top_k_ratio", type=float, default=0.1)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--p_uncond", type=float, default=0.25)
    parser.add_argument("--n_diffusion_steps", type=int, default=1000)
    parser.add_argument("--seed_steps", type=int, default=5_000)
    parser.add_argument("--eval_freq", type=int, default=5_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--pixel", action="store_true", help="Use pixel-based observations")
    parser.add_argument("--feature_dim", type=int, default=50, help="CNN encoder feature dim (pixel mode)")
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--frame_stack", type=int, default=3)
    args = parser.parse_args()

    if args.pixel:
        run_pgr_pixel(
            env_name=args.env,
            relevance_type=args.relevance,
            seed=args.seed,
            total_env_steps=args.total_env_steps,
            feature_dim=args.feature_dim,
            hidden_dim=1024,
            n_hidden=args.n_hidden,
            utd_ratio=args.utd_ratio,
            batch_size=args.batch_size,
            synthetic_ratio=args.synthetic_ratio,
            real_buffer_size=args.real_buffer_size,
            syn_buffer_size=args.syn_buffer_size,
            inner_loop_freq=args.inner_loop_freq,
            diffusion_train_steps=args.diffusion_train_steps,
            diffusion_batch_size=args.diffusion_batch_size,
            top_k_ratio=args.top_k_ratio,
            guidance_scale=args.guidance_scale,
            p_uncond=args.p_uncond,
            n_diffusion_steps=args.n_diffusion_steps,
            seed_steps=args.seed_steps,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            image_size=args.image_size,
            frame_stack=args.frame_stack,
            device=args.device,
            log_dir=args.log_dir,
            use_wandb=args.use_wandb,
        )
    else:
        run_pgr(
            env_name=args.env,
            relevance_type=args.relevance,
            seed=args.seed,
            total_env_steps=args.total_env_steps,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
            utd_ratio=args.utd_ratio,
            batch_size=args.batch_size,
            synthetic_ratio=args.synthetic_ratio,
            real_buffer_size=args.real_buffer_size,
            syn_buffer_size=args.syn_buffer_size,
            inner_loop_freq=args.inner_loop_freq,
            diffusion_train_steps=args.diffusion_train_steps,
            diffusion_batch_size=args.diffusion_batch_size,
            top_k_ratio=args.top_k_ratio,
            guidance_scale=args.guidance_scale,
            p_uncond=args.p_uncond,
            n_diffusion_steps=args.n_diffusion_steps,
            seed_steps=args.seed_steps,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            device=args.device,
            log_dir=args.log_dir,
            use_wandb=args.use_wandb,
            pixel_obs=args.pixel,
        )


if __name__ == "__main__":
    main()
