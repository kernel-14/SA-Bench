"""
Prioritized Generative Replay (PGR) Trainer.

Implements Algorithm 1 from the paper:
- Outer loop: collect real transitions, update relevance function F
- Inner loop (every T steps): train diffusion model, generate synthetic data, train policy

The key components are:
1. Real replay buffer D_real
2. Synthetic replay buffer D_syn
3. Conditional diffusion model G
4. Relevance function F (curiosity by default)
5. RL policy pi (REDQ by default)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from collections import deque

from .diffusion import ConditionalDiffusion
from .relevance import CuriosityRelevance, ReturnRelevance, TDErrorRelevance, RewardRelevance
from .replay_buffer import NormalizedReplayBuffer
from .redq import REDQAgent


class PGRTrainer:
    """
    Main PGR training loop implementing Algorithm 1.

    Hyperparameters from paper:
    - D_real, D_syn: 1M transitions each
    - Synthetic data ratio r = 0.5 (50% real, 50% synthetic per batch)
    - Batch size = 256
    - UTD ratio = 20
    - Inner loop frequency: every 10K environment steps
    - Top-k ratio for prompting: k = 0.5 * buffer_size (top 50%)
    - p_uncond = 0.25 (CFG dropout probability)
    - Guidance scale omega = 1.2
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        # PGR hyperparameters
        relevance_type: str = "curiosity",  # "curiosity", "return", "td_error", "reward"
        synthetic_ratio: float = 0.5,
        batch_size: int = 256,
        utd_ratio: int = 20,
        inner_loop_freq: int = 10_000,
        top_k_ratio: float = 0.5,
        # Diffusion hyperparameters
        diffusion_hidden_dim: int = 256,
        diffusion_n_layers: int = 4,
        diffusion_n_timesteps: int = 100,
        diffusion_lr: float = 3e-4,
        diffusion_batch_size: int = 256,
        diffusion_train_steps: int = 200_000,
        p_uncond: float = 0.25,
        guidance_scale: float = 1.2,
        # REDQ hyperparameters
        rl_hidden_dim: int = 256,
        rl_n_layers: int = 2,
        n_q_networks: int = 10,
        n_target_q: int = 2,
        rl_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        # Curiosity hyperparameters
        curiosity_feature_dim: int = 64,
        curiosity_hidden_dim: int = 256,
        curiosity_lr: float = 3e-4,
        curiosity_update_freq: int = 20,  # update every N policy steps (5% of updates)
        # Buffer sizes
        real_buffer_size: int = 1_000_000,
        syn_buffer_size: int = 1_000_000,
        n_syn_samples: int = 1_000_000,
        # Misc
        device: str = "cpu",
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.relevance_type = relevance_type
        self.synthetic_ratio = synthetic_ratio
        self.batch_size = batch_size
        self.utd_ratio = utd_ratio
        self.inner_loop_freq = inner_loop_freq
        self.top_k_ratio = top_k_ratio
        self.diffusion_train_steps = diffusion_train_steps
        self.diffusion_batch_size = diffusion_batch_size
        self.curiosity_update_freq = curiosity_update_freq
        self.n_syn_samples = n_syn_samples
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Transition dimension: s + a + s' + r
        self.transition_dim = obs_dim + action_dim + obs_dim + 1

        # ---- Replay buffers ----
        self.real_buffer = NormalizedReplayBuffer(
            obs_dim, action_dim, real_buffer_size, device
        )
        self.syn_buffer = NormalizedReplayBuffer(
            obs_dim, action_dim, syn_buffer_size, device
        )

        # ---- Diffusion model ----
        self.diffusion = ConditionalDiffusion(
            transition_dim=self.transition_dim,
            hidden_dim=diffusion_hidden_dim,
            n_layers=diffusion_n_layers,
            n_timesteps=diffusion_n_timesteps,
            p_uncond=p_uncond,
            guidance_scale=guidance_scale,
        ).to(self.device)
        self.diffusion_optimizer = torch.optim.Adam(
            self.diffusion.parameters(), lr=diffusion_lr
        )

        # ---- RL agent ----
        self.agent = REDQAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=rl_hidden_dim,
            n_layers=rl_n_layers,
            n_q_networks=n_q_networks,
            n_target_q=n_target_q,
            lr_actor=rl_lr,
            lr_critic=rl_lr,
            gamma=gamma,
            tau=tau,
            utd_ratio=utd_ratio,
            device=device,
        )

        # ---- Relevance function ----
        self.relevance_fn = self._build_relevance_fn(
            relevance_type,
            obs_dim,
            action_dim,
            curiosity_feature_dim,
            curiosity_hidden_dim,
        )
        if hasattr(self.relevance_fn, 'parameters'):
            params = list(self.relevance_fn.parameters())
            if params:
                self.relevance_optimizer = torch.optim.Adam(params, lr=curiosity_lr)
            else:
                self.relevance_optimizer = None
        else:
            self.relevance_optimizer = None

        # ---- Tracking ----
        self.total_env_steps = 0
        self.total_policy_updates = 0
        self.episode_rewards = deque(maxlen=100)
        self.metrics = {
            "episode_reward": [],
            "q_loss": [],
            "actor_loss": [],
            "diffusion_loss": [],
            "relevance_loss": [],
            "dormant_ratio": [],
        }

    def _build_relevance_fn(self, relevance_type, obs_dim, action_dim, feature_dim, hidden_dim):
        if relevance_type == "curiosity":
            fn = CuriosityRelevance(
                obs_dim=obs_dim,
                action_dim=action_dim,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
            ).to(self.device)
        elif relevance_type == "return":
            fn = ReturnRelevance()
        elif relevance_type == "td_error":
            fn = TDErrorRelevance(gamma=self.agent.gamma if hasattr(self, 'agent') else 0.99)
        elif relevance_type == "reward":
            fn = RewardRelevance()
        else:
            raise ValueError(f"Unknown relevance type: {relevance_type}")
        return fn

    def compute_relevance(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute relevance values for a batch of transitions."""
        obs = batch["obs"]
        action = batch["action"]
        next_obs = batch["next_obs"]
        reward = batch["reward"]
        done = batch["done"]

        if self.relevance_type == "curiosity":
            return self.relevance_fn.compute_relevance(obs, action, next_obs)
        elif self.relevance_type == "return":
            return self.relevance_fn.compute_relevance(
                obs, self.agent.q_networks[0], self.agent.actor
            )
        elif self.relevance_type == "td_error":
            return self.relevance_fn.compute_relevance(
                obs, action, next_obs, reward, done,
                self.agent.q_networks[0], self.agent.q_target_networks[0]
            )
        elif self.relevance_type == "reward":
            return self.relevance_fn.compute_relevance(reward)
        else:
            raise ValueError(f"Unknown relevance type: {self.relevance_type}")

    def update_relevance_fn(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update the relevance function parameters (for curiosity)."""
        if self.relevance_optimizer is None:
            return 0.0

        obs = batch["obs"]
        action = batch["action"]
        next_obs = batch["next_obs"]

        loss = self.relevance_fn.compute_loss(obs, action, next_obs)
        self.relevance_optimizer.zero_grad()
        loss.backward()
        self.relevance_optimizer.step()
        return loss.item()

    def train_diffusion(self, n_steps: int) -> float:
        """
        Train the conditional diffusion model on real buffer data.
        Inner loop step 5 in Algorithm 1.
        """
        self.real_buffer.update_normalization()
        total_loss = 0.0

        for _ in range(n_steps):
            # Sample normalized transitions with relevance values
            x, cond = self.real_buffer.sample_normalized(self.diffusion_batch_size)
            x = x.to(self.device)
            cond = cond.to(self.device)

            loss = self.diffusion.compute_loss(x, cond)
            self.diffusion_optimizer.zero_grad()
            loss.backward()
            self.diffusion_optimizer.step()
            total_loss += loss.item()

        return total_loss / max(n_steps, 1)

    def generate_synthetic_data(self, n_samples: int) -> None:
        """
        Generate synthetic transitions using the diffusion model.
        Inner loop step 6 in Algorithm 1.

        Uses the 'prompting' strategy: sample conditioning values from
        the top-k highest relevance transitions in D_real.
        """
        self.diffusion.eval()

        # Prompting strategy (Section 4.3): use top-k relevance values as conditions
        k = max(1, int(self.top_k_ratio * self.real_buffer.size))
        _, top_k_relevance = self.real_buffer.sample_top_k_normalized(k)

        # Generate in batches
        batch_size = min(1024, n_samples)
        all_transitions = []

        remaining = n_samples
        while remaining > 0:
            curr_batch = min(batch_size, remaining)

            # Sample conditioning values from top-k relevance
            cond_idx = torch.randint(0, k, (curr_batch,))
            cond = top_k_relevance[cond_idx].to(self.device)

            # Generate normalized transitions
            with torch.no_grad():
                gen_normalized = self.diffusion.sample(
                    curr_batch, cond=cond, device=self.device
                )

            all_transitions.append(gen_normalized.cpu().numpy())
            remaining -= curr_batch

        gen_normalized = np.concatenate(all_transitions, axis=0)

        # Denormalize
        gen_flat = self.real_buffer.denormalize(gen_normalized)

        # Split into components
        obs_dim = self.obs_dim
        action_dim = self.action_dim

        obs = gen_flat[:, :obs_dim]
        action = gen_flat[:, obs_dim:obs_dim + action_dim]
        next_obs = gen_flat[:, obs_dim + action_dim:2 * obs_dim + action_dim]
        reward = gen_flat[:, -1:]
        done = np.zeros((n_samples, 1), dtype=np.float32)

        # Add to synthetic buffer
        self.syn_buffer.add_batch(obs, action, next_obs, reward, done)

        self.diffusion.train()

    def sample_mixed_batch(self) -> Dict[str, torch.Tensor]:
        """
        Sample a mixed batch of real and synthetic transitions.
        Ratio r controls the fraction of synthetic data (default 0.5).
        """
        n_syn = int(self.batch_size * self.synthetic_ratio)
        n_real = self.batch_size - n_syn

        real_batch = self.real_buffer.sample(n_real)

        if n_syn > 0 and self.syn_buffer.size > 0:
            syn_batch = self.syn_buffer.sample(min(n_syn, self.syn_buffer.size))
            # Pad if needed
            if syn_batch["obs"].shape[0] < n_syn:
                extra = self.real_buffer.sample(n_syn - syn_batch["obs"].shape[0])
                syn_batch = {k: torch.cat([syn_batch[k], extra[k]], dim=0) for k in syn_batch}

            batch = {
                k: torch.cat([real_batch[k], syn_batch[k]], dim=0)
                for k in real_batch
            }
        else:
            batch = real_batch

        return batch

    def update_real_buffer_relevance(self, batch_size: int = 1024):
        """
        Recompute and update relevance values for all transitions in D_real.
        Called in the outer loop (Algorithm 1, line 3).
        """
        if self.real_buffer.size == 0:
            return

        # Process in batches to avoid OOM
        n = self.real_buffer.size
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = np.arange(start, end)
            batch = self.real_buffer._get_batch(idx)
            relevance = self.compute_relevance(batch)
            self.real_buffer.update_relevance(idx, relevance.cpu().numpy())

    def collect_transition(self, env, obs: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Collect a single transition from the environment."""
        action = self.agent.select_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        terminated = done or truncated

        # Compute relevance for this transition
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
        reward_t = torch.FloatTensor([[reward]]).to(self.device)

        batch = {
            "obs": obs_t, "action": action_t, "next_obs": next_obs_t,
            "reward": reward_t, "done": torch.FloatTensor([[float(terminated)]]).to(self.device)
        }
        relevance = self.compute_relevance(batch).item()

        self.real_buffer.add(obs, action, next_obs, reward, float(terminated), relevance)
        self.total_env_steps += 1

        return next_obs, reward, terminated, info

    def train(
        self,
        env,
        eval_env,
        total_env_steps: int = 100_000,
        eval_freq: int = 5_000,
        n_eval_episodes: int = 10,
        warmup_steps: int = 5_000,
        log_freq: int = 1_000,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        Main training loop implementing Algorithm 1.

        Args:
            env: training environment
            eval_env: evaluation environment
            total_env_steps: total environment interactions (100K in paper)
            eval_freq: evaluate every N steps
            n_eval_episodes: number of episodes for evaluation
            warmup_steps: random exploration before training
            log_freq: logging frequency
        """
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        print(f"Starting PGR training with {self.relevance_type} relevance function")
        print(f"Total env steps: {total_env_steps}, UTD: {self.utd_ratio}")
        print(f"Diffusion model params: {self.diffusion.num_parameters():,}")

        while self.total_env_steps < total_env_steps:
            # ---- Outer loop: collect real data ----
            if self.total_env_steps < warmup_steps:
                # Random exploration during warmup
                action = env.action_space.sample()
                next_obs, reward, done, truncated, info = env.step(action)
                terminated = done or truncated
                self.real_buffer.add(obs, action, next_obs, reward, float(terminated))
                self.total_env_steps += 1
            else:
                next_obs, reward, terminated, info = self.collect_transition(env, obs)

            episode_reward += reward
            episode_steps += 1

            if terminated or episode_steps >= 1000:
                self.episode_rewards.append(episode_reward)
                obs, _ = env.reset()
                episode_reward = 0.0
                episode_steps = 0
            else:
                obs = next_obs

            # ---- Update relevance function (outer loop, line 3) ----
            if (self.total_env_steps >= warmup_steps and
                    self.real_buffer.size >= self.batch_size and
                    self.relevance_type == "curiosity"):
                batch = self.real_buffer.sample(self.batch_size)
                rel_loss = self.update_relevance_fn(batch)

            # ---- Inner loop: train diffusion + policy ----
            if (self.total_env_steps >= warmup_steps and
                    self.real_buffer.size >= self.batch_size and
                    self.total_env_steps % self.inner_loop_freq == 0):

                print(f"\n[Step {self.total_env_steps}] Running inner loop...")

                # Step 5: Train diffusion model
                diff_loss = self.train_diffusion(self.diffusion_train_steps)
                print(f"  Diffusion loss: {diff_loss:.4f}")

                # Step 6: Generate synthetic data
                self.generate_synthetic_data(self.n_syn_samples)
                print(f"  Generated {self.n_syn_samples} synthetic transitions")

            # ---- Policy update (inner loop, step 7) ----
            if (self.total_env_steps >= warmup_steps and
                    self.real_buffer.size >= self.batch_size):

                for _ in range(self.utd_ratio):
                    batch = self.sample_mixed_batch()
                    rl_metrics = self.agent.update(batch)
                    self.total_policy_updates += 1

                    # Update curiosity every curiosity_update_freq steps (5% of updates)
                    if (self.relevance_type == "curiosity" and
                            self.total_policy_updates % self.curiosity_update_freq == 0):
                        real_batch = self.real_buffer.sample(self.batch_size)
                        self.update_relevance_fn(real_batch)

            # ---- Evaluation ----
            if self.total_env_steps % eval_freq == 0 and self.total_env_steps >= warmup_steps:
                eval_reward = self.evaluate(eval_env, n_eval_episodes)
                print(f"[Step {self.total_env_steps}] Eval reward: {eval_reward:.2f}")
                self.metrics["episode_reward"].append((self.total_env_steps, eval_reward))

                if save_dir:
                    self.save(os.path.join(save_dir, f"checkpoint_{self.total_env_steps}"))

        return self.metrics

    def evaluate(self, env, n_episodes: int = 10) -> float:
        """Evaluate the current policy."""
        total_reward = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            total_reward += ep_reward
        return total_reward / n_episodes

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.agent.save(os.path.join(path, "agent.pt"))
        torch.save(self.diffusion.state_dict(), os.path.join(path, "diffusion.pt"))
        if self.relevance_optimizer is not None:
            torch.save(self.relevance_fn.state_dict(), os.path.join(path, "relevance.pt"))
        np.save(os.path.join(path, "metrics.npy"), self.metrics)

    def load(self, path: str):
        self.agent.load(os.path.join(path, "agent.pt"))
        self.diffusion.load_state_dict(
            torch.load(os.path.join(path, "diffusion.pt"), map_location=self.device)
        )
        rel_path = os.path.join(path, "relevance.pt")
        if os.path.exists(rel_path):
            self.relevance_fn.load_state_dict(
                torch.load(rel_path, map_location=self.device)
            )
