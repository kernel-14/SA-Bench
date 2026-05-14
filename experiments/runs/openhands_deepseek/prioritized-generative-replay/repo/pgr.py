"""Prioritized Generative Replay (PGR) - Algorithm 1 from the paper.

Implements the outer-loop / inner-loop framework for PGR.
For pixel-based tasks, the diffusion model operates in the latent space
of the policy's CNN visual encoder (per SynthER design).
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from models.diffusion import ConditionalDiffusionModel
from models.policy import REDQPolicy, SACPolicy, DRQv2Policy
from models.curiosity import (
    ICM, PixelICM, RND, CTSRelevance, ECORelevance
)
from replay import ReplayBuffer, SyntheticBuffer, Transition
from config import RunConfig


class PGR:
    """Prioritized Generative Replay framework.

    Outer loop: agent interacts with environment, collects real data.
    Inner loop: train conditional diffusion model, generate synthetic data,
                train policy on mixed real+synthetic data.
    """

    def __init__(self, config: RunConfig, state_dim: int, action_dim: int, is_pixel: bool = False):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_pixel = is_pixel
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # For pixel-based: diffusion operates on latent states
        # The latent_dim from config is the diffusion state_dim
        self.latent_dim = config.policy.latent_dim if is_pixel else state_dim
        self.transition_dim = 2 * self.latent_dim + action_dim + 1

        # Initialize policy
        if is_pixel:
            self.policy = DRQv2Policy(
                state_latent_dim=config.policy.latent_dim,
                action_dim=action_dim,
                hidden_dim=config.policy.hidden_dims,
                n_layers=config.policy.n_hidden_layers,
                gamma=config.policy.gamma,
                tau=config.policy.tau,
                image_size=config.policy.image_size,
                image_channels=config.policy.image_channels,
                aug=config.policy.image_augmentation,
            ).to(self.device)
            self._visual_encoder = self.policy.encoder
        else:
            if config.noisy_nets:
                noisy = True
            else:
                noisy = False

            if config.bootstrapped_q:
                self.policy = REDQPolicy(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dim=config.policy.hidden_dims,
                    n_layers=config.policy.n_hidden_layers,
                    n_critics=config.policy.n_critics,
                    n_target_critics=config.policy.n_target_critics,
                    gamma=config.policy.gamma,
                    tau=config.policy.tau,
                    noisy=noisy,
                    bootstrapped=True,
                ).to(self.device)
            else:
                self.policy = REDQPolicy(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dim=config.policy.hidden_dims,
                    n_layers=config.policy.n_hidden_layers,
                    n_critics=config.policy.n_critics,
                    n_target_critics=config.policy.n_target_critics,
                    gamma=config.policy.gamma,
                    tau=config.policy.tau,
                    noisy=noisy,
                ).to(self.device)

        # Policy optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=config.policy.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            sum([list(c.parameters()) for c in self.policy.critics], []),
            lr=config.policy.critic_lr,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.policy.log_alpha], lr=config.policy.actor_lr
        )

        # Initialize relevance function
        self.relevance_fn = self._build_relevance_fn(config)

        # Initialize diffusion model (operates on latents for pixel-based)
        self.diffusion = ConditionalDiffusionModel(
            state_dim=self.latent_dim,
            action_dim=action_dim,
            n_timesteps=config.diffusion.n_timesteps,
            model_dims=config.diffusion.model_dims,
            n_residual_blocks=config.diffusion.n_residual_blocks,
            block_dims=config.diffusion.block_dims,
            time_emb_dims=config.diffusion.time_emb_dims,
            cond_emb_dims=config.diffusion.cond_emb_dims,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            p_uncond=config.diffusion.p_uncond,
        ).to(self.device)

        self.diffusion_optimizer = torch.optim.Adam(
            self.diffusion.parameters(), lr=config.diffusion.lr
        )

        # Replay buffers
        self.real_buffer = ReplayBuffer(
            capacity=config.replay.real_buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            pixel_based=is_pixel,
        )
        self.syn_buffer = SyntheticBuffer(
            capacity=config.replay.syn_buffer_capacity,
            transition_dim=self.transition_dim,
        )
        self.syn_buffer.set_split_dims(self.latent_dim, action_dim)

        # Training state
        self.total_env_steps = 0
        self.total_policy_updates = 0
        self.inner_loop_counter = 0
        self.current_episode_return = 0.0
        self.episode_returns = []

    def _build_relevance_fn(self, config: RunConfig):
        """Build the relevance function F based on config."""
        fn_type = config.relevance_fn

        if fn_type == "curiosity":
            if self.is_pixel:
                return PixelICM(
                    action_dim=self.action_dim,
                    feature_dim=config.curiosity.feature_dim,
                    hidden_dim=config.curiosity.hidden_dim,
                    image_channels=config.policy.image_channels,
                    image_size=config.policy.image_size,
                    lr=config.curiosity.lr,
                    forward_loss_weight=config.curiosity.forward_loss_weight,
                    inverse_loss_weight=config.curiosity.inverse_loss_weight,
                    intrinsic_reward_weight=config.curiosity.intrinsic_reward_weight,
                ).to(self.device)
            else:
                return ICM(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    feature_dim=config.curiosity.feature_dim,
                    hidden_dim=config.curiosity.hidden_dim,
                    lr=config.curiosity.lr,
                    forward_loss_weight=config.curiosity.forward_loss_weight,
                    inverse_loss_weight=config.curiosity.inverse_loss_weight,
                    intrinsic_reward_weight=config.curiosity.intrinsic_reward_weight,
                ).to(self.device)

        elif fn_type == "rnd":
            return RND(
                input_dim=self.state_dim,
                feature_dim=config.curiosity.rnd_feature_dim,
                bottleneck=config.curiosity.rnd_bottleneck,
                lr=config.curiosity.rnd_lr,
                pixel_based=self.is_pixel,
            ).to(self.device)

        elif fn_type == "cts":
            return CTSRelevance(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                context_bins=config.curiosity.cts_context_bins,
            )

        elif fn_type == "eco":
            return ECORelevance(
                state_dim=self.state_dim,
                feature_dim=config.curiosity.feature_dim,
                memory_size=config.curiosity.eco_memory_size,
                alpha=config.curiosity.eco_alpha,
                beta=config.curiosity.eco_beta,
                percentile=config.curiosity.eco_percentile,
                lr=config.curiosity.lr,
            ).to(self.device)

        else:
            raise ValueError(f"Unknown relevance function: {fn_type}")

    def compute_relevance(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        is_latent: bool = False,
    ) -> Optional[torch.Tensor]:
        """Compute relevance value c = F(tau) for transitions.

        Args:
            states: current states or latent states
            actions: actions
            rewards: rewards
            next_states: next states or latent next states
            is_latent: if True, states/next_states are already in latent space
                       (for pixel-based, these are latent representations from encoder)
        """
        fn_type = self.config.relevance_fn

        if fn_type == "curiosity":
            if self.is_pixel:
                if is_latent:
                    # For latent states: sample raw observations from buffer for relevance
                    # This is handled in train_diffusion separately
                    return None
                return self.relevance_fn.compute_relevance(states, actions, next_states)
            return self.relevance_fn.compute_relevance(states, actions, next_states)

        elif fn_type == "rnd":
            return self.relevance_fn.compute_relevance(states, actions, next_states)

        elif fn_type == "cts":
            return self.relevance_fn.compute_relevance(states, actions, next_states)

        elif fn_type == "eco":
            return self.relevance_fn.compute_relevance(states, actions, next_states)

        elif fn_type == "td_error":
            # Eq. (4): TD-error as relevance
            with torch.no_grad():
                next_actions, next_log_probs, _ = self.policy.actor.sample(next_states)
                q1 = self.policy.critics[0](next_states, next_actions)
                q2 = self.policy.critics[1](next_states, next_actions)
                q_next = torch.min(q1, q2)
                q_current = self.policy.critics[0](states, actions)
                td = rewards + self.config.policy.gamma * q_next - q_current
                return td.abs()

        elif fn_type == "return":
            # Eq. (3): Q-value as relevance
            with torch.no_grad():
                _, _, mean_action = self.policy.actor.sample(states)
                q_value = self.policy.critics[0](states, mean_action)
                return q_value

        elif fn_type == "reward":
            return rewards

    def update_relevance_fn(self, batch_size: int = 256):
        """Update relevance function parameters using real buffer data."""
        if len(self.real_buffer) < batch_size:
            return

        fn_type = self.config.relevance_fn

        if fn_type == "curiosity":
            states, actions, rewards, next_states, _ = self.real_buffer.sample(
                batch_size, self.device
            )
            self.relevance_fn.update(states, actions, next_states)

        elif fn_type == "rnd":
            _, _, _, next_states, _ = self.real_buffer.sample(batch_size, self.device)
            self.relevance_fn.update(next_states)

    def sample_relevance_condition(self, batch_size: int) -> Optional[torch.Tensor]:
        """Sample relevance values for CFG conditioning.

        Uses the 'prompting' strategy from Section 4.3:
        - Take top-k transitions from D_real by relevance value
        - Randomly sample from their relevance values
        """
        if self.config.relevance_fn in ("return", "td_error", "reward"):
            return None  # These require Q-function, use real buffer

        if len(self.real_buffer) < batch_size:
            return None

        # Sample transitions and compute relevance
        states, actions, rewards, next_states, _ = self.real_buffer.sample(
            min(len(self.real_buffer), 2048), self.device
        )
        relevance_values = self.compute_relevance(states, actions, rewards, next_states)

        # Get top-k
        k = max(1, int(len(relevance_values) * self.config.diffusion.top_k_ratio))
        _, top_indices = torch.topk(relevance_values.squeeze(-1), k)
        top_relevances = relevance_values[top_indices]

        # Randomly sample from top-k
        if len(top_relevances) >= batch_size:
            sampled = top_relevances[torch.randperm(len(top_relevances))[:batch_size]]
        else:
            # Repeat if not enough
            repeats = batch_size // len(top_relevances) + 1
            sampled = top_relevances.repeat(repeats)[:batch_size]

        return sampled

    def train_diffusion(self):
        """Inner loop: Train conditional diffusion model on real buffer (step 5).

        For pixel-based tasks, encodes raw pixel observations to latents on-the-fly
        for diffusion training (following SynthER design: diffusion operates in latent space).
        """
        batch_size = self.config.diffusion.batch_size

        if len(self.real_buffer) < batch_size:
            return

        n_steps = self.config.diffusion.n_grad_steps_per_loop
        for step in range(n_steps):
            # Sample real transitions
            states, actions, rewards, next_states, _ = self.real_buffer.sample(
                batch_size, self.device
            )

            if self.is_pixel:
                # Encode pixel observations to latent representations
                with torch.no_grad():
                    latent_states = self._visual_encoder(states)
                    latent_next_states = self._visual_encoder(next_states)
                x_start = torch.cat([latent_states, actions, latent_next_states, rewards], dim=-1)
                # Compute relevance on raw pixels (for PixelICM)
                cond = self.compute_relevance(states, actions, rewards, next_states, is_latent=False)
            else:
                x_start = torch.cat([states, actions, next_states, rewards], dim=-1)
                cond = self.compute_relevance(states, actions, rewards, next_states)

            # Diffusion training loss (Eq. 2)
            loss = self.diffusion.training_loss(x_start, cond=cond)

            self.diffusion_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), 1.0)
            self.diffusion_optimizer.step()

    @torch.no_grad()
    def generate_synthetic_data(self, batch_size: int = 256):
        """Inner loop: Generate synthetic transitions with CFG (step 6).

        Generates transitions and adds them to synthetic buffer.
        """
        if len(self.real_buffer) < batch_size:
            return

        # Sample relevance conditions for guidance
        cond = self.sample_relevance_condition(batch_size)
        if cond is not None:
            cond = cond.to(self.device)

        # Generate transitions with CFG
        generated = self.diffusion.sample(
            batch_size=batch_size,
            cond=cond,
            guidance_scale=self.config.diffusion.guidance_scale,
            device=self.device,
        )

        # Add to synthetic buffer
        self.syn_buffer.add(generated)

    def update_policy(self):
        """Train policy on mixed real+synthetic data (step 7)."""
        batch_size = self.config.policy.batch_size
        real_batch = int(batch_size * (1.0 - self.config.policy.synthetic_ratio))
        syn_batch = batch_size - real_batch

        # Sample real data
        if len(self.real_buffer) >= real_batch and real_batch > 0:
            real_states, real_actions, real_rewards, real_nexts, real_dones = \
                self.real_buffer.sample(real_batch, self.device)
        else:
            real_states = real_actions = real_rewards = real_nexts = real_dones = None

        # Sample synthetic data
        if self.syn_buffer.size >= syn_batch and syn_batch > 0:
            syn_states, syn_actions, syn_rewards, syn_nexts = \
                self.syn_buffer.sample(syn_batch, self.device)
            syn_dones = torch.zeros(syn_batch, 1, device=self.device)
        else:
            syn_states = syn_actions = syn_rewards = syn_nexts = syn_dones = None

        # Combine real and synthetic
        if real_states is not None and syn_states is not None:
            if self.is_pixel:
                # Encode real pixel observations to latents; syn_states are already latents
                with torch.no_grad():
                    real_latent_states = self._visual_encoder(real_states)
                    real_latent_nexts = self._visual_encoder(real_nexts)
                states = torch.cat([real_latent_states, syn_states], dim=0)
                next_states = torch.cat([real_latent_nexts, syn_nexts], dim=0)
            else:
                states = torch.cat([real_states, syn_states], dim=0)
                next_states = torch.cat([real_nexts, syn_nexts], dim=0)
            actions = torch.cat([real_actions, syn_actions], dim=0)
            rewards = torch.cat([real_rewards, syn_rewards], dim=0)
            dones = torch.cat([real_dones, syn_dones], dim=0)
        elif real_states is not None:
            if self.is_pixel:
                with torch.no_grad():
                    states = self._visual_encoder(real_states)
                    next_states = self._visual_encoder(real_nexts)
            else:
                states = real_states
                next_states = real_nexts
            actions = real_actions
            rewards = real_rewards
            dones = real_dones
        elif syn_states is not None:
            states = syn_states
            actions = syn_actions
            rewards = syn_rewards
            next_states = syn_nexts
            dones = syn_dones
        else:
            return

        # REDQ update: UTD gradient steps
        utd = self.config.scaling_utd if self.config.scaling_utd > 0 else self.config.policy.utd
        for _ in range(utd):
            # Critic update
            critic_loss = self.policy.critic_loss(states, actions, rewards, next_states, dones)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor update
            actor_loss, alpha_loss = self.policy.actor_loss(states)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Update target networks
            self.policy.update_targets()

            self.total_policy_updates += 1

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.policy.get_action(state_tensor, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def observe(self, state, action, reward, next_state, done):
        """Store transition in real buffer and update counters."""
        self.real_buffer.push(state, action, reward, next_state, done)
        self.total_env_steps += 1
        self.current_episode_return += reward

    def end_episode(self):
        """Record episode return and reset."""
        self.episode_returns.append(self.current_episode_return)
        self.current_episode_return = 0.0

    def should_update(self) -> bool:
        """Check if inner loop should run (every 10K env steps per Section 4.3/D)."""
        return (
            self.total_env_steps > 0
            and self.total_env_steps % self.config.replay.inner_loop_frequency == 0
            and self.total_env_steps > self.inner_loop_counter
        )

    def run_inner_loop(self):
        """Execute inner loop: train diffusion, generate synthetic data."""
        print(f"[PGR] Running inner loop at step {self.total_env_steps}")

        # Update relevance function with real data (step 3)
        if self.config.relevance_fn in ("curiosity", "rnd"):
            for _ in range(100):  # 5% of policy steps
                self.update_relevance_fn(batch_size=self.config.diffusion.batch_size)

        # Train diffusion model (step 5)
        self.train_diffusion()

        # Generate synthetic data and fill buffer (step 6)
        n_generations = self.config.replay.syn_buffer_capacity // self.config.diffusion.batch_size
        for _ in range(n_generations):
            self.generate_synthetic_data(batch_size=self.config.diffusion.batch_size)

        self.inner_loop_counter = self.total_env_steps

    def get_metrics(self) -> Dict[str, Any]:
        """Return training metrics."""
        avg_return = (
            np.mean(self.episode_returns[-10:]) if self.episode_returns else 0.0
        )
        return {
            "total_env_steps": self.total_env_steps,
            "total_policy_updates": self.total_policy_updates,
            "avg_return_last_10": avg_return,
            "num_episodes": len(self.episode_returns),
            "real_buffer_size": len(self.real_buffer),
            "syn_buffer_size": self.syn_buffer.size,
        }

    def save(self, path: str):
        """Save model checkpoints."""
        torch.save({
            "policy": self.policy.state_dict(),
            "diffusion": self.diffusion.state_dict(),
            "relevance_fn": self.relevance_fn.state_dict() if hasattr(self.relevance_fn, "state_dict") else None,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "diffusion_optimizer": self.diffusion_optimizer.state_dict(),
            "total_env_steps": self.total_env_steps,
            "config": self.config,
        }, path)

    def load(self, path: str):
        """Load model checkpoints."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.diffusion.load_state_dict(ckpt["diffusion"])
        if self.relevance_fn is not None and ckpt["relevance_fn"] is not None:
            self.relevance_fn.load_state_dict(ckpt["relevance_fn"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        self.diffusion_optimizer.load_state_dict(ckpt["diffusion_optimizer"])
        self.total_env_steps = ckpt["total_env_steps"]


class SYNTHER(PGR):
    """Unconditional generative replay baseline (SynthER).

    Same as PGR but with p_uncond = 1.0 (always unconditional).
    """

    def __init__(self, config: RunConfig, state_dim: int, action_dim: int, is_pixel: bool = False):
        # Override to use unconditional diffusion
        config.diffusion.p_uncond = 1.0
        config.diffusion.guidance_scale = 1.0
        config.relevance_fn = "reward"  # Dummy, not used
        super().__init__(config, state_dim, action_dim, is_pixel)

    def compute_relevance(self, states, actions, rewards, next_states):
        return None

    def sample_relevance_condition(self, batch_size: int):
        return None
