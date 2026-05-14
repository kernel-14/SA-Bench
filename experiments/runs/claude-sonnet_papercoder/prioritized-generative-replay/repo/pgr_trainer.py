```python
## pgr_trainer.py
"""Prioritized Generative Replay (PGR) Trainer.

Central orchestrator implementing Algorithm 1 from the paper:
"Prioritized Generative Replay" (PGR). Coordinates environment interaction,
real/synthetic replay buffers, relevance function updates, conditional
diffusion model training and generation, and policy optimization in a
precise outer/inner loop structure.

The outer loop collects real transitions and updates the relevance function.
The inner loop (every inner_loop_freq=10000 steps) retrains the diffusion
model and refills the synthetic buffer. The policy trains on mixed batches
from D_real ∪ D_syn at ratio r=0.5 with UTD=20.

Config references (config.yaml):
    training.total_steps:              100000  # env steps
    training.eval_freq:                5000    # eval every N steps
    training.eval_episodes:            10      # episodes per eval
    buffer.real_capacity:              1000000 # D_real size
    buffer.syn_capacity:               1000000 # D_syn size
    sampling.synthetic_ratio:          0.5     # r in paper
    sampling.batch_size:               256     # baseline batch size
    policy.utd_ratio:                  20      # gradient steps per env step
    relevance.update_freq:             20      # ICM update frequency
    diffusion.inner_loop_freq:         10000   # inner loop frequency
    diffusion.train_steps_per_loop:    10000   # diffusion gradient steps
    diffusion.top_k_fraction:          0.1     # prompting strategy top-k
    diffusion.generation_batch_size:   1024    # generation batch size
    diffusion.p_uncond:                0.25    # CFG dropout probability
    diffusion.guidance_scale:          3.0     # CFG guidance scale omega
"""

import os
import random
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from buffers.mixed_sampler import MixedSampler
from buffers.replay_buffer import ReplayBuffer
from diffusion.conditional_diffusion import ConditionalDiffusion
from envs.dmc_env import DMCEnv
from envs.gym_env import GymEnv
from evaluation import Evaluator
from policies.drqv2 import DRQv2Policy
from policies.redq import REDQPolicy
from policies.sac import SACPolicy
from relevance.base import BaseRelevance
from relevance.icm import ICMRelevance
from relevance.rnd import RNDRelevance
from utils.logger import Logger


# ---------------------------------------------------------------------------
# Thin policy-based relevance wrappers (reward, return, TD-error)
# These derive scores from the policy's Q-networks and have no learnable
# parameters of their own. update() is a no-op returning 0.0.
# ---------------------------------------------------------------------------


class _RewardRelevance(BaseRelevance):
    """Relevance function F(s, a, s', r) = r (raw reward)."""

    def __init__(self, obs_dim: int, action_dim: int, device: str = "cuda") -> None:
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, device=device)

    def score(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        r = reward.to(device=self.device, dtype=torch.float32)
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        return r.detach()

    def update(self, batch: Dict[str, torch.Tensor]) -> float:
        return 0.0

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self.score(*args, **kwargs)


class _ReturnRelevance(BaseRelevance):
    """Relevance function F(s, a, s', r) = Q(s, pi(s)) (Q-value estimate)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        policy: Any,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, device=device)
        self._policy = policy

    def score(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        obs_f = obs.to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            # Use actor to get current policy action, then evaluate Q.
            if hasattr(self._policy, "actor"):
                pi_action, _ = self._policy.actor.sample(obs_f)
            else:
                pi_action = torch.zeros(
                    obs_f.shape[0], self.action_dim, device=self.device
                )
            # Use first critic for Q estimate.
            if hasattr(self._policy, "critics") and len(self._policy.critics) > 0:
                q_val = self._policy.critics[0](obs_f, pi_action)  # (B, 1)
            elif hasattr(self._policy, "critic1"):
                q_val = self._policy.critic1(obs_f, pi_action)
            else:
                q_val = reward.to(device=self.device, dtype=torch.float32)
                if q_val.dim() == 1:
                    q_val = q_val.unsqueeze(-1)
        return q_val.detach()

    def update(self, batch: Dict[str, torch.Tensor]) -> float:
        return 0.0

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self.score(*args, **kwargs)


class _TDErrorRelevance(BaseRelevance):
    """Relevance function F(s,a,s',r) = |r + gamma*Q_target(s',a*) - Q(s,a)|."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        policy: Any,
        gamma: float = 0.99,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, device=device)
        self._policy = policy
        self._gamma = gamma

    def score(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        obs_f = obs.to(device=self.device, dtype=torch.float32)
        act_f = action.to(device=self.device, dtype=torch.float32)
        next_obs_f = next_obs.to(device=self.device, dtype=torch.float32)
        rew_f = reward.to(device=self.device, dtype=torch.float32)
        if rew_f.dim() == 1:
            rew_f = rew_f.unsqueeze(-1)

        with torch.no_grad():
            # Get next action from actor.
            if hasattr(self._policy, "actor"):
                next_action, _ = self._policy.actor.sample(next_obs_f)
            else:
                next_action = torch.zeros(
                    next_obs_f.shape[0], self.action_dim, device=self.device
                )

            # Target Q value.
            if hasattr(self._policy, "target_critics") and len(self._policy.target_critics) > 0:
                target_q = self._policy.target_critics[0](next_obs_f, next_action)
            elif hasattr(self._policy, "target_critic1"):
                target_q = self._policy.target_critic1(next_obs_f, next_action)
            else:
                target_q = rew_f

            # Current Q value.
            if hasattr(self._policy, "critics") and len(self._policy.critics) > 0:
                current_q = self._policy.critics[0](obs_f, act_f)
            elif hasattr(self._policy, "critic1"):
                current_q = self._policy.critic1(obs_f, act_f)
            else:
                current_q = rew_f

            td_error = (rew_f + self._gamma * target_q - current_q).abs()  # (B, 1)

        return td_error.detach()

    def update(self, batch: Dict[str, torch.Tensor]) -> float:
        return 0.0

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self.score(*args, **kwargs)


# ---------------------------------------------------------------------------
# PGRTrainer
# ---------------------------------------------------------------------------


class PGRTrainer:
    """Central orchestrator for Prioritized Generative Replay (PGR).

    Implements Algorithm 1 from the paper. Coordinates:
        - Online environment interaction (outer loop)
        - Relevance function updates (every icm_update_freq=20 steps)
        - Conditional diffusion model training and synthetic data generation
          (inner loop, every inner_loop_freq=10000 steps)
        - Policy optimization on mixed real+synthetic batches (UTD=20)
        - Periodic evaluation and analysis experiments

    Attributes:
        config: Hydra/OmegaConf DictConfig carrying all hyperparameters.
        env: Environment wrapper (DMCEnv or GymEnv).
        real_buffer: D_real — circular replay buffer for real transitions.
        syn_buffer: D_syn — circular replay buffer for synthetic transitions.
        mixed_sampler: Samples mixed batches from D_real ∪ D_syn at ratio r.
        policy: REDQPolicy, SACPolicy, or DRQv2Policy.
        relevance_fn: BaseRelevance subclass (ICM, RND, or policy-based).
        diffusion: ConditionalDiffusion model with CFG.
        evaluator: Evaluator for policy evaluation and analysis experiments.
        logger: Logger for W&B/TensorBoard/CSV logging.
        global_step: Current environment step counter.
        episode_step: Steps within the current episode.
        episode_return: Cumulative reward for the current episode.
        episode_count: Total number of completed episodes.
        obs: Current environment observation.
        normalizer_fitted: Whether the diffusion normalizer has been fit.
        device: PyTorch device string.
        obs_dim: Flat observation dimension (latent dim for pixel tasks).
        action_dim: Action space dimension.
        batch_size: Effective batch size (may be overridden by scaling config).
        utd_ratio: Effective UTD ratio (may be overridden by scaling config).
        synthetic_ratio: Effective synthetic data ratio.
        syn_capacity: Effective synthetic buffer capacity.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialises all PGR components from the Hydra config.

        Resolves scaling experiment overrides, instantiates the environment,
        replay buffers, policy, relevance function, diffusion model, evaluator,
        and logger. Sets random seeds for reproducibility.

        Args:
            config: Hydra/OmegaConf DictConfig loaded from config.yaml.
                All hyperparameters are read from this object. See module
                docstring for the full list of referenced config fields.
        """
        self.config: DictConfig = config
        self.device: str = str(config.hardware.device)

        # ── Set random seeds ──────────────────────────────────────────────────
        seed: int = int(config.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ── Resolve scaling experiment overrides ──────────────────────────────
        # These override the base config values for scaling experiments (Sec. 5.3).
        self.batch_size: int = int(config.sampling.batch_size)
        self.utd_ratio: int = int(config.policy.utd_ratio)
        self.synthetic_ratio: float = float(config.sampling.synthetic_ratio)
        self.syn_capacity: int = int(config.buffer.syn_capacity)
        hidden_dim: int = int(config.policy.hidden_dim)
        hidden_layers: int = int(config.policy.hidden_layers)

        # Check scaling flags in priority order: combined > high_syn_ratio > larger_network.
        if hasattr(config, "scaling"):
            if (
                hasattr(config.scaling, "combined")
                and hasattr(config.scaling.combined, "enabled")
                and bool(config.scaling.combined.enabled)
            ):
                self.utd_ratio = int(config.scaling.combined.utd_ratio)
                self.syn_capacity = int(config.scaling.combined.syn_capacity)
                self.synthetic_ratio = float(config.scaling.combined.synthetic_ratio)
                self.batch_size = int(config.scaling.combined.batch_size)
                hidden_dim = int(config.scaling.combined.hidden_dim)
                hidden_layers = int(config.scaling.combined.hidden_layers)

            elif (
                hasattr(config.scaling, "high_syn_ratio")
                and hasattr(config.scaling.high_syn_ratio, "enabled")
                and bool(config.scaling.high_syn_ratio.enabled)
            ):
                self.synthetic_ratio = float(config.scaling.high_syn_ratio.synthetic_ratio)
                self.batch_size = int(config.scaling.high_syn_ratio.batch_size)

            elif (
                hasattr(config.scaling, "larger_network")
                and hasattr(config.scaling.larger_network, "enabled")
                and bool(config.scaling.larger_network.enabled)
            ):
                hidden_dim = int(config.scaling.larger_network.hidden_dim)
                hidden_layers = int(config.scaling.larger_network.hidden_layers)
                self.batch_size = int(config.scaling.larger_network.batch_size)

        # ── Instantiate environment ───────────────────────────────────────────
        env_type: str = str(config.env.type)
        env_name: str = str(config.env.name)
        env_seed: int = int(config.env.seed)
        pixel_obs: bool = bool(config.env.pixel_obs)

        if env_type == "dmc":
            self.env: Any = DMCEnv(
                env_name=env_name,
                pixel_obs=pixel_obs,
                image_size=int(config.env.image_size),
                frame_stack=int(config.env.frame_stack),
                seed=env_seed,
            )
        elif env_type == "gym":
            self.env = GymEnv(env_name=env_name, seed=env_seed)
        else:
            raise ValueError(
                f"Unknown env.type '{env_type}'. Must be 'dmc' or 'gym'."
            )

        # ── Derive observation and action dimensions ──────────────────────────
        # For pixel-based tasks, obs_dim is the CNN latent dimension (feature_dim=50),
        # not the raw pixel dimension. The diffusion model operates in latent space.
        if pixel_obs and str(config.policy.type) == "drqv2":
            # DRQv2 encodes pixels to feature_dim=50 before storing in D_real.
            self.obs_dim: int = int(config.drqv2.feature_dim)
        else:
            self.obs_dim = self.env.observation_space_dim()

        self.action_dim: int = self.env.action_space_dim()

        # ── Instantiate replay buffers ────────────────────────────────────────
        self.real_buffer: ReplayBuffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=int(config.buffer.real_capacity),
            device=self.device,
        )
        self.syn_buffer: ReplayBuffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=self.syn_capacity,
            device=self.device,
        )

        # ── Instantiate mixed sampler ─────────────────────────────────────────
        self.mixed_sampler: MixedSampler = MixedSampler(
            real_buffer=self.real_buffer,
            syn_buffer=self.syn_buffer,
            synthetic_ratio=self.synthetic_ratio,
            device=self.device,
        )

        # ── Instantiate policy ────────────────────────────────────────────────
        policy_type: str = str(config.policy.type)

        if policy_type == "redq":
            self.policy: Any = REDQPolicy(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                num_layers=hidden_layers,
                ensemble_size=int(config.policy.ensemble_size),
                subsample_size=int(config.policy.subsample_size),
                utd_ratio=self.utd_ratio,
                gamma=float(config.policy.gamma),
                tau=float(config.policy.tau),
                lr=float(config.policy.lr),
                auto_alpha=bool(config.policy.auto_alpha),
                device=self.device,
            )
        elif policy_type == "sac":
            self.policy = SACPolicy(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                num_layers=hidden_layers,
                gamma=float(config.policy.gamma),
                tau=float(config.policy.tau),
                lr=float(config.policy.lr),
                device=self.device,
                auto_alpha=bool(config.policy.auto_alpha),
            )
        elif policy_type == "drqv2":
            self.policy = DRQv2Policy(
                obs_channels=int(config.env.frame_stack) * 3,
                action_dim=self.action_dim,
                feature_dim=int(config.drqv2.feature_dim),
                hidden_dim=int(config.drqv2.hidden_dim),
                lr=float(config.drqv2.lr),
                gamma=float(config.policy.gamma),
                tau=float(config.policy.tau),
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unknown policy.type '{policy_type}'. "
                "Must be 'redq', 'sac', or 'drqv2'."
            )

        # ── Instantiate relevance function ────────────────────────────────────
        relevance_type: str = str(config.relevance.type)

        if relevance_type == "curiosity":
            self.relevance_fn: BaseRelevance = ICMRelevance(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                latent_dim=int(config.relevance.icm.latent_dim),
                hidden_dim=int(config.relevance.icm.hidden_dim),
                num_layers=int(config.relevance.icm.num_layers),
                lr=float(config.relevance.icm.lr),
                device=self.device,
                use_cnn=False,  # State-based: MLP encoder
            )
        elif relevance_type == "rnd":
            self.relevance_fn = RNDRelevance(
                obs_dim=self.obs_dim,
                latent_dim=int(config.relevance.icm.latent_dim),
                hidden_dim=int(config.relevance.icm.hidden_dim),
                num_layers=int(config.relevance.icm.num_layers),
                lr=float(config.relevance.rnd.lr),
                use_cnn=False,
                device=self.device,
            )
        elif relevance_type == "reward":
            self.relevance_fn = _RewardRelevance(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                device=self.device,
            )
        elif relevance_type == "return":
            self.relevance_fn = _ReturnRelevance(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                policy=self.policy,
                device=self.device,
            )
        elif relevance_type == "td_error":
            self.relevance_fn = _TDErrorRelevance(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                policy=self.policy,
                gamma=float(config.policy.gamma),
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unknown relevance.type '{relevance_type}'. "
                "Must be one of: 'curiosity', 'rnd', 'reward', 'return', 'td_error'."
            )

        # ── Instantiate conditional diffusion model ───────────────────────────
        # Input dimension: concatenation of (s, a, s', r).
        diffusion_input_dim: int = self.obs_dim + self.action_dim + self.obs_dim + 1

        self.diffusion: ConditionalDiffusion = ConditionalDiffusion(
            input_dim=diffusion_input_dim,
            hidden_dim=int(config.diffusion.hidden_dim),
            num_layers=int(config.diffusion.num_layers),
            time_emb_dim=int(config.diffusion.time_emb_dim),
            cond_emb_dim=int(config.diffusion.cond_emb_dim),
            num_timesteps=int(config.diffusion.num_timesteps),
            beta_start=float(config.diffusion.beta_start),
            beta_end=float(config.diffusion.beta_end),
            p_uncond=float(config.diffusion.p_uncond),
            guidance_scale=float(config.diffusion.guidance_scale),
            lr=float(config.diffusion.lr),
            device=self.device,
        )

        # ── Instantiate evaluator and logger ──────────────────────────────────
        self.evaluator: Evaluator = Evaluator(
            env=self.env,
            policy=self.policy,
            diffusion=self.diffusion,
            relevance_fn=self.relevance_fn,
            device=self.device,
        )

        use_wandb: bool = bool(config.logging.use_wandb)
        log_dir: str = str(config.logging.log_dir)
        self.logger: Logger = Logger(
            config=config,
            use_wandb=use_wandb,
            log_dir=log_dir,
        )

        # ── Training state variables ──────────────────────────────────────────
        self.global_step: int = 0
        self.episode_step: int = 0
        self.episode_return: float = 0.0
        self.episode_count: int = 0
        self.normalizer_fitted: bool = False

        # Reset environment and get initial observation.
        self.obs: np.ndarray = self.env.reset()

        # For pixel-based tasks, encode the initial observation immediately.
        if pixel_obs and policy_type == "drqv2":
            self.obs = self._encode_pixel_obs(self.obs)

        # ── Cached config values for hot-path access ──────────────────────────
        self._pixel_obs: bool = pixel_obs
        self._policy_type: str = policy_type
        self._relevance_type: str = relevance_type
        self._total_steps: int = int(config.training.total_steps)
        self._eval_freq: int = int(config.training.eval_freq)
        self._eval_episodes: int = int(config.training.eval_episodes)
        self._icm_update_freq: int = int(config.relevance.update_freq)
        self._inner_loop_freq: int = int(config.diffusion.inner_loop_freq)
        self._diffusion_train_steps: int = int(config.diffusion.train_steps_per_loop)
        self._top_k_fraction: float = float(config.diffusion.top_k_fraction)
        self._gen_batch_size: int = int(config.diffusion.generation_batch_size)
        self._checkpoint_freq: int = int(config.logging.checkpoint_freq)
        self._checkpoint_dir: str = str(config.logging.checkpoint_dir)
        self._mse_eval_epoch: int = int(config.evaluation.mse_eval_epoch)
        self._relevance_eval_freq: int = int(config.evaluation.relevance_eval_freq)
        self._dormant_threshold: float = float(config.evaluation.dormant_threshold)
        self._relevance_num_samples: int = int(config.evaluation.relevance_num_samples)
        self._mse_num_samples: int = int(config.evaluation.mse_num_samples)

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Runs the main PGR training loop for total_steps environment steps.

        Implements Algorithm 1 from the paper:
            Outer loop: collect transitions, update relevance function
            Inner loop (every 10K steps): retrain diffusion, generate D_syn
            Policy update: UTD=20 gradient steps per env step

        Also handles periodic evaluation, analysis experiments (Fig. 5, 6),
        relevance distribution logging (Fig. 6b), and checkpointing.
        """
        print(
            f"Starting PGR training: {self._total_steps} steps, "
            f"env={self.config.env.name}, "
            f"relevance={self._relevance_type}, "
            f"device={self.device}"
        )

        for step in range(self._total_steps):
            self.global_step = step

            # ── Outer loop: collect one real transition ───────────────────────
            # Algorithm 1, line 2: "Collect transitions T_real with π"
            self._collect_transition()

            # ── Outer loop: update relevance function ─────────────────────────
            # Algorithm 1, line 3: "Update F using D_real"
            # Frequency: every icm_update_freq=20 steps = 5% of policy steps
            if (
                step % self._icm_update_freq == 0
                and len(self.real_buffer) > self.batch_size
            ):
                self._update_relevance_scores()

            # ── Inner loop: retrain diffusion + generate D_syn ────────────────
            # Algorithm 1, lines 4-8: periodic inner loop
            # Triggered every inner_loop_freq=10000 steps (not at step 0)
            if (
                step > 0
                and step % self._inner_loop_freq == 0
                and len(self.real_buffer) > self.batch_size
            ):
                self._run_inner_loop()

            # ── Policy update: UTD=20 gradient steps ─────────────────────────
            # Algorithm 1, line 7: "Train π on samples from D_real ∪ D_syn"
            if len(self.real_buffer) > self.batch_size:
                policy_metrics: Dict[str, float] = self._update_policy()

                # Log policy metrics periodically (every 1000 steps) to avoid
                # excessive logging overhead with UTD=20.
                if step % 1000 == 0:
                    prefixed: Dict[str, float] = {
                        f"policy/{k}": v for k, v in policy_metrics.items()
                    }
                    self._log_metrics(prefixed, step)

            # ── Periodic policy evaluation ────────────────────────────────────
            if step > 0 and step % self._eval_freq == 0:
                mean_return: float = self._evaluate(self._eval_episodes)
                self._log_metrics({"eval/mean_return": mean_return}, step)
                print(f"  Step {step:6d} | eval_return={mean_return:.2f}")

            # ── Periodic relevance distribution logging (Fig. 6b) ─────────────
            # Paper: "measuring the distribution of F(s,a,s',r) every 10K timesteps"
            if (
                step > 0
                and step % self._relevance_eval_freq == 0
                and len(self.real_buffer) > 0
            ):
                relevance_values: np.ndarray = (
                    self.evaluator.compute_relevance_distribution(
                        real_buffer=self.real_buffer,
                        num_samples=self._relevance_num_samples,
                    )
                )
                if len(relevance_values) > 0:
                    self.logger.log_histogram(
                        "analysis/relevance_distribution",
                        relevance_values,
                        step,
                    )

            # ── Mid-training analysis (Fig. 5, 6a) ───────────────────────────
            # Paper: "analysis is performed at epoch 50 (halfway through)"
            if step == self._mse_eval_epoch and len(self.real_buffer) > 0:
                self._run_analysis()

            # ── Periodic checkpointing ────────────────────────────────────────
            if step > 0 and step % self._checkpoint_freq == 0:
                ckpt_path: str = os.path.join(
                    self._checkpoint_dir, f"step_{step}.pt"
                )
                self.save_checkpoint(ckpt_path)

        # ── Final evaluation ──────────────────────────────────────────────────
        final_return: float = self._evaluate(self._eval_episodes)
        self._log_metrics({"eval/final_return": final_return}, self._total_steps)
        print(f"Training complete. Final eval return: {final_return:.2f}")

        # ── Final checkpoint ──────────────────────────────────────────────────
        final_ckpt_path: str = os.path.join(self._checkpoint_dir, "final.pt")
        self.save_checkpoint(final_ckpt_path)

        # ── Close logger ──────────────────────────────────────────────────────
        self.logger.close()

    def save_checkpoint(self, path: str) -> None:
        """Saves training state and all model weights to a checkpoint file.

        Saves the global step, episode count, normalizer_fitted flag, and
        state dicts for the policy, relevance function, and diffusion model.
        Replay buffers are NOT saved (too large — up to 1M transitions).

        Args:
            path: Full file path for the checkpoint. Parent directories are
                created if they do not exist.
        """
        parent_dir: str = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Build checkpoint dict.
        checkpoint: Dict[str, Any] = {
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "episode_return": self.episode_return,
            "episode_step": self.episode_step,
            "normalizer_fitted": self.normalizer_fitted,
        }

        # Save policy state — use the policy's own save() method internals
        # by extracting the state dict directly.
        if has