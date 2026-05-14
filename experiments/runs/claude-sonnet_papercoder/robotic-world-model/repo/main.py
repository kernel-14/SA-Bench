```python
## main.py
"""Entry point for the Robotic World Model (RWM) reproduction.

Orchestrates all experimental phases described in the paper:
  1. Data collection for world model pretraining (6M transitions)
  2. World model (GRU-based RWM) pretraining with autoregressive training
  3. Benchmark evaluation: prediction accuracy, noise robustness (Figs. 3, 4)
  4. Horizon ablation study (M x N grid, Fig. S8)
  5. MBPO-PPO policy training and evaluation (Fig. 5)

Usage:
    python main.py robot=anymal_d experiment=all
    python main.py robot=unitree_g1 experiment=policy
    python main.py --multirun seed=0,1,2,3,4 experiment=all

Config is loaded from config.yaml via Hydra.
"""

import logging
import os
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Core utilities
from utils.common import get_device, sample_gaussian, set_seed
from utils.logger import Logger

# Data pipeline
from data.replay_buffer import ReplayBuffer
from data.trajectory_dataset import TrajectoryDataset

# Models
from models.baselines.mlp_model import MLPModel
from models.baselines.rssm_model import RSSMModel
from models.baselines.transformer_model import TransformerModel
from models.policy import PolicyNetwork, ValueNetwork
from models.rwm import GRUWorldModel

# Training
from training.mbpo_ppo_trainer import MBPOPPOTrainer
from training.rwm_trainer import RWMTrainer

# Evaluation
from evaluation.benchmark import Benchmark
from evaluation.metrics import Metrics
from evaluation.visualizer import Visualizer

# Environment — Isaac Lab is optional; fall back to MockEnv
_ISAAC_LAB_AVAILABLE: bool = False
try:
    from envs.anymal_env import ANYmalEnv
    from envs.unitree_g1_env import UnitreeG1Env

    _ISAAC_LAB_AVAILABLE = True
except ImportError:
    _ISAAC_LAB_AVAILABLE = False

from envs.mock_env import MockEnv

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _get_robot_cfg(cfg: DictConfig) -> DictConfig:
    """Return the robot-specific sub-config (anymal_d or unitree_g1)."""
    robot: str = str(cfg.robot)
    if robot == "anymal_d":
        return cfg.anymal_d
    elif robot == "unitree_g1":
        return cfg.unitree_g1
    else:
        raise ValueError(
            f"Unknown robot '{robot}'. Expected 'anymal_d' or 'unitree_g1'. "
            "Check the 'robot' field in config.yaml."
        )


def build_model_config(cfg: DictConfig) -> SimpleNamespace:
    """Flatten the nested Hydra DictConfig into a SimpleNamespace.

    All model and trainer classes consume this namespace via attribute access.
    Merges: robot_cfg (obs/action dims, reward weights) + cfg.rwm (architecture)
            + cfg.rwm_training + cfg.mbpo_ppo + cfg.simulation + cfg.baselines.

    The SimpleNamespace also exposes the original robot sub-config under
    ``ns[robot_type]`` so that classes that do ``config[robot_type].obs_dim``
    (e.g., GRUWorldModel) work correctly.

    Args:
        cfg: Full Hydra DictConfig loaded from config.yaml.

    Returns:
        A SimpleNamespace with all fields needed by downstream modules.
    """
    robot_cfg: DictConfig = _get_robot_cfg(cfg)
    robot_type: str = str(cfg.robot)

    ns = SimpleNamespace(
        # ----------------------------------------------------------------
        # Identity
        # ----------------------------------------------------------------
        env_name=str(robot_cfg.env_name),
        robot=robot_type,
        device=str(cfg.device),
        seed=int(cfg.seed),
        num_seeds=int(cfg.num_seeds),
        experiment=str(cfg.experiment),
        env_backend=str(cfg.env_backend),

        # ----------------------------------------------------------------
        # Observation / action / privileged info dimensions (Tables S2-S5)
        # ----------------------------------------------------------------
        obs_dim=int(robot_cfg.obs_dim),
        action_dim=int(robot_cfg.action_dim),
        priv_dim=int(robot_cfg.priv_dim),
        policy_obs_dim=int(robot_cfg.policy_obs_dim),

        # Observation index slices (shared knowledge per design)
        obs_slices=OmegaConf.to_container(robot_cfg.obs_slices, resolve=True),
        policy_obs_slices=OmegaConf.to_container(
            robot_cfg.policy_obs_slices, resolve=True
        ),
        priv_slices=OmegaConf.to_container(robot_cfg.priv_slices, resolve=True),

        # Reward weights and temperature factors (Table S6)
        reward_weights=OmegaConf.to_container(
            robot_cfg.reward_weights, resolve=True
        ),
        sigma_vxy=float(robot_cfg.sigma_vxy),
        sigma_wz=float(robot_cfg.sigma_wz),

        # ----------------------------------------------------------------
        # Simulation parameters
        # ----------------------------------------------------------------
        dt=float(cfg.simulation.dt),
        control_freq_hz=int(cfg.simulation.control_freq_hz),
        num_envs_real=int(cfg.simulation.num_envs_real),
        num_envs_pretrain=int(cfg.simulation.num_envs_pretrain),

        # ----------------------------------------------------------------
        # RWM architecture (Table S7)
        # ----------------------------------------------------------------
        gru_hidden_size=int(cfg.rwm.gru_hidden_size),
        gru_num_layers=int(cfg.rwm.gru_num_layers),
        mlp_head_hidden=int(cfg.rwm.mlp_head_hidden),
        mlp_head_activation=str(cfg.rwm.mlp_head_activation),

        # Dual-autoregressive horizons (Table S10)
        history_horizon=int(cfg.rwm.history_horizon),
        forecast_horizon=int(cfg.rwm.forecast_horizon),
        forecast_decay=float(cfg.rwm.forecast_decay),

        # ----------------------------------------------------------------
        # RWM training parameters (Table S10)
        # ----------------------------------------------------------------
        max_iterations=int(cfg.rwm_training.max_iterations),
        learning_rate=float(cfg.rwm_training.learning_rate),
        weight_decay=float(cfg.rwm_training.weight_decay),
        batch_size=int(cfg.rwm_training.batch_size),
        pretrain_transitions=int(cfg.rwm_training.pretrain_transitions),

        # ----------------------------------------------------------------
        # MBPO-PPO training parameters (Table S11)
        # ----------------------------------------------------------------
        imagination_envs=int(cfg.mbpo_ppo.imagination_envs),
        imagination_steps=int(cfg.mbpo_ppo.imagination_steps),
        buffer_size=int(cfg.mbpo_ppo.buffer_size),
        ppo_lr=float(cfg.mbpo_ppo.learning_rate),
        ppo_weight_decay=float(cfg.mbpo_ppo.weight_decay),
        ppo_epochs=int(cfg.mbpo_ppo.learning_epochs),
        ppo_minibatches=int(cfg.mbpo_ppo.num_mini_batches),
        ppo_kl_target=float(cfg.mbpo_ppo.kl_divergence_target),
        gamma=float(cfg.mbpo_ppo.discount_factor),
        ppo_clip=float(cfg.mbpo_ppo.clip_range),
        entropy_coef=float(cfg.mbpo_ppo.entropy_coefficient),
        gae_lambda=float(cfg.mbpo_ppo.gae_lambda),
        ppo_max_iterations=int(cfg.mbpo_ppo.max_iterations),

        # ----------------------------------------------------------------
        # Policy and value function architecture (Table S9)
        # ----------------------------------------------------------------
        policy_hidden=list(cfg.policy.hidden_sizes),
        policy_activation=str(cfg.policy.activation),
        value_hidden=list(cfg.value_function.hidden_sizes),
        value_activation=str(cfg.value_function.activation),

        # ----------------------------------------------------------------
        # Collision handling (Section A.4.3)
        # ----------------------------------------------------------------
        terminate_on_base_contact=bool(
            cfg.collision_handling.terminate_on_base_contact
        ),
        termination_threshold=float(cfg.collision_handling.termination_threshold),
        use_pretraining=bool(cfg.collision_handling.use_pretraining),

        # ----------------------------------------------------------------
        # Logging and checkpointing
        # ----------------------------------------------------------------
        log_dir=str(cfg.log_dir),
        checkpoint_dir=str(cfg.checkpoint_dir),
        use_wandb=bool(cfg.use_wandb),
        log_interval=int(cfg.logging.log_interval),
        eval_interval=int(cfg.logging.eval_interval),
        eval_episodes=int(cfg.logging.eval_episodes),
        save_interval=int(cfg.logging.save_interval),

        # ----------------------------------------------------------------
        # Ablation study grid (Section A.4.1)
        # ----------------------------------------------------------------
        ablation_m_values=list(cfg.ablation.m_values),
        ablation_n_values=list(cfg.ablation.n_values),

        # ----------------------------------------------------------------
        # Noise robustness evaluation (Section 4.2)
        # ----------------------------------------------------------------
        noise_levels=list(cfg.noise_robustness.noise_levels),

        # ----------------------------------------------------------------
        # Baseline model configurations (Table S8)
        # ----------------------------------------------------------------
        baseline_mlp_hidden=list(cfg.baselines.mlp.hidden_sizes),
        baseline_mlp_activation=str(cfg.baselines.mlp.activation),
        baseline_mlp_ar=bool(cfg.baselines.mlp.use_autoregressive_training),
        baseline_rssm_hidden=int(cfg.baselines.rssm.hidden_size),
        baseline_rssm_layers=int(cfg.baselines.rssm.num_layers),
        baseline_rssm_latent=int(cfg.baselines.rssm.latent_dim),
        baseline_rssm_categories=int(cfg.baselines.rssm.num_categories),
        baseline_rssm_ar=bool(cfg.baselines.rssm.use_autoregressive_training),
        baseline_transformer_dmodel=int(cfg.baselines.transformer.d_model),
        baseline_transformer_nhead=int(cfg.baselines.transformer.nhead),
        baseline_transformer_layers=int(cfg.baselines.transformer.num_layers),
        baseline_transformer_context=int(cfg.baselines.transformer.context_length),
        baseline_transformer_ff=int(cfg.baselines.transformer.dim_feedforward),
        baseline_transformer_ar=bool(
            cfg.baselines.transformer.use_autoregressive_training
        ),
    )

    # ----------------------------------------------------------------
    # Attach robot sub-configs so that classes doing config[robot_type]
    # (e.g., GRUWorldModel.__init__) work correctly via __getitem__.
    # We wrap the SimpleNamespace in a thin adapter below.
    # ----------------------------------------------------------------
    # Store the raw robot DictConfig for item-access compatibility
    ns._anymal_d_cfg = cfg.anymal_d
    ns._unitree_g1_cfg = cfg.unitree_g1
    ns._baselines_cfg = cfg.baselines
    ns._rwm_cfg = cfg.rwm

    return ns


class _ConfigAdapter:
    """Thin adapter that wraps a SimpleNamespace and supports item access.

    Models like GRUWorldModel do ``config[robot_type].obs_dim`` which requires
    ``__getitem__`` support. This adapter delegates attribute access to the
    underlying SimpleNamespace and item access to the stored robot sub-configs.

    This avoids modifying any model code while keeping the config interface
    consistent with the design specification.
    """

    def __init__(self, ns: SimpleNamespace, cfg: DictConfig) -> None:
        """Initialize the adapter.

        Args:
            ns: Flat SimpleNamespace from build_model_config.
            cfg: Original Hydra DictConfig for item-access fallback.
        """
        self._ns = ns
        self._cfg = cfg

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying SimpleNamespace."""
        try:
            return getattr(self._ns, name)
        except AttributeError:
            # Fall back to the original DictConfig
            try:
                return getattr(self._cfg, name)
            except AttributeError:
                raise AttributeError(
                    f"Config has no attribute '{name}'. "
                    "Check config.yaml for the correct field name."
                )

    def __getitem__(self, key: str) -> Any:
        """Support item access for robot sub-configs and baselines.

        Handles patterns like:
            config["anymal_d"]  → robot sub-config
            config["unitree_g1"] → robot sub-config
            config["baselines"] → baselines sub-config
            config["rwm"]       → rwm sub-config
        """
        if key == "anymal_d":
            return self._cfg.anymal_d
        elif key == "unitree_g1":
            return self._cfg.unitree_g1
        elif key == "baselines":
            return self._cfg.baselines
        elif key == "rwm":
            return self._cfg.rwm
        elif key == "rwm_training":
            return self._cfg.rwm_training
        elif key == "mbpo_ppo":
            return self._cfg.mbpo_ppo
        elif key == "simulation":
            return self._cfg.simulation
        elif key == "collision_handling":
            return self._cfg.collision_handling
        elif key == "logging":
            return self._cfg.logging
        elif key == "ablation":
            return self._cfg.ablation
        elif key == "noise_robustness":
            return self._cfg.noise_robustness
        else:
            # Try the DictConfig directly
            try:
                return self._cfg[key]
            except Exception:
                raise KeyError(
                    f"Config has no item '{key}'. "
                    "Check config.yaml for the correct field name."
                )


def _make_config_adapter(ns: SimpleNamespace, cfg: DictConfig) -> _ConfigAdapter:
    """Create a _ConfigAdapter from a SimpleNamespace and DictConfig.

    Args:
        ns: Flat SimpleNamespace from build_model_config.
        cfg: Original Hydra DictConfig.

    Returns:
        A _ConfigAdapter that supports both attribute and item access.
    """
    return _ConfigAdapter(ns, cfg)


# ---------------------------------------------------------------------------
# Checkpoint directory helpers
# ---------------------------------------------------------------------------


def _get_ckpt_dir(model_config: SimpleNamespace) -> Path:
    """Return a per-robot, per-seed checkpoint directory, creating it if needed.

    Args:
        model_config: Flat model config namespace.

    Returns:
        Path to the checkpoint directory.
    """
    ckpt_dir = (
        Path(model_config.checkpoint_dir)
        / model_config.robot
        / f"seed_{model_config.seed}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _get_log_dir(model_config: SimpleNamespace) -> Path:
    """Return a per-robot, per-seed log directory, creating it if needed.

    Args:
        model_config: Flat model config namespace.

    Returns:
        Path to the log directory.
    """
    log_dir = (
        Path(model_config.log_dir)
        / model_config.robot
        / f"seed_{model_config.seed}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def _make_env(
    cfg: DictConfig,
    config_adapter: _ConfigAdapter,
    num_envs: int,
) -> Any:
    """Instantiate the appropriate environment.

    Priority:
      1. cfg.env_backend == "mock"  → MockEnv (always, for testing)
      2. Isaac Lab available        → ANYmalEnv or UnitreeG1Env
      3. Fallback                   → MockEnv with a warning

    Args:
        cfg: Full Hydra DictConfig.
        config_adapter: Config adapter for environment initialization.
        num_envs: Number of parallel environments to create.

    Returns:
        An environment instance implementing the BaseEnv interface.
    """
    env_backend: str = str(cfg.env_backend)
    robot: str = str(cfg.robot)

    if env_backend == "mock":
        log.info("Using MockEnv (env_backend=mock, num_envs=%d).", num_envs)
        return MockEnv(config_adapter, num_envs)

    if not _ISAAC_LAB_AVAILABLE:
        warnings.warn(
            "Isaac Lab not found. Falling back to MockEnv. "
            "Install Isaac Lab per https://isaac-sim.github.io/IsaacLab/ "
            "for full physics simulation experiments.",
            UserWarning,
            stacklevel=2,
        )
        log.warning("Isaac Lab unavailable. Using MockEnv (num_envs=%d).", num_envs)
        return MockEnv(config_adapter, num_envs)

    if robot == "anymal_d":
        log.info("Instantiating ANYmalEnv (num_envs=%d).", num_envs)
        return ANYmalEnv(config_adapter, num_envs)
    elif robot == "unitree_g1":
        log.info("Instantiating UnitreeG1Env (num_envs=%d).", num_envs)
        return UnitreeG1Env(config_adapter, num_envs)
    else:
        raise ValueError(
            f"Unknown robot '{robot}'. Expected 'anymal_d' or 'unitree_g1'."
        )


# ---------------------------------------------------------------------------
# Phase 1: Data collection for world model pretraining
# ---------------------------------------------------------------------------


def _collect_pretraining_data(
    cfg: DictConfig,
    config_adapter: _ConfigAdapter,
    model_config: SimpleNamespace,
    replay_buffer: ReplayBuffer,
    logger: Logger,
) -> None:
    """Collect pretraining data by rolling out a random policy in simulation.

    Per Section A.4.3 and Table 1: collect 6M state transitions using
    cfg.simulation.num_envs_pretrain=4096 parallel environments.

    The paper states pretraining does not require optimal policies (Section
    A.4.3): "RWM pretraining does not require data from optimal policies."
    We use a random policy as a conservative fallback.

    Data stored: complete trajectories (obs, actions, priv, commands) per
    episode, appended to replay_buffer via add_trajectory().

    Args:
        cfg: Full Hydra DictConfig.
        config_adapter: Config adapter for environment initialization.
        model_config: Flat model config namespace.
        replay_buffer: ReplayBuffer to populate with collected trajectories.
        logger: Logger for progress metrics.
    """
    num_envs: int = model_config.num_envs_pretrain
    target_transitions: int = model_config.pretrain_transitions
    device: torch.device = get_device(model_config.device)
    min_traj_len: int = model_config.history_horizon + model_config.forecast_horizon

    log.info(
        "Phase 1: Collecting %d pretraining transitions with %d envs "
        "(min trajectory length: %d steps).",
        target_transitions,
        num_envs,
        min_traj_len,
    )

    env = _make_env(cfg, config_adapter, num_envs)

    try:
        obs, command = env.reset()
        # obs: [num_envs, obs_dim], command: [num_envs, 3]

        # Per-env trajectory accumulators (lists of per-step tensors on CPU)
        traj_obs: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        traj_actions: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        traj_priv: List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        # Store the command at episode start (constant throughout episode)
        traj_commands: List[torch.Tensor] = [
            command[i].cpu() for i in range(num_envs)
        ]

        total_transitions: int = 0
        step_count: int = 0
        trajectories_stored: int = 0

        while total_transitions < target_transitions:
            # Random policy: uniform in [-1, 1] for joint position targets.
            # The paper uses a pretrained velocity-tracking policy for
            # pretraining data; random actions are a valid fallback since
            # Section A.4.3 confirms suboptimal policies work for pretraining.
            action: torch.Tensor = (
                torch.rand(num_envs, model_config.action_dim, device=device) * 2.0 - 1.0
            )

            next_obs, priv, reward, done, info = env.step(action)

            # Accumulate per-env transitions (move to CPU immediately to
            # avoid accumulating GPU tensors in the list)
            for env_idx in range(num_envs):
                traj_obs[env_idx].append(obs[env_idx].detach().cpu())
                traj_actions[env_idx].append(action[env_idx].detach().cpu())
                traj_priv[env_idx].append(priv[env_idx].detach().cpu())

            # Flush completed episodes to replay buffer
            done_bool: torch.Tensor = done.bool()
            done_indices: List[int] = done_bool.nonzero(as_tuple=True)[0].tolist()

            for env_idx in done_indices:
                traj_len: int = len(traj_obs[env_idx])
                if traj_len >= min_traj_len:
                    # Stack per-step tensors into trajectory tensors
                    obs_traj: torch.Tensor = torch.stack(traj_obs[env_idx])
                    # shape: [T, obs_dim]
                    act_traj: torch.Tensor = torch.stack(traj_actions[env_idx])
                    # shape: [T, action_dim]
                    priv_traj: torch.Tensor = torch.stack(traj_priv[env_idx])
                    # shape: [T, priv_dim]
                    # Expand the stored command to match trajectory length
                    cmd_traj: torch.Tensor = (
                        traj_commands[env_idx].unsqueeze(0).expand(traj_len, -1)
                    )
                    # shape: [T, 3]

                    replay_buffer.add_trajectory(
                        obs=obs_traj,
                        actions=act_traj,
                        priv=priv_traj,
                        commands=cmd_traj,
                    )
                    total_transitions += traj_len
                    trajectories_stored += 1

                # Reset accumulators for this environment
                traj_obs[env_idx] = []
                traj_actions[env_idx] = []
                traj_priv[env_idx] = []

            # Update observations for the next step.
            # For done environments, next_obs already contains the reset obs
            # (handled by the environment's internal reset logic).
            obs = next_obs

            # Update stored commands for reset environments
            if done_indices:
                _, new_commands = env.reset()
                for env_idx in done_indices:
                    traj_commands[env_idx] = new_commands[env_idx].cpu()

            step_count += 1

            # Periodic progress logging
            if step_count % 500 == 0:
                log.info(
                    "  Step %d: collected %d / %d transitions "
                    "(%d trajectories stored, buffer size: %d).",
                    step_count,
                    total_transitions,
                    target_transitions,
                    trajectories_stored,
                    len(replay_buffer),
                )
                logger.log(
                    {
                        "data_collection/transitions": total_transitions,
                        "data_collection/trajectories": trajectories_stored,
                        "data_collection/buffer_size": len(replay_buffer),
                    },
                    step=step_count,
                )

    finally:
        env.close()

    log.info(
        "Phase 1 complete. Collected %d transitions in %d trajectories "
        "(buffer size: %d).",
        total_transitions,
        trajectories_stored,
        len(replay_buffer),
    )


# ---------------------------------------------------------------------------
# Phase 2: World model pretraining
# ---------------------------------------------------------------------------


def _pretrain_world_model(
    config_adapter: _ConfigAdapter,
    model_config: SimpleNamespace,
    replay_buffer: ReplayBuffer,
    logger: Logger,
    ckpt_dir: Path,
) -> GRUWorldModel:
    """Pretrain the GRU-based world model using autoregressive training.

    Architecture: GRU(256, 256) base + MLP(128) heads (Table S7).
    Training: Adam lr=1e-4, weight_decay=1e-5, batch_size=1024,
              max_iterations=2500, M=32, N=8, alpha=1.0 (Table S10).

    If a checkpoint already exists at ckpt_dir/rwm_pretrained.pt, it is
    loaded and training is skipped (idempotent behavior for reruns).

    Args:
        config_adapter: Config adapter for model initialization.
        model_config: Flat model config namespace.
        replay_buffer: ReplayBuffer containing pretraining trajectories.
        logger: Logger for training metrics.
        ckpt_dir: Directory for saving/loading checkpoints.

    Returns:
        Trained GRUWorldModel on the configured device.
    """
    ckpt_path: Path = ckpt_dir / "rwm_pretrained.pt"
    device: torch.device = get_device(model_config.device)

    # Instantiate the world model
    world_model: GRUWorldModel = GRUWorldModel(config_adapter)
    world_model.to(device)

    # Load existing checkpoint if available (skip retraining)
    if ckpt_path.exists():
        log.info(
            "Phase 2: Loading pretrained world model from %s (skipping training).",
            ckpt_path,
        )
        rwm_trainer = RWMTrainer(world_model, config_adapter, logger)
        rwm_trainer.load_checkpoint(str(ckpt_path))
        return world_model

    log.info(
        "Phase 2: Pretraining world model for %d iterations "
        "(M=%d, N=%d, lr=%g, batch_size=%d).",
        model_config.max_iterations,
        model_config.history_horizon,
        model_config.forecast_horizon,
        model_config.learning_rate,
        model_config.batch_size,
    )

    rwm_trainer = RWMTrainer(world_model, config_adapter, logger)
    rwm_trainer.train(replay_buffer, num_iterations=model_config.max_iterations)
    rwm_trainer.save_checkpoint(str(ckpt_path))

    log.info("Phase 2 complete. Checkpoint saved to %s", ckpt_path)
    return world_model


def _train_or_load_model(
    model: torch.nn.Module,
    model_name: str,
    config_adapter: _ConfigAdapter,
    replay_buffer: ReplayBuffer,
    logger: Logger,
    ckpt_dir: Path,
    max_iterations: int,
) -> torch.nn.Module:
    """Train a world model baseline or load from checkpoint if it exists.

    Shared helper for training MLP, RSSM, Transformer, and RWM-TF baselines.
    All baselines use the same RWMTrainer with the same hyperparameters as
    the main RWM (Table S10), differing only in architecture.

    Args:
        model: Instantiated model to train (already on target device).
        model_name: Human-readable name for logging and checkpoint naming.
        config_adapter: Config adapter for RWMTrainer initialization.
        replay_buffer: ReplayBuffer containing training trajectories.
        logger: Logger for training metrics.
        ckpt_dir: Directory for saving/loading checkpoints.
        max_iterations: Number of training iterations.

    Returns:
        The trained model (same object as input, modified in-place).
    """
    # Sanitize model_name for use as a filename (replace spaces and slashes)
    safe_name: str = model_name.lower().replace(" ", "_").replace("/", "_")
    ckpt_path: Path = ckpt_dir / f"{safe_name}_baseline.pt"

    trainer = RWMTrainer(model, config_adapter, logger)

    if ckpt_path.exists():
        log.info("Loading %s from %s (skipping training).", model_name, ckpt_path)
        trainer.load_checkpoint(str(ckpt_path))
    else:
        log.info("Training %s for %d iterations...", model_name, max_iterations)
        trainer.train(replay_buffer, num_iterations=max_iterations)
        trainer.save_checkpoint(str(ckpt_path))
        log.info("%s training complete. Checkpoint saved to %s", model_name, ckpt_path)

    return model


# ---------------------------------------------------------------------------
# Phase 3: Benchmark evaluation
# ---------------------------------------------------------------------------


def _run_benchmark(
    cfg: DictConfig,
    config_adapter: _ConfigAdapter,
    model_config: SimpleNamespace,
    world_model: GRUWorldModel,
    replay_buffer: ReplayBuffer,
    logger: Logger,
    ckpt_dir: Path,
) -> None:
    """Run all benchmark experiments from Sections 4.1-4.3.

    Experiments:
      - Autoregressive prediction accuracy vs baselines (Fig. 4)
      - Noise robustness: RWM vs MLP (Fig. 3b)

    Baselines trained here (Table S8):
      - MLP (autoregressive training, same as RWM for fair comparison)
      - RSSM (teacher-forcing by default)
      - Transformer (teacher-forcing by default)
      - RWM-TF (same GRU architecture but N=1 teacher-forcing)

    All models receive the same context (M=32) during evaluation per
    Section 4.3: "All models are given the same context during training
    and evaluation."

    Args:
        cfg: Full Hydra DictConfig.
        config_adapter: Config adapter for model initialization.
        model_config: Flat model config namespace.
        world_model: Pretrained GRUWorldModel (RWM-AR).
        replay_buffer: ReplayBuffer containing training/test trajectories.
        logger: Logger for benchmark metrics.
        ckpt_dir: Directory for saving/loading baseline checkpoints.
    """
    log.info("Phase 3: Running benchmark evaluation.")
    device: torch.device = get_device(model_config.device)

    # ----------------------------------------------------------------
    # Build test dataset from the replay buffer.
    # In a full reproduction, a separate held-out set should be collected.
    # Here we use the same buffer for simplicity — the benchmark measures
    # autoregressive rollout