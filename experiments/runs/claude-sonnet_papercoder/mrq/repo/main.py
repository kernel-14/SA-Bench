```python
## main.py
"""Entry point for MR.Q (Model-based Representations for Q-learning) experiments.

This module handles:
  - Command-line argument parsing
  - Configuration construction from YAML + CLI overrides + ablation variants
  - The Trainer class that orchestrates the full training loop
  - Evaluation, logging (TensorBoard + CSV + JSON), and checkpointing

Usage:
    python main.py --env HalfCheetah-v4 --benchmark gym --seed 0
    python main.py --env cheetah-run --benchmark dmc_proprio --seed 0
    python main.py --env Alien --benchmark atari --seed 0 --ablation mse_reward
    python main.py --env cheetah-run --benchmark dmc_visual --seed 0 --ablation no_unroll
"""

import argparse
import csv
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from agent import MRQAgent
from config import (
    ATARI_HUMAN_SCORES,
    ATARI_RANDOM_SCORES,
    GYM_RANDOM_SCORES,
    GYM_TD3_SCORES,
    VALID_ABLATIONS,
    VALID_BENCHMARKS,
    Config,
)
from envs import EnvWrapper
from utils import (
    compute_aggregate_metrics,
    compute_human_normalized,
    compute_td3_normalized,
    set_seed,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MR.Q experiments.

    Returns:
        Parsed argument namespace with all CLI options.
    """
    parser = argparse.ArgumentParser(
        description="MR.Q: Model-based Representations for Q-learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help=(
            "Environment name. "
            "Gym: 'HalfCheetah-v4'. "
            "DMC: 'cheetah-run'. "
            "Atari: 'Alien'."
        ),
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(VALID_BENCHMARKS),
        help="Benchmark category.",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help=(
            "Total environment interaction steps. "
            "Overrides benchmark default if provided. "
            "Defaults: Gym=1M, DMC=500k, Atari=2.5M."
        ),
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=[
            "none",
            "linear_value",
            "dynamics_target",
            "no_target_encoder",
            "revert",
            "nonlinear_model",
            "mse_reward",
            "no_reward_scaling",
            "no_min",
            "no_lap",
            "no_mr",
            "1step_return",
            "no_unroll",
        ],
        help="Ablation variant to run (Section 5.2 of the paper).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device. Falls back to 'cpu' if CUDA is unavailable.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="Root directory for saving results, checkpoints, and logs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1000,
        help="Log training metrics every this many steps.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="Number of episodes per evaluation.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration construction
# ---------------------------------------------------------------------------


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load and parse the YAML configuration file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Parsed YAML as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: '{config_path}'. "
            f"Ensure config.yaml is in the working directory or provide "
            f"the correct path via --config."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten_yaml_config(yaml_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the nested YAML config into a flat dict matching Config fields.

    Maps YAML section keys to Config dataclass field names. Only extracts
    fields that correspond to Config attributes; nested benchmark-specific
    and ablation sections are handled separately.

    Args:
        yaml_cfg: Nested YAML dictionary loaded from config.yaml.

    Returns:
        Flat dictionary with Config field names as keys.
    """
    flat: Dict[str, Any] = {}

    # Encoder section
    enc = yaml_cfg.get("encoder", {})
    flat["lambda_dynamics"] = enc.get("dynamics_loss_weight", 1.0)
    flat["lambda_reward"] = enc.get("reward_loss_weight", 0.1)
    flat["lambda_terminal"] = enc.get("terminal_loss_weight", 0.1)
    flat["enc_horizon"] = enc.get("horizon", 5)
    flat["enc_lr"] = enc.get("learning_rate", 1e-4)
    flat["enc_weight_decay"] = enc.get("weight_decay", 1e-4)
    flat["zs_dim"] = enc.get("zs_dim", 512)
    flat["zsa_dim"] = enc.get("zsa_dim", 512)
    flat["za_dim"] = enc.get("za_dim", 256)
    flat["hidden_dim"] = enc.get("hidden_dim", 512)
    flat["reward_bins"] = enc.get("reward_bins", 65)
    flat["reward_range"] = (
        enc.get("reward_range_low", -10.0),
        enc.get("reward_range_high", 10.0),
    )

    # Value section
    val = yaml_cfg.get("value", {})
    flat["value_lr"] = val.get("learning_rate", 3e-4)
    flat["grad_clip_norm"] = val.get("gradient_clip_norm", 20.0)

    # Policy section
    pol = yaml_cfg.get("policy", {})
    flat["lambda_pre_activ"] = pol.get("pre_activation_loss_weight", 1e-5)
    flat["policy_lr"] = pol.get("learning_rate", 3e-4)
    flat["gumbel_tau"] = pol.get("gumbel_softmax_tau", 10.0)

    # TD3 section
    td3 = yaml_cfg.get("td3", {})
    flat["hq_horizon"] = td3.get("multi_step_horizon", 3)
    flat["target_noise_std"] = td3.get("target_policy_noise_std", 0.2)
    flat["target_noise_clip"] = td3.get("target_policy_noise_clip", 0.3)

    # LAP section
    lap = yaml_cfg.get("lap", {})
    flat["lap_alpha"] = lap.get("priority_alpha", 0.4)
    flat["lap_min_priority"] = lap.get("min_priority", 1.0)

    # Exploration section
    expl = yaml_cfg.get("exploration", {})
    flat["initial_random_steps"] = expl.get("initial_random_steps", 10_000)
    flat["explore_noise_std"] = expl.get("noise_std", 0.2)

    # Common section
    common = yaml_cfg.get("common", {})
    flat["discount"] = common.get("discount", 0.99)
    flat["replay_capacity"] = common.get("replay_buffer_capacity", 1_000_000)
    flat["batch_size"] = common.get("batch_size", 256)
    flat["target_update_freq"] = common.get("target_update_freq", 250)
    flat["replay_ratio"] = common.get("replay_ratio", 1)

    return flat


def _apply_ablation_to_flat_dict(
    flat: Dict[str, Any], ablation: str
) -> Dict[str, Any]:
    """Apply ablation variant overrides to the flat config dictionary.

    Maps ablation variant names to their corresponding field overrides as
    specified in config.yaml's ablations section and the Logic Analysis.

    The '1step_return' CLI string is mapped to hq_horizon=1 here.

    Args:
        flat: Flat config dictionary to modify in-place.
        ablation: Ablation variant name string from CLI.

    Returns:
        Modified flat config dictionary.
    """
    if ablation == "none":
        pass

    elif ablation == "linear_value":
        flat["value_linear"] = True

    elif ablation == "dynamics_target":
        flat["use_sa_dynamics_target"] = True

    elif ablation == "no_target_encoder":
        flat["use_target_encoder"] = False

    elif ablation == "revert":
        flat["value_linear"] = True
        flat["use_sa_dynamics_target"] = True
        flat["use_target_encoder"] = False

    elif ablation == "nonlinear_model":
        flat["nonlinear_model"] = True

    elif ablation == "mse_reward":
        flat["use_mse_reward"] = True

    elif ablation == "no_reward_scaling":
        flat["use_reward_scaling"] = False

    elif ablation == "no_min":
        flat["use_min_q"] = False

    elif ablation == "no_lap":
        flat["use_lap"] = False

    elif ablation == "no_mr":
        flat["use_encoder_loss"] = False

    elif ablation == "1step_return":
        flat["hq_horizon"] = 1

    elif ablation == "no_unroll":
        flat["enc_horizon"] = 1

    # Store the ablation name (map '1step_return' to 'one_step_return' for
    # Config validation — Config.VALID_ABLATIONS uses 'one_step_return').
    ablation_for_config: str = ablation
    if ablation == "1step_return":
        ablation_for_config = "one_step_return"
    flat["ablation"] = ablation_for_config

    return flat


def build_config(args: argparse.Namespace) -> Config:
    """Construct a fully-initialized Config from YAML file and CLI arguments.

    Performs the five-stage config construction described in the Logic Analysis:
      1. Load base config from YAML
      2. Apply benchmark-specific overrides
      3. Apply CLI overrides
      4. Apply ablation overrides
      5. Construct Config via Config.from_dict()

    Args:
        args: Parsed CLI arguments from parse_args().

    Returns:
        Fully initialized Config instance ready for use by Trainer.
    """
    # Stage 1: Load and flatten YAML config
    yaml_cfg: Dict[str, Any] = _load_yaml_config(args.config)
    flat: Dict[str, Any] = _flatten_yaml_config(yaml_cfg)

    # Stage 2: Apply benchmark-specific overrides
    env_overrides: Dict[str, Any] = Config.get_env_config(args.benchmark, args.env)
    flat.update(env_overrides)

    # Stage 3: Apply CLI overrides
    flat["env_name"] = args.env
    flat["benchmark"] = args.benchmark
    flat["seed"] = args.seed
    flat["eval_episodes"] = args.eval_episodes

    # Override total_steps if explicitly provided via CLI
    if args.total_steps is not None:
        flat["total_steps"] = args.total_steps

    # Device handling with CUDA availability check
    device_str: str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print(
            "Warning: CUDA requested but not available. "
            "Falling back to CPU."
        )
        device_str = "cpu"
    flat["device"] = device_str  # stored for reference; actual device created in Trainer

    # Stage 4: Apply ablation overrides
    flat = _apply_ablation_to_flat_dict(flat, args.ablation)

    # Stage 5: Construct Config via from_dict
    cfg: Config = Config.from_dict(flat)

    return cfg


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------


class Trainer:
    """Orchestrates the full MR.Q training loop.

    Manages environment creation, agent initialization, the training loop
    (random exploration + main learning), periodic evaluation, and result
    logging/saving.

    Attributes:
        cfg: Configuration dataclass with all hyperparameters.
        env: Training environment wrapper.
        eval_env: Evaluation environment wrapper (different seed).
        agent: MRQAgent instance.
        device: Torch device for all tensor operations.
        save_path: Directory for saving results and checkpoints.
        writer: TensorBoard SummaryWriter.
        results: List of evaluation result dicts.
        total_steps: Global step counter (Phase 2 only).
        episode_reward: Accumulated reward for the current episode.
        episode_steps: Step count for the current episode.
        episode_count: Total completed episodes.
        best_score: Best evaluation score seen so far.
        start_time: Wall-clock time at training start.
        csv_file: Open CSV file handle for incremental result writing.
        csv_writer: CSV DictWriter for the results file.
    """

    def __init__(self, cfg: Config) -> None:
        """Initialise the Trainer.

        Creates environments, agent, logging infrastructure, and save directory.

        Args:
            cfg: Fully initialized Config instance.
        """
        self.cfg: Config = cfg

        # ---------------------------------------------------------------
        # Device setup
        # ---------------------------------------------------------------
        device_str: str = getattr(cfg, "device", "cpu")
        if not isinstance(device_str, str):
            device_str = "cpu"
        self.device: torch.device = torch.device(device_str)

        # ---------------------------------------------------------------
        # Environment creation
        # ---------------------------------------------------------------
        print(
            f"Creating environments: {cfg.env_name} "
            f"(benchmark={cfg.benchmark}, seed={cfg.seed})"
        )
        self.env: EnvWrapper = EnvWrapper(
            env_name=cfg.env_name,
            benchmark=cfg.benchmark,
            seed=cfg.seed,
        )
        # Evaluation environment uses a different seed to avoid correlation
        # with the training environment's random state.
        self.eval_env: EnvWrapper = EnvWrapper(
            env_name=cfg.env_name,
            benchmark=cfg.benchmark,
            seed=cfg.seed + 1000,
        )

        # Extract environment properties for agent construction.
        state_shape: Tuple[int, ...] = self.env.state_shape
        action_dim: int = self.env.action_dim
        discrete: bool = self.env.discrete
        image_obs: bool = self.env.image_obs

        print(
            f"  state_shape={state_shape}, action_dim={action_dim}, "
            f"discrete={discrete}, image_obs={image_obs}"
        )

        # ---------------------------------------------------------------
        # Agent creation
        # ---------------------------------------------------------------
        self.agent: MRQAgent = MRQAgent(
            cfg=cfg,
            state_shape=state_shape,
            action_dim=action_dim,
            discrete=discrete,
            image_obs=image_obs,
            device=self.device,
        )

        # ---------------------------------------------------------------
        # Save directory and logging setup
        # ---------------------------------------------------------------
        # Sanitize env_name for use as a directory component
        # (replace characters that are invalid on some filesystems).
        safe_env_name: str = cfg.env_name.replace("/", "_").replace("\\", "_")
        ablation_dir: str = cfg.ablation if cfg.ablation else "none"

        self.save_path: str = os.path.join(
            cfg.save_dir,
            cfg.benchmark,
            safe_env_name,
            f"seed_{cfg.seed}",
            ablation_dir,
        )
        os.makedirs(self.save_path, exist_ok=True)
        print(f"Saving results to: {self.save_path}")

        # TensorBoard writer
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer: Optional[Any] = SummaryWriter(
                log_dir=self.save_path
            )
        except ImportError:
            print(
                "Warning: tensorboard not available. "
                "TensorBoard logging disabled."
            )
            self.writer = None

        # CSV file for incremental result writing
        csv_path: str = os.path.join(self.save_path, "results.csv")
        self._csv_file_handle = open(csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer: csv.DictWriter = csv.DictWriter(
            self._csv_file_handle,
            fieldnames=["step", "score", "time_elapsed"],
        )
        self.csv_writer.writeheader()
        self._csv_file_handle.flush()

        # ---------------------------------------------------------------
        # Training state
        # ---------------------------------------------------------------
        self.results: List[Dict[str, Any]] = []
        self.total_steps: int = 0
        self.episode_reward: float = 0.0
        self.episode_steps: int = 0
        self.episode_count: int = 0
        self.best_score: float = -math.inf
        self.start_time: float = time.time()

        # Save config at the start for reproducibility
        config_save_path: str = os.path.join(self.save_path, "config.json")
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)

        print(f"Config saved to: {config_save_path}")
        print(f"Training for {cfg.total_steps:,} steps with seed {cfg.seed}")
        print(f"Ablation: {cfg.ablation}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full training experiment.

        Executes the training loop and saves results. Ensures results are
        saved even if training is interrupted by a keyboard interrupt or
        other exception.
        """
        try:
            self._training_loop()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        except Exception as exc:
            print(f"\nTraining failed with exception: {exc}")
            raise
        finally:
            self._save_results()
            self._cleanup()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _training_loop(self) -> None:
        """Execute the full training loop: random exploration + main learning.

        Phase 1 (initial_random_steps): Collect transitions with random
        actions to warm up the replay buffer. No gradient updates.

        Phase 2 (total_steps): Main learning loop. Select actions with
        exploration noise, store transitions, update agent, evaluate
        periodically.
        """
        # ---------------------------------------------------------------
        # Phase 1: Initial random exploration
        # ---------------------------------------------------------------
        print(
            f"\nPhase 1: Random exploration "
            f"({self.cfg.initial_random_steps:,} steps)..."
        )
        state: np.ndarray = self.env.reset()

        for _ in range(self.cfg.initial_random_steps):
            action: np.ndarray = self.env.sample_action()
            next_state: np.ndarray
            reward: float
            done: bool
            info: Dict[str, Any]
            next_state, reward, done, info = self.env.step(action)

            self.agent.store_transition(state, action, reward, done, next_state)

            if done:
                state = self.env.reset()
            else:
                state = next_state

        print(
            f"  Buffer filled: {len(self.agent.replay):,} transitions. "
            f"Terminal seen: {self.agent.replay.has_terminal()}"
        )

        # ---------------------------------------------------------------
        # Phase 2: Main training loop
        # ---------------------------------------------------------------
        print(f"\nPhase 2: Main training ({self.cfg.total_steps:,} steps)...")
        state = self.env.reset()
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_count = 0

        last_log_time: float = time.time()
        last_metrics: Dict[str, float] = {}

        for step in range(1, self.cfg.total_steps + 1):
            self.total_steps = step

            # ----------------------------------------------------------
            # Action selection with exploration noise
            # ----------------------------------------------------------
            action = self.agent.select_action(state, explore=True)

            # ----------------------------------------------------------
            # Environment interaction
            # ----------------------------------------------------------
            next_state, reward, done, info = self.env.step(action)

            # ----------------------------------------------------------
            # Store transition
            # ----------------------------------------------------------
            self.agent.store_transition(state, action, reward, done, next_state)

            # ----------------------------------------------------------
            # Episode tracking
            # ----------------------------------------------------------
            self.episode_reward += reward
            self.episode_steps += 1

            # ----------------------------------------------------------
            # Learning update
            # Guard: buffer must have enough samples for both sample()
            # and sample_sequences(). After initial_random_steps=10000,
            # this is always satisfied for batch_size=256 and seq_len=6.
            # ----------------------------------------------------------
            if len(self.agent.replay) >= self.cfg.batch_size:
                try:
                    last_metrics = self.agent.update()
                except ValueError as e:
                    # sample_sequences may fail if no valid episode windows
                    # exist yet. Skip this update and continue.
                    if "No valid episode windows" in str(e) or "batch_size" in str(e):
                        pass
                    else:
                        raise

            # ----------------------------------------------------------
            # Periodic training metric logging
            # ----------------------------------------------------------
            if step % self.cfg.log_interval == 0 and last_metrics:
                if self.writer is not None:
                    for key, val in last_metrics.items():
                        self.writer.add_scalar(
                            f"train/{key}", val, step
                        )

                elapsed: float = time.time() - last_log_time
                steps_per_sec: float = self.cfg.log_interval / max(elapsed, 1e-6)
                last_log_time = time.time()

                enc_loss: float = last_metrics.get("encoder_loss", 0.0)
                val_loss: float = last_metrics.get("value_loss", 0.0)
                pol_loss: float = last_metrics.get("policy_loss", 0.0)
                print(
                    f"  Step {step:>8,}/{self.cfg.total_steps:,} "
                    f"| {steps_per_sec:.0f} steps/s "
                    f"| enc={enc_loss:.4f} "
                    f"| val={val_loss:.4f} "
                    f"| pol={pol_loss:.4f}"
                )

            # ----------------------------------------------------------
            # Periodic evaluation
            # ----------------------------------------------------------
            if step % self.cfg.eval_freq == 0:
                score: float = self._evaluate()
                self._log_results(step, score, last_metrics)

            # ----------------------------------------------------------
            # Episode reset
            # ----------------------------------------------------------
            if done:
                if self.writer is not None:
                    self.writer.add_scalar(
                        "train/episode_reward",
                        self.episode_reward,
                        step,
                    )
                    self.writer.add_scalar(
                        "train/episode_length",
                        self.episode_steps,
                        step,
                    )

                self.episode_reward = 0.0
                self.episode_steps = 0
                self.episode_count += 1
                state = self.env.reset()
            else:
                state = next_state

        print(
            f"\nTraining complete. "
            f"Best score: {self.best_score:.2f} "
            f"| Total episodes: {self.episode_count}"
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self) -> float:
        """Run evaluation episodes and return the mean episode return.

        Delegates to agent.evaluate() which runs cfg.eval_episodes episodes
        with deterministic (no-noise) action selection.

        Returns:
            Mean episode return over eval_episodes episodes.
        """
        score: float = self.agent.evaluate(
            self.eval_env, self.cfg.eval_episodes
        )
        return score

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_results(
        self,
        step: int,
        score: float,
        metrics: Dict[str, float],
    ) -> None:
        """Log evaluation results to TensorBoard, CSV, and console.

        Also saves a checkpoint if this is the best score seen so far.

        Args:
            step: Current training step.
            score: Mean evaluation episode return.
            metrics: Latest training loss metrics dict.
        """
        elapsed: float = time.time() - self.start_time

        # TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("eval/score", score, step)
            self.writer.add_scalar("eval/best_score", self.best_score, step)

        # CSV (incremental write)
        row: Dict[str, Any] = {
            "step": step,
            "score": score,
            "time_elapsed": elapsed,
        }
        self.csv_writer.writerow(row)
        self._csv_file_handle.flush()

        # In-memory results list
        result_entry: Dict[str, Any] = {
            "step": step,
            "score": score,
            "time_elapsed": elapsed,
        }
        self.results.append(result_entry)

        # Console output
        time_str: str = _format_time(elapsed)
        print(
            f"  [Eval] Step {step:>8,}/{self.cfg.total_steps:,} "
            f"| Score: {score:>10.2f} "
            f"| Best: {self.best_score:>10.2f} "
            f"| Time: {time_str}"
        )

        # Save best model checkpoint
        if score > self.best_score:
            self.best_score = score
            best_model_path: str = os.path.join(
                self.save_path, "best_model.pt"
            )
            self.agent.save(best_model_path)
            print(f"    → New best! Checkpoint saved to {best_model_path}")

    # ------------------------------------------------------------------
    # Result saving
    # ------------------------------------------------------------------

    def _save_results(self) -> None:
        """Save full results, config, and final model checkpoint to disk.

        Writes:
          - results.json: Full results with config, learning curve, best score
          - final_model.pt: Agent checkpoint at end of training
        """
        # Save final model
        final_model_path: str = os.path.join(self.save_path, "final_model.pt")
        try:
            self.agent.save(final_model_path)
            print(f"Final model saved to: {final_model_path}")
        except Exception as e:
            print(f"Warning: Could not save final model: {e}")

        # Compute normalized scores if applicable
        normalized_results: Optional[Dict[str, Any]] = None
        if self.results:
            final_score: float = self.results[-1]["score"]
            normalized_results = _compute_normalized_score(
                env_name=self.cfg.env_name,
                benchmark=self.cfg.benchmark,
                score=final_score,
            )

        # Save full results JSON
        results_json: Dict[str, Any] = {
            "config": self.cfg.to_dict(),
            "results": self.results,
            "best_score": self.best_score,
            "total_steps": self.total_steps,
            "total_time_seconds": time.time() - self.start_time,
            "normalized_score": normalized_results,
        }
        json_path: str = os.path.join(self.save_path, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2)
        print(f"Results saved to: {json_path}")

        # Print final summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Environment:  {self.cfg.env_name}")
        print(f"  Benchmark:    {self.cfg.benchmark}")
        print(f"  Seed:         {self.cfg.seed}")
        print(f"  Ablation:     {self.cfg.ablation}")
        print(f"  Total steps:  {self.total_steps:,}")
        print(f"  Best score:   {self.best_score:.2f}")
        if self.results:
            print(f"  Final score:  {self.results[-1]['score']:.2f}")
        if normalized_results is not None:
            for key, val in normalized_results.items():
                if isinstance(val, float):
                    print(f"  {key}: {val:.4f}")
        elapsed_total: float = time.time() - self.start_time
        print(f"  Total time:   {_format_time(elapsed_total)}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Release all resources: environments, TensorBoard writer, CSV file."""
        try:
            self.env.close()
        except Exception:
            pass
        try:
            self.eval_env.close()
        except Exception:
            pass
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
        try:
            self._csv_file_handle.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _format_time(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted string like '1h 23m 45s' or '5m 30s' or '45s'.
    """
    seconds = int(seconds)
    hours: int = seconds // 3600
    minutes: int = (seconds % 3600) // 60
    secs: int = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{