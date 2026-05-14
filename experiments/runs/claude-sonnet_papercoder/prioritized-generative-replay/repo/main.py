## main.py
"""Entry point for Prioritized Generative Replay (PGR) experiments.

Wires together all components via Hydra config management. Responsibilities:
    1. Seed initialization for full reproducibility
    2. Device validation (CUDA fallback to CPU)
    3. Scaling experiment config overrides
    4. PGRTrainer instantiation and training
    5. Optional baseline comparison via run_baselines()

Usage examples:
    # Default: PGR (curiosity) on quadruped-walk (state-based DMC)
    python main.py

    # Switch environment
    python main.py env.name=cheetah-run

    # Switch config file (OpenAI Gym)
    python main.py --config-name=gym env.name=Walker2d-v2

    # Switch relevance function
    python main.py relevance.type=td_error

    # Enable scaling experiment (larger network)
    python main.py scaling.larger_network.enabled=true

    # Harder sparse-reward task (300K steps)
    python main.py env.name=finger-turn-hard training.total_steps=300000

    # Pixel-based DMC
    python main.py --config-name=dmc_pixel env.name=walker-walk

    # Run SYNTHER baseline (unconditional generative replay)
    # (handled via run_baselines() or direct override)
    python main.py diffusion.guidance_scale=0.0 relevance.type=reward
"""

import os
import random
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from pgr_trainer import PGRTrainer


# ---------------------------------------------------------------------------
# Seed utilities
# ---------------------------------------------------------------------------


def set_seeds(seed: int) -> None:
    """Sets random seeds for all relevant RNG systems for reproducibility.

    Seeds PyTorch (CPU and CUDA), NumPy, and Python's built-in random module.
    Also configures cuDNN for deterministic behavior at the cost of some
    performance — acceptable for research reproducibility.

    Args:
        seed: Integer random seed. Corresponds to config.training.seed
            (default 0 in config.yaml). The same seed is used for all
            RNG systems to ensure fully deterministic training runs.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Safe even if CUDA is unavailable.
    np.random.seed(seed)
    random.seed(seed)

    # cuDNN determinism: ensures identical results across runs at the cost
    # of some performance. Required for strict reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device validation
# ---------------------------------------------------------------------------


def validate_device(cfg: DictConfig) -> DictConfig:
    """Validates the requested device and falls back to CPU if CUDA is unavailable.

    Checks whether CUDA is available when cfg.hardware.device == "cuda".
    If not, updates the config to use "cpu" and prints a warning. This
    prevents cryptic CUDA errors downstream when running on CPU-only machines.

    Args:
        cfg: Hydra/OmegaConf DictConfig. The hardware.device field is
            checked and potentially updated.

    Returns:
        Updated DictConfig with a valid device string. If CUDA was requested
        but is unavailable, hardware.device is set to "cpu".
    """
    requested_device: str = str(cfg.hardware.device)

    if requested_device == "cuda" and not torch.cuda.is_available():
        print(
            "[WARNING] CUDA requested (hardware.device='cuda') but CUDA is not "
            "available on this machine. Falling back to CPU. Training will be "
            "significantly slower. Set hardware.device='cpu' in config.yaml to "
            "suppress this warning."
        )
        # OmegaConf.update allows mutating a DictConfig field by dotted path.
        # This is safe here because we own the config object.
        OmegaConf.update(cfg, "hardware.device", "cpu", merge=True)

    elif requested_device == "cuda":
        gpu_name: str = torch.cuda.get_device_name(0)
        gpu_memory_gb: float = torch.cuda.get_device_properties(0).total_memory / (
            1024 ** 3
        )
        print(
            f"[INFO] Using CUDA device: {gpu_name} "
            f"({gpu_memory_gb:.1f} GB VRAM). "
            f"Note: PGR generation requires ~6.67 GB VRAM (Table 3 of paper)."
        )

    else:
        print(f"[INFO] Using device: {requested_device}")

    return cfg


# ---------------------------------------------------------------------------
# Scaling experiment config overrides
# ---------------------------------------------------------------------------


def apply_scaling_overrides(cfg: DictConfig) -> DictConfig:
    """Applies scaling experiment hyperparameter overrides to the config.

    Checks the three scaling flags from config.yaml (Section 5.3 of the paper)
    and mutates the config accordingly. Applied in priority order:
        combined > high_syn_ratio > larger_network

    Only one scaling experiment should be enabled at a time. If multiple are
    enabled, combined takes priority, then high_syn_ratio, then larger_network.

    Scaling experiments (paper Section 5.3):
        (a) larger_network: hidden_dim=512, hidden_layers=3, batch_size=1024
        (b) high_syn_ratio: synthetic_ratio=0.75, batch_size=512
        (c) combined: (a) + (b) + utd_ratio=40, syn_capacity=2M

    Args:
        cfg: Hydra/OmegaConf DictConfig. Scaling flags are read from
            cfg.scaling.{larger_network,high_syn_ratio,combined}.enabled.

    Returns:
        Updated DictConfig with scaling overrides applied. If no scaling
        flag is enabled, the config is returned unchanged.
    """
    # Guard: check if scaling section exists in config.
    if not hasattr(cfg, "scaling"):
        return cfg

    scaling = cfg.scaling

    # ── Priority 1: Combined scaling (a) + (b) + UTD=40 ──────────────────────
    # Paper Section 5.3: "we combine the above two insights, and show that this
    # allows us to push the UTD ratio of PGR to new heights."
    if (
        hasattr(scaling, "combined")
        and hasattr(scaling.combined, "enabled")
        and bool(scaling.combined.enabled)
    ):
        print(
            "[INFO] Scaling experiment: COMBINED "
            "(larger network + high syn ratio + UTD=40)"
        )
        OmegaConf.update(
            cfg, "policy.hidden_dim",
            int(scaling.combined.hidden_dim),
            merge=True,
        )
        OmegaConf.update(
            cfg, "policy.hidden_layers",
            int(scaling.combined.hidden_layers),
            merge=True,
        )
        OmegaConf.update(
            cfg, "sampling.synthetic_ratio",
            float(scaling.combined.synthetic_ratio),
            merge=True,
        )
        OmegaConf.update(
            cfg, "sampling.batch_size",
            int(scaling.combined.batch_size),
            merge=True,
        )
        OmegaConf.update(
            cfg, "policy.utd_ratio",
            int(scaling.combined.utd_ratio),
            merge=True,
        )
        OmegaConf.update(
            cfg, "buffer.syn_capacity",
            int(scaling.combined.syn_capacity),
            merge=True,
        )
        return cfg

    # ── Priority 2: High synthetic data ratio ─────────────────────────────────
    # Paper Section 5.3: "we double the batch size to 512 and then to 1024,
    # each time scaling r to 0.75 and 0.875, respectively."
    if (
        hasattr(scaling, "high_syn_ratio")
        and hasattr(scaling.high_syn_ratio, "enabled")
        and bool(scaling.high_syn_ratio.enabled)
    ):
        print(
            "[INFO] Scaling experiment: HIGH SYNTHETIC RATIO "
            f"(r={scaling.high_syn_ratio.synthetic_ratio}, "
            f"batch={scaling.high_syn_ratio.batch_size})"
        )
        OmegaConf.update(
            cfg, "sampling.synthetic_ratio",
            float(scaling.high_syn_ratio.synthetic_ratio),
            merge=True,
        )
        OmegaConf.update(
            cfg, "sampling.batch_size",
            int(scaling.high_syn_ratio.batch_size),
            merge=True,
        )
        return cfg

    # ── Priority 3: Larger network ────────────────────────────────────────────
    # Paper Section 5.3: "we increase the number of hidden layers from 2 to 3,
    # and their widths from 256 to 512. This results in ~6x more parameters,
    # so we also increase batch size from 256 to 1024."
    if (
        hasattr(scaling, "larger_network")
        and hasattr(scaling.larger_network, "enabled")
        and bool(scaling.larger_network.enabled)
    ):
        print(
            "[INFO] Scaling experiment: LARGER NETWORK "
            f"(hidden_dim={scaling.larger_network.hidden_dim}, "
            f"hidden_layers={scaling.larger_network.hidden_layers}, "
            f"batch={scaling.larger_network.batch_size})"
        )
        OmegaConf.update(
            cfg, "policy.hidden_dim",
            int(scaling.larger_network.hidden_dim),
            merge=True,
        )
        OmegaConf.update(
            cfg, "policy.hidden_layers",
            int(scaling.larger_network.hidden_layers),
            merge=True,
        )
        OmegaConf.update(
            cfg, "sampling.batch_size",
            int(scaling.larger_network.batch_size),
            merge=True,
        )
        return cfg

    # No scaling experiment enabled — return config unchanged.
    return cfg


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------


def run_baselines(cfg: DictConfig, baseline_type: str) -> None:
    """Runs a comparison baseline by overriding the config and training.

    Enables running SYNTHER, pure REDQ, and REDQ+CURIOSITY baselines without
    modifying the main PGR config. Each baseline is implemented by overriding
    specific config fields before instantiating PGRTrainer.

    Baseline implementations:

    "synther" — Unconditional generative replay (Lu et al., 2024):
        Sets guidance_scale=0.0 so the CFG formula reduces to unconditional
        generation: ε = 0.0 * ε_cond + 1.0 * ε_uncond = ε_uncond.
        Sets relevance.type="reward" (minimal relevance, effectively unused
        since guidance_scale=0 means the condition has no effect on generation).
        This is the direct predecessor to PGR and the primary comparison baseline.

    "redq" — Pure model-free REDQ (no diffusion):
        Sets synthetic_ratio=0.0 so MixedSampler samples 100% from D_real.
        Sets inner_loop_freq beyond total_steps so the diffusion inner loop
        never fires. The policy trains on real data only with UTD=20.

    "redq_curiosity" — REDQ with ICM exploration bonus (Fig. 3b baseline):
        Same as "redq" but adds intrinsic curiosity reward to the environment
        reward during data collection. The ICM score is weighted by
        exploration.intrinsic_reward_weight=0.1 (paper Appendix B.1).

    Args:
        cfg: Hydra/OmegaConf DictConfig. A deep copy is made before overrides
            to avoid mutating the original config.
        baseline_type: String identifier for the baseline to run. Must be one
            of: "synther", "redq", "redq_curiosity".

    Raises:
        ValueError: If baseline_type is not one of the supported options.
    """
    # Deep copy to avoid mutating the original config.
    # OmegaConf.to_container + OmegaConf.create gives a mutable copy.
    cfg_copy: DictConfig = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    )

    if baseline_type == "synther":
        # SYNTHER: unconditional generative replay.
        # guidance_scale=0.0 → ε = 0*ε_cond + 1*ε_uncond = ε_uncond (unconditional).
        # relevance.type="reward" is a minimal relevance function that has no
        # effect on generation when guidance_scale=0.
        print("[INFO] Running SYNTHER baseline (unconditional generative replay).")
        OmegaConf.update(cfg_copy, "diffusion.guidance_scale", 0.0, merge=True)
        OmegaConf.update(cfg_copy, "relevance.type", "reward", merge=True)

    elif baseline_type == "redq":
        # Pure REDQ: no diffusion, no synthetic data.
        # synthetic_ratio=0.0 → MixedSampler samples 100% from D_real.
        # inner_loop_freq > total_steps → diffusion inner loop never fires.
        print("[INFO] Running pure REDQ baseline (no diffusion).")
        OmegaConf.update(cfg_copy, "sampling.synthetic_ratio", 0.0, merge=True)
        # Disable inner loop by setting frequency beyond total training steps.
        total_steps: int = int(cfg_copy.training.total_steps)
        OmegaConf.update(
            cfg_copy, "diffusion.inner_loop_freq", total_steps + 1, merge=True
        )

    elif baseline_type == "redq_curiosity":
        # REDQ + CURIOSITY: pure REDQ with ICM exploration bonus added to reward.
        # Paper Appendix B.1: "we follow the hyperparameters of Pathak et al. (2017)
        # and set the intrinsic reward weight to 0.1."
        print(
            "[INFO] Running REDQ + CURIOSITY baseline "
            "(no diffusion, ICM exploration bonus weight=0.1)."
        )
        OmegaConf.update(cfg_copy, "sampling.synthetic_ratio", 0.0, merge=True)
        total_steps_rc: int = int(cfg_copy.training.total_steps)
        OmegaConf.update(
            cfg_copy, "diffusion.inner_loop_freq", total_steps_rc + 1, merge=True
        )
        # Set intrinsic reward weight for the exploration bonus.
        # PGRTrainer._collect_transition() reads this to add ICM score to reward.
        OmegaConf.update(
            cfg_copy,
            "exploration.intrinsic_reward_weight",
            0.1,
            merge=True,
        )
        # Keep curiosity relevance type so ICM is still instantiated and updated.
        OmegaConf.update(cfg_copy, "relevance.type", "curiosity", merge=True)

    else:
        raise ValueError(
            f"Unknown baseline_type '{baseline_type}'. "
            "Must be one of: 'synther', 'redq', 'redq_curiosity'."
        )

    # Instantiate and run the baseline trainer.
    trainer: PGRTrainer = PGRTrainer(cfg_copy)
    trainer.train()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="configs", config_name="dmc_state", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for PGR experiments.

    Orchestrates the full training pipeline:
        1. Seed all RNG systems for reproducibility
        2. Validate and potentially override the device setting
        3. Apply scaling experiment config overrides (if any flag is enabled)
        4. Print the resolved configuration for experiment tracking
        5. Instantiate PGRTrainer (handles all sub-component construction)
        6. Run training (outer loop + inner loop + policy updates)
        7. Run final evaluation and save the final checkpoint

    All training logic is encapsulated in PGRTrainer. This function is
    intentionally thin — it only handles setup and delegation.

    Args:
        cfg: Hydra/OmegaConf DictConfig loaded from configs/dmc_state.yaml
            (or the config specified via --config-name). All hyperparameters
            are read from this object. Command-line overrides are applied
            automatically by Hydra before this function is called.
    """
    # ── Step 1: Seed all RNG systems ──────────────────────────────────────────
    # Use training.seed as the primary seed. cfg.env.seed is also 0 by default
    # and is passed to the environment wrappers inside PGRTrainer.__init__.
    seed: int = int(cfg.training.seed)
    set_seeds(seed)
    print(f"[INFO] Random seed set to {seed}.")

    # ── Step 2: Validate device ───────────────────────────────────────────────
    # Falls back to CPU if CUDA is requested but unavailable.
    cfg = validate_device(cfg)

    # ── Step 3: Apply scaling experiment overrides ────────────────────────────
    # Must be applied before PGRTrainer instantiation since PGRTrainer reads
    # cfg once in __init__ and does not re-check scaling flags dynamically.
    cfg = apply_scaling_overrides(cfg)

    # ── Step 4: Print resolved configuration ─────────────────────────────────
    # OmegaConf.to_yaml gives a clean, human-readable representation of the
    # full resolved config (with all overrides applied). Useful for experiment
    # tracking and debugging.
    print("\n" + "=" * 60)
    print("PGR Experiment Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60 + "\n")

    # ── Step 5: Instantiate PGRTrainer ────────────────────────────────────────
    # PGRTrainer.__init__ constructs all sub-components:
    #   - DMCEnv or GymEnv
    #   - ReplayBuffer (D_real and D_syn)
    #   - MixedSampler
    #   - REDQPolicy / SACPolicy / DRQv2Policy
    #   - ICMRelevance / RNDRelevance / policy-based relevance
    #   - ConditionalDiffusion (with DiffusionModel + DDPMScheduler + Normalizer)
    #   - Evaluator
    #   - Logger (W&B or TensorBoard + CSV)
    trainer: PGRTrainer = PGRTrainer(cfg)

    # ── Step 6: Run training ──────────────────────────────────────────────────
    # PGRTrainer.train() implements Algorithm 1 from the paper:
    #   Outer loop: collect transitions, update relevance function
    #   Inner loop (every 10K steps): retrain diffusion, generate D_syn
    #   Policy update: UTD=20 gradient steps per env step
    #   Periodic evaluation, analysis, and checkpointing
    trainer.train()

    # ── Step 7: Final evaluation ──────────────────────────────────────────────
    # Run a final deterministic evaluation after training completes.
    # PGRTrainer.train() already runs a final evaluation internally, but we
    # run one more here for explicit reporting in the main function output.
    final_eval_episodes: int = int(cfg.training.eval_episodes)
    final_return: float = trainer._evaluate(num_episodes=final_eval_episodes)
    print(
        f"\n[RESULT] Training complete. "
        f"Final evaluation return: {final_return:.4f} "
        f"(averaged over {final_eval_episodes} episodes)"
    )

    # ── Step 8: Save final checkpoint ────────────────────────────────────────
    # Save a final checkpoint with the fully trained model weights.
    # Hydra changes the working directory to the output dir, so relative paths
    # (e.g., "checkpoints/final.pt") land in the Hydra output directory.
    checkpoint_dir: str = str(cfg.logging.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_checkpoint_path: str = os.path.join(checkpoint_dir, "final.pt")
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"[INFO] Final checkpoint saved to: {final_checkpoint_path}")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Standard Hydra entry point pattern.
    # The @hydra.main decorator handles argument parsing and config loading.
    # version_base=None suppresses Hydra 1.3 deprecation warnings.
    main()
