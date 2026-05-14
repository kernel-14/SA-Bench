"""Configuration for SCoRe training and baselines."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SCoReConfig:
    """Hyperparameters from Table 5 of the paper."""

    # Model
    base_model: str = "gemini-1.5-flash"  # or gemini-1.0-pro

    # Task-specific defaults (MATH)
    task: str = "math"  # "math" or "code"

    # Optimizer
    optimizer: str = "adam"
    learning_rate: float = 5e-6  # 5e-6 for MATH, 1e-5 for code
    learning_rate_code: float = 1e-5

    # Training steps
    training_steps: int = 3000  # 3000 for MATH, 1500 for code
    training_steps_code: int = 1500

    # Batch size
    batch_size: int = 512  # 512 for MATH, 128 for code
    batch_size_code: int = 128

    # Sampling temperature during training
    sampling_temperature: float = 1.0

    # Reward shaping parameter α (Equation 5 in paper)
    alpha: float = 10.0

    # KL penalty coefficient β₁ (default in both stages)
    beta1: float = 0.01

    # Stage I KL penalty coefficient β₂ (constrains first-turn to base model)
    beta2: float = 0.1  # 0.1 for MATH, 0.25 for code
    beta2_code: float = 0.25

    # Discount factor γ (used optionally; paper finds γ=0 works best)
    gamma: float = 0.0

    # Number of turns (l+1 in the paper; default l=1 => 2 turns)
    num_turns: int = 2

    # Max sequence length
    max_seq_length: int = 2048

    # Max generation length (new tokens)
    max_new_tokens: int = 1024

    # Evaluation
    eval_temperature: float = 0.0  # greedy decoding for evaluation

    # Gradient accumulation
    gradient_accumulation_steps: int = 1

    # Stage I training steps (paper doesn't specify exact split; we use half)
    stage1_steps_ratio: float = 0.5

    # Offline data mixing: incorporate base-model first attempts in RL
    mix_base_first_attempts: bool = True
    base_first_attempt_ratio: float = 0.3

    # Reward: binary correctness
    use_binary_reward: bool = True

    # Seed
    seed: int = 42


@dataclass
class STARConfig:
    """Configuration for STaR baseline."""

    num_iterations: int = 3  # 3 iterations following Singh et al. 2024
    training_steps_per_iteration: int = 1000
    batch_size: int = 512
    learning_rate: float = 5e-6

    # Filtering: keep only trajectories that correct an incorrect response
    filter_successful_corrections: bool = True

    # Extended dataset D_STaR^+ with correct→correct pairs
    include_correct_to_correct: bool = False


@dataclass
class PairSFTConfig:
    """Configuration for Pair-SFT baseline (Welleck et al. 2023)."""

    training_steps: int = 1000
    batch_size: int = 512
    learning_rate: float = 5e-6

    # Extended dataset D_SFT^+ with correct→correct pairs
    include_correct_to_correct: bool = False


@dataclass
class AblationConfig:
    """Configuration for ablation studies from Section 6.3."""

    # Single-turn RL (no multi-turn training)
    single_turn: bool = False

    # No Stage I (skip directly to Stage II)
    skip_stage1: bool = False

    # No reward shaping (α=0)
    no_reward_shaping: bool = False

    # Use STaR instead of REINFORCE in Stage II
    use_star_stage2: bool = False


def get_config(
    task: str = "math",
    ablation: Optional[str] = None,
) -> SCoReConfig:
    """Get configuration for a given task and ablation setting."""
    cfg = SCoReConfig()

    if task == "code":
        cfg.base_model = "gemini-1.0-pro"
        cfg.learning_rate = cfg.learning_rate_code
        cfg.training_steps = cfg.training_steps_code
        cfg.batch_size = cfg.batch_size_code
        cfg.beta2 = cfg.beta2_code
    else:
        cfg.base_model = "gemini-1.5-flash"
        cfg.beta2 = 0.1

    if ablation == "single_turn":
        cfg.num_turns = 1
    elif ablation == "skip_stage1":
        cfg.stage1_steps_ratio = 0.0
    elif ablation == "no_reward_shaping":
        cfg.alpha = 0.0
    elif ablation == "star_stage2":
        pass  # handled differently in training

    return cfg
