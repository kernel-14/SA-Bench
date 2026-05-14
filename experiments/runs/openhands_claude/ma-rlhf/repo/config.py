from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Task identifiers
# ---------------------------------------------------------------------------
TASK_TLDR = "tldr"
TASK_HH_RLHF = "hh_rlhf"
TASK_WEBGPT = "webgpt"
TASK_APPS = "apps"

SUPPORTED_TASKS = [TASK_TLDR, TASK_HH_RLHF, TASK_WEBGPT, TASK_APPS]

# ---------------------------------------------------------------------------
# Dataset paths / HuggingFace identifiers
# ---------------------------------------------------------------------------
DATASET_PATHS = {
    TASK_TLDR: "openai/summarize_from_feedback",
    TASK_HH_RLHF: "Anthropic/hh-rlhf",
    TASK_WEBGPT: "openai/webgpt_comparisons",
    TASK_APPS: "codeparrot/apps",
}

# ---------------------------------------------------------------------------
# Default base models (paper Table 5)
# ---------------------------------------------------------------------------
DEFAULT_MODELS = {
    TASK_TLDR: "google/gemma-2b",
    TASK_HH_RLHF: "google/gemma-2b",
    TASK_WEBGPT: "google/gemma-2b",
    TASK_APPS: "google/codegemma-2b",
}

# ---------------------------------------------------------------------------
# Data split ratios (paper §B.2: 20% SFT / 40% RM / 40% PPO)
# ---------------------------------------------------------------------------
SFT_SPLIT_RATIO = 0.20
RM_SPLIT_RATIO = 0.40
PPO_SPLIT_RATIO = 0.40

# For APPS: 80% PPO (no RM stage)
APPS_PPO_SPLIT_RATIO = 0.80


@dataclass
class SFTConfig:
    """Hyperparameters for the Supervised Fine-Tuning stage (paper Table 5)."""

    model_name: str = "google/gemma-2b"
    task: str = TASK_TLDR
    output_dir: str = "outputs/sft"
    max_prompt_length: int = 512
    max_response_length: int = 512

    # Per-task batch sizes
    batch_size_tldr: int = 512
    batch_size_hh_rlhf: int = 512
    batch_size_webgpt: int = 64
    batch_size_apps_2b: int = 16
    batch_size_apps_7b: int = 32

    # Per-task epochs
    epochs_tldr: int = 3
    epochs_hh_rlhf: int = 3
    epochs_webgpt_2b: int = 3
    epochs_webgpt_7b: int = 5
    epochs_apps: int = 1

    # Per-task learning rates
    lr_tldr: float = 5e-5
    lr_hh_rlhf: float = 5e-5
    lr_webgpt_2b: float = 1e-4
    lr_webgpt_7b: float = 2e-5
    lr_apps_2b: float = 5e-6
    lr_apps_7b: float = 2e-6

    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_ratio_apps: float = 0.0

    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    seed: int = 42


@dataclass
class RMConfig:
    """Hyperparameters for the Reward Modeling stage (paper Table 5)."""

    model_name: str = "google/gemma-2b"
    task: str = TASK_TLDR
    sft_model_path: str = "outputs/sft"
    output_dir: str = "outputs/rm"
    max_prompt_length: int = 512
    max_response_length: int = 512

    # Per-task batch sizes
    batch_size_tldr: int = 64
    batch_size_hh_rlhf: int = 64
    batch_size_webgpt: int = 32
    batch_size_tldr_7b: int = 128
    batch_size_hh_rlhf_7b: int = 64

    # Per-task epochs
    epochs: int = 1
    epochs_webgpt_7b: int = 32

    # Per-task learning rates
    lr_2b: float = 1e-5
    lr_webgpt_2b: float = 2e-5
    lr_7b: float = 1e-6
    lr_27b: float = 8e-6

    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1

    gradient_checkpointing: bool = True
    bf16: bool = True
    seed: int = 42


@dataclass
class PPOConfig:
    """Hyperparameters for the PPO / MA-PPO stage (paper Table 5)."""

    model_name: str = "google/gemma-2b"
    task: str = TASK_TLDR
    policy_model_path: str = "outputs/sft"
    critic_model_path: str = "outputs/rm"
    reward_model_path: str = "outputs/rm"
    output_dir: str = "outputs/ppo"

    # Sequence lengths
    max_prompt_length: int = 512
    max_response_length: int = 512

    # Batch / rollout
    batch_size: int = 256
    rollout: int = 1
    ppo_epochs: int = 1

    # Learning rates
    policy_lr_2b: float = 1.5e-5
    policy_lr_7b: float = 1e-6
    policy_lr_27b: float = 7e-7
    policy_lr_apps_2b: float = 5e-7
    policy_lr_apps_7b: float = 5e-7

    critic_lr_2b: float = 1.5e-5
    critic_lr_7b: float = 1e-6
    critic_lr_27b: float = 1e-6
    critic_lr_apps_2b: float = 5e-5
    critic_lr_apps_7b: float = 5e-5

    # Epochs per task
    epochs_webgpt: int = 4
    epochs_default: int = 1

    # PPO clipping
    clip_ratio: float = 0.2

    # GAE parameters
    gae_lambda: float = 0.95
    gae_gamma: float = 1.0

    # KL penalty coefficient (β in paper Eq. 2)
    kl_coef_default: float = 0.05
    kl_coef_7b_tldr: float = 0.05   # reduced from 0.05 for stability (paper §B.2)
    kl_coef_7b_webgpt: float = 0.1
    kl_coef_27b: float = 0.1
    kl_coef_apps: float = 0.05

    # Sampling
    temperature: float = 0.8
    temperature_apps: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    top_k_apps: int = 5

    # Warmup
    warmup_steps: int = 200
    warmup_steps_apps: int = 20

    gradient_checkpointing: bool = True
    bf16: bool = True
    seed: int = 42

    # Macro action discount factor ρ (paper §3.2.2, set to 1)
    macro_reward_discount: float = 1.0


@dataclass
class MacroActionConfig:
    """Configuration for macro action termination and value estimation."""

    # Termination strategy: 'ngram' | 'randomized_ngram' | 'parser' | 'ppl'
    termination: str = "ngram"

    # Fixed n-gram length (used when termination='ngram')
    n_gram: int = 5

    # Randomized n-gram pool (used when termination='randomized_ngram')
    randomized_ngram_lengths: List[int] = field(default_factory=lambda: [2, 3, 5, 10])
    randomized_ngram_repeat_times: int = 3

    # Parsing-based cutoff threshold C (paper §3.2.1, §B.4)
    parser_cutoff: int = 5

    # Value function σ assignment: 'equal' | 'unit' | 'position_decayed'
    sigma_assignment: str = "equal"

    # When n_gram -> infinity (treat full sequence as one macro action)
    use_full_sequence: bool = False


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    # Number of validation instances for RM scoring
    rm_eval_samples: int = 2000

    # GPT-4 / human eval instances
    pairwise_eval_samples: int = 50

    # Best-of-N sampling
    best_of_n_values: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    best_of_n_temperatures: List[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    )

    # Temperature sweep for robustness analysis
    temperature_sweep: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    # GPT-4 model for evaluation
    gpt4_model: str = "gpt-4o-2024-05-13"

    # APPS test set size
    apps_test_size: int = 5000

    # pass@k values for code evaluation
    pass_at_k_values: List[int] = field(default_factory=lambda: [1, 5])


@dataclass
class CodeRewardConfig:
    """Adaptive compiler reward for program synthesis (paper §B.5, Eq. 5)."""

    compile_error_reward: float = -1.0
    runtime_error_reward: float = -0.6
    partial_pass_base: float = -0.3
    partial_pass_scale: float = 1.3
    # R(x,y) = -0.3 + 1.3 * (N_pass / (N_pass + N_fail))  if compiled
    # R(x,y) = -0.6                                         if runtime error
    # R(x,y) = -1.0                                         if compile error


# ---------------------------------------------------------------------------
# Convenience factory: build configs from a flat namespace (argparse)
# ---------------------------------------------------------------------------

def build_ppo_config_from_args(args) -> PPOConfig:
    cfg = PPOConfig()
    for key, val in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg


def build_macro_config_from_args(args) -> MacroActionConfig:
    cfg = MacroActionConfig()
    for key, val in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
