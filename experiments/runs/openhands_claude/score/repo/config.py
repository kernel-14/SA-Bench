from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MathConfig:
    """Hyperparameters for SCoRe on MATH (Table 5, left)."""
    task: str = "math"
    base_model: str = "google/gemma-2-9b-it"  # open-source proxy for Gemini 1.5 Flash
    optimizer: str = "adam"
    learning_rate: float = 5e-6
    training_steps: int = 3000
    batch_size: int = 512
    sampling_temperature: float = 1.0
    eval_temperature: float = 0.0  # greedy decoding at eval time
    alpha: float = 10.0            # reward shaping multiplier
    beta1: float = 0.01            # KL penalty weight (both turns, Stage II)
    beta2: float = 0.1             # KL penalty weight (first turn only, Stage I)
    max_new_tokens: int = 1024
    max_prompt_length: int = 1024
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    # Dataset splits following Lightman et al. (2023)
    train_problems_from_test: int = 4500  # augment train with 4500 from test
    eval_problems: int = 500              # MATH500 evaluation set
    # Offline base-model samples to augment Stage II (Section 5.3)
    use_offline_first_attempts: bool = True
    offline_samples_per_problem: int = 4
    # Training
    stage1_steps: int = 1000
    stage2_steps: int = 2000
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 10
    output_dir: str = "checkpoints/math"
    seed: int = 42


@dataclass
class MBPPConfig:
    """Hyperparameters for SCoRe on MBPP/HumanEval (Table 5, right)."""
    task: str = "mbpp"
    base_model: str = "google/codegemma-7b-it"  # open-source proxy for Gemini 1.0 Pro
    optimizer: str = "adam"
    learning_rate: float = 1e-5
    training_steps: int = 1500
    batch_size: int = 128
    sampling_temperature: float = 1.0
    eval_temperature: float = 0.0
    alpha: float = 10.0
    beta1: float = 0.01
    beta2: float = 0.25
    max_new_tokens: int = 1024
    max_prompt_length: int = 1024
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 50
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    use_offline_first_attempts: bool = True
    offline_samples_per_problem: int = 4
    stage1_steps: int = 500
    stage2_steps: int = 1000
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    output_dir: str = "checkpoints/mbpp"
    seed: int = 42


@dataclass
class STaRConfig:
    """Hyperparameters for STaR baseline."""
    task: str = "math"
    base_model: str = "google/gemma-2-9b-it"
    learning_rate: float = 5e-6
    num_iterations: int = 3          # 3 iterations following Singh et al. (2024)
    training_steps_per_iter: int = 1000
    batch_size: int = 32
    max_new_tokens: int = 1024
    max_prompt_length: int = 1024
    output_dir: str = "checkpoints/star"
    seed: int = 42
    # Whether to include correct-to-correct pairs (D_STaR+)
    include_correct_pairs: bool = False


@dataclass
class PairSFTConfig:
    """Hyperparameters for Pair-SFT baseline (Welleck et al., 2023)."""
    task: str = "math"
    base_model: str = "google/gemma-2-9b-it"
    learning_rate: float = 5e-6
    training_steps: int = 1000
    batch_size: int = 32
    max_new_tokens: int = 1024
    max_prompt_length: int = 1024
    output_dir: str = "checkpoints/pair_sft"
    seed: int = 42
    # Whether to include correct-to-correct pairs (D_SFT+)
    include_correct_pairs: bool = False


@dataclass
class SCoReConfig:
    """Top-level config that selects task-specific settings."""
    task: str = "math"              # "math" or "mbpp"
    stage: int = 2                  # 1 or 2
    stage1_checkpoint: Optional[str] = None
    math: MathConfig = field(default_factory=MathConfig)
    mbpp: MBPPConfig = field(default_factory=MBPPConfig)
    star: STaRConfig = field(default_factory=STaRConfig)
    pair_sft: PairSFTConfig = field(default_factory=PairSFTConfig)
    wandb_project: str = "score"
    wandb_run_name: Optional[str] = None
    use_wandb: bool = False

    def get_task_config(self):
        if self.task == "math":
            return self.math
        elif self.task in ("mbpp", "humaneval"):
            return self.mbpp
        else:
            raise ValueError(f"Unknown task: {self.task}")
