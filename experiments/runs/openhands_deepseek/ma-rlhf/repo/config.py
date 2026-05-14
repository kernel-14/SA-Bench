"""Configuration and hyperparameters for MA-RLHF reproduction.

All hyperparameters are sourced from Table 5 and the paper text.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class SFTConfig:
    """Supervised Fine-Tuning configuration."""
    batch_size: int = 512
    epochs: int = 3
    learning_rate: float = 5e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    max_seq_length: int = 1024


@dataclass
class RMConfig:
    """Reward Modeling configuration."""
    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    max_seq_length: int = 1024


@dataclass
class PPOConfig:
    """PPO / MA-PPO configuration."""
    batch_size: int = 256
    policy_learning_rate: float = 1.5e-5
    critic_learning_rate: float = 1.5e-5
    epochs: int = 1
    ppo_epochs: int = 1
    rollout: int = 1
    clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    gae_gamma: float = 1.0
    kl_coefficient: float = 0.05
    max_prompt_length: int = 512
    max_response_length: int = 512
    warmup_steps: int = 200
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = 50


@dataclass
class MAPPOConfig(PPOConfig):
    """MA-PPO specific configuration extending PPO config."""
    termination: Literal["ngram", "randomized_ngram", "parser", "ppl"] = "ngram"
    n_gram: int = 5
    n_gram_list: List[int] = field(default_factory=lambda: [2, 3, 5, 10])
    n_gram_repeat_times: int = 3
    parsing_cutoff: int = 5
    value_estimation: Literal["equal", "unit", "position_decayed"] = "equal"


@dataclass
class DatasetConfig:
    """Dataset-specific configuration."""
    name: str = "tldr"
    num_train_samples: Optional[int] = None
    num_eval_samples: int = 2000
    sft_split: float = 0.2
    rm_split: float = 0.4
    ppo_split: float = 0.4
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model_name: str = "google/gemma-2b"
    task: Literal["tldr", "hh-rlhf", "webgpt", "apps"] = "tldr"
    method: Literal["vanilla_ppo", "ma_ppo"] = "ma_ppo"
    output_dir: str = "./output"
    logging_dir: str = "./logs"
    sft: SFTConfig = field(default_factory=SFTConfig)
    rm: RMConfig = field(default_factory=RMConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    ma_ppo: MAPPOConfig = field(default_factory=MAPPOConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    use_deepspeed: bool = True
    deepspeed_config: Optional[str] = None
    use_wandb: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True


def get_gemma_2b_config(task: str = "tldr") -> ExperimentConfig:
    """Configuration for Gemma-2B model."""
    cfg = ExperimentConfig(
        model_name="google/gemma-2b",
        task=task,
        sft=SFTConfig(
            batch_size=512,
            epochs=3,
            learning_rate=5e-5,
            warmup_ratio=0.1,
        ),
        rm=RMConfig(
            batch_size=64,
            epochs=1,
            learning_rate=1e-5,
            warmup_ratio=0.1,
        ),
        ppo=PPOConfig(
            batch_size=256,
            policy_learning_rate=1.5e-5,
            critic_learning_rate=1.5e-5,
            epochs=1,
            ppo_epochs=1,
            rollout=1,
            clip_ratio=0.2,
            gae_lambda=0.95,
            gae_gamma=1.0,
            kl_coefficient=0.05,
            max_prompt_length=512,
            max_response_length=512,
            warmup_steps=200,
            temperature=0.8,
            top_p=1.0,
            top_k=50,
        ),
    )
    if task == "webgpt":
        cfg.sft.batch_size = 64
        cfg.sft.learning_rate = 1e-4
        cfg.rm.batch_size = 32
        cfg.rm.learning_rate = 2e-5
        cfg.ppo.epochs = 4
        cfg.ppo.kl_coefficient = 0.1
    if task == "hh-rlhf":
        cfg.rm.batch_size = 64
    return cfg


def get_gemma_7b_config(task: str = "tldr") -> ExperimentConfig:
    """Configuration for Gemma-7B model."""
    cfg = ExperimentConfig(
        model_name="google/gemma-7b",
        task=task,
        sft=SFTConfig(
            batch_size=128,
            epochs=1,
            learning_rate=2e-5,
            warmup_ratio=0.1,
        ),
        rm=RMConfig(
            batch_size=128,
            epochs=1,
            learning_rate=1e-6,
            warmup_ratio=0.1,
        ),
        ppo=PPOConfig(
            batch_size=256,
            policy_learning_rate=1e-6,
            critic_learning_rate=1e-6,
            epochs=1,
            ppo_epochs=1,
            rollout=1,
            clip_ratio=0.2,
            gae_lambda=0.95,
            gae_gamma=1.0,
            kl_coefficient=0.01,  # Reduced from 0.05 for stability (paper §B.2)
            max_prompt_length=512,
            max_response_length=512,
            warmup_steps=200,
            temperature=0.8,
            top_p=1.0,
            top_k=50,
        ),
    )
    if task == "webgpt":
        cfg.sft.epochs = 5
        cfg.sft.batch_size = 32
        cfg.rm.epochs = 32
        cfg.rm.batch_size = 32
        cfg.rm.learning_rate = 1e-6
        cfg.ppo.epochs = 4
        cfg.ppo.kl_coefficient = 0.1
    if task == "hh-rlhf":
        cfg.rm.batch_size = 64
        cfg.ppo.kl_coefficient = 0.05
    if task == "tldr":
        cfg.rm.batch_size = 128
    return cfg


def get_gemma_27b_config(task: str = "tldr") -> ExperimentConfig:
    """Configuration for Gemma-2-27B model."""
    cfg = ExperimentConfig(
        model_name="google/gemma-2-27b",
        task=task,
        sft=SFTConfig(
            batch_size=128,
            epochs=3,
            learning_rate=5e-6,
            warmup_ratio=0.1,
        ),
        rm=RMConfig(
            batch_size=128,
            epochs=1,
            learning_rate=8e-6,
            warmup_ratio=0.1,
        ),
        ppo=PPOConfig(
            batch_size=256,
            policy_learning_rate=7e-7,
            critic_learning_rate=1e-6,
            epochs=1,
            ppo_epochs=1,
            rollout=1,
            clip_ratio=0.2,
            gae_lambda=0.95,
            gae_gamma=1.0,
            kl_coefficient=0.1,
            max_prompt_length=512,
            max_response_length=512,
            warmup_steps=0,
            temperature=0.8,
            top_p=1.0,
            top_k=50,
        ),
    )
    return cfg


def get_codegemma_2b_config() -> ExperimentConfig:
    """Configuration for CodeGemma-1.1-2B model."""
    return ExperimentConfig(
        model_name="google/codegemma-1.1-2b",
        task="apps",
        sft=SFTConfig(
            batch_size=16,
            epochs=1,
            learning_rate=5e-6,
            warmup_ratio=0.0,
        ),
        rm=RMConfig(),
        ppo=PPOConfig(
            batch_size=16,
            policy_learning_rate=5e-7,
            critic_learning_rate=5e-5,
            epochs=1,
            ppo_epochs=1,
            rollout=1,
            clip_ratio=0.2,
            gae_lambda=0.95,
            gae_gamma=1.0,
            kl_coefficient=0.05,
            max_prompt_length=600,
            max_response_length=512,
            warmup_steps=20,
            temperature=1.0,
            top_p=1.0,
            top_k=5,
        ),
    )


def get_codegemma_7b_config() -> ExperimentConfig:
    """Configuration for CodeGemma-1.1-7B-it model."""
    return ExperimentConfig(
        model_name="google/codegemma-1.1-7b-it",
        task="apps",
        sft=SFTConfig(
            batch_size=32,
            epochs=1,
            learning_rate=2e-6,
            warmup_ratio=0.0,
        ),
        rm=RMConfig(),
        ppo=PPOConfig(
            batch_size=16,
            policy_learning_rate=5e-7,
            critic_learning_rate=5e-5,
            epochs=1,
            ppo_epochs=1,
            rollout=1,
            clip_ratio=0.2,
            gae_lambda=0.95,
            gae_gamma=1.0,
            kl_coefficient=0.05,
            max_prompt_length=600,
            max_response_length=512,
            warmup_steps=20,
            temperature=1.0,
            top_p=1.0,
            top_k=5,
        ),
    )


CONFIG_MAP = {
    "gemma-2b": get_gemma_2b_config,
    "gemma-7b": get_gemma_7b_config,
    "gemma-27b": get_gemma_27b_config,
    "codegemma-2b": get_codegemma_2b_config,
    "codegemma-7b": get_codegemma_7b_config,
}
