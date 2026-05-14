import copy
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class ModelConfig:
    """Configuration for GPT and nGPT models."""

    model_type: str = "ngpt"          # "gpt" or "ngpt"
    vocab_size: int = 32000           # LLaMA-2 tokenizer vocabulary size
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    d_mlp: Optional[int] = None       # defaults to 4 * d_model
    max_seq_len: int = 4096
    rope_base: float = 10000.0

    # nGPT eigen learning rate parameters (Section 2.6, step 3)
    # alpha_init = 0.05 (~1/n_layers), alpha_scale = 1/sqrt(d_model)
    alpha_init: float = 0.05
    alpha_scale: Optional[float] = None   # set to 1/sqrt(d_model) in __post_init__

    # nGPT QK scaling parameters (Section 2.6, step 4)
    # sqk_init = 1, sqk_scale = 1/sqrt(d_model)
    sqk_init: float = 1.0
    sqk_scale: Optional[float] = None     # set to 1/sqrt(d_model) in __post_init__

    # nGPT MLP scaling parameters (Section 2.6, step 5)
    # su_init = sv_init = 1, su_scale = sv_scale = 1
    su_init: float = 1.0
    su_scale: float = 1.0
    sv_init: float = 1.0
    sv_scale: float = 1.0

    # nGPT logit scaling parameters (Section 2.6, step 6)
    # sz_init = 1, sz_scale = 1/sqrt(d_model)
    sz_init: float = 1.0
    sz_scale: Optional[float] = None      # set to 1/sqrt(d_model) in __post_init__

    def __post_init__(self):
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model
        if self.alpha_scale is None:
            self.alpha_scale = 1.0 / math.sqrt(self.d_model)
        if self.sqk_scale is None:
            self.sqk_scale = 1.0 / math.sqrt(self.d_model)
        if self.sz_scale is None:
            self.sz_scale = 1.0 / math.sqrt(self.d_model)

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


# Pre-defined model configurations matching Table 2
MODEL_CONFIGS = {
    "0.5B": ModelConfig(
        n_layers=24,
        d_model=1024,
        n_heads=16,
    ),
    "1B": ModelConfig(
        n_layers=36,
        d_model=1280,
        n_heads=20,
    ),
}


@dataclass
class TrainConfig:
    """Training configuration matching Appendix A.6."""

    # Model selection
    model_type: str = "ngpt"          # "gpt" or "ngpt"
    model_size: str = "0.5B"          # "0.5B" or "1B"

    # Data
    dataset_path: str = "data/openwebtext"
    seq_len: int = 4096               # context length (1k, 4k, or 8k in paper)
    global_batch_size: int = 512
    num_workers: int = 4

    # Optimizer — Adam for nGPT, AdamW for GPT (Table 3)
    lr: float = 2e-3                  # tuned per experiment (Appendix A.7)
    weight_decay: float = 0.0         # 0.0 for nGPT, 0.1 for GPT
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # LR schedule — cosine annealing for both (Table 3)
    warmup_steps: int = 0             # 0 for nGPT, 2000 for GPT
    max_steps: int = 100000
    min_lr_ratio: float = 0.0         # final LR = 0

    # Hardware
    dtype: str = "bfloat16"           # bfloat16 as in paper
    compile: bool = False

    # Logging / checkpointing
    log_interval: int = 100
    eval_interval: int = 1000
    eval_steps: int = 100
    save_interval: int = 5000
    out_dir: str = "checkpoints"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Distributed
    backend: str = "nccl"

    def get_model_config(self) -> ModelConfig:
        cfg = copy.deepcopy(MODEL_CONFIGS[self.model_size])
        cfg.model_type = self.model_type
        cfg.max_seq_len = self.seq_len
        # GPT uses weight decay; nGPT does not (Section 2.6, step 7)
        if self.model_type == "gpt":
            self.weight_decay = 0.1
            self.warmup_steps = 2000
        return cfg


# GPT-specific training config (baseline)
@dataclass
class GPTTrainConfig(TrainConfig):
    model_type: str = "gpt"
    weight_decay: float = 0.1
    warmup_steps: int = 2000
