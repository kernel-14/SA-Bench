from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ModelConfig:
    vocab_size: int = 50257  # natural text
    max_seq_len: int = 2048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # for smaller experiments (6M, 19M, 42M, 170M param counts)
    # n_layer=4, n_embd=256, n_head=4  -> ~6M params
    # n_layer=6, n_embd=384, n_head=6  -> ~19M params
    # n_layer=8, n_embd=512, n_head=8  -> ~42M params
    # n_layer=12, n_embd=768, n_head=12 -> ~170M params

    use_rope: bool = False
    use_learned_pos_emb: bool = True
    mask_token_id: int = 0
    pad_token_id: Optional[int] = None
    num_mask_tokens: int = 1


@dataclass
class DiffusionConfig:
    noise_schedule: Literal["linear", "cosine", "loglinear"] = "loglinear"
    alpha_0: float = 0.9999
    alpha_1: float = 1e-8
    time_steps: int = 1024
    # forward/reverse discretization uses continuous-time formulation integrated over [0, 1]
    loss_type: Literal["score_entropy", "cross_entropy"] = "score_entropy"
    # integration weight alpha'_t / (1 - alpha_t)


@dataclass
class TrainingConfig:
    batch_size: int = 128
    grad_acc_steps: int = 1
    learning_rate: float = 4e-4
    min_lr: float = 4e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_steps: int = 100_000
    warmup_steps: int = 2000
    lr_schedule: Literal["cosine", "linear"] = "cosine"
    dtype: str = "bfloat16"
    compile: bool = False
    gradient_clip: float = 1.0


@dataclass
class InferenceConfig:
    num_steps: int = 50
    strategy: Literal["vanilla", "top_probability", "top_probability_margin"] = "vanilla"
    temperature: float = 0.0
    gumbel_noise: float = 0.5
    top_k: Optional[int] = None  # deterministic number of tokens to unmask per step


@dataclass
class DataConfig:
    dataset: Literal["text", "lonaesat", "sudoku", "zebra", "humaneval_infill"] = "text"
    text_dataset: str = "slimpajama"
    # L&O-NAE-SAT
    N_latent: int = 25
    P_obs: int = 275
    vocab_size_lonaesat: int = 2
    k_arity: int = 3
    # Sudoku / Zebra
    puzzle_size: int = 9

    # π-learner interpolation
    pi_distribution: Literal["uniform", "identity", "closer", "much_closer"] = "uniform"
    n_swaps_closer: Optional[int] = None  # L // 10
    n_swaps_much_closer: Optional[int] = None  # sqrt(L)


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "./outputs"
    wandb_project: Optional[str] = None


def get_default_config(preset: str = "base") -> ExperimentConfig:
    cfg = ExperimentConfig()
    if preset == "sudoku_6m":
        cfg.model.n_layer = 4
        cfg.model.n_embd = 256
        cfg.model.n_head = 4
        cfg.model.max_seq_len = 512
        cfg.model.vocab_size = 11
        cfg.data.dataset = "sudoku"
        cfg.training.max_steps = 50000
        cfg.training.learning_rate = 0.001
        cfg.training.batch_size = 128
        cfg.inference.num_steps = 50
        cfg.inference.gumbel_noise = 0.5
    elif preset == "zebra_19m":
        cfg.model.n_layer = 6
        cfg.model.n_embd = 384
        cfg.model.n_head = 6
        cfg.model.max_seq_len = 1024
        cfg.model.vocab_size = 50
        cfg.data.dataset = "zebra"
        cfg.training.max_steps = 50000
        cfg.training.learning_rate = 0.001
        cfg.training.batch_size = 128
        cfg.inference.num_steps = 50
        cfg.inference.gumbel_noise = 0.5
    elif preset == "lonaesat_19m":
        cfg.model.n_layer = 6
        cfg.model.n_embd = 384
        cfg.model.n_head = 6
        cfg.model.max_seq_len = 512
        cfg.model.vocab_size = 4
        cfg.data.dataset = "lonaesat"
        cfg.training.max_steps = 50000
        cfg.training.learning_rate = 0.001
    elif preset == "text_170m":
        cfg.model.n_layer = 12
        cfg.model.n_embd = 768
        cfg.model.n_head = 12
        cfg.model.max_seq_len = 2048
        cfg.model.vocab_size = 50257
        cfg.data.dataset = "text"
        cfg.training.max_steps = 100000
        cfg.training.learning_rate = 4e-4
        cfg.training.batch_size = 128
    elif preset == "arm_42m_sudoku":
        cfg.model.n_layer = 8
        cfg.model.n_embd = 512
        cfg.model.n_head = 8
        cfg.model.max_seq_len = 512
        cfg.model.vocab_size = 11
        cfg.data.dataset = "sudoku"
        cfg.training.max_steps = 50000
        cfg.training.learning_rate = 0.001
    elif preset == "arm_42m_zebra":
        cfg.model.n_layer = 8
        cfg.model.n_embd = 512
        cfg.model.n_head = 8
        cfg.model.max_seq_len = 1024
        cfg.model.vocab_size = 50
        cfg.data.dataset = "zebra"
        cfg.training.max_steps = 50000
        cfg.training.learning_rate = 0.001
    elif preset == "text_pi_learner":
        cfg.model.use_rope = False
        cfg.model.use_learned_pos_emb = True
        cfg.model.n_layer = 6
        cfg.model.n_embd = 384
        cfg.model.n_head = 6
        cfg.model.max_seq_len = 2048
        cfg.model.vocab_size = 50257
        cfg.data.dataset = "text"
        cfg.training.max_steps = 100000
        cfg.training.learning_rate = 4e-4
        cfg.training.batch_size = 64
    return cfg
