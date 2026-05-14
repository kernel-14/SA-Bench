from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Model size presets (non-embedding parameter counts are approximate)
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    # 6M GPT-2 style — used for Sudoku MDM
    "6M": dict(n_layers=6, d_model=384, n_heads=6, d_ff=1536, dropout=0.1),
    # 19M — used for Zebra MDM and L&O-NAE-SAT MDM
    "19M": dict(n_layers=12, d_model=512, n_heads=8, d_ff=2048, dropout=0.1),
    # 42M — used for Sudoku/Zebra ARM baselines
    "42M": dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072, dropout=0.1),
    # 170M — used for text generative perplexity experiment
    "170M": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096, dropout=0.1),
}


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

@dataclass
class NoiseScheduleConfig:
    schedule_type: str = "linear"   # "linear" | "cosine"
    alpha_min: float = 0.0          # alpha_1 ≈ 0 (fully masked)
    alpha_max: float = 1.0          # alpha_0 ≈ 1 (fully unmasked)


# ---------------------------------------------------------------------------
# Training configurations
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # Optimizer (AdamW — Loshchilov & Hutter, 2017)
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Learning rate schedule (cosine)
    lr_max: float = 4e-4
    lr_min: float = 4e-5
    warmup_steps: int = 2000

    # Batch / sequence
    batch_size: int = 128
    seq_len: int = 2048

    # Checkpointing
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 100

    # Device
    device: str = "cuda"
    seed: int = 42


@dataclass
class SudokuTrainingConfig(TrainingConfig):
    """Hyperparameters for Sudoku experiments (Section D.2)."""
    lr_max: float = 1e-3
    lr_min: float = 1e-5
    batch_size: int = 128
    epochs: int = 300
    seq_len: int = 81          # 9x9 Sudoku grid
    warmup_steps: int = 100


@dataclass
class ZebraTrainingConfig(TrainingConfig):
    """Hyperparameters for Zebra puzzle experiments (Section D.2)."""
    lr_max: float = 1e-3
    lr_min: float = 1e-5
    batch_size: int = 128
    epochs: int = 300
    warmup_steps: int = 100


@dataclass
class NAESATTrainingConfig(TrainingConfig):
    """Hyperparameters for L&O-NAE-SAT experiments (Section C.2.1)."""
    lr_max: float = 4e-4
    lr_min: float = 4e-5
    batch_size: int = 128
    max_iters: int = 50_000     # proxy MDM trained for 5e4 iters
    warmup_steps: int = 500
    seq_len: int = 512          # N + P + 212 padding tokens


@dataclass
class TextTrainingConfig(TrainingConfig):
    """Hyperparameters for text / scaling-law experiments (Section C.1)."""
    lr_max: float = 4e-4
    lr_min: float = 4e-5
    batch_size: int = 512
    seq_len: int = 2048
    warmup_steps: int = 2000


# ---------------------------------------------------------------------------
# Inference configurations
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    num_steps: int = 50                  # reverse diffusion steps
    strategy: str = "top_prob_margin"    # "vanilla" | "top_prob" | "top_prob_margin"
    gumbel_noise_coeff: float = 0.5      # Gumbel noise added to oracle scores (Section D.2)
    temperature: float = 1.0             # sampling temperature
    # For text generation: add Gaussian noise to oracle (Section D.1.2)
    oracle_noise_std: float = 0.0


# ---------------------------------------------------------------------------
# Dataset / task configurations
# ---------------------------------------------------------------------------

@dataclass
class SudokuConfig:
    vocab_size: int = 10        # digits 0-9 (0 = mask)
    seq_len: int = 81           # 9x9 grid
    num_given: int = 30         # average number of given cells (varies per puzzle)
    data_path: str = "data/sudoku"
    train_file: str = "sudoku_train.csv"
    test_file: str = "sudoku_test.csv"
    hard_test_file: str = "sudoku_hard_test.csv"


@dataclass
class ZebraConfig:
    # Zebra / Einstein puzzle: 5 houses × 5 attributes = 25 tokens
    # Each attribute has 5 possible values → vocab_size = 6 (0=mask, 1-5=values)
    num_houses: int = 5
    num_attributes: int = 5
    num_values: int = 5
    vocab_size: int = 6         # 0=mask, 1-5=values
    seq_len: int = 25           # 5 × 5 grid
    data_path: str = "data/zebra"
    train_file: str = "zebra_train.json"
    test_file: str = "zebra_test.json"


@dataclass
class NAESATConfig:
    N: int = 25                 # number of latent tokens
    P: int = 275                # number of observation tokens (N+P=300)
    vocab_size: int = 4         # 0=mask, 1-3=values (m=3 for NAE-SAT)
    padding_value: int = 3      # extra padding token value (value=2 in paper, 0-indexed here)
    pad_len: int = 212          # padding to reach seq_len=512
    seq_len: int = 512
    num_triples_per_obs: int = 1  # each observation is one NAE triple
    data_path: str = "data/nae_sat"
    num_train: int = 100_000
    num_test: int = 10_000


@dataclass
class TextConfig:
    dataset_name: str = "cerebras/SlimPajama-627B"
    vocab_size: int = 32_000    # LLaMA tokenizer
    seq_len: int = 2048
    data_path: str = "data/slimpajama"


# ---------------------------------------------------------------------------
# π-learner permutation configurations (Section 3.2 / C.1)
# ---------------------------------------------------------------------------

@dataclass
class PiLearnerConfig:
    permutation_type: str = "random"   # "identity" | "random" | "closer" | "much_closer"
    # "closer"      → L/10 random swaps from identity
    # "much_closer" → sqrt(L) random swaps from identity
    # "random"      → Unif(S_L)
    seq_len: int = 2048
    num_permutation_samples: int = 3   # repeat each type 3 times (paper uses 3)
    use_learnable_pos_emb: bool = True  # replace RoPE with learnable embeddings


# ---------------------------------------------------------------------------
# LLaDA 8B evaluation configuration (Section 4.4 / D.3)
# ---------------------------------------------------------------------------

@dataclass
class LLaDAConfig:
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct"
    inference_strategy: str = "top_prob_margin"
    num_steps: int = 256
    # Task-specific settings
    humaneval_infill_categories: list = field(
        default_factory=lambda: ["single_line", "multi_line", "split_line"]
    )
    max_new_tokens: int = 256
    temperature: float = 0.0


# ---------------------------------------------------------------------------
# Master experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    task: str = "sudoku"           # "sudoku" | "zebra" | "nae_sat" | "text" | "llada"
    model_type: str = "mdm"        # "mdm" | "arm"
    model_size: str = "6M"
    use_ordering: bool = False      # ARM: whether to use ground-truth token order
    run_name: str = "experiment"
    output_dir: str = "outputs"
    wandb_project: str = "masked-diffusion-token-ordering"
    use_wandb: bool = False

    noise_schedule: NoiseScheduleConfig = field(default_factory=NoiseScheduleConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Task-specific configs (populated based on `task`)
    sudoku: SudokuConfig = field(default_factory=SudokuConfig)
    zebra: ZebraConfig = field(default_factory=ZebraConfig)
    nae_sat: NAESATConfig = field(default_factory=NAESATConfig)
    text: TextConfig = field(default_factory=TextConfig)
    pi_learner: PiLearnerConfig = field(default_factory=PiLearnerConfig)
    llada: LLaDAConfig = field(default_factory=LLaDAConfig)

    def get_training_config(self) -> TrainingConfig:
        if self.task == "sudoku":
            return SudokuTrainingConfig()
        elif self.task == "zebra":
            return ZebraTrainingConfig()
        elif self.task == "nae_sat":
            return NAESATTrainingConfig()
        else:
            return TextTrainingConfig()

    def get_model_config(self) -> dict:
        return MODEL_CONFIGS[self.model_size]

    def get_task_vocab_size(self) -> int:
        mapping = {
            "sudoku": self.sudoku.vocab_size,
            "zebra": self.zebra.vocab_size,
            "nae_sat": self.nae_sat.vocab_size,
            "text": self.text.vocab_size,
        }
        return mapping.get(self.task, 32_000)

    def get_task_seq_len(self) -> int:
        mapping = {
            "sudoku": self.sudoku.seq_len,
            "zebra": self.zebra.seq_len,
            "nae_sat": self.nae_sat.seq_len,
            "text": self.text.seq_len,
        }
        return mapping.get(self.task, 2048)


# ---------------------------------------------------------------------------
# Default configs for each experiment in the paper
# ---------------------------------------------------------------------------

def get_sudoku_mdm_config() -> ExperimentConfig:
    cfg = ExperimentConfig(task="sudoku", model_type="mdm", model_size="6M")
    cfg.inference.num_steps = 50
    cfg.inference.gumbel_noise_coeff = 0.5
    return cfg


def get_sudoku_arm_config(use_ordering: bool = False) -> ExperimentConfig:
    cfg = ExperimentConfig(
        task="sudoku", model_type="arm", model_size="42M",
        use_ordering=use_ordering
    )
    return cfg


def get_zebra_mdm_config() -> ExperimentConfig:
    cfg = ExperimentConfig(task="zebra", model_type="mdm", model_size="19M")
    cfg.inference.num_steps = 50
    cfg.inference.gumbel_noise_coeff = 0.5
    return cfg


def get_zebra_arm_config(use_ordering: bool = False) -> ExperimentConfig:
    cfg = ExperimentConfig(
        task="zebra", model_type="arm", model_size="42M",
        use_ordering=use_ordering
    )
    return cfg


def get_nae_sat_config(N: int = 25, P: int = 275) -> ExperimentConfig:
    cfg = ExperimentConfig(task="nae_sat", model_type="mdm", model_size="19M")
    cfg.nae_sat.N = N
    cfg.nae_sat.P = P
    return cfg


def get_text_scaling_config(permutation_type: str = "random") -> ExperimentConfig:
    cfg = ExperimentConfig(task="text", model_type="arm", model_size="170M")
    cfg.pi_learner.permutation_type = permutation_type
    return cfg
