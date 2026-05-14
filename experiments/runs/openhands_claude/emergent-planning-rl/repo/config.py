from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    grid_size: int = 8
    num_boxes: int = 4
    num_targets: int = 4
    obs_channels: int = 7
    num_actions: int = 5
    min_episode_steps: int = 115
    max_episode_steps: int = 120
    reward_step: float = -0.01
    reward_box_on_target: float = 1.0
    reward_box_off_target: float = -1.0
    reward_solved: float = 10.0


@dataclass
class DRCConfig:
    num_layers: int = 3
    num_ticks: int = 3
    hidden_channels: int = 32
    kernel_size: int = 3
    padding: int = 1
    encoder_channels: int = 32
    encoder_kernel_size: int = 3
    encoder_padding: int = 1


@dataclass
class TrainConfig:
    total_transitions: int = 250_000_000
    batch_size: int = 16
    unroll_length: int = 20
    learning_rate_start: float = 4e-4
    learning_rate_end: float = 0.0
    discount: float = 0.97
    vtrace_lambda: float = 0.97
    entropy_coef: float = 1e-2
    logit_l2_penalty: float = 1e-3
    head_l2_reg: float = 1e-5
    num_actors: int = 32
    checkpoint_interval: int = 1_000_000
    log_interval: int = 10_000
    seed: int = 42


@dataclass
class ProbeConfig:
    probe_sizes: list = field(default_factory=lambda: [1, 3])
    num_classes: int = 5
    num_seeds: int = 5
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    train_episodes: int = 3000
    test_episodes: int = 1000
    checkpoint_train_episodes: int = 1000
    checkpoint_test_episodes: int = 500


@dataclass
class InterventionConfig:
    num_levels: int = 200
    base_levels: int = 25
    num_seeds: int = 5
    alpha: float = 1.0
    max_directional_squares: int = 3


@dataclass
class DataConfig:
    boxoban_path: str = "data/boxoban-levels"
    train_split: str = "unfiltered/train"
    valid_split: str = "unfiltered/valid"
    test_split: str = "unfiltered/test"
    medium_split: str = "medium/train"
    hard_split: str = "hard/train"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    drc: DRCConfig = field(default_factory=DRCConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "cuda"
    num_thinking_steps: int = 5


CONCEPT_CLASSES = {
    "NEVER": 0,
    "UP": 1,
    "DOWN": 2,
    "LEFT": 3,
    "RIGHT": 4,
}

CONCEPT_CLASSES_INV = {v: k for k, v in CONCEPT_CLASSES.items()}

ACTION_TO_DIRECTION = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "NOOP",
}

DIRECTION_TO_DELTA = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

DELTA_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_DELTA.items()}

CELL_TYPES = {
    "WALL": 0,
    "EMPTY": 1,
    "BOX": 2,
    "AGENT": 3,
    "BOX_ON_TARGET": 4,
    "AGENT_ON_TARGET": 5,
    "TARGET": 6,
}
