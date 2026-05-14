from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ViT-B/16 architecture constants
VIT_B16_EMBED_DIM = 768
VIT_B16_NUM_HEADS = 12
VIT_B16_NUM_LAYERS = 12
VIT_B16_MLP_RATIO = 4
VIT_B16_PATCH_SIZE = 16
VIT_B16_TOTAL_PARAMS_M = 86.0  # ~86M parameters
VIT_B16_PEFT_CAP_FRACTION = 0.015  # 1.5% cap on PEFT parameters


# VTAB-1K dataset groups and task names
VTAB_NATURAL = [
    "caltech101",
    "cifar100",
    "dtd",
    "flowers102",
    "pets",
    "svhn",
    "sun397",
]

VTAB_SPECIALIZED = [
    "camelyon",
    "eurosat",
    "resisc45",
    "retinopathy",
]

VTAB_STRUCTURED = [
    "clevr_count",
    "clevr_distance",
    "dmlab",
    "kitti",
    "dsprites_loc",
    "dsprites_ori",
    "smallnorb_azimuth",
    "smallnorb_elevation",
]

VTAB_ALL_TASKS = VTAB_NATURAL + VTAB_SPECIALIZED + VTAB_STRUCTURED

VTAB_NUM_CLASSES = {
    "caltech101": 102,
    "cifar100": 100,
    "dtd": 47,
    "flowers102": 102,
    "pets": 37,
    "svhn": 10,
    "sun397": 397,
    "camelyon": 2,
    "eurosat": 10,
    "resisc45": 45,
    "retinopathy": 5,
    "clevr_count": 8,
    "clevr_distance": 6,
    "dmlab": 6,
    "kitti": 4,
    "dsprites_loc": 16,
    "dsprites_ori": 16,
    "smallnorb_azimuth": 18,
    "smallnorb_elevation": 9,
}

# Many-shot datasets
MANYSHOT_DATASETS = ["cifar100", "resisc45", "clevr_distance"]

# Distribution shift datasets
IMAGENET_SHIFT_DATASETS = ["imagenet_v2", "imagenet_r", "imagenet_s", "imagenet_a"]


@dataclass
class VTABConfig:
    """Configuration for VTAB-1K low-shot experiments."""
    backbone: str = "vit_base_patch16_224"
    pretrained: str = "imagenet21k"  # ImageNet-21K pretrained
    image_size: int = 224
    num_train_samples: int = 1000
    train_val_split: float = 0.8  # 800 train / 200 val from 1000
    batch_size: int = 64
    num_epochs: int = 100
    # LR search grid
    lr_choices: List[float] = field(default_factory=lambda: [1e-3, 1e-2])
    wd_choices: List[float] = field(default_factory=lambda: [1e-4, 1e-3])
    # Drop path rate: 0 (off) or 0.1 (on)
    drop_path_rate_choices: List[float] = field(default_factory=lambda: [0.0, 0.1])
    drop_path_rate: float = 0.1
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    # No data augmentation for VTAB-1K (consistent with prior work)
    use_augmentation: bool = False
    # ImageNet normalization
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    # PEFT parameter cap: <= 1.5% of ViT-B/16 total params (~1.29M)
    peft_param_cap_m: float = 1.29


@dataclass
class ManyShotConfig:
    """Configuration for many-shot experiments."""
    backbone: str = "vit_base_patch16_224"
    pretrained: str = "imagenet21k"
    image_size: int = 224
    train_val_split: float = 0.9  # 90/10 split
    batch_size: int = 64
    num_epochs: int = 40
    lr_choices: List[float] = field(default_factory=lambda: [5e-4, 1e-3])
    wd_choices: List[float] = field(default_factory=lambda: [1e-4, 1e-3])
    drop_path_rate: float = 0.1
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    # Dataset-specific augmentation
    augmentation: Dict[str, List[str]] = field(default_factory=lambda: {
        "cifar100": ["horizontal_flip"],
        "resisc45": ["horizontal_flip", "vertical_flip"],
        "clevr_distance": [],
    })


@dataclass
class RobustnessConfig:
    """Configuration for distribution shift robustness experiments (CLIP)."""
    backbone: str = "ViT-B/16"  # CLIP ViT-B/16
    image_size: int = 224
    num_shots: int = 100  # 100-shot ImageNet
    batch_size: int = 64
    num_epochs: int = 100
    lr: float = 3e-5
    weight_decay: float = 5e-3
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    # Strong data augmentation following CLIP paper
    use_strong_augmentation: bool = True
    mean: List[float] = field(default_factory=lambda: [0.48145466, 0.4578275, 0.40821073])
    std: List[float] = field(default_factory=lambda: [0.26862954, 0.26130258, 0.27577711])
    # WiSE mixing coefficients
    wise_alphas: List[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ])


# PEFT method hyperparameter search grids (from Table 3)
PEFT_SEARCH_GRIDS: Dict[str, Dict[str, Any]] = {
    "vpt_shallow": {
        "num_prompts": [5, 10, 50, 100, 200],
    },
    "vpt_deep": {
        "num_prompts": [5, 10, 50, 100],
    },
    "bitfit": {},
    "difffit": {},
    "layernorm": {},
    "ssf": {},
    "pfeif_adapter": {
        "scale_factor": [0.01, 0.1, 1.0, 10.0],
        "bottleneck_dim": [4, 8, 16, 32],
    },
    "houl_adapter": {
        "scale_factor": [0.01, 0.1, 1.0, 10.0],
        "bottleneck_dim": [4, 8, 16, 32],
    },
    "adaptformer": {
        "scale_factor": [0.05, 0.1, 0.2],
        "bottleneck_dim": [4, 16, 32],
    },
    "repadapter": {
        "scale_factor": [0.1, 0.5, 1.0, 5.0, 10.0],
        "bottleneck_dim": [8, 16, 32],
    },
    "convpass": {
        "scale_factor": [0.01, 0.1, 1.0, 10.0, 100.0],
        "bottleneck_dim": [8, 16],
        "xavier_init": [True, False],
    },
    "lora": {
        "rank": [1, 8, 16, 32],
    },
    "fact_tt": {
        "scale_factor": [0.01, 0.1, 1.0, 10.0, 100.0],
        "rank": [8, 16, 32],
    },
    "fact_tk": {
        "rank": [16, 32, 64],
        "scale_factor": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
}


# Default best hyperparameters (representative values from paper results)
PEFT_DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "vpt_shallow": {
        "num_prompts": 50,
    },
    "vpt_deep": {
        "num_prompts": 10,
    },
    "bitfit": {},
    "difffit": {},
    "layernorm": {},
    "ssf": {},
    "pfeif_adapter": {
        "scale_factor": 0.1,
        "bottleneck_dim": 16,
    },
    "houl_adapter": {
        "scale_factor": 0.1,
        "bottleneck_dim": 16,
    },
    "adaptformer": {
        "scale_factor": 0.1,
        "bottleneck_dim": 16,
    },
    "repadapter": {
        "scale_factor": 1.0,
        "bottleneck_dim": 16,
        "num_groups": 2,
    },
    "convpass": {
        "scale_factor": 1.0,
        "bottleneck_dim": 8,
        "kernel_size": 3,
        "xavier_init": True,
    },
    "lora": {
        "rank": 16,
    },
    "fact_tt": {
        "scale_factor": 1.0,
        "rank": 16,
    },
    "fact_tk": {
        "rank": 32,
        "scale_factor": 1.0,
    },
    "linear": {},
    "full": {},
}

# Trainable parameter counts (millions) from Table 3
PEFT_PARAM_COUNTS_M: Dict[str, float] = {
    "linear": 0.0,
    "full": 86.0,
    "vpt_shallow": 0.07,
    "vpt_deep": 0.43,
    "bitfit": 0.10,
    "difffit": 0.14,
    "layernorm": 0.04,
    "ssf": 0.21,
    "pfeif_adapter": 0.67,
    "houl_adapter": 0.77,
    "adaptformer": 0.46,
    "repadapter": 0.53,
    "convpass": 0.49,
    "lora": 0.55,
    "fact_tt": 0.13,
    "fact_tk": 0.23,
}

ALL_PEFT_METHODS = list(PEFT_DEFAULT_CONFIGS.keys())
