## config.py
"""Central configuration dataclass for MoE‑POT experiments.

All hyperparameters are stored in a single dataclass instance. The class
automatically sets model‑size‑dependent parameters in `__post_init__` and
can be instantiated from a YAML file or command‑line arguments via
`main.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import yaml


@dataclass
class Config:
    """Flat dataclass holding all configuration values.

    Default values correspond to the settings in `config.yaml`. When
    loaded from YAML, nested sections are flattened (e.g.
    ``data.spatial_resolution`` becomes ``spatial_resolution``).

    Model‑size‑dependent attributes (``attention_dim``, ``mlp_dim``,
    ``n_blocks``, ``n_heads``) are **overwritten** in `__post_init__`
    based on ``model_size``, regardless of the values passed in.
    """

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    model_size: str = "small"  # Choose from {"tiny", "small", "medium"}

    # ------------------------------------------------------------------
    # Data settings (correspond to data.* in YAML)
    # ------------------------------------------------------------------
    spatial_resolution: Tuple[int, int] = (128, 128)
    input_frames: int = 10                      # T_in
    max_channels: int = 5                       # padding constant (CNS:4 + mask)
    mask_channel_index: int = 4                 # 0‑based index of the mask channel
    noise_std: float = 0.01                     # ε for pre‑training noise injection
    balanced_sampling_weight: float = 1.0       # w_k for all datasets

    # ------------------------------------------------------------------
    # Pre‑training hyperparameters (pretrain.*)
    # ------------------------------------------------------------------
    pretrain_epochs: int = 1000
    pretrain_warmup_epochs: int = 200
    pretrain_learning_rate: float = 0.001
    pretrain_weight_decay: float = 1.0e-6
    pretrain_beta1: float = 0.9
    pretrain_beta2: float = 0.9
    pretrain_batch_size: int = 20
    pretrain_load_balance_weight: float = 0.1
    pretrain_noise_enabled: bool = True

    # ------------------------------------------------------------------
    # Fine‑tuning hyperparameters (finetune.*)
    # ------------------------------------------------------------------
    finetune_epochs: int = 200
    finetune_warmup_epochs: int = 40
    finetune_learning_rate: float = 0.001
    finetune_freeze_router: bool = True
    finetune_noise_enabled: bool = False

    # ------------------------------------------------------------------
    # Downstream task hyperparameters (downstream.*)
    # ------------------------------------------------------------------
    downstream_epochs: int = 500
    downstream_warmup_epochs: int = 100
    downstream_learning_rate: float = 0.001
    downstream_freeze_router: bool = True
    downstream_noise_enabled: bool = False

    # ------------------------------------------------------------------
    # Model component details (overwritten in __post_init__)
    # ------------------------------------------------------------------
    attention_dim: int = 1024
    mlp_dim: int = 1024
    n_blocks: int = 6
    n_heads: int = 8
    routed_experts: int = 16
    shared_experts: int = 2
    top_k: int = 4
    patch_size: int = 8

    # ------------------------------------------------------------------
    # Evaluation settings (evaluation.*)
    # ------------------------------------------------------------------
    rollout_steps: int = 10                    # future frames to predict for L2RE
    eval_warmup_inference: int = 10            # warmup iterations for timing
    eval_repeat_inference: int = 100           # measurement repeats for timing

    # ------------------------------------------------------------------
    # Paths & runtime (not in YAML, may be overridden via argparse)
    # ------------------------------------------------------------------
    data_root: str = "./data"
    output_dir: str = "./outputs"
    seed: int = 42

    # ------------------------------------------------------------------
    # Derived or special fields (non‑init, computed)
    # ------------------------------------------------------------------
    # We store the effective spatial feature map size (after patchification)
    # as a property for convenience; this is also computed in __post_init__.
    feat_h: int = field(init=False, repr=False)
    feat_w: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate model_size and auto‑set size‑dependent parameters."""
        valid_sizes = {"tiny", "small", "medium"}
        if self.model_size not in valid_sizes:
            raise ValueError(
                f"Invalid model_size '{self.model_size}'. "
                f"Choose one of {valid_sizes}."
            )

        # Override size‑dependent attributes based on chosen model_size
        size_config: Dict[str, Dict[str, int]] = {
            "tiny": {
                "attention_dim": 512,
                "mlp_dim": 512,
                "n_blocks": 4,
                "n_heads": 4,
            },
            "small": {
                "attention_dim": 1024,
                "mlp_dim": 1024,
                "n_blocks": 6,
                "n_heads": 8,
            },
            "medium": {
                "attention_dim": 1024,
                "mlp_dim": 2048,
                "n_blocks": 8,
                "n_heads": 8,
            },
        }
        params = size_config[self.model_size]
        self.attention_dim = params["attention_dim"]
        self.mlp_dim = params["mlp_dim"]
        self.n_blocks = params["n_blocks"]
        self.n_heads = params["n_heads"]

        # Ensure patch_size divides spatial resolution cleanly
        H, W = self.spatial_resolution
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Spatial resolution {self.spatial_resolution} is not "
                f"divisible by patch size {self.patch_size}."
            )
        # Pre‑compute feature map size after patchification
        self.feat_h = H // self.patch_size
        self.feat_w = W // self.patch_size

    # ------------------------------------------------------------------
    # Factory method from YAML
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Create a Config instance from a YAML file.

        The file is expected to have the same nested structure as
        ``config.yaml``. Top‑level keys (``data``, ``pretrain``, …) are
        flattened into the dataclass fields.
        """
        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if raw is None:
            raise ValueError(f"Empty YAML file: {yaml_path}")

        # Flatten nested sections
        flat: Dict[str, Any] = {}
        # Map YAML section names to the prefixes used in our field names
        section_prefix = {
            "data": "",
            "pretrain": "pretrain_",
            "finetune": "finetune_",
            "downstream": "downstream_",
            "evaluation": "eval_",
            # model_size is at top level, already handled
        }

        # First pass: handle top‑level keys like model_size
        for key, val in raw.items():
            if key in section_prefix:
                prefix = section_prefix[key]
                if isinstance(val, dict):
                    for subk, subv in val.items():
                        flat[prefix + subk] = subv
                elif key == "model_size":
                    flat["model_size"] = val
                else:
                    # Not a dict, store directly (e.g., model_size)
                    flat[key] = val
            else:
                # Keep non‑section keys as is, but we do not expect many
                flat[key] = val

        # Renaming fixes for YAML fields that differ from flat names
        # (mainly nested paths that we already prefixed, but we need to
        #  map 'data.spatial_resolution' correctly – already handled)

        return cls(**flat)

    # ------------------------------------------------------------------
    # Quick helper for training-specific hyperparams
    # ------------------------------------------------------------------
    def get_training_params(self, mode: str) -> Dict[str, Any]:
        """Return a dict with hyperparameters for the given mode.

        Args:
            mode: one of ``"pretrain"``, ``"finetune"``, ``"downstream"``.

        Returns:
            Dictionary containing ``epochs``, ``warmup_epochs``,
            ``learning_rate``, ``freeze_router``, ``noise_enabled``, etc.
        """
        if mode == "pretrain":
            return {
                "epochs": self.pretrain_epochs,
                "warmup_epochs": self.pretrain_warmup_epochs,
                "learning_rate": self.pretrain_learning_rate,
                "weight_decay": self.pretrain_weight_decay,
                "beta1": self.pretrain_beta1,
                "beta2": self.pretrain_beta2,
                "batch_size": self.pretrain_batch_size,
                "load_balance_weight": self.pretrain_load_balance_weight,
                "noise_enabled": self.pretrain_noise_enabled,
            }
        elif mode == "finetune":
            return {
                "epochs": self.finetune_epochs,
                "warmup_epochs": self.finetune_warmup_epochs,
                "learning_rate": self.finetune_learning_rate,
                "freeze_router": self.finetune_freeze_router,
                "noise_enabled": self.finetune_noise_enabled,
            }
        elif mode == "downstream":
            return {
                "epochs": self.downstream_epochs,
                "warmup_epochs": self.downstream_warmup_epochs,
                "learning_rate": self.downstream_learning_rate,
                "freeze_router": self.downstream_freeze_router,
                "noise_enabled": self.downstream_noise_enabled,
            }
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    def __repr__(self) -> str:
        """Compact representation, hiding unchanged defaults."""
        return (
            f"Config(model_size={self.model_size}, "
            f"attn_dim={self.attention_dim}, mlp_dim={self.mlp_dim}, "
            f"blocks={self.n_blocks}, heads={self.n_heads}, "
            f"feat_map=({self.feat_h},{self.feat_w}), "
            f"seed={self.seed})"
        )
