"""
config.py

Central configuration module for the Universal Neural Operators reproduction pipeline.
It reads a YAML configuration file, applies sensible defaults, validates parameters,
and exposes settings as a flat set of attributes.
"""

import yaml
from typing import Dict, Any, Optional, List
import copy
import os

# ----------------------------------------------------------------------
# Default values that mirror config.yaml and the reproduction plan.
# Any key absent in the user’s YAML will be filled from here.
# ----------------------------------------------------------------------
_DEFAULT_CONFIG = {
    "experiment_name": "default_experiment",
    "pretrain": True,                   # Flag used to distinguish pre‑training phase
    "adapter_mode": "multiphysics",     # currently unused, placeholder for future extensions
    "model": {
        "architecture": "fno",
        "fno": {
            "modes1": 12,
            "modes2": 12,
            "width": 32,
            "n_layers": 4,
            "activation": "gelu"
        },
        "mamba": {
            "d_state": 16,
            "d_conv": 4,
            "expand": 2
        },
        "perceiver": {
            "n_latent": 32,
            "n_heads": 4,
            "n_self_attn": 2,
            "fno_for_kv": False
        },
        "swin": {
            "img_size": 64,
            "window_size": 8,
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24]
        },
        "coda": {
            "modes1": 12,
            "modes2": 12,
            "width": 32
        }
    },
    "training": {
        "pretrain": {
            "epochs": 500,
            "batch_size": 16,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "grad_clip": 1.0
        },
        "finetune": {
            "epochs": 200,
            "batch_size": 16,
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-4,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "grad_clip": 1.0
        },
        "scratch": {
            "epochs": 500,
            "batch_size": 16,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "grad_clip": 1.0
        }
    },
    "data": {
        "n_train_samples": 1000,
        "n_val_samples": 125,
        "n_test_samples": 125,
        "grid_size_1d": 1024,
        "grid_size_2d": 64,
        "pdebench_pretrain_tasks": ["advection", "burgers"],
        "pdebench_finetune_task": "reaction_diffusion",
        "burgers": {
            "nu_pretrain": [0.001, 0.01, 0.02, 0.05],
            "nu_finetune": 0.1
        },
        "grayscott": {
            "F_pretrain": [0.02, 0.04, 0.06],
            "k_pretrain": [0.05, 0.06, 0.07],
            "F_finetune": 0.08,
            "k_finetune": 0.09
        },
        "navierstokes": {
            "Re_pretrain": [100, 200, 500],
            "Re_finetune": 1000
        }
    },
    "eval": {
        "metrics": ["mse", "nmae"],
        "epsilon_nmae": 1.0e-9
    },
    "logging": {
        "log_dir": "./logs",
        "checkpoint_dir": "./checkpoints",
        "tensorboard": True,
        "print_freq": 10
    }
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merges override into base. base is modified in place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class Config:
    """
    Represents the complete configuration loaded from a YAML file, augmented with
    defaults and validated. Public attributes correspond to the class diagram:
        model_params : dict
        training_params : dict
        data_params : dict
        experiment_name : str
        pretrain : bool
        adapter_mode : str
    """

    def __init__(self, raw_data: Dict[str, Any]):
        """
        Do not call directly; use from_yaml() to create a configured instance.
        """
        self.experiment_name = raw_data.get("experiment_name", "default_experiment")
        self.pretrain = raw_data.get("pretrain", True)
        self.adapter_mode = raw_data.get("adapter_mode", "multiphysics")

        # Store the hierarchical config dictionaries as public attributes.
        self.model_params = raw_data.get("model", {})
        self.training_params = raw_data.get("training", {})
        self.data_params = raw_data.get("data", {})
        self.eval_params = raw_data.get("eval", {})      # not in original diagram, but needed
        self.log_params = raw_data.get("logging", {})    # same

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from a YAML file, apply defaults, validate, and return
        a Config instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            user_config = yaml.safe_load(f) or {}

        # Deep merge user config into a copy of the default config
        merged = copy.deepcopy(_DEFAULT_CONFIG)
        _deep_merge(merged, user_config)

        # Perform validation
        cls._validate(merged)

        return cls(merged)

    # ------------------------------------------------------------------
    # Validation methods
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(raw: Dict[str, Any]) -> None:
        """Check correctness of the provided configuration and raise ValueError on errors."""
        # Model architecture check
        arch = raw.get("model", {}).get("architecture")
        allowed_archs = ["fno", "mamba_fno", "perceiver_fno", "swin_v2", "coda_no"]
        if arch not in allowed_archs:
            raise ValueError(
                f"Unsupported architecture '{arch}'. Must be one of {allowed_archs}."
            )

        # Validate training parameters (positive values)
        for phase in ["pretrain", "finetune", "scratch"]:
            train = raw.get("training", {}).get(phase, {})
            epochs = train.get("epochs")
            if not isinstance(epochs, int) or epochs <= 0:
                raise ValueError(f"training.{phase}.epochs must be a positive integer, got {epochs}")
            lr = train.get("learning_rate")
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise ValueError(f"training.{phase}.learning_rate must be positive, got {lr}")
            bs = train.get("batch_size")
            if not isinstance(bs, int) or bs <= 0:
                raise ValueError(f"training.{phase}.batch_size must be positive integer, got {bs}")
            wd = train.get("weight_decay")
            if not isinstance(wd, (int, float)) or wd < 0:
                raise ValueError(f"training.{phase}.weight_decay must be non‑negative, got {wd}")
            gc = train.get("grad_clip")
            if gc is not None and not isinstance(gc, (int, float)) or (isinstance(gc, (int, float)) and gc <= 0):
                raise ValueError(f"training.{phase}.grad_clip must be positive if set, got {gc}")

        # Data parameters
        data = raw.get("data", {})
        for key in ["n_train_samples", "n_val_samples", "n_test_samples"]:
            val = data.get(key)
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"data.{key} must be a positive integer, got {val}")
        g1d = data.get("grid_size_1d")
        if not isinstance(g1d, int) or g1d <= 0:
            raise ValueError(f"data.grid_size_1d must be positive integer, got {g1d}")
        g2d = data.get("grid_size_2d")
        if not isinstance(g2d, int) or g2d <= 0:
            raise ValueError(f"data.grid_size_2d must be positive integer, got {g2d}")

        # Swin‑v2 spatial consistency
        if arch == "swin_v2":
            img_size = raw.get("model", {}).get("swin", {}).get("img_size")
            if img_size != g2d:
                raise ValueError(
                    f"Model architecture 'swin_v2' expects img_size={img_size} "
                    f"but data.grid_size_2d={g2d}. They must match."
                )

        # PDEBench task lists
        allowed_pdebench_tasks = {"advection", "burgers", "reaction_diffusion"}
        pretrain_tasks = set(data.get("pdebench_pretrain_tasks", []))
        if not pretrain_tasks.issubset(allowed_pdebench_tasks):
            unknown = pretrain_tasks - allowed_pdebench_tasks
            raise ValueError(
                f"Unknown PDEBench pretrain tasks: {unknown}. Allowed: {allowed_pdebench_tasks}"
            )
        finetune_task = data.get("pdebench_finetune_task")
        if finetune_task not in allowed_pdebench_tasks:
            raise ValueError(
                f"Unknown PDEBench finetune task '{finetune_task}'. Allowed: {allowed_pdebench_tasks}"
            )

        # Ensure parameter lists are not empty for our custom PDEs
        for pde_name in ["burgers", "grayscott", "navierstokes"]:
            pde_params = data.get(pde_name, {})
            if "nu_pretrain" in pde_params:
                if not isinstance(pde_params["nu_pretrain"], list) or len(pde_params["nu_pretrain"]) == 0:
                    raise ValueError(f"data.{pde_name}.nu_pretrain must be a non‑empty list")
            if "F_pretrain" in pde_params:
                if not isinstance(pde_params["F_pretrain"], list) or len(pde_params["F_pretrain"]) == 0:
                    raise ValueError(f"data.{pde_name}.F_pretrain must be a non‑empty list")
            if "k_pretrain" in pde_params:
                if not isinstance(pde_params["k_pretrain"], list) or len(pde_params["k_pretrain"]) == 0:
                    raise ValueError(f"data.{pde_name}.k_pretrain must be a non‑empty list")
            if "Re_pretrain" in pde_params:
                if not isinstance(pde_params["Re_pretrain"], list) or len(pde_params["Re_pretrain"]) == 0:
                    raise ValueError(f"data.{pde_name}.Re_pretrain must be a non‑empty list")
