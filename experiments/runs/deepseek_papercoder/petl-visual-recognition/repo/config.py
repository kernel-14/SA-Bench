## config.py

import yaml
import os
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


def expand_range(value: Any, num_points: int = 5, log_scale: bool = True) -> List[float]:
    """
    Expand a range specifier into a list of concrete values.
    
    If `value` is a list of two numbers, it generates `num_points` evenly
    spaced between them (in log-space if `log_scale` is True). Otherwise,
    if it's already a list, it returns it unchanged; if a single number,
    wraps it in a list.
    """
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            if log_scale:
                start, end = np.log10(value[0]), np.log10(value[1])
                return (10 ** np.linspace(start, end, num_points)).tolist()
            else:
                return np.linspace(value[0], value[1], num_points).tolist()
        # Already a fully specified list
        return value
    else:
        # Single value
        return [float(value)]


class Config:
    # Allowed PEFT methods as per paper (exact keys used in config.yaml)
    _ALLOWED_PEFT_METHODS = {
        "vpt_shallow", "vpt_deep", "bitfit", "difffit", "layernorm",
        "ssf", "pfeif_adapter", "houl_adapter", "adaptformer",
        "repadapter", "convpass", "lora", "fact_tt", "fact_tk"
    }
    _ALLOWED_BACKBONES = {"vit_base_patch16_224_in21k", "ViT-B/16"}

    def __init__(self, config_dict: dict, overrides: Optional[dict] = None):
        """
        Initialize a Config object from a dictionary (typically loaded from YAML).
        
        Args:
            config_dict: raw dictionary from YAML.
            overrides: optional dict of overrides to update config (using dot‑notation keys).
        """
        self.raw = config_dict
        if overrides:
            self._apply_overrides(overrides)
        self._parse()
        self._validate()

    @classmethod
    def from_file(cls, path: str, overrides: Optional[dict] = None) -> "Config":
        """Load configuration from a YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)
        return cls(raw, overrides)

    @staticmethod
    def parse_config(path: str) -> "Config":
        """Alias for from_file (matches design diagram)."""
        return Config.from_file(path)

    # ------------------------------------------------------------------ #
    #  Internal parsing and validation
    # ------------------------------------------------------------------ #
    def _parse(self):
        """Extract and pre-process configuration sections."""
        # Top-level sections
        self.backbone = self.raw.get("backbone", {})
        self.training = self.raw.get("training", {})
        self.peft_methods = self.raw.get("peft_methods", {})
        self.datasets = self.raw.get("datasets", {})
        self.misc = self.raw.get("misc", {})

        # Expand learning rate / weight decay ranges for VTAB and many-shot
        for mode in ["vtab", "many_shot"]:
            if mode in self.training:
                cfg = self.training[mode]
                if "learning_rate" in cfg:
                    cfg["learning_rate"] = expand_range(cfg["learning_rate"])
                if "weight_decay" in cfg:
                    cfg["weight_decay"] = expand_range(cfg["weight_decay"])

        # Ensure robustness values are not expanded (they are single floats)
        # No expansion needed.

    def _validate(self):
        """Validate required fields, types, and values."""
        # --- Backbone ---
        if not isinstance(self.backbone, dict):
            raise ValueError("'backbone' section must be a dict")
        bn = self.backbone.get("name", "")
        if bn not in self._ALLOWED_BACKBONES:
            raise ValueError(f"Backbone name '{bn}' not supported; choose from {self._ALLOWED_BACKBONES}")
        drop_rate = self.backbone.get("drop_path_rate", None)
        if drop_rate is None:
            raise ValueError("backbone.drop_path_rate is required")
        if drop_rate not in (0.0, 0.1):
            raise ValueError("backbone.drop_path_rate must be 0.0 or 0.1")
        param_total = self.backbone.get("param_total", 0)
        if not isinstance(param_total, (int, float)) or param_total <= 0:
            raise ValueError("backbone.param_total must be a positive number")

        # --- Training ---
        for mode in ["vtab", "many_shot", "robustness"]:
            if mode not in self.training:
                raise ValueError(f"Missing training section: '{mode}'")
            cfg = self.training[mode]
            if mode in ["vtab", "many_shot"]:
                # Check lists
                for key in ["learning_rate", "weight_decay"]:
                    vals = cfg.get(key, [])
                    if not isinstance(vals, list) or len(vals) == 0:
                        raise ValueError(f"training.{mode}.{key} must be a non-empty list")
                    if not all(isinstance(v, (int, float)) for v in vals):
                        raise ValueError(f"All elements of {key} must be numeric")
                # Additional VTAB checks
                if mode == "vtab":
                    if "val_split_ratio" not in cfg or not (0.0 < cfg["val_split_ratio"] < 1.0):
                        raise ValueError("training.vtab.val_split_ratio must be in (0,1)")
            else:  # robustness
                for key in ["learning_rate", "weight_decay"]:
                    v = cfg.get(key)
                    if not isinstance(v, (int, float)):
                        raise ValueError(f"training.robustness.{key} must be a single float")
                if "wise_alphas" not in cfg or not isinstance(cfg["wise_alphas"], list):
                    raise ValueError("training.robustness.wise_alphas must be a list of floats")

        # --- PEFT methods ---
        if not isinstance(self.peft_methods, dict):
            raise ValueError("'peft_methods' must be a dict")
        for meth, hyper in self.peft_methods.items():
            if meth not in self._ALLOWED_PEFT_METHODS:
                raise ValueError(f"Unknown PEFT method: {meth}")
            if not isinstance(hyper, dict):
                raise ValueError(f"Hyperparameters for {meth} must be a dict")
            # Method-specific validation could be added, but for now we trust the config.

        # --- Datasets ---
        if not isinstance(self.datasets, dict):
            raise ValueError("'datasets' section must be a dict")
        if "vtab_root" not in self.datasets or not isinstance(self.datasets["vtab_root"], str):
            raise ValueError("datasets.vtab_root must be a string path")
        if "many_shot" not in self.datasets or not isinstance(self.datasets["many_shot"], dict):
            raise ValueError("datasets.many_shot must be a dict")
        if "robustness" not in self.datasets or not isinstance(self.datasets["robustness"], dict):
            raise ValueError("datasets.robustness must be a dict")

        # --- Misc ---
        if not isinstance(self.misc, dict):
            raise ValueError("'misc' must be a dict")
        if "seed" not in self.misc or not isinstance(self.misc["seed"], int):
            raise ValueError("misc.seed must be an integer")
        if "num_workers" not in self.misc or not isinstance(self.misc["num_workers"], int):
            raise ValueError("misc.num_workers must be an integer")
        if "device" not in self.misc or not isinstance(self.misc["device"], str):
            raise ValueError("misc.device must be a string")

        logger.info("Configuration validated successfully.")

    def _apply_overrides(self, overrides: dict):
        """Apply override values using dot‑notation keys, e.g. 'backbone.drop_path_rate'."""
        for key, value in overrides.items():
            parts = key.split('.')
            d = self.raw
            for part in parts[:-1]:
                if part not in d:
                    # If the parent key doesn't exist, create a nested dict temporarily
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value

    # ------------------------------------------------------------------ #
    #  Convenience accessors
    # ------------------------------------------------------------------ #
    def get_training_params(self, experiment_type: str) -> dict:
        """
        Return a flat dictionary of training hyperparameters for the given experiment type.
        """
        if experiment_type not in ["vtab", "many_shot", "robustness"]:
            raise ValueError(f"Invalid experiment type: {experiment_type}")
        return self.training[experiment_type]

    def get_peft_hyperparams(self, method_name: str) -> dict:
        """
        Return the hyperparameter search grid for a specific PEFT method.
        """
        if method_name not in self._ALLOWED_PEFT_METHODS:
            raise ValueError(f"Unknown PEFT method: {method_name}")
        return self.peft_methods.get(method_name, {})

    def get_dataset_path(self, task: Optional[str] = None, mode: str = "vtab") -> str:
        """
        Return the absolute filesystem path for a dataset.
        - For VTAB mode: task must be provided, returns the task‑specific directory.
        - For many_shot mode: task is one of 'cifar100', 'resisc45', 'clevr_distance'.
        - For robustness mode: task is one of the robustness keys (e.g. 'imagenet_train').
        """
        if mode == "vtab":
            if task is None:
                raise ValueError("task must be provided for VTAB mode")
            # VTAB datasets are stored under vtab_root/task_name
            base = self.datasets["vtab_root"]
            return os.path.join(base, task)
        elif mode == "many_shot":
            if task is None:
                raise ValueError("task must be provided for many_shot mode")
            return self.datasets["many_shot"][task]
        elif mode == "robustness":
            if task is None:
                raise ValueError("task must be provided for robustness mode")
            return self.datasets["robustness"][task]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_backbone_name(self) -> str:
        """Return the name of the backbone to be used."""
        return self.backbone.get("name", "")

    # ------------------------------------------------------------------ #
    #  Additional helpers
    # ------------------------------------------------------------------ #
    def get_method_hyperparam_combinations(self, method_name: str) -> List[Dict[str, Any]]:
        """
        Generate a list of all hyperparameter combinations (dicts) for a given method.
        This is used by the hyperparameter tuner to create separate trials.
        """
        hyper = self.get_peft_hyperparams(method_name)
        if not hyper:
            # Method has no tunable hyperparameters → one empty combo
            return [{}]
        # Create Cartesian product of all hyperparameter value lists
        import itertools
        keys = list(hyper.keys())
        values_lists = [hyper[k] for k in keys]
        combinations = []
        for combo in itertools.product(*values_lists):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)
        return combinations

    def __repr__(self) -> str:
        return f"Config(backbone={self.backbone.get('name')}, peft methods={list(self.peft_methods.keys())})"
