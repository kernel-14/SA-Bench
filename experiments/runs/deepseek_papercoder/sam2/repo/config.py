# config.py

import os
import yaml
from collections.abc import Mapping
from typing import Any, Dict, Optional, Union

class AttrDict(dict):
    """
    A dictionary subclass that allows attribute-like access to its keys.
    This enables dot notation (e.g., config.model.resolution) while still
    supporting standard dictionary operations.
    """
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


def _to_attrdict(obj: Any) -> Any:
    """
    Recursively convert a nested dictionary (or list) into AttrDict objects.
    Leaves strings, numbers, booleans, and None unchanged.
    """
    if isinstance(obj, Mapping):
        # Convert each mapping to an AttrDict and recurse on values
        return AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_attrdict(item) for item in obj]
    else:
        return obj


class Config:
    """
    Central configuration class for SAM 2 reproduction.

    Loads a YAML file and provides hierarchial, dot-notation access to all
    parameters (model architecture, training hyperparameters, data paths,
    augmentation settings, and evaluation protocols). Post-loading tweaks
    ensure compatibility with the paper's specifications (e.g., handling the
    missing internal dataset).

    Typical usage:
        config = Config("config.yaml")
        lr = config.training.pretrain.learning_rate
        sav_root = config.data.sav_root

    Attributes:
        _cfg (AttrDict): The root of the configuration tree. All sections
            (model, training, data, augmentation, evaluation) are accessible
            as attributes of this object.
    """

    def __init__(self, config_path: str) -> None:
        """
        Args:
            config_path: Path to a YAML configuration file (e.g., "config.yaml").
        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file cannot be parsed.
        """
        with open(config_path, 'r') as f:
            raw = yaml.safe_load(f)
        # Convert all nested dicts/lists to AttrDict for easy dot access
        self._cfg: AttrDict = _to_attrdict(raw)

        # Post-processing adjustments mirroring the paper's experimental setup
        self._normalize_paths()
        self._handle_internal_dataset()

    def _normalize_paths(self) -> None:
        """
        Convert all dataset root paths to absolute paths to avoid ambiguity
        when scripts are executed from different directories.
        """
        data_keys = [
            "sa1b_root", "sav_root", "davis_root", "mose_root",
            "ytv_root", "internal_root"
        ]
        for key in data_keys:
            if key in self._cfg.data and self._cfg.data[key] is not None:
                self._cfg.data[key] = os.path.abspath(self._cfg.data[key])

    def _handle_internal_dataset(self) -> None:
        """
        If the internal dataset root is None (not released), remove the
        'internal' key from the data mix weights and renormalize the remaining
        weights so they sum to 1. This aligns the training mixture with the
        public datasets (SA-V + VOS + SA-1B) used in the paper's ablation.
        """
        if self._cfg.data.internal_root is None:
            mix = self._cfg.data.mix_weights
            if "internal" in mix:
                del mix["internal"]
            total = sum(mix.values())
            if total > 0:
                for k in mix:
                    mix[k] /= total

    # Allow top-level access via dot notation (e.g., config.model)
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            # Avoid recursion when accessing private attributes
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if name in self._cfg:
            return self._cfg[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # Allow dict-like access (e.g., config["model"])
    def __getitem__(self, key: str) -> Any:
        return self._cfg[key]

    def __repr__(self) -> str:
        return f"Config({dict(self._cfg)})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the underlying configuration as a plain (non-AttrDict) dictionary.
        Useful for serialisation or passing to libraries that expect standard dicts.
        """
        def _to_builtin(obj):
            if isinstance(obj, AttrDict):
                return {k: _to_builtin(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_to_builtin(item) for item in obj]
            else:
                return obj
        return _to_builtin(self._cfg)

