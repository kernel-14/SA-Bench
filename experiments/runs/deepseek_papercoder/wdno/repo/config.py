## config.py
"""
Central configuration module for the Wavelet Diffusion Neural Operator (WDNO) reproduction.
Reads a YAML file and provides a structured, validated Config object accessible by all other modules.
"""

import yaml
import copy
from typing import Dict, Any, List, Optional, Union


class Config:
    """
    Loads, validates, and provides access to experiment‑specific settings from config.yaml.

    Usage:
        config = Config.from_yaml('config.yaml')
        train_cfg = config.get_training_config()
        wav_cfg  = config.get_wavelet_config()
        ...
    """

    _SUPPORTED_EXPERIMENTS = {
        "burgers_1d_sim",
        "burgers_1d_ctrl",
        "advection_1d",
        "cfd_1d",
        "fluid_2d_sim",
        "fluid_2d_ctrl",
        "era5",
    }

    # Mapping of (experiment, task_type, model_role) -> number of condition channels
    # model_role: 'base' for BRM, 'super' for SRM (super adds extra low‑res channels)
    _COND_CHANNELS: Dict[tuple, int] = {
        # 1D Burgers simulation – condition: u0 (2) + f (4) = 6
        ("burgers_1d_sim", "sim", "base"): 6,
        ("burgers_1d_sim", "sim", "super"): 6 + 4,   # extra low‑res = 4
        # 1D Burgers control – condition: u0 (2) + u_T (2) = 4
        ("burgers_1d_ctrl", "ctrl", "base"): 4,
        ("burgers_1d_ctrl", "ctrl", "super"): 4 + 4,
        # 1D Advection – condition: only u0 (2)
        ("advection_1d", "sim", "base"): 2,
        ("advection_1d", "sim", "super"): 2 + 4,
        # 1D Compressible Navier‑Stokes – condition: u0 (2)
        ("cfd_1d", "sim", "base"): 2,
        ("cfd_1d", "sim", "super"): 2 + 4,
        # 2D incompressible fluid simulation – condition: initial density (4) + control (8) = 12
        ("fluid_2d_sim", "sim", "base"): 12,
        ("fluid_2d_sim", "sim", "super"): 12 + 8,  # extra low‑res = 8 subbands from 3D wavelet
        # 2D incompressible fluid control – condition: initial density (4)
        ("fluid_2d_ctrl", "ctrl", "base"): 4,
        ("fluid_2d_ctrl", "ctrl", "super"): 4 + 8,
        # ERA5 – condition: past frames (8 subbands from 3D wavelet)
        ("era5", "sim", "base"): 8,
        ("era5", "sim", "super"): 8 + 8,
    }

    # Number of wavelet subbands depending on transform dimensionality
    _WAVELET_SUBBANDS = {
        2: 4,   # 2D DWT → LL, LH, HL, HH
        3: 8,   # 3D DWT → 8 subbands
    }

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        """
        Initialise from a dictionary (already loaded YAML).
        Performs basic validation.
        """
        if not isinstance(config_dict, dict):
            raise TypeError("config_dict must be a dictionary")
        if "experiment" not in config_dict:
            raise ValueError("Configuration missing required top‑level key 'experiment'")
        experiment = str(config_dict["experiment"])
        if experiment not in self._SUPPORTED_EXPERIMENTS:
            raise ValueError(
                f"Unsupported experiment '{experiment}'. Must be one of {self._SUPPORTED_EXPERIMENTS}"
            )

        self._cfg = copy.deepcopy(config_dict)
        # Provide some defaults for optional fields
        self._cfg.setdefault("seed", 42)
        self._cfg.setdefault("device", "cuda")

        # Data section existence
        if "data" not in self._cfg:
            raise ValueError("Missing 'data' section in configuration")
        if "wavelet" not in self._cfg:
            raise ValueError("Missing 'wavelet' section")
        if "model_base" not in self._cfg:
            raise ValueError("Missing 'model_base' section")
        if "model_super" not in self._cfg:
            self._cfg["model_super"] = {}  # able to fall back gracefully, but we require it
        if "model_3d" not in self._cfg:
            raise ValueError("Missing 'model_3d' section")
        if "diffusion" not in self._cfg:
            raise ValueError("Missing 'diffusion' section")
        if "training" not in self._cfg:
            raise ValueError("Missing 'training' section")
        if "control" not in self._cfg:
            self._cfg["control"] = {}
        if "eval" not in self._cfg:
            self._cfg["eval"] = {"num_samples": 50}

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the full configuration dictionary (for logging/saving)."""
        return copy.deepcopy(self._cfg)

    # ----------------------------------------------------------------------
    # Experiment identification helpers
    # ----------------------------------------------------------------------
    def _is_1d(self) -> bool:
        return "1d" in self._cfg["experiment"]

    def _is_2d(self) -> bool:
        exp = self._cfg["experiment"]
        return "2d" in exp or "era5" in exp

    def _is_control(self) -> bool:
        return "ctrl" in self._cfg["experiment"]

    def _task_type(self) -> str:
        """Return 'sim' or 'ctrl'."""
        return "ctrl" if self._is_control() else "sim"

    def _model_role(self, role: str = "base") -> str:
        """Return the role ('base' or 'super') – used for condition channel lookup."""
        return role

    # ----------------------------------------------------------------------
    # Basic accessors
    # ----------------------------------------------------------------------
    def get_experiment_name(self) -> str:
        return self._cfg["experiment"]

    def get_seed(self) -> int:
        return int(self._cfg["seed"])

    def get_device(self) -> str:
        return self._cfg.get("device", "cuda")

    def get_eval_config(self) -> Dict[str, Any]:
        return self._cfg["eval"]

    # ----------------------------------------------------------------------
    # Data configuration
    # ----------------------------------------------------------------------
    def get_data_config(self) -> Dict[str, Any]:
        """
        Returns the data section, adjusting super_res_scales for 2D experiments.
        For 2D, the paper only shows spatial super‑resolution (64→128).
        """
        data_cfg = copy.deepcopy(self._cfg["data"])
        # For 2D, override super_res_scales if needed
        if self._is_2d():
            # Only one level of super‑resolution: original and 2x spatial
            data_cfg["super_res_scales"] = [1, 0.5]
        # Ensure downsampling method exists
        data_cfg.setdefault("downsampling", "average")
        return data_cfg

    # ----------------------------------------------------------------------
    # Wavelet configuration
    # ----------------------------------------------------------------------
    def get_wavelet_config(self) -> Dict[str, Any]:
        """
        Returns wavelet parameters based on experiment dimensionality.
        Keys: wname, mode, level, ndim, rec_tol.
        """
        wavelet_cfg = self._cfg["wavelet"]
        if self._is_2d():
            wname = wavelet_cfg["wname_2d"]
            mode = wavelet_cfg["mode_2d"]
            ndim = 3
        else:
            wname = wavelet_cfg["wname_1d"]
            mode = wavelet_cfg["mode_1d"]
            ndim = 2
        return {
            "wname": wname,
            "mode": mode,
            "level": wavelet_cfg.get("level", 1),
            "ndim": ndim,
            "rec_tol": wavelet_cfg.get("rec_tol", 1.0e-6),
        }

    # ----------------------------------------------------------------------
    # Model configurations (BRM, SRM, 3D)
    # ----------------------------------------------------------------------
    def get_base_model_config(self) -> Dict[str, Any]:
        """
        Returns the Base‑Resolution Model (BRM) U‑Net configuration.
        Computes input channels as noisy wavelet subbands + condition channels.
        """
        base_cfg = copy.deepcopy(self._cfg["model_base"])
        ndim = 2 if self._is_1d() else 3   # 1D experiments use 2D U‑Net, 2D use 3D U‑Net
        n_subbands = self._WAVELET_SUBBANDS[ndim]
        cond_channels = self._get_condition_channels(role="base")
        base_cfg["in_channels"] = n_subbands + cond_channels
        base_cfg["out_channels"] = base_cfg.get("out_channels", n_subbands)
        return base_cfg

    def get_super_model_config(self) -> Dict[str, Any]:
        """
        Returns Super‑Resolution Model (SRM) U‑Net configuration.
        Inherits from base model, adds extra low‑res wavelet channels.
        """
        base_cfg = self.get_base_model_config()  # already enriched with computed channels
        super_cfg = copy.deepcopy(self._cfg.get("model_super", {}))
        ndim = 2 if self._is_1d() else 3
        n_subbands = self._WAVELET_SUBBANDS[ndim]
        # The SRM takes additional low‑res wavelet as condition.
        # The number of extra channels equals n_subbands.
        extra_low_channels = n_subbands
        cond_channels = self._get_condition_channels(role="super")
        # total in_channels includes noisy wavelet + condition (which already includes extra low‑res)
        # but the base model config returns in_channels = n_subbands + base_cond.
        # For SRM, condition is low‑res wavelet + high‑res condition, so total cond_channels = base_cond + n_subbands.
        # So we just override the in_channels computed by get_base_model_config.
        # We'll recompute here.
        total_in = n_subbands + cond_channels
        # Update the base config with SRM specific overrides if any (e.g., attention dims, etc.)
        for k, v in super_cfg.items():
            if k == "extra_channels_low":
                continue
            base_cfg[k] = v
        base_cfg["in_channels"] = total_in
        base_cfg["out_channels"] = n_subbands
        base_cfg["extra_channels_low"] = extra_low_channels
        return base_cfg

    def get_3d_model_config(self) -> Dict[str, Any]:
        """
        Returns configuration for the 3D U‑Net used in 2D fluid/ERA5 experiments.
        Computes input channels as 8 (noisy wavelet) + condition channels.
        """
        model3d_cfg = copy.deepcopy(self._cfg["model_3d"])
        # 3D wavelet always yields 8 subbands
        n_subbands = 8
        cond_channels = self._get_condition_channels(role="base", ndim=3)
        model3d_cfg["in_channels"] = n_subbands + cond_channels
        model3d_cfg["out_channels"] = model3d_cfg.get("out_channels", n_subbands)
        return model3d_cfg

    def _get_condition_channels(self, role: str = "base", ndim: Optional[int] = None) -> int:
        """
        Looks up the number of condition channels (excluding noisy wavelet channels)
        based on experiment, task type, and model role.
        If ndim is None, it's inferred from experiment.
        """
        if ndim is None:
            ndim = 3 if self._is_2d() else 2
        exp = self._cfg["experiment"]
        task = self._task_type()
        key = (exp, task, role)
        if key not in self._COND_CHANNELS:
            raise KeyError(f"No condition channel mapping defined for {key}. Please add it to Config._COND_CHANNELS.")
        return self._COND_CHANNELS[key]

    # ----------------------------------------------------------------------
    # Diffusion configuration
    # ----------------------------------------------------------------------
    def get_diffusion_config(self) -> Dict[str, Any]:
        """
        Returns diffusion process parameters, selecting DDIM steps and
        guidance weight based on experiment type.
        """
        diff_cfg = copy.deepcopy(self._cfg["diffusion"])
        # DDIM steps
        if self._is_2d():
            diff_cfg["ddim_steps"] = diff_cfg.get("ddim_steps_2d", 100)
        else:
            diff_cfg["ddim_steps"] = diff_cfg.get("ddim_steps_1d", 50)
        # If a specific guidance_weight is not set, default to 0 (simulation) or a
        # positive value for control (will be overridden later by get_control_config).
        if "guidance_weight" not in diff_cfg:
            diff_cfg["guidance_weight"] = 1.0 if self._is_control() else 0.0
        return diff_cfg

    # ----------------------------------------------------------------------
    # Training configuration
    # ----------------------------------------------------------------------
    def get_training_config(self) -> Dict[str, Any]:
        """
        Returns training hyperparameters. Batch size and steps depend on
        dimensionality.
        """
        train_cfg = copy.deepcopy(self._cfg["training"])
        if self._is_2d():
            train_cfg.setdefault("batch_size", self._cfg["training"].get("batch_size_2d", 8))
        else:
            train_cfg.setdefault("batch_size", self._cfg["training"].get("batch_size_1d", 16))
        # Add default steps if not present
        train_cfg.setdefault("steps_brm", 190000)
        train_cfg.setdefault("steps_srm", 190000)
        train_cfg.setdefault("steps_ctrl_surrogate", 50000)
        return train_cfg

    # ----------------------------------------------------------------------
    # Control configuration
    # ----------------------------------------------------------------------
    def get_control_config(self) -> Dict[str, Any]:
        """
        Returns control‑specific settings: guidance lambda and surrogate model info.
        """
        ctrl_cfg = copy.deepcopy(self._cfg.get("control", {}))
        # Lambda
        if "lambda" not in ctrl_cfg:
            if self._is_2d():
                ctrl_cfg["lambda"] = ctrl_cfg.get("lambda_2d", 11.5)
            else:
                ctrl_cfg["lambda"] = ctrl_cfg.get("lambda_1d", 120000)
        # Surrogate
        surrogate = ctrl_cfg.get("surrogate", {})
        surrogate.setdefault("type", "unet_2d" if self._is_1d() else "unet_3d")
        surrogate.setdefault("lr", 1.0e-4)
        surrogate.setdefault("epochs", 500)
        surrogate.setdefault("batch_size", 16)
        ctrl_cfg["surrogate"] = surrogate
        return ctrl_cfg

    # ----------------------------------------------------------------------
    # Convenience method for general hyperparameter dict
    # ----------------------------------------------------------------------
    def get_all_configs(self) -> Dict[str, Any]:
        """
        Returns a single dictionary with all sub‑configurations needed by
        the main pipeline. Useful for passing a single object to constructors.
        """
        return {
            "experiment": self.get_experiment_name(),
            "seed": self.get_seed(),
            "device": self.get_device(),
            "data": self.get_data_config(),
            "wavelet": self.get_wavelet_config(),
            "model_base": self.get_base_model_config(),
            "model_super": self.get_super_model_config(),
            "model_3d": self.get_3d_model_config(),
            "diffusion": self.get_diffusion_config(),
            "training": self.get_training_config(),
            "control": self.get_control_config(),
            "eval": self.get_eval_config(),
        }
