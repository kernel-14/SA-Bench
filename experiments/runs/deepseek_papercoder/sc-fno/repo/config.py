# config.py
# ============================================================================
# Purpose: Load and validate experiment settings from a YAML configuration file.
#          Provides a single, typed Config object used by all other modules.
# ============================================================================

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class Config:
    """Central configuration object for the SC-FNO reproduction.

    Attributes are populated from a YAML file via `Config.from_yaml`.

    Example:
        cfg = Config.from_yaml("config.yaml")
        print(cfg.equation)           # 'pde1'
        print(cfg.sol_params['M'])   # 5
    """

    global_params: Dict[str, Any]   # global settings (seed, dirs, device)
    equation: str                   # active equation key
    sol_params: Dict[str, Any]      # equation‑specific parameters (discretization, ranges, etc.)
    data_params: Dict[str, Any]     # dataset generation options
    model_params: Dict[str, Any]    # FNO architecture hyperparameters
    training_params: Dict[str, Any] # training loop configuration (loss, optimizer, epochs, ...)
    eval_params: Dict[str, Any]     # evaluation settings
    inversion_params: Dict[str, Any] = field(default_factory=dict)  # optional inversion experiments
    data_volume_params: Optional[Dict[str, Any]] = None  # optional data‑volume study

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Factory method: parse a YAML file and return a frozen Config instance.

        Args:
            path: Path to the `config.yaml` file.

        Returns:
            A fully populated Config dataclass.

        Raises:
            FileNotFoundError: if the YAML file cannot be read.
            ValueError: if the chosen equation is not defined in the YAML.
        """
        yaml_path = Path(path)
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        # ------------------------------------------------------------------
        # 1. Global parameters (required section)
        # ------------------------------------------------------------------
        global_params = raw.get("global", {})
        if not isinstance(global_params, dict):
            raise ValueError("Configuration must contain a 'global' section.")

        # ------------------------------------------------------------------
        # 2. Equation selection
        # ------------------------------------------------------------------
        equation = raw.get("equation")
        if equation is None:
            raise ValueError("Configuration must specify an 'equation' key.")

        equations_dict = raw.get("equations", {})
        if equation not in equations_dict:
            raise ValueError(
                f"Equation '{equation}' not found in the 'equations' section. "
                f"Available: {list(equations_dict.keys())}"
            )
        sol_params = equations_dict[equation]

        # ------------------------------------------------------------------
        # 3. Data generation parameters
        # ------------------------------------------------------------------
        data_params = raw.get("data", {})
        if not isinstance(data_params, dict):
            raise ValueError("Configuration must contain a 'data' section.")

        # ------------------------------------------------------------------
        # 4. Model architecture parameters
        # ------------------------------------------------------------------
        model_params = raw.get("model", {})
        if not isinstance(model_params, dict):
            raise ValueError("Configuration must contain a 'model' section.")

        # ------------------------------------------------------------------
        # 5. Training parameters (includes loss and PINN sub‑dicts)
        # ------------------------------------------------------------------
        training_params = raw.get("training", {})
        if not isinstance(training_params, dict):
            raise ValueError("Configuration must contain a 'training' section.")

        # ------------------------------------------------------------------
        # 6. Evaluation parameters
        # ------------------------------------------------------------------
        eval_params = raw.get("evaluation", {})
        if not isinstance(eval_params, dict):
            # evaluation section is mandatory; provide a default but prefer explicit.
            eval_params = {}

        # ------------------------------------------------------------------
        # 7. Optional inversion section
        # ------------------------------------------------------------------
        inversion_dict = raw.get("inversion", {})
        if not isinstance(inversion_dict, dict):
            inversion_dict = {}

        # ------------------------------------------------------------------
        # 8. Optional data‑volume study parameters
        # ------------------------------------------------------------------
        data_volume = raw.get("data_volume", None)
        if data_volume is not None and not isinstance(data_volume, dict):
            data_volume = None

        # ------------------------------------------------------------------
        # Assemble and return the immutable Config object
        # ------------------------------------------------------------------
        return cls(
            global_params=global_params,
            equation=equation,
            sol_params=sol_params,
            data_params=data_params,
            model_params=model_params,
            training_params=training_params,
            eval_params=eval_params,
            inversion_params=inversion_dict,
            data_volume_params=data_volume,
        )

    def __post_init__(self):
        """Optional validation after initialization. (left empty for now)"""
        pass
