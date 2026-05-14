"""
config.py – Configuration module for reproducing the adversarial manipulation experiments.

This module defines the Config class, which loads and validates settings from a YAML file
(default: config.yaml) and exposes them as typed attributes and properties.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class Config:
    """
    Central configuration object for the reproduction pipeline.

    All hyperparameters, model lists, prompt definitions, API keys, simulation settings,
    and mitigation parameters are loaded from a YAML file and validated.

    Usage:
        config = Config("config.yaml")
        print(config.models)
        print(config.detector_feature_types)
    """

    def __init__(self, path: str = "config.yaml") -> None:
        """
        Load configuration from the given YAML file.

        Args:
            path: Path to the configuration file.
        """
        self._raw: Dict[str, Any] = {}
        self.load(path)

    def load(self, path: str = "config.yaml") -> None:
        """
        Load and validate configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

        # Validate required keys and types, set derived attributes.
        self._validate()

        # ------ Convenience top‑level attributes (directly from raw) ------
        self.seed: int = self._raw.get("seed", 42)
        self.output_dir: str = self._raw["output_dir"]
        self.cache_dir: str = self._raw["cache_dir"]

        self.models: List[str] = self._raw["models"]
        self.prompt_categories: Dict[str, str] = self._raw["prompt_categories"]
        self.num_prompts_per_category: int = self._raw.get("num_prompts_per_category", 200)
        self.num_responses_per_prompt: int = self._raw.get("num_responses_per_prompt", 50)
        self.max_output_tokens: int = self._raw.get("max_output_tokens", 512)
        self.temperature: Optional[float] = self._raw.get("temperature", None)
        self.top_p: Optional[float] = self._raw.get("top_p", None)

        # ------ Dataset paths ------
        self.dataset_paths: Dict[str, str] = self._raw["dataset_paths"]

        # ------ API configuration ------
        self.api: Dict[str, Any] = self._raw["api"]

        # ------ Detector configuration (nested dict) ------
        self.detector: Dict[str, Any] = self._raw["detector"]

        # ------ Simulation configuration ------
        self.simulation: Dict[str, Any] = self._raw["simulation"]

        # ------ Mitigation configuration ------
        if "mitigation" in self._raw:
            self.mitigation: Dict[str, Any] = self._raw["mitigation"]
        else:
            self.mitigation = {"enabled": False}

        # ------ Logging configuration ------
        self.logging: Dict[str, Any] = self._raw.get("logging", {"level": "INFO", "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"})

        # Ensure output/cache directories exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validation logic
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        """
        Ensure all required configuration keys and types are correct.

        Raises:
            ValueError: if any validation fails.
        """
        required_top = [
            "models",
            "prompt_categories",
            "num_prompts_per_category",
            "num_responses_per_prompt",
            "max_output_tokens",
            "detector",
            "simulation",
            "api",
            "dataset_paths",
            "output_dir",
            "cache_dir",
        ]
        for key in required_top:
            if key not in self._raw:
                raise ValueError(f"Missing required configuration key: '{key}'")

        # Models
        if not isinstance(self._raw["models"], list) or not all(isinstance(m, str) for m in self._raw["models"]):
            raise ValueError("'models' must be a list of strings")

        # Prompt categories
        if not isinstance(self._raw["prompt_categories"], dict):
            raise ValueError("'prompt_categories' must be a dict")

        # Numeric ranges
        self._check_positive_int("num_prompts_per_category")
        self._check_positive_int("num_responses_per_prompt")
        self._check_positive_int("max_output_tokens")

        # Detector section
        detector = self._raw["detector"]
        if not isinstance(detector, dict):
            raise ValueError("'detector' must be a dictionary")
        if "feature_types" not in detector:
            raise ValueError("'detector.feature_types' is required")
        allowed_features = {"bow", "tfidf", "length_word", "length_char"}
        for ft in detector["feature_types"]:
            if ft not in allowed_features:
                raise ValueError(f"Invalid feature type: '{ft}'. Allowed: {allowed_features}")

        if "test_size" in detector:
            test_size = detector["test_size"]
            if not (0 < test_size < 1):
                raise ValueError("detector.test_size must be between 0 and 1 (exclusive)")

        if "identity_prompts" not in detector or not isinstance(detector["identity_prompts"], list):
            raise ValueError("detector.identity_prompts must be a list of strings")

        if "keyword_map" not in detector or not isinstance(detector["keyword_map"], dict):
            raise ValueError("detector.keyword_map must be a dictionary")

        # Simulation section
        sim = self._raw["simulation"]
        if not isinstance(sim, dict):
            raise ValueError("'simulation' must be a dictionary")
        if "num_models" not in sim or sim["num_models"] <= 0:
            raise ValueError("simulation.num_models must be a positive integer")
        if "k_factor" not in sim or sim["k_factor"] <= 0:
            raise ValueError("simulation.k_factor must be positive")
        if "tie_prob" not in sim or not (0 <= sim["tie_prob"] <= 1):
            raise ValueError("simulation.tie_prob must be between 0 and 1")
        if "attacker_accuracy" not in sim or not (0 <= sim["attacker_accuracy"] <= 1):
            raise ValueError("simulation.attacker_accuracy must be between 0 and 1")
        valid_strategies = {"do_nothing", "random_upvote", "vote_tie", "vote_tie_both_bad"}
        if sim.get("non_target_strategy", "do_nothing") not in valid_strategies:
            raise ValueError(f"Invalid non_target_strategy. Allowed: {valid_strategies}")

        # Mitigation section (optional)
        if "mitigation" in self._raw:
            mit = self._raw["mitigation"]
            if not isinstance(mit, dict):
                raise ValueError("'mitigation' must be a dictionary if present")
            if mit.get("enabled", False):
                if "noise_scales" not in mit:
                    raise ValueError("mitigation.noise_scales is required when mitigation is enabled")
                if not all(isinstance(s, (int, float)) and s >= 0 for s in mit["noise_scales"]):
                    raise ValueError("mitigation.noise_scales must be a list of non-negative numbers")
                if "significance_level" in mit:
                    sl = mit["significance_level"]
                    if not (0 < sl < 1):
                        raise ValueError("mitigation.significance_level must be in (0,1)")

        # API section
        api = self._raw["api"]
        if not isinstance(api, dict):
            raise ValueError("'api' must be a dictionary")
        for key_env in ["openai_key_env", "anthropic_key_env", "google_key_env", "together_key_env"]:
            if key_env not in api:
                raise ValueError(f"Missing API environment variable name: '{key_env}'")

        # Dataset paths must be present
        dp = self._raw["dataset_paths"]
        if not isinstance(dp, dict):
            raise ValueError("'dataset_paths' must be a dictionary")
        required_ds = ["lmsys_chat_root", "math_path", "advbench_path", "alpaca_code_path"]
        for ds_key in required_ds:
            if ds_key not in dp:
                raise ValueError(f"Missing dataset path: '{ds_key}'")

    def _check_positive_int(self, key: str) -> None:
        """Helper to assert a configuration value is a positive integer."""
        val = self._raw.get(key)
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"'{key}' must be a positive integer, got {val}")

    # ------------------------------------------------------------------
    # Convenient properties (to provide typed access as per design)
    # ------------------------------------------------------------------
    @property
    def detector_feature_types(self) -> List[str]:
        """Feature types used by the training‑based detector."""
        return self.detector["feature_types"]

    @property
    def detector_classifier(self) -> str:
        """Classifier type for training‑based detector."""
        return self.detector.get("classifier", "logistic_regression")

    @property
    def detector_test_size(self) -> float:
        """Test split fraction for detector training."""
        return self.detector.get("test_size", 0.2)

    @property
    def detector_random_state(self) -> int:
        """Random state for detector training splits."""
        return self.detector.get("random_state", 42)

    @property
    def identity_prompts(self) -> List[str]:
        """Prompts used for identity‑probing detection."""
        return self.detector["identity_prompts"]

    @property
    def identity_num_queries(self) -> int:
        """Number of queries per prompt for identity‑probing accuracy."""
        return self.detector.get("identity_num_queries", 1000)

    @property
    def keyword_map(self) -> Dict[str, List[str]]:
        """Mapping from model family to list of identifying keywords."""
        return self.detector["keyword_map"]

    @property
    def simulation_num_models(self) -> int:
        return self.simulation["num_models"]

    @property
    def simulation_initial_rating_mean(self) -> float:
        return self.simulation.get("initial_rating_mean", 1000.0)

    @property
    def simulation_initial_rating_std(self) -> float:
        return self.simulation.get("initial_rating_std", 200.0)

    @property
    def simulation_k_factor(self) -> float:
        return self.simulation["k_factor"]

    @property
    def simulation_tie_prob(self) -> float:
        return self.simulation["tie_prob"]

    @property
    def simulation_num_genuine_votes(self) -> int:
        return self.simulation.get("num_genuine_votes", 1600000)

    @property
    def simulation_attacker_accuracy(self) -> float:
        return self.simulation["attacker_accuracy"]

    @property
    def simulation_non_target_strategy(self) -> str:
        return self.simulation.get("non_target_strategy", "do_nothing")

    @property
    def simulation_target_models(self) -> Dict[str, List[str]]:
        return self.simulation.get("target_models", {})

    @property
    def simulation_target_movements(self) -> Dict[str, List[int]]:
        return self.simulation.get("target_movements", {})

    @property
    def simulation_ablations(self) -> Dict[str, Any]:
        return self.simulation.get("ablations", {})

    @property
    def mitigation_enabled(self) -> bool:
        return self.mitigation.get("enabled", False)

    @property
    def mitigation_noise_scales(self) -> List[float]:
        return self.mitigation.get("noise_scales", [])

    @property
    def mitigation_significance_level(self) -> float:
        return self.mitigation.get("significance_level", 0.01)

    @property
    def mitigation_num_null_simulations(self) -> int:
        return self.mitigation.get("num_null_simulations", 1000)

    @property
    def mitigation_num_attacker_votes(self) -> int:
        return self.mitigation.get("num_attacker_votes", 100)

    @property
    def mitigation_strategies(self) -> List[str]:
        return self.mitigation.get("strategies", ["naive", "smart"])

    @property
    def logging_level(self) -> str:
        return self.logging.get("level", "INFO")

    @property
    def logging_format(self) -> str:
        return self.logging.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<Config: models={len(self.models)} | output={self.output_dir}>"
