## config.py
"""Configuration dataclasses for the adversarial leaderboard manipulation paper.

This module defines ModelConfig and Config dataclasses that translate config.yaml
into typed Python objects used throughout the project. It has zero internal
project dependencies and must be imported first by all other modules.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Valid API provider identifiers.
_VALID_PROVIDERS = frozenset({"openai", "anthropic", "google", "together"})


@dataclass
class ModelConfig:
    """Configuration for a single LLM entry from the models list in config.yaml.

    Attributes:
        name: Exact model identifier string used in API calls and cache keys,
            e.g. "claude-3-5-sonnet-20240620".
        organization: Company name, e.g. "Anthropic", "OpenAI", "Google",
            "Meta", "Mistral AI", "Alibaba".
        api_provider: API endpoint provider, one of "openai", "anthropic",
            "google", "together". Note that Google's Gemma models use "together"
            since they are accessed via the Together AI inference API.
        keywords: Name/organization strings used by IdentityProbingDetector for
            keyword matching, e.g. ["Claude", "Anthropic"]. Case-insensitive
            matching is handled by the detector, not here.
        max_tokens: Maximum output tokens for API calls. Default 512 per
            Appendix A.3: "we set the output length to 512 tokens."
    """

    name: str
    organization: str
    api_provider: str
    keywords: List[str]
    max_tokens: int = 512


@dataclass
class Config:
    """Top-level configuration object for all experiments.

    Attributes:
        models: All 22 LLM configurations from Appendix A.1 / Table 6.
        prompt_categories: The 8 category names extracted from
            data_collection.prompt_categories[*].name in YAML.
        n_prompts_per_category: Number of prompts sampled per category.
            Default 200 per Section 2.3.
        n_responses_per_model: Responses collected per model per prompt.
            Default 50 per Section 2.3.
        n_identity_queries: Queries per identity-probing prompt per model.
            Default 1000 per Section 2.4.1.
        train_test_split: Fraction of data used for training in the
            training-based detector. Default 0.8 per Section 2.3.
        random_state: Global random seed for all numpy/sklearn operations.
            Default 42 per Section 2.3.
        detection_accuracy: Assumed detector accuracy for main simulation.
            Default 0.95 per Section 3.1.
        bt_scale_factor: Scale factor s in the Bradley-Terry logistic model.
            Default 1.0 per simulation config.
        significance_level: Alpha for the malicious user hypothesis test.
            Default 0.01 per Section 4.2.3.
        output_dir: Root directory for all output files (tables, figures, logs).
        cache_dir: Root directory for disk-based response cache.
        raw: Full parsed YAML dict, giving downstream modules access to nested
            config values (e.g. simulation.high_ranked_targets,
            mitigations.cost_model.c_detector) without bloating this interface.
    """

    models: List[ModelConfig]
    prompt_categories: List[str]
    n_prompts_per_category: int = 200
    n_responses_per_model: int = 50
    n_identity_queries: int = 1000
    train_test_split: float = 0.8
    random_state: int = 42
    detection_accuracy: float = 0.95
    bt_scale_factor: float = 1.0
    significance_level: float = 0.01
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration values and create required directories."""
        # Validate each model's api_provider.
        for model in self.models:
            if model.api_provider not in _VALID_PROVIDERS:
                raise ValueError(
                    f"Model '{model.name}' has invalid api_provider "
                    f"'{model.api_provider}'. Must be one of {_VALID_PROVIDERS}."
                )
            if not model.name:
                raise ValueError("ModelConfig.name must be a non-empty string.")
            if not model.keywords:
                raise ValueError(
                    f"ModelConfig '{model.name}' must have at least one keyword."
                )
            if model.max_tokens <= 0:
                raise ValueError(
                    f"ModelConfig '{model.name}' max_tokens must be positive, "
                    f"got {model.max_tokens}."
                )

        # Validate scalar fields.
        if not (0.0 < self.train_test_split < 1.0):
            raise ValueError(
                f"train_test_split must be in (0, 1), got {self.train_test_split}."
            )
        if not (0.0 < self.detection_accuracy <= 1.0):
            raise ValueError(
                f"detection_accuracy must be in (0, 1], got {self.detection_accuracy}."
            )
        if not (0.0 < self.significance_level < 1.0):
            raise ValueError(
                f"significance_level must be in (0, 1), got {self.significance_level}."
            )
        if self.n_prompts_per_category <= 0:
            raise ValueError(
                f"n_prompts_per_category must be positive, "
                f"got {self.n_prompts_per_category}."
            )
        if self.n_responses_per_model <= 0:
            raise ValueError(
                f"n_responses_per_model must be positive, "
                f"got {self.n_responses_per_model}."
            )
        if self.n_identity_queries <= 0:
            raise ValueError(
                f"n_identity_queries must be positive, "
                f"got {self.n_identity_queries}."
            )
        if self.random_state < 0:
            raise ValueError(
                f"random_state must be non-negative, got {self.random_state}."
            )
        if self.bt_scale_factor <= 0.0:
            raise ValueError(
                f"bt_scale_factor must be positive, got {self.bt_scale_factor}."
            )
        if not self.prompt_categories:
            raise ValueError("prompt_categories must contain at least one entry.")

        # Ensure output and cache directories exist.
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Create standard output subdirectories used by downstream modules.
        for subdir in ("tables", "figures", "responses", "simulation"):
            Path(self.output_dir, subdir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Construct a Config instance from a YAML configuration file.

        Args:
            path: Path to the config.yaml file.

        Returns:
            A fully validated Config instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            KeyError: If a required YAML key is missing.
            ValueError: If any configuration value fails validation.
        """
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with yaml_path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh)

        # --- Build ModelConfig list from models[*] ---
        models: List[ModelConfig] = []
        for entry in raw["models"]:
            models.append(
                ModelConfig(
                    name=str(entry["name"]),
                    organization=str(entry["organization"]),
                    api_provider=str(entry["api_provider"]),
                    keywords=[str(k) for k in entry["keywords"]],
                    max_tokens=int(entry.get("max_tokens", 512)),
                )
            )

        # --- Extract prompt_categories from data_collection.prompt_categories[*].name ---
        data_collection: Dict[str, Any] = raw["data_collection"]
        prompt_categories: List[str] = [
            str(cat["name"]) for cat in data_collection["prompt_categories"]
        ]

        # --- Extract scalar fields from their respective YAML paths ---
        n_prompts_per_category: int = int(
            data_collection.get("n_prompts_per_category", 200)
        )
        n_responses_per_model: int = int(
            data_collection.get("n_responses_per_model", 50)
        )
        n_identity_queries: int = int(
            data_collection.get("n_identity_queries", 1000)
        )

        training_based: Dict[str, Any] = raw["training_based_detector"]
        train_test_split: float = float(
            training_based.get("train_test_split", 0.8)
        )

        reproducibility: Dict[str, Any] = raw["reproducibility"]
        random_state: int = int(reproducibility.get("random_state", 42))

        simulation: Dict[str, Any] = raw["simulation"]
        detection_accuracy: float = float(
            simulation.get("detection_accuracy", 0.95)
        )
        bt_scale_factor: float = float(
            simulation.get("bt_scale_factor", 1.0)
        )

        mitigations: Dict[str, Any] = raw["mitigations"]
        malicious_detection: Dict[str, Any] = mitigations[
            "malicious_user_detection"
        ]
        significance_level: float = float(
            malicious_detection.get("significance_level", 0.01)
        )

        output_dir: str = str(raw.get("output_dir", "outputs"))
        cache_dir: str = str(raw.get("cache_dir", "cache"))

        return cls(
            models=models,
            prompt_categories=prompt_categories,
            n_prompts_per_category=n_prompts_per_category,
            n_responses_per_model=n_responses_per_model,
            n_identity_queries=n_identity_queries,
            train_test_split=train_test_split,
            random_state=random_state,
            detection_accuracy=detection_accuracy,
            bt_scale_factor=bt_scale_factor,
            significance_level=significance_level,
            output_dir=output_dir,
            cache_dir=cache_dir,
            raw=raw,
        )

    def get_model_by_name(self, name: str) -> Optional[ModelConfig]:
        """Return the ModelConfig for the given model name, or None if not found.

        Args:
            name: Exact model name string, e.g. "claude-3-5-sonnet-20240620".

        Returns:
            The matching ModelConfig, or None.
        """
        for model in self.models:
            if model.name == name:
                return model
        return None

    def get_models_by_provider(self, provider: str) -> List[ModelConfig]:
        """Return all ModelConfigs for a given API provider.

        Args:
            provider: One of "openai", "anthropic", "google", "together".

        Returns:
            List of ModelConfig instances with the matching api_provider.
        """
        return [m for m in self.models if m.api_provider == provider]

    def get_model_names(self) -> List[str]:
        """Return the list of all model name strings.

        Returns:
            List of model name strings in the order they appear in config.yaml.
        """
        return [m.name for m in self.models]
