"""
prompts.py – Prompt loading and sampling for adversarial manipulation reproduction.

This module implements the PromptLoader class, which loads prompt texts from locally
stored JSON datasets and provides reproducible random sampling.  The datasets are
expected to have been pre‑processed into a unified ``all_prompts.json`` format as
described in the project design.
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np

# Import the Config class for type annotations (no circular dependency at runtime)
from config import Config


class PromptLoader:
    """
    Loads prompts from local dataset files and draws reproducible random samples.

    Each dataset category (English, Chinese, Coding, Safety, …) is stored as a single
    ``all_prompts.json`` file containing a JSON array of string prompts.  The mapping
    between logical category names and their disk locations is read from the global
    configuration.

    Attributes:
        config: The fully loaded configuration object.
        rng: A numpy :class:`~numpy.random.RandomState` seeded with ``config.seed``
            to guarantee deterministic sampling across runs.
    """

    # ------------------------------------------------------------------
    # Internal mapping of the dataset identifier (from config.prompt_categories)
    # to the key in config.dataset_paths where the root directory can be found.
    # ------------------------------------------------------------------
    _DATASET_PATH_KEYS = {
        "alpaca-code": "alpaca_code_path",
        "math":         "math_path",
        "advbench":     "advbench_path",
    }

    def __init__(self, config: Config) -> None:
        """
        Create a PromptLoader bound to the given configuration.

        Args:
            config: The application configuration (parsed from ``config.yaml``).
        """
        if not isinstance(config, Config):
            raise TypeError("config must be an instance of Config")

        self.config: Config = config
        # Seeded random state for reproducible prompt sampling
        self.rng: np.random.RandomState = np.random.RandomState(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_category(self, category: str) -> List[str]:
        """
        Load *all* prompts associated with a single category.

        The location of the category is resolved in two steps:
          1. Look up the category key in ``config.prompt_categories`` to obtain an
             *identifier* string (e.g. ``"lmsys-chat-1m/english"`` or ``"alpaca-code"``).
          2. Map that identifier to a filesystem directory via ``config.dataset_paths``.
             For LMSYS‑Chat‑1M categories the identifier contains a ``/`` that separates
             the root name from a language subfolder; other categories map directly to
             a single dataset root.

        Args:
            category: One of the keys defined in ``config.prompt_categories``.
                Expected values: ``"english"``, ``"chinese"``, ``"spanish"``,
                ``"indonesian"``, ``"persian"``, ``"coding"``, ``"math"``, ``"safety"``.

        Returns:
            A list of prompt strings in the order they appear in the JSON file.

        Raises:
            ValueError: If ``category`` is not a known prompt category.
            FileNotFoundError: If the underlying ``all_prompts.json`` cannot be found
                (e.g. because the dataset has not been pre‑processed).
            json.JSONDecodeError: If the JSON file is malformed.
        """
        # 1. Resolve identifier from config
        prompt_categories = self.config.prompt_categories
        if category not in prompt_categories:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Known categories: {list(prompt_categories.keys())}"
            )
        identifier: str = prompt_categories[category]

        # 2. Determine root directory and optional subfolder
        if identifier.startswith("lmsys-chat-1m/"):
            # Identifier format: "lmsys-chat-1m/<language>"
            dataset_key: str = "lmsys_chat_root"
            subfolder: str = identifier.split("/", 1)[1]
        else:
            # Special dataset (alpaca-code, math, advbench)
            if identifier in self._DATASET_PATH_KEYS:
                dataset_key = self._DATASET_PATH_KEYS[identifier]
            else:
                # Fallback for unknown identifiers – treat as literal path key
                dataset_key = identifier
            subfolder = ""

        # 3. Build full path to the all_prompts.json file
        dataset_root: str = self.config.dataset_paths[dataset_key]
        full_path: str = os.path.join(dataset_root, subfolder, "all_prompts.json")

        # 4. Load and return
        if not os.path.isfile(full_path):
            raise FileNotFoundError(
                f"Prompt dataset not found for category '{category}'. "
                f"Expected file: {full_path}"
            )

        with open(full_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        if not isinstance(prompts, list):
            raise ValueError(
                f"Expected a JSON array in {full_path}, got {type(prompts).__name__}"
            )

        return prompts

    def sample_prompts(self, category: str, n: int) -> List[str]:
        """
        Draw a reproducible random sample of *n* prompts without replacement.

        Args:
            category: Same as for :meth:`load_category`.
            n: Number of prompts to sample.

        Returns:
            A list of *n* prompt strings.

        Raises:
            ValueError: If ``n`` is larger than the total number of available prompts.
            (Also raises any exceptions propagated from :meth:`load_category`.)
        """
        # Load all prompts for the category
        all_prompts: List[str] = self.load_category(category)

        if n > len(all_prompts):
            raise ValueError(
                f"Insufficient prompts for category '{category}': "
                f"requested {n}, available {len(all_prompts)}"
            )

        # Deterministic sample using the seeded random state
        indices: np.ndarray = self.rng.choice(len(all_prompts), size=n, replace=False)
        sampled: List[str] = [all_prompts[i] for i in indices]

        return sampled

