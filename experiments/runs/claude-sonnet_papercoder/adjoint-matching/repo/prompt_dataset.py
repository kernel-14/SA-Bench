## prompt_dataset.py
"""Prompt dataset management for Adjoint Matching experiments.

This module provides the PromptDataset class that manages the text prompt pool
used throughout training and evaluation. Per the paper (Appendix G), each of
3 independent runs samples 40k training prompts and 1k test prompts from a
total pool of 100k prompts. The splits are disjoint and reproducible via seed.

Configuration alignment (config.yaml):
    data.total_prompts: 100000       → num_prompts argument
    data.num_train_prompts: 40000    → train_size in get_train_test_split()
    data.num_test_prompts: 1000      → test_size in get_train_test_split()
    data.prompts_file: "data/prompts.txt" → prompts_file argument
    training.seed: 42                → seed argument
    training.batch_size: 40          → batch_size in get_batch()

This file has NO dependencies on other project files. Only stdlib and
optional third-party packages (numpy, datasets) are used.
"""

from __future__ import annotations

import logging
import os
import random
import warnings
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic prompt templates (fallback when no file/dataset is available)
# ---------------------------------------------------------------------------

_SUBJECTS: List[str] = [
    "cat", "dog", "bird", "horse", "lion", "tiger", "elephant", "giraffe",
    "mountain", "ocean", "forest", "desert", "city", "village", "castle",
    "woman", "man", "child", "artist", "scientist", "chef", "warrior",
    "flower", "tree", "river", "waterfall", "sunset", "sunrise", "moon",
    "dragon", "robot", "spaceship", "lighthouse", "bridge", "temple",
]

_ADJECTIVES: List[str] = [
    "beautiful", "majestic", "colorful", "ancient", "futuristic", "serene",
    "dramatic", "vibrant", "mysterious", "elegant", "powerful", "delicate",
    "golden", "silver", "ethereal", "rustic", "modern", "enchanted",
    "photorealistic", "detailed", "stunning", "breathtaking", "cinematic",
]

_SETTINGS: List[str] = [
    "at sunset", "in the rain", "under moonlight", "in a forest",
    "on a mountain top", "by the ocean", "in a city", "in space",
    "in a garden", "in winter", "in autumn", "at dawn", "in fog",
    "underwater", "in the clouds", "in a desert", "in a meadow",
    "in a cave", "on a cliff", "in a marketplace",
]

_STYLES: List[str] = [
    "oil painting", "watercolor", "digital art", "photograph",
    "pencil sketch", "impressionist painting", "concept art",
    "illustration", "3D render", "cinematic shot",
]

_TEMPLATES: List[str] = [
    "A photo of a {subject}",
    "A {adjective} {subject}",
    "A {adjective} {subject} {setting}",
    "{subject} {setting}, {style}",
    "A {style} of a {adjective} {subject}",
    "Beautiful {subject} {setting}",
    "A {adjective} {subject}, {style}",
    "{subject} in a {adjective} landscape",
    "Portrait of a {adjective} {subject}",
    "A {style} depicting a {subject} {setting}",
    "High quality photo of {subject} {setting}",
    "A {adjective} scene with {subject}",
    "{subject} and {subject} {setting}",
    "A {adjective} {subject} with {adjective} background",
    "Close-up of a {adjective} {subject}",
]


def _generate_synthetic_prompts(n: int, seed: int = 42) -> List[str]:
    """Generate n synthetic prompts from templates for fallback use.

    Uses a seeded RNG to ensure reproducibility. The generated prompts
    cover diverse subjects, adjectives, settings, and styles to approximate
    a real prompt distribution.

    Args:
        n: Number of synthetic prompts to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of n synthetic prompt strings.
    """
    rng = random.Random(seed)
    prompts: List[str] = []

    for _ in range(n):
        template: str = rng.choice(_TEMPLATES)
        # Fill template slots
        prompt: str = template.format(
            subject=rng.choice(_SUBJECTS),
            adjective=rng.choice(_ADJECTIVES),
            setting=rng.choice(_SETTINGS),
            style=rng.choice(_STYLES),
        )
        prompts.append(prompt)

    return prompts


# ---------------------------------------------------------------------------
# HuggingFace dataset column name heuristics
# ---------------------------------------------------------------------------

_HF_TEXT_COLUMN_CANDIDATES: List[str] = [
    "prompt", "text", "caption", "description", "sentence",
    "query", "input", "content", "title",
]


def _extract_text_column(dataset_split: "Any") -> List[str]:  # type: ignore[name-defined]
    """Extract text prompts from a HuggingFace dataset split.

    Tries common column names in order of preference. Returns the first
    column found that contains string data.

    Args:
        dataset_split: A HuggingFace Dataset object with column access.

    Returns:
        List of prompt strings extracted from the dataset.

    Raises:
        ValueError: If no suitable text column is found.
    """
    available_columns: List[str] = dataset_split.column_names

    for candidate in _HF_TEXT_COLUMN_CANDIDATES:
        if candidate in available_columns:
            raw_texts = dataset_split[candidate]
            # Filter out None and non-string entries
            texts: List[str] = [
                str(t).strip()
                for t in raw_texts
                if t is not None and str(t).strip()
            ]
            if texts:
                logger.info(
                    "Extracted %d prompts from HuggingFace column '%s'.",
                    len(texts),
                    candidate,
                )
                return texts

    raise ValueError(
        f"Could not find a text column in HuggingFace dataset. "
        f"Available columns: {available_columns}. "
        f"Tried: {_HF_TEXT_COLUMN_CANDIDATES}"
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PromptDataset:
    """Manages the text prompt pool for Adjoint Matching fine-tuning.

    Handles loading prompts from a file, HuggingFace dataset, or synthetic
    fallback. Provides reproducible train/test splits and batch sampling.

    Per the paper (Appendix G):
        - Total prompt pool: 100k prompts
        - Training prompts per run: 40k (sampled from pool)
        - Test prompts per run: 1k (held-out, disjoint from train)
        - 3 independent runs, each with a different seed

    RNG isolation: Three separate random.Random instances are used to prevent
    interference between loading, splitting, and batch sampling operations.

    Attributes:
        seed: Random seed used for all RNG operations.
        all_prompts: Full list of loaded prompts (up to num_prompts entries).
        train_prompts: Training prompts (populated by get_train_test_split()).
        test_prompts: Test prompts (populated by get_train_test_split()).

    Example:
        >>> dataset = PromptDataset("data/prompts.txt", num_prompts=100000, seed=42)
        >>> train, test = dataset.get_train_test_split(40000, 1000)
        >>> batch = dataset.get_batch(40)
        >>> len(batch)
        40
    """

    def __init__(
        self,
        prompts_file: Optional[str] = "data/prompts.txt",
        num_prompts: int = 100000,
        seed: int = 42,
    ) -> None:
        """Initialize the prompt dataset.

        Loads prompts from the specified source, caps to num_prompts entries
        using a seeded shuffle for reproducibility, and initializes RNG
        instances for splitting and batch sampling.

        Args:
            prompts_file: Path to a .txt file (one prompt per line), a
                HuggingFace dataset identifier string (e.g.,
                "poloclub/diffusiondb"), or None for synthetic fallback.
                From config.yaml: data.prompts_file = "data/prompts.txt".
            num_prompts: Maximum number of prompts to load from the source.
                From config.yaml: data.total_prompts = 100000.
            seed: Random seed for reproducibility across 3 independent runs.
                From config.yaml: training.seed = 42.
                For multiple runs, use seed=42, 43, 44 etc.

        Raises:
            ValueError: If num_prompts <= 0.
        """
        if num_prompts <= 0:
            raise ValueError(
                f"num_prompts must be positive, got {num_prompts}."
            )

        self.seed: int = seed
        self.num_prompts: int = num_prompts

        # Three isolated RNG instances to prevent cross-contamination
        # between loading, splitting, and batch sampling operations.
        self._load_rng: random.Random = random.Random(seed)
        self._split_rng: random.Random = random.Random(seed)
        self._batch_rng: random.Random = random.Random(seed)

        # NumPy RNG for any array-based operations
        self._np_rng: np.random.RandomState = np.random.RandomState(seed)

        # Load and cap prompts
        raw_prompts: List[str] = self.load(prompts_file)

        if len(raw_prompts) == 0:
            warnings.warn(
                "No prompts loaded from source. Generating synthetic prompts.",
                UserWarning,
                stacklevel=2,
            )
            raw_prompts = _generate_synthetic_prompts(num_prompts, seed=seed)

        if len(raw_prompts) < num_prompts:
            warnings.warn(
                f"Source has only {len(raw_prompts)} prompts, "
                f"but num_prompts={num_prompts} was requested. "
                f"Using all {len(raw_prompts)} available prompts.",
                UserWarning,
                stacklevel=2,
            )
            # Use all available prompts without capping
            self.all_prompts: List[str] = raw_prompts
        else:
            # Shuffle with seeded RNG before capping to ensure reproducibility
            # when the source has more than num_prompts entries.
            indices: List[int] = list(range(len(raw_prompts)))
            self._load_rng.shuffle(indices)
            capped_indices: List[int] = indices[:num_prompts]
            self.all_prompts = [raw_prompts[i] for i in capped_indices]

        logger.info(
            "PromptDataset initialized: %d prompts loaded (seed=%d).",
            len(self.all_prompts),
            seed,
        )

        # Train/test splits are populated by get_train_test_split()
        self.train_prompts: List[str] = []
        self.test_prompts: List[str] = []
        self._split_done: bool = False

    def load(self, path: Optional[str]) -> List[str]:
        """Load prompts from a file, HuggingFace dataset, or synthetic fallback.

        Tries three loading strategies in order:
        1. Local file (if path is a valid file path)
        2. HuggingFace dataset (if path is a non-None string and not a file)
        3. Synthetic fallback (if path is None or all other strategies fail)

        Args:
            path: File path, HuggingFace dataset ID, or None.

        Returns:
            List of prompt strings. May be empty if all strategies fail
            (caller handles this case in __init__).
        """
        # --- Branch 1: Local file ---
        if path is not None and os.path.isfile(path):
            return self._load_from_file(path)

        # --- Branch 2: HuggingFace dataset ---
        if path is not None and not os.path.isfile(path):
            hf_result: Optional[List[str]] = self._load_from_huggingface(path)
            if hf_result is not None:
                return hf_result
            # Fall through to synthetic if HF loading failed

        # --- Branch 3: Synthetic fallback ---
        logger.warning(
            "Using synthetic prompts as fallback "
            "(path=%r is not a valid file or HuggingFace dataset).",
            path,
        )
        return _generate_synthetic_prompts(self.num_prompts, seed=self.seed)

    def _load_from_file(self, path: str) -> List[str]:
        """Load prompts from a plain text file (one prompt per line).

        Args:
            path: Absolute or relative path to the .txt file.

        Returns:
            List of non-empty stripped prompt strings.
        """
        prompts: List[str] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped: str = line.strip()
                    if stripped:  # Filter empty lines
                        prompts.append(stripped)
            logger.info(
                "Loaded %d prompts from file '%s'.", len(prompts), path
            )
        except OSError as exc:
            warnings.warn(
                f"Failed to read prompts file '{path}': {exc}. "
                f"Falling back to synthetic prompts.",
                UserWarning,
                stacklevel=3,
            )
            return []
        return prompts

    def _load_from_huggingface(self, dataset_id: str) -> Optional[List[str]]:
        """Load prompts from a HuggingFace dataset.

        Attempts to import the `datasets` library and load the specified
        dataset. Handles common dataset structures by trying multiple
        text column names.

        Supported dataset IDs (examples):
            - "poloclub/diffusiondb": Uses "2m_random_1k" config
            - "yuvalkirstain/pickapic_v2": Uses "train" split
            - Any dataset with a "prompt", "text", or "caption" column

        Args:
            dataset_id: HuggingFace dataset identifier string.

        Returns:
            List of prompt strings if successful, None if loading fails.
        """
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError:
            warnings.warn(
                "The 'datasets' library is not installed. "
                "Cannot load HuggingFace dataset. "
                "Install with: pip install datasets",
                UserWarning,
                stacklevel=4,
            )
            return None

        try:
            logger.info(
                "Attempting to load HuggingFace dataset '%s'...", dataset_id
            )

            # Special handling for known datasets
            if "diffusiondb" in dataset_id.lower():
                # DiffusionDB has multiple configs; use a small one for testing
                dataset = load_dataset(
                    dataset_id,
                    "2m_random_1k",
                    trust_remote_code=True,
                )
            else:
                # Generic loading: try without specifying config
                dataset = load_dataset(dataset_id, trust_remote_code=True)

            # Get the training split (most datasets have a "train" split)
            if hasattr(dataset, "keys"):
                # DatasetDict: pick "train" or first available split
                split_names: List[str] = list(dataset.keys())
                split_name: str = "train" if "train" in split_names else split_names[0]
                dataset_split = dataset[split_name]
            else:
                # Already a Dataset object
                dataset_split = dataset

            # Extract text column
            prompts: List[str] = _extract_text_column(dataset_split)
            logger.info(
                "Loaded %d prompts from HuggingFace dataset '%s'.",
                len(prompts),
                dataset_id,
            )
            return prompts

        except Exception as exc:  # pylint: disable=broad-except
            warnings.warn(
                f"Failed to load HuggingFace dataset '{dataset_id}': {exc}. "
                f"Falling back to synthetic prompts.",
                UserWarning,
                stacklevel=4,
            )
            return None

    def get_train_test_split(
        self,
        train_size: int = 40000,
        test_size: int = 1000,
    ) -> Tuple[List[str], List[str]]:
        """Split the prompt pool into disjoint train and test sets.

        Uses the seeded RNG to shuffle indices before splitting, ensuring:
        1. Disjointness: train and test sets share no prompts
        2. Reproducibility: same seed always produces the same split
        3. Paper alignment: 40k train / 1k test per run (Appendix G)

        For 3 independent runs (Appendix G), instantiate PromptDataset with
        different seeds (seed=42, 43, 44) to get different splits.

        Populates self.train_prompts and self.test_prompts as side effects.

        Args:
            train_size: Number of training prompts.
                From config.yaml: data.num_train_prompts = 40000.
            test_size: Number of test prompts.
                From config.yaml: data.num_test_prompts = 1000.

        Returns:
            Tuple (train_prompts, test_prompts) where:
                train_prompts: List of train_size prompt strings
                test_prompts: List of test_size prompt strings
            Both lists are disjoint subsets of self.all_prompts.

        Raises:
            ValueError: If train_size + test_size > len(self.all_prompts),
                indicating the prompt pool is too small for the requested split.
            ValueError: If train_size <= 0 or test_size <= 0.
        """
        if train_size <= 0:
            raise ValueError(
                f"train_size must be positive, got {train_size}."
            )
        if test_size <= 0:
            raise ValueError(
                f"test_size must be positive, got {test_size}."
            )

        total_needed: int = train_size + test_size
        available: int = len(self.all_prompts)

        if total_needed > available:
            raise ValueError(
                f"Requested train_size={train_size} + test_size={test_size} "
                f"= {total_needed} prompts, but only {available} prompts are "
                f"available in the pool. "
                f"Either reduce train_size/test_size or increase num_prompts "
                f"when constructing PromptDataset."
            )

        # Create shuffled index list using the split RNG (isolated from batch RNG)
        indices: List[int] = list(range(available))
        self._split_rng.shuffle(indices)

        # Take first train_size as training, next test_size as test
        train_indices: List[int] = indices[:train_size]
        test_indices: List[int] = indices[train_size: train_size + test_size]

        # Build prompt lists from indices
        self.train_prompts = [self.all_prompts[i] for i in train_indices]
        self.test_prompts = [self.all_prompts[i] for i in test_indices]
        self._split_done = True

        logger.info(
            "Train/test split: %d train prompts, %d test prompts (seed=%d).",
            len(self.train_prompts),
            len(self.test_prompts),
            self.seed,
        )

        return self.train_prompts, self.test_prompts

    def get_batch(self, batch_size: int = 40) -> List[str]:
        """Sample a random batch of prompts from the training set.

        Uses sampling with replacement, which is acceptable given the large
        training set size (40k) relative to batch size (40). The batch RNG
        is isolated from the split RNG to prevent interference.

        Must be called after get_train_test_split() has been called.

        Args:
            batch_size: Number of prompts to sample.
                From config.yaml: training.batch_size = 40.

        Returns:
            List of batch_size prompt strings sampled from self.train_prompts
            with replacement.

        Raises:
            RuntimeError: If get_train_test_split() has not been called yet.
            ValueError: If batch_size <= 0.
        """
        if not self._split_done or len(self.train_prompts) == 0:
            raise RuntimeError(
                "Call get_train_test_split() before get_batch(). "
                "self.train_prompts is empty."
            )

        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {batch_size}."
            )

        # Sample with replacement using the isolated batch RNG
        batch: List[str] = self._batch_rng.choices(
            self.train_prompts, k=batch_size
        )
        return batch

    def __len__(self) -> int:
        """Return the total number of prompts in the pool.

        Returns:
            Length of self.all_prompts.
        """
        return len(self.all_prompts)

    def __repr__(self) -> str:
        """Human-readable representation of the dataset state."""
        split_status: str = (
            f"train={len(self.train_prompts)}, test={len(self.test_prompts)}"
            if self._split_done
            else "not split yet"
        )
        return (
            f"PromptDataset("
            f"total={len(self.all_prompts)}, "
            f"seed={self.seed}, "
            f"split=({split_status})"
            f")"
        )
