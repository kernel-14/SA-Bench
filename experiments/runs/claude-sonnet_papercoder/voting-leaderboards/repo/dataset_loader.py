## dataset_loader.py
"""Dataset loader for prompt collection used in the training-based detector.

This module provides the DatasetLoader class, which loads and samples prompts
from four source datasets used in Section 2.3 of the paper:
  - LMSYS-Chat-1M: Normal chat prompts in English, Chinese, Spanish,
    Indonesian, and Persian.
  - MATH (Hendrycks et al., 2021): Math problem prompts.
  - AdvBench (Zou et al., 2023): Safety-violating instruction prompts.
  - HumanEval (OpenAI): Coding task prompts.

The output of load_all_categories() feeds directly into DataCollector, which
uses the prompts to query all 22 LLMs and collect 50 responses per model per
prompt (Section 2.3: "200 prompts per category", "50 responses per model").

All sampling is seeded via numpy.random.default_rng(config.random_state) for
full reproducibility. This module has no internal project dependencies beyond
config.py and utils/logger.py.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# AdvBench remote URL — used when the local CSV is not present.
# Source: https://github.com/llm-attacks/llm-attacks
# ---------------------------------------------------------------------------
_ADVBENCH_REMOTE_URL: str = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)

# Local path where AdvBench CSV is stored (relative to project root).
_ADVBENCH_LOCAL_PATH: str = os.path.join("data", "advbench", "harmful_behaviors.csv")

# Minimum prompt length (characters) after stripping whitespace.
# Prompts shorter than this are excluded before sampling.
_MIN_PROMPT_LENGTH: int = 10

# Mapping from config source names to loader dispatch keys.
_SOURCE_DISPATCH: Dict[str, str] = {
    "lmsys_chat_1m": "lmsys",
    "math": "math",
    "advbench": "advbench",
    "humaneval": "humaneval",
}


class DatasetLoader:
    """Loads and samples prompts from four source datasets for the paper's experiments.

    Provides one public method per dataset source and a unified
    load_all_categories() method that dispatches based on the source field in
    config.data_collection.prompt_categories.

    All sampling uses a seeded numpy RNG for reproducibility. Dataset loading
    is lazy — no data is fetched until a load_* method is called.

    Attributes:
        config: The global Config object from config.py.
        rng: A seeded numpy.random.Generator used for all sampling operations.

    Example:
        >>> from config import Config
        >>> config = Config.from_yaml("config.yaml")
        >>> loader = DatasetLoader(config)
        >>> prompts = loader.load_all_categories()
        >>> len(prompts["english"])
        200
    """

    def __init__(self, config: Config) -> None:
        """Initialize the DatasetLoader with a Config instance.

        Stores the config and initializes a seeded numpy RNG. No datasets are
        loaded at init time — all loading is deferred to method calls.

        Args:
            config: The global Config object. Uses config.random_state (42) for
                seeding the RNG and config.n_prompts_per_category (200) as the
                default sample size in load_all_categories().
        """
        self.config: Config = config
        # Seed from config.reproducibility.random_state (42) for reproducibility.
        self.rng: np.random.Generator = np.random.default_rng(config.random_state)
        logger.info(
            "DatasetLoader initialized with random_state=%d, "
            "n_prompts_per_category=%d.",
            config.random_state,
            config.n_prompts_per_category,
        )

    # -----------------------------------------------------------------------
    # Public loader methods
    # -----------------------------------------------------------------------

    def load_lmsys_chat(self, language: str, n_samples: int) -> List[str]:
        """Load and sample prompts from LMSYS-Chat-1M for a given language.

        Filters the dataset by the built-in "language" column (ISO 639-1 code)
        as the primary strategy. Falls back to langdetect on the first user turn
        if the language column is absent or yields insufficient samples.

        Extracts the first user turn from each conversation. Applies a minimum
        length filter (>= 10 characters) before sampling.

        Args:
            language: ISO 639-1 language code, e.g. "en", "zh", "es", "id", "fa".
                Must match the values in the LMSYS-Chat-1M "language" column.
            n_samples: Number of prompts to return. If fewer than n_samples
                prompts are available after filtering, returns all available
                prompts and logs a warning.

        Returns:
            List of up to n_samples prompt strings (first user turns) in the
            requested language. May be shorter than n_samples if the filtered
            pool is insufficient.

        Raises:
            Exception: If the HuggingFace dataset cannot be loaded (network
                error, authentication required, etc.). Caller (load_all_categories)
                catches this and logs the error.

        Example:
            >>> loader = DatasetLoader(config)
            >>> prompts = loader.load_lmsys_chat("en", 200)
            >>> len(prompts) <= 200
            True
        """
        # Import here to avoid making datasets a hard import-time dependency
        # when only other loaders are used.
        import datasets as hf_datasets  # type: ignore[import]

        logger.info(
            "Loading LMSYS-Chat-1M for language='%s', n_samples=%d.",
            language,
            n_samples,
        )

        # Load the full training split. This is a large dataset (~1M rows);
        # HuggingFace caches it locally after the first download.
        dataset = hf_datasets.load_dataset(
            "lmsys/lmsys-chat-1m",
            split="train",
            trust_remote_code=True,
        )

        # --- Primary filter: use the built-in "language" column ---
        filtered_prompts: List[str] = self._filter_lmsys_by_language_column(
            dataset, language
        )

        # --- Fallback: langdetect on first user turn ---
        if len(filtered_prompts) < n_samples:
            logger.warning(
                "LMSYS language column filter yielded only %d prompts for "
                "language='%s' (need %d). Attempting langdetect fallback.",
                len(filtered_prompts),
                language,
                n_samples,
            )
            fallback_prompts: List[str] = self._filter_lmsys_by_langdetect(
                dataset, language, n_samples * 10  # scan a larger subset
            )
            # Merge and deduplicate, preserving order.
            seen: set = set(filtered_prompts)
            for p in fallback_prompts:
                if p not in seen:
                    filtered_prompts.append(p)
                    seen.add(p)
            logger.info(
                "After langdetect fallback: %d prompts available for language='%s'.",
                len(filtered_prompts),
                language,
            )

        # Apply minimum length filter.
        filtered_prompts = [
            p for p in filtered_prompts if len(p.strip()) >= _MIN_PROMPT_LENGTH
        ]

        # Sample without replacement.
        return self._sample_prompts(filtered_prompts, n_samples, f"lmsys/{language}")

    def load_math(self, n_samples: int) -> List[str]:
        """Load and sample math problem prompts from the MATH dataset.

        Uses the "hendrycks/competition_math" HuggingFace dataset. Extracts
        the "problem" field from each row.

        Args:
            n_samples: Number of prompts to return. The MATH training split
                has ~7,500 problems, which is more than sufficient for
                n_samples=200.

        Returns:
            List of up to n_samples math problem strings.

        Raises:
            Exception: If the HuggingFace dataset cannot be loaded.

        Example:
            >>> loader = DatasetLoader(config)
            >>> prompts = loader.load_math(200)
            >>> len(prompts) <= 200
            True
        """
        import datasets as hf_datasets  # type: ignore[import]

        logger.info("Loading MATH dataset, n_samples=%d.", n_samples)

        dataset = hf_datasets.load_dataset(
            "hendrycks/competition_math",
            split="train",
            trust_remote_code=True,
        )

        # Extract "problem" field from each row.
        problems: List[str] = []
        for row in dataset:
            problem: str = str(row.get("problem", "")).strip()
            if len(problem) >= _MIN_PROMPT_LENGTH:
                problems.append(problem)

        logger.info(
            "MATH dataset: %d valid problems available (need %d).",
            len(problems),
            n_samples,
        )

        return self._sample_prompts(problems, n_samples, "math")

    def load_advbench(self, n_samples: int) -> List[str]:
        """Load and sample safety-violating prompts from AdvBench.

        Tries to load from a local CSV file first. If not found, downloads
        from the llm-attacks GitHub repository and caches locally.

        The "goal" column contains the harmful instruction strings used as
        prompts. Models will typically refuse these, and the refusal responses
        themselves serve as the distinguishing signal between models.

        Args:
            n_samples: Number of prompts to return. AdvBench has 520 behaviors,
                which is more than sufficient for n_samples=200.

        Returns:
            List of up to n_samples harmful instruction strings.

        Raises:
            Exception: If the local file is absent and the remote download fails.

        Example:
            >>> loader = DatasetLoader(config)
            >>> prompts = loader.load_advbench(200)
            >>> len(prompts) <= 200
            True
        """
        logger.info("Loading AdvBench dataset, n_samples=%d.", n_samples)

        # Ensure the local CSV is available (download if needed).
        local_path: str = self._ensure_advbench_local()

        # Load with pandas.
        df: pd.DataFrame = pd.read_csv(local_path)

        # The CSV has a "goal" column with the harmful instruction strings.
        if "goal" not in df.columns:
            # Fallback: try the first column if "goal" is absent.
            logger.warning(
                "AdvBench CSV does not have a 'goal' column. "
                "Available columns: %s. Using first column.",
                list(df.columns),
            )
            goals: List[str] = df.iloc[:, 0].astype(str).tolist()
        else:
            goals = df["goal"].astype(str).tolist()

        # Apply minimum length filter.
        goals = [g.strip() for g in goals if len(g.strip()) >= _MIN_PROMPT_LENGTH]

        logger.info(
            "AdvBench: %d valid goals available (need %d).",
            len(goals),
            n_samples,
        )

        return self._sample_prompts(goals, n_samples, "advbench")

    def load_humaneval(self, n_samples: int) -> List[str]:
        """Load and sample coding prompts from the HumanEval dataset.

        Uses the "openai_humaneval" HuggingFace dataset. Extracts the "prompt"
        field (Python function signature + docstring) from each row.

        HumanEval has only 164 problems. If n_samples > 164, returns all 164
        prompts and logs a warning. The downstream classifier training will
        still work with fewer prompts.

        Args:
            n_samples: Number of prompts to return. If n_samples > 164 (the
                full HumanEval test split size), all 164 prompts are returned.

        Returns:
            List of up to n_samples coding prompt strings.

        Raises:
            Exception: If the HuggingFace dataset cannot be loaded.

        Example:
            >>> loader = DatasetLoader(config)
            >>> prompts = loader.load_humaneval(200)
            >>> len(prompts) <= 164  # HumanEval has only 164 problems
            True
        """
        import datasets as hf_datasets  # type: ignore[import]

        logger.info("Loading HumanEval dataset, n_samples=%d.", n_samples)

        dataset = hf_datasets.load_dataset(
            "openai_humaneval",
            split="test",
            trust_remote_code=True,
        )

        # Extract "prompt" field from each row.
        prompts: List[str] = []
        for row in dataset:
            prompt: str = str(row.get("prompt", "")).strip()
            if len(prompt) >= _MIN_PROMPT_LENGTH:
                prompts.append(prompt)

        if len(prompts) < n_samples:
            logger.warning(
                "HumanEval has only %d problems but %d were requested. "
                "Returning all %d available prompts.",
                len(prompts),
                n_samples,
                len(prompts),
            )

        logger.info(
            "HumanEval: %d valid prompts available (need %d).",
            len(prompts),
            n_samples,
        )

        return self._sample_prompts(prompts, n_samples, "humaneval")

    def load_all_categories(self) -> Dict[str, List[str]]:
        """Load and sample prompts for all configured prompt categories.

        Iterates over config.data_collection.prompt_categories and dispatches
        to the appropriate loader based on the "source" field. Returns a dict
        mapping category name -> list of prompt strings.

        Each category loader call is wrapped in a try/except so that a failure
        in one category (e.g., network error for a specific dataset) does not
        crash the entire pipeline. Failed categories are stored as empty lists
        and logged as errors.

        Returns:
            Dict mapping category name (e.g., "english", "math") to a list of
            up to n_prompts_per_category prompt strings. Categories that failed
            to load have empty lists.

        Example:
            >>> loader = DatasetLoader(config)
            >>> prompts = loader.load_all_categories()
            >>> set(prompts.keys()) == {"english", "chinese", "spanish",
            ...                          "indonesian", "persian", "coding",
            ...                          "math", "safety"}
            True
        """
        result: Dict[str, List[str]] = {}
        n_prompts: int = self.config.n_prompts_per_category

        # Read prompt_categories from the raw config dict since Config only
        # exposes the category names, not the full source/language metadata.
        raw_categories: List[Dict[str, Any]] = (
            self.config.raw
            .get("data_collection", {})
            .get("prompt_categories", [])
        )

        for cat_entry in raw_categories:
            category_name: str = str(cat_entry.get("name", ""))
            source: str = str(cat_entry.get("source", ""))
            language: str = str(cat_entry.get("language", "en"))

            if not category_name:
                logger.warning("Skipping prompt category entry with empty name.")
                continue

            logger.info(
                "Loading category '%s' (source='%s', language='%s', n=%d).",
                category_name,
                source,
                language,
                n_prompts,
            )

            try:
                prompts: List[str] = self._dispatch_loader(
                    source=source,
                    language=language,
                    n_samples=n_prompts,
                    category_name=category_name,
                )
                result[category_name] = prompts
                logger.info(
                    "Loaded %d prompts for category '%s' (source: %s).",
                    len(prompts),
                    category_name,
                    source,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Failed to load category '%s' (source='%s'): %s. "
                    "Storing empty list and continuing.",
                    category_name,
                    source,
                    exc,
                    exc_info=True,
                )
                result[category_name] = []

        return result

    # -----------------------------------------------------------------------
    # Private helper methods
    # -----------------------------------------------------------------------

    def _dispatch_loader(
        self,
        source: str,
        language: str,
        n_samples: int,
        category_name: str,
    ) -> List[str]:
        """Dispatch to the appropriate loader based on the source field.

        Args:
            source: Dataset source identifier from config, one of
                "lmsys_chat_1m", "math", "advbench", "humaneval".
            language: ISO 639-1 language code (used only for lmsys_chat_1m).
            n_samples: Number of prompts to load.
            category_name: Human-readable category name for logging.

        Returns:
            List of prompt strings from the appropriate dataset.

        Raises:
            ValueError: If source is not a recognized value.
        """
        if source == "lmsys_chat_1m":
            return self.load_lmsys_chat(language=language, n_samples=n_samples)
        elif source == "math":
            return self.load_math(n_samples=n_samples)
        elif source == "advbench":
            return self.load_advbench(n_samples=n_samples)
        elif source == "humaneval":
            return self.load_humaneval(n_samples=n_samples)
        else:
            raise ValueError(
                f"Unknown dataset source '{source}' for category '{category_name}'. "
                f"Must be one of: {sorted(_SOURCE_DISPATCH.keys())}."
            )

    def _filter_lmsys_by_language_column(
        self,
        dataset: Any,
        language: str,
    ) -> List[str]:
        """Filter LMSYS-Chat-1M by the built-in 'language' column.

        Iterates the dataset and collects first user turns where the row's
        "language" field matches the requested language code.

        Args:
            dataset: A HuggingFace Dataset object for lmsys/lmsys-chat-1m.
            language: ISO 639-1 language code to filter by.

        Returns:
            List of first-user-turn strings matching the language filter.
            May be empty if the language column is absent or has no matches.
        """
        prompts: List[str] = []

        # Check if the "language" column exists in the dataset features.
        if "language" not in dataset.features:
            logger.warning(
                "LMSYS-Chat-1M dataset does not have a 'language' column. "
                "Language column filter will return empty list."
            )
            return prompts

        for row in dataset:
            row_language: str = str(row.get("language", "")).strip().lower()
            if row_language != language.lower():
                continue

            first_turn: Optional[str] = self._extract_first_user_turn(
                row.get("conversation", [])
            )
            if first_turn and len(first_turn.strip()) >= _MIN_PROMPT_LENGTH:
                prompts.append(first_turn.strip())

        logger.debug(
            "LMSYS language column filter: found %d prompts for language='%s'.",
            len(prompts),
            language,
        )
        return prompts

    def _filter_lmsys_by_langdetect(
        self,
        dataset: Any,
        language: str,
        max_scan: int,
    ) -> List[str]:
        """Filter LMSYS-Chat-1M using langdetect on the first user turn.

        Used as a fallback when the built-in language column is absent or
        yields insufficient samples. Scans up to max_scan rows to limit
        runtime cost.

        Args:
            dataset: A HuggingFace Dataset object for lmsys/lmsys-chat-1m.
            language: ISO 639-1 language code to detect.
            max_scan: Maximum number of rows to scan before stopping.

        Returns:
            List of first-user-turn strings detected as the requested language.
        """
        try:
            from langdetect import detect, LangDetectException  # type: ignore[import]
        except ImportError:
            logger.warning(
                "langdetect is not installed. Skipping langdetect fallback."
            )
            return []

        prompts: List[str] = []
        scanned: int = 0

        for row in dataset:
            if scanned >= max_scan:
                break
            scanned += 1

            first_turn: Optional[str] = self._extract_first_user_turn(
                row.get("conversation", [])
            )
            if not first_turn or len(first_turn.strip()) < _MIN_PROMPT_LENGTH:
                continue

            try:
                detected_lang: str = detect(first_turn)
                if detected_lang == language:
                    prompts.append(first_turn.strip())
            except LangDetectException:
                # langdetect raises LangDetectException for very short or
                # ambiguous texts. Skip silently.
                continue

        logger.debug(
            "langdetect fallback: found %d prompts for language='%s' "
            "after scanning %d rows.",
            len(prompts),
            language,
            scanned,
        )
        return prompts

    def _extract_first_user_turn(
        self,
        conversation: List[Dict[str, str]],
    ) -> Optional[str]:
        """Extract the content of the first user turn from a conversation list.

        Each conversation in LMSYS-Chat-1M is a list of dicts with "role" and
        "content" keys. This method finds the first dict where role == "user"
        and returns its "content" string.

        Args:
            conversation: List of turn dicts, each with "role" and "content".
                May be empty or contain only assistant turns.

        Returns:
            The content string of the first user turn, or None if no user turn
            is found.

        Example:
            >>> turns = [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi!"}]
            >>> loader._extract_first_user_turn(turns)
            'Hello!'
        """
        if not conversation:
            return None

        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            role: str = str(turn.get("role", "")).strip().lower()
            content: str = str(turn.get("content", "")).strip()
            if role == "user" and content:
                return content

        return None

    def _ensure_advbench_local(self) -> str:
        """Ensure the AdvBench CSV is available locally, downloading if needed.

        Checks for the file at _ADVBENCH_LOCAL_PATH. If absent, downloads from
        _ADVBENCH_REMOTE_URL and saves to the local path. Creates parent
        directories as needed.

        Returns:
            The local file path string where the CSV is available.

        Raises:
            urllib.error.URLError: If the remote download fails (network error,
                URL not found, etc.).
            OSError: If the local directory cannot be created or the file
                cannot be written.
        """
        if os.path.exists(_ADVBENCH_LOCAL_PATH):
            logger.debug(
                "AdvBench CSV found at local path: %s", _ADVBENCH_LOCAL_PATH
            )
            return _ADVBENCH_LOCAL_PATH

        # Create parent directory if it does not exist.
        parent_dir: str = os.path.dirname(_ADVBENCH_LOCAL_PATH)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        logger.info(
            "AdvBench CSV not found locally. Downloading from %s to %s.",
            _ADVBENCH_REMOTE_URL,
            _ADVBENCH_LOCAL_PATH,
        )

        urllib.request.urlretrieve(_ADVBENCH_REMOTE_URL, _ADVBENCH_LOCAL_PATH)

        logger.info(
            "AdvBench CSV downloaded successfully to %s.", _ADVBENCH_LOCAL_PATH
        )
        return _ADVBENCH_LOCAL_PATH

    def _sample_prompts(
        self,
        prompts: List[str],
        n_samples: int,
        source_label: str,
    ) -> List[str]:
        """Sample up to n_samples prompts without replacement using self.rng.

        If len(prompts) <= n_samples, returns all prompts (no sampling needed).
        Logs a warning if fewer than n_samples prompts are available.

        Args:
            prompts: Pool of candidate prompt strings to sample from.
            n_samples: Desired number of prompts to return.
            source_label: Human-readable label for logging (e.g., "lmsys/en").

        Returns:
            List of up to n_samples prompt strings sampled without replacement.
            Order is determined by the RNG for reproducibility.
        """
        n_available: int = len(prompts)

        if n_available == 0:
            logger.warning(
                "No prompts available for source '%s'. Returning empty list.",
                source_label,
            )
            return []

        if n_available < n_samples:
            logger.warning(
                "Source '%s': only %d prompts available, but %d were requested. "
                "Returning all %d available prompts.",
                source_label,
                n_available,
                n_samples,
                n_available,
            )
            # Return all available prompts in a shuffled order for consistency.
            indices: np.ndarray = self.rng.permutation(n_available)
            return [prompts[int(i)] for i in indices]

        # Sample n_samples indices without replacement.
        sampled_indices: np.ndarray = self.rng.choice(
            n_available, size=n_samples, replace=False
        )
        sampled: List[str] = [prompts[int(i)] for i in sampled_indices]

        logger.debug(
            "Sampled %d/%d prompts from source '%s'.",
            len(sampled),
            n_available,
            source_label,
        )
        return sampled
