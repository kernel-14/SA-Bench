## data_collector.py
"""Data collection orchestrator for the adversarial leaderboard manipulation paper.

This module provides the DataCollector class, which sits between the prompt
sources (DatasetLoader) and the de-anonymization/simulation modules. It
orchestrates all API calls to collect model responses, using a disk-based
cache to avoid redundant queries across runs.

Two collection modes are supported:
  1. Training-based detector mode: 50 responses per model per prompt across
     8 categories × 200 prompts (Section 2.3, Appendix A.3).
  2. Identity-probing mode: 1,000 responses per model per identity prompt
     across 5 prompts (Section 2.4.1).

All responses are stored in a ResponseDataset (data_structures.py) and cached
to disk via the Cache utility (utils/cache.py) for resumable collection.

Total estimated data volume:
  - Training-based: 22 models × 8 categories × 200 prompts × 50 responses
    = 1,760,000 response strings.
  - Identity-probing: 22 models × 5 prompts × 1,000 responses
    = 110,000 response strings.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from tqdm import tqdm

from api_client import APIClient
from config import Config, ModelConfig
from data_structures import ResponseDataset
from utils.cache import Cache
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Category label used for identity-probing cache keys.
# Namespaces identity-probing entries separately from training-based entries
# to prevent collisions if any identity prompt text appears in other categories.
# ---------------------------------------------------------------------------
_IDENTITY_PROBING_CATEGORY: str = "identity_probing"

# ---------------------------------------------------------------------------
# Minimum number of responses required in a cache entry to consider it valid.
# If a cached entry has fewer responses than expected (e.g., from an interrupted
# run), the entry is treated as a cache miss and re-queried.
# ---------------------------------------------------------------------------
_MIN_VALID_RESPONSES: int = 1


class DataCollector:
    """Orchestrates LLM API calls to collect model responses with disk caching.

    Provides three public methods:
      - collect_responses: Collect responses for one model across a list of prompts.
      - collect_all_responses: Collect responses for all models across all categories.
      - collect_identity_probing_responses: Collect identity-probing responses for one model.
      - collect_all_identity_responses: Collect identity-probing responses for all models.

    All API calls are routed through the appropriate APIClient based on each
    model's api_provider field. Responses are cached to disk using SHA256-keyed
    entries to enable resumable collection across interrupted runs.

    Attributes:
        config: The global Config object from config.py.
        clients: Dict mapping provider name strings to initialized APIClient
            instances. Must contain keys for all providers used by models in
            config.models: "openai", "anthropic", "google", "together".
        cache: Disk-based Cache instance for persisting API responses.
        n_responses: Number of responses to collect per model per prompt for
            the training-based detector. From config.data_collection.n_responses_per_model (50).
        n_identity_queries: Number of responses to collect per model per
            identity-probing prompt. From config.data_collection.n_identity_queries (1000).
        max_tokens: Maximum output tokens per API call. From
            config.data_collection.max_tokens (512).

    Example:
        >>> import os
        >>> from config import Config
        >>> from api_client import APIClient
        >>> from utils.cache import Cache
        >>> config = Config.from_yaml("config.yaml")
        >>> clients = {
        ...     "openai": APIClient("openai", os.environ["OPENAI_API_KEY"]),
        ...     "anthropic": APIClient("anthropic", os.environ["ANTHROPIC_API_KEY"]),
        ...     "google": APIClient("google", os.environ["GOOGLE_API_KEY"]),
        ...     "together": APIClient("together", os.environ["TOGETHER_API_KEY"]),
        ... }
        >>> cache = Cache(config.cache_dir)
        >>> collector = DataCollector(config, clients, cache)
        >>> prompts_by_category = {"english": ["Hello!", "How are you?"]}
        >>> dataset = collector.collect_all_responses(prompts_by_category)
        >>> len(dataset.get_all_categories())
        1
    """

    def __init__(
        self,
        config: Config,
        clients: Dict[str, APIClient],
        cache: Cache,
    ) -> None:
        """Initialize the DataCollector.

        Stores configuration, API clients, and cache. Reads collection
        parameters from the config object. No API calls are made at init time.

        Args:
            config: The global Config object. Provides model list, collection
                parameters (n_responses_per_model, n_identity_queries, max_tokens),
                and random state.
            clients: Dict mapping provider name strings ("openai", "anthropic",
                "google", "together") to initialized APIClient instances. The
                caller (main.py) is responsible for initializing all four clients
                before constructing DataCollector. A KeyError will be raised
                during collection if a required provider key is missing.
            cache: Initialized Cache instance pointing to config.cache_dir.
                Used for all cache read/write operations during collection.

        Raises:
            KeyError: During collection (not at init time) if a model's
                api_provider is not present in the clients dict.
        """
        self.config: Config = config
        self.clients: Dict[str, APIClient] = clients
        self.cache: Cache = cache

        # Read collection parameters from the raw config dict to match the
        # nested YAML structure (data_collection section).
        data_collection_cfg: Dict = config.raw.get("data_collection", {})
        self.n_responses: int = int(
            data_collection_cfg.get("n_responses_per_model", 50)
        )
        self.n_identity_queries: int = int(
            data_collection_cfg.get("n_identity_queries", 1000)
        )
        self.max_tokens: int = int(
            data_collection_cfg.get("max_tokens", 512)
        )

        logger.info(
            "DataCollector initialized: %d models, n_responses=%d, "
            "n_identity_queries=%d, max_tokens=%d.",
            len(config.models),
            self.n_responses,
            self.n_identity_queries,
            self.max_tokens,
        )

    def collect_responses(
        self,
        model_config: ModelConfig,
        prompts: List[str],
        category: str,
    ) -> Dict[str, List[str]]:
        """Collect responses for a single model across a list of prompts.

        For each prompt, checks the disk cache first. On a cache hit with
        sufficient responses, returns the cached data without an API call.
        On a cache miss (or insufficient cached responses), queries the API
        and stores the result in the cache.

        Args:
            model_config: Configuration for the model to query. Provides the
                model name (for API calls and cache keys) and api_provider
                (for client routing).
            prompts: List of prompt strings to collect responses for. Typically
                200 prompts per category from DatasetLoader.load_all_categories().
            category: Prompt category name, e.g. "english", "math", "safety".
                Used as part of the cache key to namespace entries by category.
                Must match the category names in config.data_collection.prompt_categories.

        Returns:
            Dict mapping each prompt string to a list of response strings.
            Each list contains exactly n_responses_per_model (50) strings under
            normal conditions. If an API call fails after all retries, the
            prompt is mapped to an empty list and an error is logged.

        Raises:
            KeyError: If model_config.api_provider is not in self.clients.

        Example:
            >>> responses = collector.collect_responses(
            ...     model_config=config.models[0],
            ...     prompts=["Hello!", "What is 2+2?"],
            ...     category="english",
            ... )
            >>> len(responses["Hello!"])
            50
        """
        # Route to the correct API client based on the model's provider.
        client: APIClient = self.clients[model_config.api_provider]
        result: Dict[str, List[str]] = {}

        for prompt in prompts:
            cache_key: str = self.cache.make_key(
                model_config.name, prompt, category
            )

            # --- Cache lookup ---
            cached_responses: Optional[List[str]] = None
            if self.cache.exists(cache_key):
                cached_value = self.cache.get(cache_key)
                # Validate that the cached entry has sufficient responses.
                # An entry with fewer than _MIN_VALID_RESPONSES responses
                # indicates an interrupted or failed previous collection run.
                if (
                    cached_value is not None
                    and isinstance(cached_value, list)
                    and len(cached_value) >= _MIN_VALID_RESPONSES
                ):
                    cached_responses = cached_value
                    logger.debug(
                        "Cache HIT: model='%s', category='%s', "
                        "prompt='%.50s...', n_cached=%d.",
                        model_config.name,
                        category,
                        prompt,
                        len(cached_responses),
                    )

            if cached_responses is not None:
                result[prompt] = cached_responses
                continue

            # --- Cache miss: query the API ---
            logger.debug(
                "Cache MISS: querying model='%s', category='%s', "
                "prompt='%.50s...', n_samples=%d.",
                model_config.name,
                category,
                prompt,
                self.n_responses,
            )

            try:
                responses: List[str] = client.query(
                    model_name=model_config.name,
                    prompt=prompt,
                    n_samples=self.n_responses,
                    max_tokens=self.max_tokens,
                )
                # Store in cache for future runs.
                self.cache.set(cache_key, responses)
                result[prompt] = responses
                logger.debug(
                    "Collected %d responses: model='%s', category='%s', "
                    "prompt='%.50s...'.",
                    len(responses),
                    model_config.name,
                    category,
                    prompt,
                )
            except Exception as exc:  # pylint: disable=broad-except
                # Log the error and store an empty list so the caller can
                # detect the failure without crashing the entire collection run.
                logger.error(
                    "API call failed for model='%s', category='%s', "
                    "prompt='%.50s...': %s. Storing empty list.",
                    model_config.name,
                    category,
                    prompt,
                    exc,
                    exc_info=True,
                )
                result[prompt] = []

        return result

    def collect_all_responses(
        self,
        prompts_by_category: Dict[str, List[str]],
    ) -> ResponseDataset:
        """Collect responses for all models across all prompt categories.

        Iterates over all categories, all models within each category, and
        all prompts within each (category, model) pair. Assembles all collected
        responses into a ResponseDataset.

        This is the primary data collection method for the training-based
        detector experiments (Section 2.3). The returned ResponseDataset
        should be saved to disk by the caller (main.py) via dataset.save().

        Args:
            prompts_by_category: Dict mapping category name strings to lists
                of prompt strings. Produced by DatasetLoader.load_all_categories().
                Keys must match config.data_collection.prompt_categories[*].name.

        Returns:
            A ResponseDataset containing all collected responses, structured as
            data[category][prompt][model_name] = List[str]. Categories or
            prompts where collection failed have empty response lists.

        Example:
            >>> prompts = {"english": ["Hello!", "How are you?"], "math": ["2+2=?"]}
            >>> dataset = collector.collect_all_responses(prompts)
            >>> responses = dataset.get_responses("english", "Hello!", "gpt-4o-2024-05-13")
            >>> len(responses)
            50
        """
        dataset: ResponseDataset = ResponseDataset()

        # Compute total work units for the outer progress bar.
        # Each work unit = one (model, category) pair.
        total_pairs: int = len(prompts_by_category) * len(self.config.models)
        logger.info(
            "Starting full response collection: %d categories × %d models = "
            "%d (model, category) pairs.",
            len(prompts_by_category),
            len(self.config.models),
            total_pairs,
        )

        # Outer loop: iterate over categories.
        for category, prompts in prompts_by_category.items():
            n_prompts: int = len(prompts)
            logger.info(
                "Collecting responses for category='%s' (%d prompts).",
                category,
                n_prompts,
            )

            if n_prompts == 0:
                logger.warning(
                    "Category '%s' has no prompts. Skipping.", category
                )
                continue

            # Inner loop: iterate over all 22 models with a tqdm progress bar.
            # leave=False keeps the terminal clean by removing the bar when done.
            model_bar = tqdm(
                self.config.models,
                desc=f"Category: {category}",
                unit="model",
                leave=True,
            )
            for model_config in model_bar:
                model_bar.set_postfix({"model": model_config.name[:30]})
                logger.info(
                    "  Collecting: model='%s', category='%s', n_prompts=%d.",
                    model_config.name,
                    category,
                    n_prompts,
                )

                # Collect responses for this (model, category) pair.
                responses_by_prompt: Dict[str, List[str]] = self.collect_responses(
                    model_config=model_config,
                    prompts=prompts,
                    category=category,
                )

                # Add all collected responses to the dataset.
                for prompt, responses in responses_by_prompt.items():
                    dataset.add_responses(
                        category=category,
                        prompt=prompt,
                        model_name=model_config.name,
                        responses=responses,
                    )

                logger.info(
                    "  Completed: model='%s', category='%s', "
                    "%d/%d prompts with responses.",
                    model_config.name,
                    category,
                    sum(1 for r in responses_by_prompt.values() if r),
                    n_prompts,
                )

        logger.info(
            "Full response collection complete. "
            "ResponseDataset contains %d (category, prompt, model) entries.",
            len(dataset),
        )
        return dataset

    def collect_identity_probing_responses(
        self,
        model_config: ModelConfig,
    ) -> Dict[str, List[str]]:
        """Collect identity-probing responses for a single model.

        Queries the model with each of the 5 identity-probing prompts defined
        in deanonymization/identity_probing.py, collecting n_identity_queries
        (1,000) responses per prompt. Uses the cache to avoid redundant queries.

        The identity-probing prompts are imported from the identity_probing
        module at call time (not at import time) to avoid circular imports,
        since identity_probing.py may import from this module in some
        configurations.

        Args:
            model_config: Configuration for the model to query. Provides the
                model name and api_provider for client routing.

        Returns:
            Dict mapping each identity-probing prompt string to a list of
            n_identity_queries (1,000) response strings. If an API call fails,
            the prompt is mapped to an empty list.

        Example:
            >>> responses = collector.collect_identity_probing_responses(
            ...     model_config=config.models[0]
            ... )
            >>> len(responses)  # 5 identity prompts
            5
            >>> len(responses["Who are you?"])  # 1000 responses per prompt
            1000
        """
        # Import IDENTITY_PROMPTS here (not at module level) to avoid circular
        # imports. deanonymization/identity_probing.py imports from config.py
        # but not from data_collector.py, so this is safe.
        from deanonymization.identity_probing import IDENTITY_PROMPTS  # pylint: disable=import-outside-toplevel

        client: APIClient = self.clients[model_config.api_provider]
        result: Dict[str, List[str]] = {}

        logger.info(
            "Collecting identity-probing responses for model='%s' "
            "(%d prompts × %d queries each).",
            model_config.name,
            len(IDENTITY_PROMPTS),
            self.n_identity_queries,
        )

        for prompt in IDENTITY_PROMPTS:
            # Use a dedicated category label to namespace identity-probing
            # cache entries separately from training-based detector entries.
            cache_key: str = self.cache.make_key(
                model_config.name, prompt, _IDENTITY_PROBING_CATEGORY
            )

            # --- Cache lookup ---
            cached_responses: Optional[List[str]] = None
            if self.cache.exists(cache_key):
                cached_value = self.cache.get(cache_key)
                if (
                    cached_value is not None
                    and isinstance(cached_value, list)
                    and len(cached_value) >= _MIN_VALID_RESPONSES
                ):
                    cached_responses = cached_value
                    logger.debug(
                        "Cache HIT (identity): model='%s', prompt='%s', "
                        "n_cached=%d.",
                        model_config.name,
                        prompt,
                        len(cached_responses),
                    )

            if cached_responses is not None:
                result[prompt] = cached_responses
                continue

            # --- Cache miss: query the API ---
            logger.debug(
                "Cache MISS (identity): querying model='%s', prompt='%s', "
                "n_samples=%d.",
                model_config.name,
                prompt,
                self.n_identity_queries,
            )

            try:
                responses: List[str] = client.query(
                    model_name=model_config.name,
                    prompt=prompt,
                    n_samples=self.n_identity_queries,
                    max_tokens=self.max_tokens,
                )
                self.cache.set(cache_key, responses)
                result[prompt] = responses
                logger.debug(
                    "Collected %d identity-probing responses: "
                    "model='%s', prompt='%s'.",
                    len(responses),
                    model_config.name,
                    prompt,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "API call failed (identity probing): model='%s', "
                    "prompt='%s': %s. Storing empty list.",
                    model_config.name,
                    prompt,
                    exc,
                    exc_info=True,
                )
                result[prompt] = []

        return result

    def collect_all_identity_responses(
        self,
    ) -> Dict[str, Dict[str, List[str]]]:
        """Collect identity-probing responses for all 22 models.

        Calls collect_identity_probing_responses for each model in
        config.models and assembles the results into a nested dict.

        Returns:
            Nested dict with structure:
                model_name -> prompt_string -> List[str] of responses.
            The outer key is the exact model name string (e.g.,
            "claude-3-5-sonnet-20240620"), the inner key is one of the 5
            identity-probing prompt strings, and the value is a list of
            n_identity_queries (1,000) response strings.

            This structure is consumed directly by
            IdentityProbingDetector.evaluate_all().

        Example:
            >>> all_responses = collector.collect_all_identity_responses()
            >>> len(all_responses)  # 22 models
            22
            >>> len(all_responses["claude-3-5-sonnet-20240620"])  # 5 prompts
            5
            >>> len(all_responses["claude-3-5-sonnet-20240620"]["Who are you?"])
            1000
        """
        all_responses: Dict[str, Dict[str, List[str]]] = {}

        logger.info(
            "Starting identity-probing collection for %d models.",
            len(self.config.models),
        )

        # Wrap model iteration with a tqdm progress bar.
        model_bar = tqdm(
            self.config.models,
            desc="Identity probing",
            unit="model",
            leave=True,
        )
        for model_config in model_bar:
            model_bar.set_postfix({"model": model_config.name[:30]})
            logger.info(
                "Collecting identity-probing responses for model='%s'.",
                model_config.name,
            )

            responses: Dict[str, List[str]] = (
                self.collect_identity_probing_responses(model_config)
            )
            all_responses[model_config.name] = responses

            # Log a summary of how many responses were collected per prompt.
            for prompt, resp_list in responses.items():
                logger.info(
                    "  model='%s', prompt='%s': %d responses collected.",
                    model_config.name,
                    prompt,
                    len(resp_list),
                )

        logger.info(
            "Identity-probing collection complete for %d models.",
            len(all_responses),
        )
        return all_responses
