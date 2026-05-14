"""
responses.py – Response collection for adversarial manipulation reproduction.

This module implements the :class:`ResponseCollector`, which orchestrates the sampling
of prompts from each category (via :class:`PromptLoader`) and the querying of all
target models (via :class:`ModelAPI`) to obtain multiple responses per prompt.
Collected responses are cached on disk (JSON per category) to avoid expensive re‑queries.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# -- third-party progress bar (already listed in required packages)
from tqdm import tqdm

# -- project modules (no circular import because we only use Config type)
from config import Config
from prompts import PromptLoader
from api import ModelAPI
from utils import load_json, save_json

logger = logging.getLogger(__name__)


class ResponseCollector:
    """
    Collects model responses for all prompt‑category pairs and persistently caches them.

    The class is designed to be used by the training‑based detector, which needs
    balanced datasets of responses for each (prompt, model) pairing.  Caching ensures
    that expensive API calls are only made once.

    Attributes:
        config:          The global configuration (loaded from ``config.yaml``).
        api:             Unified LLM query interface.
        loader:          Prompt loader that provides reproducible samples.
        cache_dir:       Directory where per‑category JSON files are stored.
        categories_cache: In‑memory mirror of the per‑category caches, keyed by
                          category name.  Each value is a dict of the form
                          ``{prompt: {model: [resp1, resp2, ...]}}``.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Config,
        api: ModelAPI,
        loader: PromptLoader,
    ) -> None:
        """
        Initialise the response collector.

        Args:
            config: Application configuration.
            api:    Authenticated API wrapper for all models.
            loader: Prompt loader that provides sampling from local datasets.
        """
        if not isinstance(config, Config):
            raise TypeError("config must be an instance of Config")
        if not isinstance(api, ModelAPI):
            raise TypeError("api must be an instance of ModelAPI")
        if not isinstance(loader, PromptLoader):
            raise TypeError("loader must be an instance of PromptLoader")

        self.config: Config = config
        self.api: ModelAPI = api
        self.loader: PromptLoader = loader
        self.cache_dir: str = config.cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.categories_cache: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    # ------------------------------------------------------------------
    def collect_for_prompt(
        self,
        prompt: str,
        models: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Query every model *num_responses_per_prompt* times for a single prompt.

        Args:
            prompt: The text prompt to send to each model.
            models: List of model identifiers to query.  If ``None``, uses
                    ``config.models``.

        Returns:
            A dictionary mapping model name to a list of its generated responses.

        Raises:
            RuntimeError: If any model fails after the API retries.
        """
        if models is None:
            models = self.config.models

        num_responses = self.config.num_responses_per_prompt
        responses: Dict[str, List[str]] = {}

        for model in models:
            logger.debug("Collecting %d responses for model '%s' on prompt (len=%d)",
                         num_responses, model, len(prompt))
            model_responses: List[str] = []
            for i in range(num_responses):
                try:
                    resp = self.api.query(
                        model=model,
                        prompt=prompt,
                        max_tokens=self.config.max_output_tokens,
                        temperature=self.config.temperature,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to query model '%s' on iteration %d/%d: %s",
                        model, i + 1, num_responses, exc,
                    )
                    raise RuntimeError(
                        f"Could not collect response {i + 1}/{num_responses} "
                        f"for model '{model}'" 
                    ) from exc
                model_responses.append(resp)
                # Pacing to avoid hitting API rate limits
                time.sleep(self.config.api.get("request_interval", 0.5))

            responses[model] = model_responses

        return responses

    # ------------------------------------------------------------------
    def collect_all(self) -> None:
        """
        Iterate over all prompt categories, sample prompts, collect responses, and
        persist each category to its own JSON cache file.

        Missing or incomplete prompts are re‑queried; already complete prompts are
        skipped to enable resumption of interrupted collection.
        """
        for category in self.config.prompt_categories:
            cache_path = os.path.join(self.cache_dir, f"{category}.json")
            # Load existing cache if present
            if os.path.isfile(cache_path):
                category_cache: Dict[str, Dict[str, List[str]]] = load_json(cache_path)
                logger.info("Loaded existing cache for category '%s' (%d prompts)",
                            category, len(category_cache))
            else:
                category_cache = {}

            # Sample prompts for this category (reproducible)
            prompts = self.loader.sample_prompts(
                category=category,
                n=self.config.num_prompts_per_category,
            )
            logger.info("Processing category '%s' with %d prompts",
                        category, len(prompts))

            for prompt in tqdm(prompts, desc=f"Collecting {category}", unit="prompt"):
                # Check if this prompt is already fully collected
                if prompt in category_cache:
                    cached_data = category_cache[prompt]
                    complete = all(
                        (model in cached_data and
                         len(cached_data[model]) == self.config.num_responses_per_prompt)
                        for model in self.config.models
                    )
                else:
                    complete = False

                if complete:
                    logger.debug("Prompt already complete, skipping.")
                    continue

                # Collect fresh responses for this prompt
                fresh_data = self.collect_for_prompt(prompt, self.config.models)
                category_cache[prompt] = fresh_data

                # Periodically save to disk to avoid losing progress
                # (save after every prompt – acceptable for ~200 prompts)
                save_json(category_cache, cache_path)

            # Final save after the loop (already saved incrementally)
            save_json(category_cache, cache_path)
            # Keep the latest cache in memory for quick access
            self.categories_cache[category] = category_cache
            logger.info("Category '%s' collection complete (%d prompts cached).",
                        category, len(category_cache))

    # ------------------------------------------------------------------
    def get_data(self, category: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Return the full response cache for *category*, loading it from disk if not
        already in memory.

        Args:
            category: Name of the prompt category (e.g. ``"english"``, ``"coding"``).

        Returns:
            Dictionary ``{prompt: {model: [response, ...]}}``.

        Raises:
            FileNotFoundError: If the cache file does not exist (collection has not
                been run yet).
        """
        if category not in self.categories_cache:
            cache_path = os.path.join(self.cache_dir, f"{category}.json")
            if not os.path.isfile(cache_path):
                raise FileNotFoundError(
                    f"No cache found for category '{category}'. "
                    f"Run collect_all() first."
                )
            self.categories_cache[category] = load_json(cache_path)
        return self.categories_cache[category]

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"<ResponseCollector: models={len(self.config.models)}, "
                f"categories={len(self.config.prompt_categories)}>")

