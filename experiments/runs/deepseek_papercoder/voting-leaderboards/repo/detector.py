"""
detector.py – Implementation of the de‑anonymisation detectors for the adversarial
              manipulation reproduction.

Contains two classes:
  - IdentityProbingDetector : queries identity‑probing prompts and performs keyword
    matching to identify a target model.
  - TrainingDetector        : builds a binary classifier (logistic regression) over
    text features (BoW, TF‑IDF, length) to distinguish a target model from all
    others, using previously collected response data.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import Config
from api import ModelAPI
from features import FeatureExtractor

logger = logging.getLogger(__name__)


# =========================================================================
# Identity‑Probing Detector
# =========================================================================

class IdentityProbingDetector:
    """
    Evaluates the accuracy of identity‑probing prompts for model de‑anonymisation.

    For each model and prompt defined in the configuration, the detector sends the
    prompt ``identity_num_queries`` times, then checks whether any of the model's
    pre‑defined keywords appear in the response (case‑insensitive).  The fraction
    of hits is returned as the detection accuracy.

    Attributes:
        config:           Application configuration (parsed from ``config.yaml``).
        api:              Authenticated API wrapper for querying all models.
        model_keywords:   Mapping from full model name to a list of keywords to
                          search for in the response.
    """

    def __init__(
        self,
        config: Config,
        api: ModelAPI,
    ) -> None:
        """
        Initialise the identity‑probing detector.

        Args:
            config: The global configuration object.
            api:    A ModelAPI instance for querying the language models.
        """
        if not isinstance(config, Config):
            raise TypeError("config must be an instance of Config")
        if not isinstance(api, ModelAPI):
            raise TypeError("api must be an instance of ModelAPI")

        self.config: Config = config
        self.api: ModelAPI = api

        # -- Build per‑model keyword lists from the configuration --
        self.model_keywords: Dict[str, List[str]] = {}
        keyword_map: Dict[str, List[str]] = self.config.detector.get("keyword_map", {})

        for model_name in self.config.models:
            matched_keywords: List[str] = []

            # Find the first family key that appears as a substring in the model name.
            # This heuristic works for all models in Appendix A.1.
            for family_key, keywords in keyword_map.items():
                if family_key.lower() in model_name.lower():
                    matched_keywords = keywords
                    break

            if not matched_keywords:
                logger.warning(
                    "No keyword family found for model '%s'. Detection will always fail.",
                    model_name,
                )
            self.model_keywords[model_name] = matched_keywords

        logger.info(
            "IdentityProbingDetector initialised for %d models, %d prompts.",
            len(self.config.models),
            len(self.config.identity_prompts),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        target_model: str,
        prompt: str,
        num_queries: int,
    ) -> float:
        """
        Query *target_model* *num_queries* times with *prompt* and compute the
        fraction of responses that contain any keyword.

        Args:
            target_model: Full model identifier (e.g. ``"gpt-4o-mini-2024-07-18"``).
            prompt:       The identity‑probing prompt text.
            num_queries:  Number of times to query the model.

        Returns:
            Detection accuracy as a float between 0 and 1.
        """
        keywords = self.model_keywords.get(target_model, [])
        if not keywords:
            # No keywords configured – output 0 for safety.
            return 0.0

        # Normalise keywords for case‑insensitive matching once
        lower_keywords = [kw.lower() for kw in keywords]

        hits = 0
        for i in range(num_queries):
            try:
                response = self.api.query(
                    model=target_model,
                    prompt=prompt,
                    max_tokens=self.config.max_output_tokens,
                    temperature=self.config.temperature,
                )
            except Exception as exc:
                logger.error(
                    "Query %d/%d for model '%s' failed: %s",
                    i + 1, num_queries, target_model, exc,
                )
                # Consider the failed query as a miss (conservative)
                continue

            response_lower = response.lower()
            if any(kw in response_lower for kw in lower_keywords):
                hits += 1

        accuracy = hits / num_queries if num_queries > 0 else 0.0
        logger.debug(
            "detect(model=%s, prompt='%s') -> %.4f (%d/%d hits)",
            target_model, prompt[:30], accuracy, hits, num_queries,
        )
        return accuracy

    def run_all_prompts(self) -> pd.DataFrame:
        """
        Execute the identity‑probing detection experiment for every model and every
        identity‑probing prompt defined in the configuration.

        Returns:
            A DataFrame with columns ``model``, ``prompt``, ``accuracy``.
        """
        rows: List[Dict[str, Any]] = []

        for model in self.config.models:
            for prompt in self.config.identity_prompts:
                accuracy = self.detect(
                    target_model=model,
                    prompt=prompt,
                    num_queries=self.config.identity_num_queries,
                )
                rows.append({
                    "model": model,
                    "prompt": prompt,
                    "accuracy": accuracy,
                })

        df = pd.DataFrame(rows)
        logger.info(
            "Identity‑probing evaluation complete. Mean accuracy over all models/prompts: %.4f",
            df["accuracy"].mean(),
        )
        return df


# =========================================================================
# Training‑Based Detector
# =========================================================================

class TrainingDetector:
    """
    Builds and evaluates binary classifiers that distinguish a target model from
    all other models using supervised learning on response features.

    The detector expects pre‑collected response data (loaded via :meth:`load_data`)
    and a prompt category mapping.  For each (prompt, target_model) pair and each
    feature type, a balanced dataset is constructed, a logistic regression
    classifier is trained, and test accuracy is computed.  Per‑category averages
    are then reported.

    Attributes:
        config:               Application configuration.
        feature_extractor:    A :class:`FeatureExtractor` instance (stored for
                              potential future use).
        response_data:        Nested dict ``{prompt: {model: [responses]}}``.
        category_prompts:     Dict ``{category: [prompt_text, ...]}`` mapping each
                              prompt category to the list of prompts that belong
                              to it.
    """

    def __init__(
        self,
        config: Config,
        feature_extractor: FeatureExtractor,
    ) -> None:
        """
        Initialise the training‑based detector.

        Args:
            config:           Application configuration.
            feature_extractor: An instance of :class:`FeatureExtractor`. Not
                               actively used by the current implementation
                               but kept for interface compatibility.
        """
        if not isinstance(config, Config):
            raise TypeError("config must be an instance of Config")
        if not isinstance(feature_extractor, FeatureExtractor):
            raise TypeError("feature_extractor must be an instance of FeatureExtractor")

        self.config: Config = config
        self.feature_extractor: FeatureExtractor = feature_extractor

        # Will be populated by load_data()
        self.response_data: Optional[Dict[str, Dict[str, List[str]]]] = None
        self.category_prompts: Optional[Dict[str, List[str]]] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(
        self,
        response_data: Dict[str, Dict[str, List[str]]],
        category_prompts: Dict[str, List[str]],
    ) -> None:
        """
        Provide the detector with pre‑collected response data and prompt groupings.

        Args:
            response_data:    A dictionary ``{prompt: {model: [response_str, ...]}}``.
                              Usually obtained from :class:`ResponseCollector`.
            category_prompts: A dictionary ``{category_name: [prompt_text, ...]}``
                              linking each category to its prompt texts.
        """
        if not isinstance(response_data, dict):
            raise TypeError("response_data must be a dictionary")
        if not isinstance(category_prompts, dict):
            raise TypeError("category_prompts must be a dictionary")

        # Basic sanity checks
        for prompt, model_map in response_data.items():
            if not isinstance(model_map, dict):
                raise ValueError(
                    f"response_data['{prompt}'] must be a dict, got {type(model_map)}"
                )
            for model, responses in model_map.items():
                if not isinstance(responses, list):
                    raise ValueError(
                        f"Responses for prompt '{prompt}', model '{model}' must be a list"
                    )
                # Ensure all entries are strings
                if not all(isinstance(r, str) for r in responses):
                    raise ValueError(
                        f"Non‑string response found for prompt '{prompt}', model '{model}'"
                    )

        self.response_data = response_data
        self.category_prompts = category_prompts
        logger.info(
            "TrainingDetector data loaded: %d prompts, %d categories.",
            len(response_data), len(category_prompts),
        )

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def prepare_dataset(
        self,
        target_model: str,
        prompt: str,
        feature_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a balanced binary dataset for a single (prompt, target_model) pair.

        The positive class consists of *num_responses_per_prompt* responses from the
        target model.  The negative class is obtained by randomly sampling the same
        number of responses from all other models for the same prompt.

        Args:
            target_model: The model to be identified (positive class).
            prompt:       The prompt text used to generate the responses.
            feature_type: One of ``"bow"``, ``"tfidf"``, ``"length_word"``,
                          ``"length_char"``.

        Returns:
            A tuple ``(X, y)`` where *X* is a dense feature matrix of shape
            ``(2*num_responses, n_features)`` and *y* is the corresponding binary
            label vector.

        Raises:
            KeyError: If the prompt is not present in the loaded response data.
            ValueError: If there are not enough responses to build a balanced set.
        """
        if self.response_data is None:
            raise RuntimeError("No response data loaded. Call load_data() first.")

        # -- Positive samples --
        if prompt not in self.response_data:
            raise KeyError(f"Prompt not found in response data: '{prompt}'")
        model_map = self.response_data[prompt]
        if target_model not in model_map:
            raise KeyError(
                f"Target model '{target_model}' not found for prompt '{prompt}'"
            )
        positives = model_map[target_model]
        if len(positives) < self.config.num_responses_per_prompt:
            raise ValueError(
                f"Insufficient positive responses for model '{target_model}', "
                f"prompt '{prompt}': expected {self.config.num_responses_per_prompt}, "
                f"got {len(positives)}"
            )
        pos_texts = positives[:self.config.num_responses_per_prompt]  # ensure exact length

        # -- Negative samples --
        # Gather all responses from other models for the same prompt.
        neg_pool: List[str] = []
        for model, responses in model_map.items():
            if model == target_model:
                continue
            neg_pool.extend(responses)

        if len(neg_pool) < self.config.num_responses_per_prompt:
            raise ValueError(
                f"Insufficient negative responses for prompt '{prompt}': "
                f"need {self.config.num_responses_per_prompt}, got {len(neg_pool)}"
            )

        # Deterministic sampling (global seed set in main.py)
        neg_texts = random.sample(neg_pool, self.config.num_responses_per_prompt)

        # -- Combined texts and labels --
        texts = pos_texts + neg_texts
        y = np.array([1] * len(pos_texts) + [0] * len(neg_texts), dtype=int)

        # -- Feature extraction --
        X = self._extract_features(texts, feature_type)

        return X, y

    def _extract_features(
        self,
        texts: List[str],
        feature_type: str,
    ) -> np.ndarray:
        """
        Convert a list of raw text responses into a numeric feature matrix.

        For BoW and TF‑IDF a fresh vectorizer is fitted on the provided texts,
        ensuring independence between different prompt–model experiments.

        Args:
            texts:        List of response strings.
            feature_type: One of ``"bow"``, ``"tfidf"``, ``"length_word"``,
                          ``"length_char"``.

        Returns:
            Dense 2‑D numpy array of shape ``(len(texts), n_features)``.
        """
        if feature_type == "bow":
            vectorizer = CountVectorizer(max_features=5000)
            X_sparse = vectorizer.fit_transform(texts)
            return X_sparse.toarray().astype(float)

        elif feature_type == "tfidf":
            vectorizer = TfidfVectorizer(max_features=5000)
            X_sparse = vectorizer.fit_transform(texts)
            return X_sparse.toarray().astype(float)

        elif feature_type == "length_word":
            # Feature: number of space‑separated tokens
            lengths = np.array([[len(t.split())] for t in texts], dtype=float)
            return lengths

        elif feature_type == "length_char":
            # Feature: total character count
            lengths = np.array([[len(t)] for t in texts], dtype=float)
            return lengths

        else:
            raise ValueError(
                f"Unsupported feature_type '{feature_type}'. "
                f"Allowed: bow, tfidf, length_word, length_char."
            )

    # ------------------------------------------------------------------
    # Training & evaluation
    # ------------------------------------------------------------------

    def train_one(
        self,
        prompt: str,
        target_model: str,
        feature_type: str,
    ) -> float:
        """
        Train a logistic regression classifier for a single (prompt, target_model)
        pair and evaluate its test accuracy.

        Args:
            prompt:       The prompt text.
            target_model: The target model (positive class).
            feature_type: The feature representation to use.

        Returns:
            Test accuracy as a float.
        """
        # -- Obtain features and labels --
        X, y = self.prepare_dataset(target_model, prompt, feature_type)

        # -- Train / test split (reproducible) --
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.detector.get("test_size", 0.2),
            random_state=self.config.detector.get("random_state", 42),
            stratify=y,
        )

        # -- Classifier (default hyper‑parameters as in the paper) --
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)

        # -- Evaluate --
        accuracy = clf.score(X_test, y_test)
        return float(accuracy)

    def evaluate_category(
        self,
        category: str,
        target_model: str,
        feature_type: str,
    ) -> float:
        """
        Compute the mean test accuracy across all prompts belonging to a given
        category.

        Args:
            category:     The prompt category (e.g. ``"english"``, ``"coding"``).
            target_model: The target model identifier.
            feature_type: The feature type.

        Returns:
            Mean accuracy as a float.
        """
        if self.category_prompts is None:
            raise RuntimeError("Category prompts not loaded. Call load_data() first.")

        prompts = self.category_prompts.get(category, [])
        if not prompts:
            logger.warning("No prompts found for category '%s'. Returning 0.0.", category)
            return 0.0

        accuracies: List[float] = []
        for prompt in prompts:
            try:
                acc = self.train_one(prompt, target_model, feature_type)
                accuracies.append(acc)
            except Exception as exc:
                logger.error(
                    "Error training for prompt '%s', target '%s', feature '%s': %s",
                    prompt, target_model, feature_type, exc,
                )
                # Skip invalid prompts – the average is over successful ones only.
                # This matches the paper’s approach where all 200 prompts are used.
                continue

        if not accuracies:
            logger.warning(
                "No successful training steps for category '%s', target '%s'. Returning 0.0.",
                category, target_model,
            )
            return 0.0

        mean_acc = float(np.mean(accuracies))
        logger.debug(
            "evaluate_category(%s, %s, %s) -> %.4f (N=%d)",
            category, target_model, feature_type, mean_acc, len(accuracies),
        )
        return mean_acc

    def run_full_evaluation(self) -> pd.DataFrame:
        """
        Perform the complete training‑based detector evaluation over all feature
        types, target models, and prompt categories.

        Returns:
            A DataFrame with columns ``feature_type``, ``target_model``, ``category``,
            ``accuracy``.
        """
        if self.response_data is None or self.category_prompts is None:
            raise RuntimeError(
                "Response data and category prompts must be loaded via load_data() "
                "before calling run_full_evaluation()."
            )

        feature_types = self.config.detector.get("feature_types", ["bow"])
        target_models = self.config.models
        categories = list(self.config.prompt_categories.keys())

        rows: List[Dict[str, Any]] = []
        for ft in feature_types:
            for tm in target_models:
                for cat in categories:
                    accuracy = self.evaluate_category(cat, tm, ft)
                    rows.append({
                        "feature_type": ft,
                        "target_model": tm,
                        "category": cat,
                        "accuracy": accuracy,
                    })

        df = pd.DataFrame(rows)
        logger.info(
            "Training‑based evaluation complete. %d combinations evaluated.", len(df),
        )
        return df

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<TrainingDetector: data_loaded={self.response_data is not None}, "
            f"models={len(self.config.models)}, categories={len(self.config.prompt_categories)}>"
        )
