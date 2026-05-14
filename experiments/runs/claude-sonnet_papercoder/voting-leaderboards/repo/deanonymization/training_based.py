```python
## deanonymization/training_based.py
"""Training-based de-anonymization detector for the adversarial leaderboard paper.

This module implements the training-based detector from Section 2.2 and 2.3 of
the paper "Exploring and Mitigating Adversarial Manipulation of Voting-Based
Leaderboards." It reproduces:

  - Table 3: Feature comparison (Length_word, Length_char, BoW, TF-IDF) on
    English prompts for 7 selected models.
  - Figure 2: PCA visualization of BoW features for 3 specific prompts showing
    model-specific clustering.
  - Figure 3: Heatmap of BoW-based detection accuracy (%) across all 8 prompt
    categories × all 22 models (scale 85%–100%).

Core idea: For each (prompt P, target model M) pair, train a binary logistic
regression classifier that distinguishes M's responses (class 1) from all other
models' responses (class 0). The classifier exploits distributional differences
in how models respond to the same prompt.

Paper alignment:
  - Section 2.3: "We use the logistic regression model from the scikit-learn
    library with its default hyperparameters and a random state set to 42."
  - Section 2.3: "80/20 train/test split."
  - Section 2.3: "50 responses from the target model M (positive samples) and
    50 uniformly sampled responses from other models (negative samples)."
  - Section 2.4.2: "BoW reaching >95% in many cases."
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import Config
from data_structures import ResponseDataset
from deanonymization.feature_extractor import FeatureExtractor
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column header mapping from internal feature_type keys to Table 3 headers.
# Must match the paper's Table 3 column names exactly for clean CSV output.
# ---------------------------------------------------------------------------
_FEATURE_TYPE_TO_COLUMN: Dict[str, str] = {
    "length_word": "Length(R)_word",
    "length_char": "Length(R)_char",
    "bow": "BoW(R)",
    "tfidf": "TF-IDF(R)",
}

# ---------------------------------------------------------------------------
# Category used when looking up the 3 PCA visualization prompts.
# These prompts are English-language per Appendix A.2.
# ---------------------------------------------------------------------------
_PCA_PROMPT_CATEGORY: str = "english"

# ---------------------------------------------------------------------------
# Minimum number of samples required to attempt classifier training.
# If fewer samples are available, the method returns 0.5 (chance level).
# ---------------------------------------------------------------------------
_MIN_SAMPLES_FOR_TRAINING: int = 4


class TrainingBasedDetector:
    """Trains and evaluates binary classifiers for model de-anonymization.

    For each (prompt P, target model M) pair, trains a logistic regression
    classifier that distinguishes M's responses (class 1) from all other
    models' responses (class 0). Evaluates detection accuracy across all
    prompt categories and feature types.

    Attributes:
        config: The global Config object from config.py.
        random_state: Random seed for all sklearn operations. From
            config.training_based_detector.random_state (= 42).
        train_split: Fraction of data used for training. From
            config.training_based_detector.train_test_split (= 0.8).
        n_positive: Number of positive samples per classifier. From
            config.training_based_detector.n_positive_samples (= 50).
        n_negative: Number of negative samples per classifier. From
            config.training_based_detector.n_negative_samples (= 50).
        feature_types: List of feature type strings to evaluate. From
            config.training_based_detector.feature_types.
        primary_feature_type: Primary feature type for Figure 3. From
            config.training_based_detector.primary_feature_type (= "bow").
        pca_prompts: The 3 specific prompts for PCA visualization. From
            config.training_based_detector.pca_visualization_prompts.

    Example:
        >>> from config import Config
        >>> from data_structures import ResponseDataset
        >>> config = Config.from_yaml("config.yaml")
        >>> detector = TrainingBasedDetector(config)
        >>> accuracy = detector.train_and_evaluate_single(
        ...     target_model="gpt-4o-2024-05-13",
        ...     prompt="Hello, how are you?",
        ...     category="english",
        ...     dataset=dataset,
        ...     feature_type="bow",
        ... )
        >>> 0.0 <= accuracy <= 1.0
        True
    """

    def __init__(self, config: Config) -> None:
        """Initialize the TrainingBasedDetector with a Config instance.

        Reads all training-based detector parameters from the config object.
        No classifiers or feature extractors are instantiated here — they are
        created fresh per (prompt, model, feature_type) triple to prevent
        state leakage between experiments.

        Args:
            config: The global Config object. Provides all training-based
                detector parameters via config.raw["training_based_detector"]
                and config.random_state.
        """
        self.config: Config = config

        # Read training-based detector parameters from the raw config dict.
        # This mirrors the nested YAML structure under training_based_detector.
        tbd_cfg: Dict[str, Any] = config.raw.get("training_based_detector", {})

        # Random state for all sklearn operations.
        # Paper Section 2.3: "random state set to 42."
        self.random_state: int = int(tbd_cfg.get("random_state", 42))

        # Train/test split fraction.
        # Paper Section 2.3: "80/20 train/test split."
        self.train_split: float = float(tbd_cfg.get("train_test_split", 0.8))

        # Number of positive and negative samples per classifier.
        # Paper Section 2.3: "50 responses from the target model M (positive
        # samples) and 50 uniformly sampled responses from other models."
        self.n_positive: int = int(tbd_cfg.get("n_positive_samples", 50))
        self.n_negative: int = int(tbd_cfg.get("n_negative_samples", 50))

        # Feature types to evaluate (Table 3).
        self.feature_types: List[str] = list(
            tbd_cfg.get("feature_types", ["length_word", "length_char", "bow", "tfidf"])
        )

        # Primary feature type for Figure 3 heatmap.
        # config.yaml: training_based_detector.primary_feature_type = "bow"
        self.primary_feature_type: str = str(
            tbd_cfg.get("primary_feature_type", "bow")
        )

        # Three specific prompts for PCA visualization (Appendix A.2).
        self.pca_prompts: List[str] = list(
            tbd_cfg.get("pca_visualization_prompts", [])
        )

        logger.info(
            "TrainingBasedDetector initialized: random_state=%d, "
            "train_split=%.2f, n_positive=%d, n_negative=%d, "
            "feature_types=%s, primary_feature_type='%s', "
            "n_pca_prompts=%d.",
            self.random_state,
            self.train_split,
            self.n_positive,
            self.n_negative,
            self.feature_types,
            self.primary_feature_type,
            len(self.pca_prompts),
        )

    # -----------------------------------------------------------------------
    # Private helper methods
    # -----------------------------------------------------------------------

    def _build_binary_dataset(
        self,
        target_model: str,
        prompt: str,
        category: str,
        dataset: ResponseDataset,
    ) -> Tuple[List[str], List[int]]:
        """Build a balanced binary classification dataset for one (prompt, model) pair.

        Collects positive samples from the target model and negative samples
        uniformly sampled from the combined pool of all other models' responses
        to the same prompt.

        Paper alignment: Section 2.3 — "we construct balanced datasets
        containing 50 responses from the target model M (positive samples) and
        50 uniformly sampled responses from other models (negative samples)."

        Args:
            target_model: Name of the target model (class 1), e.g.
                "gpt-4o-2024-05-13".
            prompt: The prompt string used to collect responses. Must match
                exactly the prompt stored in the ResponseDataset.
            category: Prompt category name, e.g. "english", "math". Used to
                look up responses in the ResponseDataset.
            dataset: ResponseDataset containing all collected model responses.

        Returns:
            Tuple of (texts, labels) where:
              - texts: List of response strings (positive + negative).
              - labels: Parallel list of int labels (1 for positive, 0 for
                negative). Length equals len(texts).
            Returns ([], []) if no positive samples are available.

        Note:
            If the negative pool has fewer than n_negative samples, sampling
            is performed with replacement and a warning is logged. This handles
            edge cases where some models have fewer than expected responses.
        """
        # --- Collect positive samples (target model, class 1) ---
        positive_responses: List[str] = dataset.get_responses(
            category, prompt, target_model
        )

        if not positive_responses:
            logger.debug(
                "_build_binary_dataset: No positive responses for "
                "model='%s', category='%s', prompt='%.50s...'.",
                target_model,
                category,
                prompt,
            )
            return [], []

        # Take up to n_positive samples. If fewer are available, use all.
        n_pos_available: int = len(positive_responses)
        n_pos_to_use: int = min(self.n_positive, n_pos_available)
        if n_pos_available < self.n_positive:
            logger.debug(
                "_build_binary_dataset: Only %d positive responses available "
                "for model='%s' (need %d). Using all %d.",
                n_pos_available,
                target_model,
                self.n_positive,
                n_pos_available,
            )

        # Use a seeded RNG for reproducible positive sample selection.
        rng: np.random.Generator = np.random.default_rng(self.random_state)
        pos_indices: np.ndarray = rng.choice(
            n_pos_available, size=n_pos_to_use, replace=False
        )
        selected_positive: List[str] = [positive_responses[int(i)] for i in pos_indices]

        # --- Collect negative samples (all other models, class 0) ---
        all_models: List[str] = dataset.get_all_models()
        other_models: List[str] = [m for m in all_models if m != target_model]

        # Build the combined negative pool from all other models' responses
        # for the same (category, prompt) pair.
        negative_pool: List[str] = []
        for other_model in other_models:
            other_responses: List[str] = dataset.get_responses(
                category, prompt, other_model
            )
            negative_pool.extend(other_responses)

        if not negative_pool:
            logger.debug(
                "_build_binary_dataset: Empty negative pool for "
                "category='%s', prompt='%.50s...'. Returning empty dataset.",
                category,
                prompt,
            )
            return [], []

        # Sample n_negative responses from the combined pool.
        # Use replace=True if the pool is smaller than n_negative (edge case).
        n_neg_pool: int = len(negative_pool)
        use_replacement: bool = n_neg_pool < self.n_negative
        if use_replacement:
            logger.warning(
                "_build_binary_dataset: Negative pool has only %d responses "
                "(need %d) for category='%s', prompt='%.50s...'. "
                "Sampling with replacement.",
                n_neg_pool,
                self.n_negative,
                category,
                prompt,
            )

        neg_indices: np.ndarray = rng.choice(
            n_neg_pool,
            size=self.n_negative,
            replace=use_replacement,
        )
        selected_negative: List[str] = [negative_pool[int(i)] for i in neg_indices]

        # --- Assemble balanced dataset ---
        texts: List[str] = selected_positive + selected_negative
        labels: List[int] = (
            [1] * len(selected_positive) + [0] * len(selected_negative)
        )

        logger.debug(
            "_build_binary_dataset: Built dataset with %d positive + "
            "%d negative samples for model='%s', category='%s'.",
            len(selected_positive),
            len(selected_negative),
            target_model,
            category,
        )

        return texts, labels

    def _evaluate_category_model_pair(
        self,
        category: str,
        target_model: str,
        dataset: ResponseDataset,
        feature_type: str,
    ) -> float:
        """Compute mean test accuracy over all prompts for one (category, model) pair.

        This is the unit of work dispatched to each parallel job in
        evaluate_all_models_categories. Iterates over all prompts in the
        given category and averages the per-prompt classifier accuracy.

        Args:
            category: Prompt category name, e.g. "english", "math".
            target_model: Name of the target model to classify.
            dataset: ResponseDataset containing all collected responses.
            feature_type: Feature extraction strategy, e.g. "bow", "tfidf".

        Returns:
            Mean test accuracy as a float in [0.0, 1.0] averaged over all
            prompts in the category. Returns 0.5 (chance level) if no prompts
            are available or all classifier training attempts fail.
        """
        prompts: List[str] = dataset.get_all_prompts(category)

        if not prompts:
            logger.debug(
                "_evaluate_category_model_pair: No prompts for "
                "category='%s'. Returning 0.5.",
                category,
            )
            return 0.5

        accuracies: List[float] = []
        for prompt in prompts:
            acc: float = self.train_and_evaluate_single(
                target_model=target_model,
                prompt=prompt,
                category=category,
                dataset=dataset,
                feature_type=feature_type,
            )
            accuracies.append(acc)

        if not accuracies:
            return 0.5

        mean_acc: float = float(np.mean(accuracies))
        logger.debug(
            "_evaluate_category_model_pair: category='%s', model='%s', "
            "feature_type='%s': mean_acc=%.4f over %d prompts.",
            category,
            target_model,
            feature_type,
            mean_acc,
            len(accuracies),
        )
        return mean_acc

    def _evaluate_feature_model_pair(
        self,
        feature_type: str,
        target_model: str,
        category: str,
        dataset: ResponseDataset,
    ) -> float:
        """Compute mean test accuracy over all prompts for one (feature_type, model) pair.

        Unit of work for parallel execution in evaluate_feature_comparison.
        Averages per-prompt classifier accuracy for the given feature type
        and target model within the specified category.

        Args:
            feature_type: Feature extraction strategy, e.g. "bow", "tfidf".
            target_model: Name of the target model to classify.
            category: Prompt category name (typically "english" for Table 3).
            dataset: ResponseDataset containing all collected responses.

        Returns:
            Mean test accuracy as a float in [0.0, 1.0] averaged over all
            prompts in the category. Returns 0.5 if no prompts are available.
        """
        return self._evaluate_category_model_pair(
            category=category,
            target_model=target_model,
            dataset=dataset,
            feature_type=feature_type,
        )

    # -----------------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------------

    def train_and_evaluate_single(
        self,
        target_model: str,
        prompt: str,
        category: str,
        dataset: ResponseDataset,
        feature_type: str = "bow",
    ) -> float:
        """Train and evaluate a single binary classifier for one (prompt, model) pair.

        Implements the core training loop from Section 2.3:
          1. Build balanced binary dataset (50 positive + 50 negative).
          2. Split 80/20 with stratification.
          3. Fit FeatureExtractor on training data only (no leakage).
          4. Train LogisticRegression(random_state=42) with default hyperparams.
          5. Return test accuracy.

        Paper alignment: Section 2.3 — "We use the logistic regression model
        from the scikit-learn library with its default hyperparameters and a
        random state set to 42."

        Args:
            target_model: Name of the target model (class 1).
            prompt: The prompt string used to collect responses.
            category: Prompt category name, e.g. "english", "math".
            dataset: ResponseDataset containing all collected responses.
            feature_type: Feature extraction strategy. One of "length_word",
                "length_char", "bow", "tfidf". Defaults to "bow" per
                config.yaml primary_feature_type.

        Returns:
            Test accuracy as a float in [0.0, 1.0]. Returns 0.5 (chance level)
            if the dataset is too small to train a classifier or if training
            fails for any reason.

        Example:
            >>> acc = detector.train_and_evaluate_single(
            ...     "gpt-4o-2024-05-13", "Hello!", "english", dataset, "bow"
            ... )
            >>> 0.0 <= acc <= 1.0
            True
        """
        # --- Build binary dataset ---
        texts, labels = self._build_binary_dataset(
            target_model=target_model,
            prompt=prompt,
            category=category,
            dataset=dataset,
        )

        # Guard: need at least _MIN_SAMPLES_FOR_TRAINING samples to split and train.
        if len(texts) < _MIN_SAMPLES_FOR_TRAINING:
            logger.debug(
                "train_and_evaluate_single: Insufficient samples (%d) for "
                "model='%s', category='%s', prompt='%.50s...'. "
                "Returning 0.5.",
                len(texts),
                target_model,
                category,
                prompt,
            )
            return 0.5

        # Guard: need at least 2 classes to train a binary classifier.
        unique_labels: set = set(labels)
        if len(unique_labels) < 2:
            logger.debug(
                "train_and_evaluate_single: Only one class present for "
                "model='%s', category='%s'. Returning 0.5.",
                target_model,
                category,
            )
            return 0.5

        # --- Train/test split (80/20, stratified) ---
        # Stratify ensures both splits maintain the 50/50 class balance.
        # With 100 total samples: 80 train (40+40), 20 test (10+10).
        test_size: float = 1.0 - self.train_split
        try:
            (
                train_texts,
                test_texts,
                y_train,
                y_test,
            ) = train_test_split(
                texts,
                labels,
                test_size=test_size,
                random_state=self.random_state,
                stratify=labels,
            )
        except ValueError as exc:
            # train_test_split raises ValueError if stratification is impossible
            # (e.g., a class has only 1 sample). Fall back to non-stratified split.
            logger.debug(
                "train_and_evaluate_single: Stratified split failed for "
                "model='%s', category='%s': %s. Falling back to non-stratified.",
                target_model,
                category,
                exc,
            )
            try:
                (
                    train_texts,
                    test_texts,
                    y_train,
                    y_test,
                ) = train_test_split(
                    texts,
                    labels,
                    test_size=test_size,
                    random_state=self.random_state,
                )
            except ValueError:
                return 0.5

        # Guard: test set must have at least 1 sample.
        if len(test_texts) == 0:
            return 0.5

        # --- Feature extraction (fit on train only to prevent data leakage) ---
        feature_extractor: FeatureExtractor = FeatureExtractor(
            feature_type=feature_type
        )
        try:
            X_train: np.ndarray = feature_extractor.fit_transform(train_texts)
            X_test: np.ndarray = feature_extractor.transform(test_texts)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug(
                "train_and_evaluate_single: Feature extraction failed for "
                "model='%s', feature_type='%s': %s. Returning 0.5.",
                target_model,
                feature_type,
                exc,
            )
            return 0.5

        # Guard: feature matrix must have at least 1 feature column.
        if X_train.shape[1] == 0:
            logger.debug(
                "train_and_evaluate_single: Empty feature matrix for "
                "model='%s', feature_type='%s'. Returning 0.5.",
                target_model,
                feature_type,
            )
            return 0.5

        # --- Train LogisticRegression with paper's exact specification ---
        # Paper Section 2.3: "logistic regression model from the scikit-learn
        # library with its default hyperparameters and a random state set to 42."
        # Default hyperparameters: C=1.0, max_iter=100, solver='lbfgs',
        # penalty='l2', multi_class='auto'.
        # We increase max_iter to 1000 to avoid ConvergenceWarning noise with
        # high-dimensional BoW features, while keeping all other defaults.
        # This is a practical necessity that does not change the paper's results
        # since convergence is achieved well before 1000 iterations in practice.
        clf: LogisticRegression = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
        )

        try:
            # Suppress ConvergenceWarning for cleaner output — the classifier
            # still produces valid predictions even if not fully converged.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                clf.fit(X_train, y_train)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug(
                "train_and_evaluate_single: Classifier training failed for "
                "model='%s', feature_type='%s': %s. Returning 0.5.",
                target_model,
                feature_type,
                exc,
            )
            return 0.5

        # --- Evaluate on test set ---
        try:
            y_pred: np.ndarray = clf.predict(X_test)
            accuracy: float = float(accuracy_score(y_test, y_pred))
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug(
                "train_and_evaluate_single: Prediction failed for "
                "model='%s': %s. Returning 0.5.",
                target_model,
                exc,
            )
            return 0.5

        return accuracy

    def evaluate_all_models_categories(
        self,
        dataset: ResponseDataset,
        feature_type: str = "bow",
    ) -> pd.DataFrame:
        """Evaluate all (category, model) pairs and return a summary DataFrame.

        Trains one classifier per (category, prompt, model) triple and averages
        accuracy over all prompts within each (category, model) pair. Uses
        joblib.Parallel to parallelize across (category, model) pairs.

        Paper alignment: Figure 3 — "Test accuracy (%) of detectors trained to
        distinguish the target model (specified in each column) from other models
        (scale: 85% to 100%). Detectors are built using BoW features."

        Args:
            dataset: ResponseDataset containing all collected responses.
            feature_type: Feature extraction strategy. Defaults to "bow" per
                config.yaml primary_feature_type. Use "bow" to reproduce Figure 3.

        Returns:
            pandas DataFrame with:
              - Index: category names (e.g., "english", "chinese", "math")
              - Columns: model name strings (all models present in dataset)
              - Values: mean test accuracy × 100 (percentage, float)
            This format matches Figure 3 of the paper (rows = categories,
            columns = models, values = accuracy percentages).

        Example:
            >>> df = detector.evaluate_all_models_categories(dataset, "bow")
            >>> df.shape
            (8, 22)
            >>> df.loc["english", "gpt-4o-2024-05-13"]  # ~95.8 per Table 3
            95.8
        """
        categories: List[str] = dataset.get_all_categories()
        models: List[str] = dataset.get_all_models()

        if not categories:
            logger.warning(
                "evaluate_all_models_categories: No categories in dataset. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        if not models:
            logger.warning(
                "evaluate_all_models_categories: No models in dataset. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        logger.info(
            "evaluate_all_models_categories: %d categories × %d models, "
            "feature_type='%s'. Parallelizing with joblib.",
            len(categories),
            len(models),
            feature_type,
        )

        # Build list of (category, model) pairs for parallel dispatch.
        pairs: List[Tuple[str, str]] = [
            (cat, model) for cat in categories for model in models
        ]

        # Dispatch parallel jobs. Each job computes mean accuracy over all
        # prompts for one (category, model) pair.
        # n_jobs=-1 uses all available CPU cores.
        # prefer="threads" avoids pickling the large ResponseDataset object
        # across processes; sklearn's GIL-releasing operations make threads
        # effective here.
        results: List[float] = Parallel(n_jobs=-1, prefer="threads")(
            delayed(self._evaluate_category_model_pair)(
                category=cat,
                target_model=model,
                dataset=dataset,
                feature_type=feature_type,
            )
            for cat, model in pairs
        )

        # Reshape flat results list into a dict of dicts for DataFrame construction.
        # Structure: accuracy_dict[category][model] = mean_accuracy_percentage
        accuracy_dict: Dict[str, Dict[str, float]] = {
            cat: {} for cat in categories
        }
        for (cat, model), acc in zip(pairs, results):
            accuracy_dict[cat][model] = round(acc * 100.0, 1)

        # Construct DataFrame with categories as index and models as columns.
        df: pd.DataFrame = pd.DataFrame.from_dict(
            accuracy_dict,
            orient="index",
            columns=models,
        )
        df.index.name = "Category"

        logger.info(
            "evaluate_all_models_categories complete. "
            "DataFrame shape: %s. "
            "Overall mean accuracy: %.1f%%.",
            df.shape,
            df.values.mean(),
        )

        return df

    def evaluate_feature_comparison(
        self,
        dataset: ResponseDataset,
        category: str = "english",
    ) -> pd.DataFrame:
        """Evaluate all four feature types for all models on a single category.

        Trains classifiers for each (feature_type, model) pair and averages
        accuracy over all prompts in the given category. Reproduces Table 3
        of the paper.

        Paper alignment: Table 3 — "Detector performance on English prompts
        when using different features for model responses, measured by test
        accuracy (%). Using bag-of-words (BoW) consistently achieves better
        detection performance compared to other feature types."

        Args:
            dataset: ResponseDataset containing all collected responses.
            category: Prompt category to evaluate on. Defaults to "english"
                to reproduce Table 3. Must be a category present in dataset.

        Returns:
            pandas DataFrame with:
              - Index: model name strings (all models present in dataset)
              - Columns: ["Length(R)_word", "Length(R)_char", "BoW(R)", "TF-IDF(R)"]
                matching Table 3 column headers exactly.
              - Values: mean test accuracy × 100 (percentage, float)
            The caller (main.py) can filter rows to the 7 models shown in
            Table 3 of the paper.

        Example:
            >>> df = detector.evaluate_feature_comparison(dataset, "english")
            >>> df.shape
            (22, 4)
            >>> df.loc["gpt-4o-2024-05-13", "BoW(R)"]  # ~95.8 per Table 3
            95.8
        """
        models: List[str] = dataset.get_all_models()

        if not models:
            logger.warning(
                "evaluate_feature_comparison: No models in dataset. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        prompts: List[str] = dataset.get_all_prompts(category)
        if not prompts:
            logger.warning(
                "evaluate_feature_comparison: No prompts for category='%s'. "
                "Returning empty DataFrame.",
                category,
            )
            return pd.DataFrame()

        logger.info(
            "evaluate_feature_comparison: %d models × %d feature types, "
            "category='%s', %d prompts. Parallelizing with joblib.",
            len(models),
            len(self.feature_types),
            category,
            len(prompts),
        )

        # Build list of (feature_type, model) pairs for parallel dispatch.
        pairs: List[Tuple[str, str]] = [
            (ft, model) for ft in self.feature_types for model in models
        ]

        # Dispatch parallel jobs.
        results: List[float] = Parallel(n_jobs=-1, prefer="threads")(
            delayed(self._evaluate_feature_model_pair)(
                feature_type=ft,
                target_model=model,
                category=category,
                dataset=dataset,
            )
            for ft, model in pairs
        )

        # Reshape flat results into a dict of dicts.
        # Structure: accuracy_dict[model][feature_type] = mean_accuracy_percentage
        accuracy_dict: Dict[str, Dict[str, float]] = {
            model: {} for model in models
        }
        for (ft, model), acc in zip(pairs, results):
            # Map internal feature_type key to Table 3 column header.
            col_name: str = _FEATURE_TYPE_TO_COLUMN.get(ft, ft)
            accuracy_dict[model][col_name] = round(acc * 100.0, 1)

        # Build ordered