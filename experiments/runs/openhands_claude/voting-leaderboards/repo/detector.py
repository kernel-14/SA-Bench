"""
Model detectors for de-anonymizing LLM responses.

Implements the two detector types from Section 2.2:
  1. Identity-probing detector: checks if model reveals its identity in response
  2. Training-based detector: logistic regression on text features (BoW/TF-IDF/Length)

The training-based detector is the primary contribution, achieving >95% accuracy
on most models using BoW features (Table 3, Figure 3).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from config import MODEL_FAMILY_KEYWORDS, DetectorConfig
from data import DetectorDataset, build_detector_dataset, train_test_split_dataset
from features import FeatureType, extract_features


# ---------------------------------------------------------------------------
# Identity-probing detector (Section 2.2, 2.3)
# ---------------------------------------------------------------------------

def identity_probing_detect(
    response: str,
    target_model: str,
    keywords: Optional[List[str]] = None,
) -> bool:
    """
    Detect if a response comes from the target model by checking for
    model family name or organization name in the response text.

    The classifier predicts a positive match if the model's name (e.g., "Llama")
    or organization (e.g., "Meta") appears anywhere in the response (Section 2.3).

    Args:
        response: The model's text response.
        target_model: The target model identifier.
        keywords: Optional override for keywords to search. If None, uses
                  MODEL_FAMILY_KEYWORDS from config.

    Returns:
        True if the response is predicted to come from the target model.
    """
    if keywords is None:
        keywords = MODEL_FAMILY_KEYWORDS.get(target_model, [])

    response_lower = response.lower()
    for keyword in keywords:
        if keyword.lower() in response_lower:
            return True
    return False


def evaluate_identity_probing_detector(
    responses: List[str],
    target_model: str,
    true_labels: List[int],
    keywords: Optional[List[str]] = None,
) -> float:
    """
    Evaluate the identity-probing detector on a set of responses.

    Args:
        responses: List of model responses.
        target_model: The target model to detect.
        true_labels: Ground truth labels (1 = target model, 0 = other).
        keywords: Optional keyword override.

    Returns:
        Detection accuracy as a float in [0, 1].
    """
    predictions = [
        int(identity_probing_detect(r, target_model, keywords))
        for r in responses
    ]
    return accuracy_score(true_labels, predictions)


# ---------------------------------------------------------------------------
# Training-based detector (Section 2.2, 2.3)
# ---------------------------------------------------------------------------

@dataclass
class TrainingBasedDetector:
    """
    Binary classifier that distinguishes a target model from all others.

    Uses logistic regression with text features (BoW, TF-IDF, or Length).
    Trained per (prompt, model) pair as described in Section 2.3.

    The paper uses sklearn's LogisticRegression with default hyperparameters
    and random_state=42.
    """
    target_model: str
    feature_type: FeatureType
    random_state: int = 42

    def __post_init__(self) -> None:
        self.classifier = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
        )
        self._feature_extractor_train_data: Optional[List[str]] = None
        self._fitted = False

    def fit(
        self,
        train_responses: List[str],
        train_labels: List[int],
    ) -> "TrainingBasedDetector":
        """
        Fit the detector on training data.

        Args:
            train_responses: Training responses.
            train_labels: Binary labels (1 = target model, 0 = other).
        """
        self._feature_extractor_train_data = train_responses
        X_train, _ = extract_features(train_responses, train_responses, self.feature_type)
        self.classifier.fit(X_train, train_labels)
        self._fitted = True
        return self

    def predict(self, responses: List[str]) -> np.ndarray:
        """
        Predict whether each response comes from the target model.

        Args:
            responses: List of text responses.

        Returns:
            Binary predictions array of shape (n_samples,).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        X_train = self._feature_extractor_train_data
        _, X_test = extract_features(X_train, responses, self.feature_type)
        return self.classifier.predict(X_test)

    def predict_proba(self, responses: List[str]) -> np.ndarray:
        """
        Return probability estimates for each response.

        Returns:
            Array of shape (n_samples, 2) with [P(class=0), P(class=1)].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        X_train = self._feature_extractor_train_data
        _, X_test = extract_features(X_train, responses, self.feature_type)
        return self.classifier.predict_proba(X_test)

    def score(
        self,
        test_responses: List[str],
        test_labels: List[int],
    ) -> float:
        """
        Compute test accuracy.

        Args:
            test_responses: Test responses.
            test_labels: Ground truth labels.

        Returns:
            Accuracy in [0, 1].
        """
        predictions = self.predict(test_responses)
        return accuracy_score(test_labels, predictions)


def train_detector_for_prompt(
    dataset: DetectorDataset,
    feature_type: FeatureType = "bow",
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> Tuple[TrainingBasedDetector, float]:
    """
    Train and evaluate a detector for a single (prompt, target_model) pair.

    Implements the training procedure from Section 2.3:
    - 80/20 train/test split
    - Logistic regression with default sklearn hyperparameters
    - random_state=42

    Args:
        dataset: DetectorDataset for a specific prompt and target model.
        feature_type: Feature type to use.
        train_ratio: Fraction of data for training.
        random_state: Random seed.

    Returns:
        Tuple of (fitted detector, test accuracy).
    """
    train_data, test_data = train_test_split_dataset(dataset, train_ratio, random_state)

    detector = TrainingBasedDetector(
        target_model=dataset.prompt,  # prompt used as identifier
        feature_type=feature_type,
        random_state=random_state,
    )
    detector.fit(train_data.responses, train_data.labels)
    test_acc = detector.score(test_data.responses, test_data.labels)

    return detector, test_acc


# ---------------------------------------------------------------------------
# Multi-prompt detector (aggregates across prompts)
# ---------------------------------------------------------------------------

@dataclass
class MultiPromptDetector:
    """
    Aggregates predictions from multiple per-prompt detectors.

    In practice, the attacker trains one detector per prompt and uses
    the best-performing prompt for the attack (Section 2.2).
    """
    target_model: str
    feature_type: FeatureType = "bow"
    random_state: int = 42

    def __post_init__(self) -> None:
        self.detectors: List[TrainingBasedDetector] = []
        self.prompt_accuracies: List[float] = []

    def add_detector(self, detector: TrainingBasedDetector, accuracy: float) -> None:
        self.detectors.append(detector)
        self.prompt_accuracies.append(accuracy)

    def predict_best(self, responses: List[str]) -> np.ndarray:
        """Use the best-performing detector (highest validation accuracy)."""
        if not self.detectors:
            raise RuntimeError("No detectors added.")
        best_idx = int(np.argmax(self.prompt_accuracies))
        return self.detectors[best_idx].predict(responses)

    def predict_majority_vote(self, responses: List[str]) -> np.ndarray:
        """Majority vote across all detectors."""
        if not self.detectors:
            raise RuntimeError("No detectors added.")
        all_preds = np.stack([d.predict(responses) for d in self.detectors], axis=0)
        return (all_preds.mean(axis=0) >= 0.5).astype(int)

    def mean_accuracy(self) -> float:
        """Average test accuracy across all prompts (reported in Table 3, Figure 3)."""
        if not self.prompt_accuracies:
            return 0.0
        return float(np.mean(self.prompt_accuracies))


# ---------------------------------------------------------------------------
# Detector evaluation across models and categories (Figure 3)
# ---------------------------------------------------------------------------

def evaluate_detector_across_models(
    responses_by_prompt_model: Dict[str, Dict[str, List[str]]],
    target_models: List[str],
    all_models: List[str],
    feature_type: FeatureType = "bow",
    num_positive: int = 50,
    num_negative: int = 50,
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Evaluate the training-based detector for each target model.

    Reproduces the results in Table 3 and Figure 3.

    Args:
        responses_by_prompt_model: {prompt -> {model -> [responses]}}
        target_models: Models to evaluate as targets.
        all_models: Full list of models (for negative sampling).
        feature_type: Feature type to use.
        num_positive: Positive samples per prompt.
        num_negative: Negative samples per prompt.
        train_ratio: Train/test split ratio.
        random_state: Random seed.

    Returns:
        Dict mapping target model name to mean test accuracy across prompts.
    """
    results: Dict[str, float] = {}

    for target_model in target_models:
        prompt_accuracies = []

        for prompt, model_responses in responses_by_prompt_model.items():
            if target_model not in model_responses:
                continue

            dataset = build_detector_dataset(
                responses_by_model=model_responses,
                target_model=target_model,
                prompt=prompt,
                category="unknown",
                num_positive=num_positive,
                num_negative=num_negative,
                random_seed=random_state,
            )

            if len(dataset.responses) < 4:
                continue

            _, test_acc = train_detector_for_prompt(
                dataset=dataset,
                feature_type=feature_type,
                train_ratio=train_ratio,
                random_state=random_state,
            )
            prompt_accuracies.append(test_acc)

        results[target_model] = float(np.mean(prompt_accuracies)) if prompt_accuracies else 0.0

    return results


def evaluate_detector_by_category(
    responses_by_category: Dict[str, Dict[str, Dict[str, List[str]]]],
    target_models: List[str],
    feature_type: FeatureType = "bow",
    num_positive: int = 50,
    num_negative: int = 50,
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate detector accuracy per category per target model (Figure 3).

    Args:
        responses_by_category: {category -> {prompt -> {model -> [responses]}}}
        target_models: Models to evaluate.
        feature_type: Feature type.
        num_positive: Positive samples per prompt.
        num_negative: Negative samples per prompt.
        train_ratio: Train/test split.
        random_state: Random seed.

    Returns:
        Dict: {category -> {target_model -> mean_accuracy}}
    """
    results: Dict[str, Dict[str, float]] = {}

    for category, responses_by_prompt_model in responses_by_category.items():
        results[category] = evaluate_detector_across_models(
            responses_by_prompt_model=responses_by_prompt_model,
            target_models=target_models,
            all_models=target_models,
            feature_type=feature_type,
            num_positive=num_positive,
            num_negative=num_negative,
            train_ratio=train_ratio,
            random_state=random_state,
        )

    return results
