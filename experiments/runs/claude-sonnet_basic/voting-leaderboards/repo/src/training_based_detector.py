"""
Training-based detector for de-anonymizing model responses.

This module implements the training-based attack described in Section 2.2 of the paper.
The attacker uses supervised learning to differentiate between models' responses to the
same prompt, using simple text features (Length, TF-IDF, BoW) and logistic regression.

Key details from the paper (Section 2.3):
- 200 prompts per category, 50 responses per model per prompt
- 80/20 train/test split
- Logistic regression from scikit-learn with default hyperparameters, random_state=42
- Features: Length (word/character), TF-IDF, Bag-of-Words
"""

import numpy as np
import logging
from typing import Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Feature types supported
FEATURE_TYPES = ["length_word", "length_char", "bow", "tfidf"]


def extract_length_features(responses: list, mode: str = "word") -> np.ndarray:
    """
    Extract length features from responses.

    Args:
        responses: List of response strings
        mode: "word" for word count, "char" for character count

    Returns:
        2D numpy array of shape (n_responses, 1)
    """
    if mode == "word":
        lengths = [len(r.split()) for r in responses]
    elif mode == "char":
        lengths = [len(r) for r in responses]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'word' or 'char'.")

    return np.array(lengths).reshape(-1, 1)


class ModelDetector:
    """
    Training-based model detector using text features and logistic regression.

    Implements the binary classifier f_M described in Section 2.2 of the paper.
    Given a prompt-response pair, outputs 1 if the response comes from the target model.
    """

    def __init__(
        self,
        feature_type: str = "bow",
        random_state: int = 42,
        max_features: int = 10000,
    ):
        """
        Initialize the detector.

        Args:
            feature_type: Type of features to use ("length_word", "length_char",
                         "bow", "tfidf")
            random_state: Random state for reproducibility (default: 42 as in paper)
            max_features: Maximum number of features for BoW/TF-IDF
        """
        if feature_type not in FEATURE_TYPES:
            raise ValueError(
                f"Unknown feature type: {feature_type}. "
                f"Choose from {FEATURE_TYPES}"
            )

        self.feature_type = feature_type
        self.random_state = random_state
        self.max_features = max_features
        self.classifier = None
        self.vectorizer = None
        self.scaler = None
        self._is_fitted = False

    def _build_vectorizer(self):
        """Build the text vectorizer based on feature type."""
        if self.feature_type == "bow":
            return CountVectorizer(max_features=self.max_features)
        elif self.feature_type == "tfidf":
            return TfidfVectorizer(max_features=self.max_features)
        else:
            return None  # Length features don't need a vectorizer

    def _extract_features(self, responses: list, fit: bool = False) -> np.ndarray:
        """
        Extract features from responses.

        Args:
            responses: List of response strings
            fit: Whether to fit the vectorizer (True for training, False for inference)

        Returns:
            Feature matrix
        """
        if self.feature_type == "length_word":
            return extract_length_features(responses, mode="word")
        elif self.feature_type == "length_char":
            return extract_length_features(responses, mode="char")
        elif self.feature_type in ["bow", "tfidf"]:
            if fit:
                self.vectorizer = self._build_vectorizer()
                return self.vectorizer.fit_transform(responses).toarray()
            else:
                return self.vectorizer.transform(responses).toarray()
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

    def fit(
        self,
        target_responses: list,
        other_responses: list,
        test_size: float = 0.2,
    ) -> dict:
        """
        Train the detector on target model responses vs. other model responses.

        Constructs balanced datasets with 50 responses from target model (positive)
        and 50 uniformly sampled responses from other models (negative), as described
        in Section 2.3.

        Args:
            target_responses: List of responses from the target model (positive class)
            other_responses: List of responses from other models (negative class)
            test_size: Fraction of data to use for testing (default: 0.2 as in paper)

        Returns:
            Dictionary with training results including test accuracy
        """
        # Create labels
        pos_labels = [1] * len(target_responses)
        neg_labels = [0] * len(other_responses)

        all_responses = target_responses + other_responses
        all_labels = pos_labels + neg_labels

        # Extract features
        X = self._extract_features(all_responses, fit=True)
        y = np.array(all_labels)

        # Scale features for length-based features
        if self.feature_type in ["length_word", "length_char"]:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Train logistic regression with default hyperparameters (as in paper)
        self.classifier = LogisticRegression(random_state=self.random_state)
        self.classifier.fit(X_train, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        self._is_fitted = True

        return {
            "test_accuracy": test_accuracy * 100,  # Convert to percentage
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_positive_train": int(y_train.sum()),
            "n_negative_train": int((1 - y_train).sum()),
        }

    def predict(self, response: str) -> int:
        """
        Predict whether a response is from the target model.

        Args:
            response: Response text to classify

        Returns:
            1 if predicted to be from target model, 0 otherwise
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        X = self._extract_features([response], fit=False)

        if self.feature_type in ["length_word", "length_char"] and self.scaler:
            X = self.scaler.transform(X)

        return int(self.classifier.predict(X)[0])

    def predict_proba(self, response: str) -> float:
        """
        Get the probability that a response is from the target model.

        Args:
            response: Response text to classify

        Returns:
            Probability of being from target model (class 1)
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        X = self._extract_features([response], fit=False)

        if self.feature_type in ["length_word", "length_char"] and self.scaler:
            X = self.scaler.transform(X)

        return float(self.classifier.predict_proba(X)[0][1])


def build_balanced_dataset(
    target_responses: list,
    other_model_responses: dict,
    n_per_class: int = 50,
    random_state: int = 42,
) -> tuple:
    """
    Build a balanced dataset for training the detector.

    As described in Section 2.3: "we construct balanced datasets containing 50 responses
    from the target model M (positive samples) and 50 uniformly sampled responses from
    other models (negative samples)."

    Args:
        target_responses: List of responses from the target model
        other_model_responses: Dictionary mapping model_name -> list of responses
        n_per_class: Number of samples per class (default: 50 as in paper)
        random_state: Random state for reproducibility

    Returns:
        Tuple of (positive_responses, negative_responses)
    """
    rng = np.random.RandomState(random_state)

    # Sample positive examples from target model
    if len(target_responses) >= n_per_class:
        pos_indices = rng.choice(len(target_responses), n_per_class, replace=False)
        positive = [target_responses[i] for i in pos_indices]
    else:
        positive = target_responses
        logger.warning(
            f"Only {len(target_responses)} target responses available, "
            f"expected {n_per_class}"
        )

    # Collect all other model responses
    all_other = []
    for model_name, responses in other_model_responses.items():
        all_other.extend(responses)

    # Sample negative examples uniformly from other models
    if len(all_other) >= n_per_class:
        neg_indices = rng.choice(len(all_other), n_per_class, replace=False)
        negative = [all_other[i] for i in neg_indices]
    else:
        negative = all_other
        logger.warning(
            f"Only {len(all_other)} other model responses available, "
            f"expected {n_per_class}"
        )

    return positive, negative


def train_detector_for_prompt(
    prompt: str,
    target_model: str,
    target_responses: list,
    other_model_responses: dict,
    feature_type: str = "bow",
    n_per_class: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Train a detector for a specific prompt-model pair (P, M).

    As described in Section 2.3: "We then train a logistic regression classifier for
    each prompt-model pair (P, M) using an 80/20 train/test split."

    Args:
        prompt: The prompt used to generate responses
        target_model: The target model name
        target_responses: Responses from the target model for this prompt
        other_model_responses: Dict mapping model_name -> responses for this prompt
        feature_type: Type of features to use
        n_per_class: Number of samples per class
        test_size: Fraction for test split
        random_state: Random state

    Returns:
        Tuple of (detector, results_dict)
    """
    # Build balanced dataset
    positive, negative = build_balanced_dataset(
        target_responses, other_model_responses, n_per_class, random_state
    )

    # Train detector
    detector = ModelDetector(
        feature_type=feature_type, random_state=random_state
    )
    results = detector.fit(positive, negative, test_size=test_size)
    results["prompt"] = prompt
    results["target_model"] = target_model
    results["feature_type"] = feature_type

    return detector, results


def evaluate_detector_across_prompts(
    target_model: str,
    all_responses: dict,
    feature_type: str = "bow",
    n_per_class: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Evaluate the detector across all prompts for a target model.

    As described in Section 2.3: "We evaluate the classifier using the average test
    accuracy across all prompts."

    Args:
        target_model: The target model name
        all_responses: Dict mapping prompt -> {model_name -> list of responses}
        feature_type: Type of features to use
        n_per_class: Number of samples per class
        test_size: Fraction for test split
        random_state: Random state

    Returns:
        Dictionary with per-prompt results and average accuracy
    """
    per_prompt_results = {}
    accuracies = []

    for prompt, model_responses in all_responses.items():
        if target_model not in model_responses:
            logger.warning(
                f"Target model {target_model} not found for prompt '{prompt[:50]}...'"
            )
            continue

        target_responses = model_responses[target_model]
        other_model_responses = {
            m: r for m, r in model_responses.items() if m != target_model
        }

        _, results = train_detector_for_prompt(
            prompt=prompt,
            target_model=target_model,
            target_responses=target_responses,
            other_model_responses=other_model_responses,
            feature_type=feature_type,
            n_per_class=n_per_class,
            test_size=test_size,
            random_state=random_state,
        )

        per_prompt_results[prompt] = results
        accuracies.append(results["test_accuracy"])

    avg_accuracy = np.mean(accuracies) if accuracies else 0.0

    return {
        "target_model": target_model,
        "feature_type": feature_type,
        "average_accuracy": avg_accuracy,
        "per_prompt_results": per_prompt_results,
        "n_prompts": len(accuracies),
    }
