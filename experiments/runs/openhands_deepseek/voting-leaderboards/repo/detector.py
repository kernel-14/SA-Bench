"""De-anonymization of model responses (Section 2).

Implements two types of target model detectors:
  1. Identity-probing detector (Section 2.2): Uses prompts designed to elicit
     identifying information about the target model.
  2. Training-based detector (Section 2.2): Uses supervised learning with
     BoW, TF-IDF, and response length features to classify model identity.

The problem is formulated as binary classification between the target model
(class 1) and all other models (class 0), as described in Section 2.1.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data import (
    generate_synthetic_responses,
    extract_features,
    prepare_detector_dataset,
    IDENTITY_PROBING_PROMPTS,
    MODEL_NAME_PATTERNS,
)


class IdentityProbingDetector:
    """Detect target model by asking identity-revealing prompts (Section 2.2).

    Experiments use five prompts:
      - "Who are you?" (most effective, >90% accuracy for all models)
      - "Which model are you?"
      - "What is your model name?"
      - "How should I refer to you as an AI?"
      - "How would you define your role or identity as an AI?"

    The classifier predicts a positive match if the model's name
    (e.g., "Llama") or organization (e.g., "Meta") appears anywhere
    in the response.
    """

    def __init__(self, target_model_name: str, prompts: Optional[List[str]] = None):
        self.target_model_name = target_model_name
        self.prompts = prompts or IDENTITY_PROBING_PROMPTS
        self._target_keywords = self._extract_keywords(target_model_name)

    @staticmethod
    def _extract_keywords(model_name: str) -> List[str]:
        """Extract identifying keywords from model name."""
        keywords = []
        lower = model_name.lower()
        for pattern in MODEL_NAME_PATTERNS:
            if pattern in lower:
                keywords.append(pattern)
        parts = lower.replace("-", " ").replace("_", " ").split()
        for p in parts:
            if len(p) > 2 and p.isalpha():
                keywords.append(p)
        return list(set(keywords))

    def detect(self, response: str) -> bool:
        """Check if response contains any target model identifier."""
        lower_response = response.lower()
        return any(kw in lower_response for kw in self._target_keywords)

    def evaluate(
        self,
        model_response_fn,
        num_queries: int = 1000,
    ) -> Dict[str, float]:
        """Evaluate detection accuracy across all identity-probing prompts.

        For each prompt, queries the target model and records whether
        the detector correctly identifies it. Reports average accuracy
        across num_queries per prompt. (Section 2.3)
        """
        results = {}
        for prompt in self.prompts:
            correct = 0
            for _ in range(num_queries):
                response = model_response_fn(self.target_model_name, prompt)
                if self.detect(response):
                    correct += 1
            results[prompt] = correct / num_queries
        return results


class TrainingBasedDetector:
    """Training-based target model detector using supervised learning (Section 2.2).

    The attacker:
    1. Selects a prompt (or set of prompts) and queries all models
    2. Gathers response dataset D_M for target model and D_M' for others
    3. Trains a binary classifier f_{M,P} to de-anonymize the target model

    Features (Section 2.3):
      - Length(R): word count or character count
      - BoW(R): bag-of-words representation
      - TF-IDF(R): term frequency-inverse document frequency

    Uses logistic regression from scikit-learn with default hyperparameters
    and random_state=42 (Section 2.3).
    """

    def __init__(
        self,
        target_model_name: str,
        other_model_names: List[str],
        feature_type: str = "bow",
        random_state: int = 42,
    ):
        self.target_model_name = target_model_name
        self.other_model_names = other_model_names
        self.feature_type = feature_type
        self.random_state = random_state
        self.classifier: Optional[LogisticRegression] = None
        self.vectorizer: Optional[object] = None

    def collect_data(
        self,
        prompts: List[str],
        num_responses_per_model: int = 50,
        output_tokens: int = 512,
        model_query_fn=None,
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        """Collect responses for target and other models across multiple prompts.

        Implements data collection as described in Section 2.3:
        - 50 responses per model for each prompt
        - 512 output tokens
        """
        target_all = []
        other_all = []
        target_lengths_word = []
        other_lengths_word = []

        for prompt in prompts:
            for _ in range(num_responses_per_model):
                response = model_query_fn(self.target_model_name, prompt, output_tokens)
                target_all.append(response)
                target_lengths_word.append(len(response.split()))

            for other_name in self.other_model_names:
                response = model_query_fn(other_name, prompt, output_tokens)
                other_all.append(response)
                other_lengths_word.append(len(response.split()))

        return target_all, other_all, target_lengths_word, other_lengths_word

    def train(
        self,
        target_responses: List[str],
        other_responses: List[str],
        target_lengths_word: Optional[List[int]] = None,
        other_lengths_word: Optional[List[int]] = None,
    ) -> float:
        """Train the binary classifier on balanced dataset.

        Uses an 80/20 train/test split as specified in Section 2.3.
        Returns test accuracy.
        """
        features, labels, vec = prepare_detector_dataset(
            target_responses,
            other_responses,
            target_lengths_word,
            other_lengths_word,
            feature_type=self.feature_type,
        )
        self.vectorizer = vec

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=0.2,
            random_state=self.random_state,
            stratify=labels,
        )

        self.classifier = LogisticRegression(random_state=self.random_state, max_iter=1000)
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc

    def predict(self, response: str) -> int:
        """Predict whether a response comes from the target model.

        Returns 1 if predicted as target model, 0 otherwise.
        """
        if self.classifier is None or self.vectorizer is None:
            raise RuntimeError("Detector must be trained before prediction.")

        if self.feature_type in ("length_word",):
            length = len(response.split())
            features = np.array([length]).reshape(1, -1)
        elif self.feature_type == "length_character":
            features = np.array([len(response)]).reshape(1, -1)
        else:
            features = self.vectorizer.transform([response]).toarray()

        return self.classifier.predict(features)[0]

    def predict_with_proba(self, response: str) -> float:
        """Return probability that the response comes from the target model."""
        if self.classifier is None or self.vectorizer is None:
            raise RuntimeError("Detector must be trained before prediction.")

        if self.feature_type in ("length_word",):
            features = np.array([[len(response.split())]])
        elif self.feature_type == "length_character":
            features = np.array([[len(response)]])
        else:
            features = self.vectorizer.transform([response]).toarray()

        return self.classifier.predict_proba(features)[0, 1]


def evaluate_all_features(
    target_model_name: str,
    other_model_names: List[str],
    prompts: List[str],
    model_query_fn,
    num_responses: int = 50,
    output_tokens: int = 512,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate training-based detector with all four feature types (Table 3).

    Tests:
      - Length(R)_word
      - Length(R)_character
      - BoW(R)
      - TF-IDF(R)
    """
    feature_types = ["length_word", "length_character", "bow", "tfidf"]
    results = {}

    for ftype in feature_types:
        detector = TrainingBasedDetector(
            target_model_name=target_model_name,
            other_model_names=other_model_names,
            feature_type=ftype,
            random_state=random_state,
        )

        target_responses = []
        other_responses = []
        target_lengths = []
        other_lengths = []

        for prompt in prompts:
            for _ in range(num_responses):
                r = model_query_fn(target_model_name, prompt, output_tokens)
                target_responses.append(r)
                target_lengths.append(len(r.split()))

            for other_name in other_model_names:
                r = model_query_fn(other_name, prompt, output_tokens)
                other_responses.append(r)
                other_lengths.append(len(r.split()))

        acc = detector.train(target_responses, other_responses, target_lengths, other_lengths)
        results[ftype] = acc

    return results


def evaluate_prompt_categories(
    target_model_name: str,
    other_model_names: List[str],
    category_prompts: Dict[str, List[str]],
    model_query_fn,
    num_responses: int = 50,
    output_tokens: int = 512,
    feature_type: str = "bow",
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate detection accuracy across prompt categories (Figure 3).

    Tests eight prompt categories:
      - English, Chinese, Spanish (normal chat, high-resource)
      - Indonesian, Persian (normal chat, low-resource)
      - Coding, Math, Safety-violating (specialty)
    """
    results = {}

    for category, prompts in category_prompts.items():
        detector = TrainingBasedDetector(
            target_model_name=target_model_name,
            other_model_names=other_model_names,
            feature_type=feature_type,
            random_state=random_state,
        )

        target_responses = []
        other_responses = []
        target_lengths = []
        other_lengths = []

        for prompt in prompts:
            for _ in range(num_responses):
                r = model_query_fn(target_model_name, prompt, output_tokens)
                target_responses.append(r)
                target_lengths.append(len(r.split()))

            for other_name in other_model_names:
                r = model_query_fn(other_name, prompt, output_tokens)
                other_responses.append(r)
                other_lengths.append(len(r.split()))

        acc = detector.train(target_responses, other_responses, target_lengths, other_lengths)
        results[category] = acc

    return results
