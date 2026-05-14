"""
De-anonymization of Model Responses (Section 2)

This module implements two types of target model detectors:
1. Identity-probing detector (Section 2.2, 2.4.1)
2. Training-based detector (Section 2.2, 2.4.2)

Problem Formulation (Section 2.1):
Given a target model M from the public set of models, the attacker aims to build
a classifier f_M that, given a prompt-response pair (P, R), outputs 1 if and only
if the response comes from the target model.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Identity-Probing Detector (Section 2.2, 2.4.1)
# =============================================================================

IDENTITY_PROBING_PROMPTS = [
    "Who are you?",
    "Which model are you?",
    "What is your model name?",
    "How should I refer to you as an AI?",
    "How would you define your role or identity as an AI?",
]

# Mapping of model name keywords (as described in paper)
MODEL_IDENTITY_MAP = {
    "claude": ["Claude", "Anthropic"],
    "gemini": ["Gemini", "Google"],
    "gpt": ["GPT", "OpenAI", "ChatGPT"],
    "gemma": ["Gemma", "Google"],
    "llama": ["Llama", "Meta"],
    "mixtral": ["Mixtral", "Mistral"],
    "qwen": ["Qwen", "Alibaba"],
    "chatglm": ["ChatGLM", "Tsinghua"],
    "dolly": ["Dolly", "Databricks"],
    "stablelm": ["StableLM", "Stability"],
    "fastchat": ["FastChat", "LMSYS"],
}


class IdentityProbingDetector:
    """Detector that uses identity-probing prompts to identify target model.
    
    As described in Section 2.2: the attacker crafts a prompt P designed to
    elicit identifying information about the target model (e.g., its name).
    The classifier predicts the model as a positive match if its name or 
    organization appears anywhere in the response.
    """
    
    def __init__(self, target_model: str, model_identity_map: Optional[Dict] = None):
        """
        Args:
            target_model: The model identifier string (e.g., "llama-3.1-70b-instruct")
            model_identity_map: Mapping from model family names to possible keywords
        """
        self.target_model = target_model
        self.model_identity_map = model_identity_map or MODEL_IDENTITY_MAP
        self.target_keywords = self._get_target_keywords()
        
    def _get_target_keywords(self) -> List[str]:
        """Determine which keywords correspond to the target model."""
        model_lower = self.target_model.lower()
        for family, keywords in self.model_identity_map.items():
            if family in model_lower:
                return keywords
        # Fallback: use the model name itself
        return [self.target_model]
    
    def predict(self, response: str) -> int:
        """Check if response contains target model identity markers.
        
        Returns 1 if the target model is detected, 0 otherwise.
        """
        for keyword in self.target_keywords:
            if re.search(re.escape(keyword), response, re.IGNORECASE):
                return 1
        return 0
    
    def evaluate(self, target_responses: List[str], other_responses: List[str]) -> float:
        """Evaluate detection accuracy on test data.
        
        Args:
            target_responses: Responses from the target model
            other_responses: Responses from other models
            
        Returns:
            accuracy: fraction of correct predictions
        """
        y_true = [1] * len(target_responses) + [0] * len(other_responses)
        y_pred = []
        for r in target_responses:
            y_pred.append(self.predict(r))
        for r in other_responses:
            y_pred.append(self.predict(r))
        return accuracy_score(y_true, y_pred)


# =============================================================================
# Training-Based Detector (Section 2.2, 2.4.2)
# =============================================================================

# Prompt categories as defined in Table 1
PROMPT_CATEGORIES = {
    "english": "Normal chat, high-resource language (English)",
    "chinese": "Normal chat, high-resource language (Chinese)",
    "spanish": "Normal chat, high-resource language (Spanish)",
    "indonesian": "Normal chat, low-resource language (Indonesian)",
    "persian": "Normal chat, low-resource language (Persian)",
    "coding": "Specialty chat - Coding",
    "math": "Specialty chat - Math",
    "safety": "Specialty chat - Safety-violating",
}


class TextFeatureExtractor:
    """Extract text features from model responses as described in Section 2.
    
    Features:
    - Length (word and character) 
    - Bag-of-Words (BoW)
    - TF-IDF
    """
    
    def __init__(self, feature_type: str = "bow", max_features: int = 5000):
        """
        Args:
            feature_type: One of "length_word", "length_char", "bow", "tfidf"
            max_features: Max features for BoW/TF-IDF vectorizer
        """
        self.feature_type = feature_type
        self.max_features = max_features
        self.vectorizer = None
        self._fitted = False
        
    def _extract_length_features(self, texts: List[str]) -> np.ndarray:
        """Extract response length features.
        
        Returns:
            Array of shape (n_samples, n_features) where n_features is 1 or 2
            depending on whether word or character (or both) length is used.
        """
        if self.feature_type == "length_word":
            return np.array([[len(t.split())] for t in texts])
        elif self.feature_type == "length_char":
            return np.array([[len(t)] for t in texts])
        elif self.feature_type == "length_both":
            return np.array([[len(t.split()), len(t)] for t in texts])
        else:
            raise ValueError(f"Unknown length feature type: {self.feature_type}")
    
    def fit(self, texts: List[str]):
        """Fit the feature extractor on training texts."""
        if self.feature_type in ("bow", "tfidf"):
            if self.feature_type == "bow":
                self.vectorizer = CountVectorizer(max_features=self.max_features)
            else:
                self.vectorizer = TfidfVectorizer(max_features=self.max_features)
            self.vectorizer.fit(texts)
        self._fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to feature vectors."""
        if not self._fitted:
            self.fit(texts)
        
        if self.feature_type in ("length_word", "length_char", "length_both"):
            return self._extract_length_features(texts)
        elif self.feature_type in ("bow", "tfidf"):
            return self.vectorizer.transform(texts).toarray()
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)


class TrainingBasedDetector:
    """Training-based detector as described in Section 2.2.
    
    Uses supervised learning (logistic regression) to differentiate between
    models' responses to the same prompt. The attacker queries models to
    gather responses, then trains a binary classifier f_{M,P} to de-anonymize
    the target model M.
    
    Key design choices from paper:
    - Logistic regression from scikit-learn with default hyperparameters, random_state=42
    - 80/20 train/test split
    - Balanced datasets: 50 positive samples (target) + 50 negative samples (other models)
    """
    
    def __init__(
        self,
        target_model: str,
        feature_type: str = "bow",
        max_features: int = 5000,
        random_state: int = 42,
    ):
        """
        Args:
            target_model: Name of the target model
            feature_type: Type of text features to use
            max_features: Max features for vectorizer
            random_state: Random seed for reproducibility
        """
        self.target_model = target_model
        self.feature_type = feature_type
        self.max_features = max_features
        self.random_state = random_state
        
        self.feature_extractor = TextFeatureExtractor(
            feature_type=feature_type, 
            max_features=max_features
        )
        self.classifier = None
        
    def _build_pipeline(self) -> Pipeline:
        """Build classification pipeline.
        
        For length features, use standard scaling before logistic regression.
        For BoW/TF-IDF, the features are already frequency-based.
        """
        if self.feature_type in ("length_word", "length_char", "length_both"):
            return Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(random_state=self.random_state, max_iter=1000))
            ])
        else:
            return Pipeline([
                ('clf', LogisticRegression(random_state=self.random_state, max_iter=1000))
            ])
    
    def train(
        self, 
        target_responses: List[str], 
        other_responses: List[str],
        verbose: bool = False
    ) -> Dict:
        """
        Train the detector on balanced dataset.
        
        As described in Section 2.3: 
        - Balanced datasets with 50 responses from target model (positive)
          and 50 uniformly sampled responses from other models (negative)
        - 80/20 train/test split
        - Logistic regression with default hyperparameters, random_state=42
        
        Args:
            target_responses: Responses from the target model
            other_responses: Responses from other models
            verbose: Whether to log details
            
        Returns:
            Dict with training results including accuracy
        """
        # Build balanced dataset
        X_texts = target_responses + other_responses
        y = np.array([1] * len(target_responses) + [0] * len(other_responses))
        
        # Train/test split (80/20 as specified in paper)
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            X_texts, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        if verbose:
            logger.info(f"Training data: {len(X_train_texts)} samples")
            logger.info(f"Test data: {len(X_test_texts)} samples")
            logger.info(f"Positive samples: {y_train.sum()}, Negative: {len(y_train) - y_train.sum()}")
        
        # Fit feature extractor on training data
        X_train = self.feature_extractor.fit_transform(X_train_texts)
        X_test = self.feature_extractor.transform(X_test_texts)
        
        # Train classifier
        self.classifier = self._build_pipeline()
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, self.classifier.predict(X_train))
        
        if verbose:
            logger.info(f"Train accuracy: {train_accuracy:.4f}")
            logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "n_train": len(X_train_texts),
            "n_test": len(X_test_texts),
            "feature_type": self.feature_type,
        }
    
    def predict(self, responses: List[str]) -> np.ndarray:
        """Predict whether responses are from the target model."""
        if self.classifier is None:
            raise ValueError("Detector must be trained before prediction")
        X = self.feature_extractor.transform(responses)
        return self.classifier.predict(X)
    
    def predict_proba(self, responses: List[str]) -> np.ndarray:
        """Get prediction probabilities."""
        if self.classifier is None:
            raise ValueError("Detector must be trained before prediction")
        X = self.feature_extractor.transform(responses)
        return self.classifier.predict_proba(X)


def evaluate_prompt_for_detection(
    prompt: str,
    target_model_responses: List[str],
    other_model_responses: List[str],
    feature_types: List[str] = ["length_word", "length_char", "bow", "tfidf"],
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Evaluate how well a given prompt distinguishes the target model from others.
    
    This is used for prompt selection (Section 2.2): the attacker scores each
    prompt on its ability to distinguish models.
    
    Args:
        prompt: The prompt text
        target_model_responses: List of responses from target model for this prompt
        other_model_responses: List of responses from other models for this prompt
        feature_types: List of feature types to try
        random_state: Random seed
        
    Returns:
        Dict mapping feature_type -> accuracy
    """
    results = {}
    for ft in feature_types:
        detector = TrainingBasedDetector(
            target_model="target",
            feature_type=ft,
            random_state=random_state,
        )
        train_result = detector.train(target_model_responses, other_model_responses)
        results[ft] = train_result["test_accuracy"]
    return results


# =============================================================================
# PCA Visualization helper (for Figure 2 reproduction)
# =============================================================================

def compute_pca_visualization(
    responses_by_model: Dict[str, List[str]],
    n_components: int = 2,
    max_features: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA of BoW features for visualization (Figure 2).
    
    Args:
        responses_by_model: Dict mapping model_name -> list of responses
        n_components: Number of PCA components
        max_features: Max features for BoW vectorizer
        
    Returns:
        projections: PCA projections (n_samples, n_components)
        labels: Integer labels for each sample
        model_names: List of model names
    """
    from sklearn.decomposition import PCA
    
    # Collect all responses
    all_responses = []
    labels = []
    model_names = list(responses_by_model.keys())
    
    for i, (model, responses) in enumerate(responses_by_model.items()):
        all_responses.extend(responses)
        labels.extend([i] * len(responses))
    
    # BoW features
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(all_responses).toarray()
    
    # PCA
    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(X)
    
    return projections, np.array(labels), model_names


# =============================================================================
# Prompt selection utilities
# =============================================================================

def select_best_prompts(
    prompt_candidates: List[str],
    get_responses_fn,
    target_model: str,
    other_models: List[str],
    n_best: int = 10,
    feature_type: str = "bow",
) -> List[Tuple[str, float]]:
    """
    Select the best prompts for de-anonymization (Section 2.2).
    
    The attacker scores prompts on their ability to distinguish models.
    
    Args:
        prompt_candidates: List of candidate prompts
        get_responses_fn: Function that given prompt and model returns list of responses
        target_model: Target model identifier
        other_models: List of other model identifiers
        n_best: Number of best prompts to return
        feature_type: Feature type for detection
        
    Returns:
        List of (prompt, accuracy) tuples, sorted by accuracy descending
    """
    prompt_scores = []
    for prompt in prompt_candidates:
        target_responses = get_responses_fn(prompt, target_model)
        other_responses = []
        for model in other_models:
            other_responses.extend(get_responses_fn(prompt, model))
        
        scores = evaluate_prompt_for_detection(
            prompt, target_responses, other_responses, [feature_type]
        )
        prompt_scores.append((prompt, scores[feature_type]))
        logger.info(f"Prompt score: {scores[feature_type]:.3f} - {prompt[:80]}...")
    
    prompt_scores.sort(key=lambda x: x[1], reverse=True)
    return prompt_scores[:n_best]
