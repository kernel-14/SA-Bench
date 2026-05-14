"""
Text feature extraction for the training-based model detector.

Implements the three feature types described in Section 2.3:
  - Length(R): response length in words or characters
  - TF-IDF(R): term frequency–inverse document frequency features
  - BoW(R): bag-of-words features
"""

from typing import List, Literal, Tuple

import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

FeatureType = Literal["length_word", "length_char", "bow", "tfidf"]


def extract_length_features(
    responses: List[str],
    unit: Literal["word", "char"] = "word",
) -> np.ndarray:
    """
    Extract response length as a single scalar feature.

    Args:
        responses: List of text responses.
        unit: "word" for word count, "char" for character count.

    Returns:
        Array of shape (n_samples, 1).
    """
    if unit == "word":
        lengths = np.array([len(r.split()) for r in responses], dtype=np.float64)
    else:
        lengths = np.array([len(r) for r in responses], dtype=np.float64)
    return lengths.reshape(-1, 1)


class BagOfWordsExtractor:
    """
    Bag-of-words feature extractor wrapping sklearn CountVectorizer.

    The paper uses BoW as the primary feature for the training-based detector
    (Section 2.3, Table 3).
    """

    def __init__(self, max_features: int = 50000, binary: bool = False) -> None:
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            binary=binary,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )
        self._fitted = False

    def fit(self, responses: List[str]) -> "BagOfWordsExtractor":
        self.vectorizer.fit(responses)
        self._fitted = True
        return self

    def transform(self, responses: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = self.vectorizer.transform(responses)
        if issparse(X):
            X = X.toarray()
        return X.astype(np.float64)

    def fit_transform(self, responses: List[str]) -> np.ndarray:
        self.fit(responses)
        return self.transform(responses)


class TFIDFExtractor:
    """
    TF-IDF feature extractor wrapping sklearn TfidfVectorizer.

    Used as an alternative to BoW in Table 3 of the paper.
    """

    def __init__(self, max_features: int = 50000) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, responses: List[str]) -> "TFIDFExtractor":
        self.vectorizer.fit(responses)
        self._fitted = True
        return self

    def transform(self, responses: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = self.vectorizer.transform(responses)
        if issparse(X):
            X = X.toarray()
        return X.astype(np.float64)

    def fit_transform(self, responses: List[str]) -> np.ndarray:
        self.fit(responses)
        return self.transform(responses)


def extract_features(
    train_responses: List[str],
    test_responses: List[str],
    feature_type: FeatureType,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a feature extractor on training responses and transform both splits.

    Args:
        train_responses: Responses for fitting the extractor.
        test_responses: Responses to transform using the fitted extractor.
        feature_type: One of "length_word", "length_char", "bow", "tfidf".

    Returns:
        Tuple of (train_features, test_features) as numpy arrays.
    """
    if feature_type == "length_word":
        X_train = extract_length_features(train_responses, unit="word")
        X_test = extract_length_features(test_responses, unit="word")
    elif feature_type == "length_char":
        X_train = extract_length_features(train_responses, unit="char")
        X_test = extract_length_features(test_responses, unit="char")
    elif feature_type == "bow":
        extractor = BagOfWordsExtractor()
        X_train = extractor.fit_transform(train_responses)
        X_test = extractor.transform(test_responses)
    elif feature_type == "tfidf":
        extractor = TFIDFExtractor()
        X_train = extractor.fit_transform(train_responses)
        X_test = extractor.transform(test_responses)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    return X_train, X_test


def compute_pca_embeddings(
    responses: List[str],
    n_components: int = 2,
    feature_type: FeatureType = "bow",
) -> np.ndarray:
    """
    Compute PCA-reduced embeddings for visualization (Figure 2 in the paper).

    Args:
        responses: List of text responses.
        n_components: Number of principal components.
        feature_type: Feature type to use before PCA.

    Returns:
        Array of shape (n_samples, n_components).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if feature_type == "bow":
        extractor = BagOfWordsExtractor()
        X = extractor.fit_transform(responses)
    elif feature_type == "tfidf":
        extractor = TFIDFExtractor()
        X = extractor.fit_transform(responses)
    else:
        X = extract_length_features(responses, unit="word" if feature_type == "length_word" else "char")

    scaler = StandardScaler(with_mean=True)
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X_scaled)
