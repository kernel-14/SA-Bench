"""
features.py – Lightweight feature extraction for text responses.

Implements the FeatureExtractor class, which converts raw text into numeric
feature vectors (BoW, TF-IDF, word/character length).  It follows the design
specification and is used by the training‑based detector (detector.py).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureExtractor:
    """Build numeric features for model responses.

    Supports four feature types:
      - ``"bow"``          : bag‑of‑words counts.
      - ``"tfidf"``        : TF‑IDF weighted vectors.
      - ``"length_word"``  : number of space‑separated tokens.
      - ``"length_char"``  : number of characters.

    The ``fit()`` method builds a vocabulary for ``bow``/``tfidf``;
    ``transform()`` returns a 2‑D numpy array ready for a classifier.

    Attributes:
        feature_type:    the active feature representation.
        max_features:    maximum vocabulary size (ignored for length features).
        fitted:          whether ``fit()`` has been called.
        vectorizer:      underlying sklearn vectorizer (``None`` for length features).
    """

    def __init__(
        self,
        feature_type: str,
        max_features: int = 5000,
        **vectorizer_kwargs,
    ) -> None:
        """
        Args:
            feature_type: One of ``"bow"``, ``"tfidf"``, ``"length_word"``,
                          ``"length_char"``.
            max_features: Maximum number of features for BoW / TF‑IDF vectorizers.
                          Ignored for length features.  Default 5000.
            **vectorizer_kwargs: Extra keyword arguments passed to the sklearn
                                 vectorizer (e.g., ``stop_words``, ``ngram_range``).
        """
        allowed = {"bow", "tfidf", "length_word", "length_char"}
        if feature_type not in allowed:
            raise ValueError(
                f"Invalid feature_type '{feature_type}'. Allowed: {allowed}."
            )

        self.feature_type: str = feature_type
        self.max_features: int = max_features
        self.vectorizer_kwargs = vectorizer_kwargs
        self.fitted: bool = False

        # Instantiate the appropriate sklearn vectorizer
        if feature_type == "bow":
            self.vectorizer = CountVectorizer(
                max_features=max_features, **vectorizer_kwargs
            )
        elif feature_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features, **vectorizer_kwargs
            )
        else:  # length features do not need a vectorizer
            self.vectorizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, texts: List[str]) -> None:
        """Fit the feature extractor to a corpus of texts.

        For BoW and TF‑IDF this builds the vocabulary; for length features it is
        a no‑op (but marks the extractor as fitted).

        Args:
            texts: A list of raw response strings.
        """
        if self.feature_type in ("bow", "tfidf"):
            if self.vectorizer is None:
                raise RuntimeError("Vectorizer not initialised – this is a bug.")
            self.vectorizer.fit(texts)
        self.fitted = True

    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert a list of raw texts into a dense feature matrix.

        The returned array has shape ``(len(texts), n_features)`` where
        ``n_features`` is vocabulary size (BoW/TF‑IDF) or 1 (length features).

        Args:
            texts: Response strings to transform.

        Returns:
            2‑D numpy array of float64 features.

        Raises:
            RuntimeError: If ``fit()`` has not been called before transforming
                BoW/TF‑IDF features.
        """
        if self.feature_type == "bow":
            if not self.fitted or self.vectorizer is None:
                raise RuntimeError(
                    "BoW vectorizer must be fitted before calling transform()."
                )
            # Convert sparse matrix to dense (small datasets only)
            return self.vectorizer.transform(texts).toarray()

        elif self.feature_type == "tfidf":
            if not self.fitted or self.vectorizer is None:
                raise RuntimeError(
                    "TF‑IDF vectorizer must be fitted before calling transform()."
                )
            return self.vectorizer.transform(texts).toarray()

        elif self.feature_type == "length_word":
            # Each response becomes a 1‑D array with a single column (word count)
            return np.array(
                [[self.extract_length(t)[0]] for t in texts], dtype=float
            )

        elif self.feature_type == "length_char":
            return np.array(
                [[self.extract_length(t)[1]] for t in texts], dtype=float
            )

        else:
            # Should never reach here because constructor validation would have caught it.
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    # ------------------------------------------------------------------
    # Static utility
    # ------------------------------------------------------------------
    @staticmethod
    def extract_length(text: str) -> Tuple[int, int]:
        """Compute word and character counts for a single string.

        Word count is obtained by splitting on whitespace; character count is
        the full length of the string.

        Args:
            text: The input string.

        Returns:
            A tuple ``(word_count, char_count)``.
        """
        word_count = len(text.split())
        char_count = len(text)
        return word_count, char_count

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<FeatureExtractor type={self.feature_type}, fitted={self.fitted}>"

