## deanonymization/feature_extractor.py
"""Feature extraction module for the training-based de-anonymization detector.

This module implements the four feature extraction strategies evaluated in
Section 2.3 and Table 3 of the paper:
  - Length(R)_word: Response length in words.
  - Length(R)_char: Response length in characters.
  - BoW(R): Bag-of-words representation.
  - TF-IDF(R): Term frequency-inverse document frequency representation.

The FeatureExtractor class provides a consistent fit/transform interface
that prevents data leakage by separating vocabulary learning (fit) from
feature computation (transform). A new instance is created per
(prompt, model, feature_type) triple in TrainingBasedDetector to ensure
no vocabulary bleed between experiments.

Alignment with paper:
  - Table 3: All four feature types evaluated on English prompts.
  - Figure 3: BoW features used as primary_feature_type per config.yaml.
  - Figure 2: BoW features used for PCA visualization.
  - Section 2.4.2: "BoW reaching >95% in many cases."

No internal project dependencies — only numpy and scikit-learn are required.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ---------------------------------------------------------------------------
# Valid feature type identifiers — must match config.yaml
# training_based_detector.feature_types values exactly.
# ---------------------------------------------------------------------------
_VALID_FEATURE_TYPES: frozenset = frozenset(
    {"length_word", "length_char", "bow", "tfidf"}
)


class FeatureExtractor:
    """Converts raw LLM response strings into numeric feature arrays.

    Supports four feature types from Section 2.3 of the paper:
      - 'length_word': Word count of each response. Shape (N, 1).
      - 'length_char': Character count of each response. Shape (N, 1).
      - 'bow': Bag-of-words using CountVectorizer. Shape (N, vocab_size).
      - 'tfidf': TF-IDF using TfidfVectorizer. Shape (N, vocab_size).

    The typical usage pattern in TrainingBasedDetector is:
        extractor = FeatureExtractor(feature_type='bow')
        X_train = extractor.fit_transform(train_texts)  # fits vocabulary
        X_test = extractor.transform(test_texts)         # uses fitted vocab

    A new FeatureExtractor instance must be created for each (prompt, model,
    feature_type) triple to prevent vocabulary bleed across experiments.

    Attributes:
        feature_type: One of 'length_word', 'length_char', 'bow', 'tfidf'.
        vectorizer: The fitted sklearn vectorizer for 'bow' and 'tfidf' types.
            None for length-based types and before fit() is called.

    Example:
        >>> extractor = FeatureExtractor(feature_type='bow')
        >>> X_train = extractor.fit_transform(["Hello world", "Goodbye world"])
        >>> X_train.shape
        (2, 2)
        >>> X_test = extractor.transform(["Hello there"])
        >>> X_test.shape
        (1, 2)

        >>> length_extractor = FeatureExtractor(feature_type='length_word')
        >>> X = length_extractor.fit_transform(["Hello world", "One two three"])
        >>> X.tolist()
        [[2], [3]]
    """

    def __init__(self, feature_type: str = "bow") -> None:
        """Initialize the FeatureExtractor with the specified feature type.

        Stores the feature type and initializes the vectorizer slot to None.
        No sklearn objects are instantiated here — instantiation is deferred
        to fit() to ensure a fresh vectorizer for each training split.

        Args:
            feature_type: Feature extraction strategy. Must be one of
                'length_word', 'length_char', 'bow', 'tfidf'. Defaults to
                'bow', which is the primary_feature_type in config.yaml and
                achieves the highest accuracy in Table 3 of the paper.

        Raises:
            ValueError: If feature_type is not one of the four valid values.

        Example:
            >>> extractor = FeatureExtractor(feature_type='tfidf')
            >>> extractor.feature_type
            'tfidf'
            >>> extractor.vectorizer is None
            True
        """
        if feature_type not in _VALID_FEATURE_TYPES:
            raise ValueError(
                f"Unknown feature_type '{feature_type}'. "
                f"Must be one of {sorted(_VALID_FEATURE_TYPES)}."
            )

        self.feature_type: str = feature_type
        # Vectorizer is None for length-based types and before fit() is called
        # for bow/tfidf types. Populated in fit() for bow and tfidf only.
        self.vectorizer: Optional[CountVectorizer | TfidfVectorizer] = None

    def fit(self, texts: List[str]) -> None:
        """Learn vocabulary or statistics from training texts.

        For 'bow' and 'tfidf' types, instantiates and fits the appropriate
        sklearn vectorizer on the provided texts. For length-based types,
        this method is a no-op since no parameters need to be learned.

        IMPORTANT: This method must only be called on training data to prevent
        data leakage. The test split must use transform() alone.

        Args:
            texts: List of response strings from the training split. Typically
                80 strings (80% of 100 total samples) per the paper's 80/20
                train/test split described in Section 2.3.

        Returns:
            None.

        Example:
            >>> extractor = FeatureExtractor(feature_type='bow')
            >>> extractor.fit(["Hello world", "Goodbye world"])
            >>> extractor.vectorizer is not None
            True
            >>> sorted(extractor.vectorizer.vocabulary_.keys())
            ['goodbye', 'hello', 'world']
        """
        if self.feature_type == "bow":
            # Instantiate a fresh CountVectorizer with all default parameters.
            # No max_features cap or min_df filter — the paper uses "simple
            # features" without tuning, and restricting vocabulary would reduce
            # the >95% accuracy reported in Table 3.
            self.vectorizer = CountVectorizer()
            self.vectorizer.fit(texts)

        elif self.feature_type == "tfidf":
            # Instantiate a fresh TfidfVectorizer with all default parameters.
            # IDF weights are computed from training texts only, preventing
            # leakage of test document frequencies into the vocabulary.
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(texts)

        # For 'length_word' and 'length_char': no parameters to learn.
        # self.vectorizer remains None. Method is intentionally a no-op.

    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to a 2D numeric feature array using the fitted extractor.

        For 'bow' and 'tfidf', uses the vocabulary learned during fit().
        For length-based types, computes features directly from text without
        any learned parameters.

        All four feature types return a 2D array with shape (len(texts), F)
        where F is the feature dimensionality:
          - 'length_word': F = 1
          - 'length_char': F = 1
          - 'bow': F = vocab_size (number of unique tokens in training data)
          - 'tfidf': F = vocab_size (same vocabulary as bow for same training data)

        Args:
            texts: List of response strings to transform. May be training or
                test data — this method does not modify the fitted state.
                Empty strings are handled gracefully: length features return 0,
                BoW/TF-IDF return all-zero rows.

        Returns:
            2D numpy array of shape (len(texts), F) with dtype float64.
            Length features return integer-valued floats in shape (N, 1).
            BoW returns non-negative integer counts in shape (N, vocab_size).
            TF-IDF returns non-negative float weights in shape (N, vocab_size).

        Raises:
            RuntimeError: If transform() is called before fit() for 'bow' or
                'tfidf' types (i.e., self.vectorizer is None).

        Example:
            >>> extractor = FeatureExtractor(feature_type='length_word')
            >>> X = extractor.transform(["Hello world", "One two three four"])
            >>> X.tolist()
            [[2], [4]]

            >>> extractor = FeatureExtractor(feature_type='bow')
            >>> extractor.fit(["Hello world"])
            >>> X = extractor.transform(["Hello world", "Hello there"])
            >>> X.shape
            (2, 2)
        """
        if self.feature_type == "length_word":
            # Word count via whitespace splitting. Returns shape (N, 1).
            # Empty string: ''.split() returns [], len([]) = 0. Valid.
            return np.array(
                [[len(t.split())] for t in texts],
                dtype=np.float64,
            )

        elif self.feature_type == "length_char":
            # Character count including spaces and punctuation. Shape (N, 1).
            # Empty string: len('') = 0. Valid.
            return np.array(
                [[len(t)] for t in texts],
                dtype=np.float64,
            )

        elif self.feature_type == "bow":
            if self.vectorizer is None:
                raise RuntimeError(
                    "FeatureExtractor.transform() called before fit() for "
                    "feature_type='bow'. Call fit() on training data first."
                )
            # transform() returns a scipy sparse matrix; .toarray() converts
            # to a dense numpy array of shape (N, vocab_size).
            return self.vectorizer.transform(texts).toarray().astype(np.float64)

        elif self.feature_type == "tfidf":
            if self.vectorizer is None:
                raise RuntimeError(
                    "FeatureExtractor.transform() called before fit() for "
                    "feature_type='tfidf'. Call fit() on training data first."
                )
            # Same pattern as bow: sparse -> dense numpy array.
            return self.vectorizer.transform(texts).toarray().astype(np.float64)

        else:
            # This branch is unreachable if __init__ validated feature_type,
            # but included for defensive completeness.
            raise ValueError(
                f"Unrecognized feature_type '{self.feature_type}' in transform()."
            )

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit on texts and return their transformed feature array.

        Convenience method for use on training data only. Equivalent to
        calling fit(texts) followed by transform(texts). The test split
        must always use transform() alone to prevent data leakage.

        This is the primary entry point called by TrainingBasedDetector on
        the training split:
            X_train = extractor.fit_transform(train_texts)
            X_test = extractor.transform(test_texts)

        Args:
            texts: List of training response strings. The vocabulary (for
                'bow'/'tfidf') or no-op (for length types) is learned from
                exactly these texts.

        Returns:
            2D numpy array of shape (len(texts), F) — same shape and dtype
            as transform() would return after fit().

        Example:
            >>> extractor = FeatureExtractor(feature_type='tfidf')
            >>> X = extractor.fit_transform(["Hello world", "Goodbye world"])
            >>> X.shape
            (2, 2)
            >>> extractor.vectorizer is not None
            True
        """
        self.fit(texts)
        return self.transform(texts)
