# feature_extractor.py

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Dict


class FeatureExtractor:
    """Processes model responses and extracts features for classification tasks."""

    def __init__(self, config: dict) -> None:
        """
        Initializes FeatureExtractor with configuration.

        Args:
            config (dict): Configuration dictionary with feature extraction settings.
        """
        self.tfidf_enabled = config.get("features", {}).get("tfidf_enabled", True)
        self.bow_enabled = config.get("features", {}).get("bow_enabled", True)
        self.length_enabled = config.get("features", {}).get("length_enabled", True)

        # Initialize vectorizer attributes
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000,  # Limit vocabulary size for efficiency
            min_df=2,  # Minimum document frequency
        ) if self.tfidf_enabled else None

        self.bow_vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000,  # Limit vocabulary size for efficiency
            min_df=2,  # Minimum document frequency
        ) if self.bow_enabled else None

    def extract_length_features(self, responses: Dict[str, Dict[str, str]]) -> np.ndarray:
        """
        Extracts length features (word count or character count) from responses.

        Args:
            responses (Dict[str, Dict[str, str]]): Dictionary mapping model names
                                                  to dictionaries of prompts and responses.

        Returns:
            np.ndarray: Feature array of shape (number of responses, 1).
        """
        lengths = []
        # Iterate through model responses
        for model_name, model_responses in responses.items():
            for response in model_responses.values():
                # Calculate response length (word count)
                lengths.append(len(response.split()))  # Simple tokenization using space separation

        return np.array(lengths).reshape(-1, 1)  # Return as 2D array (num_samples, 1)

    def extract_tfidf_features(self, responses: Dict[str, Dict[str, str]]) -> csr_matrix:
        """
        Extracts TF-IDF features from responses.

        Args:
            responses (Dict[str, Dict[str, str]]): Dictionary mapping model names
                                                  to dictionaries of prompts and responses.

        Returns:
            csr_matrix: Sparse matrix of TF-IDF features.
        """
        if not self.tfidf_enabled:
            raise ValueError("TF-IDF extraction is not enabled in the configuration.")

        # Flatten responses into a list of strings
        all_responses = []
        for model_name, model_responses in responses.items():
            for response in model_responses.values():
                all_responses.append(response)

        # Fit-transform responses into TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_responses)
        return tfidf_matrix  # Sparse format for memory efficiency

    def extract_bow_features(self, responses: Dict[str, Dict[str, str]]) -> csr_matrix:
        """
        Extracts Bag-of-Words (BoW) features from responses.

        Args:
            responses (Dict[str, Dict[str, str]]): Dictionary mapping model names
                                                  to dictionaries of prompts and responses.

        Returns:
            csr_matrix: Sparse matrix of BoW features.
        """
        if not self.bow_enabled:
            raise ValueError("BoW extraction is not enabled in the configuration.")

        # Flatten responses into a list of strings
        all_responses = []
        for model_name, model_responses in responses.items():
            for response in model_responses.values():
                all_responses.append(response)

        # Fit-transform responses into BoW matrix
        bow_matrix = self.bow_vectorizer.fit_transform(all_responses)
        return bow_matrix  # Sparse format for memory efficiency
